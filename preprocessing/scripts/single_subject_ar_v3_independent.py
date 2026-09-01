"""
Single-Subject AutoReject v3 Pipeline -- Independent Onset/Response Rejection (Localizer)
Designed to be called by the SLURM job array with a subject ID argument.

Unlike single_subject_ar_v3.py, onset and response epoch rejection are fully
independent here -- a trial being AutoReject-rejected in one stream no longer
forces the other stream's epoch for that trial to be dropped too. Each saved
epoch set (onset, response) reflects only its own flat + AutoReject decisions,
so the two are generally different lengths.

Per-trial survivability is recorded on the filtered logfile via two new
columns, 'include_onset_epoch' and 'include_response_epoch' ('yes'/'no'),
rather than being implicit in which trials got dropped.

Baseline correction (-101..-1ms relative to the PROMPT, event 160 -- not
board onset, event 161) is applied BEFORE either stream's AutoReject runs,
not after. Confirmed event order per trial: 162 (fixation cross) -> 160
(prompt) -> 161 (board onset) -> 164 (response). A baseline window ending
at 161 sits directly against the board-onset boundary -- vulnerable to
trigger jitter and to the 1-40Hz IIR filter (applied during preprocessing)
smearing the board-onset evoked response backward into it. Anchoring to 160
instead moves the baseline a further ~1s earlier, away from any
task-relevant stimulus transition. -101..-1ms rather than -100..0ms: MNE's
tmin/tmax are inclusive, so a window ending at exactly 0 would include the
sample AT the 160 trigger itself -- the very instant the prompt appears --
contaminating the "pre-stimulus" baseline with one sample of post-onset
signal. Shifting by one sample (1ms at this sfreq) avoids that while keeping
the same 101-sample window width. This same off-by-one was already present
in the prior -100..0-relative-to-161 baseline; fixed here as part of moving
the anchor, not a separate unrelated change.

Baselining before AutoReject (rather than after) is mathematically
equivalent to computing it from AutoReject-cleaned data afterward:
AutoReject's decisions are entirely peak-to-peak-based, which is invariant
to a constant offset, and channel interpolation is a fixed linear spatial
operator (see MNE's _do_interp_dots: a matmul broadcast over time with no
cross-sample dependency) that commutes with constant-offset subtraction. So
baselining first and letting AutoReject run on already-baselined data
produces identical final values to baselining a cleaned signal afterward --
without needing any private AutoReject internals.

Baseline now comes from its own Epochs object (baseline_epochs, built from
raw and anchored to each trial's own 160 event), not from cropping
epochs_onset_full's own leading edge as before -- 160 precedes 161 by ~1s,
well outside epochs_onset_full's own tmin, so there's no window within the
onset epoch itself to reuse anymore. This introduces a new possible gap
that didn't exist when baseline was derived from epochs_onset_full
directly: a trial could survive flat rejection in the onset and/or response
stream but get flat-dropped from baseline_epochs specifically. Same
defensive philosophy as before -- this assumes that never happens, and
raises a clear error rather than silently omitting a baseline if it ever
does.

Usage:
    python single_subject_ar_v3_independent.py --subject R2649 --base_dir /scratch/djn8068/meg_project/data
"""

import argparse
import pandas as pd
import mne
import pickle
import os
import sys
import traceback
from sklearn.preprocessing import LabelEncoder
import numpy as np
from autoreject import AutoReject

mne.set_log_level(verbose='WARNING')

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description='AutoReject v3 for a single subject -- independent onset/response rejection')
parser.add_argument('--subject', type=str, required=True,
                    help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                    default='/scratch/djn8068/meg_project/data',
                    help='Base directory containing subject folders')
args = parser.parse_args()

subj     = args.subject
base_dir = args.base_dir

print(f'\n{"="*60}')
print(f'PROCESSING (independent onset/response): {subj}')
print(f'{"="*60}')
print(f'Base dir: {base_dir}')

# =============================================================================
# PATHS
# =============================================================================

data_dir   = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
raw_path   = os.path.join(preproc_dir, f'{subj}_localizer_ICA-raw_final.fif')

if not os.path.exists(raw_path):
    print(f'ERROR: ICA file not found at {raw_path}')
    sys.exit(1)

cv_loss_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_independent_cv_loss.pkl')
if os.path.exists(cv_loss_path):
    print(f'SKIPPING {subj}: ar_v3_independent cv_loss file already exists')
    sys.exit(0)

# =============================================================================
# CONFIGURATION
# =============================================================================

epoch_config = {
    "tmin_161": -0.1, "tmax_161": 1.0,
    "tmin_164": -1.0, "tmax_164": 0.1,
    "decim": 1,
    "photodiode_realignment": True,
    "flat_criteria": dict(mag=1e-15)
}

le = LabelEncoder()
le.fit(['none', '2', '2gap', '3', '4'])

# =============================================================================
# LOAD RAW DATA
# =============================================================================

print(f'  Loading {raw_path}')
raw = mne.io.read_raw_fif(raw_path, preload=True)
events = mne.find_events(raw, min_duration=0.0025)
picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
flat_criteria = epoch_config["flat_criteria"]

if epoch_config["photodiode_realignment"]:
    events[:, 0] += 60

# =============================================================================
# LOAD LOGFILE
# =============================================================================

logfile_path = os.path.join(data_dir, f'{subj}_logfile.csv')
if not os.path.exists(logfile_path):
    print(f'ERROR: Logfile not found: {logfile_path}')
    sys.exit(1)
logfile = pd.read_csv(logfile_path)

# =============================================================================
# TRIAL SELECTION
# =============================================================================

print('  Selecting trials...')

events_160 = events[events[:, 2] == 160]
events_161 = events[events[:, 2] == 161]
events_164 = events[events[:, 2] == 164]

print(f"    Prompt events (160): {len(events_160)}")
print(f"    Onset events (161): {len(events_161)}")
print(f"    Offset events (164): {len(events_164)}")

if len(events_161) != len(events_164) or len(events_160) != len(events_161):
    print(f"    WARNING: Unequal prompt/onset/offset event counts")

filtered_events_160_temp = []
filtered_events_161_temp = []
for i in range(len(events_161)):
    if pd.notna(logfile.iloc[i]['acc']):
        filtered_events_160_temp.append(events_160[i])
        filtered_events_161_temp.append(events_161[i])
filtered_events_160_temp = np.array(filtered_events_160_temp)
filtered_events_161_temp = np.array(filtered_events_161_temp)
print(f"    Onset events with responses: {len(filtered_events_161_temp)}")

logfile = logfile[(logfile['task'] == 'localizer') & (logfile['acc'].notna())]
logfile = logfile.reset_index(drop=True)

filtered_events_160 = []
filtered_events_161 = []
filtered_events_164 = []
for i in range(len(logfile)):
    if logfile.iloc[i]['acc'] == True:
        filtered_events_160.append(filtered_events_160_temp[i])
        filtered_events_161.append(filtered_events_161_temp[i])
        filtered_events_164.append(events_164[i])

filtered_events_160 = np.array(filtered_events_160)
filtered_events_161 = np.array(filtered_events_161)
filtered_events_164 = np.array(filtered_events_164)
print(f"    Correct trials kept: {len(filtered_events_161)}")

logfile = logfile[(logfile['task'] == 'localizer') & (logfile['acc'] == True)]
logfile = logfile.reset_index(drop=True)

# Store the absolute MEG sample of the 164 event (post-photodiode-offset) for
# each trial, same as single_subject_ar_v3_main.py, for downstream
# response-locked analyses that want the raw sample rather than re-deriving
# it from the separately-saved response epochs.
logfile = logfile.copy()
logfile['response_sample'] = filtered_events_164[:, 0].astype(int)

# =============================================================================
# CREATE EPOCHS
# =============================================================================

print('  Creating epochs...')
epochs_onset_full = mne.Epochs(
    raw, filtered_events_161,
    event_id={'161': 161},
    tmin=epoch_config["tmin_161"],
    tmax=epoch_config["tmax_161"],
    baseline=None,
    picks=picks_meg,
    decim=epoch_config["decim"],
    preload=True,
    reject=None,
    flat=flat_criteria
)

epochs_response_full = mne.Epochs(
    raw, filtered_events_164,
    event_id={'164': 164},
    tmin=epoch_config["tmin_164"],
    tmax=epoch_config["tmax_164"],
    baseline=None,
    picks=picks_meg,
    decim=epoch_config["decim"],
    preload=True,
    reject=None,
    flat=flat_criteria
)

# Track flat-dropped epochs using drop_log (matches original script exactly)
flat_dropped_onset = set(
    i for i, log in enumerate(epochs_onset_full.drop_log) if log
)
flat_dropped_response = set(
    i for i, log in enumerate(epochs_response_full.drop_log) if log
)

if flat_dropped_onset:
    print(f"  WARNING: {len(flat_dropped_onset)} onset epoch(s) dropped by flat criteria: {sorted(flat_dropped_onset)}")
if flat_dropped_response:
    print(f"  WARNING: {len(flat_dropped_response)} response epoch(s) dropped by flat criteria: {sorted(flat_dropped_response)}")

# Map positions in epochs objects back to original logfile indices
onset_full_kept = np.array(
    [i for i, log in enumerate(epochs_onset_full.drop_log) if not log]
)
response_full_kept = np.array(
    [i for i, log in enumerate(epochs_response_full.drop_log) if not log]
)

print(f"    Onset epochs after flat rejection: {len(epochs_onset_full)}")
print(f"    Response epochs after flat rejection: {len(epochs_response_full)}")

# =============================================================================
# BASELINE CORRECTION -- applied BEFORE either stream's AutoReject runs
# =============================================================================
# -101..-1ms relative to the PROMPT (event 160), not board onset (161) --
# see module docstring for why (161-anchored baseline sits against the
# board-onset boundary; 160-anchored moves it ~1s earlier) and why -101..-1
# rather than -100..0 (MNE's tmax is inclusive, so -100..0 would include the
# sample AT the 160 trigger itself). Built as its own Epochs object here,
# not cropped from epochs_onset_full -- 160 precedes 161 by ~1s, well
# outside epochs_onset_full's own tmin, so there's no window inside the
# onset epoch itself to reuse anymore.

print('\n  Creating baseline epochs (relative to prompt, event 160)...')
baseline_epochs = mne.Epochs(
    raw, filtered_events_160,
    event_id={'160': 160},
    tmin=-0.101, tmax=-0.001,
    baseline=None,
    picks=picks_meg,
    decim=epoch_config["decim"],
    preload=True,
    reject=None,
    flat=flat_criteria
)
flat_dropped_baseline = set(
    i for i, log in enumerate(baseline_epochs.drop_log) if log
)
if flat_dropped_baseline:
    print(f"  WARNING: {len(flat_dropped_baseline)} baseline epoch(s) dropped by flat criteria: {sorted(flat_dropped_baseline)}")
baseline_full_kept = np.array(
    [i for i, log in enumerate(baseline_epochs.drop_log) if not log]
)

print('\n  Applying pre-AutoReject baseline correction...')
baseline_by_position = baseline_epochs.get_data().mean(axis=-1)
# row p corresponds to row p of baseline_epochs, i.e. trial baseline_full_kept[p]

# New possible gap vs. the old onset-derived baseline (see docstring): a
# trial could survive flat rejection in the onset and/or response stream
# but get flat-dropped from baseline_epochs specifically, since it's now an
# independently-constructed Epochs object rather than a crop of
# epochs_onset_full. Checked explicitly for both streams, same defensive
# philosophy as before -- raise rather than silently drop a baseline.
trial_to_baseline_pos = {int(t): pos for pos, t in enumerate(baseline_full_kept)}
orphan_missing_baseline = sorted(set(
    [int(t) for t in onset_full_kept if int(t) not in trial_to_baseline_pos] +
    [int(t) for t in response_full_kept if int(t) not in trial_to_baseline_pos]
))
if orphan_missing_baseline:
    raise RuntimeError(
        f"{len(orphan_missing_baseline)} trial(s) present in the onset and/or response "
        f"stream have no baseline source because they were dropped from baseline_epochs "
        f"by FLAT criteria at construction time: trial indices {orphan_missing_baseline}. "
        f"This script assumes flat drops never remove a trial from the baseline stream; "
        f"if this fires, a raw-signal baseline fallback needs to be added for these trial(s)."
    )

baseline_positions_onset = [trial_to_baseline_pos[int(t)] for t in onset_full_kept]
epochs_onset_full._data -= baseline_by_position[baseline_positions_onset][:, :, None]

baseline_positions_response = [trial_to_baseline_pos[int(t)] for t in response_full_kept]
epochs_response_full._data -= baseline_by_position[baseline_positions_response][:, :, None]

# =============================================================================
# AUTOREJECT v3 -- ONSET
# =============================================================================

print('\n  Running AutoReject v3 on onset epochs...')
ar_onset = AutoReject(
    n_interpolate=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
    consensus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    random_state=42,
    n_jobs=-1,
    thresh_method='bayesian_optimization',
    verbose=True
)
epochs_onset_clean, reject_log_onset = ar_onset.fit_transform(
    epochs_onset_full, return_log=True
)

surviving_onset = reject_log_onset.labels[~reject_log_onset.bad_epochs]
print(f'\n  Onset AutoReject summary:')
print(f'    Learned n_interpolate: {ar_onset.n_interpolate_}')
print(f'    Learned consensus: {ar_onset.consensus_}')
print(f'    Epochs dropped: {reject_log_onset.bad_epochs.sum()}')
print(f'    Max interpolated channels per epoch (surviving): {(surviving_onset == 2).sum(axis=1).max()}')
print(f'    Max flagged-unfixed channels per epoch (surviving): {(surviving_onset == 1).sum(axis=1).max()}')

# =============================================================================
# AUTOREJECT v3 -- RESPONSE
# =============================================================================

print('\n  Running AutoReject v3 on response epochs...')
ar_response = AutoReject(
    n_interpolate=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
    consensus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    random_state=42,
    n_jobs=-1,
    thresh_method='bayesian_optimization',
    verbose=True
)
epochs_response_clean, reject_log_response = ar_response.fit_transform(
    epochs_response_full, return_log=True
)

surviving_response = reject_log_response.labels[~reject_log_response.bad_epochs]
print(f'\n  Response AutoReject summary:')
print(f'    Learned n_interpolate: {ar_response.n_interpolate_}')
print(f'    Learned consensus: {ar_response.consensus_}')
print(f'    Epochs dropped: {reject_log_response.bad_epochs.sum()}')
print(f'    Max interpolated channels per epoch (surviving): {(surviving_response == 2).sum(axis=1).max()}')
print(f'    Max flagged-unfixed channels per epoch (surviving): {(surviving_response == 1).sum(axis=1).max()}')

# =============================================================================
# INDEPENDENT BAD-TRIAL TRACKING -- no pairing/enforcement between streams
# =============================================================================

bad_onset_ar      = set(onset_full_kept[reject_log_onset.bad_epochs])
bad_response_ar   = set(response_full_kept[reject_log_response.bad_epochs])
bad_onset_indices = flat_dropped_onset | bad_onset_ar
bad_response_indices = flat_dropped_response | bad_response_ar

all_trial_indices = set(range(len(logfile)))
good_indices_onset    = sorted(all_trial_indices - bad_onset_indices)
good_indices_response = sorted(all_trial_indices - bad_response_indices)

print(f"\n  Rejection summary (independent):")
print(f"    Bad onset epochs:    {len(bad_onset_indices)}")
print(f"    Bad response epochs: {len(bad_response_indices)}")
print(f"    Bad in at least one stream (informational only, not enforced): "
      f"{len(bad_onset_indices | bad_response_indices)}")
print(f"    Onset survivors:    {len(good_indices_onset)}")
print(f"    Response survivors: {len(good_indices_response)}")

# Baseline correction already applied to both streams before AutoReject ran
# (see BASELINE CORRECTION section above) -- nothing further needed here.

# =============================================================================
# FINAL VERIFICATION
# =============================================================================

if len(epochs_onset_clean) != len(good_indices_onset):
    print(f'ERROR: Onset mismatch: epochs={len(epochs_onset_clean)}, '
          f'expected={len(good_indices_onset)}')
    sys.exit(1)
if len(epochs_response_clean) != len(good_indices_response):
    print(f'ERROR: Response mismatch: epochs={len(epochs_response_clean)}, '
          f'expected={len(good_indices_response)}')
    sys.exit(1)
print(f"\n  Final onset trial count:    {len(epochs_onset_clean)}")
print(f"  Final response trial count: {len(epochs_response_clean)}")

# =============================================================================
# METADATA
# =============================================================================

logfile['coded_ground_truth'] = le.transform(logfile['ground_truth'])

logfile['include_onset_epoch'] = 'no'
logfile.loc[good_indices_onset, 'include_onset_epoch'] = 'yes'
logfile['include_response_epoch'] = 'no'
logfile.loc[good_indices_response, 'include_response_epoch'] = 'yes'

epochs_onset_clean.metadata    = logfile.iloc[good_indices_onset].reset_index(drop=True)
epochs_response_clean.metadata = logfile.iloc[good_indices_response].reset_index(drop=True)

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

# Filtered logfile -- ALL eligible trials, tagged with independent per-stream
# survivability (moved here from right after trial selection, since the
# survivability columns aren't known until AutoReject has run)
ar_logfile_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_independent_filtered_logfile.csv')
logfile.to_csv(ar_logfile_path, index=False)
print(f"\n  Saved: {ar_logfile_path}")

# Rejection report -- independent onset/response summaries, no shared
# good_indices/final_trial_count (that would no longer mean anything)
rejection_report = {
    'total_trials': len(all_trial_indices),
    'onset': {
        'bad_indices':        sorted(bad_onset_indices),
        'good_indices':       good_indices_onset,
        'final_trial_count':  len(good_indices_onset),
    },
    'response': {
        'bad_indices':        sorted(bad_response_indices),
        'good_indices':       good_indices_response,
        'final_trial_count':  len(good_indices_response),
    },
    'bad_in_at_least_one_stream_informational_only': sorted(bad_onset_indices | bad_response_indices),
}
report_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_independent_rejection_report.p')
with open(report_path, 'wb') as f:
    pickle.dump(rejection_report, f)
print(f"  Saved: {report_path}")

# Reject logs
for log, label in [(reject_log_onset,    'ar_v3_independent_reject_log_onset'),
                   (reject_log_response, 'ar_v3_independent_reject_log_response')]:
    log_path = os.path.join(preproc_dir, f'{subj}_localizer_{label}.npz')
    if os.path.exists(log_path):
        os.remove(log_path)
    log.save(log_path)
print(f"  Saved AR v3 independent reject logs")

# CV loss landscape -- shape: (n_consensus, n_interpolate, n_folds)
n_interpolate_grid = np.array([1, 2, 4, 8, 12, 16, 20, 24, 28, 32])
consensus_grid     = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
loss_data = {
    'onset': {
        'loss':               ar_onset.loss_['mag'],
        'consensus':          consensus_grid,
        'n_interpolate':      n_interpolate_grid,
        'best_consensus':     float(ar_onset.consensus_['mag']),
        'best_n_interpolate': int(ar_onset.n_interpolate_['mag']),
    },
    'response': {
        'loss':               ar_response.loss_['mag'],
        'consensus':          consensus_grid,
        'n_interpolate':      n_interpolate_grid,
        'best_consensus':     float(ar_response.consensus_['mag']),
        'best_n_interpolate': int(ar_response.n_interpolate_['mag']),
    }
}
with open(cv_loss_path, 'wb') as f:
    pickle.dump(loss_data, f)
print(f"  Saved: {cv_loss_path}")

print('  Saving epochs...')

onset_fif_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_independent_onset-epo.fif')
epochs_onset_clean.save(onset_fif_path, overwrite=True)
print(f"  Saved: {onset_fif_path}")

response_fif_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_independent_offset-epo.fif')
epochs_response_clean.save(response_fif_path, overwrite=True)
print(f"  Saved: {response_fif_path}")

print(f'\n{"="*60}')
print(f'COMPLETE: {subj}')
print(f'{"="*60}')
