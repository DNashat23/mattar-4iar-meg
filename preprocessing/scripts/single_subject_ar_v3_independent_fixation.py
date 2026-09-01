"""
Single-Subject AutoReject v3 Pipeline -- Fixation-Locked Epochs (Localizer, Independent)
Designed to be called by the SLURM job array with a subject ID argument.

Loads the post-ICA localizer raw and the subject's *_localizer_fixations.csv
(one row per fixation during board presentation). For every fixation whose
trial was correct/responded AND whose fixated square was part of the winning
answer (in_answer == True), three timepoints are derived -- onset, median
(midpoint), and offset of that fixation, each relative to the trial's own
161 (board onset) event -- and epoched independently (-500ms..+500ms, 1001ms).
Each of the three fixation-timepoint types (onset/median/offset) is its own
independent stream: independently epoched, independently AutoRejected, no
pairing between them, exactly mirroring single_subject_ar_v3_independent.py's
onset/response independence, just extended to three streams instead of two.

LOGFILE INPUT: if {subj}_localizer_logfile.csv exists at the base of the
subject's folder, it's used as the starting logfile (this naming implies
radio-noise removal already trimmed it to match the raw data -- see
remove_radio_noise.py). Otherwise falls back to the standard
{subj}_logfile.csv. Both have identical column structure, so no other
logic needs to branch on which one was loaded.

TRIAL MATCHING: the fixations CSV has no totaltrialnum column, and
totaltrialnum is NOT used anywhere in this matching (confirmed: it isn't used
for filtering in single_subject_ar_v3.py, _main.py, or _independent.py either
-- only ever as a human-readable label in the old interactive fixation
scripts). Positional indexing (matching logfile row i to raw event i) still
works for the LOGFILE-TO-RAW-EVENTS join, because that stays strictly 1:1
even after radio-noise removal (both sides are trimmed together, by
remove_radio_noise.py, using the same trial indices). But a trial can have
zero, one, or many fixations, so the FIXATIONS-CSV-TO-TRIAL join can't be
positional at all -- it uses (task, block, trial) as a label-based key instead,
which (unlike a positional/totaltrialnum-style index) survives arbitrary
upstream row removal unchanged, since surviving rows keep their original
(task, block, trial) values no matter how many other rows were dropped before
them. 'task' is included because most subjects' logfiles contain both
'localizer' and 'main' rows with overlapping block/trial numbers.

Each fixation is checked independently per timepoint-type against
[trial's 161 sample, trial's 164 sample] -- a fixation can contribute fewer
than 3 epochs if e.g. only its offset-timepoint falls out of bounds, even
though its onset/median timepoints are fine.

Baseline (-101..-1ms relative to that trial's 160 event -- the PROMPT, not
board onset 161) is computed directly from raw for every trial in the
canonical trial list, before AutoReject runs, and applied per-trial to
every fixation epoch (any type) belonging to that trial -- same mechanism
as single_subject_ar_v3_independent.py's current baseline design (see that
script's docstring for why 160 instead of 161 -- a window ending at 161
sits against the board-onset boundary -- and why -101..-1ms instead of
-100..0ms -- MNE's tmax is inclusive, so -100..0 would include the sample
AT the 160 trigger itself). No divergence from that design here, just
retargeted to the corrected anchor event. Fixation sample offsets
themselves (onset_161_sample below) are still computed relative to each
trial's own 161 event, unaffected by this -- fixations only make sense
relative to when the board actually appears, since that's what's being
fixated on; only the BASELINE anchor moved, not the fixation-timing anchor.

Usage:
    python single_subject_ar_v3_independent_fixation.py --subject R2649 --base_dir /scratch/djn8068/meg_project/data
"""

import argparse
import glob
import pickle
import os
import sys

import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import LabelEncoder
from autoreject import AutoReject

mne.set_log_level(verbose='WARNING')

FIXATION_EVENT_ID = 9999
FIXATION_HALF_WINDOW = 0.5  # seconds on either side of each fixation timepoint
FIXATION_TYPES = ['onset', 'median', 'offset']

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description='AutoReject v3 for a single subject -- independent fixation-locked epochs')
parser.add_argument('--subject', type=str, required=True,
                    help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                    default='/scratch/djn8068/meg_project/data',
                    help='Base directory containing subject folders')
parser.add_argument('--debug', action='store_true',
                    help='Stop before AutoReject (which is slow) -- run everything through '
                         'event matching, fixation filtering, epoch construction, and baseline, '
                         'print diagnostics, and save *_DEBUG-epo.fif files for manual inspection')
args = parser.parse_args()

subj     = args.subject
base_dir = args.base_dir

print(f'\n{"="*60}')
print(f'PROCESSING (independent fixation epochs): {subj}')
print(f'{"="*60}')
print(f'Base dir: {base_dir}')

# =============================================================================
# PATHS
# =============================================================================

data_dir    = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
raw_path    = os.path.join(preproc_dir, f'{subj}_localizer_ICA-raw_final.fif')

if not os.path.exists(raw_path):
    print(f'ERROR: ICA file not found at {raw_path}')
    sys.exit(1)

fixation_csv_matches = glob.glob(os.path.join(data_dir, f'{subj}_*_localizer_fixations.csv'))
if len(fixation_csv_matches) != 1:
    print(f'ERROR: Expected exactly 1 fixations CSV for {subj}, found {len(fixation_csv_matches)}: '
          f'{fixation_csv_matches}')
    sys.exit(1)
fixation_csv_path = fixation_csv_matches[0]

cv_loss_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_cv_loss.pkl')
if os.path.exists(cv_loss_path):
    print(f'SKIPPING {subj}: ar_v3_independent_fixation cv_loss file already exists')
    sys.exit(0)

# =============================================================================
# CONFIGURATION
# =============================================================================

flat_criteria = dict(mag=1e-15)

le = LabelEncoder()
le.fit(['none', '2', '2gap', '3', '4'])

# =============================================================================
# LOAD RAW DATA
# =============================================================================

print(f'  Loading {raw_path}')
raw = mne.io.read_raw_fif(raw_path, preload=True)
sfreq = raw.info['sfreq']
events = mne.find_events(raw, min_duration=0.0025)
picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)

events[:, 0] += 60  # photodiode realignment

# =============================================================================
# LOAD LOGFILE
# =============================================================================

localizer_logfile_path = os.path.join(data_dir, f'{subj}_localizer_logfile.csv')
standard_logfile_path  = os.path.join(data_dir, f'{subj}_logfile.csv')

if os.path.exists(localizer_logfile_path):
    logfile_path = localizer_logfile_path
    print(f'  Using radio-noise-adjusted logfile: {logfile_path}')
elif os.path.exists(standard_logfile_path):
    logfile_path = standard_logfile_path
    print(f'  Using standard logfile: {logfile_path}')
else:
    print(f'ERROR: Logfile not found: {standard_logfile_path}')
    sys.exit(1)
logfile = pd.read_csv(logfile_path)

# =============================================================================
# TRIAL SELECTION (identical logic to single_subject_ar_v3_independent.py)
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
logfile = logfile.copy()
logfile['response_sample'] = filtered_events_164[:, 0].astype(int)
logfile['coded_ground_truth'] = le.transform(logfile['ground_truth'])

K = len(logfile)
print(f"    Canonical trial list (K): {K}")

# (task, block, trial) -> position i in the canonical K-trial list. Label-based,
# not positional -- survives arbitrary upstream row removal (radio noise,
# no-response, incorrect) since surviving rows keep their original
# (task, block, trial) values regardless of how many other rows were dropped.
# 'task' is included even though `logfile` is already filtered to localizer-only
# at this point (defensive -- most subjects' logfiles mix 'main' and 'localizer'
# rows with overlapping block/trial numbers, so this key would collide across
# tasks if this lookup-building pattern ever got reused somewhere less filtered).
trial_lookup = {
    (row['task'], int(row['block']), int(row['trial'])): i
    for i, row in logfile.iterrows()
}

# =============================================================================
# PER-TRIAL BASELINE (raw-sourced, computed before AutoReject runs)
# =============================================================================

print('\n  Computing per-trial baseline (-101..-1ms relative to prompt, event 160)...')
baseline_epochs = mne.Epochs(
    raw, filtered_events_160,
    event_id={'160': 160},
    tmin=-0.101, tmax=-0.001,
    baseline=None,
    picks=picks_meg,
    decim=1,
    preload=True,
    reject=None,
    flat=flat_criteria
)
baseline_flat_dropped = [i for i, log in enumerate(baseline_epochs.drop_log) if log]
if baseline_flat_dropped:
    raise RuntimeError(
        f"{len(baseline_flat_dropped)} trial(s) dropped by FLAT criteria while computing "
        f"the baseline window, before any fixation epoching happened: trial indices "
        f"{baseline_flat_dropped}. This script assumes flat drops never occur; if this "
        f"fires, a fallback needs to be added for these trial(s)."
    )
baseline_by_trial = baseline_epochs.get_data().mean(axis=-1)  # shape (K, n_channels)

# =============================================================================
# LOAD AND FILTER FIXATIONS CSV
# =============================================================================

print(f'\n  Loading fixations: {fixation_csv_path}')
fixations = pd.read_csv(fixation_csv_path)
print(f'    Total fixation rows: {len(fixations)}')

fixations = fixations[fixations['acc'] == True]
print(f'    After acc == True: {len(fixations)}')
fixations = fixations[fixations['in_answer'] == True]
print(f'    After in_answer == True: {len(fixations)}')
fixations = fixations.reset_index(drop=True)

# =============================================================================
# MATCH FIXATIONS TO TRIALS, COMPUTE ONSET/MEDIAN/OFFSET SAMPLES
# =============================================================================

print('\n  Matching fixations to trials and computing timepoints...')

# fixation_records[type] holds dicts: sample, trial_position, fixation_row_id, metadata_row
fixation_records = {t: [] for t in FIXATION_TYPES}
unmatched_trials = []
out_of_bounds_count = {t: 0 for t in FIXATION_TYPES}
n_matched_rows = 0

for row_id, frow in fixations.iterrows():
    key = (frow['task'], int(frow['block']), int(frow['trial']))
    if key not in trial_lookup:
        unmatched_trials.append(key)
        continue
    n_matched_rows += 1

    trial_pos = trial_lookup[key]
    onset_161_sample = int(filtered_events_161[trial_pos, 0])
    trial_164_sample = int(filtered_events_164[trial_pos, 0])

    # Cross-file consistency: the fixation's own trial-level fields should
    # agree with the (independently-loaded) logfile row for the same trial.
    # A disagreement means the two files are misaligned for this trial --
    # a real problem, not something to filter around silently.
    logfile_row = logfile.iloc[trial_pos]
    if frow['ground_truth'] != logfile_row['ground_truth']:
        raise RuntimeError(
            f"ground_truth mismatch for {key}: fixations CSV says "
            f"'{frow['ground_truth']}', logfile says '{logfile_row['ground_truth']}'. "
            f"Fixations CSV and logfile appear to be misaligned for this trial."
        )
    if frow['board_shown'] != logfile_row['board_shown']:
        raise RuntimeError(
            f"board_shown mismatch for {key}: fixations CSV says "
            f"'{frow['board_shown']}', logfile says '{logfile_row['board_shown']}'. "
            f"Fixations CSV and logfile appear to be misaligned for this trial."
        )

    onset_s  = frow['fix_onset_s']
    offset_s = frow['fix_offset_s']
    if onset_s > offset_s:
        raise RuntimeError(
            f"fix_onset_s ({onset_s}) > fix_offset_s ({offset_s}) for fixation row {row_id} "
            f"(trial {key}) -- a fixation's offset can't precede its own onset. "
            f"This indicates a data anomaly in the fixations CSV, not something to silently work around."
        )
    median_s = (onset_s + offset_s) / 2.0

    for ftype, t_s in [('onset', onset_s), ('median', median_s), ('offset', offset_s)]:
        sample = onset_161_sample + int(round(t_s * sfreq))
        if sample < onset_161_sample or sample > trial_164_sample:
            out_of_bounds_count[ftype] += 1
            continue
        fixation_records[ftype].append({
            'sample': sample,
            'trial_position': trial_pos,
            'fixation_row_id': row_id,
            'metadata_row': frow,
        })

if unmatched_trials:
    print(f"    WARNING: {len(unmatched_trials)} fixation row(s) referenced a (block, trial) "
          f"not in the canonical trial list: {unmatched_trials[:20]}"
          f"{' ...' if len(unmatched_trials) > 20 else ''}")
else:
    print("    All fixation rows matched to a canonical trial.")
print("    ground_truth / board_shown cross-checked against logfile for every matched "
      "trial -- no mismatches (would have raised otherwise).")

for ftype in FIXATION_TYPES:
    n_candidate = len(fixation_records[ftype])
    n_oob = out_of_bounds_count[ftype]
    print(f"    {ftype}: {n_candidate} candidate epoch(s), {n_oob} excluded as out-of-bounds")
    # Internal arithmetic consistency: every matched row contributes exactly
    # one in-bounds-or-excluded outcome per type.
    assert n_candidate + n_oob == n_matched_rows, (
        f"{ftype}: candidate ({n_candidate}) + out-of-bounds ({n_oob}) != "
        f"matched rows ({n_matched_rows})"
    )

# Final defensive check: every record that made it this far should have
# acc == True and in_answer == True in its own metadata, re-verified directly
# rather than just trusting the earlier DataFrame filters worked correctly.
for ftype in FIXATION_TYPES:
    for r in fixation_records[ftype]:
        m = r['metadata_row']
        assert m['acc'] == True, f"{ftype} record for row {r['fixation_row_id']} has acc != True"
        assert m['in_answer'] == True, f"{ftype} record for row {r['fixation_row_id']} has in_answer != True"
print(f"    Verified acc == True and in_answer == True for every candidate epoch, all {len(FIXATION_TYPES)} types.")

# =============================================================================
# BUILD, BASELINE, AND AUTOREJECT EACH FIXATION-TYPE STREAM INDEPENDENTLY
# =============================================================================

epochs_clean_by_type   = {}
reject_log_by_type     = {}
ar_by_type              = {}
survived_row_ids_by_type = {t: set() for t in FIXATION_TYPES}

for ftype in FIXATION_TYPES:
    records = fixation_records[ftype]
    if not records:
        print(f'\n  Skipping {ftype}: no candidate epochs')
        continue

    print(f'\n  === {ftype.upper()} fixation epochs ===')
    records_sorted = sorted(records, key=lambda r: r['sample'])
    events_arr = np.array([[r['sample'], 0, FIXATION_EVENT_ID] for r in records_sorted])

    ep_full = mne.Epochs(
        raw, events_arr,
        event_id={'fixation': FIXATION_EVENT_ID},
        tmin=-FIXATION_HALF_WINDOW, tmax=FIXATION_HALF_WINDOW,
        baseline=None,
        picks=picks_meg,
        decim=1,
        preload=True,
        reject=None,
        flat=flat_criteria
    )

    kept_records = [records_sorted[k] for k in ep_full.selection]
    n_flat_dropped = len(events_arr) - len(ep_full)
    if n_flat_dropped:
        print(f'    WARNING: {n_flat_dropped} {ftype} epoch(s) dropped by flat criteria')

    baseline_vals = baseline_by_trial[[r['trial_position'] for r in kept_records]]
    ep_full._data -= baseline_vals[:, :, None]

    meta_rows = []
    for r in kept_records:
        row = r['metadata_row'].copy()
        row['fixation_type']   = ftype
        row['totaltrialnum']   = logfile.iloc[r['trial_position']]['totaltrialnum']
        row['response_sample'] = logfile.iloc[r['trial_position']]['response_sample']
        row['coded_ground_truth'] = logfile.iloc[r['trial_position']]['coded_ground_truth']
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    ep_full.metadata = meta_df

    if args.debug:
        print(f'    [DEBUG] {ftype}: {len(ep_full)} epochs, data shape {ep_full.get_data().shape}')
        print(f'    [DEBUG] metadata columns: {list(meta_df.columns)}')

        # Rigorous baseline check: independently recompute what the FIRST kept
        # epoch should look like straight from raw (slice raw around its
        # fixation sample, subtract that trial's own baseline_by_trial value)
        # and compare against what mne.Epochs + our subtraction actually
        # produced. This is a real correctness check, not a superficial one --
        # a mean over some arbitrary window of the fixation epoch's own time
        # axis wouldn't mean anything, since that axis is centered on the
        # fixation timepoint, not the trial's 161 onset.
        r0 = kept_records[0]
        n_samples_half = int(round(FIXATION_HALF_WINDOW * sfreq))
        raw_slice = raw.get_data(
            picks=picks_meg,
            start=r0['sample'] - n_samples_half,
            stop=r0['sample'] + n_samples_half + 1,
        )
        expected = raw_slice - baseline_by_trial[r0['trial_position']][:, None]
        actual = ep_full.get_data()[0]
        max_abs_diff = np.abs(expected - actual).max()
        print(f'    [DEBUG] baseline recheck on first epoch (trial_position={r0["trial_position"]}, '
              f'sample={r0["sample"]}): max abs diff vs independently recomputed = {max_abs_diff:.3e} '
              f'({"OK" if max_abs_diff < 1e-20 else "MISMATCH"})')

        print(f'    [DEBUG] sample metadata row:')
        print(meta_df.iloc[0].to_string())
        debug_path = os.path.join(
            preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_{ftype}_DEBUG-epo.fif')
        ep_full.save(debug_path, overwrite=True)
        print(f'    [DEBUG] Saved: {debug_path}')
        continue

    print(f'    Running AutoReject on {len(ep_full)} {ftype} epochs...')
    ar = AutoReject(
        n_interpolate=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
        consensus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        random_state=42,
        n_jobs=-1,
        thresh_method='bayesian_optimization',
        verbose=True
    )
    ep_clean, reject_log = ar.fit_transform(ep_full, return_log=True)

    print(f'    Learned n_interpolate: {ar.n_interpolate_}')
    print(f'    Learned consensus: {ar.consensus_}')
    print(f'    Epochs dropped: {reject_log.bad_epochs.sum()}')
    print(f'    Final {ftype} epoch count: {len(ep_clean)}')

    epochs_clean_by_type[ftype] = ep_clean
    reject_log_by_type[ftype]   = reject_log
    ar_by_type[ftype]           = ar

    surviving_kept_records = [kept_records[k] for k in np.where(~reject_log.bad_epochs)[0]]
    survived_row_ids_by_type[ftype] = set(r['fixation_row_id'] for r in surviving_kept_records)

if args.debug:
    print(f'\n{"="*60}')
    print(f'DEBUG RUN COMPLETE (stopped before AutoReject): {subj}')
    print('No rejection_report.p / cv_loss.pkl / filtered_fixations.csv were written --')
    print('those filenames are reserved for the real run and must not be pre-empted,')
    print('or the skip-guard would think a real run already finished.')
    print(f'{"="*60}')
    sys.exit(0)

# =============================================================================
# ANNOTATE FIXATIONS CSV WITH PER-TYPE SURVIVABILITY
# =============================================================================

for ftype in FIXATION_TYPES:
    col = f'include_{ftype}_epoch'
    fixations[col] = 'no'
    fixations.loc[list(survived_row_ids_by_type[ftype]), col] = 'yes'

annotated_path = os.path.join(
    preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_filtered_fixations.csv')
fixations.to_csv(annotated_path, index=False)
print(f'\n  Saved: {annotated_path}')

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

rejection_report = {'total_candidate_fixation_rows': len(fixations)}
loss_data = {}

for ftype in FIXATION_TYPES:
    if ftype not in epochs_clean_by_type:
        continue
    ar = ar_by_type[ftype]
    reject_log = reject_log_by_type[ftype]

    rejection_report[ftype] = {
        'n_candidate_epochs': len(fixation_records[ftype]),
        'n_out_of_bounds':    out_of_bounds_count[ftype],
        'n_rejected':         int(reject_log.bad_epochs.sum()),
        'final_epoch_count':  len(epochs_clean_by_type[ftype]),
    }

    n_interpolate_grid = np.array([1, 2, 4, 8, 12, 16, 20, 24, 28, 32])
    consensus_grid      = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    loss_data[ftype] = {
        'loss':               ar.loss_['mag'],
        'consensus':          consensus_grid,
        'n_interpolate':      n_interpolate_grid,
        'best_consensus':     float(ar.consensus_['mag']),
        'best_n_interpolate': int(ar.n_interpolate_['mag']),
    }

    log_path = os.path.join(
        preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_reject_log_{ftype}.npz')
    if os.path.exists(log_path):
        os.remove(log_path)
    reject_log.save(log_path)

    epo_path = os.path.join(
        preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_{ftype}-epo.fif')
    epochs_clean_by_type[ftype].save(epo_path, overwrite=True)
    print(f'  Saved: {epo_path}')

report_path = os.path.join(
    preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_rejection_report.p')
with open(report_path, 'wb') as f:
    pickle.dump(rejection_report, f)
print(f'  Saved: {report_path}')

with open(cv_loss_path, 'wb') as f:
    pickle.dump(loss_data, f)
print(f'  Saved: {cv_loss_path}')

print(f'\n{"="*60}')
print(f'COMPLETE: {subj}')
print(f'{"="*60}')
