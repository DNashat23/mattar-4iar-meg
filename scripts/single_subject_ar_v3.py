"""
Single-Subject AutoReject v3 Pipeline
Designed to be called by the SLURM job array with a subject ID argument.

Usage:
    python ar_v3_single_subject.py --subject R2649 --base_dir /scratch/djn8068/meg_project/data
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

parser = argparse.ArgumentParser(description='AutoReject v3 for a single subject')
parser.add_argument('--subject', type=str, required=True,
                    help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                    default='/scratch/djn8068/meg_project/data',
                    help='Base directory containing subject folders')
args = parser.parse_args()

subj     = args.subject
base_dir = args.base_dir

print(f'\n{"="*60}')
print(f'PROCESSING: {subj}')
print(f'{"="*60}')
print(f'Base dir: {base_dir}')

# =============================================================================
# PATHS
# =============================================================================

data_dir   = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
raw_path   = os.path.join(preproc_dir, f'{subj}_localizer_ICA_no_ocular-raw.fif')

# Guard: ICA file must exist
if not os.path.exists(raw_path):
    print(f'ERROR: ICA file not found at {raw_path}')
    sys.exit(1)

# Skip if cv_loss file already exists — means full run completed successfully
cv_loss_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_cv_loss.pkl')
if os.path.exists(cv_loss_path):
    print(f'SKIPPING {subj}: ar_v3 cv_loss file already exists')
    sys.exit(0)

# =============================================================================
# CONFIGURATION
# =============================================================================

epoch_config = {
    "tmin_161": -0.1, "tmax_161": 1.0,
    # "baseline_161": (-0.1, 0),      # unused — baseline is applied manually below
    "tmin_164": -1.0, "tmax_164": 0.1,
    # "baseline_164": (-1.0, -0.9),   # unused — response epochs share the onset epoch's baseline instead, applied manually below
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

events_161 = events[events[:, 2] == 161]
events_164 = events[events[:, 2] == 164]

print(f"    Onset events (161): {len(events_161)}")
print(f"    Offset events (164): {len(events_164)}")

if len(events_161) != len(events_164):
    print(f"    WARNING: Unequal onset and offset events")

filtered_events_161_temp = []
for i in range(len(events_161)):
    if pd.notna(logfile.iloc[i]['acc']):
        filtered_events_161_temp.append(events_161[i])
filtered_events_161_temp = np.array(filtered_events_161_temp)
print(f"    Onset events with responses: {len(filtered_events_161_temp)}")

logfile = logfile[(logfile['task'] == 'localizer') & (logfile['acc'].notna())]
logfile = logfile.reset_index(drop=True)

filtered_events_161 = []
filtered_events_164 = []
for i in range(len(logfile)):
    if logfile.iloc[i]['acc'] == True:
        filtered_events_161.append(filtered_events_161_temp[i])
        filtered_events_164.append(events_164[i])

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

# Save v3 filtered logfile
ar_logfile_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_filtered_logfile.csv')
logfile.to_csv(ar_logfile_path, index=False)
print(f"    Saved filtered logfile: {ar_logfile_path}")

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
# AUTOREJECT v3 — ONSET
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
# AUTOREJECT v3 — RESPONSE
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
# MAP AR BAD EPOCHS BACK TO ORIGINAL INDICES
# =============================================================================

bad_onset_ar      = set(onset_full_kept[reject_log_onset.bad_epochs])
bad_response_ar   = set(response_full_kept[reject_log_response.bad_epochs])
bad_onset_indices = flat_dropped_onset | bad_onset_ar
bad_response_indices = flat_dropped_response | bad_response_ar
all_bad_indices   = bad_onset_indices | bad_response_indices

print(f"\n  Rejection summary:")
print(f"    Bad onset epochs:    {len(bad_onset_indices)}")
print(f"    Bad response epochs: {len(bad_response_indices)}")
print(f"    Total bad trials:    {len(all_bad_indices)}")

# =============================================================================
# ENFORCE TRIAL PAIRING
# =============================================================================

onset_surviving    = onset_full_kept[~reject_log_onset.bad_epochs]
response_surviving = response_full_kept[~reject_log_response.bad_epochs]

response_extra = bad_response_indices - bad_onset_indices
onset_positions_to_drop = sorted([
    int(np.where(onset_surviving == i)[0][0]) for i in response_extra
])
if onset_positions_to_drop:
    epochs_onset_clean.drop(onset_positions_to_drop, reason='PAIRED_REJECTION')
    print(f"    Dropped {len(onset_positions_to_drop)} onset epoch(s) for pairing")

onset_extra = bad_onset_indices - bad_response_indices
response_positions_to_drop = sorted([
    int(np.where(response_surviving == i)[0][0]) for i in onset_extra
])
if response_positions_to_drop:
    epochs_response_clean.drop(response_positions_to_drop, reason='PAIRED_REJECTION')
    print(f"    Dropped {len(response_positions_to_drop)} response epoch(s) for pairing")

all_trial_indices = set(range(len(logfile)))
good_indices = sorted(all_trial_indices - all_bad_indices)
logfile = logfile.iloc[good_indices].reset_index(drop=True)

epochs_onset    = epochs_onset_clean
epochs_response = epochs_response_clean

# Apply pre-onset baseline correction to both epoch types.
# The baseline window is the 100ms before event 161, computed from the onset
# epoch and applied identically to the response epoch so both share the same
# reference signal.
baseline_mask  = (epochs_onset.times >= -0.1) & (epochs_onset.times <= 0)
baseline_means = epochs_onset.get_data()[:, :, baseline_mask].mean(axis=-1, keepdims=True)
epochs_onset._data   -= baseline_means
epochs_response._data -= baseline_means

# =============================================================================
# FINAL VERIFICATION
# =============================================================================

if len(epochs_onset) != len(epochs_response) or len(epochs_onset) != len(logfile):
    print(f'ERROR: Mismatch: onset={len(epochs_onset)}, '
          f'response={len(epochs_response)}, logfile={len(logfile)}')
    sys.exit(1)
print(f"  Final trial count: {len(epochs_onset)}")

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

# Rejection report
rejection_report = {
    'total_trials':        len(all_trial_indices),
    'bad_onset_indices':   sorted(list(bad_onset_indices)),
    'bad_response_indices': sorted(list(bad_response_indices)),
    'all_bad_indices':     sorted(list(all_bad_indices)),
    'good_indices':        good_indices,
    'final_trial_count':   len(good_indices)
}
report_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_rejection_report.p')
with open(report_path, 'wb') as f:
    pickle.dump(rejection_report, f)
print(f"  Saved: {report_path}")

# Reject logs
for log, label in [(reject_log_onset,    'ar_v3_reject_log_onset'),
                   (reject_log_response, 'ar_v3_reject_log_response')]:
    log_path = os.path.join(preproc_dir, f'{subj}_localizer_{label}.npz')
    if os.path.exists(log_path):
        os.remove(log_path)
    log.save(log_path)
print(f"  Saved AR v3 reject logs")

# CV loss landscape — shape: (n_consensus, n_interpolate, n_folds)
# Stored as negative log-likelihood internally; we save as-is and negate when plotting
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
loss_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_cv_loss.pkl')
with open(loss_path, 'wb') as f:
    pickle.dump(loss_data, f)
print(f"  Saved: {loss_path}")

# Add metadata
logfile['coded_ground_truth'] = le.transform(logfile['ground_truth'])
epochs_onset.metadata    = logfile
epochs_response.metadata = logfile

print('  Saving epochs...')

# Onset .fif
onset_fif_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_onset-epo.fif')
epochs_onset.save(onset_fif_path, overwrite=True)
print(f"  Saved: {onset_fif_path}")

# Response .fif
response_fif_path = os.path.join(preproc_dir, f'{subj}_localizer_ar_v3_offset-epo.fif')
epochs_response.save(response_fif_path, overwrite=True)
print(f"  Saved: {response_fif_path}")

print(f'\n{"="*60}')
print(f'COMPLETE: {subj}')
print(f'{"="*60}')