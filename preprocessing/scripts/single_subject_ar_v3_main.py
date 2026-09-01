"""
Single-Subject AutoReject v3 Pipeline — Main Task
Designed to be called by the SLURM job array with a subject ID argument.

Epochs are locked to event 163 (board onset), spanning -100ms to +15s.
The absolute MEG sample of event 164 (response) is stored in the metadata
column 'response_sample' for use in downstream response-locked analyses.

Usage:
    python single_subject_ar_v3_main.py --subject R2649 --base_dir /scratch/djn8068/meg_project/data
"""

import argparse
import pandas as pd
import mne
import pickle
import os
import sys
import numpy as np
from autoreject import AutoReject

mne.set_log_level(verbose='WARNING')

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description='AutoReject v3 — Main Task')
parser.add_argument('--subject', type=str, required=True,
                    help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                    default='/scratch/djn8068/meg_project/data',
                    help='Base directory containing subject folders')
args = parser.parse_args()

subj     = args.subject
base_dir = args.base_dir

print(f'\n{"="*60}')
print(f'PROCESSING (main): {subj}')
print(f'{"="*60}')
print(f'Base dir: {base_dir}')

# =============================================================================
# PATHS
# =============================================================================

data_dir    = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
raw_path    = os.path.join(preproc_dir, f'{subj}_main_ICA_no_ocular-raw.fif')

if not os.path.exists(raw_path):
    print(f'ERROR: ICA file not found at {raw_path}')
    sys.exit(1)

# Skip guard: if cv_loss already exists, this subject completed successfully
cv_loss_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_cv_loss.pkl')
if os.path.exists(cv_loss_path):
    print(f'SKIPPING {subj}: main ar_v3 cv_loss already exists')
    sys.exit(0)

# =============================================================================
# CONFIGURATION
# =============================================================================

TMIN              = -0.1   # 100ms before board onset
TMAX              = 15.0   # 15s after board onset
BASELINE          = (-0.1, 0)
FLAT_CRITERIA     = dict(mag=1e-15)
PHOTODIODE_OFFSET = 60     # samples

# =============================================================================
# LOAD RAW DATA
# =============================================================================

print(f'  Loading {raw_path}')
raw   = mne.io.read_raw_fif(raw_path, preload=True)
sfreq = raw.info['sfreq']

events = mne.find_events(raw, min_duration=0.0025)
events[:, 0] += PHOTODIODE_OFFSET

picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)

events_163 = events[events[:, 2] == 163]  # board onset — one per trial
events_164 = events[events[:, 2] == 164]  # response — one per responded trial

print(f'  Board onset events (163): {len(events_163)}')
print(f'  Response events (164):    {len(events_164)}')

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

# Filter to main task rows — i-th row aligns with i-th events_163
logfile_main = logfile[logfile['task'] == 'main'].reset_index(drop=True)
print(f'    Main task trials: {len(logfile_main)}')

# Responded trials: rt, answer_x, and answer_y are all NaN together when no
# response was given — require all three to be present to count as responded.
has_response = (
    logfile_main['rt'].notna()
    & logfile_main['answer_x'].notna()
    & logfile_main['answer_y'].notna()
).values
events_163_responded = events_163[has_response]
logfile_responded    = logfile_main[has_response].reset_index(drop=True)
print(f'    Trials with response: {len(logfile_responded)}')

if len(events_163_responded) != len(events_164):
    print(f'  WARNING: {len(events_163_responded)} responded 163 events '
          f'but {len(events_164)} 164 events — check event alignment')

# =============================================================================
# RESPONSE SAMPLE
# =============================================================================

# Store the absolute MEG sample of the 164 event (post-photodiode-offset)
# for each trial. Downstream response-locked analyses use this to re-align
# epochs to the response rather than to board onset.
logfile_responded = logfile_responded.copy()
logfile_responded['response_sample'] = [
    int(events_164[i, 0]) for i in range(len(events_163_responded))
]

# =============================================================================
# SAVE FILTERED LOGFILE
# =============================================================================

ar_logfile_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_filtered_logfile.csv')
logfile_responded.to_csv(ar_logfile_path, index=False)
print(f'    Saved filtered logfile: {ar_logfile_path}')

# =============================================================================
# CREATE EPOCHS
# =============================================================================

print('  Creating epochs...')
epochs_full = mne.Epochs(
    raw, events_163_responded,
    event_id={'163': 163},
    tmin=TMIN,
    tmax=TMAX,
    baseline=None,
    picks=picks_meg,
    decim=1,
    preload=True,
    reject=None,
    flat=FLAT_CRITERIA
)

flat_dropped = set(i for i, log in enumerate(epochs_full.drop_log) if log)
if flat_dropped:
    print(f'  WARNING: {len(flat_dropped)} epoch(s) dropped by flat criteria: '
          f'{sorted(flat_dropped)}')

kept_indices = np.array([i for i, log in enumerate(epochs_full.drop_log) if not log])
print(f'  Epochs after flat rejection: {len(epochs_full)}')

# =============================================================================
# AUTOREJECT
# =============================================================================

print('\n  Running AutoReject...')
ar = AutoReject(
    n_interpolate=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
    consensus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    random_state=42,
    n_jobs=-1,
    thresh_method='bayesian_optimization',
    verbose=True
)
epochs_clean, reject_log = ar.fit_transform(epochs_full, return_log=True)

print(f'\n  AutoReject summary:')
print(f'    Learned n_interpolate: {ar.n_interpolate_}')
print(f'    Learned consensus:     {ar.consensus_}')
print(f'    Epochs dropped:        {reject_log.bad_epochs.sum()}')
print(f'    Final trial count:     {len(epochs_clean)}')

# =============================================================================
# UPDATE LOGFILE TO SURVIVING TRIALS
# =============================================================================

bad_ar       = set(kept_indices[reject_log.bad_epochs])
all_bad      = flat_dropped | bad_ar
all_indices  = set(range(len(logfile_responded)))
good_indices = sorted(all_indices - all_bad)

logfile_final = logfile_responded.iloc[good_indices].reset_index(drop=True)

if len(epochs_clean) != len(logfile_final):
    print(f'ERROR: Mismatch: epochs={len(epochs_clean)}, logfile={len(logfile_final)}')
    sys.exit(1)

epochs_clean.metadata = logfile_final
print(f'  Final trial count: {len(epochs_clean)}')

# Apply pre-onset baseline correction manually.
# The baseline window is the 100ms before event 163, computed and subtracted
# per trial per channel.
baseline_mask  = (epochs_clean.times >= -0.1) & (epochs_clean.times <= 0)
baseline_means = epochs_clean.get_data()[:, :, baseline_mask].mean(axis=-1, keepdims=True)
epochs_clean._data -= baseline_means

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

# Rejection report
rejection_report = {
    'total_trials':      len(all_indices),
    'all_bad_indices':   sorted(list(all_bad)),
    'good_indices':      good_indices,
    'final_trial_count': len(good_indices)
}
report_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_rejection_report.p')
with open(report_path, 'wb') as f:
    pickle.dump(rejection_report, f)
print(f'  Saved: {report_path}')

# Reject log
log_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_reject_log.npz')
if os.path.exists(log_path):
    os.remove(log_path)
reject_log.save(log_path)
print(f'  Saved: {log_path}')

# CV loss landscape
n_interpolate_grid = np.array([1, 2, 4, 8, 12, 16, 20, 24, 28, 32])
consensus_grid     = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
loss_data = {
    'loss':               ar.loss_['mag'],
    'consensus':          consensus_grid,
    'n_interpolate':      n_interpolate_grid,
    'best_consensus':     float(ar.consensus_['mag']),
    'best_n_interpolate': int(ar.n_interpolate_['mag']),
}
with open(cv_loss_path, 'wb') as f:
    pickle.dump(loss_data, f)
print(f'  Saved: {cv_loss_path}')

# Epoch FIF — single file, 163-locked, -100ms to +15s
onset_fif_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_onset-epo.fif')
epochs_clean.save(onset_fif_path, overwrite=True)
print(f'  Saved: {onset_fif_path}')

print(f'\n{"="*60}')
print(f'COMPLETE: {subj}')
print(f'{"="*60}')
