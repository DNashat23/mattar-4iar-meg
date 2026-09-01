"""
Single-Subject AutoReject v3 Pipeline -- Fixation-Locked Epoching (Main Task)
Interactive script, run manually per subject (NOT a SLURM job array target --
it requires typed input at the terminal for fixation timepoints).

Loads the post-ICA main-task raw, applies the same trial-selection filtering
(no response) and builds the same starting epoch batch as
single_subject_ar_v3_main.py, then prints which trials were dropped. From
there it either:
  (1) creates fixation-locked epochs (500ms on either side of user-entered
      fixation timepoints, in ms relative to board onset) for every
      surviving trial,
  (2) creates fixation-locked epochs for a user-specified subset of trials, or
  (3) saves the un-rejected starting epoch batch so a later script can run
      AutoReject on it, unchanged from the current pipeline.

Each fixation the user enters becomes its own epoch row (-500ms to +500ms
around that timepoint), carrying a copy of that trial's full metadata plus
'fixation_time_ms' and 'fixation_index_in_trial'. A trial with 5 fixations
therefore contributes 5 rows to the saved epochs, all sharing the same
trial-level metadata.

Usage:
    python single_subject_ar_v3_main_fixation.py --subject R2649 --base_dir /path/to/data
"""

import argparse
import pandas as pd
import mne
import os
import sys
import numpy as np

mne.set_log_level(verbose='WARNING')

FIXATION_EVENT_ID = 9999
FIXATION_HALF_WINDOW = 0.5  # seconds on either side of each fixation

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description='Fixation-locked epoching prep -- Main Task')
parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                     default='/scratch/djn8068/meg_project/data',
                     help='Base directory containing subject folders')
args = parser.parse_args()

subj     = args.subject
base_dir = args.base_dir

print(f'\n{"="*60}')
print(f'FIXATION EPOCHING PREP (main): {subj}')
print(f'{"="*60}')

# =============================================================================
# PATHS
# =============================================================================

data_dir    = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
raw_path    = os.path.join(preproc_dir, f'{subj}_main_ICA_no_ocular-raw.fif')

if not os.path.exists(raw_path):
    print(f'ERROR: ICA file not found at {raw_path}')
    sys.exit(1)

# =============================================================================
# CONFIGURATION  (mirrors single_subject_ar_v3_main.py)
# =============================================================================

TMIN              = -0.1   # 100ms before board onset
TMAX              = 15.0   # 15s after board onset
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

events_163 = events[events[:, 2] == 163]  # board onset -- one per trial
events_164 = events[events[:, 2] == 164]  # response -- one per responded trial

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
# TRIAL SELECTION  (identical logic to single_subject_ar_v3_main.py)
# =============================================================================

print('  Selecting trials...')

logfile_main = logfile[logfile['task'] == 'main'].reset_index(drop=True)
all_totaltrialnum = logfile_main['totaltrialnum'].tolist()
print(f'    Main task trials: {len(logfile_main)}')

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
          f'but {len(events_164)} 164 events -- check event alignment')

logfile_responded = logfile_responded.copy()
logfile_responded['response_sample'] = [
    int(events_164[i, 0]) for i in range(len(events_163_responded))
]

kept_totaltrialnum = set(logfile_responded['totaltrialnum'].tolist())
dropped_totaltrialnum = sorted(set(all_totaltrialnum) - kept_totaltrialnum)
print(f"\n  Trials dropped (no response): {len(dropped_totaltrialnum)}")
print(f"    totaltrialnum values dropped: {dropped_totaltrialnum}")

# =============================================================================
# CREATE STARTING EPOCH BATCH  (identical to single_subject_ar_v3_main.py, pre-AutoReject)
# =============================================================================

print('\n  Creating starting epoch batch...')
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
print(f'  Epochs after flat rejection: {len(epochs_full)}')

onset_sample_by_position = events_163_responded[:, 0]

# =============================================================================
# FIXATION ENTRY
# =============================================================================

def prompt_int(prompt_text):
    while True:
        raw_val = input(prompt_text).strip()
        try:
            return int(raw_val)
        except ValueError:
            print('    Please enter a whole number.')

def prompt_fixation_value(prompt_text):
    while True:
        raw_val = input(prompt_text).strip()
        try:
            return float(raw_val)
        except ValueError:
            print('    Please enter a number (or 0 to move on).')

def collect_fixations_for_trial(totaltrialnum):
    fixations = []
    idx = 1
    print(f"\n  Trial totaltrialnum={int(totaltrialnum)} -- enter fixation timepoints "
          f"in ms since board onset, 0 to move to next trial:")
    while True:
        val = prompt_fixation_value(f"    fixation #{idx} (ms) > ")
        if val == 0:
            break
        fixations.append(val)
        idx += 1
    return fixations

print(f"\n  Valid trial numbers (totaltrialnum) available for fixation entry:")
print(f"    {sorted(int(t) for t in logfile_responded['totaltrialnum'])}")

print("\n  Enter fixations for:")
print("    1 = all trials")
print("    2 = specific trials")
print("    3 = no trials (save starting batch for AutoReject)")
choice = None
while choice not in (1, 2, 3):
    choice = prompt_int("  Choice (1/2/3) > ")

# =============================================================================
# CHOICE 3: SAVE STARTING BATCH, END
# =============================================================================

if choice == 3:
    print('\n  Saving starting epoch batch (pre-AutoReject)...')
    logfile_path_out = os.path.join(preproc_dir, f'{subj}_main_ar_v3_filtered_logfile.csv')
    logfile_responded.to_csv(logfile_path_out, index=False)
    print(f"    Saved: {logfile_path_out}")

    onset_starting_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_starting_onset-epo.fif')
    epochs_full.save(onset_starting_path, overwrite=True)
    print(f"    Saved: {onset_starting_path}")

    print(f'\n{"="*60}')
    print(f'DONE -- starting batch ready for AutoReject: {subj}')
    print(f'{"="*60}')
    sys.exit(0)

# =============================================================================
# CHOICE 1 / 2: FIXATION ENTRY
# =============================================================================

totaltrialnum_to_position = {
    int(row['totaltrialnum']): pos for pos, row in logfile_responded.iterrows()
}

fixation_records = []  # each: dict(sample, metadata_row, fixation_time_ms, fixation_index_in_trial, baseline_mean)

def register_trial_fixations(position, totaltrialnum):
    fixations_ms = collect_fixations_for_trial(totaltrialnum)
    if not fixations_ms:
        return
    onset_sample = int(onset_sample_by_position[position])
    baseline_start = onset_sample - int(round(0.1 * sfreq))
    baseline_data = raw.get_data(picks=picks_meg, start=baseline_start, stop=onset_sample)
    baseline_mean = baseline_data.mean(axis=1, keepdims=True)  # (n_channels, 1)
    metadata_row = logfile_responded.iloc[position]
    for fix_idx, fix_ms in enumerate(fixations_ms, start=1):
        fixation_sample = onset_sample + int(round((fix_ms / 1000.0) * sfreq))
        fixation_records.append({
            'sample': fixation_sample,
            'metadata_row': metadata_row,
            'fixation_time_ms': fix_ms,
            'fixation_index_in_trial': fix_idx,
            'baseline_mean': baseline_mean,
        })

if choice == 1:
    for position in range(len(logfile_responded)):
        totaltrialnum = logfile_responded.iloc[position]['totaltrialnum']
        register_trial_fixations(position, totaltrialnum)

elif choice == 2:
    while True:
        trial_input = prompt_int("\n  Enter trial number (totaltrialnum), or 0 to finish > ")
        if trial_input == 0:
            break
        if trial_input not in totaltrialnum_to_position:
            print(f"    Trial {trial_input} not in the surviving trial set -- try again.")
            continue
        register_trial_fixations(totaltrialnum_to_position[trial_input], trial_input)

if not fixation_records:
    print('\n  No fixations entered -- nothing to save. Exiting without writing an epochs file.')
    sys.exit(0)

# =============================================================================
# BUILD FIXATION-LOCKED EPOCHS
# =============================================================================

fixation_records.sort(key=lambda r: r['sample'])
events_fix = np.array([[r['sample'], 0, FIXATION_EVENT_ID] for r in fixation_records])

print(f'\n  Building {len(events_fix)} fixation-locked epochs '
      f'(+/-{int(FIXATION_HALF_WINDOW*1000)}ms)...')

epochs_fix = mne.Epochs(
    raw, events_fix,
    event_id={'fixation': FIXATION_EVENT_ID},
    tmin=-FIXATION_HALF_WINDOW,
    tmax=FIXATION_HALF_WINDOW,
    baseline=None,
    picks=picks_meg,
    decim=1,
    preload=True,
    reject=None,
    flat=FLAT_CRITERIA
)

n_dropped = len(events_fix) - len(epochs_fix)
if n_dropped:
    print(f"  WARNING: {n_dropped} fixation epoch(s) dropped (flat criteria or out-of-bounds window)")

kept_records = [fixation_records[i] for i in epochs_fix.selection]

# Apply per-trial baseline correction (reuses the trial's task-onset baseline)
baseline_stack = np.concatenate([r['baseline_mean'][np.newaxis, :, :] for r in kept_records], axis=0)
epochs_fix._data -= baseline_stack

# Attach metadata: one row per fixation, all fields from that trial replicated
metadata_rows = []
for r in kept_records:
    row = r['metadata_row'].copy()
    row['fixation_time_ms'] = r['fixation_time_ms']
    row['fixation_index_in_trial'] = r['fixation_index_in_trial']
    metadata_rows.append(row)
metadata_df = pd.DataFrame(metadata_rows).reset_index(drop=True)
epochs_fix.metadata = metadata_df

# =============================================================================
# SAVE
# =============================================================================

fixation_path = os.path.join(preproc_dir, f'{subj}_main_ar_v3_fixation-epo.fif')
epochs_fix.save(fixation_path, overwrite=True)
print(f"\n  Saved: {fixation_path}")
print(f"  Total fixation epochs saved: {len(epochs_fix)}")

print(f'\n{"="*60}')
print(f'DONE: {subj}')
print(f'{"="*60}')
