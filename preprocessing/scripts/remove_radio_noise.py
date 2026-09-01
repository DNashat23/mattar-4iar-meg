"""
Radio Noise Removal + Full Preprocessing Pipeline
For subjects whose data contains radio noise contamination.

Replaces preprocessing_meg.py for these subjects. Produces the same
ICA-raw.fif output that ar_v3 expects.

Steps:
  1. Load raw data and calculate baseline SNR
  2. Bandpass filter (1–40 Hz)
  3a. Interactive radio noise annotation and physical segment removal
  3b. Interactive bad channel marking and interpolation (interpolated
      channels are treated as good afterward, not excluded downstream)
  4. Empty room projection (same bad channels interpolated here too,
      before noise projectors are computed)
  5. ICA

Usage: set subj and dtype below, then run interactively.
"""

import mne
import numpy as np
import pandas as pd
import os
import sys
import glob
import pickle
from snr_both import (SNRTracker, calculate_snr_onset, calculate_snr_response,
                      create_temp_epochs_onset, create_temp_epochs_response)

mne.set_log_level(verbose='WARNING')

# =============================================================================
# CONFIGURATION
# =============================================================================

subj  = 'R2737'
dtype = 'localizer'   # 'localizer' or 'main'

base_dir        = '/Users/dariushnashat/Documents/Mattar Lab Internship/'
data_dir        = os.path.join(base_dir, subj)
preproc_dir     = os.path.join(data_dir, 'preprocessing')
empty_room_file = f'{data_dir}/raw/{subj}_EmptyRoom_nr.sqd'
overwrite       = True

onset_event_id    = 161 if dtype == 'localizer' else 163
response_event_id = 164

os.makedirs(preproc_dir, exist_ok=True)

print(f'\n{"="*60}')
print(f'PREPROCESSING (radio noise): {subj} | {dtype}')
print(f'{"="*60}')

# =============================================================================
# STEP 1: Load raw data
# =============================================================================

print('\nSTEP 1: Loading raw data files...')
raw_files = sorted(glob.glob(f'{data_dir}/raw/*{dtype}*.fif'))
if not raw_files:
    print(f'ERROR: No {dtype} raw files found in {data_dir}/raw/')
    sys.exit(1)

raws = []
for f in raw_files:
    print(f'  Loading {os.path.basename(f)}')
    raws.append(mne.io.read_raw_fif(f, preload=True))

raw   = mne.concatenate_raws(raws)
sfreq = raw.info['sfreq']
print(f'  Sampling rate: {sfreq} Hz')
print(f'  Duration: {raw.times[-1]:.1f} seconds')

snr_tracker = SNRTracker(base_dir)
snr_tracker.start_subject(subj)

print('\n  Calculating baseline SNR...')
print('  === Onset Epochs ===')
temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
snr_onset_global, _, _ = calculate_snr_onset(temp_epochs_onset)
snr_tracker.record_snr('onset', 'baseline', snr_onset_global)

print('\n  === Response Epochs ===')
temp_epochs_response = create_temp_epochs_response(raw)
snr_response_global, _, _ = calculate_snr_response(temp_epochs_response)
snr_tracker.record_snr('response', 'baseline', snr_response_global)

# =============================================================================
# STEP 2: Bandpass filter (1–40 Hz)
# =============================================================================

print('\nSTEP 2: Applying frequency filter (1–40 Hz)...')
raw.filter(1, 40, picks='meg', method='iir')

print('  === Onset Epochs ===')
temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
snr_onset_global, _, _ = calculate_snr_onset(temp_epochs_onset)
snr_tracker.record_snr('onset', 'after_filtering', snr_onset_global)

print('\n  === Response Epochs ===')
temp_epochs_response = create_temp_epochs_response(raw)
snr_response_global, _, _ = calculate_snr_response(temp_epochs_response)
snr_tracker.record_snr('response', 'after_filtering', snr_response_global)

# =============================================================================
# LOAD LOGFILE AND VERIFY EVENTS
# =============================================================================

logfile_path = os.path.join(data_dir, f'{subj}_logfile.csv')
if not os.path.exists(logfile_path):
    print(f'ERROR: Logfile not found: {logfile_path}')
    sys.exit(1)

logfile      = pd.read_csv(logfile_path)
logfile_task = logfile[logfile['task'] == dtype].reset_index(drop=True)
print(f'\n  Logfile rows for {dtype}: {len(logfile_task)}')

if dtype == 'localizer':
    has_response = logfile_task['acc'].notna().values
else:
    has_response = logfile_task['answer_x'].notna().values

print('\nVerifying event-logfile alignment...')
events          = mne.find_events(raw, min_duration=0.0025)
events_onset    = events[events[:, 2] == onset_event_id]
events_response = events[events[:, 2] == response_event_id]

print(f'  Onset events    ({onset_event_id}): {len(events_onset)}')
print(f'  Response events ({response_event_id}): {len(events_response)}')
print(f'  Logfile rows:              {len(logfile_task)}')
print(f'  Logfile responded trials:  {has_response.sum()}')


if len(events_onset) != len(logfile_task):
    print(f'ERROR: {len(events_onset)} onset events but {len(logfile_task)} logfile rows — cannot align.')
    sys.exit(1)
if len(events_response) != has_response.sum():
    print(f'ERROR: {len(events_response)} response events but {has_response.sum()} responded logfile rows.')
    sys.exit(1)
print('  Alignment verified.')

## for R2664 
'''
if len(events_onset) != len(logfile_task):
    print(f'ERROR: {len(events_onset)} onset events but {len(logfile_task)} logfile rows — cannot align.')
    sys.exit(1)
if len(events_response) > has_response.sum():                     # ← new branch
    print(f'WARNING: {len(events_response)} response events but {has_response.sum()} responded logfile rows — continuing.')
elif len(events_response) < has_response.sum():                   # ← old check, narrowed
    print(f'ERROR: {len(events_response)} response events but {has_response.sum()} responded logfile rows.')
    sys.exit(1)
print('  Alignment verified.')
'''

responded_trial_indices = np.where(has_response)[0]

# =============================================================================
# STEP 3a: Radio noise annotation and removal
# =============================================================================

print('\nSTEP 3a: Radio noise removal.')
print('Mark contaminated segments as BAD annotations, then CLOSE the window.')
raw.plot(block=True)

bad_annotations = [ann for ann in raw.annotations
                   if 'BAD' in ann['description'] or 'bad' in ann['description']]

if not bad_annotations:
    print('  No bad annotations found — proceeding without radio noise removal.')
    raw_clean        = raw
    logfile_clean    = logfile_task
else:
    print(f'\n  Processing {len(bad_annotations)} bad annotation(s)...')
    bad_trial_indices = set()
    # Collected across ALL annotations before exiting, so a problem with
    # annotation 1 doesn't hide problems with annotations 2, 3, ... — you
    # get every fix you need to make in one pass instead of discovering
    # them one restart at a time. Still hard-exits below if anything's
    # wrong — this only changes WHEN it exits, not WHETHER it exits.
    errors = []

    for ann_idx, ann in enumerate(bad_annotations):
        ann_start_s = ann['onset']
        ann_end_s   = ann['onset'] + ann['duration']
        ann_start   = ann_start_s * sfreq
        ann_end     = ann_end_s   * sfreq

        print(f'\n  Annotation {ann_idx + 1}: {ann_start_s:.3f}s → {ann_end_s:.3f}s '
              f'(duration {ann["duration"]:.3f}s)')

        onset_in_ann    = np.where((events_onset[:, 0]    >= ann_start) &
                                    (events_onset[:, 0]    <= ann_end))[0]
        response_in_ann = np.where((events_response[:, 0] >= ann_start) &
                                    (events_response[:, 0] <= ann_end))[0]

        print(f'    Onset events inside:    {len(onset_in_ann)}  — trial indices {list(onset_in_ann)}')
        print(f'    Response events inside: {len(response_in_ann)}  — response indices {list(response_in_ann)}')

        if len(onset_in_ann) == 0 and len(response_in_ann) == 0:
            print('    No events in this annotation — skipping.')
            continue

        all_in_ann = []
        for i in onset_in_ann:
            all_in_ann.append((events_onset[i, 0], 'onset', int(i)))
        for j in response_in_ann:
            trial_idx = int(responded_trial_indices[j])
            all_in_ann.append((events_response[j, 0], 'response', trial_idx))
        all_in_ann.sort(key=lambda x: x[0])

        first_event = all_in_ann[0]
        last_event  = all_in_ann[-1]
        ann_ok = True

        if first_event[1] == 'response':
            trial_idx  = first_event[2]
            onset_time = events_onset[trial_idx, 0] / sfreq
            msg = (f'Annotation {ann_idx + 1} ({ann_start_s:.3f}s → {ann_end_s:.3f}s): '
                   f'first event is a RESPONSE (trial {trial_idx}). The onset trigger for '
                   f'this trial is at {onset_time:.3f}s, which is BEFORE your annotation '
                   f'start ({ann_start_s:.3f}s). Extend the annotation to before {onset_time:.3f}s.')
            print(f'\n  *** EDGE CASE — {msg} ***')
            errors.append(msg)
            ann_ok = False

        if last_event[1] == 'onset':
            trial_idx = last_event[2]
            if has_response[trial_idx]:
                j_idx = np.where(responded_trial_indices == trial_idx)[0]
                if len(j_idx) > 0:
                    response_time = events_response[j_idx[0], 0] / sfreq
                    msg = (f'Annotation {ann_idx + 1} ({ann_start_s:.3f}s → {ann_end_s:.3f}s): '
                           f'last event is an ONSET (trial {trial_idx}, has response). The '
                           f'response trigger for this trial is at {response_time:.3f}s, which is '
                           f'AFTER your annotation end ({ann_end_s:.3f}s). Extend the annotation '
                           f'to after {response_time:.3f}s.')
                else:
                    msg = (f'Annotation {ann_idx + 1} ({ann_start_s:.3f}s → {ann_end_s:.3f}s): '
                           f'last event is an ONSET (trial {trial_idx}, has response). The response '
                           f'trigger falls after the annotation end ({ann_end_s:.3f}s). Extend the '
                           f'annotation to cover the response.')
                print(f'\n  *** EDGE CASE — {msg} ***')
                errors.append(msg)
                ann_ok = False

        if not ann_ok:
            # Skip adding this annotation's trials — its bounds are wrong,
            # so keep checking the rest and report everything at the end.
            continue

        for _, event_type, trial_idx in all_in_ann:
            bad_trial_indices.add(trial_idx)

    if errors:
        print(f'\n{"="*64}')
        print(f'  {len(errors)} of {len(bad_annotations)} annotation(s) need to be fixed:')
        print(f'{"="*64}')
        for i, msg in enumerate(errors, 1):
            print(f'  {i}. {msg}')
        print(f'\n  Restart the script and redraw ALL annotations, fixing the issue(s) above.')
        sys.exit(1)

    print(f'\n  Total contaminated trials: {len(bad_trial_indices)}')

    good_indices  = sorted(set(range(len(logfile_task))) - bad_trial_indices)
    logfile_clean = logfile_task.iloc[good_indices].reset_index(drop=True)
    print(f'  Remaining trials: {len(logfile_clean)}')

    original_duration = raw.times[-1]
    bad_intervals     = sorted([(ann['onset'], ann['onset'] + ann['duration'])
                                 for ann in bad_annotations])
    good_segments = []
    current_time  = 0.0
    for start_bad, end_bad in bad_intervals:
        if start_bad > current_time:
            good_segments.append((current_time, start_bad))
        current_time = max(current_time, end_bad)
    if current_time < original_duration:
        good_segments.append((current_time, original_duration))

    print(f'  Keeping {len(good_segments)} clean segment(s)...')
    good_raws = [raw.copy().crop(tmin=s, tmax=e) for s, e in good_segments]
    raw_clean  = mne.concatenate_raws(good_raws) if len(good_raws) > 1 else good_raws[0]
    print(f'  Duration after removal: {raw_clean.times[-1]:.1f}s')

# Save filtered logfile
logfile_out = os.path.join(preproc_dir, f'{subj}_{dtype}_nr_filtered_logfile.csv')
logfile_clean.to_csv(logfile_out, index=False)
print(f'  Saved filtered logfile: {logfile_out}')

print('\n  === Onset Epochs (post radio noise removal) ===')
temp_epochs_onset = create_temp_epochs_onset(raw_clean, onset_event_id=onset_event_id)
snr_onset_global, _, _ = calculate_snr_onset(temp_epochs_onset)
snr_tracker.record_snr('onset', 'after_radio_noise', snr_onset_global)

print('\n  === Response Epochs (post radio noise removal) ===')
temp_epochs_response = create_temp_epochs_response(raw_clean)
snr_response_global, _, _ = calculate_snr_response(temp_epochs_response)
snr_tracker.record_snr('response', 'after_radio_noise', snr_response_global)

# =============================================================================
# STEP 3b: Bad channel detection and interpolation
# =============================================================================

print('\nSTEP 3b: Bad channel detection and interpolation...')
print('  Opening interactive plot for bad channel marking...')
raw_clean.plot()
input('  Mark bad channels, then press Enter to continue...')

bad_channels = raw_clean.info['bads'].copy()
print(f'  Bad channels marked: {bad_channels}')

bad_channels_path = f'{preproc_dir}/{subj}_{dtype}_bad_channels.p'
with open(bad_channels_path, 'wb') as f:
    pickle.dump(bad_channels, f)
print(f'  Bad channels saved to: {bad_channels_path}')

if bad_channels:
    raw_clean.interpolate_bads()
    print('  Bad channels interpolated (now treated as good for all downstream steps)')
else:
    print('  No bad channels marked')

print('  === Onset Epochs ===')
temp_epochs_onset = create_temp_epochs_onset(raw_clean, onset_event_id=onset_event_id)
snr_onset_global, _, _ = calculate_snr_onset(temp_epochs_onset)
snr_tracker.record_snr('onset', 'after_interpolation', snr_onset_global)

print('\n  === Response Epochs ===')
temp_epochs_response = create_temp_epochs_response(raw_clean)
snr_response_global, _, _ = calculate_snr_response(temp_epochs_response)
snr_tracker.record_snr('response', 'after_interpolation', snr_response_global)

# =============================================================================
# STEP 4: Empty room projection
# =============================================================================

print('\nSTEP 4: Empty room projection...')
print(f'  Rank before empty room projection: {mne.compute_rank(raw_clean)}')
try:
    empty_room_raw = mne.io.read_raw_kit(
        empty_room_file, mrk=None, elp=None, hsp=None,
        stim='<', preload=True, verbose=True
    )
    print(f'  Empty room duration: {empty_room_raw.times[-1]:.1f} seconds')

    empty_room_raw.filter(1, 40, picks='meg', method='iir')

    # Empty room recordings have no digitization of their own (no head
    # present, loaded with mrk=elp=hsp=None) so MEG bad-channel
    # interpolation can't fit a sphere model without borrowing the
    # subject's head position/shape from this same session. 'dig' is a
    # protected Info field, so it has to go through _unlock().
    with empty_room_raw.info._unlock():
        empty_room_raw.info['dig'] = raw_clean.info['dig']
        empty_room_raw.info['dev_head_t'] = raw_clean.info['dev_head_t']

    empty_room_raw.info['bads'] = bad_channels.copy()
    if bad_channels:
        empty_room_raw.interpolate_bads()
        print(f'  Bad channels interpolated in empty room: {bad_channels}')
    else:
        print('  No bad channels to interpolate in empty room')

    print('  Computing environmental noise projectors...')
    empty_room_projs = mne.compute_proj_raw(
        empty_room_raw,
        duration=2,
        n_grad=0,
        n_mag=0.9,
        reject={'mag': 3e-12},
        flat={'mag': 1e-15}
    )

    mne.viz.plot_projs_topomap(empty_room_projs, colorbar=True,
                               vlim='joint', info=empty_room_raw.info)

    print(f'  Created {len(empty_room_projs)} noise projectors')
    for i, proj in enumerate(empty_room_projs):
        print(f'    {i+1}. {proj["desc"]}')

    raw_clean.add_proj(empty_room_projs, remove_existing=False)
    raw_clean.apply_proj()
    print('  Applied projectors to main data')
    print(f'  Rank after empty room projection: {mne.compute_rank(raw_clean)}')

except Exception as e:
    print(f'  Error with empty room data: {e}')

print('  === Onset Epochs ===')
temp_epochs_onset = create_temp_epochs_onset(raw_clean, onset_event_id=onset_event_id)
snr_onset_global, _, _ = calculate_snr_onset(temp_epochs_onset)
snr_tracker.record_snr('onset', 'after_empty_room', snr_onset_global)

print('\n  === Response Epochs ===')
temp_epochs_response = create_temp_epochs_response(raw_clean)
snr_response_global, _, _ = calculate_snr_response(temp_epochs_response)
snr_tracker.record_snr('response', 'after_empty_room', snr_response_global)

# =============================================================================
# STEP 5: ICA
# =============================================================================

print('\nSTEP 5: ICA artifact removal...')
print(f'  Rank before ICA: {mne.compute_rank(raw_clean)}')

ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
print('  Fitting ICA...')
ica.fit(raw_clean, reject={'mag': 5e-12})
ica.save(f'{preproc_dir}/{subj}_{dtype}_icasolution-ica.fif', overwrite=overwrite)

'''
print('\n  ICA component explained variance:')
var_explained = []
for idx in range(ica.n_components_):
    ratio = ica.get_explained_variance_ratio(raw_clean, components=idx).get('mag', 0.0)
    var_explained.append((idx, ratio))
for comp_idx, var in sorted(var_explained, key=lambda x: x[1], reverse=True):
    print(f'    Component {comp_idx:3d}: {var * 100:.3f}%')
'''   

print('\n  Review components and mark artifacts...')
ica.plot_sources(raw_clean)
input('  Press Enter to see ICA topographies...')
ica.plot_components()

artifact_components = ica.exclude.copy()
print(f'  Excluding components: {artifact_components}')

raw_clean = ica.apply(raw_clean)
print('  ICA applied to data')
print(f'  Rank after ICA: {mne.compute_rank(raw_clean)}')

print('  === Onset Epochs ===')
temp_epochs_onset = create_temp_epochs_onset(raw_clean, onset_event_id=onset_event_id)
snr_onset_global, _, _ = calculate_snr_onset(temp_epochs_onset)
snr_tracker.record_snr('onset', 'after_ica', snr_onset_global)

print('\n  === Response Epochs ===')
temp_epochs_response = create_temp_epochs_response(raw_clean)
snr_response_global, _, _ = calculate_snr_response(temp_epochs_response)
snr_tracker.record_snr('response', 'after_ica', snr_response_global)

raw_clean.plot()
input('  Review final cleaned data, press Enter to continue...')

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

print('\nSaving preprocessed data...')
raw_clean.save(f'{preproc_dir}/{subj}_{dtype}_ICA_no_ocular-raw.fif', overwrite=overwrite)
with open(f'{preproc_dir}/{subj}_{dtype}-info.p', 'wb') as f:
    pickle.dump(raw_clean.info, f)

snr_tracker.save_subject()

print(f'\n{"="*60}')
print(f'COMPLETE: {subj} | {dtype}')
print(f'{"="*60}')
