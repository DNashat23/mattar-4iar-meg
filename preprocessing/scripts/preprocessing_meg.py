"""
MEG Preprocessing Pipeline
Processes raw MEG data through filtering, artifact removal, and ICA.
Epoching is handled downstream by single_subject_ar_v3.py.
"""

import mne
import pickle
import glob
import os
import numpy as np
from snr_both import (SNRTracker, calculate_snr_onset, calculate_snr_response,
                         create_temp_epochs_onset, create_temp_epochs_response)

# Suppress MNE info messages
mne.set_log_level(verbose='WARNING')
print("Starting MEG preprocessing pipeline...")

# ============================================================================
# CONFIGURATION
# ============================================================================

subj   = 'R2781'          #  id of participant whose data you want to preprocess
dtypes = ['main']    #  choose between localizer / main
base_dir = f'/Users/dariushnashat/Documents/Mattar Lab Internship/'
data_dir = os.path.join(base_dir, subj)
empty_room_file = f'{data_dir}/raw/{subj}_EmptyRoom_nr.sqd'
overwrite = True

# If remove_radio_noise.py has been run for this subject/dtype, set this path
# to the cleaned raw it produced. Leave as None to load from raw/ as usual.
nr_raw_path = f'{data_dir}/preprocessing/{subj}_localizer_nr-raw.fif'

# Event ID for stimulus onset: 161 for localizer, 163 for main
onset_event_ids = {'localizer': 161, 'main': 163}

# Initialize SNR tracker
snr_tracker = SNRTracker(base_dir)

# ============================================================================
# PREPROCESSING
# ============================================================================

for dtype in dtypes:
    preproc_dir = os.path.join(data_dir, 'preprocessing')
    os.makedirs(preproc_dir, exist_ok=True)
    onset_event_id = onset_event_ids.get(dtype, 161)

    # Start tracking for this subject
    snr_tracker.start_subject(subj)
    
    # Check if preprocessing already done
    if not overwrite and os.path.exists(f'{preproc_dir}/{subj}_{dtype}_ica_exclude.p'):
        print(f'{subj} | {dtype} : ICA ALREADY COMPLETED')
    else:
        print(f'\n{"="*60}')
        print(f'PREPROCESSING: {subj} | {dtype}')
        print(f'{"="*60}\n')
        
        # ------------------------------------------------------------------------
        # STEP 1: Load raw data
        # ------------------------------------------------------------------------
        print('STEP 1: Loading raw data files...')
        raws = []
        for file in sorted(glob.glob(f'{data_dir}/raw/*{dtype}*.fif')):
            print(f'  Loading {os.path.basename(file)}')
            raws.append(mne.io.read_raw_fif(file, preload=True))
        if not raws:
            raise Exception(f'{subj} | {dtype} : NO RAW FILES FOUND')
        raw = mne.concatenate_raws(raws)
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")
        
        # Calculate baseline SNR for both onset and response
        print("\n  Calculating baseline SNR...")
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'baseline', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'baseline', snr_response_global)
        
        # ------------------------------------------------------------------------
        # STEP 2: Frequency filtering
        # ------------------------------------------------------------------------
        print('\nSTEP 2: Applying frequency filter (1-40 Hz)...')
        raw.filter(1, 40, method='iir')
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'after_filtering', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'after_filtering', snr_response_global)
        
        # ------------------------------------------------------------------------
        # STEP 3: Bad channel marking and interpolation
        # ------------------------------------------------------------------------
        print('\nSTEP 3: Bad channel detection and interpolation...')
        print('  Opening interactive plot for bad channel marking...')
        original_duration = raw.times[-1]
        
        raw.plot()
        input('  Mark bad channels and bad segments, then press Enter to continue...')
        
        bad_channels = raw.info['bads'].copy()
        print(f'  Bad channels marked: {bad_channels}')

        # Save bad channels to file so the minimal preprocessing pipeline
        # can load them without requiring a new interactive session
        bad_channels_path = f'{preproc_dir}/{subj}_{dtype}_bad_channels.p'
        with open(bad_channels_path, 'wb') as f:
            pickle.dump(bad_channels, f)
        print(f'  Bad channels saved to: {bad_channels_path}')

        if bad_channels:
            raw.interpolate_bads()
            print(f'  Bad channels interpolated')
            # Preserve bad channel info for record-keeping
            raw.info['bads'] = bad_channels
        else:
            print('  No bad channels marked')
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'after_interpolation', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'after_interpolation', snr_response_global)
        
        # ------------------------------------------------------------------------
        # STEP 4: Empty room projection (environmental noise removal)
        # ------------------------------------------------------------------------
        print('\nSTEP 5: Empty room projection...')
        print(f'  Rank before empty room projection: {mne.compute_rank(raw)}')
        try:
            empty_room_raw = mne.io.read_raw_kit(
                empty_room_file, mrk=None, elp=None, hsp=None, 
                stim='<', preload=True, verbose=True
            )
            print(f"  Empty room duration: {empty_room_raw.times[-1]:.1f} seconds")
            
            # Apply same filtering
            empty_room_raw.filter(1, 40, method='iir')

            # Empty room recordings have no digitization of their own (no head
            # present, loaded with mrk=elp=hsp=None) so MEG bad-channel
            # interpolation can't fit a sphere model without borrowing the
            # subject's head position/shape from this same session. 'dig' is
            # a protected Info field, so it has to go through _unlock().
            with empty_room_raw.info._unlock():
                empty_room_raw.info['dig'] = raw.info['dig']
                empty_room_raw.info['dev_head_t'] = raw.info['dev_head_t']

            # Mark and interpolate the same bad channels in empty room data
            empty_room_raw.info['bads'] = bad_channels.copy()
            if bad_channels:
                empty_room_raw.interpolate_bads()
                print(f'  Bad channels interpolated in empty room: {bad_channels}')
            else:
                print('  No bad channels to interpolate in empty room')

            # Compute noise projectors
            print("  Computing environmental noise projectors...")
            empty_room_projs = mne.compute_proj_raw(
                empty_room_raw, 
                duration=2,
                n_grad=0,
                n_mag=0.9,
                reject={'mag': 3e-12},
                flat={'mag': 1e-15}
            )
            
            mne.viz.plot_projs_topomap(empty_room_projs, colorbar=True, 
                                       vlim="joint", info=empty_room_raw.info)
            
            print(f"  Created {len(empty_room_projs)} noise projectors")
            for i, proj in enumerate(empty_room_projs):
                print(f"    {i+1}. {proj['desc']}")
            
            # Apply to main data
            raw.add_proj(empty_room_projs, remove_existing=False)
            raw.apply_proj()
            print(f"  Applied projectors to main data")
            print(f'  Rank after empty room projection: {mne.compute_rank(raw)}')

        except Exception as e:
            print(f"  Error with empty room data: {e}")
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'after_empty_room', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'after_empty_room', snr_response_global)
        
        # ------------------------------------------------------------------------
        # STEP 6: ICA (Independent Component Analysis)
        # ------------------------------------------------------------------------
        print('\nSTEP 6: ICA artifact removal...')
        print(f'  Rank before ICA: {mne.compute_rank(raw)}')

        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
        print('  Fitting ICA...')
        ica.fit(raw, reject={'mag': 5e-12})
        ica.save(f'{preproc_dir}/{subj}_{dtype}_icasolution-ica.fif', overwrite=overwrite)

        '''
        # Explained variance per component (mag only — KIT has no gradiometers)
        print('\n  ICA component explained variance:')
        var_explained = []
        for idx in range(ica.n_components_):
            ratio = ica.get_explained_variance_ratio(raw, components=idx).get('mag', 0.0)
            var_explained.append((idx, ratio))
        for comp_idx, var in sorted(var_explained, key=lambda x: x[1], reverse=True):
            print(f'    Component {comp_idx:3d}: {var * 100:.3f}%')
        '''

        # Interactive component selection
        print('\n  Review components and mark artifacts...')
        ica.plot_sources(raw)
        input('  Press Enter to see ICA topographies...')
        ica.plot_components()

        artifact_components = ica.exclude.copy()
        print(f'  Excluding components: {artifact_components}')

        # Apply ICA
        raw = ica.apply(raw)
        print('  ICA applied to data')
        print(f'  Rank after ICA: {mne.compute_rank(raw)}')
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw, onset_event_id=onset_event_id)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'after_ica', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'after_ica', snr_response_global)
        
        # Review cleaned data
        raw.plot()
        input('  Review final cleaned data, press Enter to continue...')
        
        # ------------------------------------------------------------------------
        # Save preprocessed data
        # ------------------------------------------------------------------------
        print('\nSaving preprocessed data...')
        # Localizer output is named ICA-raw_final (matching what
        # single_subject_ar_v3_independent.py / _fixation.py and
        # transfer_localizer_files.py already expect); main task keeps
        # ICA_no_ocular-raw (matching single_subject_ar_v3_main.py /
        # _fixation.py and transfer_main_ica.py).
        ica_raw_suffix = 'ICA-raw_final' if dtype == 'localizer' else 'ICA_no_ocular-raw'
        raw.save(f'{preproc_dir}/{subj}_{dtype}_{ica_raw_suffix}.fif', overwrite=overwrite)
        with open(f'{preproc_dir}/{subj}_{dtype}-info.p', 'wb') as f:
            pickle.dump(raw.info, f)
        
        # Save SNR results
        snr_tracker.save_subject()
        
        print(f'\n{"="*60}')
        print(f'PREPROCESSING COMPLETE: {subj} | {dtype}')
        print(f'{"="*60}\n')

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*60)
print('SNR TRACKING SUMMARY')
print('='*60)
snr_tracker.get_summary()

print('\n' + '='*60)
print('PIPELINE COMPLETE')
print('='*60)