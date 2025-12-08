"""
MEG Preprocessing Pipeline
Processes raw MEG data through filtering, artifact removal, and epoching
"""

import pandas as pd
import scipy.io as sio
import mne
import pickle
import glob
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from snr_both import (SNRTracker, calculate_snr_onset, calculate_snr_response, 
                         create_temp_epochs_onset, create_temp_epochs_response)

# Suppress MNE info messages
mne.set_log_level(verbose='WARNING')
print("Starting MEG preprocessing pipeline...")

# ============================================================================
# CONFIGURATION
# ============================================================================

subj = 'R2649'
dtypes = ['localizer']  # localizer / main
epoching_types = ['localizer']
base_dir = f'/Users/dariushnashat/Documents/Mattar Lab Internship/' 
data_dir = os.path.join(base_dir, subj)
empty_room_file = f'{data_dir}/raw/{subj}_EmptyRoom_nr.sqd'
overwrite = True
epochsonly = False

# Epoch configurations for different task types
epoch_configs = {
    'localizer': {
        "event_ids": {'161': 161, '164': 164},
        "tmin_161": -0.1, "tmax_161": 1.0, "baseline_161": (-0.1, 0),
        "tmin_164": -1.0, "tmax_164": 0.1, "baseline_164": (-1.0, -0.9),
        "decim": 1,
        "expected_nb_events": 1008,
        "photodiode_realignment": True,
        "reject_criteria": dict(mag=4e-12, grad=4e-12),
        "flat_criteria": dict(mag=1e-15, grad=1e-13)
    },
    'main': {
        "event_ids": {'sequences4': 2},
        "tmin": -0.1, "tmax": 0.9, "baseline": (-0.1, 0),
        "decim": 1,
        "expected_nb_events": 960,
        "photodiode_realignment": True,
        "reject_criteria": dict(mag=4e-12, grad=4e-12),
        "flat_criteria": dict(mag=1e-15, grad=1e-13)
    }
}

# Initialize SNR tracker
snr_tracker = SNRTracker(base_dir)

# ============================================================================
# PREPROCESSING
# ============================================================================

for dtype in dtypes:
    preproc_dir = os.path.join(data_dir, 'preprocessing')
    os.makedirs(preproc_dir, exist_ok=True)
    
    # Start tracking for this subject
    snr_tracker.start_subject(subj)
    
    # Check if preprocessing already done
    if epochsonly:
        print(f'{subj} | {dtype} : skipping preprocessing - will use existing ICA results')
    elif not overwrite and os.path.exists(f'{preproc_dir}/{subj}_{dtype}_ica_exclude.p'):
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
        temp_epochs_onset = create_temp_epochs_onset(raw)
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
        temp_epochs_onset = create_temp_epochs_onset(raw)
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
        
        if bad_channels:
            raw.interpolate_bads()
            print(f'  Bad channels interpolated')
            # Preserve bad channel info for record-keeping
            raw.info['bads'] = bad_channels
        else:
            print('  No bad channels marked')
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'after_interpolation', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'after_interpolation', snr_response_global)
        
        # ------------------------------------------------------------------------
        # STEP 4: Radio noise removal
        # ------------------------------------------------------------------------
        print('\nSTEP 4: Radio noise removal...')
        bad_annotations = [ann for ann in raw.annotations 
                          if 'BAD' in ann['description'] or 'bad' in ann['description']]
        
        if bad_annotations:
            print(f'  Found {len(bad_annotations)} radio noise annotations')
            bad_segments = []
            total_bad_time = 0
            
            for ann in bad_annotations:
                if ann['duration'] > 0:
                    start_time = ann['onset']
                    end_time = ann['onset'] + ann['duration']
                    bad_segments.append((start_time, end_time))
                    total_bad_time += ann['duration']
                    print(f"    Bad segment: {start_time:.2f}s to {end_time:.2f}s ({ann['duration']:.2f}s)")
            
            if bad_segments:
                # Extract clean segments
                good_segments = []
                current_time = 0.0
                bad_segments.sort(key=lambda x: x[0])
                
                for start_bad, end_bad in bad_segments:
                    if start_bad > current_time:
                        good_segments.append((current_time, start_bad))
                    current_time = max(current_time, end_bad)
                
                if current_time < original_duration:
                    good_segments.append((current_time, original_duration))
                
                # Crop and concatenate clean segments
                if good_segments:
                    good_raws = []
                    for start_time, end_time in good_segments:
                        raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)
                        good_raws.append(raw_segment)
                    
                    if len(good_raws) > 1:
                        raw = mne.concatenate_raws(good_raws)
                        print(f'  Concatenated {len(good_raws)} clean segments')
                    else:
                        raw = good_raws[0]
                    
                    cleaned_duration = raw.times[-1]
                    removed_time = original_duration - cleaned_duration
                    print(f'  Data duration after cleaning: {cleaned_duration:.1f}s')
                    print(f'  Removed {removed_time:.1f}s of contaminated data')
                else:
                    raise Exception("No clean data segments found!")
        else:
            print('  No radio noise annotations found')
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw)
        snr_onset_global, snr_onset_raw, _ = calculate_snr_onset(temp_epochs_onset)
        snr_tracker.record_snr('onset', 'after_radio_noise', snr_onset_global)
        
        print("\n  === Response Epochs ===")
        temp_epochs_response = create_temp_epochs_response(raw)
        snr_response_global, snr_response_raw, _ = calculate_snr_response(temp_epochs_response)
        snr_tracker.record_snr('response', 'after_radio_noise', snr_response_global)
        
        # ------------------------------------------------------------------------
        # STEP 5: Empty room projection (environmental noise removal)
        # ------------------------------------------------------------------------
        print('\nSTEP 5: Empty room projection...')
        try:
            empty_room_raw = mne.io.read_raw_kit(
                empty_room_file, mrk=None, elp=None, hsp=None, 
                stim='<', preload=True, verbose=True
            )
            print(f"  Empty room duration: {empty_room_raw.times[-1]:.1f} seconds")
            
            # Apply same filtering
            empty_room_raw.filter(1, 40, method='iir')
            
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
            
        except Exception as e:
            print(f"  Error with empty room data: {e}")
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw)
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
        
        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
        print('  Fitting ICA...')
        ica.fit(raw, reject={'mag': 5e-12})
        ica.save(f'{preproc_dir}/{subj}_{dtype}_icasolution-ica.fif', overwrite=overwrite)
        
        # Interactive component selection
        print('  Review components and mark artifacts...')
        ica.plot_sources(raw)
        input('  Press Enter to see ICA topographies...')
        ica.plot_components()
        
        artifact_components = ica.exclude.copy()
        print(f'  Excluding components: {artifact_components}')
        
        # Apply ICA
        raw = ica.apply(raw)
        print('  ICA applied to data')
        
        print("  === Onset Epochs ===")
        temp_epochs_onset = create_temp_epochs_onset(raw)
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
        raw.save(f'{preproc_dir}/{subj}_{dtype}_ICA-raw.fif', overwrite=overwrite)
        with open(f'{preproc_dir}/{subj}_{dtype}-info.p', 'wb') as f:
            pickle.dump(raw.info, f)
        
        # Save SNR results
        snr_tracker.save_subject()
        
        print(f'\n{"="*60}')
        print(f'PREPROCESSING COMPLETE: {subj} | {dtype}')
        print(f'{"="*60}\n')

# ============================================================================
# EPOCHING
# ============================================================================

# Pre-fit label encoder
le = LabelEncoder()
le.fit(['none', '2', '2gap', '3', '4'])

for dtype in epoching_types:
    print(f'\n{"="*60}')
    print(f'EPOCHING: {subj} | {dtype}')
    print(f'{"="*60}\n')
    
    configs = epoch_configs[dtype]
    raw_path = f'{data_dir}/preprocessing/{subj}_{dtype}_ICA-raw.fif'
    
    # Check if epochs already exist
    if not overwrite and os.path.exists(f'{data_dir}/preprocessing/{subj}_{dtype}_onset-epo.fif'):
        print(f'{subj} | {dtype} : EPOCHS ALREADY CREATED')
        continue
    
    # Load preprocessed data
    print('Loading preprocessed data...')
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    events = mne.find_events(raw, min_duration=0.0025)
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
    
    # Detect channel types
    has_mag = len(mne.pick_types(raw.info, meg='mag')) > 0
    has_grad = len(mne.pick_types(raw.info, meg='grad')) > 0
    print(f"Channel types: MAG={has_mag}, GRAD={has_grad}")
    
    # Create rejection criteria based on available channels
    reject_criteria = {}
    flat_criteria = {}
    if has_mag:
        reject_criteria['mag'] = configs["reject_criteria"]['mag']
        flat_criteria['mag'] = configs["flat_criteria"]['mag']
    if has_grad:
        reject_criteria['grad'] = configs["reject_criteria"]['grad']
        flat_criteria['grad'] = configs["flat_criteria"]['grad']
    
    print(f"Rejection criteria: {reject_criteria}")
    print(f"Flat criteria: {flat_criteria}")
    
    # Photodiode realignment
    if configs["photodiode_realignment"]:
        events[:, 0] += 60
    
    # Load logfile
    logfile = pd.read_csv(f'{data_dir}/{subj}_logfile.csv')
    
    # ------------------------------------------------------------------------
    # Localizer-specific processing
    # ------------------------------------------------------------------------
    if dtype == 'localizer':
        print('\nProcessing localizer trials...')
        
        # Find onset and offset events
        events_161 = events[events[:, 2] == 161]
        events_164 = events[events[:, 2] == 164]
        
        print(f"Onset events (161): {len(events_161)}")
        print(f"Offset events (164): {len(events_164)}")
        
        if len(events_161) != len(events_164):
            print(f"WARNING: Unequal onset and offset events")
        
        # Filter for trials with responses
        filtered_events_161_temp = []
        for i in range(len(events_161)):
            if pd.notna(logfile.iloc[i]['acc']):
                filtered_events_161_temp.append(events_161[i])
        filtered_events_161_temp = np.array(filtered_events_161_temp)
        print(f"Onset events with responses: {len(filtered_events_161_temp)}")
        
        # Keep only trials with responses in logfile
        logfile = logfile[(logfile['task'] == 'localizer') & (logfile['acc'].notna())]
        logfile = logfile.reset_index(drop=True)
        
        # Filter for correct responses only
        filtered_events_161 = []
        filtered_events_164 = []
        for i in range(len(logfile)):  # FIXED: Use len(logfile) after filtering
            if logfile.iloc[i]['acc'] == True:
                # FIXED: Use filtered_events_161_temp which matches filtered logfile
                filtered_events_161.append(filtered_events_161_temp[i])
                filtered_events_164.append(events_164[i])
        
        filtered_events_161 = np.array(filtered_events_161)
        filtered_events_164 = np.array(filtered_events_164)
        
        print(f"Correct trials kept: {len(filtered_events_161)}")
        
        # Keep only correct trials in logfile
        logfile = logfile[(logfile['task'] == 'localizer') & (logfile['acc'] == True)]
        logfile = logfile.reset_index(drop=True)
        
        # Save filtered logfile
        output_csv_path = f'{data_dir}/preprocessing/{subj}_{dtype}_filtered_logfile.csv'
        logfile.to_csv(output_csv_path, index=False)
        print(f"Saved filtered logfile to: {output_csv_path}")
        
        # Create epochs
        print('\nCreating epochs...')
        epochs_onset = mne.Epochs(
            raw, filtered_events_161, 
            event_id={'161': 161},
            tmin=configs["tmin_161"], 
            tmax=configs["tmax_161"],
            baseline=configs["baseline_161"], 
            picks=picks_meg,
            decim=configs["decim"], 
            preload=True,
            reject=reject_criteria,
            flat=flat_criteria
        )
        
        epochs_response = mne.Epochs(
            raw, filtered_events_164, 
            event_id={'164': 164},
            tmin=configs["tmin_164"], 
            tmax=configs["tmax_164"],
            baseline=configs["baseline_164"], 
            picks=picks_meg,
            decim=configs["decim"], 
            preload=True,
            reject=reject_criteria,
            flat=flat_criteria
        )
        
        # Check for rejected epochs
        print('\nEpoch rejection analysis...')
        bad_onset_indices = set()
        bad_response_indices = set()
        
        if hasattr(epochs_onset, 'drop_log'):
            for i, log in enumerate(epochs_onset.drop_log):
                if log:
                    bad_onset_indices.add(i)
                    print(f"  Onset epoch {i} dropped: {log}")
        
        if hasattr(epochs_response, 'drop_log'):
            for i, log in enumerate(epochs_response.drop_log):
                if log:
                    bad_response_indices.add(i)
                    print(f"  Response epoch {i} dropped: {log}")
        
        all_bad_indices = bad_onset_indices.union(bad_response_indices)
        
        print(f"\nRejection summary:")
        print(f"  Bad onset epochs: {len(bad_onset_indices)}")
        print(f"  Bad response epochs: {len(bad_response_indices)}")
        print(f"  Total bad trials: {len(all_bad_indices)}")
        
        # Recreate epochs without bad trials
        if all_bad_indices:
            all_trial_indices = set(range(len(logfile)))
            good_indices = sorted(list(all_trial_indices - all_bad_indices))
            
            print(f"  Keeping {len(good_indices)} good trials")
            
            good_events_161 = filtered_events_161[good_indices]
            good_events_164 = filtered_events_164[good_indices]
            good_logfile = logfile.iloc[good_indices].reset_index(drop=True)
            
            epochs_onset = mne.Epochs(
                raw, good_events_161, 
                event_id={'161': 161},
                tmin=configs["tmin_161"], 
                tmax=configs["tmax_161"],
                baseline=configs["baseline_161"], 
                picks=picks_meg,
                decim=configs["decim"], 
                preload=True,
                reject=reject_criteria,
                flat=flat_criteria
            )
            
            epochs_response = mne.Epochs(
                raw, good_events_164, 
                event_id={'164': 164},
                tmin=configs["tmin_164"], 
                tmax=configs["tmax_164"],
                baseline=configs["baseline_164"], 
                picks=picks_meg,
                decim=configs["decim"], 
                preload=True,
                reject=reject_criteria,
                flat=flat_criteria
            )
            
            logfile = good_logfile
            
            # Save rejection report
            rejection_report = {
                'total_trials': len(all_trial_indices),
                'bad_onset_indices': sorted(list(bad_onset_indices)),
                'bad_response_indices': sorted(list(bad_response_indices)),
                'all_bad_indices': sorted(list(all_bad_indices)),
                'good_indices': good_indices,
                'final_trial_count': len(good_indices)
            }
            
            with open(f'{data_dir}/preprocessing/{subj}_{dtype}_rejection_report.p', 'wb') as f:
                pickle.dump(rejection_report, f)
            print(f"  Saved rejection report")
        else:
            print("  No epochs rejected - all trials kept")
        
        # Final verification
        if len(epochs_onset) != len(epochs_response) or len(epochs_onset) != len(logfile):
            raise Exception(
                f"Mismatch: onset={len(epochs_onset)}, "
                f"response={len(epochs_response)}, logfile={len(logfile)}"
            )
        
        print(f"\nFinal verification: {len(epochs_onset)} trials")
    
    # ------------------------------------------------------------------------
    # Add metadata and save
    # ------------------------------------------------------------------------
    logfile['coded_ground_truth'] = le.transform(logfile['ground_truth'])
    epochs_onset.metadata = logfile
    epochs_response.metadata = logfile
    
    # Save epochs
    print('\nSaving epochs...')
    
    # Save as .mat
    sio.savemat(
        f'{data_dir}/preprocessing/{subj}_{dtype}.mat',
        mdict={
            'x': epochs_onset.get_data(), 
            'y': epochs_onset.metadata.coded_ground_truth.tolist()
        }
    )
    
    # Save as .fif
    epochs_onset.save(
        f'{data_dir}/preprocessing/{subj}_{dtype}_onset-epo.fif', 
        overwrite=overwrite
    )
    print(f'  Onset epochs saved')
    
    epochs_response.save(
        f'{data_dir}/preprocessing/{subj}_{dtype}_offset-epo.fif', 
        overwrite=overwrite
    )
    print(f'  Response epochs saved')
    
    print(f'\n{"="*60}')
    print(f'EPOCHING COMPLETE: {subj} | {dtype}')
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