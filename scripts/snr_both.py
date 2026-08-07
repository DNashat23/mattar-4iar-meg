"""
SNR Tracker Module
Handles SNR calculation and tracking across preprocessing stages for both onset and response epochs
"""

import pandas as pd
import numpy as np
import os
import mne


class SNRTracker:
    """Tracks SNR improvements across preprocessing stages for onset and response epochs"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.spreadsheet_path = os.path.join(base_dir, 'snr_analysis_master.csv')
        
        # Define columns for tracking both onset and response
        self.columns = [
            'subject',
            # Onset SNR values at each stage
            'onset_baseline_snr',
            'onset_after_filtering_snr',
            'onset_after_interpolation_snr', 
            'onset_after_radio_noise_snr',
            'onset_after_empty_room_snr',
            'onset_after_ica_snr',
            'onset_final_snr',
            # Response SNR values at each stage
            'response_baseline_snr',
            'response_after_filtering_snr',
            'response_after_interpolation_snr', 
            'response_after_radio_noise_snr',
            'response_after_empty_room_snr',
            'response_after_ica_snr',
            'response_final_snr',
            # Onset absolute changes (delta SNR)
            'onset_delta_filtering',
            'onset_delta_interpolation',
            'onset_delta_radio_noise',
            'onset_delta_empty_room',
            'onset_delta_ica',
            'onset_total_improvement',
            # Response absolute changes (delta SNR)
            'response_delta_filtering',
            'response_delta_interpolation',
            'response_delta_radio_noise',
            'response_delta_empty_room',
            'response_delta_ica',
            'response_total_improvement',
            # Onset relative contributions (%)
            'onset_rel_filtering_pct',
            'onset_rel_interpolation_pct', 
            'onset_rel_radio_noise_pct',
            'onset_rel_empty_room_pct',
            'onset_rel_ica_pct',
            # Response relative contributions (%)
            'response_rel_filtering_pct',
            'response_rel_interpolation_pct', 
            'response_rel_radio_noise_pct',
            'response_rel_empty_room_pct',
            'response_rel_ica_pct'
        ]
        
        self.load_or_create_spreadsheet()
    
    def load_or_create_spreadsheet(self):
        """Load existing spreadsheet or create a new one"""
        if os.path.exists(self.spreadsheet_path):
            try:
                self.df = pd.read_csv(self.spreadsheet_path)
                print(f"Loaded existing spreadsheet with {len(self.df)} subjects")
            except Exception as e:
                print(f"Error loading spreadsheet: {e}")
                print("Creating new spreadsheet...")
                self.df = pd.DataFrame(columns=self.columns)
        else:
            print("Creating new spreadsheet...")
            self.df = pd.DataFrame(columns=self.columns)
    
    def start_subject(self, subject_id):
        """Initialize tracking for a new subject"""
        self.subject_id = subject_id
        self.current_row = {col: np.nan for col in self.columns}
        self.current_row['subject'] = subject_id
        print(f"\nStarting SNR tracking for subject: {subject_id}")
    
    def record_snr(self, epoch_type, stage, snr_value):
        """
        Record SNR value for a specific stage and epoch type
        
        Parameters:
        -----------
        epoch_type : str
            Either 'onset' or 'response'
        stage : str
            Stage name (e.g., 'baseline', 'after_filtering', etc.)
        snr_value : float
            SNR value to record
        """
        if not hasattr(self, 'current_row'):
            raise ValueError("Must call start_subject() first")
        
        stage_column = f"{epoch_type}_{stage}_snr"
        if stage_column in self.columns:
            self.current_row[stage_column] = snr_value
            print(f"  Recorded {epoch_type} {stage} SNR: {snr_value:.3f}")
        else:
            print(f"  Warning: Unknown stage '{epoch_type}_{stage}'")
    
    def calculate_changes_and_contributions(self, epoch_type):
        """
        Calculate delta SNRs and relative contributions for a specific epoch type
        
        Parameters:
        -----------
        epoch_type : str
            Either 'onset' or 'response'
        """
        if not hasattr(self, 'current_row'):
            raise ValueError("Must call start_subject() first")
        
        # Get SNR values in order
        stages = ['baseline', 'after_filtering', 'after_interpolation', 
                 'after_radio_noise', 'after_empty_room', 'after_ica']
        
        snr_values = []
        valid_stages = []
        
        for stage in stages:
            snr_col = f"{epoch_type}_{stage}_snr"
            if not pd.isna(self.current_row[snr_col]):
                snr_values.append(self.current_row[snr_col])
                valid_stages.append(stage)
        
        if len(snr_values) < 2:
            print(f"Warning: Need at least 2 SNR values to calculate changes for {epoch_type}")
            return
        
        # Calculate absolute changes (deltas)
        delta_mapping = {
            'after_filtering': f'{epoch_type}_delta_filtering',
            'after_interpolation': f'{epoch_type}_delta_interpolation', 
            'after_radio_noise': f'{epoch_type}_delta_radio_noise',
            'after_empty_room': f'{epoch_type}_delta_empty_room',
            'after_ica': f'{epoch_type}_delta_ica'
        }
        
        prev_snr = self.current_row[f'{epoch_type}_baseline_snr']
        for i, stage in enumerate(valid_stages[1:], 1):
            curr_snr = self.current_row[f"{epoch_type}_{stage}_snr"]
            delta_col = delta_mapping.get(stage)
            if delta_col:
                delta = curr_snr - prev_snr
                self.current_row[delta_col] = delta
                print(f"  {epoch_type.capitalize()} delta {stage}: {delta:+.3f}")
            prev_snr = curr_snr
        
        # Set final SNR and total improvement
        self.current_row[f'{epoch_type}_final_snr'] = snr_values[-1]
        self.current_row[f'{epoch_type}_total_improvement'] = snr_values[-1] - snr_values[0]
        
        # Calculate relative contributions (as percentages of total gross movement)
        rel_mapping = {
            f'{epoch_type}_delta_filtering': f'{epoch_type}_rel_filtering_pct',
            f'{epoch_type}_delta_interpolation': f'{epoch_type}_rel_interpolation_pct',
            f'{epoch_type}_delta_radio_noise': f'{epoch_type}_rel_radio_noise_pct',
            f'{epoch_type}_delta_empty_room': f'{epoch_type}_rel_empty_room_pct',
            f'{epoch_type}_delta_ica': f'{epoch_type}_rel_ica_pct'
        }

        delta_values = [
            self.current_row[d] for d in rel_mapping
            if not pd.isna(self.current_row[d])
        ]
        sum_abs_deltas = sum(abs(v) for v in delta_values)

        if sum_abs_deltas > 0:
            for delta_col, rel_col in rel_mapping.items():
                if not pd.isna(self.current_row[delta_col]):
                    contribution = (self.current_row[delta_col] / sum_abs_deltas) * 100
                    self.current_row[rel_col] = round(contribution, 2)
                    print(f"  {epoch_type.capitalize()} relative contribution {delta_col}: {contribution:.2f}%")
    
    def save_subject(self):
        """Save the current subject's data to the spreadsheet"""
        if not hasattr(self, 'current_row'):
            raise ValueError("Must call start_subject() first")
    
        # Calculate all changes and contributions for both epoch types
        print("\nCalculating onset SNR changes and contributions...")
        self.calculate_changes_and_contributions('onset')
        print("\nCalculating response SNR changes and contributions...")
        self.calculate_changes_and_contributions('response')
    
        # Round ALL numeric values to 2 decimal places before saving
        numeric_columns = [col for col in self.columns if col != 'subject']
        for col in numeric_columns:
            if not pd.isna(self.current_row[col]) and isinstance(self.current_row[col], (int, float)):
                self.current_row[col] = round(self.current_row[col], 2)
    
        # Check if subject already exists
        existing_idx = self.df[self.df['subject'] == self.subject_id].index
    
        if len(existing_idx) > 0:
            # Update existing row
            for col, value in self.current_row.items():
                if not pd.isna(value):
                    self.df.loc[existing_idx[0], col] = value
            print(f"\nUpdated existing data for subject {self.subject_id}")
        else:
            # Add new row
            new_row_df = pd.DataFrame([self.current_row])
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
            print(f"\nAdded new data for subject {self.subject_id}")
    
        # Save to CSV
        self.df.to_csv(self.spreadsheet_path, index=False)
        print(f"Spreadsheet saved to: {self.spreadsheet_path}")
    
        # Clean up
        delattr(self, 'current_row')
        delattr(self, 'subject_id')
    
    def get_summary(self):
        """Print a summary of the current spreadsheet"""
        if len(self.df) == 0:
            print("No data in spreadsheet yet")
            return
        
        print(f"\n=== SNR Spreadsheet Summary ===")
        print(f"Total subjects: {len(self.df)}")
        print(f"Subjects: {', '.join(self.df['subject'].tolist())}")
        
        # Show means for key metrics - both onset and response
        print("\n--- Onset Epochs ---")
        onset_cols = ['onset_baseline_snr', 'onset_final_snr', 'onset_total_improvement']
        for col in onset_cols:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                if not pd.isna(mean_val):
                    print(f"Mean {col}: {mean_val:.3f}")
        
        print("\n--- Response Epochs ---")
        response_cols = ['response_baseline_snr', 'response_final_snr', 'response_total_improvement']
        for col in response_cols:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                if not pd.isna(mean_val):
                    print(f"Mean {col}: {mean_val:.3f}")


def calculate_snr_onset(epochs, time_window=None, top_percentile=0.2):
    """
    Calculate SNR for onset epochs (100-400ms post-stimulus)
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to calculate SNR from
    time_window : tuple, optional
        (start, end) time window in seconds. Default is (0.1, 0.4)
    top_percentile : float
        Percentile of top SNR values to use for global calculation
    
    Returns:
    --------
    global_snr : float
        Mean of top percentile SNR values
    raw_snr : float
        Mean of all SNR values (100%)
    snrs : ndarray
        SNR values for each channel-time combination
    """
    if time_window is None:
        time_window = (0.1, 0.4)  # 100-400ms post-stimulus
        
    time_mask = (epochs.times >= time_window[0]) & (epochs.times <= time_window[1])
    data = epochs.get_data()[:, :, time_mask]

    # Get dimensions
    n_epochs, n_channels, n_timepoints = data.shape
    
    print(f"  [ONSET] Time window: {time_window[0]*1000:.0f}ms to {time_window[1]*1000:.0f}ms")
    print(f"  [ONSET] Data shape: {n_epochs} epochs × {n_channels} channels × {n_timepoints} timepoints")

    # Calculate SNR for each channel-time combination
    snrs = np.zeros((data.shape[1], data.shape[2]))
    for ch in range(data.shape[1]):
        for t in range(data.shape[2]):
            signal_mean = np.mean(data[:, ch, t])
            signal_std = np.std(data[:, ch, t], ddof=0)
            standard_error = signal_std / np.sqrt(data.shape[0])
            
            if signal_std > 0:
                snrs[ch, t] = abs(signal_mean) / standard_error
            else:
                snrs[ch, t] = 0

    # Calculate raw SNR (average of all SNR values)
    flat_snrs = snrs.flatten()
    raw_snr = np.mean(flat_snrs)
    
    # Calculate global SNR (average of top percentile)
    threshold_idx = int((1 - top_percentile) * len(flat_snrs))
    top_snrs = np.partition(flat_snrs, threshold_idx)[threshold_idx:]
    global_snr = np.mean(top_snrs)
    
    print(f"  [ONSET] Raw SNR (100%): {raw_snr:.2f}")
    print(f"  [ONSET] Global SNR (top {top_percentile*100}%): {global_snr:.2f}")

    return global_snr, raw_snr, snrs


def calculate_snr_response(epochs, time_window=None, top_percentile=0.2):
    """
    Calculate SNR for response epochs (-400ms to -100ms pre-response)
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to calculate SNR from
    time_window : tuple, optional
        (start, end) time window in seconds. Default is (-0.4, -0.1)
    top_percentile : float
        Percentile of top SNR values to use for global calculation
    
    Returns:
    --------
    global_snr : float
        Mean of top percentile SNR values
    raw_snr : float
        Mean of all SNR values (100%)
    snrs : ndarray
        SNR values for each channel-time combination
    """
    if time_window is None:
        time_window = (-0.4, -0.1)  # 400-100ms pre-response
        
    time_mask = (epochs.times >= time_window[0]) & (epochs.times <= time_window[1])
    data = epochs.get_data()[:, :, time_mask]

    # Get dimensions
    n_epochs, n_channels, n_timepoints = data.shape
    
    print(f"  [RESPONSE] Time window: {time_window[0]*1000:.0f}ms to {time_window[1]*1000:.0f}ms")
    print(f"  [RESPONSE] Data shape: {n_epochs} epochs × {n_channels} channels × {n_timepoints} timepoints")

    # Calculate SNR for each channel-time combination
    snrs = np.zeros((data.shape[1], data.shape[2]))
    for ch in range(data.shape[1]):
        for t in range(data.shape[2]):
            signal_mean = np.mean(data[:, ch, t])
            signal_std = np.std(data[:, ch, t], ddof=0)
            standard_error = signal_std / np.sqrt(data.shape[0])
            
            if signal_std > 0:
                snrs[ch, t] = abs(signal_mean) / standard_error
            else:
                snrs[ch, t] = 0

    # Calculate raw SNR (average of all SNR values)
    flat_snrs = snrs.flatten()
    raw_snr = np.mean(flat_snrs)
    
    # Calculate global SNR (average of top percentile)
    threshold_idx = int((1 - top_percentile) * len(flat_snrs))
    top_snrs = np.partition(flat_snrs, threshold_idx)[threshold_idx:]
    global_snr = np.mean(top_snrs)
    
    print(f"  [RESPONSE] Raw SNR (100%): {raw_snr:.2f}")
    print(f"  [RESPONSE] Global SNR (top {top_percentile*100}%): {global_snr:.2f}")

    return global_snr, raw_snr, snrs


def create_temp_epochs_onset(raw, onset_event_id=161):
    """
    Create temporary onset epochs for SNR calculation

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw data object
    onset_event_id : int
        Event ID to epoch around. 161 for localizer, 163 for main.

    Returns:
    --------
    temp_epochs : mne.Epochs
        Temporary epochs for SNR calculation
    """
    events = mne.find_events(raw, min_duration=0.0025)
    events[:, 0] += 60
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)

    events_filtered = events[events[:, 2] == onset_event_id]
    temp_epochs = mne.Epochs(
        raw, events_filtered,
        event_id={str(onset_event_id): onset_event_id},
        tmin=-0.1, tmax=1.0,
        baseline=(-0.1, 0),
        picks=picks_meg,
        decim=1,
        preload=True,
        reject=None,
        verbose=False
    )

    return temp_epochs


def create_temp_epochs_response(raw):
    """
    Create temporary response epochs for SNR calculation
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw data object
    
    Returns:
    --------
    temp_epochs : mne.Epochs
        Temporary epochs for SNR calculation
    """
    events = mne.find_events(raw, min_duration=0.0025)
    events[:, 0] += 60
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)

    events_filtered = events[events[:, 2] == 164]
    temp_epochs = mne.Epochs(
        raw, events_filtered, 
        event_id={'164': 164},
        tmin=-1.0, tmax=0.1,
        baseline=(-1.0, -0.9), 
        picks=picks_meg,
        decim=1, 
        preload=True,
        reject=None,
        verbose=False
    )
    
    return temp_epochs