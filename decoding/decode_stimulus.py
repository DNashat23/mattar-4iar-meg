import pandas as pd
import numpy as np
import mne
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer, cross_val_multiscore, SlidingEstimator, Scaler
import matplotlib.pyplot as plt

mne.set_log_level(verbose='WARNING')

# =============================================================================
# CONFIGURATION
# =============================================================================

subj = 'R2649'
trial_type = 'localizer'
base_dir = '/Users/dariushnashat/Documents/Mattar Lab Internship'
data_dir = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')

# Rejection thresholds (same as preprocessing)
reject_criteria = dict(mag=4e-12, grad=4e-12)
flat_criteria   = dict(mag=1e-15, grad=1e-13)

# Time windows
TMIN_ITI      = -0.5   # ITI epoch: 500ms of blank screen before stimulus
TMAX_ITI      =  0.0
TMIN_STIMULUS = -0.1   # Stimulus epoch: 100ms before to 400ms after onset
TMAX_STIMULUS =  0.4

# Labels
LABEL_ITI      = 0     # blank screen
LABEL_STIMULUS = 1     # stimulus present

# =============================================================================
# LOAD ICA-CLEANED RAW FILE
# =============================================================================

# --- Switch between fully preprocessed and minimally preprocessed raw ---
# To revert to the full pipeline, comment out the minimal line and
# uncomment the ICA line below.
# raw_path = os.path.join(preproc_dir, f'{subj}_{trial_type}_ICA-raw.fif')
raw_path = os.path.join(preproc_dir, f'{subj}_{trial_type}_minimally_preprocessed-raw.fif')
print(f"Loading minimally preprocessed raw file: {raw_path}")
raw = mne.io.read_raw_fif(raw_path, preload=True)
print(f"  Sampling rate: {raw.info['sfreq']} Hz")

# Pick MEG channels only
picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)

# Build rejection criteria based on channel types actually present
has_mag  = len(mne.pick_types(raw.info, meg='mag'))  > 0
has_grad = len(mne.pick_types(raw.info, meg='grad')) > 0
active_reject = {}
active_flat   = {}
if has_mag:
    active_reject['mag'] = reject_criteria['mag']
    active_flat['mag']   = flat_criteria['mag']
if has_grad:
    active_reject['grad'] = reject_criteria['grad']
    active_flat['grad']   = flat_criteria['grad']
print(f"  Channel types present — MAG: {has_mag}, GRAD: {has_grad}")
print(f"  Rejection criteria: {active_reject}")
print(f"  Flat criteria:      {active_flat}")

# =============================================================================
# FIND EVENTS AND APPLY PHOTODIODE REALIGNMENT
# =============================================================================

print("\nFinding events...")
events = mne.find_events(raw, min_duration=0.0025)
events[:, 0] += 60  # photodiode realignment (same as preprocessing)

events_161 = events[events[:, 2] == 161]
print(f"  Onset events (161) found: {len(events_161)}")

# =============================================================================
# LOAD FILTERED LOGFILE TO IDENTIFY GOOD TRIALS
# =============================================================================

# --- Switch between full and minimal pipeline filtered logfiles ---
# logfile_path = os.path.join(preproc_dir, f'{subj}_{trial_type}_filtered_logfile.csv')
logfile_path = os.path.join(preproc_dir, f'{subj}_{trial_type}_minimally_preprocessed_filtered_logfile.csv')
print(f"\nLoading filtered logfile: {logfile_path}")
logfile = pd.read_csv(logfile_path)
n_good_trials = len(logfile)
print(f"  Good trials in filtered logfile: {n_good_trials}")

# The filtered logfile rows correspond 1-to-1 with events_161 after
# trial-level filtering (responses + accuracy). We use its length to
# index the matching subset of events.
# NOTE: this mirrors what the preprocessing script does at lines 414-416.
if len(events_161) < n_good_trials:
    raise ValueError(
        f"Fewer events ({len(events_161)}) than logfile rows ({n_good_trials}). "
        "Check that the correct raw and logfile are paired."
    )

good_events_161 = events_161[:n_good_trials]
print(f"  Using first {n_good_trials} onset events to match logfile")

# =============================================================================
# EXTRACT EPOCHS — NO BASELINE CORRECTION
# =============================================================================

print("\n=== EXTRACTING EPOCHS ===")

# --- ITI epochs: blank screen (-0.5 to 0s relative to onset) ---
print(f"\nExtracting ITI epochs ({TMIN_ITI}s to {TMAX_ITI}s)...")
epochs_iti = mne.Epochs(
    raw, good_events_161,
    event_id={'161': 161},
    tmin=TMIN_ITI, tmax=TMAX_ITI,
    baseline=None,          # no baseline correction
    picks=picks_meg,
    preload=True,
    reject=active_reject,
    flat=active_flat
)
print(f"  ITI epochs after rejection: {len(epochs_iti)} / {n_good_trials}")

# --- Stimulus epochs: board on screen (-0.1 to 0.4s relative to onset) ---
print(f"\nExtracting stimulus epochs ({TMIN_STIMULUS}s to {TMAX_STIMULUS}s)...")
epochs_stim = mne.Epochs(
    raw, good_events_161,
    event_id={'161': 161},
    tmin=TMIN_STIMULUS, tmax=TMAX_STIMULUS,
    baseline=None,          # no baseline correction
    picks=picks_meg,
    preload=True,
    reject=active_reject,
    flat=active_flat
)
print(f"  Stimulus epochs after rejection: {len(epochs_stim)} / {n_good_trials}")

# =============================================================================
# ALIGN TRIALS: DROP ANY TRIAL REJECTED IN EITHER EPOCH SET
# =============================================================================

print("\n=== ALIGNING TRIALS ACROSS EPOCH SETS ===")

# MNE stores a drop_log entry per original event; entries are empty tuples
# for kept trials and tuples of reason strings for dropped ones.
iti_dropped  = {i for i, log in enumerate(epochs_iti.drop_log)  if log}
stim_dropped = {i for i, log in enumerate(epochs_stim.drop_log) if log}
all_dropped  = iti_dropped | stim_dropped

print(f"  Trials dropped in ITI epochs:      {len(iti_dropped)}")
print(f"  Trials dropped in stimulus epochs: {len(stim_dropped)}")
print(f"  Trials dropped in either:          {len(all_dropped)}")

# Re-extract both epoch sets keeping only the mutually clean trials
good_indices  = sorted(set(range(n_good_trials)) - all_dropped)
clean_events  = good_events_161[good_indices]
n_clean       = len(clean_events)
print(f"  Clean trials kept: {n_clean}")

print("\nRe-extracting clean epochs...")
epochs_iti_clean = mne.Epochs(
    raw, clean_events,
    event_id={'161': 161},
    tmin=TMIN_ITI, tmax=TMAX_ITI,
    baseline=None,
    picks=picks_meg,
    preload=True,
    reject=active_reject,
    flat=active_flat
)

epochs_stim_clean = mne.Epochs(
    raw, clean_events,
    event_id={'161': 161},
    tmin=TMIN_STIMULUS, tmax=TMAX_STIMULUS,
    baseline=None,
    picks=picks_meg,
    preload=True,
    reject=active_reject,
    flat=active_flat
)

print(f"  Final ITI epochs:      {len(epochs_iti_clean)}")
print(f"  Final stimulus epochs: {len(epochs_stim_clean)}")

if len(epochs_iti_clean) != len(epochs_stim_clean):
    raise ValueError(
        f"Epoch count mismatch after realignment: "
        f"ITI={len(epochs_iti_clean)}, stimulus={len(epochs_stim_clean)}"
    )

# =============================================================================
# BUILD COMBINED DATASET
# =============================================================================

print("\n=== BUILDING COMBINED DATASET ===")

X_iti  = epochs_iti_clean.get_data()   # (N, n_channels, n_times_iti)
X_stim = epochs_stim_clean.get_data()  # (N, n_channels, n_times_stim)

# Both windows are 500ms long — verify timepoint counts match
n_times_iti  = X_iti.shape[2]
n_times_stim = X_stim.shape[2]
print(f"  ITI timepoints:      {n_times_iti}  ({TMIN_ITI}s to {TMAX_ITI}s)")
print(f"  Stimulus timepoints: {n_times_stim} ({TMIN_STIMULUS}s to {TMAX_STIMULUS}s)")

if n_times_iti != n_times_stim:
    raise ValueError(
        f"Timepoint mismatch: ITI has {n_times_iti}, stimulus has {n_times_stim}. "
        "Both windows must be the same length for concatenation."
    )

# Concatenate along the trial axis
X = np.concatenate([X_iti, X_stim], axis=0)   # (2N, n_channels, n_times)
y = np.concatenate([
    np.full(n_clean, LABEL_ITI),
    np.full(n_clean, LABEL_STIMULUS)
], axis=0)                                      # (2N,)

print(f"  Combined data shape: {X.shape}  (trials, channels, timepoints)")
print(f"  Labels: {n_clean} ITI (0) + {n_clean} stimulus (1) = {len(y)} total")

# =============================================================================
# NORMALISE
# =============================================================================

# Use stimulus epochs' info for the Scaler (both have same channels/info)
scaler = Scaler(info=epochs_stim_clean.info, scalings='mean')
X_norm = scaler.fit_transform(X)
print(f"\nNormalized data shape: {X_norm.shape}")

# =============================================================================
# TEMPORAL DECODING: STIMULUS PRESENCE (ITI vs. STIMULUS)
# =============================================================================

print("\n" + "="*50)
print("TEMPORAL DECODING: STIMULUS PRESENCE (ITI vs. STIMULUS)")
print("="*50)

clf = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf_pipeline = make_pipeline(Vectorizer(), clf)
sliding      = SlidingEstimator(clf_pipeline, scoring='accuracy', n_jobs=1)

print("Running cross-validated temporal decoding (this may take a moment)...")
scores_folds = cross_val_multiscore(sliding, X_norm, y, cv=cv, n_jobs=1)
scores       = scores_folds.mean(axis=0)   # (n_timepoints,)

# Time axis — use the stimulus epoch times (-0.1 to 0.4s), since that is
# the window where the neural signal of interest lives. At each timepoint T,
# the classifier compares ITI trial data against stimulus trial data, so the
# meaningful label is "where in the stimulus epoch does decoding peak".
times = epochs_stim_clean.times  # -0.1 to 0.4s

print(f"\nTemporal Decoding Results:")
print(f"  Time window: {times[0]:.3f}s to {times[-1]:.3f}s (relative to stimulus onset)")
print(f"  Timepoints: {len(times)}")
print(f"  Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Peak accuracy: {scores.max():.3f} at {times[scores.argmax()]:.3f}s post-stimulus onset")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

chance_level = 0.5   # binary classification
above_chance = scores > chance_level
pct_above    = (above_chance.sum() / len(above_chance)) * 100

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"  Chance level:                    {chance_level:.3f}")
print(f"  Mean accuracy:                   {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Peak accuracy:                   {scores.max():.3f} at {times[scores.argmax()]:.3f}s post-stimulus onset")
print(f"  % timepoints above chance:       {pct_above:.1f}%")
print(f"  N trials per class:              {n_clean}")
print(f"  Total trials:                    {len(y)}")

# =============================================================================
# VISUALISATION
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

peak_time = times[scores.argmax()]

ax.plot(times, scores, linewidth=2.5, color='#2E86AB', label=f'Subject {subj}')
ax.axhline(chance_level, color='red', linestyle='--', linewidth=2,
           label=f'Chance ({chance_level:.2f})', alpha=0.7)
ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
           alpha=0.5, label='Stimulus Onset')
ax.axvline(peak_time, color='green', linestyle=':', linewidth=2,
           alpha=0.7, label=f'Peak ({peak_time:.3f}s)')
ax.fill_between(times, chance_level, scores,
                where=(scores > chance_level),
                alpha=0.2, color='#2E86AB', label='Above chance')

ax.set_xlabel('Time relative to stimulus onset (s)', fontsize=12)
ax.set_ylabel('Classification Accuracy', fontsize=12)
ax.set_title(
    f'Temporal Decoding Sanity Check: {subj} — Stimulus Presence (ITI vs. Stimulus) [Minimal Preprocessing]',
    fontsize=13, fontweight='bold'
)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.3, 1.0])
ax.set_xlim([times[0], times[-1]])

plt.tight_layout()

analysis_dir = os.path.join(data_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)
# --- Output filenames distinguish minimal from full pipeline results ---
# plot_path = os.path.join(analysis_dir, f'{subj}_temporal_decoding_stimulus_presence.png')
plot_path = os.path.join(analysis_dir, f'{subj}_temporal_decoding_stimulus_presence_minimal.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {plot_path}")
plt.show()

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = {
    'subject_id':       subj,
    'decoded_label':    'stimulus_presence',
    'label_encoding':   {0: 'ITI (blank)', 1: 'stimulus present'},
    'scores':           scores,
    'scores_per_fold':  scores_folds,
    'times':            times,           # time axis (-0.1 to 0.4s, relative to stimulus onset)
    'mean_accuracy':    scores.mean(),
    'std_accuracy':     scores.std(),
    'peak_accuracy':    scores.max(),
    'peak_time':        times[scores.argmax()],  # seconds post-stimulus onset
    'chance_level':     chance_level,
    'n_trials_per_class': n_clean,
    'tmin_iti':         TMIN_ITI,
    'tmax_iti':         TMAX_ITI,
    'tmin_stimulus':    TMIN_STIMULUS,
    'tmax_stimulus':    TMAX_STIMULUS,
    'n_timepoints':     len(times),
}

# results_path = os.path.join(analysis_dir, f'{subj}_temporal_decoding_stimulus_presence_results.pkl')
results_path = os.path.join(analysis_dir, f'{subj}_temporal_decoding_stimulus_presence_minimal_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n{'='*50}")
print(f"Results saved to: {results_path}")
print(f"{'='*50}")
print("\nSaved data structure:")
print(f"  - scores:            {scores.shape}  (mean accuracy per timepoint)")
print(f"  - scores_per_fold:   {scores_folds.shape}  (per-fold accuracy)")
print(f"  - times:             {times.shape}  (time values in seconds)")
print(f"  - label_encoding:    {{0: 'ITI (blank)', 1: 'stimulus present'}}")
print(f"  - Summary statistics (mean, std, peak, etc.)")
print("\nStimulus presence decoding complete!")