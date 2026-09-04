"""
Single-Subject Temporal Decoding: Prompt Color (black vs. white)
Designed to be called by the SLURM job array with a subject ID argument.

Loads onset epochs from the independent onset/response rejection pipeline
(single_subject_ar_v3_independent.py), i.e.
{subj}_{trial_type}_ar_v3_independent_onset-epo.fif, and writes results to
{subj}_ar_v3_independent_temporal_decoding_prompt_color_results.pkl -- the
naming compile_bw_decoding_onset_independent.py expects.

Usage:
    python decode_bw_single_subject.py --subject R2649 --base_dir /scratch/djn8068/meg_project/data
"""

import argparse
import numpy as np
import mne
import pickle
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mne.decoding import Scaler, cross_val_multiscore, Vectorizer, SlidingEstimator

mne.set_log_level(verbose='WARNING')

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description='Temporal decoding for a single subject')
parser.add_argument('--subject', type=str, required=True,
                    help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                    default='/scratch/djn8068/meg_project/data',
                    help='Base directory containing subject folders')
parser.add_argument('--trial_type', type=str, default='localizer',
                    help='Trial type (default: localizer)')
parser.add_argument('--tmin', type=float, default=-0.1,
                    help='Crop start time in seconds (default: -0.1)')
parser.add_argument('--tmax', type=float, default=0.75,
                    help='Crop end time in seconds (default: 0.75)')
parser.add_argument('--n_splits', type=int, default=5,
                    help='Number of CV folds (default: 5)')
args = parser.parse_args()

subj       = args.subject
base_dir   = args.base_dir
trial_type = args.trial_type
TMIN       = args.tmin
TMAX       = args.tmax
N_SPLITS   = args.n_splits

print(f'\n{"="*60}')
print(f'DECODING: {subj}')
print(f'{"="*60}')
print(f'Base dir:   {base_dir}')
print(f'Trial type: {trial_type}')
print(f'Time window: {TMIN}s to {TMAX}s')
print(f'CV folds:   {N_SPLITS}')

# =============================================================================
# PATHS
# =============================================================================

data_dir    = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
analysis_dir = os.path.join(data_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)

onset_path = os.path.join(preproc_dir, f'{subj}_{trial_type}_ar_v3_independent_onset-epo.fif')

if not os.path.exists(onset_path):
    print(f'ERROR: AR onset epoch file not found: {onset_path}')
    print('Exiting — check that batch_autoreject.py completed for this subject.')
    sys.exit(1)

# =============================================================================
# LOAD EPOCHS
# =============================================================================

print(f'\nLoading: {onset_path}')
onset_epochs = mne.read_epochs(onset_path, preload=True)

if onset_epochs.metadata is None or 'prompt_color' not in onset_epochs.metadata.columns:
    print('ERROR: metadata missing or no prompt_color column.')
    sys.exit(1)

onset_epochs = onset_epochs.copy().crop(tmin=TMIN, tmax=TMAX)
print(f'Epochs shape: {onset_epochs.get_data().shape}  '
      f'({onset_epochs.tmin:.3f}s to {onset_epochs.tmax:.3f}s)')

# =============================================================================
# LABELS
# =============================================================================

X = onset_epochs.get_data()  # (n_trials, n_channels, n_times)

le = LabelEncoder()
y_raw = onset_epochs.metadata['prompt_color'].to_numpy()
y = le.fit_transform(y_raw)  # 'black' -> 0, 'white' -> 1

n_classes = len(np.unique(y))
if n_classes < 2:
    print(f'ERROR: Only {n_classes} class found in prompt_color — need at least 2.')
    sys.exit(1)

unique_vals, counts = np.unique(y, return_counts=True)
print(f'Labels: {dict(zip(le.inverse_transform(unique_vals), counts))}')
print(f'Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}')

# =============================================================================
# NORMALIZE
# =============================================================================

scaler = Scaler(info=onset_epochs.info, scalings='mean')
X_norm = scaler.fit_transform(X)

# =============================================================================
# SLIDING ESTIMATOR DECODING
# =============================================================================

print(f'\nRunning sliding estimator ({N_SPLITS}-fold CV)...')

clf_pipeline = make_pipeline(
    Vectorizer(),
    LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
)
# n_jobs=-1 inside SlidingEstimator parallelises across timepoints
# n_jobs=1 on cross_val_multiscore avoids oversubscription with the above
sliding = SlidingEstimator(clf_pipeline, scoring='accuracy', n_jobs=-1)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

scores_folds = cross_val_multiscore(sliding, X_norm, y, cv=cv, n_jobs=1)
scores = scores_folds.mean(axis=0)  # (n_timepoints,)
times  = onset_epochs.times

peak_time = times[scores.argmax()]
print(f'Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}')
print(f'Peak accuracy: {scores.max():.3f} at {peak_time:.3f}s')

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = {
    'subject_id':     subj,
    'decoded_label':  'prompt_color',
    'label_encoding': dict(zip(le.classes_, le.transform(le.classes_))),
    'scores':         scores,
    'scores_per_fold': scores_folds,
    'times':          times,
    'mean_accuracy':  float(scores.mean()),
    'std_accuracy':   float(scores.std()),
    'peak_accuracy':  float(scores.max()),
    'peak_time':      float(peak_time),
    'chance_level':   1.0 / n_classes,
    'n_classes':      n_classes,
    'n_trials':       len(y),
    'time_window':    (float(times[0]), float(times[-1])),
    'n_timepoints':   len(times),
    'tmin':           TMIN,
    'tmax':           TMAX,
    'n_splits':       N_SPLITS,
}

results_path = os.path.join(analysis_dir,
                            f'{subj}_ar_v3_independent_temporal_decoding_prompt_color_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f'\nResults saved: {results_path}')
print(f'\n{"="*60}')
print(f'COMPLETE: {subj}')
print(f'{"="*60}')