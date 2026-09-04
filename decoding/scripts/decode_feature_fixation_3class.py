"""
Temporal decoding: prompt_feature -- 3-class version, fixation epochs

Identical pipeline to decode_feature_fixation.py, with one difference: the
'2' and '2gap' prompt_feature labels are merged into a single combined class
('2_or_2gap') before encoding, leaving 3 classes total: 2_or_2gap / 3 / 4.

Why merge: from ar_v3_independent_trial_feature_summary_group_averages.csv
(trial-level, onset stream), the per-feature group means are roughly
2 -> 23.8, 2gap -> 21.2, 3 -> 51.7, 4 -> 56.4 trials/subject. As 4 separate
classes, '2gap' (21.2) vs '4' (56.4) is a ~2.7x imbalance. Merged, 2_or_2gap
(~45.0) sits much closer to 3 (~51.7) and 4 (~56.4) -- largest/smallest ratio
drops to ~1.25x, giving 3 genuinely comparable class sizes (and a chance
level of 1/3 that's actually representative of the data, rather than one a
majority-class classifier could beat by exploiting imbalance). Exact counts
at the fixation-epoch level will differ from these trial-level numbers, but
the same relative pattern should hold since fixation counts scale with the
trials they come from.

CV is grouped by trial-of-origin (StratifiedGroupKFold on (block, trial)),
not plain StratifiedKFold: a single trial can contribute more than one
fixation epoch (see single_subject_ar_v3_independent_fixation.py -- a trial
may have zero, one, or many qualifying fixations), and sibling epochs from
the same trial share both a label and correlated trial-level signal (drift,
head position, subject state). Without grouping, siblings can split across
train/test, letting the classifier exploit trial identity instead of
genuine timepoint-locked encoding. StratifiedGroupKFold still tries to keep
class proportions balanced across folds subject to that grouping
constraint. decode_feature_fixation.py (4-class script) now uses the same
grouped CV.

Fixation-window filter ('_fixwin' in the output filenames): after the
ground_truth filter, epochs are further restricted to fixations that
occurred 1.0-2.0s before that trial's response (164) event -- i.e. not the
final second immediately before the response (possibly confounded with
motor preparation / decision execution) and not more than 2s before it
(possibly early exploratory scanning). Computed directly from data already
in the saved epoch files: response_sample (per-trial, in epochs.metadata)
minus the epoch's own anchor sample (epochs.events[:, 0]), divided by
sfreq. LogisticRegression now also uses class_weight='balanced', since this
filter is expected to shift/worsen class balance relative to the
unfiltered epochs.

Temporal averaging ('_avg10ms' in the output filenames): right after X is
extracted, every 10 consecutive timepoints (non-overlapping bins, 10ms each
at sfreq=1000Hz) are averaged down to one -- 1001 samples -> 1000 (final
leftover sample dropped) -> 100 bins. This happens BEFORE the Scaler, not
after: the Scaler's own per-channel mean/std are then computed on the
smoothed 100-bin signal, which is what actually carries the noise reduction
through scaling and into the classifier, rather than scaling the original
noisy signal and averaging afterward (mathematically not the same, since
the Scaler's std would differ between the two orderings). `times` is
rebuilt as the corresponding bin-mean timestamps at this same point, and
NOT reassigned from epochs.times later on (that 1001-length array no
longer matches scores_temporal's length after binning).

Per-fold scaling ('_noleak' in the output filenames): there is no longer a
single global Scaler.fit_transform(X) applied before CV. Fitting one Scaler
on the full dataset let each fold's held-out epochs contribute to the
per-channel mean/std used to normalize them -- mild (unsupervised, affine)
but still leakage. Scaling now happens inside an explicit per-fold loop
(replacing cross_val_multiscore), refit on each training split only and
applied (not refit) to that fold's test split. The Scaler is still applied
to the full 3D (n_epochs, n_channels, n_times) array before SlidingEstimator
slices it, same as before -- putting it inside clf_pipeline instead would
have SlidingEstimator clone it per timepoint and fit each clone on an
already-2D slice, silently changing "one mean/std per channel pooled across
the whole epoch" into "one mean/std per channel per timepoint." The
class-presence-per-fold check (above) still previews folds via
cv.split(X, y, groups) -- same X, y, groups, and cv (deterministic,
int random_state) as the scoring loop below uses, so those folds are
guaranteed identical and the check doesn't need to be repeated there.
results['scores_folds'] now also stores the full (5, n_times) per-fold
score array alongside the fold-averaged 'scores', so per-fold spread isn't
discarded.

Everything else -- loading, scaling, SlidingEstimator, summary stats,
plotting, results schema -- is identical to the 4-class script. See
decode_feature_fixation.py for the full step-by-step docstring.

Usage:
    python decode_feature_fixation_3class.py --subject R2649 --base_dir /scratch/djn8068/meg_project/data
"""

import argparse
import os
import pickle

import numpy as np
import mne
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mne.decoding import Scaler, Vectorizer, SlidingEstimator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mne.set_log_level(verbose='WARNING')

FIXATION_TYPES = ['onset', 'median', 'offset']

# '2' and '2gap' collapse onto the same combined label; '3' and '4' pass through unchanged.
FEATURE_MERGE_MAP = {'2': '2_or_2gap', '2gap': '2_or_2gap', '3': '3', '4': '4'}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description='3-class prompt_feature decoding on fixation epochs (2/2gap merged)')
parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g. R2649)')
parser.add_argument('--base_dir', type=str,
                    default='/scratch/djn8068/meg_project/data',
                    help='Base directory containing subject folders')
args = parser.parse_args()

subj = args.subject
base_dir = args.base_dir

data_dir = os.path.join(base_dir, subj)
preproc_dir = os.path.join(data_dir, 'preprocessing')
analysis_dir = os.path.join(data_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)

print(f'\n{"="*60}')
print(f'PROCESSING (3-class prompt_feature, fixation epochs): {subj}')
print(f'{"="*60}')

# =============================================================================
# ONE FIXATION TYPE AT A TIME
# =============================================================================

for epoch_type in FIXATION_TYPES:
    print(f'\n{"="*60}')
    print(f'{subj} -- {epoch_type} fixation epochs')
    print(f'{"="*60}')

    # '_groupcv_fixwin_classcheck_avg10ms_noleak' suffix distinguishes these
    # from earlier runs already sitting in analysis_dir -- new name so this
    # rerun writes fresh output instead of hitting the already-exists skip
    # guard below and instead of overwriting older results, which are worth
    # keeping around for comparison. Each segment marks one accumulated
    # change: '_classcheck' verifies every class survives in every fold's
    # training set before scoring; '_avg10ms' averages every 10 consecutive
    # timepoints before the Scaler; '_noleak' refits the Scaler per-fold
    # (train split only) instead of once globally before CV.
    results_path = os.path.join(
        analysis_dir, f'{subj}_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_fixwin_classcheck_avg10ms_noleak_results.pkl')
    if os.path.exists(results_path):
        print(f'  SKIPPING: results already exist at {results_path}')
        continue

    epoch_path = os.path.join(
        preproc_dir, f'{subj}_localizer_ar_v3_independent_fixation_{epoch_type}-epo.fif')
    if not os.path.exists(epoch_path):
        print(f'  ERROR: epoch file not found: {epoch_path} -- skipping this stream')
        continue

    print(f'  Loading {epoch_path}')
    epochs = mne.read_epochs(epoch_path, preload=True)
    print(f'  Epochs — Time: {epochs.tmin:.3f} to {epochs.tmax:.3f}s, Shape: {epochs.get_data().shape}')

    # Exclude ground_truth == 'none' trials -- same reasoning as the 4-class script.
    n_before = len(epochs)
    epochs = epochs[epochs.metadata['ground_truth'] != 'none']
    n_excluded = n_before - len(epochs)
    print(f'  Excluded {n_excluded} trial(s) with ground_truth == \'none\' ({n_before} -> {len(epochs)})')

    # Fixation-window filter: keep only fixations that occurred 1.0-2.0s
    # before that trial's response (164) event. response_sample (per-trial,
    # absolute raw sample) is already in epochs.metadata; the epoch's own
    # anchor sample (the fixation onset/median/offset instant it was built
    # around) is epochs.events[:, 0] -- both come straight from the saved
    # epoch files, no re-epoching needed.
    sfreq = epochs.info['sfreq']
    response_sample = epochs.metadata['response_sample'].to_numpy()
    anchor_sample = epochs.events[:, 0]
    time_before_response = (response_sample - anchor_sample) / sfreq
    in_fix_window = (time_before_response >= 1.0) & (time_before_response <= 2.0)
    n_before_fixwin = len(epochs)
    epochs = epochs[in_fix_window]
    print(f'  Fixation window (1.0-2.0s before response): kept {len(epochs)} / {n_before_fixwin} epochs')

    if len(epochs) < 20:
        print(f'  WARNING: only {len(epochs)} epochs remain -- skipping (too few for 5-fold CV)')
        continue

    # =========================================================================
    # EXTRACT X, y -- merge '2'/'2gap' before encoding
    # =========================================================================

    X = epochs.get_data()  # (n_trials, n_channels, n_times)

    # Average every 10 consecutive timepoints (non-overlapping bins) to
    # reduce noise before fitting the classifier. Done here, right after
    # extracting the raw epoch data, so the Scaler's own mean/std (below)
    # are computed on the smoothed signal, not the original per-millisecond
    # noise -- averaging first is what actually carries the noise reduction
    # through the rest of the pipeline; scaling first and averaging after
    # would give the Scaler different (noisier) statistics to work with.
    # 1001 samples isn't a clean multiple of 10, so the final leftover
    # sample is dropped: 1001 -> 1000 -> 100 bins of 10ms each (sfreq=1000Hz).
    n_times_full = X.shape[-1]
    n_bins = n_times_full // 10
    X = X[:, :, :n_bins * 10].reshape(X.shape[0], X.shape[1], n_bins, 10).mean(axis=-1)
    times = epochs.times[:n_bins * 10].reshape(n_bins, 10).mean(axis=-1)
    print(f'  Averaged every 10 timepoints: {n_times_full} -> {n_bins} bins, shape now {X.shape}')

    y_raw = epochs.metadata['prompt_feature'].to_numpy()
    y_merged = np.array([FEATURE_MERGE_MAP[label] for label in y_raw])

    le = LabelEncoder()
    y = le.fit_transform(y_merged)

    print(f'  Label encoding (post-merge): {dict(zip(le.classes_, le.transform(le.classes_)))}')
    print(f'  Number of fixation epochs: {len(y)}')
    unique_vals, counts = np.unique(y, return_counts=True)
    print(f'  Class distribution: {dict(zip(le.inverse_transform(unique_vals), counts))}')

    if len(unique_vals) < 3:
        print(f'  WARNING: only {len(unique_vals)} of 3 classes present -- skipping this stream')
        continue

    # A trial can contribute multiple fixation epochs (onset/median/offset
    # are each derived per-fixation, and a trial can have several qualifying
    # fixations -- see single_subject_ar_v3_independent_fixation.py). Group
    # by trial-of-origin so CV never splits siblings across train/test.
    groups = (epochs.metadata['block'].astype(str) + '_' + epochs.metadata['trial'].astype(str)).to_numpy()
    n_groups = len(np.unique(groups))
    print(f'  Number of unique trials (CV groups): {n_groups} (from {len(y)} epochs)')

    if n_groups < 5:
        print(f'  WARNING: only {n_groups} unique trials -- too few for 5-fold grouped CV -- skipping this stream')
        continue

    # =========================================================================
    # TEMPORAL DECODING
    # =========================================================================
    # NOTE: no global Scaler.fit_transform(X) here -- scaling now happens
    # per-fold, inside the loop below, refit on each training split only.
    # A single Scaler fit on the full X before CV would let each fold's
    # held-out epochs contribute to the per-channel mean/std used to
    # normalize them -- mild (unsupervised, affine) but still leakage.
    # X stays exactly the array avg10ms already built (E2, 157, 100) all
    # the way through this section -- never re-fetched from epochs here,
    # so the averaging step is untouched.

    classifier = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced')
    # Grouped by trial-of-origin (see `groups` above) so sibling fixation
    # epochs from the same trial can't split across train/test.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Verify every class is represented in every fold's training set.
    # StratifiedGroupKFold only makes a best-effort attempt at
    # stratification (unlike plain StratifiedKFold, it does NOT guarantee
    # n_splits <= min class count) -- if a class's epochs are concentrated
    # in very few trials, grouping can push all of them into a single
    # fold, leaving every OTHER fold's training set entirely missing that
    # class for that CV iteration. LogisticRegression.fit() silently
    # infers classes_ from whatever's present in the training data (no
    # error), and .predict() can never output a class that wasn't in
    # training -- so cross_val_multiscore would score every test epoch of
    # that missing class as automatically wrong, not because of a real
    # deficit but because the model was structurally incapable of
    # predicting it. No exception fires anywhere in that chain, so this
    # has to be checked explicitly, in advance, rather than trusted to
    # surface on its own. random_state is an int, so this .split() call is
    # deterministic and reproduces exactly the folds the per-fold scoring
    # loop below will use (same X, y, groups, cv), so this check remains
    # valid for that loop without needing to be repeated there.
    all_classes = set(np.unique(y))
    folds_missing_classes = {}
    for fold_i, (train_idx, _) in enumerate(cv.split(X, y, groups)):
        missing = all_classes - set(np.unique(y[train_idx]))
        if missing:
            folds_missing_classes[fold_i] = list(le.inverse_transform(sorted(missing)))

    if folds_missing_classes:
        print(f'  WARNING: {len(folds_missing_classes)} / 5 fold(s) missing a class from their '
              f'training set -- skipping this stream: {folds_missing_classes}')
        continue
    print(f"  Verified: all {len(all_classes)} classes present in every fold's training set")

    clf_pipeline = make_pipeline(Vectorizer(), classifier)

    # Per-fold scoring loop, replacing cross_val_multiscore, specifically so
    # the Scaler can be refit on each training split only (see the NOTE
    # above the removed global Scaler.fit_transform(X) call). The Scaler
    # must be applied to the full (n_epochs, n_channels, n_times) array
    # BEFORE SlidingEstimator slices it -- if it were placed inside
    # clf_pipeline instead (make_pipeline(Scaler(...), Vectorizer(),
    # classifier)), SlidingEstimator would clone that pipeline once per
    # timepoint and each clone would independently fit the Scaler on a
    # single already-sliced (n_epochs, n_channels) 2D timepoint -- silently
    # turning "one mean/std per channel, pooled across the whole epoch
    # window" into "one mean/std per channel per timepoint," a materially
    # different (and unintended) preprocessing choice. So the Scaler is
    # fit explicitly here, per fold, on the full 3D training slice, before
    # SlidingEstimator ever sees it.
    n_times = X.shape[-1]
    scores_folds = np.zeros((5, n_times))
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        scaler_fold = Scaler(scalings='mean')
        X_train = scaler_fold.fit_transform(X[train_idx])
        X_test = scaler_fold.transform(X[test_idx])

        sliding_fold = SlidingEstimator(clone(clf_pipeline), scoring='accuracy', n_jobs=1)
        sliding_fold.fit(X_train, y[train_idx])
        scores_folds[fold_i] = sliding_fold.score(X_test, y[test_idx])

    scores_temporal = scores_folds.mean(axis=0)
    # NOTE: `times` was already set above (the 10-sample-averaged bin
    # centers) when X was built -- not reassigned from epochs.times here,
    # since that would revert to the original 1001-sample axis and no
    # longer match scores_temporal's length (n_bins).

    print(f'  Mean accuracy across time: {scores_temporal.mean():.3f} ± {scores_temporal.std():.3f}')
    peak_time = times[scores_temporal.argmax()]
    print(f'  Peak accuracy: {scores_temporal.max():.3f} at {peak_time:.3f}s')

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================

    n_classes = len(np.unique(y))
    chance_level = 1.0 / n_classes
    above_chance = scores_temporal > chance_level
    pct_above_chance = (above_chance.sum() / len(above_chance)) * 100
    print(f'  Chance level: {chance_level:.3f}  |  % timepoints above chance: {pct_above_chance:.1f}%')

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(times, scores_temporal, linewidth=2.5, label=f'Subject {subj}', color='#2E86AB')
    ax.axhline(chance_level, color='red', linestyle='--', linewidth=2, label=f'Chance ({chance_level:.3f})', alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Fixation (t=0)')
    ax.axvline(peak_time, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Peak ({peak_time:.3f}s)')
    ax.fill_between(times, chance_level, scores_temporal,
                    where=(scores_temporal > chance_level),
                    alpha=0.2, color='#2E86AB', label='Above chance')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title(f'Temporal Decoding: {subj} — Prompt Feature (2∪2gap/3/4), {epoch_type} fixation (1-2s pre-response)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim([times[0], times[-1]])
    plt.tight_layout()

    plot_path = os.path.join(
        analysis_dir, f'{subj}_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_fixwin_classcheck_avg10ms_noleak.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved plot: {plot_path}')

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    results = {
        'subject_id': subj,
        'decoded_label': 'prompt_feature_3class',
        'epoch_type': epoch_type,
        'feature_merge_map': FEATURE_MERGE_MAP,
        'label_encoding': dict(zip(le.classes_, le.transform(le.classes_))),
        'scores': scores_temporal,
        'scores_folds': scores_folds,
        'times': times,
        'mean_accuracy': scores_temporal.mean(),
        'std_accuracy': scores_temporal.std(),
        'peak_accuracy': scores_temporal.max(),
        'peak_time': peak_time,
        'chance_level': chance_level,
        'n_classes': n_classes,
        'n_trials': len(y),
        'n_unique_trials': n_groups,
        'cv': 'StratifiedGroupKFold',
        'class_weight': 'balanced',
        'fixation_window_s': (1.0, 2.0),
        'averaging_bin_size': 10,
        'scaler': 'per-fold (refit on train split only)',
        'time_window': (times[0], times[-1]),
        'n_timepoints': len(times)
    }

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'  Saved: {results_path}')

print(f'\n{"="*60}')
print(f'COMPLETE: {subj}')
print(f'{"="*60}')
