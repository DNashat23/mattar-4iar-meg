"""
Group-level temporal decoding analysis -- Prompt Feature, fixation epochs
4-class and 3-class, all three fixation streams (onset/median/offset),
across three decoding-pipeline generations (pre-fix / GroupCV / GroupCV +
fixation-window) -- 15 configs total, each producing its own group plot.

Loosely based on compile_bw_decoding_onset_independent.py, but: (1) plots
grand mean +/- SEM only, no per-subject overlay lines -- with cohorts this
size, individual traces are visual noise, not signal; (2) does NOT glob all
R* folders -- it uses the fixed 45-subject cohort from slurm_decode_feature_
fixation.sh / slurm_decode_feature_fixation_3class.sh (the 48-subject
independent-pipeline cohort minus R2652, R2667, R2760, which have no
*_localizer_fixations.csv), since that's the exact set the SLURM arrays ran
the decoders on. Not every subject/stream survives every config -- missing
files are skipped (see compile_one's missing_subjects handling), and each
plot's title reports however many of the 45 actually loaded (n_subjects),
which varies per config/stream.

Reads, per subject, from {subj}/analysis/ (three filename generations):
  {subj}_temporal_decoding_prompt_feature_[3class_]fixation_{onset,median,offset}_results.pkl
  {subj}_temporal_decoding_prompt_feature_3class_fixation_{onset,median,offset}_groupcv_results.pkl
  {subj}_temporal_decoding_prompt_feature_[3class_]fixation_{onset,median,offset}_groupcv_fixwin_results.pkl

Writes, per config:
  group_temporal_decoding_prompt_feature_[3class_]fixation_{epoch_type}[_groupcv][_fixwin].png / _results.pkl

Usage:
    python compile_feature_fixation_decoding.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

base_dir = '/Users/dariushnashat/Documents/Mattar Lab Internship'

SUBJECTS = [
    'R2662', 'R2665', 'R2671', 'R2675', 'R2726', 'R2728', 'R2312', 'R2538',
    'R2611', 'R2612', 'R2621', 'R2649', 'R2650', 'R2657', 'R2663', 'R2669',
    'R2672', 'R2674', 'R2676', 'R2678', 'R2679', 'R2682', 'R2683', 'R2684',
    'R2688', 'R2700', 'R2701', 'R2702', 'R2733', 'R2734', 'R2743', 'R2749',
    'R2763', 'R2765', 'R2772', 'R2775', 'R2776', 'R2780', 'R2781', 'R2787',
    'R2792', 'R2795', 'R2796', 'R2737', 'R2664',
]

FIXATION_TYPES = ['onset', 'median', 'offset']

# One config per decoding type: (result filename template, output prefix, title)
CONFIGS = []
for epoch_type in FIXATION_TYPES:
    CONFIGS.append({
        'key': f'fixation_{epoch_type}',
        'result_filename': f'{{subj}}_temporal_decoding_prompt_feature_fixation_{epoch_type}_results.pkl',
        'out_prefix': f'group_temporal_decoding_prompt_feature_fixation_{epoch_type}',
        'title': f'Prompt Feature (2/2gap/3/4), {epoch_type} fixation',
    })
for epoch_type in FIXATION_TYPES:
    CONFIGS.append({
        'key': f'3class_fixation_{epoch_type}',
        'result_filename': f'{{subj}}_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_results.pkl',
        'out_prefix': f'group_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}',
        'title': f'Prompt Feature (2∪2gap/3/4), {epoch_type} fixation',
    })

# GroupCV rerun of the 3-class decoder: decode_feature_fixation_3class.py was
# patched to use StratifiedGroupKFold (grouped by trial-of-origin) instead of
# plain StratifiedKFold, since a single trial can contribute multiple fixation
# epochs (sharing a label and correlated trial-level signal) that must not
# split across train/test. Written to '_groupcv'-suffixed filenames so the
# pre-fix results above are left untouched for comparison.
for epoch_type in FIXATION_TYPES:
    CONFIGS.append({
        'key': f'3class_fixation_{epoch_type}_groupcv',
        'result_filename': f'{{subj}}_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_results.pkl',
        'out_prefix': f'group_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv',
        'title': f'Prompt Feature (2∪2gap/3/4), {epoch_type} fixation -- GroupCV (trial-grouped)',
    })

# GroupCV + fixation-window rerun: both decode_feature_fixation.py (4-class)
# and decode_feature_fixation_3class.py (3-class) now also restrict epochs to
# fixations 1.0-2.0s before that trial's response, and use
# class_weight='balanced' (see both scripts' docstrings). Not every
# subject/stream survives -- '2' is the fastest-solved, already-smallest
# condition, so some subjects lose it entirely (all fixations occur before
# the window even starts) or lose it only in the offset stream (the
# closest-to-response fixation); a few subjects also fall under the
# minimum-epoch/minimum-trial-group floor across every class. compile_one's
# missing-subject handling already accounts for this -- n_subjects in each
# plot reflects however many of the 45 actually survived for that stream.
for epoch_type in FIXATION_TYPES:
    CONFIGS.append({
        'key': f'fixation_{epoch_type}_groupcv_fixwin',
        'result_filename': f'{{subj}}_temporal_decoding_prompt_feature_fixation_{epoch_type}_groupcv_fixwin_results.pkl',
        'out_prefix': f'group_temporal_decoding_prompt_feature_fixation_{epoch_type}_groupcv_fixwin',
        'title': f'Prompt Feature (2/2gap/3/4), {epoch_type} fixation -- GroupCV + fixation window (1-2s pre-response)',
    })
for epoch_type in FIXATION_TYPES:
    CONFIGS.append({
        'key': f'3class_fixation_{epoch_type}_groupcv_fixwin',
        'result_filename': f'{{subj}}_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_fixwin_results.pkl',
        'out_prefix': f'group_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_fixwin',
        'title': f'Prompt Feature (2∪2gap/3/4), {epoch_type} fixation -- GroupCV + fixation window (1-2s pre-response)',
    })

# GroupCV + fixation-window + class-presence-per-fold check + 10ms temporal
# averaging + per-fold Scaler (no global-fit leakage): the latest round of
# fixes to decode_feature_fixation_3class.py. '_classcheck' verifies every
# class survives in every fold's training set before scoring (a fold could
# otherwise be structurally incapable of predicting a class that got
# concentrated into too few trials); '_avg10ms' averages every 10 consecutive
# timepoints (10ms at 1000Hz) before scaling, to reduce noise; '_noleak'
# refits the Scaler on each fold's training split only, instead of once
# globally before CV. Only run for the 3-class decoder so far -- the 4-class
# script has the same code but hasn't been resubmitted on the cluster yet.
for epoch_type in FIXATION_TYPES:
    CONFIGS.append({
        'key': f'3class_fixation_{epoch_type}_groupcv_fixwin_classcheck_avg10ms_noleak',
        'result_filename': f'{{subj}}_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_fixwin_classcheck_avg10ms_noleak_results.pkl',
        'out_prefix': f'group_temporal_decoding_prompt_feature_3class_fixation_{epoch_type}_groupcv_fixwin_classcheck_avg10ms_noleak',
        'title': f'Prompt Feature (2∪2gap/3/4), {epoch_type} fixation -- GroupCV + fixwin + 10ms avg + no-leak scaling',
    })


def compile_one(config):
    print("=" * 70)
    print(f"GROUP-LEVEL TEMPORAL DECODING -- {config['key']}")
    print("=" * 70)

    # Skip if this config's outputs already exist -- this loop reprocesses
    # every config in CONFIGS on every invocation (there's no per-config
    # memory otherwise), so without this guard, adding one new config at
    # the bottom of the list means silently rewriting every PNG/pkl for
    # every older config too, on every run. Content should be identical
    # either way (same source per-subject pkls, same processing code), but
    # the rewrite still touches every file's mtime, which is what actually
    # prompted this guard. Delete a config's outputs manually to force a
    # recompute of just that one.
    plot_path = os.path.join(base_dir, f"{config['out_prefix']}.png")
    group_results_path = os.path.join(base_dir, f"{config['out_prefix']}_results.pkl")
    if os.path.exists(plot_path) and os.path.exists(group_results_path):
        print(f"SKIPPING: outputs already exist ({plot_path})")
        return

    all_results = []
    subject_ids = []
    missing_subjects = []
    failed_subjects = []

    for subj in SUBJECTS:
        pkl_path = os.path.join(base_dir, subj, 'analysis', config['result_filename'].format(subj=subj))

        if not os.path.exists(pkl_path):
            missing_subjects.append(subj)
            continue

        try:
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
            all_results.append(results)
            subject_ids.append(subj)
        except Exception as e:
            failed_subjects.append((subj, str(e)))
            print(f"  ERROR loading {subj}: {e}")

    print(f"Loaded {len(subject_ids)} / {len(SUBJECTS)} subjects")
    if missing_subjects:
        print(f"Missing results for: {missing_subjects}")
    if failed_subjects:
        print(f"Failed to load: {failed_subjects}")

    if len(subject_ids) == 0:
        print("ERROR: no subject data found -- skipping this type")
        return

    # -------------------------------------------------------------------
    # ORGANIZE DATA
    # -------------------------------------------------------------------

    all_scores = []
    times = None
    chance_level = None

    for i, result in enumerate(all_results):
        all_scores.append(result['scores'])
        if times is None:
            times = result['times']
            chance_level = result['chance_level']
        if not np.allclose(times, result['times']):
            print(f"  WARNING: {subject_ids[i]} has different timepoints!")

    all_scores = np.array(all_scores)  # (n_subjects, n_timepoints)
    n_subjects = len(subject_ids)

    print(f"Data shape: {all_scores.shape} (subjects x timepoints)")
    print(f"Time window: {times[0]:.3f}s to {times[-1]:.3f}s")
    print(f"Chance level: {chance_level:.3f}")

    # -------------------------------------------------------------------
    # GROUP STATISTICS
    # -------------------------------------------------------------------

    grand_mean = all_scores.mean(axis=0)
    grand_std = all_scores.std(axis=0)
    grand_sem = stats.sem(all_scores, axis=0)
    ci_95 = 1.96 * grand_sem

    print(f"Grand average accuracy: {grand_mean.mean():.3f} +/- {grand_std.mean():.3f}")
    print(f"Peak grand average: {grand_mean.max():.3f} at {times[grand_mean.argmax()]:.3f}s")

    # -------------------------------------------------------------------
    # OVERLAY PLOT
    # -------------------------------------------------------------------

    # Mean +/- SEM only -- no per-subject overlay lines. With cohorts this
    # size (up to 45 subjects), individual traces just add visual noise;
    # the grand average +/- SEM band is what's actually being reported on.
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    ax.plot(times, grand_mean, linewidth=3.5, color='black',
            label=f'Grand Average (n={n_subjects})', zorder=100)
    ax.fill_between(times, grand_mean - ci_95, grand_mean + ci_95,
                     alpha=0.3, color='gray', label='95% CI', zorder=99)
    ax.axhline(chance_level, color='red', linestyle='--', linewidth=2.5,
               label=f'Chance ({chance_level:.3f})', alpha=0.8, zorder=98)
    ax.axvline(0, color='black', linestyle='--', linewidth=2,
               alpha=0.6, label='Fixation (t=0)', zorder=97)

    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f"Temporal Decoding -- {config['title']}, n={n_subjects} subjects\n"
                 f"Independent AR pipeline, independently AutoRejected fixation stream",
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_ylim([0.2, 0.6])
    ax.set_xlim([times[0], times[-1]])

    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    # plot_path already computed at the top of this function (skip-guard check)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # -------------------------------------------------------------------
    # SAVE GROUP RESULTS
    # -------------------------------------------------------------------

    group_results = {
        'subject_ids': subject_ids,
        'decoded_label': all_results[0]['decoded_label'],
        'epoch_type': all_results[0]['epoch_type'],
        'pipeline': 'ar_v3_independent',
        'all_scores': all_scores,
        'times': times,
        'grand_mean': grand_mean,
        'grand_std': grand_std,
        'grand_sem': grand_sem,
        'ci_95': ci_95,
        'chance_level': chance_level,
        'n_subjects': n_subjects,
    }

    # group_results_path already computed at the top of this function (skip-guard check)
    with open(group_results_path, 'wb') as f:
        pickle.dump(group_results, f)
    print(f"Group results saved: {group_results_path}")

    # -------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------

    summary_data = []
    for i, subj_id in enumerate(subject_ids):
        summary_data.append({
            'Subject': subj_id,
            'N_Trials': all_results[i]['n_trials'],
            'Mean_Accuracy': f"{all_results[i]['mean_accuracy']:.3f}",
            'Peak_Accuracy': f"{all_results[i]['peak_accuracy']:.3f}",
            'Peak_Time': f"{all_results[i]['peak_time']:.3f}s",
        })
    print(pd.DataFrame(summary_data).to_string(index=False))
    print(f"\nGROUP STATISTICS ({config['key']}):")
    print(f"  Grand mean accuracy: {grand_mean.mean():.3f} +/- {grand_std.mean():.3f}")
    print(f"  Peak grand average:  {grand_mean.max():.3f} at {times[grand_mean.argmax()]:.3f}s")
    print(f"  Chance level:        {chance_level:.3f}\n")


for config in CONFIGS:
    compile_one(config)

print("=" * 70)
print(f"ALL {len(CONFIGS)} GROUP-LEVEL COMPILATIONS COMPLETE")
print("=" * 70)
