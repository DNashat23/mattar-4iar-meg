"""
Group-level temporal decoding analysis — Prompt Color (black vs. white)
Independent onset/response rejection pipeline (single_subject_ar_v3_independent.py)

Same as compile_bw_decoding_onset.py, but points at the
*_ar_v3_independent_temporal_decoding_prompt_color_results.pkl files produced
from the independently-AutoRejected onset epochs, instead of the original
pegged-pipeline results. Kept as a separate script so the original's outputs
(group_temporal_decoding_prompt_color_overlay.png / _results.pkl) are never
overwritten.

This script:
1. Searches through all subject folders (RXXXX format)
2. Finds *_ar_v3_independent_temporal_decoding_prompt_color_results.pkl files
   in analysis folders
3. Loads all results and creates an overlay plot
4. Shows individual subject curves + grand average

Usage:
    python compile_bw_decoding_onset_independent.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import glob

# Configuration
base_dir = '/Users/dariushnashat/Documents/Mattar Lab Internship'

print("="*70)
print("GROUP-LEVEL TEMPORAL DECODING ANALYSIS — PROMPT COLOR (black vs. white)")
print("Independent onset/response rejection pipeline")
print("="*70)

# =============================================================================
# FIND ALL SUBJECT FOLDERS AND RESULT FILES
# =============================================================================

print("\nSearching for subject data...")

# Find all folders matching RXXXX pattern
subject_folders = glob.glob(os.path.join(base_dir, 'R*'))
subject_folders = [f for f in subject_folders if os.path.isdir(f)]

print(f"Found {len(subject_folders)} potential subject folders")

# Find all .pkl files in analysis folders
all_results = []
subject_ids = []
failed_subjects = []

for subj_folder in sorted(subject_folders):
    subj_id = os.path.basename(subj_folder)

    # Look for the independent-pipeline prompt_color pkl file in analysis folder
    pkl_path = os.path.join(subj_folder, 'analysis', f'{subj_id}_ar_v3_independent_temporal_decoding_prompt_color_results.pkl')

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)

            all_results.append(results)
            subject_ids.append(subj_id)
            print(f"  ✓ Loaded {subj_id}: {results['n_trials']} trials, peak={results['peak_accuracy']:.3f}")

        except Exception as e:
            failed_subjects.append((subj_id, str(e)))
            print(f"  ✗ Error loading {subj_id}: {e}")

print(f"\n{'='*70}")
print(f"Successfully loaded: {len(subject_ids)} subjects")
if failed_subjects:
    print(f"Failed to load: {len(failed_subjects)} subjects")
print(f"{'='*70}")

if len(subject_ids) == 0:
    print("\nERROR: No subject data found!")
    exit()

# =============================================================================
# ORGANIZE DATA FOR PLOTTING
# =============================================================================

print("\nOrganizing data...")

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

# Convert to numpy array: (n_subjects, n_timepoints)
all_scores = np.array(all_scores)
n_subjects = len(subject_ids)

print(f"Data shape: {all_scores.shape} (subjects × timepoints)")
print(f"Time window: {times[0]:.3f}s to {times[-1]:.3f}s")
print(f"Chance level: {chance_level:.3f}")

# =============================================================================
# CALCULATE GROUP STATISTICS
# =============================================================================

print("\nCalculating group statistics...")

grand_mean = all_scores.mean(axis=0)
grand_std = all_scores.std(axis=0)
grand_sem = stats.sem(all_scores, axis=0)
ci_95 = 1.96 * grand_sem

print(f"Grand average accuracy: {grand_mean.mean():.3f} ± {grand_std.mean():.3f}")
print(f"Peak grand average: {grand_mean.max():.3f} at {times[grand_mean.argmax()]:.3f}s")

# =============================================================================
# CREATE OVERLAY PLOT
# =============================================================================

print("\nCreating plot...")

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))

for i, subj_id in enumerate(subject_ids):
    ax.plot(times, all_scores[i], alpha=0.4, linewidth=1.5,
            color=colors[i], label=f'{subj_id}')

ax.plot(times, grand_mean, linewidth=3.5, color='black',
        label=f'Grand Average (n={n_subjects})', zorder=100)

ax.fill_between(times, grand_mean - ci_95, grand_mean + ci_95,
                alpha=0.3, color='gray', label='95% CI', zorder=99)

ax.axhline(chance_level, color='red', linestyle='--', linewidth=2.5,
          label=f'Chance ({chance_level:.3f})', alpha=0.8, zorder=98)
ax.axvline(0, color='black', linestyle='--', linewidth=2,
          alpha=0.6, label='Stimulus Onset', zorder=97)

ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
ax.set_title(f'Temporal Decoding — Prompt Color (black vs. white), n={n_subjects} subjects\n'
             f'Independent onset/response rejection pipeline',
             fontsize=16, fontweight='bold', pad=20)

ax.set_ylim([0.2, 0.8])
ax.set_xlim([times[0], times[-1]])

if n_subjects <= 10:
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
else:
    handles, labels = ax.get_legend_handles_labels()
    important_labels = ['Grand Average', '95% CI', 'Chance', 'Stimulus Onset']
    important_handles = [h for h, l in zip(handles, labels) if any(imp in l for imp in important_labels)]
    important_label_texts = [l for l in labels if any(imp in l for imp in important_labels)]
    ax.legend(important_handles, important_label_texts, loc='upper right', fontsize=11, framealpha=0.9)

ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()

output_path = os.path.join(base_dir, 'group_temporal_decoding_prompt_color_overlay_independent.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved: {output_path}")

# =============================================================================
# SAVE GROUP RESULTS
# =============================================================================

print("\nSaving group results...")

group_results = {
    'subject_ids': subject_ids,
    'decoded_label': 'prompt_color',
    'pipeline': 'ar_v3_independent',
    'all_scores': all_scores,  # (n_subjects, n_timepoints)
    'times': times,
    'grand_mean': grand_mean,
    'grand_std': grand_std,
    'grand_sem': grand_sem,
    'ci_95': ci_95,
    'chance_level': chance_level,
    'n_subjects': n_subjects
}

group_results_path = os.path.join(base_dir, 'group_temporal_decoding_prompt_color_results_independent.pkl')
with open(group_results_path, 'wb') as f:
    pickle.dump(group_results, f)

print(f"Group results saved: {group_results_path}")

# =============================================================================
# CREATE SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("SUBJECT SUMMARY TABLE")
print("="*70)

summary_data = []
for i, subj_id in enumerate(subject_ids):
    summary_data.append({
        'Subject': subj_id,
        'N_Trials': all_results[i]['n_trials'],
        'Mean_Accuracy': f"{all_results[i]['mean_accuracy']:.3f}",
        'Peak_Accuracy': f"{all_results[i]['peak_accuracy']:.3f}",
        'Peak_Time': f"{all_results[i]['peak_time']:.3f}s"
    })

df = pd.DataFrame(summary_data)
print(df.to_string(index=False))
print("="*70)

print("\nGROUP STATISTICS:")
print(f"  Grand mean accuracy: {grand_mean.mean():.3f} ± {grand_std.mean():.3f}")
print(f"  Peak grand average:  {grand_mean.max():.3f} at {times[grand_mean.argmax()]:.3f}s")
print(f"  Chance level:        {chance_level:.3f}")

print("\n" + "="*70)
print("GROUP-LEVEL ANALYSIS COMPLETE!")
print("="*70)
