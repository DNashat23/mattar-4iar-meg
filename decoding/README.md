# Feature Decoding Pipeline

## Overview

This pipeline runs temporal (per-timepoint) decoding of `prompt_feature` —
the pattern length (`2` / `2gap` / `3` / `4`) cued by the prompt on each
localizer trial — from fixation-locked MEG epochs, for PI Marcelo Mattar and
Dr. Jonathan Nicholas's Four-in-a-Row project (Research Assistant: Dariush
Nashat).

Input epochs come from `single_subject_ar_v3_independent_fixation.py` (see
`../../preprocessing/scripts/`), which independently AutoRejects three
fixation-timepoint streams per trial — **onset**, **median**, and **offset**
of each qualifying fixation — producing
`{subj}_localizer_ar_v3_independent_fixation_{onset,median,offset}-epo.fif`.
A single trial can contribute more than one fixation epoch to a stream (zero,
one, or many qualifying fixations per trial), so decoding treats fixation
epochs, not trials, as the unit of analysis, with trial-of-origin tracked
separately for cross-validation grouping (see below).

There are two decoders, run independently per subject:

1. **`decode_feature_fixation.py`** — 4-class (`2` / `2gap` / `3` / `4`).
2. **`decode_feature_fixation_3class.py`** — 3-class, merging `2` and `2gap`
   into a combined `2_or_2gap` class. `2gap` is the smallest class and `4` the
   largest (~2.7x apart at the trial level); merging brings the largest/smallest
   ratio down to ~1.25x, giving three comparably-sized classes and a chance
   level (`1/3`) that's actually representative of the data.

Both loop internally over all three fixation streams for the given subject,
skipping any stream whose output already exists.

A third script, **`compile_feature_fixation_decoding.py`**, aggregates every
subject's per-stream results into group-level plots.

---

## Prompt color (black/white) decoding

A separate, simpler decoder — **`decode_bw_single_subject.py`** — decodes
`prompt_color` (black vs. white, binary) from **trial-level** onset epochs
(`{subj}_{trial_type}_ar_v3_independent_onset-epo.fif`, from
`single_subject_ar_v3_independent.py`), cropped to `-0.1s → 0.75s`. Since
each trial contributes exactly one onset epoch here (no fixation-level
multiplicities), a plain `StratifiedKFold` (5-fold) is sufficient — there's
no sibling-epoch grouping concern like the fixation-based feature decoders
above. Designed for a SLURM job array (`slurm_decode_bw.sh`, 48-subject
independent-pipeline cohort); **`compile_bw_decoding_onset_independent.py`**
aggregates the per-subject results into a group-level overlay plot (subject
curves + grand average).

---

## Per-subject decoding steps

Both decoders run the same pipeline, once per fixation stream:

1. **Load epochs** (`mne.read_epochs`) and exclude trials where
   `ground_truth == 'none'` — those aren't genuine instances of "the subject
   is looking at pattern X."
2. **Fixation-window filter**: keep only fixations that occurred **1.0–2.0s
   before that trial's response** (`164`) event — not the final second before
   the response (possibly confounded with motor preparation/execution), and
   not more than 2s before it (possibly early exploratory scanning). Computed
   directly from `response_sample` (in `epochs.metadata`) minus the epoch's
   own anchor sample (`epochs.events[:, 0]`), divided by `sfreq`. This filter
   worsens class balance, so `LogisticRegression` below uses
   `class_weight='balanced'`.
3. **Minimum-data guards**: skip the stream if fewer than 20 epochs remain, if
   not all classes survive, or if fewer than 5 unique trials remain (too few
   for 5-fold grouped CV).
4. **10ms temporal averaging**: every 10 consecutive timepoints (1000Hz →
   100ms bins → 100 bins total, final leftover sample dropped) are averaged
   into one, done *before* scaling so the per-channel scaling statistics are
   computed on the smoothed signal rather than the raw per-millisecond noise.
5. **Grouped 5-fold cross-validation** (`StratifiedGroupKFold`, grouped by
   `block_trial`): because a single trial can contribute multiple fixation
   epochs that share both a label and correlated trial-level signal (drift,
   head position, subject state), grouping guarantees sibling epochs from the
   same trial never split across train/test — otherwise the classifier could
   exploit trial identity instead of genuine timepoint-locked encoding.
6. **Per-fold class-presence check**: `StratifiedGroupKFold` only makes a
   best-effort attempt at stratification — unlike plain `StratifiedKFold`, it
   does *not* guarantee every class survives in every fold's training set. If
   a class's epochs are concentrated in very few trials, grouping can push all
   of them into one fold, silently leaving that class unpredictable in every
   other fold (no error is raised anywhere in that chain — `.fit()` just
   infers `classes_` from whatever's present). This is checked explicitly in
   advance; the stream is skipped if any fold fails it.
7. **Per-fold scaling** (`mne.decoding.Scaler(scalings='mean')`, i.e.
   `StandardScaler` per channel): refit on each fold's training split only,
   never fit globally before CV — a single global fit would let each fold's
   held-out epochs leak into the per-channel mean/std used to normalize them.
8. **Classification**: `LogisticRegression(class_weight='balanced')` wrapped
   in `SlidingEstimator`, so each timepoint gets its own independently-trained
   classifier. Per-fold accuracy is scored and averaged across folds.
9. **Save** the per-timepoint accuracy curve (`scores`), the full per-fold
   score array (`scores_folds`, shape `(5, n_times)`), and summary statistics,
   as both a `.pkl` and a plot per subject/stream.

---

## Group-level compilation (`compile_feature_fixation_decoding.py`)

Reads every subject's per-stream `.pkl`, computes the grand mean accuracy
curve and a 95% CI band (`1.96 * SEM`) across subjects, and plots mean ± CI
only — no individual subject traces (with cohorts this size, individual
traces are visual noise, not signal).

Uses a fixed 45-subject cohort (the 48-subject independent-pipeline cohort
minus `R2652`, `R2667`, `R2760`, which have no `*_localizer_fixations.csv`) —
the same set the SLURM arrays below run on. Not every subject/stream survives
every config (the fixation-window filter in particular can eliminate a
subject's `2`/`2gap` epochs entirely, or just in the offset stream); missing
subjects are skipped and each plot's title reports however many of the 45
actually loaded.

`CONFIGS` holds one entry per methodology generation this pipeline has gone
through (pre-fix → grouped CV → + fixation window → + class-check/10ms
averaging/per-fold scaling), so older, superseded results stay available for
comparison rather than being overwritten. **The current, correct methodology
is the `_groupcv_fixwin_classcheck_avg10ms_noleak` suffix** — everything
without that full suffix predates one or more of the leakage/imbalance fixes
above and should not be treated as the reference result. As of this writing
that suffix has only been run for the 3-class decoder; the 4-class script has
the same pipeline but hasn't yet been resubmitted on the cluster with it.

Each config is skipped if its output plot and results `.pkl` already exist,
so re-running the script after adding a new config doesn't recompute (and
touch the mtime of) every earlier one.

---

## Running on the cluster

```
python decode_feature_fixation.py        --subject R2611 --base_dir /path/to/data
python decode_feature_fixation_3class.py --subject R2611 --base_dir /path/to/data
```

`slurm_decode_feature_fixation.sh` / `slurm_decode_feature_fixation_3class.sh`
are SLURM job arrays over the fixed 45-subject cohort (`--array=0-44%5`, 5
concurrent). Each array task runs one subject; the script internally loops
over all three fixation streams and skips any that already have output.

`compile_feature_fixation_decoding.py` takes no arguments — it reads
`base_dir` directly from a hardcoded path at the top of the script and should
be run locally (or wherever `base_dir` points) after transferring the
per-subject `analysis/*_results.pkl` files back from the cluster.

---

## Outputs (in `{subj}/analysis/`)

| File | Contents |
|---|---|
| `{subj}_temporal_decoding_prompt_feature_[3class_]fixation_{onset,median,offset}_groupcv_fixwin_classcheck_avg10ms_noleak_results.pkl` | Per-timepoint accuracy (`scores`), per-fold scores, times, chance level, and pipeline metadata for one subject/stream |
| `{subj}_temporal_decoding_prompt_feature_[3class_]fixation_{onset,median,offset}_groupcv_fixwin_classcheck_avg10ms_noleak.png` | Per-subject accuracy-over-time plot |

Filename suffix key (each marks one accumulated pipeline change, in order):

| Suffix | Meaning |
|---|---|
| `groupcv` | `StratifiedGroupKFold` grouped by trial-of-origin, instead of plain `StratifiedKFold` |
| `fixwin` | Fixation-window filter (1.0–2.0s before response) |
| `classcheck` | Explicit per-fold check that every class survives in every fold's training set |
| `avg10ms` | Every 10 consecutive timepoints averaged before scaling |
| `noleak` | Scaler refit per-fold (training split only), not fit globally before CV |

Group-level outputs (in `base_dir`, one per `CONFIGS` entry):

| File | Contents |
|---|---|
| `group_temporal_decoding_prompt_feature_[3class_]fixation_{epoch_type}[...]_results.pkl` | Grand mean, grand SEM/95% CI, all per-subject scores, subject IDs |
| `group_temporal_decoding_prompt_feature_[3class_]fixation_{epoch_type}[...].png` | Group-level accuracy-over-time plot (mean ± 95% CI) |
