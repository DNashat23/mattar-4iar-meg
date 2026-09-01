# MEG Preprocessing Pipeline

## Overview

This pipeline preprocesses raw magnetoencephalography (MEG) data for PI Marcelo
Mattar and Dr. Jonathan Nicholas's Four-in-a-Row project (Research Assistant:
Dariush Nashat). It is implemented in Python using MNE, and tracks
Signal-to-Noise Ratio (SNR) at every stage to measure the sequential
improvement in signal quality as each preprocessing step runs.

The pipeline has two stages, run as separate scripts:

1. **`preprocessing_meg.py`** — loads raw MEG data, filters it, interpolates
   bad channels, removes environmental noise via an empty-room projection,
   and runs ICA to remove artifacts (eye blinks, heartbeats, muscle activity).
   Produces one cleaned, continuous raw `.fif` file per subject/task.
2. **`single_subject_ar_v3_independent.py`** (localizer) / **`single_subject_ar_v3_main.py`**
   (main task) — load that cleaned raw file, epoch it around task events, and
   run [AutoReject](https://autoreject.github.io/) to reject or repair bad
   trials/channels per epoch.

SNR tracking across both the filtering and interpolation/projection/ICA
stages is handled by **`snr_both.py`**, shared by all of the above.

---

## Directory Structure

Each subject needs the following layout under `base_dir`:

```
base_dir/
  R2611/                          # subject ID
    R2611_logfile.csv             # behavioral logfile
    raw/
      R2611_localizer1_..._anonymized.fif
      R2611_localizer2_..._anonymized.fif
      R2611_main1_..._anonymized.fif
      R2611_EmptyRoom_nr.sqd
    preprocessing/                # output directory — can start empty
```

Raw `.fif` files must already exist (created upstream via the KIT/Yokogawa
MEG system's export + an anonymization step) — this pipeline does not do
raw sensor-format conversion.

---

## Stage 1: `preprocessing_meg.py`

This script is meant to be run interactively (set `subj`/`dtypes` at the top,
then run cell-by-cell or step through in an interactive Python session) —
several steps pause for you to visually mark bad channels and review ICA
components.

### Configuration

```python
subj   = 'R2611'          # subject ID
dtypes = ['localizer']    # 'localizer', 'main', or both in one run
base_dir = '/path/to/data/'
empty_room_file = f'{base_dir}/{subj}/raw/{subj}_EmptyRoom_nr.sqd'
overwrite = True

onset_event_ids = {'localizer': 161, 'main': 163}   # response is always 164
```

### Steps

1. **Load raw data + baseline SNR.** All `.fif` files matching `*{dtype}*.fif`
   in `raw/` are loaded and concatenated. Baseline SNR is computed for both
   onset- and response-locked temporary epochs (see below).
2. **Frequency filtering.** A 1–40 Hz bandpass filter (`raw.filter(1, 40,
   method='iir')`) removes slow drift and high-frequency noise.
3. **Bad channel interpolation.** `raw.plot()` opens an interactive window —
   click channel names to mark them bad, close the window, and press Enter.
   Marked channels are saved to `{subj}_{dtype}_bad_channels.p` and then
   **interpolated** (`raw.interpolate_bads()`). Interpolated channels are
   treated as good for every step from here on — they are *not* excluded
   from the empty-room projection or the ICA fit that follow.
4. **Empty-room projection.** The empty-room recording is loaded, filtered
   identically, and has the *same* bad channels interpolated (empty-room
   recordings have no head digitization of their own, so this borrows the
   subject's `dig`/`dev_head_t` to make interpolation possible). SSP noise
   projectors are then computed from it and applied to the subject's data.
5. **ICA.** `mne.preprocessing.ICA(n_components=0.95, method='fastica')` is
   fit on the fully-interpolated data. `ica.plot_sources()` and
   `ica.plot_components()` let you review and mark artifact components
   interactively; marked components are removed with `ica.apply()`.

### Handling radio noise

While reviewing the data for bad channels in step 3 — or from a prior note
that a subject's recording is contaminated — if you notice radio noise,
**exit `preprocessing_meg.py` and run `remove_radio_noise.py` instead.** It
replaces `preprocessing_meg.py` entirely for that subject/task and produces
the same final raw output (see Outputs below) that AutoReject expects
downstream.

It follows the exact same pipeline (load → filter → interpolate bad channels
→ empty-room projection → ICA), with one extra step inserted right before bad
channel marking: `raw.plot(block=True)` opens with the plot ready for
annotation — press **`a`** to enter annotation mode, then click-and-drag
across each contaminated segment to mark it `BAD`. If there's no radio noise
after all, just close the window without marking anything and the script
proceeds using the raw data unchanged.

**Annotations must fully contain any trial they touch** — an annotation
can't cut a trial in half. Concretely: if an annotation's first captured
trigger is a response (meaning that trial's own onset trigger falls *before*
the annotation start) or its last captured trigger is an onset whose paired
response falls *after* the annotation end, the script exits with an error
listing exactly which timepoint to extend that annotation to. It checks
every annotation before exiting, not just the first bad one, so you get every
fix needed in one pass — restart the script and redraw all annotations with
the suggested corrections applied. Annotations with no trial triggers inside
them are left alone.

Once every annotation is valid, the contaminated segments are physically
cropped out and the remaining clean segments concatenated back together; the
logfile is filtered to match (`{subj}_{dtype}_nr_filtered_logfile.csv`), and
the pipeline continues into bad-channel marking as normal.

### Outputs (in `preprocessing/`)

The final cleaned raw file's name depends on `dtype`, matching what each
Stage 2 script expects (see below): **localizer** → `{subj}_localizer_ICA-raw_final.fif`,
**main** → `{subj}_main_ICA_no_ocular-raw.fif`.

| File | Contents |
|---|---|
| `{subj}_localizer_ICA-raw_final.fif` / `{subj}_main_ICA_no_ocular-raw.fif` | Final cleaned, continuous raw data |
| `{subj}_{dtype}_icasolution-ica.fif` | Fitted ICA solution |
| `{subj}_{dtype}_bad_channels.p` | List of interpolated channel names |
| `{subj}_{dtype}-info.p` | Final `raw.info`, pickled |
| `snr_analysis_master.csv` (in `base_dir`) | SNR at every stage, every subject |

If a file exceeds MNE's ~2GB single-file limit, it's split automatically into
the base filename above plus a `-1` suffixed continuation (e.g.
`..._ICA-raw_final.fif` + `..._ICA-raw_final-1.fif`). This is transparent —
loading the base filename with `mne.io.read_raw_fif()` reads the full
recording automatically as long as both files stay in the same folder.

---

## SNR Tracking (`snr_both.py`)

At every stage above (baseline, after filtering, after interpolation, after
empty-room projection, after ICA), temporary epochs are built and scored so
you can see how much each step helped.

**Onset epochs**: locked to the task's onset trigger, window `-0.1s` to
`1.0s`, scored over the `100–400ms` post-stimulus window.
**Response epochs**: locked to the response trigger (164), window `-1.0s` to
`0.1s`, scored over the `-400ms` to `-100ms` pre-response window.

For each channel/timepoint combination, SNR is `abs(signal_mean) /
standard_error` across trials. Two summary numbers are reported:

- **Raw SNR (100%)** — mean SNR across all channel/timepoint combinations.
- **Global SNR (top 20%)** — mean of just the strongest 20% of those values,
  a more robust signal-focused estimate.

This approach follows [Gonzalez-Moreno et al.,
2014](https://www.sciencedirect.com/science/article/abs/pii/S0165027013003713).

The `SNRTracker` class accumulates baseline/delta/percent-contribution
values for every subject into `snr_analysis_master.csv` in `base_dir`.

---

## Stage 2: Epoching + AutoReject

`single_subject_ar_v3_independent.py` and `single_subject_ar_v3_main.py` are
designed to be run non-interactively (e.g. from a SLURM job array):

```
python single_subject_ar_v3_independent.py --subject R2611 --base_dir /path/to/data
python single_subject_ar_v3_main.py        --subject R2611 --base_dir /path/to/data
```

Both find events, filter trials against the subject's logfile, build epochs,
and run `AutoReject` (grid search over `n_interpolate` and `consensus`,
Bayesian-optimized) to reject or repair individual trial/channel
combinations. `single_subject_ar_v3_main.py` loads
`{subj}_main_ICA_no_ocular-raw.fif`; `single_subject_ar_v3_independent.py`
loads `{subj}_localizer_ICA-raw_final.fif` — see Stage 1's Outputs section
for why the localizer and main raw files are named differently.

### Localizer (`single_subject_ar_v3_independent.py`)

- Keeps only **correct** trials with a response.
- Builds two epoch sets: **onset** (`161`, `-0.1s → 1.0s`) and **response**
  (`164`, `-1.0s → 0.1s`).
- **Onset and response are rejected fully independently** by AutoReject: a
  trial being rejected in one stream does *not* force the other stream's
  epoch for that trial to be dropped too, so the two saved epoch sets are
  generally different lengths. Each trial's independent survivability is
  recorded on the filtered logfile via `include_onset_epoch` /
  `include_response_epoch` (`yes`/`no`).
- Baseline correction is applied manually (not via `mne.Epochs`'s
  `baseline=` argument), and is anchored to the **prompt event (`160`)**,
  not board onset (`161`). The localizer trial sequence is fixation cross
  (`162`) → prompt (`160`) → board onset (`161`) → response (`164`), so a
  baseline window ending at `161` sits directly against the board-onset
  boundary — vulnerable to trigger jitter and to Stage 1's 1–40 Hz filter
  smearing the board-onset evoked response backward into it. The baseline is
  instead the mean of each trial's own `-101ms → -1ms` window relative to
  `160` — comfortably inside the fixation-cross display, before the prompt
  (let alone the board) ever appears — subtracted from *both* the onset and
  response epochs so they share one reference point. (`-101ms` rather than
  `-100ms`: MNE's epoch windows are inclusive on both ends, so a window
  ending at exactly `0` would include the sample at the `160` trigger
  itself.)

### Main task (`single_subject_ar_v3_main.py`)

- Keeps all trials with a response (no correctness filter — the main task
  isn't scored the same way as the localizer).
- Builds a single, long epoch locked to board onset (`163`, `-0.1s → 15.0s`).
  There's no separate response epoch — instead, each trial's response
  sample (event `164`, absolute MEG sample) is stored in the epoch/logfile
  metadata as `response_sample`, so response-locked analyses can re-slice
  this epoch downstream as needed.
- Baseline correction uses the same manual `-0.1s → 0s` pre-onset window.

### Outputs (in `preprocessing/`)

**Localizer** (`single_subject_ar_v3_independent.py`):

| File | Contents |
|---|---|
| `{subj}_localizer_ar_v3_independent_onset-epo.fif` | Cleaned onset-locked epochs |
| `{subj}_localizer_ar_v3_independent_offset-epo.fif` | Cleaned response-locked epochs |
| `{subj}_localizer_ar_v3_independent_filtered_logfile.csv` | Logfile rows kept, tagged with `include_onset_epoch` / `include_response_epoch` |
| `{subj}_localizer_ar_v3_independent_rejection_report.p` | Which trials were dropped from each stream, and why |
| `{subj}_localizer_ar_v3_independent_reject_log_onset.npz` / `_response.npz` | Per-epoch, per-channel AutoReject labels, one file per stream |
| `{subj}_localizer_ar_v3_independent_cv_loss.pkl` | AutoReject's consensus/n_interpolate grid search results, per stream |

**Main task** (`single_subject_ar_v3_main.py`):

| File | Contents |
|---|---|
| `{subj}_main_ar_v3_onset-epo.fif` | Cleaned main-task epochs |
| `{subj}_main_ar_v3_filtered_logfile.csv` | Logfile rows kept, incl. `response_sample` |
| `{subj}_main_ar_v3_rejection_report.p` | Which trials were dropped, and why |
| `{subj}_main_ar_v3_reject_log*.npz` | Per-epoch, per-channel AutoReject labels |
| `{subj}_main_ar_v3_cv_loss.pkl` | AutoReject's consensus/n_interpolate grid search results |

---

## References

Gonzalez-Moreno, A., Aurtenetxe, S., Lopez-Garcia, M.E., del Pozo, F.,
Maestu, F., & Nevado, A. (2014). [Signal-to-noise ratio of the MEG signal
after preprocessing](https://www.sciencedirect.com/science/article/abs/pii/S0165027013003713).
*Journal of Neuroscience Methods*, 235, 78-87.
