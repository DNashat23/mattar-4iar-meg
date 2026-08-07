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
2. **`single_subject_ar_v3.py`** (localizer) / **`single_subject_ar_v3_main.py`**
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

### Outputs (in `preprocessing/`)

| File | Contents |
|---|---|
| `{subj}_{dtype}_ICA_no_ocular-raw.fif` | Final cleaned, continuous raw data |
| `{subj}_{dtype}_icasolution-ica.fif` | Fitted ICA solution |
| `{subj}_{dtype}_bad_channels.p` | List of interpolated channel names |
| `{subj}_{dtype}-info.p` | Final `raw.info`, pickled |
| `snr_analysis_master.csv` (in `base_dir`) | SNR at every stage, every subject |

If a file exceeds MNE's ~2GB single-file limit, it's split automatically into
`..._ICA_no_ocular-raw.fif` + `..._ICA_no_ocular-raw-1.fif`. This is
transparent — loading the base filename with `mne.io.read_raw_fif()` reads
the full recording automatically as long as both files stay in the same
folder.

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

`single_subject_ar_v3.py` and `single_subject_ar_v3_main.py` are designed to
be run non-interactively (e.g. from a SLURM job array):

```
python single_subject_ar_v3.py      --subject R2611 --base_dir /path/to/data
python single_subject_ar_v3_main.py --subject R2611 --base_dir /path/to/data
```

Both load `{subj}_{dtype}_ICA_no_ocular-raw.fif`, find events, filter trials
against the subject's logfile, build epochs, and run `AutoReject` (grid
search over `n_interpolate` and `consensus`, Bayesian-optimized) to reject or
repair individual trial/channel combinations.

### Localizer (`single_subject_ar_v3.py`)

- Keeps only **correct** trials with a response.
- Builds two epoch sets: **onset** (`161`, `-0.1s → 1.0s`) and **response**
  (`164`, `-1.0s → 0.1s`), run through AutoReject independently.
- **Trial pairing is enforced**: if AutoReject rejects a trial in one epoch
  set but not the other, the corresponding trial is dropped from the other
  set too, so onset and response epochs always contain the same trials.
- Baseline correction is applied manually (not via `mne.Epochs`'s
  `baseline=` argument): the mean of each trial's own `-0.1s → 0s` window
  from the **onset** epoch is subtracted from *both* the onset and response
  epochs, so they share one reference point.

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

| File | Contents |
|---|---|
| `{subj}_localizer_ar_v3_onset-epo.fif` / `_offset-epo.fif` | Cleaned localizer epochs |
| `{subj}_main_ar_v3_onset-epo.fif` | Cleaned main-task epochs |
| `{subj}_{dtype}_ar_v3_filtered_logfile.csv` | Logfile rows kept, incl. `response_sample` |
| `{subj}_{dtype}_ar_v3_rejection_report.p` | Which trials were dropped, and why |
| `{subj}_{dtype}_ar_v3_reject_log*.npz` | Per-epoch, per-channel AutoReject labels |
| `{subj}_{dtype}_ar_v3_cv_loss.pkl` | AutoReject's consensus/n_interpolate grid search results |

---

## References

Gonzalez-Moreno, A., Aurtenetxe, S., Lopez-Garcia, M.E., del Pozo, F.,
Maestu, F., & Nevado, A. (2014). [Signal-to-noise ratio of the MEG signal
after preprocessing](https://www.sciencedirect.com/science/article/abs/pii/S0165027013003713).
*Journal of Neuroscience Methods*, 235, 78-87.
