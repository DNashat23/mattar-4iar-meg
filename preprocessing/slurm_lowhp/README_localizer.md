# Localizer at 0.1 Hz high-pass (instead of 1 Hz), on Torch

The localizer was cleaned at 1–40 Hz (`preprocessing_meg.py` / `remove_radio_noise.py`): the 1 Hz high-pass ran
*before* bad-channel interpolation, empty-room projection and ICA, so everything slower than ~1 s is gone from every
file after it. We want the localizer decoding to use the same 0.1–40 Hz data as the main-task analyses.

Nothing manual has to be redone. Every step after the filter is a fixed linear map of the channels, so
`scripts/lowhp_reconstruct.py --task localizer`:
1. estimates that map exactly from your existing cleaned file and the original `_nr.sqd` recordings, and
2. applies it to the originals filtered at 0.1–40 Hz.

You get the same bad channels and the same ICA components, at 0.1–40 Hz. Your epoching scripts then run as before,
with one new option, `--reuse_ar_dir`. Instead of fitting AutoReject again, it applies the decisions your 1–40 Hz run
saved: the same epochs are dropped, and the same channels are interpolated in each epoch, using AutoReject's own
interpolation. There are two reasons for this:
- slow drift would distort AutoReject's peak-to-peak thresholds;
- every stream keeps exactly the epochs of the 1 Hz run, so 1 Hz and 0.1 Hz decoding are directly comparable.

Without `--reuse_ar_dir` the scripts behave exactly as before.

**Tested on R2312.** The reconstruction matches the cleaned file on held-out samples (R² = 1.000000 on every channel,
about 1 minute). On the 1 Hz data, reusing the saved AutoReject decisions reproduces a fresh AutoReject fit exactly
(same 211 / 222 / 227 onset / median / offset epochs, identical data), in ~2 minutes instead of ~35. The 0.1 Hz
epochs are the same epochs; their data correlate 0.47 with the 1 Hz data and have 2.2× the s.d. (the slow components).

## What goes where on Torch
| path | what | status |
|---|---|---|
| `/scratch/djn8068/meg_project/data/<subj>/preprocessing/<subj>_localizer_ICA-raw_final.fif` | cleaned 1–40 Hz localizer | already there |
| `/scratch/djn8068/meg_project/data/<subj>/preprocessing/*_reject_log_*.npz` | 1–40 Hz AutoReject decisions | already there |
| `/scratch/djn8068/meg_project/data/<subj>/<subj>_logfile.csv`, `<subj>_*_localizer_fixations.csv` | logfile, fixations | already there |
| `/scratch/djn8068/meg_project/sqd/<subj>/*localizer*_nr.sqd` | original recordings: 98 files, 19.3 GB | **upload (step 2)** |
| `/scratch/djn8068/meg_project/scripts/` | 5 scripts (new or updated) | **upload (step 1)** |
| `/scratch/djn8068/meg_project/data_lowhp/` | new data, epochs and decoding results | created by the jobs |

`data/` is only read, never written. Everything new goes to `data_lowhp/`, in the same layout as `data/`.

## 1. Get the code (on your Mac)
```bash
cd <your clone of mattar-4iar-meg>
git fetch origin
git checkout lowhp-localizer
git diff main -- preprocessing/scripts/single_subject_ar_v3_independent.py preprocessing/scripts/single_subject_ar_v3_independent_fixation.py
```
The diff should show only the new `--reuse_ar_dir` option. If your copies of these two scripts on Torch have edits that
aren't in git, merge them before step 1b, because the copy below overwrites them.
```bash
# 1b. copy the scripts to Torch
rsync -av preprocessing/scripts/lowhp_reconstruct.py preprocessing/scripts/reuse_autoreject.py \
  preprocessing/scripts/single_subject_ar_v3_main_rtpad.py preprocessing/scripts/trial_matching.py \
  preprocessing/scripts/single_subject_ar_v3_independent.py preprocessing/scripts/single_subject_ar_v3_independent_fixation.py \
  preprocessing/slurm_lowhp/slurm_lowhp_localizer.sh preprocessing/slurm_lowhp/verify_sqd.py \
  preprocessing/slurm_lowhp/sqd_manifest_localizer.csv \
  djn8068@dtn.torch.hpc.nyu.edu:/scratch/djn8068/meg_project/scripts/
```
(`single_subject_ar_v3_main_rtpad.py` and `trial_matching.py` are only imported, for the file-loading helpers.)

## 2. Upload the `_nr` localizer recordings from Box (on your Mac; 19.3 GB)
```bash
cd <your clone of mattar-4iar-meg>/preprocessing/slurm_lowhp
for m in $(tail -n +2 sqd_manifest_localizer.csv | cut -d, -f1 | sort -u); do
  rsync -av --include='*locali[sz]er*_nr.sqd' --exclude='*' \
    "$HOME/Library/CloudStorage/Box-Box/4IAR_MEG/$m/" djn8068@dtn.torch.hpc.nyu.edu:/scratch/djn8068/meg_project/sqd/$m/
done
```
- Only the `_nr` files are needed: the noise-reduced recordings the cleaned files were made from.
- R2652's files are spelled `localiser`; the pattern covers both spellings.
- Box Drive downloads each file as rsync reads it. If it stalls on a file, download that file from the Box website, or
  right-click the subject folder in Finder → *Make Available Offline*, then rerun the loop. rsync skips files that are
  already complete.

## 3. Check the upload and the inputs (on Torch)
```bash
# ssh into Torch as usual, then:
cd /scratch/djn8068/meg_project/scripts
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && source activate /scratch/djn8068/conda_envs/meg_env
python verify_sqd.py /scratch/djn8068/meg_project/sqd sqd_manifest_localizer.csv   # expect: 98 files expected, 98 OK
python -c "import autoreject, mne; print(autoreject.__version__, mne.__version__)"   # autoreject must be >= 0.4
sbatch --export=ALL,STEP=check -J lowhp_check slurm_lowhp_localizer.sh             # cheap, ~1 min per subject
```
When the check jobs finish:
```bash
cd /scratch/djn8068/meg_project/logs
grep -h '"alignment"' lowhp_check_*.out | sort | uniq -c   # expect 48 lines in total: "direct" (or "segments")
grep -L '"alignment"' lowhp_check_*.out                    # logs WITHOUT a result = problems; open them
```
- **direct**: the recording was used whole.
- **segments**: radio-noise segments were cut out by `remove_radio_noise.py`; each cleaned segment is then found in the
  original by signal matching.
- R2671 and R2728 have three localizer runs. The runs whose total length equals the cleaned file are chosen
  automatically (`runs_used` in the log).

## 4. Test on one subject (R2312 = array index 0)
```bash
cd /scratch/djn8068/meg_project/scripts
sbatch --array=0 --export=ALL,STEP=reconstruct -J lowhp_reconstruct slurm_lowhp_localizer.sh
# when it finishes (~5 min):
grep "R2" ../logs/lowhp_reconstruct_*_0.out     # expect "held-out R2 min 1.000000 ..." and "written"
sbatch --array=0 --export=ALL,STEP=fixation -J lowhp_fixation slurm_lowhp_localizer.sh
sbatch --array=0 --export=ALL,STEP=onset    -J lowhp_onset    slurm_lowhp_localizer.sh
# when they finish (~5 min):
grep -h "Reused AutoReject\|COMPLETE\|ERROR" ../logs/lowhp_{fixation,onset}_*_0.out
```
Then confirm the new epochs are the same epochs as the 1 Hz ones, with different data:
```bash
python - <<'EOF'
import mne, numpy as np
D = "/scratch/djn8068/meg_project"
for f in ["fixation_onset", "fixation_median", "fixation_offset", "onset", "offset"]:
    a = mne.read_epochs(f"{D}/data/R2312/preprocessing/R2312_localizer_ar_v3_independent_{f}-epo.fif", verbose=False)
    b = mne.read_epochs(f"{D}/data_lowhp/R2312/preprocessing/R2312_localizer_ar_v3_independent_{f}-epo.fif", verbose=False)
    same = len(a) == len(b) and (a.events[:, 0] == b.events[:, 0]).all()
    print(f, len(a), len(b), "same epochs" if same else "DIFFERENT EPOCHS",
          "| corr 1 Hz vs 0.1 Hz %.2f" % np.corrcoef(a.get_data().ravel(), b.get_data().ravel())[0, 1])
EOF
```
- **Expected:** "same epochs" for all five, and a correlation of about 0.5 (R2312 locally: 0.47), because the
  0.1 Hz data add slow components.
- **If a job exits with "does not match these epochs":** the 1 Hz reject log in `data/` was made from a different
  cleaned file or logfile than the one in `data/` now. Tell me which subject.

## 5. Run everyone
```bash
sbatch --export=ALL,STEP=reconstruct -J lowhp_reconstruct slurm_lowhp_localizer.sh
# after ALL reconstruct jobs finish:
grep -l "ERROR" ../logs/lowhp_reconstruct_*.out           # expect nothing
ls /scratch/djn8068/meg_project/data_lowhp/*/preprocessing/*_localizer_ICA-raw_final.fif | wc -l   # expect 48
sbatch --export=ALL,STEP=fixation -J lowhp_fixation slurm_lowhp_localizer.sh   # R2652, R2667, R2760 skip (no fixations CSV)
sbatch --export=ALL,STEP=onset    -J lowhp_onset    slurm_lowhp_localizer.sh
```
- A subject is not written if the reconstruction is not exact (R² < 0.999 on held-out samples). Its
  `data_lowhp/<subj>/preprocessing/<subj>_localizer_lowhp_report.json` says why. Send it to me.
- Resubmitting a step is safe. `fixation` and `onset` skip subjects that already finished (the scripts' cv_loss
  skip-guard); `reconstruct` simply redoes them.

## 6. Decode
Run your decoding scripts unchanged, with `--base_dir /scratch/djn8068/meg_project/data_lowhp`, e.g. in
`slurm_decode_feature_fixation.sh` set `DATA_DIR=${PROJECT_DIR}/data_lowhp`. Results go to
`data_lowhp/<subj>/analysis/`, so the 1 Hz results in `data/` are untouched. To compile locally, rsync
`data_lowhp/*/analysis/` into a separate local folder and point `compile_feature_fixation_decoding.py`'s `base_dir`
at it.

## Things to know about 0.1 Hz data
- **Slow drift.** Below 1 Hz the data contain large slow drift. Epochs are still baseline-corrected on −101 to −1 ms
  before the prompt (160). Fixation epochs come 1–2 s or more after that baseline, so each epoch carries its own drift
  offset. This adds trial-to-trial variance; it is not a confound for `prompt_feature`, which is interleaved within
  blocks.
- **Black/white is blocked.** In each localizer block every prompt has the same colour (e.g. R2312: blocks 1 and 4
  white, 2 and 3 black). A slow signal that differs between blocks therefore looks like colour information. This is
  true at 1 Hz too, but much more so at 0.1 Hz. With shuffled 5-fold CV, black/white decoding of 0.1 Hz data is not a
  valid sanity check. Use folds that leave out whole blocks: test on one black and one white block, train on the other
  two.
