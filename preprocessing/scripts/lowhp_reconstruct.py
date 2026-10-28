"""Rebuild the main-task (or localizer) cleaned continuous MEG with a LOWER high-pass (default 0.1 Hz instead of 1 Hz), reusing every
manual preprocessing decision without redoing it.

Why this works. After filtering, the preprocessing (preprocessing_meg.py / remove_radio_noise.py) only applies LINEAR,
time-invariant channel operations: bad-channel interpolation, empty-room SSP projection, ICA component removal
(ICA removal is affine: it subtracts the projection of (X - pca_mean_)). So the saved cleaned raw satisfies, sample by
sample on the magnetometers,
        cleaned = L @ filt_1-40(original) + c
for a fixed matrix L (n_mag x n_mag) and offset c. L and c are estimated by least squares from the original recording
re-filtered as in the pipeline (IIR 1-40 Hz, MNE defaults, over the whole concatenated recording) and the saved cleaned
raw, then applied to the original filtered HP-40 Hz:
        cleaned_lowhp = L @ filt_HP-40(original) + c
The fit must be essentially exact on held-out samples (R2 >= MIN_R2 for every channel); otherwise nothing is written.

Original recording = the KIT '_nr' runs of the task, <raw_root>/<MEG ID>/*<task><k>*_nr.sqd, in run order. Runs are chosen
automatically: the in-order subset whose total length equals the cleaned recording ('direct'). If no subset matches
(segments were cut out, as in remove_radio_noise.py), all runs are used and each cleaned segment (between 'boundary'
annotations) is located in the original by signal matching ('segments').
main (default): cleaned raw and logfile found under --base_dir as single_subject_ar_v3_main_rtpad.py does (Main/ or
preprocessing/); output <out_base>/<MEG ID>/preprocessing/<MEG ID>_main_ICA_no_ocular-raw.fif (+ logfile), the layout
single_subject_ar_v3_main_rtpad.py reads with --base_dir <out_base>.
--task localizer: cleaned raw <base_dir>/<MEG ID>/{preprocessing,Localizer}/<MEG ID>_localizer_ICA-raw_final.fif; output
<out_base>/<MEG ID>/preprocessing/<MEG ID>_localizer_ICA-raw_final.fif, with the logfiles and the fixations CSV copied
to <out_base>/<MEG ID>/, the layout single_subject_ar_v3_independent[_fixation].py read with --base_dir <out_base>.
Both write <out_base>/<MEG ID>/preprocessing/<MEG ID>_<task>_lowhp_report.json.

    python lowhp_reconstruct.py --subject R2312 --base_dir ... --raw_root ... --out_base ... [--task localizer]
                                [--hp 0.1] [--check_only]
"""
import argparse
import itertools
import json
import re
import shutil
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from single_subject_ar_v3_main_rtpad import cluster_paths, load_raw  # noqa: E402

mne.set_log_level("WARNING")
MIN_R2 = 0.999
MIN_SEG = 50             # boundary marks closer than this are merged
FIT_STEP = 25            # every 25th sample is used to estimate L; the samples in between are held out


def mags(raw):
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, meg="mag", ref_meg=False)]


def run_files(raw_root, mid, task):
    pat = {"main": "main", "localizer": "locali[sz]er"}[task] + r"\D*?(\d+)"   # main1, mainr1, localiser1, ...
    files = [f for f in (Path(raw_root) / mid).glob("*_nr.sqd") if re.search(pat, f.name)]
    files = sorted(files, key=lambda f: int(re.search(pat, f.name).group(1)))
    if not files:
        sys.exit(f"ERROR: no {task} *_nr.sqd runs for {mid} under {Path(raw_root) / mid}")
    return files


def localizer_paths(base_dir, mid):
    """cleaned localizer raw, and the files next to it that the localizer epoching scripts read (logfiles, fixations)"""
    d = Path(base_dir) / mid
    for sub_dir in ("preprocessing", "Localizer", ""):
        if (d / sub_dir / f"{mid}_localizer_ICA-raw_final.fif").exists():
            raw_path = str(d / sub_dir / f"{mid}_localizer_ICA-raw_final.fif")
            break
    else:
        sys.exit(f"ERROR: no {mid}_localizer_ICA-raw_final.fif under {d}")
    side = [p for p in [d / f"{mid}_logfile.csv", d / f"{mid}_localizer_logfile.csv"] if p.exists()]
    if not side:
        sys.exit(f"ERROR: no logfile for {mid} under {d}")
    return raw_path, side + sorted(d.glob(f"{mid}_*_localizer_fixations.csv"))


def choose_runs(files, n_clean):
    """in-order subset of runs whose total length equals the cleaned recording; else all runs"""
    lens = [mne.io.read_raw_kit(f, preload=False).n_times for f in files]
    for k in range(len(files), 0, -1):
        for combo in itertools.combinations(range(len(files)), k):
            if sum(lens[i] for i in combo) == n_clean:
                return [files[i] for i in combo], lens, "direct"
    return files, lens, "segments"


def locate(A, tmpl):
    """offset o maximizing the normalized cross-correlation of tmpl (ch x n) with A[:, o:o+n], averaged over
    channels; returns (o, score). Score ~1 for an exact match (cleaned ~ L @ A, L near identity per channel)."""
    C, n = tmpl.shape
    Lh = A.shape[1]
    nfft = 1 << int(np.ceil(np.log2(Lh + n)))
    score = np.zeros(Lh - n + 1)
    for ch in range(C):
        t = tmpl[ch] - tmpl[ch].mean()
        tn = np.sqrt((t ** 2).sum())
        if tn == 0:
            continue
        x = A[ch]
        num = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(t, nfft)), nfft)[:Lh - n + 1]
        c1 = np.concatenate([[0], np.cumsum(x)])
        c2 = np.concatenate([[0], np.cumsum(x * x)])
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        den = np.sqrt(np.maximum(s2 - s1 * s1 / n, 1e-300)) * tn
        score += num / den
    score /= C
    o = int(np.argmax(score))
    return o, float(score[o])


def segment_map(A, clean_data, clean, kind):
    """[(clean_start, clean_stop, orig_start)] in data-array samples, and the match score of each segment"""
    if kind == "direct":
        return [(0, clean.n_times, 0)], [1.0]
    sf = clean.info["sfreq"]
    bnd = sorted({int(round((o - clean.first_time) * sf)) for o, d in zip(clean.annotations.onset, clean.annotations.description)
                  if "boundary" in d.lower()})
    edges = [0]
    for b in bnd:                                   # MNE can leave two boundary marks a sample apart; merge marks
        if b - edges[-1] >= MIN_SEG and clean.n_times - b >= MIN_SEG:   # < MIN_SEG samples apart (no slivers)
            edges.append(b)
    edges.append(clean.n_times)
    segs, scores = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        n = min(5000, b - a)                                          # first 5 s of the segment, 20 channels
        o, sc = locate(A[:20], clean_data[:20, a:a + n])
        segs.append((a, b, o))
        scores.append(sc)
    return segs, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="MEG ID")
    ap.add_argument("--base_dir", required=True, help="layout with the existing cleaned main raw + logfile")
    ap.add_argument("--raw_root", required=True, help="folder with <MEG ID>/<...>_main<k>_..._nr.sqd")
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--task", choices=["main", "localizer"], default="main")
    ap.add_argument("--hp", type=float, default=0.1)
    ap.add_argument("--check_only", action="store_true", help="find files and runs, report, no filtering")
    a = ap.parse_args()
    mid = a.subject
    if a.task == "main":
        clean_path, log_path = cluster_paths(a.base_dir, mid)
        side, out_name = [log_path], f"{mid}_main_ICA_no_ocular-raw.fif"
    else:
        clean_path, side = localizer_paths(a.base_dir, mid)
        out_name = f"{mid}_localizer_ICA-raw_final.fif"
    clean, _ = load_raw(clean_path)
    files, lens, kind = choose_runs(run_files(a.raw_root, mid, a.task), clean.n_times)
    rep = {"subject": mid, "task": a.task, "hp": a.hp, "cleaned": clean_path, "cleaned_samples": int(clean.n_times),
           "runs_available": [f.name for f in run_files(a.raw_root, mid, a.task)], "run_samples": [int(x) for x in lens],
           "runs_used": [f.name for f in files], "alignment": kind}
    out_dir = Path(a.out_base) / mid / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.check_only:
        print(json.dumps(rep, indent=1))
        return
    orig = mne.concatenate_raws([mne.io.read_raw_kit(f, preload=True) for f in files]) if len(files) > 1 \
        else mne.io.read_raw_kit(files[0], preload=True)
    ch = mags(clean)
    missing = [c for c in ch if c not in orig.ch_names]
    if missing:
        sys.exit(f"ERROR: {mid}: channels missing from the original: {missing[:5]}")
    orig.pick(ch)
    A_raw = orig.copy().filter(1, 40, method="iir")                   # as preprocessing_meg.py
    A = A_raw._data
    clean.load_data()
    ci = [clean.ch_names.index(x) for x in ch]
    Y = clean._data
    segs, scores = segment_map(A, Y[ci], clean, kind)
    rep["segments"] = [list(map(int, s)) for s in segs]
    rep["segment_match"] = [round(x, 4) for x in scores]
    if min(scores) < 0.2:                                           # wrong positions score ~0.05
        (out_dir / f"{mid}_{a.task}_lowhp_report.json").write_text(json.dumps(rep, indent=1))
        sys.exit(f"ERROR: {mid}: could not locate every cleaned segment in the original (match scores "
                 f"{[round(x, 3) for x in scores]}); not written")
    SC = 1e15                                                       # fit in fT: tesla-scale values next to the
                                                                    # intercept would fall below lstsq's rank cutoff
    def fit(segs):
        half = FIT_STEP // 2
        Xa = np.concatenate([A[:, o:o + (b - s):FIT_STEP] for s, b, o in segs], 1)
        Ya = np.concatenate([Y[ci, s:b:FIT_STEP] for s, b, o in segs], 1)
        Xh = np.concatenate([A[:, o + half:o + (b - s):FIT_STEP] for s, b, o in segs], 1)
        Yh = np.concatenate([Y[ci, s + half:b:FIT_STEP] for s, b, o in segs], 1)
        n = min(Xh.shape[1], Yh.shape[1])
        Xh, Yh = Xh[:, :n], Yh[:, :n]
        Lc = np.linalg.lstsq(np.vstack([Xa * SC, np.ones(Xa.shape[1])]).T, (Ya * SC).T, rcond=None)[0].T
        L, c = Lc[:, :-1], Lc[:, -1:] / SC
        res = Yh - (L @ Xh + c)
        r2 = 1 - (res ** 2).sum(1) / ((Yh - Yh.mean(1, keepdims=True)) ** 2).sum(1)
        return L, c, r2

    L, c, r2 = fit(segs)
    if kind == "segments":
        # refine each segment to the exact sample: with the fitted map the error is ~0 at the true offset and large
        # even one sample off (the correlation-based location can be a sample or two out on smooth 1 kHz data)
        refined = []
        for s, b, o in segs:
            m = min(b - s, 20000)
            errs = {}
            for d in range(-5, 6):
                if o + d < 0 or o + d + m > A.shape[1]:
                    continue
                e = Y[ci, s:s + m] - (L @ A[:, o + d:o + d + m] + c)
                errs[d] = float((e ** 2).mean())
            refined.append((s, b, o + min(errs, key=errs.get)))
        rep["segment_shift"] = [int(r[2] - g[2]) for r, g in zip(refined, segs)]
        segs = refined
        rep["segments"] = [list(map(int, x)) for x in segs]
        L, c, r2 = fit(segs)
    del A_raw, A
    rep.update(r2_min=float(r2.min()), r2_median=float(np.median(r2)),
               worst_channel=ch[int(np.argmin(r2))], identity_like=float(np.median(np.diag(L))))
    print(f"{mid}: runs {[f.name for f in files]}; alignment {kind} ({len(segs)} segment(s)); "
          f"held-out R2 min {r2.min():.6f} median {np.median(r2):.6f}")
    if r2.min() < MIN_R2:
        (out_dir / f"{mid}_{a.task}_lowhp_report.json").write_text(json.dumps(rep, indent=1))
        sys.exit(f"ERROR: {mid}: linear reconstruction not exact (R2 min {r2.min():.4f}, {rep['worst_channel']}); "
                 f"not written")
    orig.filter(a.hp, 40, method="iir")                                # HP-40 Hz, in place
    B = orig._data
    for s, b, o in segs:
        for t in range(s, b, 200_000):
            e = min(b, t + 200_000)
            Y[ci, t:e] = L @ B[:, o + (t - s):o + (e - s)] + c
    with clean.info._unlock():
        clean.info["highpass"] = a.hp
    out = out_dir / out_name
    clean.save(out, overwrite=True, split_size="2GB")
    for f in side:
        shutil.copy(f, Path(a.out_base) / mid / Path(f).name)
    rep["output"] = str(out)
    (out_dir / f"{mid}_{a.task}_lowhp_report.json").write_text(json.dumps(rep, indent=1))
    print(f"{mid}: written {out}")


if __name__ == "__main__":
    main()
