"""
Single-Subject AutoReject v3 — Main Task, variable-length epochs (board onset -> response + 0.5 s)

Differences from single_subject_ar_v3_main.py (which is left unchanged):

1. Trial matching. Logged trials are matched to MEG triggers by RT fingerprint + timing
   (trial_matching.match_session) instead of by position, so subjects with trials missing from the
   recording, or a logged response without a response trigger, are labelled correctly.
2. Epoch window. Each responded trial with a confirmed response trigger becomes one epoch from
   -0.1 s before board onset to 0.5 s after the response. Windows are capped at 15 s after onset
   (truncated=True) and zero-padded to a fixed -0.1 .. 15 s (15.1 s). n_valid_samples gives the real
   length; mask the padding in analyses.
3. Photodiode. The board appears on screen ~60 ms after its trigger, so board onset is shifted
   +PHOTODIODE_OFFSET samples (time 0 = board actually visible). Response triggers are NOT shifted:
   they are sent on the first screen refresh after the button press (<= 1 frame), not delayed by the
   display. (The original script shifted both.)
4. AutoReject on continuous data. AR needs equal-length epochs, so it runs on consecutive
   CHUNK_S-second chunks of the continuous recording (the chunks overlapping any trial window), with
   the same settings as the localizer (n_interpolate / consensus grids, Bayesian optimisation,
   random_state 42). It is fitted on a random subset of up to N_FIT chunks and applied to all of them
   in batches (memory). Within a chunk, bad channels are interpolated or the chunk is rejected. Epochs
   are then assembled from the cleaned chunks; an epoch is dropped if any chunk it needs was rejected
   or dropped (BAD annotation, flat channels), or if its window contains a zero-duration
   'BAD boundary' (a join between recordings), which mne's reject_by_annotation does not catch. Note: a channel may be interpolated in one
   chunk and not the next, which can leave a small step at chunk edges.
5. Baseline. Mean over -0.1 .. 0 s inclusive (101 samples at 1 kHz), the same samples MNE's
   baseline=(-0.1, 0) uses. It is subtracted from the real samples only, so the padding stays exactly
   zero. (MNE's own baseline would also shift the padding.)

Usage (local layout; MNE + autoreject environment):
    python single_subject_ar_v3_main_rtpad.py --subject SUB54
    python single_subject_ar_v3_main_rtpad.py --subject SUB54 --skip_ar     # test assembly only
Cluster layout (as single_subject_ar_v3_main.py: <base_dir>/<MEG ID>/preprocessing/<MEG ID>_main_ICA_no_ocular-raw.fif
and <base_dir>/<MEG ID>/<MEG ID>_logfile.csv); the ID map is a 2-column CSV (MEG ID, Behavioral ID):
    python single_subject_ar_v3_main_rtpad.py --subject R2726 --ar_mode epochs \
        --base_dir /scratch/djn8068/meg_project/data --ids meg_ids.csv --out_root /scratch/$USER/meg_rtpad
    ... --check_only    load, match and report trial counts without AutoReject (input check)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trial_matching import RT_TOL, bad_times, match_session  # noqa: E402

mne.set_log_level("WARNING")

# ----------------------------------------------------------------------------- configuration
PHOTODIODE_OFFSET = 60   # samples, board onset only
TMIN = -0.1              # s before (shifted) board onset
POST = 0.5               # s kept after the response
MAX_AFTER_ONSET = 15.0   # s; windows are capped here
BASELINE = (-0.1, 0.0)   # inclusive, as MNE
FLAT_CRITERIA = dict(mag=1e-15)
CHUNK_S = 1.0
N_FIT = 600
BATCH = 400
AR_KW = dict(n_interpolate=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
             consensus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
             random_state=42, thresh_method="bayesian_optimization")
XS = [-305, -229, -153, -77, -1, 75, 151, 227, 303]   # task grid (4IAR_MEG_Task/task.py)
YS = [113, 37, -39, -115]

PROJ = Path("/Users/jonathannicholas/postdoc/mattarlab/projects/4IAR_MEG")
BOX = Path(os.path.expanduser("~/Library/CloudStorage/Box-Box/4IAR_MEG/Preprocessed"))


def resolve_ids(subject, ids=None):
    """Return (behavioral SUB id, MEG id) from either form, via an ID map (columns 'Behavioral ID',
    'MEG ID'; default: the local FINAL_subject_list.csv)."""
    sl = pd.read_csv(ids or PROJ / "full_data" / "FINAL_subject_list.csv", usecols=["Behavioral ID", "MEG ID"])
    if subject.upper().startswith("SUB"):
        row = sl[sl["Behavioral ID"] == subject.upper()]
    else:
        row = sl[sl["MEG ID"] == subject.upper()]
    if row.empty:
        sys.exit(f"ERROR: {subject} not in {ids or 'FINAL_subject_list.csv'}")
    return row["Behavioral ID"].iloc[0], row["MEG ID"].iloc[0]


def load_raw(p1, p2=None):
    """Continuous data; joins split parts even if part 2 was renamed after saving. Returns (raw,
    seamless_joins): a >2 GB recording saved as two files is one continuous recording, but
    concatenate_raws marks the join 'BAD boundary' / 'EDGE boundary'. When part 2 starts on the sample
    after part 1 ends, those marks are removed (checked on 33 subjects: contiguous, no signal step)."""
    try:
        return mne.io.read_raw_fif(p1, preload=False), []
    except ValueError:
        parts = [mne.io.read_raw_fif(p1, preload=False, on_split_missing="ignore")]
        p2 = p2 or str(p1)[:-4] + "-1.fif"     # X.fif -> X-1.fif (MNE's split naming)
        if os.path.exists(p2):
            parts.append(mne.io.read_raw_fif(p2, preload=False, on_split_missing="ignore"))
        else:   # part 1's header says the recording continues in another file
            sys.exit(f"ERROR: split part 2 not found ({p2}); pass it with --raw2")
        seamless = parts[1].first_samp == parts[0].last_samp + 1
        t_join = parts[0].n_times / parts[0].info["sfreq"]        # data time of part 2's first sample
        join_samp = parts[1].first_samp
        raw = mne.concatenate_raws(parts)                         # (extends parts[0] in place)
        joins = []
        if seamless:
            drop = [k for k, (o, d) in enumerate(zip(raw.annotations.onset - raw.first_time, raw.annotations.description))
                    if d in ("BAD boundary", "EDGE boundary") and abs(o - t_join) < 1.5 / raw.info["sfreq"]]
            raw.annotations.delete(drop)
            joins = [join_samp]
            print(f"  split parts are contiguous: removed {len(drop)} boundary marks at {t_join:.3f} s")
        return raw, joins


def cluster_paths(base_dir, mid):
    """Raw and logfile under the cluster layout; the raw may also carry the Box name."""
    d = Path(base_dir) / mid
    for name in (f"{mid}_main_ICA_no_ocular-raw.fif", f"{mid}_main_ICA-raw_final.fif"):
        for sub_dir in ("preprocessing", "Main", ""):
            if (d / sub_dir / name).exists():
                raw_path = str(d / sub_dir / name)
                break
        else:
            continue
        break
    else:
        sys.exit(f"ERROR: no main ICA raw for {mid} under {d}")
    logs = [d / f"{mid}_logfile.csv"] + sorted(d.glob("*_MAIN_logfile.csv"))
    logs = [p for p in logs if p.exists()]
    if not logs:
        sys.exit(f"ERROR: no logfile for {mid} under {d}")
    return raw_path, str(logs[0])


def bad_spans(raw):
    """[start, stop) data-array samples of BAD annotations. Zero-duration ones (the 'BAD boundary'
    marks where recordings were joined) become (b, b): reject_by_annotation ignores those, so they
    are checked explicitly."""
    sf, out = raw.info["sfreq"], []
    for a in raw.annotations:
        if a["description"].upper().startswith("BAD"):
            b = int(round((a["onset"] - raw.first_time) * sf))
            out.append((b, b + int(round(a["duration"] * sf))))
    return out


def crosses(spans, a, b):
    """True if data samples [a, b) touch a BAD span or contain a boundary (discontinuity before b0)."""
    return any((s0 < b and s1 > a) if s1 > s0 else (a < s0 < b) for s0, s1 in spans)


def ar_chunks(raw, trials, picks, spans, args, out_dir, tag, label):
    """AR on consecutive CHUNK_S chunks covering the trial windows; a trial is kept only if every chunk
    it needs survives. Returns ({trial position: cleaned segment}, excluded, info)."""
    sf, first = raw.info["sfreq"], raw.first_samp
    L = int(round(CHUNK_S * sf))
    need = sorted({k for t in trials for k in range((t["start"] - first) // L, (t["end"] - 1 - first) // L + 1)})
    chunk_ev = np.array([[first + k * L, 0, 1] for k in need])
    chunks = mne.Epochs(raw, chunk_ev, event_id={"chunk": 1}, tmin=0, tmax=(L - 1) / sf, baseline=None,
                        picks=picks, reject_by_annotation=True, flat=FLAT_CRITERIA, preload=True)
    split = [n for n, k in enumerate(need) if not chunks.drop_log[n] and crosses(spans, k * L, (k + 1) * L)]
    chunks.drop(np.searchsorted([n for n, d in enumerate(chunks.drop_log) if not d], split), reason="BAD boundary")
    kept_k = [need[n] for n, d in enumerate(chunks.drop_log) if not d]
    dropped_k = {need[n]: list(d) for n, d in enumerate(chunks.drop_log) if d}
    print(f"  chunks needed {len(need)} | dropped before AR (BAD boundary / flat) {len(dropped_k)}")

    clean, labels, bad_chunks = {}, {}, []
    if args.skip_ar:
        X = chunks.get_data()
        clean = {k: X[n] for n, k in enumerate(kept_k)}
    else:
        from autoreject import AutoReject
        rng = np.random.default_rng(42)
        fit_idx = np.sort(rng.choice(len(chunks), min(args.n_fit, len(chunks)), replace=False))
        ar = AutoReject(n_jobs=args.n_jobs, verbose=False, **AR_KW)
        print(f"  fitting AutoReject on {len(fit_idx)} of {len(chunks)} chunks ...")
        ar.fit(chunks[fit_idx])
        print(f"    learned n_interpolate {ar.n_interpolate_}, consensus {ar.consensus_}")
        for b0 in range(0, len(chunks), BATCH):
            sel = np.arange(b0, min(b0 + BATCH, len(chunks)))
            ep_clean, rlog = ar.transform(chunks[sel], return_log=True)
            Xc, n_good = ep_clean.get_data(), 0
            for n, s in enumerate(sel):
                k = kept_k[s]
                labels[k] = rlog.labels[n]
                if rlog.bad_epochs[n]:
                    bad_chunks.append(k)
                else:
                    clean[k] = Xc[n_good]
                    n_good += 1
        save_ar(ar, out_dir, tag, n_fit=len(fit_idx))
        np.savez(out_dir / f"{tag}_reject_log_chunks.npz", chunk_index=np.array(sorted(labels)),
                 labels=np.stack([labels[k] for k in sorted(labels)]), bad_chunks=np.array(sorted(bad_chunks)),
                 chunk_samples=L, first_samp=first)
        print(f"  AR rejected {len(bad_chunks)} of {len(chunks)} chunks")

    seg_of, excluded = {}, []
    for n, t in enumerate(trials):
        ks = range((t["start"] - first) // L, (t["end"] - 1 - first) // L + 1)
        if crosses(spans, t["start"] - first, t["end"] - first):
            excluded.append((*label(t), "BAD boundary/flat"))
            continue
        missing = [k for k in ks if k not in clean]
        if missing:
            excluded.append((*label(t), "BAD boundary/flat" if any(k in dropped_k for k in missing)
                             else "rejected by AutoReject"))
            continue
        seg = np.concatenate([clean[k] for k in ks], axis=1)
        off = (t["start"] - first) - ks[0] * L
        seg_of[n] = seg[:, off: off + (t["end"] - t["start"])]
    return seg_of, excluded, {"chunks_needed": len(need), "chunks_dropped_pre_ar": dropped_k,
                              "chunks_rejected_by_ar": bad_chunks, "labels": labels}


def ar_fixed_epochs(raw, trials, picks, spans, n_pre, n_total, args, out_dir, tag, label):
    """As single_subject_ar_v3_main.py: fixed epochs TMIN .. MAX_AFTER_ONSET around every (shifted)
    board onset, flat rejection, AutoReject fit_transform on all of them; each cleaned epoch is then
    cut at its own window end. Epochs MNE cannot make (BAD annotation with duration, end of recording,
    flat) and trials whose kept window contains a zero-duration BAD boundary are excluded."""
    sf, first = raw.info["sfreq"], raw.first_samp
    ev = np.array([[t["onset_sample"], 0, 163] for t in trials])
    ep = mne.Epochs(raw, ev, event_id={"board": 163}, tmin=TMIN, tmax=TMIN + (n_total - 1) / sf, baseline=None,
                    picks=picks, reject_by_annotation=True, flat=FLAT_CRITERIA, preload=True)
    kept = [n for n, d in enumerate(ep.drop_log) if not d]
    excluded = [(*label(trials[n]), "MNE drop: " + "/".join(d)) for n, d in enumerate(ep.drop_log) if d]
    print(f"  fixed epochs {len(trials)} | dropped before AR {len(excluded)} | shape {ep.get_data().shape}")
    info = {"n_epochs_ar": len(kept)}
    if args.skip_ar:
        X, good = ep.get_data(), list(range(len(kept)))
    elif args.reuse_ar_dir:
        # apply the AutoReject decisions of an earlier run (e.g. the 1-40 Hz data) instead of refitting: same
        # rejected epochs, same channels interpolated per epoch (lower high-pass data is dominated by slow drift,
        # which would distort AutoReject's peak-to-peak thresholds)
        src = Path(args.reuse_ar_dir) / out_dir.name / f"{tag}_reject_log.npz"
        z = np.load(src, allow_pickle=True)
        names = [ep.ch_names[i] for i in range(len(ep.ch_names))]
        if list(z["ch_names"]) != names or len(z["bad_epochs"]) != len(ep):
            sys.exit(f"ERROR: {src} does not match these epochs ({len(z['bad_epochs'])} vs {len(ep)} epochs, "
                     f"channels equal: {list(z['ch_names']) == names})")
        bad_ep, labels = z["bad_epochs"].astype(bool), z["labels"]
        groups = {}
        for p in np.flatnonzero(~bad_ep):
            groups.setdefault(tuple(np.flatnonzero(labels[p] == 2)), []).append(p)
        Xall = ep.get_data()
        for chs, idx in groups.items():
            if not chs:
                continue
            sub = ep[idx].copy()
            sub.info["bads"] = [names[c] for c in chs]
            sub.interpolate_bads(reset_bads=True)
            Xall[idx] = sub.get_data()
        X = Xall[~bad_ep]
        good = list(np.cumsum(~bad_ep) - 1)
        rlog = type("R", (), {"bad_epochs": bad_ep})()
        for p in np.flatnonzero(bad_ep):
            excluded.append((*label(trials[kept[p]]), "rejected by AutoReject (reused)"))
        info.update(rejected_by_ar=int(bad_ep.sum()), interpolated_per_epoch=(labels == 2).sum(1).tolist(),
                    reused_ar=str(src))
        print(f"  reused AutoReject decisions from {src}: {int(bad_ep.sum())} rejected, "
              f"{int((labels[~bad_ep] == 2).sum())} channel-epochs interpolated")
    else:
        from autoreject import AutoReject
        ar = AutoReject(n_jobs=args.n_jobs, verbose=False, **AR_KW)
        print(f"  AutoReject fit_transform on {len(ep)} epochs of {len(ep.times)} samples ...")
        ep_clean, rlog = ar.fit_transform(ep, return_log=True)
        print(f"    learned n_interpolate {ar.n_interpolate_}, consensus {ar.consensus_}")
        rlog.save(out_dir / f"{tag}_reject_log.npz", overwrite=True)
        save_ar(ar, out_dir, tag, n_fit=len(ep))
        X = ep_clean.get_data()
        good = list(np.cumsum(~rlog.bad_epochs) - 1)
        for p in np.flatnonzero(rlog.bad_epochs):
            excluded.append((*label(trials[kept[p]]), "rejected by AutoReject"))
        info.update(rejected_by_ar=int(rlog.bad_epochs.sum()),
                    interpolated_per_epoch=(rlog.labels == 2).sum(1).tolist())
        print(f"  AR rejected {int(rlog.bad_epochs.sum())} of {len(ep)} epochs")
    seg_of = {}
    for p, n in enumerate(kept):
        if not args.skip_ar and rlog.bad_epochs[p]:
            continue
        t = trials[n]
        if crosses(spans, t["start"] - first, t["end"] - first):
            excluded.append((*label(t), "BAD boundary/flat"))
            continue
        seg_of[n] = X[good[p], :, : t["end"] - t["start"]]
    return seg_of, excluded, info


def save_ar(ar, out_dir, tag, n_fit):
    ar.save(out_dir / f"{tag}_autoreject.hdf5", overwrite=True)   # fitted model: reuse with read_auto_reject
    with open(out_dir / f"{tag}_cv_loss.pkl", "wb") as f:
        pickle.dump({"loss": ar.loss_["mag"], "consensus": np.array(AR_KW["consensus"]),
                     "n_interpolate": np.array(AR_KW["n_interpolate"]),
                     "best_consensus": float(ar.consensus_["mag"]),
                     "best_n_interpolate": int(ar.n_interpolate_["mag"]), "n_fit": n_fit}, f)


def main():
    ap = argparse.ArgumentParser(description="AutoReject v3 — main task, board onset -> RT + 0.5 s")
    ap.add_argument("--subject", required=True, help="SUB## or R####")
    ap.add_argument("--raw", help="preprocessed main ICA raw (part 1); default: local Box copy")
    ap.add_argument("--raw2", help="split part 2, if renamed and not next to --raw")
    ap.add_argument("--logfile", help="behavioral logfile; default: full_data/behavior/<N>_MAIN_logfile.csv")
    ap.add_argument("--out_dir", help="default: behavior_modeling/data/meg_epochs_main_rtpad/<SUB>")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_fit", type=int, default=N_FIT)
    ap.add_argument("--ar_mode", choices=["chunks", "epochs"], default="chunks",
                    help="chunks: AR on 1 s chunks of the continuous data; epochs: AR on fixed -0.1..15 s "
                         "epochs as in single_subject_ar_v3_main.py, then cut at response + 0.5 s")
    ap.add_argument("--skip_ar", action="store_true", help="assemble epochs from uncleaned chunks (testing)")
    ap.add_argument("--base_dir", help="cluster layout root (<base_dir>/<MEG ID>/...); sets --raw/--logfile")
    ap.add_argument("--ids", help="CSV with 'Behavioral ID' and 'MEG ID' (default: local subject list)")
    ap.add_argument("--out_root", help="write to <out_root>/<SUB>/ (alternative to --out_dir)")
    ap.add_argument("--check_only", action="store_true", help="load and match, write a check row, no AR")
    ap.add_argument("--reuse_ar_dir", help="apply the AutoReject decisions saved by an earlier --ar_mode epochs run "
                                           "(<dir>/<SUB>/<tag>_reject_log.npz) instead of fitting AutoReject")
    ap.add_argument("--overwrite", action="store_true", help="redo a subject whose epochs already exist")
    ap.add_argument("--redo_split_joins", action="store_true",
                    help="redo an existing subject only if an AR epoch spans a seamless split-file join "
                         "(trials earlier lost to the spurious boundary mark); otherwise skip")
    args = ap.parse_args()

    sub, mid = resolve_ids(args.subject, args.ids)
    if args.base_dir:
        raw_path, log_path = cluster_paths(args.base_dir, mid)
    else:
        raw_path = str(BOX / mid / "Main" / f"{mid}_main_ICA-raw_final.fif")
        log_path = str(PROJ / "full_data" / "behavior" / f"{int(sub[3:])}_MAIN_logfile.csv")
    raw_path, log_path = args.raw or raw_path, args.logfile or log_path
    out_dir = Path(args.out_dir or (Path(args.out_root) / sub if args.out_root else
                                    PROJ / "behavior_modeling" / "data" / "meg_epochs_main_rtpad" / sub))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{mid}_main_ar_v3_rtpad" + ("_fixedAR" if args.ar_mode == "epochs" else "") + ("_noAR" if args.skip_ar else "")
    done = (out_dir / f"{tag}_onset-epo.fif").exists()
    if not args.check_only and not args.overwrite and done and not args.redo_split_joins:
        print(f"SKIPPING {sub}: {out_dir / (tag + '_onset-epo.fif')} exists (use --overwrite)")
        return
    print(f"\n{'=' * 60}\nPROCESSING (main, RT-padded): {sub} / {mid}\n{'=' * 60}\n  raw: {raw_path}\n  log: {log_path}")

    # --- data and triggers ---------------------------------------------------------------------
    raw, joins = load_raw(raw_path, args.raw2)
    sf = raw.info["sfreq"]
    assert sf == 1000.0, f"expected 1 kHz, got {sf}"
    ev = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)
    board = ev[ev[:, 2] == 163, 0]   # absolute samples, raw trigger times
    resp = ev[ev[:, 2] == 164, 0]
    to_s = lambda s: (s - raw.first_samp) / sf  # noqa: E731
    print(f"  board onsets (163): {len(board)}   responses (164): {len(resp)}")

    # --- trial matching -------------------------------------------------------------------------
    log = pd.read_csv(log_path)
    log = log[log.task == "main"].sort_values(["block", "trial"]).reset_index(drop=True)
    assign, delta, resp_idx = match_session(log, to_s(board), to_s(resp), bad_times(raw))
    responded = (log.rt.notna() & log.answer_x.notna() & log.answer_y.notna()).to_numpy()
    status = np.where(assign < 0, "not in MEG", np.where(~responded, "no response", ""))
    for i in np.flatnonzero(status == ""):
        j = assign[i]
        if not (np.isfinite(delta[j]) and abs(delta[j] - log.rt[i]) < RT_TOL):
            status[i] = "response trigger not confirmed"
    print(f"  logged trials {len(log)} | in MEG {int((assign >= 0).sum())} | responded & confirmed "
          f"{int((status == '').sum())}")
    if args.check_only:   # one row per subject, for comparing cluster inputs with the local audit
        row = {"sub": sub, "meg_id": mid, "raw": raw_path, "n_times": raw.n_times, "first_samp": raw.first_samp,
               "n_channels": len(raw.ch_names), "n_annotations": len(raw.annotations),
               "n_bad_annot": int(sum(d.upper().startswith("BAD") for d in raw.annotations.description)),
               "board_onsets": len(board), "responses": len(resp), "logged": len(log),
               "in_meg": int((assign >= 0).sum()), "confirmed": int((status == "").sum()),
               "first_board_sample": int(board[0]) if len(board) else -1,
               "missing": ";".join(f"b{int(b)}t{int(t)}" for b, t in zip(log.block[assign < 0], log.trial[assign < 0]))}
        pd.DataFrame([row]).to_csv(out_dir / f"{mid}_input_check.csv", index=False)
        print(json.dumps({k: v for k, v in row.items() if k != "missing"}, default=str))
        return

    # --- windows (absolute samples) -------------------------------------------------------------
    n_pre, n_post = int(round(-TMIN * sf)), int(round(POST * sf))
    n_cap = int(round(MAX_AFTER_ONSET * sf))
    n_total = n_pre + n_cap                       # -0.1 .. 15 s  -> 15100 samples
    trials = []
    for i in np.flatnonzero(status == ""):
        j = assign[i]
        onset = board[j] + PHOTODIODE_OFFSET
        r = resp[resp_idx[j]]
        end = r + n_post
        trials.append(dict(i=i, onset_sample=int(onset), response_sample=int(r), start=int(onset - n_pre),
                           end=int(min(end, onset + n_cap)), truncated=bool(end > onset + n_cap)))

    if done and not args.overwrite and not args.check_only:   # only reached with --redo_split_joins
        lo, hi = -n_pre, n_total - n_pre     # fixed AR epoch relative to onset
        hit = [t for t in trials for j in joins if t["onset_sample"] + lo < j < t["onset_sample"] + hi]
        if not hit:
            print(f"SKIPPING {sub}: no epoch spans a seamless split join; existing output kept")
            return
        print(f"  REDO: {len(hit)} trial(s) span the split join and were lost to the boundary mark before")
    first = raw.first_samp
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False)
    spans = bad_spans(raw)
    seg_of, excluded, ar_info = {}, [], {}   # trial position -> cleaned real-data segment (n_ch, n_valid)
    label = lambda t: (int(log.block[t["i"]]), int(log.trial[t["i"]]))  # noqa: E731
    if args.ar_mode == "epochs":
        seg_of, excluded, ar_info = ar_fixed_epochs(raw, trials, picks, spans, n_pre, n_total, args, out_dir, tag,
                                                    label)
    else:
        seg_of, excluded, ar_info = ar_chunks(raw, trials, picks, spans, args, out_dir, tag, label)

    # --- assemble epochs ------------------------------------------------------------------------
    data, meta, events = [], [], []
    bl = np.arange(n_pre + int(round(BASELINE[1] * sf)) + 1)   # samples -0.1 .. 0 inclusive
    for n, t in enumerate(trials):
        if n not in seg_of:
            continue
        seg = seg_of[n]
        seg = seg - seg[:, bl].mean(axis=1, keepdims=True)
        x = np.zeros((len(picks), n_total), dtype=np.float32)
        x[:, : seg.shape[1]] = seg
        r = log.iloc[t["i"]]
        meta.append({"block": int(r.block), "trial": int(r.trial), "board_shown": r.board_shown,
                     "color": r.prompt_color, "move": YS.index(int(r.answer_y)) * 9 + XS.index(int(r.answer_x)),
                     "rt_log": float(r.rt), "rt_trigger": (t["response_sample"] - (t["onset_sample"] - PHOTODIODE_OFFSET)) / sf,
                     "rt_from_display": (t["response_sample"] - t["onset_sample"]) / sf,
                     "onset_sample": t["onset_sample"], "response_sample": t["response_sample"],
                     "n_valid_samples": int(seg.shape[1]), "valid_end_s": TMIN + seg.shape[1] / sf,
                     "truncated": t["truncated"]})
        data.append(x)
        events.append([t["onset_sample"], 0, 163])

    md = pd.DataFrame(meta)
    info = mne.pick_info(raw.info, picks)
    epochs = mne.EpochsArray(np.stack(data), info, events=np.array(events), tmin=TMIN,
                             event_id={"board": 163}, metadata=md, baseline=None)
    epochs.save(out_dir / f"{tag}_onset-epo.fif", fmt="single", overwrite=True)
    md.to_csv(out_dir / f"{tag}_metadata.csv", index=False)

    report = {"subject": sub, "meg_id": mid, "logged_trials": len(log),
              "status_counts": pd.Series(status).replace("", "epoch candidate").value_counts().to_dict(),
              "candidates": len(trials), "final_trial_count": len(epochs),
              "excluded": excluded, "not_epoched": [(int(log.block[i]), int(log.trial[i]), status[i])
                                                   for i in np.flatnonzero(status != "")],
              "ar_mode": args.ar_mode, "ar": {k: v for k, v in ar_info.items() if k != "labels"},
              "skip_ar": args.skip_ar,
              "settings": dict(PHOTODIODE_OFFSET=PHOTODIODE_OFFSET, TMIN=TMIN, POST=POST,
                               MAX_AFTER_ONSET=MAX_AFTER_ONSET, BASELINE=BASELINE, CHUNK_S=CHUNK_S,
                               n_fit=args.n_fit, AR=AR_KW)}
    with open(out_dir / f"{tag}_rejection_report.p", "wb") as f:
        pickle.dump(report, f)
    print(json.dumps({k: report[k] for k in ("status_counts", "candidates", "final_trial_count")}, indent=1))
    print(f"  excluded at assembly: {len(excluded)} ({pd.Series([e[2] for e in excluded]).value_counts().to_dict()})")
    print(f"  truncated at {MAX_AFTER_ONSET:.0f} s: {int(md.truncated.sum())} | valid length median "
          f"{md.valid_end_s.median():.2f} s\n  saved to {out_dir}\n{'=' * 60}\nCOMPLETE: {sub}\n{'=' * 60}")


if __name__ == "__main__":
    main()
