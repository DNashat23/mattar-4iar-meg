"""Order-preserving matching of logged main-task trials to MEG board-onset triggers.

Positional matching (i-th trigger = i-th logged trial) breaks when any trial is missing from the
MEG recording or a logged response has no response trigger. This matcher identifies trials instead:

1. RT fingerprints: for each board onset (163), delta = first response trigger (164) after it and
   before the next onset (within 20 s), minus the onset. A logged trial whose RT equals delta within
   RT_TOL is a candidate pair. RTs are logged to the ms, so coincidences are rare.
2. Anchors: the longest chain of candidate pairs increasing in both log order and MEG order, so
   assignments never cross. Consecutive anchors in the same block must also agree on onset spacing:
   the MEG gap equals the logged gap within TOL (the two clocks drift < 15 ms per block), unless a
   BAD boundary lies between them (data were removed there), in which case the MEG gap may only be
   shorter. Without this, short RTs that coincide within RT_TOL can win (SUB34 localizer: b4t1 and
   b4t6 were taken for the onsets that really are b4t13 and b4t14). Ties go to the smaller summed
   RT error.
3. Fill: trials without a fingerprint (e.g. timeouts) are placed from their logged onset time
   relative to an already matched neighbour in the same block (tolerance TOL, no boundary in
   between), repeated until nothing changes.

Validated on 46 subjects' main task and the downloaded localizer epochs. Times are in seconds on
one common MEG clock (data time, i.e. from the first sample); raw trigger times (no photodiode
shift) must be used, since the shift is applied afterwards.
"""
import numpy as np

TOL = 0.05      # s, onset-time agreement for fill
RT_TOL = 0.03   # s, response trigger - onset vs logged RT


def bad_times(raw):
    """Data-time onsets (s) of BAD annotations, e.g. the 'BAD boundary' marks at recording joins."""
    return np.array([o - raw.first_time for o, d in zip(raw.annotations.onset, raw.annotations.description)
                     if d.upper().startswith("BAD")])


def match_session(log, board_t, resp_t, boundaries=()):
    """log: DataFrame of rows sorted by (block, trial) with columns block, onset_time, rt, answer_x
    (answer_x non-null marks a response). board_t / resp_t: sorted MEG times (s) of onset / response
    triggers. boundaries: MEG times (s) of BAD annotations (bad_times(raw)).
    Returns (assign, delta, resp_idx):
      assign[i]   index into board_t matched to log row i, or -1 if the trial is not in the MEG
      delta[j]    response trigger - onset for onset j (NaN if none)
      resp_idx[j] index into resp_t of that response trigger (-1 if none)"""
    n_log, n_meg = len(log), len(board_t)
    bnd = np.sort(np.asarray(boundaries, float))
    cut = lambda t0, t1: np.searchsorted(bnd, t1) > np.searchsorted(bnd, t0, side="right")  # noqa: E731
    delta, resp_idx = np.full(n_meg, np.nan), np.full(n_meg, -1)
    for j in range(n_meg):
        nxt = board_t[j + 1] if j + 1 < n_meg else np.inf
        k = np.flatnonzero((resp_t > board_t[j]) & (resp_t < min(nxt, board_t[j] + 20)))
        if len(k):
            resp_idx[j], delta[j] = k[0], resp_t[k[0]] - board_t[j]

    rt = log.rt.to_numpy(float)
    t_log, blk = log.onset_time.to_numpy(float), log.block.to_numpy()
    has_resp = log.answer_x.notna().to_numpy() & np.isfinite(rt)
    pairs = sorted((i, j) for j in range(n_meg) if np.isfinite(delta[j])
                   for i in np.flatnonzero(has_resp & (np.abs(rt - delta[j]) < RT_TOL)))
    err = [abs(rt[i] - delta[j]) for i, j in pairs]

    def compatible(b, a):
        (ib, jb), (ia, ja) = pairs[b], pairs[a]
        if not (ib < ia and jb < ja):
            return False
        if blk[ib] != blk[ia]:
            return True
        d_meg, d_log = board_t[ja] - board_t[jb], t_log[ia] - t_log[ib]
        return d_meg < d_log + TOL if cut(board_t[jb], board_t[ja]) else abs(d_meg - d_log) < TOL

    best = [(1, -e) for e in err]
    prev = [-1] * len(pairs)
    for a in range(len(pairs)):
        for b in range(a):
            cand = (best[b][0] + 1, best[b][1] - err[a])
            if cand > best[a] and compatible(b, a):
                best[a], prev[a] = cand, b
    assign = np.full(n_log, -1)
    if pairs:
        k = max(range(len(pairs)), key=lambda q: best[q])
        while k >= 0:
            assign[pairs[k][0]] = pairs[k][1]
            k = prev[k]

    used = np.zeros(n_meg, bool)
    used[assign[assign >= 0]] = True
    changed = True
    while changed:
        changed = False
        for i in np.flatnonzero(assign < 0):
            before = [k for k in range(i - 1, -1, -1) if assign[k] >= 0][:1]
            after = [k for k in range(i + 1, n_log) if assign[k] >= 0][:1]
            lo = assign[before[0]] if before else -1
            hi = assign[after[0]] if after else n_meg
            for ia in before + after:
                if blk[ia] != blk[i]:
                    continue
                ref = board_t[assign[ia]]
                pred = ref + (t_log[i] - t_log[ia])
                cand = [j for j in range(lo + 1, hi) if not used[j] and abs(board_t[j] - pred) < TOL
                        and not cut(min(ref, board_t[j]), max(ref, board_t[j]))]
                if cand:
                    j = min(cand, key=lambda j: abs(board_t[j] - pred))
                    assign[i], used[j], changed = j, True, True
                    break
    return assign, delta, resp_idx
