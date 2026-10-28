"""Apply the AutoReject decisions saved by an earlier run (a RejectLog .npz) to new epochs of the same trials, instead
of fitting AutoReject again: the same epochs are dropped and the same channels are interpolated in each epoch, with
AutoReject's own interpolation, so the result equals what AutoReject.transform would have produced.

Used for the 0.1 Hz high-pass rebuild (lowhp_reconstruct.py): data below 1 Hz are dominated by slow drift, which would
distort AutoReject's peak-to-peak thresholds, and reusing the 1-40 Hz decisions keeps the trial set identical.
"""
import os
import sys

import mne
import numpy as np
from autoreject import read_reject_log
from autoreject.autoreject import _get_interp_chs, _interpolate_bad_epochs
from autoreject.utils import _compute_dots


def apply_reject_log(epochs, log_path):
    """(cleaned epochs, reject log); exits if the saved log does not belong to these epochs"""
    if not os.path.exists(log_path):
        sys.exit(f"ERROR: no saved AutoReject log at {log_path}")
    log = read_reject_log(log_path)
    if list(log.ch_names) != list(epochs.ch_names) or len(log.bad_epochs) != len(epochs):
        sys.exit(f"ERROR: {log_path} does not match these epochs ({len(log.bad_epochs)} vs {len(epochs)} epochs, "
                 f"channels equal: {list(log.ch_names) == list(epochs.ch_names)})")
    picks = np.flatnonzero(np.any(~np.isnan(log.labels), axis=0))
    ep = epochs.copy()
    # MEG interpolation geometry, computed once as AutoReject.fit does (otherwise it is recomputed for every set of
    # bad channels, which takes hours)
    dots = _compute_dots(mne.pick_info(ep.info, mne.pick_types(ep.info, meg=True, eeg=False, exclude=[]), copy=True),
                         templates=None)
    _interpolate_bad_epochs(ep, interp_channels=_get_interp_chs(log.labels, log.ch_names, picks), picks=picks,
                            dots=dots, verbose=False)
    if log.bad_epochs.any():
        ep.drop(np.flatnonzero(log.bad_epochs), reason="AUTOREJECT")
    print(f"    Reused AutoReject decisions from {log_path}: {int(log.bad_epochs.sum())} epochs dropped, "
          f"{int((log.labels[~log.bad_epochs] == 2).sum())} channel-epochs interpolated")
    return ep, log
