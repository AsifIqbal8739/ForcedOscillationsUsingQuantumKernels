"""Test-time attack / processing variants for the Reviewer-4 robustness controls.

Training always uses the published protocol (run_isone_nonstationary_experiments.make_event:
fractional linear-interpolation shift, base 8-20 ms + spatial gradient 8-20 ms + N(0,2 ms),
edge hold). Only the held-out test event is generated with a variant, on the same shifted
(second-half) PMU footprint and with the same genuine windows.
"""
import numpy as np
from scipy.interpolate import CubicSpline
import run_isone_nonstationary_experiments as ns

FS = ns.FS; L = int(ns.OUTER_SEC * FS); ST = int(ns.STRIDE_SEC * FS)

def published_offsets(rng, width):
    return rng.uniform(.008, .020) + np.linspace(-1, 1, width) * rng.uniform(.008, .020) + rng.normal(0, .002, width)

def shift(x, delta, method="linear"):
    """y_j(t) = x_j(t + delta_j(t)); delta has shape (len(x), ncols) in seconds."""
    t = np.arange(len(x)) / FS; y = x.copy()
    for j in range(x.shape[1]):
        d = delta[:, j]
        if not np.any(d): continue
        tq = t + d
        if method == "linear":
            y[:, j] = np.interp(tq, t, x[:, j], left=x[0, j], right=x[-1, j])
        elif method == "cubic":
            y[:, j] = CubicSpline(t, x[:, j], extrapolate=False)(np.clip(tq, t[0], t[-1]))
        elif method == "fourier":   # band-limited shift of a constant delay; mirror-padded to avoid wrap
            dd = float(np.mean(d)); z = np.r_[x[::-1, j], x[:, j], x[::-1, j]]
            f = np.fft.rfftfreq(len(z), 1 / FS); zs = np.fft.irfft(np.fft.rfft(z) * np.exp(2j * np.pi * f * dd), len(z))
            y[:, j] = zs[len(x):2 * len(x)]
    return y

def offsets_for(variant, rng, width, n_t):
    """Return delta (n_t x width) in seconds for the attacked PMUs."""
    t = np.arange(n_t) / FS
    if variant in ("published", "interp_cubic", "interp_fourier", "crop"):
        return np.tile(published_offsets(rng, width), (n_t, 1))
    if variant == "profile_constant":     # no spatial gradient
        return np.tile(rng.uniform(.008, .020) + rng.normal(0, .002, width), (n_t, 1))
    if variant == "profile_ramp":         # clock-frequency error: offset grows linearly across the window
        end = published_offsets(rng, width); return t[:, None] / t[-1] * (2 * end)[None, :]
    if variant == "profile_sinusoid":     # periodic timing manipulation (Eq. 4 with periodic delta)
        base = published_offsets(rng, width); f = rng.uniform(0.1, 0.5); ph = rng.uniform(0, 2 * np.pi)
        return base[None, :] * (1 + 0.5 * np.sin(2 * np.pi * f * t + ph))[:, None]
    if variant == "mag_low":              # below training range
        s = rng.uniform(2, 6) / 8
        return np.tile(s * published_offsets(rng, width), (n_t, 1))
    if variant == "mag_high":             # above training range
        return np.tile(rng.uniform(25, 50) / 14 * published_offsets(rng, width), (n_t, 1))
    if variant == "unseen_randomwalk":    # Brownian clock jitter, zero start, ~10 ms rms at window end
        steps = rng.normal(0, 0.010 / np.sqrt(n_t), (n_t, width)); return np.cumsum(steps, axis=0)
    if variant == "unseen_delay":         # replay-style integer-sample delay (1-3 reports = 33-100 ms)
        return np.tile(rng.integers(1, 4, width) / FS, (n_t, 1)).astype(float)
    raise ValueError(variant)

def make_test_event(raw, seed, variant):
    F, VM, VA = raw
    rng = np.random.default_rng(seed + 0)        # same seed stream position as augment_event(naug=1)
    n = VA.shape[1]; width = max(1, n // 2); attacked = np.arange(n - width, n)
    method = {"interp_cubic": "cubic", "interp_fourier": "fourier"}.get(variant, "linear")
    crop = 0
    out = {k: [] for k in ["static", "multiscale", "structured"]}; y = []; pair = []
    for i in range(0, len(VA) - L + 1, ST):
        fw, vw, aw = F[i:i + L], VM[i:i + L], VA[i:i + L]
        delta = np.zeros((L, n)); delta[:, attacked] = offsets_for(variant, rng, width, L)
        if variant == "crop":
            crop = int(np.ceil(np.abs(delta).max() * FS)) + 1
        sl = slice(crop, L - crop) if crop else slice(None)
        g = ns.reps(fw[sl], vw[sl], aw[sl])
        a = ns.reps(shift(fw, delta, method)[sl], shift(vw, delta, method)[sl], shift(aw, delta, method)[sl])
        pid = len(pair) // 2
        for k in out: out[k].append(g[k])
        y.append(1); pair.append(pid)
        for k in out: out[k].append(a[k])
        y.append(0); pair.append(pid)
    return {k: np.asarray(v) for k, v in out.items()}, np.asarray(y), np.asarray(pair)

VARIANTS = ["interp_cubic", "interp_fourier", "crop", "profile_constant", "profile_ramp", "profile_sinusoid",
            "mag_low", "mag_high", "unseen_randomwalk", "unseen_delay"]
