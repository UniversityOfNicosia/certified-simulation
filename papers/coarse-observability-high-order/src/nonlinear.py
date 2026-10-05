"""
Nonlinear schemes for E6: MUSCL flux limiters and WENO5 variants.

All schemes solve u_t + a u_x = 0, a > 0, on a periodic grid, in
conservation (flux-difference) form, so the spatial mean is preserved to
machine precision.

MUSCL (flux-limiter form, forward Euler; TVD for lambda in (0, 1]):

    u_i^{n+1} = u_i - lambda (F_{i+1/2} - F_{i-1/2}),
    F_{i+1/2} = u_i + (1 - lambda)/2 * phi(r_i) * (u_{i+1} - u_i),
    r_i = (u_i - u_{i-1}) / (u_{i+1} - u_i),

with phi = minmod / MC / van Leer / superbee.  phi == 1 recovers
Lax-Wendroff exactly and phi == r recovers Beam-Warming exactly (used as
self-tests).

WENO5 (Jiang-Shu, mapped, and Z weights) advanced with SSP-RK3: the
forward-Euler pairing is antidissipative (article prop:weno-fe), so the
Shu-Osher three-stage update is used throughout.  With the weights frozen
at their linear values the flux reduces to the `schemes.weno5l` circulant
(self-test).

All steppers accept states of shape (N,) or (N, C) and act columnwise,
so entire collision ensembles evolve in one vectorized call.

Registry: STEPPERS maps name -> (stepper(U, lam), label).
"""

from __future__ import annotations

import numpy as np

import schemes as sch

# ----------------------------------------------------------------------
# Flux limiters phi(r) (vectorized)
# ----------------------------------------------------------------------


def phi_minmod(r: np.ndarray) -> np.ndarray:
    return np.clip(r, 0.0, 1.0)


def phi_superbee(r: np.ndarray) -> np.ndarray:
    return np.maximum(0.0,
                      np.maximum(np.minimum(2.0 * r, 1.0),
                                 np.minimum(r, 2.0)))


def phi_vanleer(r: np.ndarray) -> np.ndarray:
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def phi_mc(r: np.ndarray) -> np.ndarray:
    return np.maximum(0.0,
                      np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))


LIMITERS = {
    "minmod": phi_minmod,
    "mc": phi_mc,
    "vanleer": phi_vanleer,
    "superbee": phi_superbee,
}

# ----------------------------------------------------------------------
# MUSCL step (forward Euler)
# ----------------------------------------------------------------------


def muscl_flux(u: np.ndarray, lam: float, phi_fn) -> np.ndarray:
    """Limited interface fluxes F_{i+1/2}; u of shape (N,) or (N, C)."""
    dm = u - np.roll(u, 1, axis=0)        # u_i - u_{i-1}
    dp = np.roll(u, -1, axis=0) - u       # u_{i+1} - u_i
    r = np.divide(dm, dp, out=np.zeros_like(u), where=(dp != 0.0))
    phi = np.where(dp != 0.0, phi_fn(r), 0.0)
    return u + 0.5 * (1.0 - lam) * phi * dp


def muscl_step(u: np.ndarray, lam: float, phi_fn) -> np.ndarray:
    """One MUSCL flux-limiter step; u of shape (N,) or (N, C)."""
    F = muscl_flux(u, lam, phi_fn)
    return u - lam * (F - np.roll(F, 1, axis=0))


# ----------------------------------------------------------------------
# WENO5 flux and steps
# ----------------------------------------------------------------------

_D_LIN = (0.1, 0.6, 0.3)
_EPS_JS = 1.0e-6       # Jiang-Shu / mapped regularization
_EPS_Z = 1.0e-40       # Borges et al. (WENO-Z) regularization


def _weno5_flux(u: np.ndarray, weights: str) -> np.ndarray:
    """Upwind-biased fifth-order flux F_{i+1/2} (a > 0), vectorized."""
    um2 = np.roll(u, 2, axis=0)
    um1 = np.roll(u, 1, axis=0)
    up1 = np.roll(u, -1, axis=0)
    up2 = np.roll(u, -2, axis=0)

    f0 = (2.0 * um2 - 7.0 * um1 + 11.0 * u) / 6.0
    f1 = (-um1 + 5.0 * u + 2.0 * up1) / 6.0
    f2 = (2.0 * u + 5.0 * up1 - up2) / 6.0

    if weights == "linear":
        d0, d1, d2 = _D_LIN
        return d0 * f0 + d1 * f1 + d2 * f2

    b0 = (13.0 / 12.0) * (um2 - 2.0 * um1 + u) ** 2 \
        + 0.25 * (um2 - 4.0 * um1 + 3.0 * u) ** 2
    b1 = (13.0 / 12.0) * (um1 - 2.0 * u + up1) ** 2 \
        + 0.25 * (um1 - up1) ** 2
    b2 = (13.0 / 12.0) * (u - 2.0 * up1 + up2) ** 2 \
        + 0.25 * (3.0 * u - 4.0 * up1 + up2) ** 2

    if weights in ("js", "m"):
        alpha = [d / (_EPS_JS + b) ** 2
                 for d, b in zip(_D_LIN, (b0, b1, b2))]
    elif weights == "z":
        tau5 = np.abs(b0 - b2)
        alpha = [d * (1.0 + tau5 / (b + _EPS_Z))
                 for d, b in zip(_D_LIN, (b0, b1, b2))]
    else:
        raise ValueError(f"unknown weights '{weights}'")

    asum = alpha[0] + alpha[1] + alpha[2]
    om = [a / asum for a in alpha]

    if weights == "m":
        # Henrick-Aslam-Powers mapping, then renormalize.
        om = [w * (d + d * d - 3.0 * d * w + w * w)
              / (d * d + w * (1.0 - 2.0 * d))
              for d, w in zip(_D_LIN, om)]
        osum = om[0] + om[1] + om[2]
        om = [w / osum for w in om]

    return om[0] * f0 + om[1] * f1 + om[2] * f2


def weno5_fe_step(u: np.ndarray, lam: float, weights: str) -> np.ndarray:
    """One forward-Euler step (antidissipative for the linear weights;
    kept for the self-test and the linear-weights consistency check)."""
    F = _weno5_flux(u, weights)
    return u - lam * (F - np.roll(F, 1, axis=0))


def weno5_ssprk3_step(u: np.ndarray, lam: float, weights: str) -> np.ndarray:
    """Shu-Osher SSP-RK3 built on the WENO5 forward-Euler operator."""
    u1 = weno5_fe_step(u, lam, weights)
    u2 = 0.75 * u + 0.25 * weno5_fe_step(u1, lam, weights)
    return u / 3.0 + (2.0 / 3.0) * weno5_fe_step(u2, lam, weights)


def weno5_ssprk3_flux(u: np.ndarray, lam: float, weights: str) -> np.ndarray:
    """Realized flux of the full SSP-RK3 step: unrolling the Shu-Osher
    stages gives u^{n+1} = u - lam*(F - roll(F)) with
    F = (F(u) + F(u1) + 4 F(u2)) / 6."""
    F0 = _weno5_flux(u, weights)
    u1 = u - lam * (F0 - np.roll(F0, 1, axis=0))
    F1 = _weno5_flux(u1, weights)
    u2 = 0.75 * u + 0.25 * (u1 - lam * (F1 - np.roll(F1, 1, axis=0)))
    F2 = _weno5_flux(u2, weights)
    return (F0 + F1 + 4.0 * F2) / 6.0


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

def _muscl(name):
    fn = LIMITERS[name]
    return lambda u, lam: muscl_step(u, lam, fn)


def _weno(weights):
    return lambda u, lam: weno5_ssprk3_step(u, lam, weights)


STEPPERS: dict[str, tuple] = {
    "muscl-minmod":   (_muscl("minmod"),   "MUSCL minmod"),
    "muscl-mc":       (_muscl("mc"),       "MUSCL MC"),
    "muscl-vanleer":  (_muscl("vanleer"),  "MUSCL van Leer"),
    "muscl-superbee": (_muscl("superbee"), "MUSCL superbee"),
    "weno5-js":       (_weno("js"),        "WENO5-JS + RK3"),
    "weno5-m":        (_weno("m"),         "WENO5-M + RK3"),
    "weno5-z":        (_weno("z"),         "WENO5-Z + RK3"),
}


def _muscl_flux_fn(name):
    fn = LIMITERS[name]
    return lambda u, lam: muscl_flux(u, lam, fn)


def _weno_flux_fn(weights):
    return lambda u, lam: weno5_ssprk3_flux(u, lam, weights)


# Realized interface fluxes of the full step, keyed as in STEPPERS.
FLUXES = {
    "muscl-minmod":   _muscl_flux_fn("minmod"),
    "muscl-mc":       _muscl_flux_fn("mc"),
    "muscl-vanleer":  _muscl_flux_fn("vanleer"),
    "muscl-superbee": _muscl_flux_fn("superbee"),
    "weno5-js":       _weno_flux_fn("js"),
    "weno5-m":        _weno_flux_fn("m"),
    "weno5-z":        _weno_flux_fn("z"),
}


# ----------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def self_test() -> None:
    rng = np.random.default_rng(1)
    N, lam = 48, 0.37
    u = rng.standard_normal(N)

    # phi == 1 reproduces Lax-Wendroff; phi == r reproduces Beam-Warming.
    A_lw = sch.circulant_matrix(sch.lax_wendroff(lam), N)
    A_bw = sch.circulant_matrix(sch.beam_warming(lam), N)
    v = muscl_step(u, lam, lambda r: np.ones_like(r))
    _check(np.allclose(v, A_lw @ u, atol=1e-13), "phi=1 does not give LW")
    v = muscl_step(u, lam, lambda r: r)
    _check(np.allclose(v, A_bw @ u, atol=1e-13), "phi=r does not give BW")

    # Linear WENO weights reproduce the weno5l circulant (forward Euler).
    A_w5 = sch.circulant_matrix(sch.SCHEMES["weno5l"][0](lam), N)
    v = weno5_fe_step(u, lam, "linear")
    _check(np.allclose(v, A_w5 @ u, atol=1e-12),
           "linear-weights WENO5 flux does not match weno5l")

    # Conservation and column independence for every registered stepper.
    U = rng.standard_normal((N, 7))
    for name, (step, _label) in STEPPERS.items():
        V = step(U, 0.5)
        _check(np.allclose(V.mean(axis=0), U.mean(axis=0), atol=1e-14),
               f"{name}: mean drift")
        v0 = step(U[:, 0], 0.5)
        _check(np.allclose(v0, V[:, 0], atol=1e-14),
               f"{name}: columns are not independent")

    # Realized flux reproduces the full step (incl. the RK3 unrolling).
    for name, (step, _label) in STEPPERS.items():
        for lam2 in (0.9, 0.5, 0.1):
            F = FLUXES[name](U, lam2)
            recon = U - lam2 * (F - np.roll(F, 1, axis=0))
            _check(np.allclose(recon, step(U, lam2), atol=1e-13),
                   f"{name}: realized flux does not reproduce the step")

    # Smooth-data weight consistency: nonlinear weights approach the
    # linear ones on a well-resolved sine (away from machine effects).
    xs = (np.arange(N) + 0.5) / N
    us = 0.5 + 0.25 * np.sin(2 * np.pi * xs)
    for w in ("js", "m", "z"):
        fnl = _weno5_flux(us, w)
        flin = _weno5_flux(us, "linear")
        _check(np.max(np.abs(fnl - flin)) < 5e-3,
               f"weno5-{w}: smooth-data flux far from linear weights")

    # TVD sanity: minmod MUSCL does not increase total variation.
    step_u = np.where(xs < 0.5, 1.0, 0.0)
    tv = lambda z: np.abs(z - np.roll(z, 1)).sum()
    z = step_u.copy()
    for _ in range(100):
        z = muscl_step(z, 0.5, phi_minmod)
    _check(tv(z) <= tv(step_u) + 1e-12, "minmod MUSCL increased TV")

    print("nonlinear.py self-test: all checks passed.")


if __name__ == "__main__":
    self_test()
