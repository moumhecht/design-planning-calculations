import numpy as np
from scipy.optimize import brentq, least_squares

# ── Physical constants ───────────────────────────────────────────────
e    = 1.602176634e-19
h    = 6.626e-34
hbar = 1.05457e-34
fF   = 1e-15
nH   = 1e-9

# ── Physical bounds on inductance (fabricated Xmon/resonator inductors) ─
L_MIN_NH = 0.1
L_MAX_NH = 5.0

# ── Circuit functions ────────────────────────────────────────────────
def Ctot(CR, Cg, CQ):
    return CR*CQ + CR*Cg + CQ*Cg

def ECQ(CR, Cg, CQ):
    return (e**2/2) * (CR + Cg) / Ctot(CR, Cg, CQ)

def ECR(CR, Cg, CQ):
    return (e**2/2) * (CQ + Cg) / Ctot(CR, Cg, CQ)

def EL(L):
    return hbar**2 / (4 * e**2 * L)

def omegaQ(ratioQ, CR, Cg, CQ):
    ec  = ECQ(CR, Cg, CQ)
    val = 8 * ratioQ * ec**2
    return np.sqrt(val) - ec if val > 0 else np.nan

def omegaR(L, CR, Cg, CQ):
    val = 8 * EL(L) * ECR(CR, Cg, CQ)
    return np.sqrt(val) if val > 0 else np.nan

def g(ratioQ, L, CR, Cg, CQ):
    ec  = ECQ(CR, Cg, CQ)
    ecr = ECR(CR, Cg, CQ)
    el  = EL(L)
    inside = (el / (32*ecr)) * (ratioQ * ec / (32*ec))
    if inside <= 0:
        return np.nan
    prefactor  = (-4*e**2) * (Cg / Ctot(CR, Cg, CQ))
    correction = 1 + (-ec) / (2 * omegaQ(ratioQ, CR, Cg, CQ))
    return prefactor * (inside**0.25) * correction


def get_ratio(omegaQ_Hz, ECQ_Hz):
    return ((omegaQ_Hz + ECQ_Hz)**2) / (8 * ECQ_Hz**2)


def make_initial_guess(omegaQ_Hz, omegaR_Hz, ECQ_Hz, g_Hz):
    """
    Physically motivated starting point that scales correctly
    across fQ: 3-10 GHz, fR: 4-11 GHz, EC: 100-300 MHz.
    """

    CQ0 = (e**2 / (2 * ECQ_Hz * h)) / fF

    L_seed   = 0.5 * (L_MIN_NH + L_MAX_NH) * nH
    EL_seed  = hbar**2 / (4 * e**2 * L_seed)
    ECR_seed = (omegaR_Hz * h)**2 / (8 * EL_seed)
    CR0      = (e**2 / (2 * ECR_seed)) / fF


    Cg0 = max(0.1, abs(g_Hz) / omegaQ_Hz * 10)


    ECR0 = (e**2/2) / (CR0 * fF)
    L0   = hbar**2 / (4 * e**2 * (omegaR_Hz * h)**2 / (8 * ECR0)) / nH


    CQ0 = np.clip(CQ0,  10,  600)
    CR0 = np.clip(CR0,  30, 3000)
    Cg0 = np.clip(Cg0, 0.05,  30)
    L0  = np.clip(L0,  L_MIN_NH, L_MAX_NH)   # <-- hard physical clamp on L

    return CQ0, CR0, Cg0, L0

def find_bracket(func, lo, hi, n=500):
    """Scan [lo,hi] and return the first sub-interval with a sign change."""
    xs = np.linspace(lo, hi, n)
    ys = np.array([func(x) for x in xs])
    finite = np.isfinite(ys)
    if finite.sum() < 2:
        return False, None, None
    xs_f, ys_f = xs[finite], ys[finite]
    idx = np.where(np.diff(np.sign(ys_f)))[0]
    if len(idx) == 0:
        return False, None, None
    return True, xs_f[idx[0]], xs_f[idx[0]+1]

def solve_g(targetOmegaQ_Hz, targetOmegaR_Hz, targetECQ_Hz, targetg_Hz,
          verbose=True):
    """
    Solve for (CQ, CR, Cg, L) given target frequencies and energies.

    Supported ranges
    ----------------
    fQ  : 3  – 10  GHz
    fR  : 4  – 11  GHz   (must be > fQ)
    ECQ : 100 – 300 MHz
    g   : any negative value in MHz range
    L   : constrained to 0.1 - 5 nH (physically realizable on-chip inductor)
    """

    # targets in Joules
    targetOmegaQ = targetOmegaQ_Hz * h
    targetOmegaR = targetOmegaR_Hz * h
    targetECQ    = targetECQ_Hz    * h
    targetg      = targetg_Hz      * h
    ratioQ       = get_ratio(targetOmegaQ_Hz, targetECQ_Hz)

    if verbose:
        print(f"── Targets ──────────────────────────────")
        print(f"  fQ  = {targetOmegaQ_Hz/1e9:.3f} GHz")
        print(f"  fR  = {targetOmegaR_Hz/1e9:.3f} GHz")
        print(f"  ECQ = {targetECQ_Hz/1e6:.1f} MHz")
        print(f"  g   = {targetg_Hz/1e6:.1f} MHz")
        print(f"  EJ/EC (derived) = {ratioQ:.4f}")

    CQ0, CR0, Cg0, L0 = make_initial_guess(
        targetOmegaQ_Hz, targetOmegaR_Hz, targetECQ_Hz, targetg_Hz)

    CQ = CQ0 * fF
    CR = CR0 * fF
    Cg = Cg0 * fF
    L  = L0  * nH

    if verbose:
        print(f"\n── Initial guess ────────────────────────")
        print(f"  CQ = {CQ0:.2f} fF")
        print(f"  CR = {CR0:.2f} fF")
        print(f"  Cg = {Cg0:.3f} fF")
        print(f"  L  = {L0:.3f} nH")

    converged = False
    best_res   = np.inf
    best_state = (CQ, CR, Cg, L)
    for iteration in range(500):

        def res_ECQ(CR_fF):
            return ECQ(CR_fF*fF, Cg, CQ) - targetECQ
        ok, lo, hi = find_bracket(res_ECQ, 5, 15000)
        if ok:
            CR = brentq(res_ECQ, lo, hi, xtol=1e-8, rtol=1e-12) * fF

        def res_omegaQ(CQ_fF):
            v = omegaQ(ratioQ, CR, Cg, CQ_fF*fF)
            return (v - targetOmegaQ) if np.isfinite(v) else 1e30
        ok, lo, hi = find_bracket(res_omegaQ, 5, 3000)
        if ok:
            CQ = brentq(res_omegaQ, lo, hi, xtol=1e-8, rtol=1e-12) * fF


        def res_omegaR(L_nH):
            v = omegaR(L_nH*nH, CR, Cg, CQ)
            return (v - targetOmegaR) if np.isfinite(v) else 1e30
        ok, lo, hi = find_bracket(res_omegaR, L_MIN_NH, L_MAX_NH)
        if ok:
            L = brentq(res_omegaR, lo, hi, xtol=1e-10, rtol=1e-12) * nH
        else:

            edge_lo = res_omegaR(L_MIN_NH)
            edge_hi = res_omegaR(L_MAX_NH)
            L = (L_MIN_NH if abs(edge_lo) < abs(edge_hi) else L_MAX_NH) * nH

        def res_g(Cg_fF):
            v = g(ratioQ, L, CR, Cg_fF*fF, CQ)
            return (v - targetg) if np.isfinite(v) else 1e30
        ok, lo, hi = find_bracket(res_g, 0.001, 500)
        if ok:
            Cg = brentq(res_g, lo, hi, xtol=1e-10, rtol=1e-12) * fF


        gv = g(ratioQ, L, CR, Cg, CQ)
        if not np.isfinite(gv):
            continue

        r1 = abs(ECQ(CR,Cg,CQ)           - targetECQ)    / abs(targetECQ)
        r2 = abs(omegaQ(ratioQ,CR,Cg,CQ) - targetOmegaQ) / abs(targetOmegaQ)
        r3 = abs(omegaR(L,CR,Cg,CQ)      - targetOmegaR) / abs(targetOmegaR)
        r4 = abs(gv                       - targetg)       / abs(targetg)
        max_res = max(r1, r2, r3, r4)

        if max_res < best_res:
            best_res   = max_res
            best_state = (CQ, CR, Cg, L)

        if verbose and iteration % 20 == 0:
            print(f"  iter {iteration:3d} | max_res={max_res:.2e} | "
                  f"ECQ={ECQ(CR,Cg,CQ)/h/1e6:.2f}MHz "
                  f"fQ={omegaQ(ratioQ,CR,Cg,CQ)/h/1e9:.4f}GHz "
                  f"fR={omegaR(L,CR,Cg,CQ)/h/1e9:.4f}GHz "
                  f"g={gv/h/1e6:.3f}MHz "
                  f"L={L/nH:.4f}nH")

        if max_res < 1e-8:
            converged = True
            break

    CQ, CR, Cg, L = best_state
    if verbose:
        print(f"\n  (best iterate had max_res={best_res:.2e}, used as polish start)")

    x0 = [CQ/fF, CR/fF, Cg/fF, L/nH]


    lower = [x0[0]*0.1, x0[1]*0.1, x0[2]*0.1, L_MIN_NH]
    upper = [x0[0]*10.0, x0[1]*10.0, x0[2]*10.0, L_MAX_NH]


    x0[3] = float(np.clip(x0[3], L_MIN_NH, L_MAX_NH))

    def residuals_polish(x_sc):
        CQp, CRp, Cgp, Lp = x_sc[0]*fF, x_sc[1]*fF, x_sc[2]*fF, x_sc[3]*nH
        gv = g(ratioQ, Lp, CRp, Cgp, CQp)
        if not np.isfinite(gv):
            return [1e6]*4
        return [
            (ECQ(CRp,Cgp,CQp)           - targetECQ)    / abs(targetECQ),
            (omegaQ(ratioQ,CRp,Cgp,CQp) - targetOmegaQ) / abs(targetOmegaQ),
            (omegaR(Lp,CRp,Cgp,CQp)     - targetOmegaR) / abs(targetOmegaR),
            (gv                          - targetg)       / abs(targetg),
        ]

    try:
        res = least_squares(residuals_polish, x0,
                            bounds=(lower, upper),
                            method='trf',
                            ftol=1e-14, xtol=1e-14, gtol=1e-14,
                            max_nfev=50000)
        CQ, CR, Cg, L = [v*u for v,u in zip(res.x, [fF,fF,fF,nH])]
    except Exception as ex:
        if verbose:
            print(f"  Polish skipped: {ex}")

    gv = g(ratioQ, L, CR, Cg, CQ)

    # ── Final residuals ───────────────────────────────────────────────
    r1 = abs(ECQ(CR,Cg,CQ)/h/1e6           - targetECQ_Hz/1e6)    / (targetECQ_Hz/1e6)
    r2 = abs(omegaQ(ratioQ,CR,Cg,CQ)/h/1e9 - targetOmegaQ_Hz/1e9) / (targetOmegaQ_Hz/1e9)
    r3 = abs(omegaR(L,CR,Cg,CQ)/h/1e9      - targetOmegaR_Hz/1e9) / (targetOmegaR_Hz/1e9)
    r4 = abs(gv/h/1e6 - targetg_Hz/1e6) / abs(targetg_Hz/1e6) if np.isfinite(gv) else np.nan
    max_res = max(r for r in [r1,r2,r3,r4] if np.isfinite(r))

    print(f"\n── Solution ─────────────────────────────")
    print(f"  CQ = {CQ/fF:.4f} fF")
    print(f"  CR = {CR/fF:.4f} fF")
    print(f"  Cg = {Cg/fF:.4f} fF")
    print(f"  L  = {L/nH:.4f}  nH")
    print(f"\n── Verification ─────────────────────────")
    print(f"  ECQ = {ECQ(CR,Cg,CQ)/h/1e6:.4f} MHz   (target: {targetECQ_Hz/1e6:.1f})")
    print(f"  fQ  = {omegaQ(ratioQ,CR,Cg,CQ)/h/1e9:.4f} GHz   (target: {targetOmegaQ_Hz/1e9:.3f})")
    print(f"  fR  = {omegaR(L,CR,Cg,CQ)/h/1e9:.4f} GHz   (target: {targetOmegaR_Hz/1e9:.3f})")
    print(f"  g   = {gv/h/1e6:.4f} MHz   (target: {targetg_Hz/1e6:.1f})")
    print(f"\n  max_residual = {max_res:.2e}  {'✓ converged' if max_res < 1e-4 else '✗ check targets'}")
    print(f"  L in [{L_MIN_NH}, {L_MAX_NH}] nH: {'✓' if L_MIN_NH - 1e-9 <= L/nH <= L_MAX_NH + 1e-9 else '✗ OUT OF RANGE'}")

    return {
        "CQ_fF": CQ/fF, "CR_fF": CR/fF, "Cg_fF": Cg/fF, "L_nH": L/nH,
        "EJ_EC": ratioQ,
        "ECQ_MHz": ECQ(CR,Cg,CQ)/h/1e6,
        "fQ_GHz" : omegaQ(ratioQ,CR,Cg,CQ)/h/1e9,
        "fR_GHz" : omegaR(L,CR,Cg,CQ)/h/1e9,
        "g_MHz"  : gv/h/1e6 if np.isfinite(gv) else np.nan,
        "max_res": max_res,
        "converged": max_res < 1e-4,
    }


if __name__ == "__main__":
    # Same targets as the Mathematica notebook (fQ=4.5GHz, fR=5.75GHz,
    # ECQ=200MHz, g=-50MHz) but ratioQ is now derived from fQ & ECQ
    # instead of being fixed at 70.
    solve(targetOmegaQ_Hz=4.5e9, targetOmegaR_Hz=5.75e9,
          targetECQ_Hz=200e6, targetg_Hz=-50e6)