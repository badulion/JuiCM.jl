# Reference data

## `shin2004_lock_exchange.csv` — rung V3 (lock exchange)

**Source.** Shin, J.O., Dalziel, S.B. & Linden, P.F. (2004), *Gravity
currents produced by lock exchange*, J. Fluid Mech. **521**, 1–34,
doi:10.1017/S0022112004002369. Benjamin, T.B. (1968), *Gravity currents
and related phenomena*, J. Fluid Mech. **31**, 209–248.

**Quantity.** The front Froude number of the slumping (constant-speed)
phase of a full-depth lock release (lock depth = channel depth),
F = U/√(g′H). Shin et al. report F ≈ 0.48–0.50 for this case, close to
Benjamin's energy-conserving value 0.5; the wider laboratory literature
scatters over 0.35–0.50.

**Digitization method.** None: the values are the published scalar
slumping-phase Froude numbers (no curve was digitized). The front
position curve used by the test is the constant-speed line
x_f(t) = F·√(g′H)·t with the band centre F = 0.49.

**Tolerance.** Issue #40 acceptance (ratified 2026-10-16): within ~10 %
of the published value, i.e. F ∈ [0.9·0.48, 1.1·0.50] = [0.432, 0.55],
and front position within 10 % of the constant-speed curve. Changing
this tolerance requires a dedicated issue and a human decision
(AGENTS.md hard rule 2).

**Model-side note.** The hydrostatic two-layer shallow-water limit of
the same geometry (1-D Rusanov/MUSCL solver, converged) gives F = 0.545;
Benjamin's 0.5 needs the energy-conserving (non-hydrostatic) head. The
JuiCM V3 configuration (20 z* layers, 24-cell-wide channel) measures
F = 0.477–0.483 on the 2/10/25 % contours.
