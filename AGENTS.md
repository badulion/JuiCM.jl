# AGENTS.md — JuiCM.jl Agent Workflow

Operating instructions for AI coding agents (Claude, Codex, Copilot, etc.)
working in this repository. If you are an AI agent making changes here, follow
this file exactly. Read `DESIGN.md` first — it is the source of truth for the
architecture, the validation ladder, and the phase plan. Never contradict it;
if a task requires deviating from `DESIGN.md`, stop and open an issue instead.

---

## Project context

- **What this is:** a from-scratch, fully-coupled ocean-atmosphere GCM in
  pure Julia (cubed-sphere horizontal, z-star ocean vertical, single-node first).
- **The architecture is the research contribution.** Do not pull in GCM code
  from other packages (no ClimaCore, no Oceananigans, no SpeedyWeather).
  Infrastructure deps (NCDatasets, OffsetArrays, SeawaterPolynomials,
  KernelAbstractions) are allowed; dycores, coupler, and physics are ours.
- **Numerical bugs look like physics.** Green tests prove code runs, not that
  the physics is right. This is why the workflow below is strict.

---

## The workflow

All work is managed via **GitHub issues and milestones**. No work happens
without an issue.

### 1. Work only from issues

1. Check the milestone for the current phase (Phases are defined in
   `DESIGN.md` §5). Every issue belongs to a milestone.
2. Before starting any task, either pick an open issue or create one.
   A new issue must state: what to build, which design doc section it
   implements, and the acceptance criterion (usually a validation-ladder rung
   from `DESIGN.md` §4).
3. One issue → one branch → one PR. Branch naming: `<issue-number>-short-slug`
   (e.g. `12-cubed-sphere-metrics`). If a task doesn't fit in one PR, it's
   multiple issues.
4. Reference the issue in the PR body (`Closes #N`). A PR without an issue
   reference must not be opened.

### 2. Docs and tests before a PR is submitted

A PR is not ready for submission unless **both** exist in the same PR as the
code:

- **Docs:** every new module, type, or public function is documented
  (docstrings for public API; `docs/` updates when user-facing behavior
  changes). A physics module PR documents its equations and their source.
- **Tests:** every new behavior is covered by tests in `test/`. Numerical
  code gets quantitative tests, not just "runs without error" — conservation
  checks, convergence order, or comparison against reference values
  (validation-ladder rungs live in `test/` and are part of the suite).
  Bug-fix PRs add a regression test that fails before the fix.

Write docs and tests **before** opening the PR, not as a follow-up. A PR
without docs and tests is incomplete — do not submit it.

### 3. Merge gate: all tests pass locally

1. Run the full test suite locally:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
2. The **entire** suite must pass. Never skip, `@test_skip`, or weaken a test
   to make it green. If a validation test fails, that is the finding — report
   it in the issue, don't merge around it.
3. If and only if the full suite passes locally, the PR may be merged
   (auto-merge is authorized for this repo).
4. After merge: close the issue (the `Closes #N` reference does this), delete
   the branch, and post a one-paragraph summary of what landed in the issue
   or PR thread.

Local tests are the merge gate for now. When CI is added later, this section
changes to require green local **and** CI — do not treat this file as
permitting CI-free merges after that change exists.

---

## Hard rules

1. **No new GCM dependencies.** `DESIGN.md` §6 has the allowlist. Anything that
   is itself a dynamical core, coupler, or physics package is forbidden.
2. **Never weaken or delete a passing validation test.** Validation-ladder
   tests (Williamson, Held-Suarez, gravity current, overflow, spin-up budgets)
   are the project's immune system. Changing tolerances requires a dedicated
   issue with justification, and a human decision.
3. **No direct indexing into `Field.data`** outside `src/Fields/`. Physics code
   goes through operators — this is what keeps the MPI escape hatch
   (`DESIGN.md` §3.3) viable.
4. **No force-push to `main`.** Ever.
5. **Don't merge a failing build, even if the failure looks unrelated.**
   Investigate first; unrelated breakage is a finding worth an issue.
6. **Physics modules implement the interface** (`compute_tendency!` on
   `AbstractPhysicsModule`). If an apparently needed change to the interface
   itself comes up, that's a design decision — open an issue, cite the
   `DESIGN.md` section, don't improvise.

## When stuck or uncertain

- Ambiguity about architecture → stop, open an issue with the question and
  the options.
- Test suite failure you can't explain in 30 minutes → open an issue with
  the failure output; don't paper over it.
- Anything touching `DESIGN.md` (grid, coordinates, validation criteria) →
  always ask first, never decide alone.

---

*Auto-merge is authorized because the human owner trusts green local tests as
the gate. That trust depends on rules 2–5 above holding. If they break, say so
in the PR.*
