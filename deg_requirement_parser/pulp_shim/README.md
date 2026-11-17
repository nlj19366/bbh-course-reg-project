# pulp_shim

A stand-in for PuLP that solves with `scipy.optimize.milp` (HiGHS) instead of CBC.
It exists so `bip.py` can run unchanged in the browser through Pyodide, where
CBC can't run. On a normal machine, keep using real PuLP.

## Using it

Put this folder first on `sys.path`, and `import pulp` picks up the shim:

```bash
PYTHONPATH=pulp_shim python check_bip.py --program-dir out_v2
```

In the browser, the app loads `pulp.py` as the module named `pulp` before `bip.py`,
after `await pyodide.loadPackage("scipy")`.

## What it supports

Only what `bip.py` and `finder.py` use:
- binary `LpVariable`s;
- `lpSum`, linear expressions, and `<=` / `>=` / `==` constraints;
- `LpProblem` with `LpMinimize`, and `prob +=` for constraints and the objective;
- `solve`, `status`, `LpStatus`, `value()`.

Anything else raises `NotImplementedError`, and an unexpected solver status raises
`RuntimeError`. So a gap shows up as an error rather than as a wrong verdict.

## How it was checked

- `compare.py` runs the same workload with real PuLP + CBC and with the shim, and
  diffs the results. The workload:
  - every golden case, on `examples/` and on `out_v2/`;
  - every demo student, on every node of their programme's tree;
  - six course lists against all 443 programmes.

  On 4 Nov 2026: 2,703 results, 0 differences. The CBC run took 55 s and the shim
  run took 16 s.
- `check_bip.py` (both tree sets) and `test_bip.py` pass with the shim.
- In Pyodide 0.26.4 (scipy 1.12), `scipy.optimize.milp` matched CBC on 2,586
  problems that `bip.py` built, with a median of 0.8 ms per solve.

Re-run after changing `bip.py` or this file:

```bash
python pulp_shim/compare.py
```

## Known difference

When several assignments use equally few courses, the shim may put a course in a
different slot than CBC does. The "courses counted" lists can then differ.
Verdicts and unmet requirements are compared and don't.
