"""
A stand-in for PuLP, backed by scipy.optimize.milp (HiGHS), so bip.py can run in
the browser through Pyodide, where PuLP's CBC solver can't run.

It implements only what bip.py and demo/build_demo.py use. Anything else raises
NotImplementedError, so a gap shows up as an error rather than a wrong verdict.
Every variable is binary, because every variable bip.py makes is.

Select it by putting this folder first on sys.path; in the browser it is loaded
as the module named "pulp". See README.md for how it was checked against PuLP.
Works with scipy 1.12 (Pyodide 0.26) and later.
"""

from __future__ import annotations

from numbers import Number

LpMinimize = 1
LpMaximize = -1
LpStatusOptimal = 1
LpStatusNotSolved = 0
LpStatusInfeasible = -1
LpStatus = {1: "Optimal", 0: "Not Solved", -1: "Infeasible", -2: "Unbounded", -3: "Undefined"}

_LE, _GE, _EQ = "<=", ">=", "=="


class LpAffineExpression:
    """sum(coef * var) + constant. Variables are keyed by object identity."""

    __slots__ = ("terms", "constant")

    def __init__(self, terms=None, constant=0.0):
        self.terms: dict = dict(terms) if terms else {}
        self.constant = float(constant)

    @staticmethod
    def of(x) -> "LpAffineExpression":
        if isinstance(x, LpAffineExpression):
            return x
        if isinstance(x, LpVariable):
            return LpAffineExpression({x: 1.0})
        if isinstance(x, Number) and not isinstance(x, bool):
            return LpAffineExpression(constant=x)
        raise NotImplementedError(f"pulp_shim: can't use {type(x).__name__} in an expression")

    def copy(self) -> "LpAffineExpression":
        return LpAffineExpression(self.terms, self.constant)

    def __add__(self, other):
        o = LpAffineExpression.of(other)
        out = self.copy()
        for v, c in o.terms.items():
            out.terms[v] = out.terms.get(v, 0.0) + c
        out.constant += o.constant
        return out

    __radd__ = __add__

    def __neg__(self):
        return LpAffineExpression({v: -c for v, c in self.terms.items()}, -self.constant)

    def __sub__(self, other):
        return self + (-LpAffineExpression.of(other))

    def __rsub__(self, other):
        return LpAffineExpression.of(other) - self

    def __mul__(self, other):
        if not isinstance(other, Number) or isinstance(other, bool):
            raise NotImplementedError("pulp_shim: only multiplication by a number is supported")
        return LpAffineExpression({v: c * other for v, c in self.terms.items()}, self.constant * other)

    __rmul__ = __mul__

    def __le__(self, other):
        return LpConstraint(self - other, _LE)

    def __ge__(self, other):
        return LpConstraint(self - other, _GE)

    def __eq__(self, other):  # noqa: D105 (PuLP semantics: builds a constraint)
        return LpConstraint(self - other, _EQ)

    __hash__ = object.__hash__

    def value(self):
        total = self.constant
        for v, c in self.terms.items():
            if v.varValue is None:
                return None
            total += c * v.varValue
        return total


class LpVariable:
    __slots__ = ("name", "varValue")

    def __init__(self, name: str, lowBound=None, upBound=None, cat: str = "Continuous"):
        if cat != "Binary":
            raise NotImplementedError("pulp_shim: only binary variables are supported")
        self.name = name
        self.varValue = None

    def _expr(self) -> LpAffineExpression:
        return LpAffineExpression({self: 1.0})

    def __add__(self, other):
        return self._expr() + other

    __radd__ = __add__

    def __sub__(self, other):
        return self._expr() - other

    def __rsub__(self, other):
        return LpAffineExpression.of(other) - self._expr()

    def __neg__(self):
        return -self._expr()

    def __mul__(self, other):
        return self._expr() * other

    __rmul__ = __mul__

    def __le__(self, other):
        return self._expr() <= other

    def __ge__(self, other):
        return self._expr() >= other

    def __eq__(self, other):
        return self._expr() == other

    __hash__ = object.__hash__

    def value(self):
        return self.varValue

    def __repr__(self):
        return self.name


class LpConstraint:
    """expr (sense) 0."""

    __slots__ = ("expr", "sense")

    def __init__(self, expr: LpAffineExpression, sense: str):
        self.expr = expr
        self.sense = sense

    def __bool__(self):
        raise NotImplementedError("pulp_shim: a constraint has no truth value")


def lpSum(items) -> LpAffineExpression:
    out = LpAffineExpression()
    for x in items:
        out = out + x
    return out


def value(x):
    if x is None:
        return None
    if isinstance(x, Number):
        return x
    return x.value()


class PULP_CBC_CMD:
    """Accepted for compatibility; the solve always uses scipy's HiGHS."""

    def __init__(self, *args, **kwargs):
        pass


class LpProblem:
    def __init__(self, name: str = "NoName", sense: int = LpMinimize):
        if sense != LpMinimize:
            raise NotImplementedError("pulp_shim: only minimisation is supported")
        self.name = name
        self.constraints: list[LpConstraint] = []
        self.objective = LpAffineExpression()
        self.status = LpStatusNotSolved

    def __iadd__(self, other):
        if isinstance(other, LpConstraint):
            self.constraints.append(other)
        elif isinstance(other, (LpAffineExpression, LpVariable)):
            self.objective = LpAffineExpression.of(other)
        elif other is True:  # PuLP also ignores a constraint that is trivially true
            pass
        else:
            raise NotImplementedError(f"pulp_shim: can't add {type(other).__name__} to a problem")
        return self

    def solve(self, solver=None) -> int:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_array

        index: dict = {}

        def col(v):
            if v not in index:
                index[v] = len(index)
            return index[v]

        for v in self.objective.terms:
            col(v)
        rows = []
        for con in self.constraints:
            terms = {col(v): c for v, c in con.expr.terms.items() if c != 0}
            rhs = -con.expr.constant
            if not terms:  # a constant row: check it here, as CBC would
                ok = {_LE: 0 <= rhs + 1e-9, _GE: 0 >= rhs - 1e-9, _EQ: abs(rhs) <= 1e-9}[con.sense]
                if not ok:
                    return self._finish(LpStatusInfeasible, index, None)
                continue
            rows.append((terms, con.sense, rhs))

        n = len(index)
        if n == 0:
            return self._finish(LpStatusOptimal, index, np.zeros(0))
        c = np.zeros(n)
        for v, coef in self.objective.terms.items():
            c[index[v]] += coef
        kwargs = {"integrality": np.ones(n), "bounds": Bounds(0, 1)}
        if rows:
            data, ri, ci, lb, ub = [], [], [], [], []
            for r, (terms, sense, rhs) in enumerate(rows):
                for j, coef in terms.items():
                    ri.append(r)
                    ci.append(j)
                    data.append(coef)
                lb.append(rhs if sense in (_GE, _EQ) else -np.inf)
                ub.append(rhs if sense in (_LE, _EQ) else np.inf)
            A = csr_array((data, (ri, ci)), shape=(len(rows), n))
            kwargs["constraints"] = LinearConstraint(A, lb, ub)
        res = milp(c, **kwargs)
        if res.status == 0:
            return self._finish(LpStatusOptimal, index, res.x)
        if res.status == 2:
            return self._finish(LpStatusInfeasible, index, None)
        raise RuntimeError(f"pulp_shim: milp returned status {res.status} ({res.message})")

    def _finish(self, status: int, index: dict, x) -> int:
        self.status = status
        for v, j in index.items():
            v.varValue = None if x is None else float(round(x[j]))
        return status
