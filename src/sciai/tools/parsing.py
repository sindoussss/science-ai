"""Safe expression parsing.

``sympify`` and ``parse_expr`` call ``eval``, so model text never reaches them
directly. First layer: a token whitelist (names, numbers, a fixed operator set,
whitelisted function names only before "(" and no attribute access, strings,
brackets, keywords or dunders). Second layer: ``parse_expr`` runs with an empty
``__builtins__`` and a namespace holding only whitelisted SymPy objects. The
sandbox process (tools/sandbox.py) is the third layer.
"""
from __future__ import annotations

import builtins
import io
import keyword
import re
import tokenize
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import auto_number, auto_symbol, convert_xor, parse_expr

MAX_LEN = 2000
MAX_DIGITS = 60
MAX_LITERAL_EXPONENT = 1000

FUNCTIONS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh,
    "atanh": sp.atanh, "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "cbrt": sp.cbrt, "root": sp.root, "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceiling": sp.ceiling, "factorial": sp.factorial, "gamma": sp.gamma,
    "binomial": sp.binomial, "erf": sp.erf, "Min": sp.Min, "Max": sp.Max, "re": sp.re,
    "im": sp.im, "conjugate": sp.conjugate, "Heaviside": sp.Heaviside,
}
CONSTANTS: dict[str, Any] = {"pi": sp.pi, "E": sp.E, "I": sp.I, "oo": sp.oo, "inf": sp.oo}

_PY_BUILTINS = frozenset(dir(builtins))
_BIG_FUNCS = {"factorial", "gamma", "binomial"}
MAX_FACTORIAL_ARG = 5_000
_ALLOWED_OPS = {"+", "-", "*", "/", "**", "^", "(", ")", ","}
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,29}$")
_NUMBER_RE = re.compile(r"^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_ASSUMPTIONS = {"real", "positive", "negative", "nonnegative", "nonpositive", "integer", "nonzero"}

_GLOBALS_BASE: dict[str, Any] = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    # Needed by parse_expr(evaluate=False), which rewrites operators into these.
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
}


class ParseError(ValueError):
    pass


def _check_tokens(text: str) -> None:
    if len(text) > MAX_LEN:
        raise ParseError(f"expression longer than {MAX_LEN} characters")
    if not text.strip():
        raise ParseError("empty expression")
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT,
                                  tokenize.DEDENT)]
    except (tokenize.TokenError, IndentationError) as exc:
        raise ParseError(f"cannot tokenize: {exc}") from None
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        prev = toks[i - 1] if i > 0 else None
        if tok.type == tokenize.NAME:
            name = tok.string
            if keyword.iskeyword(name) or not _NAME_RE.match(name) or (
                    name in _PY_BUILTINS and name not in FUNCTIONS):
                raise ParseError(f"name {name!r} is not allowed")
            followed_by_call = nxt is not None and nxt.string == "("
            if followed_by_call and name not in FUNCTIONS:
                raise ParseError(f"unknown function {name!r}; use * for multiplication")
            if not followed_by_call and name in FUNCTIONS:
                raise ParseError(f"{name!r} is a function and must be called, e.g. {name}(x)")
        elif tok.type == tokenize.NUMBER:
            if not _NUMBER_RE.match(tok.string):
                raise ParseError(f"number {tok.string!r} is not allowed")
            if len(re.sub(r"\D", "", tok.string)) > MAX_DIGITS:
                raise ParseError("number literal too long")
            if prev is not None and prev.string == "(" and i >= 2 and toks[i - 2].string in _BIG_FUNCS:
                if abs(float(tok.string)) > MAX_FACTORIAL_ARG:
                    raise ParseError(f"argument of {toks[i - 2].string} is too large")
            if prev is not None and prev.string in ("**", "^"):
                if abs(float(tok.string)) > MAX_LITERAL_EXPONENT:
                    raise ParseError("exponent literal too large")
        elif tok.type == tokenize.OP:
            if tok.string not in _ALLOWED_OPS:
                raise ParseError(f"operator {tok.string!r} is not allowed")
        else:
            raise ParseError(f"token {tok.string!r} is not allowed")


def make_symbol(name: str, assumptions: dict[str, str] | None = None) -> sp.Symbol:
    if not _NAME_RE.match(name) or name in FUNCTIONS or name in CONSTANTS or keyword.iskeyword(name):
        raise ParseError(f"invalid variable name {name!r}")
    kind = (assumptions or {}).get(name)
    if kind is None:
        return sp.Symbol(name)
    if kind not in _ASSUMPTIONS:
        raise ParseError(f"unknown assumption {kind!r} for {name!r}")
    return sp.Symbol(name, **{kind: True})


def parse(text: str, assumptions: dict[str, str] | None = None) -> sp.Expr:
    if not isinstance(text, str):
        text = str(text)
    _check_tokens(text)
    # Parse unevaluated first so size bombs (9**9**9, factorial(10**6)) are
    # rejected before SymPy tries to compute them.
    _guard_size(_parse_raw(text, assumptions, evaluate=False))
    return _parse_raw(text, assumptions, evaluate=True)


def _parse_raw(text: str, assumptions: dict[str, str] | None, evaluate: bool) -> sp.Basic:
    local: dict[str, Any] = {**FUNCTIONS, **CONSTANTS}
    for name in (assumptions or {}):
        local[name] = make_symbol(name, assumptions)
    try:
        expr = parse_expr(
            text,
            local_dict=local,
            global_dict=dict(_GLOBALS_BASE),
            transformations=(auto_symbol, auto_number, convert_xor),
            evaluate=evaluate,
        )
    except ParseError:
        raise
    except Exception as exc:  # SyntaxError, TypeError from bad call shapes, ...
        raise ParseError(f"cannot parse {text!r}: {exc}") from None
    if not isinstance(expr, sp.Basic):
        raise ParseError(f"{text!r} did not parse to an expression")
    return expr


MAX_RESULT_LOG10 = 10_000


def _mp_value(e: sp.Basic) -> Any:
    """Value of a symbol-free subtree in mpmath (huge exponents are cheap there).
    Returns None for anything not handled, which is treated as harmless."""
    import mpmath

    mpmath.mp.dps = 15
    if e.is_Integer or e.is_Rational:
        return mpmath.mpf(e.p) / mpmath.mpf(e.q)
    if e.is_Float:
        return mpmath.mpf(str(e))
    if e is sp.pi:
        return mpmath.pi
    if e is sp.E:
        return mpmath.e
    if isinstance(e, (sp.Add, sp.Mul, sp.Pow)):
        vals = [_mp_value(a) for a in e.args]
        if any(v is None for v in vals):
            return None
        try:
            if isinstance(e, sp.Add):
                return mpmath.fsum(vals)
            if isinstance(e, sp.Mul):
                return mpmath.fprod(vals)
            return mpmath.power(vals[0], vals[1])
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    return None


def _guard_size(expr: sp.Basic) -> None:
    import mpmath

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and not node.free_symbols:
            base, exp = _mp_value(node.args[0]), _mp_value(node.args[1])
            if base is None or exp is None:
                continue
            if abs(base) not in (0, 1) and abs(exp) > 1e6:
                raise ParseError("power is too large to evaluate")
            v = _mp_value(node)
            if v is not None and v != 0 and abs(mpmath.log10(abs(v))) > MAX_RESULT_LOG10:
                raise ParseError("number is too large to evaluate")
        elif isinstance(node, sp.factorial) and not node.free_symbols:
            v = _mp_value(node.args[0])
            if v is not None and abs(v) > MAX_FACTORIAL_ARG:
                raise ParseError("factorial argument is too large")


def parse_relation(text: str, assumptions: dict[str, str] | None = None) -> sp.Basic:
    """Parse "lhs = rhs" (or "lhs == rhs") into Eq; a bare expression means expr = 0."""
    if any(op in text for op in ("<=", ">=", "!=")):
        raise ParseError("inequalities are not supported here")
    parts = re.split(r"==|=", text)
    if len(parts) == 1:
        return sp.Eq(parse(text, assumptions), 0)
    if len(parts) != 2:
        raise ParseError("expected exactly one '='")
    return sp.Eq(parse(parts[0], assumptions), parse(parts[1], assumptions))


def var(name: str, assumptions: dict[str, str] | None = None) -> sp.Symbol:
    return make_symbol(name, assumptions)


def canonical(expr: sp.Basic) -> str:
    return sp.srepr(expr)


def expr_result(expr: sp.Basic) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": "expr", "value": str(expr), "srepr": sp.srepr(expr),
                           "latex": sp.latex(expr)}
    if isinstance(expr, sp.Expr) and expr.is_number:
        try:
            approx = complex(sp.N(expr, 17))
            out["numeric"] = approx.real if abs(approx.imag) < 1e-15 else [approx.real, approx.imag]
        except (TypeError, ValueError):
            pass
    return out


def number_result(value: float, **extra: Any) -> dict[str, Any]:
    return {"kind": "number", "value": float(value), **extra}


def from_result(result: dict[str, Any], assumptions: dict[str, str] | None = None) -> sp.Basic:
    """Rebuild a SymPy object from a stored result (used by checkers)."""
    kind = result.get("kind")
    if kind == "number":
        return sp.Float(result["value"])
    if kind == "expr":
        return parse(result["value"], assumptions)
    raise ParseError(f"cannot rebuild a {kind!r} result as an expression")
