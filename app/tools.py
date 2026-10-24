"""
tools.py - Finance calculator tools the model can call (OpenAI "function calling").

Why: LLMs are good with words but unreliable with arithmetic (e.g. (1 + 0.015)^12 - 1).
For calculation questions we give gpt-4o-mini a list of exact Python functions. The model
decides which function to call and with which numbers; OUR code computes the result; the
model then writes the answer using that exact result (and still cites the textbook for
the method). The UI shows every calculation that was made, so it can be checked.

Each tool = a Python function + a JSON schema that describes it to the model.
"""

import ast
import json
import math
import operator

# ---------------------------------------------------------------------------
# The functions (pure Python, no API)
# ---------------------------------------------------------------------------


def rule_of_72(rate_percent: float) -> dict:
    """Approximate years for money to double at rate_percent per period."""
    years = 72 / rate_percent
    exact = math.log(2) / math.log(1 + rate_percent / 100)
    return {"years_rule_of_72": round(years, 2), "years_exact": round(exact, 2)}


def future_value(present_value: float, rate_percent: float, years: float,
                 compounds_per_year: int = 1) -> dict:
    m = compounds_per_year
    fv = present_value * (1 + rate_percent / 100 / m) ** (m * years)
    return {"future_value": round(fv, 2)}


def present_value(future_value: float, rate_percent: float, years: float,
                  compounds_per_year: int = 1) -> dict:
    m = compounds_per_year
    pv = future_value / (1 + rate_percent / 100 / m) ** (m * years)
    return {"present_value": round(pv, 2)}


def effective_annual_rate(rate_percent: float, periods_per_year: int,
                          rate_is_per_period: bool = False) -> dict:
    """EAR from a nominal annual rate (APR) or from a periodic rate (e.g. 1.5% per month)."""
    periodic = rate_percent / 100 if rate_is_per_period else rate_percent / 100 / periods_per_year
    ear = (1 + periodic) ** periods_per_year - 1
    return {"periodic_rate_percent": round(periodic * 100, 4),
            "effective_annual_rate_percent": round(ear * 100, 4)}


def npv(rate_percent: float, cash_flows: list[float]) -> dict:
    """cash_flows[0] is at time 0 (usually negative: the initial investment)."""
    r = rate_percent / 100
    value = sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
    return {"npv": round(value, 2), "decision": "accept (NPV > 0)" if value > 0 else "reject (NPV <= 0)"}


def irr(cash_flows: list[float]) -> dict:
    """Internal rate of return by bisection (the rate where NPV = 0)."""
    def f(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
    lo, hi = -0.99, 10.0
    if f(lo) * f(hi) > 0:
        return {"irr_percent": None, "note": "no single IRR found for these cash flows"}
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return {"irr_percent": round(mid * 100, 4)}


def payback_period(initial_investment: float, cash_flows: list[float]) -> dict:
    """Years until cumulative cash inflows repay the initial investment (with fraction of year)."""
    remaining = abs(initial_investment)
    for year, cf in enumerate(cash_flows, start=1):
        if cf >= remaining:
            return {"payback_years": round(year - 1 + remaining / cf, 3)}
        remaining -= cf
    return {"payback_years": None, "note": "investment not recovered within the cash flows given"}


def loan_payment(principal: float, annual_rate_percent: float, years: float,
                 payments_per_year: int = 12) -> dict:
    r = annual_rate_percent / 100 / payments_per_year
    n = int(round(years * payments_per_year))
    pmt = principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)
    return {"payment": round(pmt, 2), "number_of_payments": n,
            "total_paid": round(pmt * n, 2), "total_interest": round(pmt * n - principal, 2)}


def savings_plan(goal_amount: float, annual_rate_percent: float, years: float,
                 current_savings: float = 0.0, payments_per_year: int = 12) -> dict:
    """Payment to save each period to reach a goal (current savings keep growing too)."""
    r = annual_rate_percent / 100 / payments_per_year
    n = int(round(years * payments_per_year))
    fv_current = current_savings * (1 + r) ** n
    missing = max(0.0, goal_amount - fv_current)
    pmt = missing / n if r == 0 else missing * r / ((1 + r) ** n - 1)
    return {"payment_per_period": round(pmt, 2), "number_of_payments": n,
            "current_savings_grow_to": round(fv_current, 2), "total_deposited": round(pmt * n, 2),
            "interest_earned": round(goal_amount - current_savings - pmt * n, 2) if goal_amount > fv_current else None}


def savings_growth(payment_per_period: float, annual_rate_percent: float, years: float,
                   current_savings: float = 0.0, payments_per_year: int = 12) -> dict:
    """Future value of regular savings (ordinary annuity) plus current savings."""
    r = annual_rate_percent / 100 / payments_per_year
    n = int(round(years * payments_per_year))
    fv_payments = payment_per_period * n if r == 0 else payment_per_period * ((1 + r) ** n - 1) / r
    fv_current = current_savings * (1 + r) ** n
    total = fv_payments + fv_current
    return {"future_value": round(total, 2), "total_deposited": round(payment_per_period * n + current_savings, 2),
            "interest_earned": round(total - payment_per_period * n - current_savings, 2)}


def financial_ratio(name: str, numerator: float, denominator: float) -> dict:
    """Any ratio: current ratio, debt ratio, P/E ... = numerator / denominator."""
    return {"ratio": name, "value": round(numerator / denominator, 4)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos, ast.Mod: operator.mod}


def calculator(expression: str) -> dict:
    """Safe arithmetic with simpleeval (course spec); our AST evaluator is the fallback when
    simpleeval is not installed. Never Python eval()."""
    try:
        from simpleeval import simple_eval
        value = simple_eval(str(expression).replace("^", "**"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"expression": expression, "result": round(value, 6)}
        raise ValueError("the result is not a number")
    except ImportError:
        pass

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** % ( ) are allowed")
    return {"expression": expression, "result": round(ev(ast.parse(expression, mode="eval")), 6)}


FUNCTIONS = {f.__name__: f for f in [rule_of_72, future_value, present_value, effective_annual_rate,
                                     npv, irr, payback_period, loan_payment, savings_plan,
                                     savings_growth, financial_ratio, calculator]}

# ---------------------------------------------------------------------------
# JSON schemas shown to the model
# ---------------------------------------------------------------------------
_num = {"type": "number"}
_int = {"type": "integer"}
_flows = {"type": "array", "items": {"type": "number"}}


def _tool(name, description, props, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOL_SCHEMAS = [
    _tool("rule_of_72", "Years for money to double at a given % rate (rule of 72 and exact).",
          {"rate_percent": _num}, ["rate_percent"]),
    _tool("future_value", "Future value of a single amount with compound interest.",
          {"present_value": _num, "rate_percent": _num, "years": _num, "compounds_per_year": _int},
          ["present_value", "rate_percent", "years"]),
    _tool("present_value", "Present value of a single future amount.",
          {"future_value": _num, "rate_percent": _num, "years": _num, "compounds_per_year": _int},
          ["future_value", "rate_percent", "years"]),
    _tool("effective_annual_rate", "Effective annual rate from an APR (nominal) or from a periodic rate "
          "such as 1.5% per month (then rate_is_per_period=true).",
          {"rate_percent": _num, "periods_per_year": _int, "rate_is_per_period": {"type": "boolean"}},
          ["rate_percent", "periods_per_year"]),
    _tool("npv", "Net present value; cash_flows[0] is the time-0 flow (negative investment).",
          {"rate_percent": _num, "cash_flows": _flows}, ["rate_percent", "cash_flows"]),
    _tool("irr", "Internal rate of return of a cash-flow series (cash_flows[0] at time 0).",
          {"cash_flows": _flows}, ["cash_flows"]),
    _tool("payback_period", "Payback period in years for an initial investment and yearly inflows.",
          {"initial_investment": _num, "cash_flows": _flows}, ["initial_investment", "cash_flows"]),
    _tool("loan_payment", "Periodic payment, total paid and total interest of an amortising loan.",
          {"principal": _num, "annual_rate_percent": _num, "years": _num, "payments_per_year": _int},
          ["principal", "annual_rate_percent", "years"]),
    _tool("savings_plan", "Amount to save each period to reach a savings goal in a number of years "
          "(current savings also grow at the same rate).",
          {"goal_amount": _num, "annual_rate_percent": _num, "years": _num, "current_savings": _num,
           "payments_per_year": _int}, ["goal_amount", "annual_rate_percent", "years"]),
    _tool("savings_growth", "Future value of regular savings (e.g. monthly) plus current savings.",
          {"payment_per_period": _num, "annual_rate_percent": _num, "years": _num, "current_savings": _num,
           "payments_per_year": _int}, ["payment_per_period", "annual_rate_percent", "years"]),
    _tool("financial_ratio", "Compute a named ratio = numerator / denominator (current ratio, etc.).",
          {"name": {"type": "string"}, "numerator": _num, "denominator": _num},
          ["name", "numerator", "denominator"]),
    _tool("calculator", "Exact arithmetic on an expression with + - * / ** % and parentheses.",
          {"expression": {"type": "string"}}, ["expression"]),
]


def run_tool(name: str, arguments_json: str) -> dict:
    """Execute one tool call requested by the model. Errors are returned, never raised."""
    try:
        args = json.loads(arguments_json or "{}")
        if name not in FUNCTIONS:
            return {"error": f"unknown tool {name}"}
        return FUNCTIONS[name](**args)
    except Exception as exc:                      # noqa: BLE001
        return {"error": str(exc)}
