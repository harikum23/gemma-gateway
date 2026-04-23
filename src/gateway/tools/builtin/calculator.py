from __future__ import annotations

import ast
import operator
from typing import Any

# Allowed binary operators
_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNOPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOPS:
            raise ValueError(f"unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Div and right == 0:
            raise ValueError("division by zero")
        if op_type is ast.FloorDiv and right == 0:
            raise ValueError("division by zero")
        if op_type is ast.Mod and right == 0:
            raise ValueError("modulo by zero")
        return _BINOPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNOPS:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        return _UNOPS[op_type](_eval_node(node.operand))
    raise ValueError(f"unsafe expression node: {type(node).__name__}")


async def run(args: dict, settings: Any, redis_client: Any = None) -> str:
    expression: str = args.get("expression", "").strip()
    if not expression:
        return "error: missing 'expression' argument"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        # Return int if whole number, else float
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ValueError as exc:
        return f"error: {exc}"
    except SyntaxError as exc:
        return f"error: invalid syntax: {exc}"
    except Exception as exc:
        return f"error: {exc}"
