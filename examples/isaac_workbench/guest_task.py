import ast
import json
import resource
import signal
from pathlib import Path
from types import FrameType

ALLOWED = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)
FUNCTIONS = {
    "sum": sum,
    "max": max,
    "min": min,
    "len": len,
    "abs": abs,
    "sorted": sorted,
    "reversed": reversed,
    "list": list,
    "str": str,
}


def timeout_handler(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("expression exceeded one second")


def grade(expression: str, task: dict) -> int:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        assert isinstance(node, ALLOWED), f"unsupported syntax: {type(node).__name__}"
        if isinstance(node, ast.Name):
            assert not node.id.startswith("_"), "private names are not available"
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
    compiled = compile(tree, "<physical-keyboard>", "eval")
    passed = 0
    for case in task["cases"]:
        signal.alarm(1)
        try:
            value = eval(compiled, {"__builtins__": FUNCTIONS}, case["inputs"])
            passed += int(value == case["expected"])
        finally:
            signal.alarm(0)
    return passed


def main() -> None:
    task = json.loads(Path("/task.json").read_text())
    resource.setrlimit(resource.RLIMIT_AS, (192 * 1024**2, 192 * 1024**2))
    signal.signal(signal.SIGALRM, timeout_handler)
    with open("/dev/ttyS0", "w", buffering=1) as serial:
        print("\033[2J\033[HCarrot / Linux programming exercise", flush=True)
        print(task["name"] + "\n" + task["instruction"], flush=True)
        print("Submit with ENTER. Backspace edits. No host command execution.", flush=True)
        serial.write("WORKBENCH_READY\n")
        attempt = 0
        while True:
            expression = input(">>> ")
            attempt += 1
            result = {
                "episode": task["episode"],
                "task": task["name"],
                "attempt": attempt,
                "passed": 0,
                "total": len(task["cases"]),
            }
            try:
                result["passed"] = grade(expression, task)
            except (
                SyntaxError,
                AssertionError,
                NameError,
                TypeError,
                ValueError,
                ZeroDivisionError,
                IndexError,
                KeyError,
                TimeoutError,
                MemoryError,
            ) as error:
                print(type(error).__name__ + ": " + str(error)[:120], flush=True)
            result["success"] = result["passed"] == result["total"]
            print(f"Tests: {result['passed']}/{result['total']}", flush=True)
            serial.write("WORKBENCH_GRADE " + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
