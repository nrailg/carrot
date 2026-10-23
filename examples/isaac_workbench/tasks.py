from dataclasses import dataclass


@dataclass(frozen=True)
class CodingTask:
    name: str
    instruction: str
    cases: list[dict]
    demonstration: str

    def guest_spec(self, episode: str) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "cases": self.cases,
            "episode": episode,
        }


CODING_TASKS = {
    "sum_two": CodingTask(
        "sum_two",
        "Return the sum of integers a and b. Write one Python expression.",
        [
            {"inputs": {"a": 17, "b": 25}, "expected": 42},
            {"inputs": {"a": -9, "b": 4}, "expected": -5},
            {"inputs": {"a": 0, "b": 0}, "expected": 0},
        ],
        "a+b\n",
    ),
    "max_list": CodingTask(
        "max_list",
        "Return the largest integer in the nonempty list values.",
        [
            {"inputs": {"values": [3, 9, 2]}, "expected": 9},
            {"inputs": {"values": [-7, -2, -9]}, "expected": -2},
            {"inputs": {"values": [4]}, "expected": 4},
        ],
        "max(values)\n",
    ),
    "reverse_text": CodingTask(
        "reverse_text",
        "Reverse string text, including spaces. Write one Python expression.",
        [
            {"inputs": {"text": "robot"}, "expected": "tobor"},
            {"inputs": {"text": "a b"}, "expected": "b a"},
            {"inputs": {"text": ""}, "expected": ""},
        ],
        "text[::-1]\n",
    ),
}
