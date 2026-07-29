"""Shared LLM response parser for synthesis generators.

This module exposes a single public function that turns a raw LLM response into
a list of plain ``{"question": ..., "answer": ...}`` dictionaries. It performs
no validation and constructs no domain object; callers decide which entries to
keep and how to wrap them.
"""

from __future__ import annotations

import json
import re


def parse_question_answer_response(raw_response: str) -> list[dict[str, str]]:
    """Parse a raw LLM response into raw question/answer pairs.

    Applies the same five-tier fallback used by both question and adversarial
    generators:

    1. A single JSON object.
    2. A JSON array of objects.
    3. Individual JSON objects found anywhere in the text.
    4. A strict regex matching ``{"question": "...", "answer": "..."}``.
    5. A loose fallback regex pairing standalone ``"question"`` and
       ``"answer"`` values.

    The returned dictionaries have string ``question`` and ``answer`` values.
    Empty strings are preserved so that callers can apply their own validation
    rules (e.g. adversarial generators may keep empty answers).

    Args:
        raw_response: The raw text returned by the LLM.

    Returns:
        A list of parsed question/answer dictionaries. May be empty if no pairs
        could be extracted.
    """
    parsed: list[dict[str, str]] = []

    # Clean code blocks and normalize text
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    # Normalize Python-style single quotes to JSON double quotes,
    # but only when text uses Python-style dict formatting (single-quoted keys),
    # to avoid breaking apostrophes in valid JSON (e.g. "company's").
    has_python_style = bool(re.search(r"\{\s*'[^']*'\s*:", text))
    if has_python_style:
        text = text.replace("'", '"')

    # Strategy 0: Try parsing as a single JSON object (non-array response)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            parsed.append(
                {
                    "question": data.get("question", ""),
                    "answer": data.get("answer", ""),
                }
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 1: Try to extract JSON array and parse it
    try:
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx > start_idx:
            array_text = text[start_idx : end_idx + 1]
        else:
            array_text = text

        # Clean the JSON
        array_text = re.sub(r",\s*([}\]])", r"\1", array_text)
        array_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", array_text)

        data = json.loads(array_text)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parsed.append(
                        {
                            "question": item.get("question", ""),
                            "answer": item.get("answer", ""),
                        }
                    )
        if parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract individual JSON objects and parse them
    try:
        obj_pattern = re.compile(r"\{[^{}]+\}", re.DOTALL)
        objects = obj_pattern.findall(text)

        for obj_str in objects:
            try:
                # Clean the object
                obj_str = re.sub(r",\s*([}\]])", r"\1", obj_str)
                obj_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", obj_str)

                item = json.loads(obj_str)
                if isinstance(item, dict):
                    parsed.append(
                        {
                            "question": item.get("question", ""),
                            "answer": item.get("answer", ""),
                        }
                    )
            except (json.JSONDecodeError, ValueError):
                continue

        if parsed:
            return parsed
    except Exception:
        pass

    # Strategy 3: Extract question-answer pairs using regex
    qa_pattern = re.compile(
        r'\{\s*"question"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]+)"\s*\}',
        re.IGNORECASE,
    )
    matches = qa_pattern.findall(text)

    for question, answer in matches:
        parsed.append({"question": question, "answer": answer})

    if parsed:
        return parsed

    # Strategy 4: Try to find any question-answer patterns
    question_pattern = re.compile(r'"question"\s*:\s*"([^"]+)"', re.IGNORECASE)
    answer_pattern = re.compile(r'"answer"\s*:\s*"([^"]+)"', re.IGNORECASE)

    questions = question_pattern.findall(text)
    answers = answer_pattern.findall(text)

    for question, answer in zip(questions, answers, strict=False):
        parsed.append({"question": question, "answer": answer})

    return parsed
