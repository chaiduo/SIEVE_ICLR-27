#!/usr/bin/env python3
"""Validate new comparison scores against a labeled reference campaign."""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any


MATCH_FIELDS = (
    "sample_uid",
    "orig_id",
    "semantic_group_id",
    "split",
    "run_index",
    "injected",
    "clean_answer",
    "pred_answer",
    "fault",
)
REQUIRED_SCORE_FIELDS = (
    "ranger_score",
    "drdna_score",
    "has_non_finite",
    "steps_observed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


FAULT_IDENTITY_FIELDS = (
    "mode",
    "module",
    "component",
    "layer_index",
    "op_type",
    "forward",
    "dtype",
    "idx",
    "bit_positions",
    "bit_categories",
    "before",
    "after",
)


def normalized(value: Any, *, field: str) -> str:
    if field == "fault" and isinstance(value, dict):
        value = {key: value.get(key) for key in FAULT_IDENTITY_FIELDS}
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def main() -> int:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    reference_path = args.reference.resolve()
    rows = 0
    with candidate_path.open(encoding="utf-8") as candidate_stream, reference_path.open(
        encoding="utf-8"
    ) as reference_stream:
        for line_number, pair in enumerate(
            zip_longest(candidate_stream, reference_stream), start=1
        ):
            candidate_line, reference_line = pair
            if candidate_line is None or reference_line is None:
                raise ValueError(
                    "Candidate/reference row counts differ at line "
                    f"{line_number}"
                )
            candidate = json.loads(candidate_line)
            reference = json.loads(reference_line)
            missing = [
                field for field in REQUIRED_SCORE_FIELDS if field not in candidate
            ]
            if missing:
                raise ValueError(
                    f"Candidate line {line_number} misses score fields: {missing}"
                )
            for field in MATCH_FIELDS:
                if normalized(candidate.get(field), field=field) != normalized(
                    reference.get(field), field=field
                ):
                    raise ValueError(
                        f"Campaign mismatch at line {line_number}, field={field}, "
                        f"candidate_uid={candidate.get('sample_uid')}, "
                        f"reference_uid={reference.get('sample_uid')}"
                    )
            rows += 1
            if rows % 5000 == 0:
                print(f"[validate-comparison] rows={rows}", flush=True)

    result = {
        "status": "matched",
        "rows": rows,
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "matched_fields": list(MATCH_FIELDS),
        "required_score_fields": list(REQUIRED_SCORE_FIELDS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
