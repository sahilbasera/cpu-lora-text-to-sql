"""Independent tests for the generated English-to-SQL dataset."""

from __future__ import annotations

import json
import operator
import sqlite3
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import prepare_data


COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "=": operator.eq,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

EXPECTED_FILTERS = {
    "country_eq": (("country", "=", "country"),),
    "amount_gt": (("amount", ">", "amount"),),
    "amount_ge": (("amount", ">=", "amount"),),
    "amount_lt": (("amount", "<", "amount"),),
    "amount_le": (("amount", "<=", "amount"),),
    "country_amount_gt": (("country", "=", "country"), ("amount", ">", "amount")),
    "amount_range": (("amount", ">=", "low"), ("amount", "<=", "high")),
}

GOLDEN_SQL_CASES = {
    "country_eq": (
        {"country": "India"},
        "SELECT * FROM orders WHERE country = 'India';",
    ),
    "amount_gt": (
        {"amount": 100},
        "SELECT * FROM orders WHERE amount > 100;",
    ),
    "amount_ge": (
        {"amount": 100},
        "SELECT * FROM orders WHERE amount >= 100;",
    ),
    "amount_lt": (
        {"amount": 100},
        "SELECT * FROM orders WHERE amount < 100;",
    ),
    "amount_le": (
        {"amount": 100},
        "SELECT * FROM orders WHERE amount <= 100;",
    ),
    "country_amount_gt": (
        {"country": "India", "amount": 100},
        "SELECT * FROM orders WHERE country = 'India' AND amount > 100;",
    ),
    "amount_range": (
        {"low": 50, "high": 150},
        "SELECT * FROM orders WHERE amount >= 50 AND amount <= 150;",
    ),
}

REQUIRED_RECORD_KEYS = {
    "id",
    "split",
    "query_family",
    "template_family",
    "generalization_type",
    "values",
    "components",
    "schema",
    "question",
    "sql",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file without using code from the generator."""
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_database_rows(path: Path) -> list[tuple[int, str, float]]:
    """Read raw order rows from one fixture."""
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT id, country, amount FROM orders").fetchall()


def python_oracle(
    rows: list[tuple[int, str, float]], components: dict[str, Any]
) -> list[tuple[int, str, float]]:
    """Apply component filters in Python rather than evaluating SQL."""
    column_indexes = {"id": 0, "country": 1, "amount": 2}

    def row_matches(row: tuple[int, str, float]) -> bool:
        outcomes = []
        for filter_spec in components["filters"]:
            comparison = COMPARISONS[filter_spec["operator"]]
            actual_value = row[column_indexes[filter_spec["column"]]]
            outcomes.append(comparison(actual_value, filter_spec["value"]))
        return all(outcomes)

    return [row for row in rows if row_matches(row)]


def execute(path: Path, sql: str) -> list[tuple[int, str, float]]:
    """Execute trusted test SQL and return all result rows."""
    with sqlite3.connect(path) as connection:
        return connection.execute(sql).fetchall()


class PreparedDataTests(unittest.TestCase):
    """Verify dataset integrity, semantics, and fixture discrimination."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records_by_split = {
            split: load_jsonl(prepare_data.DATA_DIR / f"{split}.jsonl")
            for split in prepare_data.SPLITS
        }
        cls.fixture_paths = sorted(prepare_data.FIXTURES_DIR.glob("*.db"))
        cls.rows_by_fixture = {
            path: load_database_rows(path) for path in cls.fixture_paths
        }

    def test_manifest_and_split_counts(self) -> None:
        manifest_path = prepare_data.DATA_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["seed"], prepare_data.SEED)
        self.assertEqual(manifest["query_families"], sorted(EXPECTED_FILTERS))
        self.assertEqual(
            manifest["splits"],
            {split: len(records) for split, records in self.records_by_split.items()},
        )
        self.assertEqual(len(self.fixture_paths), 3)

        for split, records in self.records_by_split.items():
            family_counts = Counter(record["query_family"] for record in records)
            self.assertEqual(
                family_counts,
                Counter(
                    {
                        family_name: prepare_data.EXAMPLES_PER_FAMILY[split]
                        for family_name in EXPECTED_FILTERS
                    }
                ),
            )

    def test_records_have_required_fields_and_no_duplicates(self) -> None:
        all_records = [
            record for records in self.records_by_split.values() for record in records
        ]
        for record in all_records:
            self.assertEqual(set(record), REQUIRED_RECORD_KEYS, record["id"])
            self.assertNotIn("{", record["question"], record["id"])
            self.assertEqual(record["schema"], prepare_data.SCHEMA, record["id"])

        self.assertEqual(len({record["id"] for record in all_records}), len(all_records))
        self.assertEqual(
            len({record["question"].casefold() for record in all_records}),
            len(all_records),
        )

    def test_template_families_do_not_cross_splits(self) -> None:
        templates_by_split = {
            split: {
                template_id
                for family in prepare_data.FAMILIES.values()
                for template_id, _ in family["templates"][split]
            }
            for split in prepare_data.SPLITS
        }
        for left_index, left_split in enumerate(prepare_data.SPLITS):
            for right_split in prepare_data.SPLITS[left_index + 1 :]:
                self.assertTrue(
                    templates_by_split[left_split].isdisjoint(templates_by_split[right_split]),
                    f"Template leakage between {left_split} and {right_split}",
                )

    def test_family_components_match_independent_specification(self) -> None:
        for records in self.records_by_split.values():
            for record in records:
                expected = EXPECTED_FILTERS[record["query_family"]]
                actual_filters = record["components"]["filters"]

                self.assertEqual(record["components"]["table"], "orders", record["id"])
                self.assertEqual(record["components"]["projection"], ["*"], record["id"])
                self.assertEqual(len(actual_filters), len(expected), record["id"])

                for actual, (column, comparison, value_name) in zip(
                    actual_filters, expected, strict=True
                ):
                    self.assertEqual(actual["column"], column, record["id"])
                    self.assertEqual(actual["operator"], comparison, record["id"])
                    self.assertEqual(actual["value"], record["values"][value_name], record["id"])

                expected_connector = "AND" if len(expected) > 1 else None
                self.assertEqual(
                    record["components"]["connector"], expected_connector, record["id"]
                )

    def test_sql_builder_against_golden_cases(self) -> None:
        for family_name, (values, expected_sql) in GOLDEN_SQL_CASES.items():
            with self.subTest(family=family_name):
                actual_sql = prepare_data.build_sql(
                    prepare_data.FAMILIES[family_name], values
                )
                self.assertEqual(actual_sql, expected_sql)

    def test_sql_results_match_independent_python_oracle(self) -> None:
        for records in self.records_by_split.values():
            for record in records:
                for fixture_path in self.fixture_paths:
                    with self.subTest(record=record["id"], fixture=fixture_path.name):
                        expected_rows = python_oracle(
                            self.rows_by_fixture[fixture_path], record["components"]
                        )
                        sql_rows = execute(fixture_path, record["sql"])
                        self.assertEqual(Counter(sql_rows), Counter(expected_rows))

    def test_boundary_fixture_detects_common_mutations(self) -> None:
        boundary_path = prepare_data.FIXTURES_DIR / "orders_boundaries.db"

        for amount in prepare_data.FIELD_VALUES["amount"]:
            with self.subTest(amount=amount, mutation="> versus >="):
                strict = execute(boundary_path, f"SELECT * FROM orders WHERE amount > {amount};")
                inclusive = execute(boundary_path, f"SELECT * FROM orders WHERE amount >= {amount};")
                self.assertNotEqual(Counter(strict), Counter(inclusive))

            with self.subTest(amount=amount, mutation="< versus <="):
                strict = execute(boundary_path, f"SELECT * FROM orders WHERE amount < {amount};")
                inclusive = execute(boundary_path, f"SELECT * FROM orders WHERE amount <= {amount};")
                self.assertNotEqual(Counter(strict), Counter(inclusive))

        correct_combination = execute(
            boundary_path,
            "SELECT * FROM orders WHERE country = 'India' AND amount > 100;",
        )
        missing_country = execute(boundary_path, "SELECT * FROM orders WHERE amount > 100;")
        missing_amount = execute(
            boundary_path, "SELECT * FROM orders WHERE country = 'India';"
        )
        self.assertNotEqual(Counter(correct_combination), Counter(missing_country))
        self.assertNotEqual(Counter(correct_combination), Counter(missing_amount))

        inclusive_range = execute(
            boundary_path,
            "SELECT * FROM orders WHERE amount >= 50 AND amount <= 150;",
        )
        wrong_lower_edge = execute(
            boundary_path,
            "SELECT * FROM orders WHERE amount > 50 AND amount <= 150;",
        )
        wrong_upper_edge = execute(
            boundary_path,
            "SELECT * FROM orders WHERE amount >= 50 AND amount < 150;",
        )
        self.assertNotEqual(Counter(inclusive_range), Counter(wrong_lower_edge))
        self.assertNotEqual(Counter(inclusive_range), Counter(wrong_upper_edge))


if __name__ == "__main__":
    unittest.main(verbosity=2)
