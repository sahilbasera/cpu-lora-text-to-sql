"""Create deterministic English-to-SQL datasets and SQLite fixtures."""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
from pathlib import Path
from typing import Any


SEED = 20260830
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"

SCHEMA = """CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    country TEXT,
    amount REAL
);"""

SPLITS = ("pilot", "train", "validation", "test")
EXAMPLES_PER_FAMILY = {
    "pilot": 6,
    "train": 28,
    "validation": 8,
    "test": 12,
}
GENERALIZATION_TYPES = {
    "pilot": "pilot_calibration",
    "train": "training",
    "validation": "unseen_phrasing_validation",
    "test": "unseen_phrasing_test",
}

FIELD_VALUES = {
    "country": ("India", "Brazil", "Canada", "France", "Japan", "Kenya", "Mexico", "Spain"),
    "amount": (25, 50, 75, 100, 125, 150, 200, 500),
    "low": (25, 50, 75, 100),
    "high": (125, 150, 200, 500),
}


def templates(*items: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """Return template definitions while keeping the family declaration readable."""
    return items


FAMILIES: dict[str, dict[str, Any]] = {
    "country_eq": {
        "fields": ("country",),
        "filters": (("country", "=", "country"),),
        "templates": {
            "pilot": templates(
                ("pilot_show_from", "Show orders from {country}."),
                ("pilot_list_country", "List orders placed in {country}."),
            ),
            "train": templates(
                ("train_get_from", "Get orders from {country}."),
                ("train_find_country", "Find orders in {country}."),
                ("train_return_from", "Return every order from {country}."),
                ("train_country_orders", "Which orders came from {country}?"),
            ),
            "validation": templates(
                ("validation_originating", "Show purchases originating in {country}."),
                ("validation_only_country", "Give me only the {country} orders."),
            ),
            "test": templates(
                ("test_country_purchases", "What purchases were made in {country}?"),
                ("test_located_country", "Retrieve orders whose country is {country}."),
                ("test_belong_country", "List all orders belonging to {country}."),
            ),
        },
    },
    "amount_gt": {
        "fields": ("amount",),
        "filters": (("amount", ">", "amount"),),
        "templates": {
            "pilot": templates(
                ("pilot_more_than", "Show orders worth more than {amount}."),
                ("pilot_over", "List orders over {amount}."),
            ),
            "train": templates(
                ("train_above", "Find orders above {amount}."),
                ("train_greater", "Get orders with an amount greater than {amount}."),
                ("train_exceed", "Return orders that exceed {amount}."),
                ("train_cost_more", "Which orders cost more than {amount}?"),
            ),
            "validation": templates(
                ("validation_surpass", "Show purchases whose value surpasses {amount}."),
                ("validation_higher", "List orders higher than {amount} in value."),
            ),
            "test": templates(
                ("test_exceeding", "Retrieve purchases exceeding {amount}."),
                ("test_beyond", "What orders have amounts beyond {amount}?"),
                ("test_strictly_above", "Find orders strictly above {amount}."),
            ),
        },
    },
    "amount_ge": {
        "fields": ("amount",),
        "filters": (("amount", ">=", "amount"),),
        "templates": {
            "pilot": templates(
                ("pilot_at_least", "Show orders worth at least {amount}."),
                ("pilot_minimum", "List orders with a minimum amount of {amount}."),
            ),
            "train": templates(
                ("train_no_less", "Find orders worth no less than {amount}."),
                ("train_or_more", "Get orders for {amount} or more."),
                ("train_amount_ge", "Return orders whose amount is at least {amount}."),
                ("train_reach", "Which orders reach or exceed {amount}?"),
            ),
            "validation": templates(
                ("validation_floor", "Show purchases with a value floor of {amount}."),
                ("validation_min_value", "List orders valued {amount} and upward."),
            ),
            "test": templates(
                ("test_not_below", "Retrieve orders not below {amount}."),
                ("test_meet_minimum", "What purchases meet a minimum of {amount}?"),
                ("test_equal_or_above", "Find orders with amounts equal to or above {amount}."),
            ),
        },
    },
    "amount_lt": {
        "fields": ("amount",),
        "filters": (("amount", "<", "amount"),),
        "templates": {
            "pilot": templates(
                ("pilot_less_than", "Show orders worth less than {amount}."),
                ("pilot_under", "List orders under {amount}."),
            ),
            "train": templates(
                ("train_below", "Find orders below {amount}."),
                ("train_lower", "Get orders with an amount lower than {amount}."),
                ("train_cost_less", "Return orders that cost less than {amount}."),
                ("train_fall_short", "Which orders fall short of {amount}?"),
            ),
            "validation": templates(
                ("validation_beneath", "Show purchases valued beneath {amount}."),
                ("validation_smaller", "List orders with values smaller than {amount}."),
            ),
            "test": templates(
                ("test_strictly_below", "Retrieve orders strictly below {amount}."),
                ("test_cheaper", "What purchases are cheaper than {amount}?"),
                ("test_fail_reach", "Find orders whose amount does not reach {amount}."),
            ),
        },
    },
    "amount_le": {
        "fields": ("amount",),
        "filters": (("amount", "<=", "amount"),),
        "templates": {
            "pilot": templates(
                ("pilot_at_most", "Show orders worth at most {amount}."),
                ("pilot_maximum", "List orders with a maximum amount of {amount}."),
            ),
            "train": templates(
                ("train_no_more", "Find orders worth no more than {amount}."),
                ("train_or_less", "Get orders for {amount} or less."),
                ("train_amount_le", "Return orders whose amount is at most {amount}."),
                ("train_not_exceed", "Which orders do not exceed {amount}?"),
            ),
            "validation": templates(
                ("validation_ceiling", "Show purchases with a value ceiling of {amount}."),
                ("validation_up_to", "List orders valued up to and including {amount}."),
            ),
            "test": templates(
                ("test_not_above", "Retrieve orders not above {amount}."),
                ("test_within_maximum", "What purchases stay within a maximum of {amount}?"),
                ("test_equal_or_below", "Find orders with amounts equal to or below {amount}."),
            ),
        },
    },
    "country_amount_gt": {
        "fields": ("country", "amount"),
        "filters": (("country", "=", "country"), ("amount", ">", "amount")),
        "templates": {
            "pilot": templates(
                ("pilot_from_more", "Show orders from {country} worth more than {amount}."),
                ("pilot_country_over", "List {country} orders over {amount}."),
            ),
            "train": templates(
                ("train_from_above", "Find orders from {country} above {amount}."),
                ("train_country_greater", "Get {country} orders with amounts greater than {amount}."),
                ("train_exceed_in", "Return orders exceeding {amount} in {country}."),
                ("train_cost_more_country", "Which orders from {country} cost more than {amount}?"),
            ),
            "validation": templates(
                ("validation_surpass_country", "Show purchases in {country} whose value surpasses {amount}."),
                ("validation_country_high", "List high-value {country} orders above {amount}."),
            ),
            "test": templates(
                ("test_exceeding_in", "Retrieve purchases exceeding {amount} that came from {country}."),
                ("test_country_beyond", "What {country} orders have amounts beyond {amount}?"),
                ("test_strict_country", "Find orders strictly above {amount} with country {country}."),
            ),
        },
    },
    "amount_range": {
        "fields": ("low", "high"),
        "filters": (("amount", ">=", "low"), ("amount", "<=", "high")),
        "templates": {
            "pilot": templates(
                ("pilot_between", "Show orders with amounts between {low} and {high}, inclusive."),
                ("pilot_range", "List orders with amounts from {low} through {high}, including both endpoints."),
            ),
            "train": templates(
                ("train_within", "Find orders with amounts between {low} and {high}, including both endpoints."),
                ("train_interval", "Get orders in the inclusive amount interval {low} to {high}."),
                ("train_low_high", "Return orders worth at least {low} and at most {high}."),
                ("train_bounded", "Which orders have amounts bounded by {low} and {high}, including both?"),
            ),
            "validation": templates(
                ("validation_band", "Show purchases in the closed value band {low}–{high}."),
                ("validation_floor_ceiling", "List orders with an amount floor of {low} and ceiling of {high}, both inclusive."),
            ),
            "test": templates(
                ("test_not_outside", "Retrieve orders with amounts no lower than {low} and no higher than {high}."),
                ("test_inclusive_span", "What purchases have amounts in the inclusive span {low} to {high}?"),
                ("test_limits", "Find orders whose amount stays within limits {low} and {high}, including both limits."),
            ),
        },
    },
}


def parameter_sets(fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Build every valid value combination for a query family."""
    combinations = itertools.product(*(FIELD_VALUES[field] for field in fields))
    values = [dict(zip(fields, combination, strict=True)) for combination in combinations]
    return [item for item in values if not {"low", "high"} <= item.keys() or item["low"] < item["high"]]


def sql_literal(value: Any) -> str:
    """Format one controlled Python value as a SQLite literal."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def build_sql(family: dict[str, Any], values: dict[str, Any]) -> str:
    """Create canonical SQL from a family's filter specification."""
    conditions = [
        f"{column} {operator} {sql_literal(values[value_name])}"
        for column, operator, value_name in family["filters"]
    ]
    return "SELECT * FROM orders WHERE " + " AND ".join(conditions) + ";"


def build_components(family: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Store the expected SQL structure for later diagnostic evaluation."""
    filters = [
        {"column": column, "operator": operator, "value": values[value_name]}
        for column, operator, value_name in family["filters"]
    ]
    return {
        "table": "orders",
        "projection": ["*"],
        "filters": filters,
        "connector": "AND" if len(filters) > 1 else None,
    }


def generate_split(split: str, rng: random.Random) -> list[dict[str, Any]]:
    """Generate one split by sampling only from that split's phrasing templates."""
    records: list[dict[str, Any]] = []
    for family_name, family in FAMILIES.items():
        candidates = []
        for template_id, template_text in family["templates"][split]:
            for values in parameter_sets(family["fields"]):
                candidates.append((template_id, template_text, values))

        rng.shuffle(candidates)
        requested = EXAMPLES_PER_FAMILY[split]
        if len(candidates) < requested:
            raise ValueError(f"{family_name}/{split} has only {len(candidates)} candidates")

        for index, (template_id, template_text, values) in enumerate(candidates[:requested], start=1):
            records.append(
                {
                    "id": f"{split}_{family_name}_{index:03d}",
                    "split": split,
                    "query_family": family_name,
                    "template_family": template_id,
                    "generalization_type": GENERALIZATION_TYPES[split],
                    "values": values,
                    "components": build_components(family, values),
                    "schema": SCHEMA,
                    "question": template_text.format(**values),
                    "sql": build_sql(family, values),
                }
            )

    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one JSON object per line using stable key ordering."""
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def create_database(path: Path, rows: list[tuple[str, float]]) -> None:
    """Create one disposable SQLite database from country/amount rows."""
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(SCHEMA)
        connection.executemany("INSERT INTO orders (country, amount) VALUES (?, ?)", rows)


def create_fixtures() -> list[Path]:
    """Create databases that expose boundary, country, and empty-result mistakes."""
    countries = FIELD_VALUES["country"]
    thresholds = FIELD_VALUES["amount"]

    boundary_rows = [
        (country, float(threshold + offset))
        for country in countries
        for threshold in thresholds
        for offset in (-0.5, 0, 0.5)
    ]
    mixed_rows = [
        (country, float(amount))
        for country, amount in zip(
            itertools.cycle(countries),
            (10, 25, 49.5, 50, 75, 100, 125, 150, 200, 500, 750, 1000),
        )
    ]
    sparse_rows = [
        ("India", 25.0),
        ("India", 500.0),
        ("Brazil", 100.0),
        ("France", 150.0),
        ("Japan", 75.0),
        ("Kenya", 200.0),
        ("Mexico", 50.0),
        ("Spain", 125.0),
    ]

    fixture_rows = {
        "orders_boundaries.db": boundary_rows,
        "orders_mixed.db": mixed_rows,
        "orders_sparse.db": sparse_rows,
    }
    paths = []
    for filename, rows in fixture_rows.items():
        path = FIXTURES_DIR / filename
        create_database(path, rows)
        paths.append(path)
    return paths


def validate(records_by_split: dict[str, list[dict[str, Any]]], fixture_paths: list[Path]) -> None:
    """Fail fast on leakage, malformed SQL, duplicate examples, or useless references."""
    all_records = [record for records in records_by_split.values() for record in records]

    ids = [record["id"] for record in all_records]
    questions = [record["question"].casefold() for record in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs found")
    if len(questions) != len(set(questions)):
        raise ValueError("Duplicate questions found across splits")

    template_splits: dict[str, set[str]] = {}
    for record in all_records:
        template_splits.setdefault(record["template_family"], set()).add(record["split"])
    leaked_templates = {name: splits for name, splits in template_splits.items() if len(splits) > 1}
    if leaked_templates:
        raise ValueError(f"Template leakage found: {leaked_templates}")

    for record in all_records:
        if not record["sql"].startswith("SELECT ") or record["sql"].count(";") != 1:
            raise ValueError(f"Unsafe reference SQL in {record['id']}")

        produced_rows = 0
        for fixture_path in fixture_paths:
            with sqlite3.connect(fixture_path) as connection:
                produced_rows += len(connection.execute(record["sql"]).fetchall())
        if produced_rows == 0:
            raise ValueError(f"Reference query never returns a row: {record['id']}")


def write_manifest(records_by_split: dict[str, list[dict[str, Any]]], fixture_paths: list[Path]) -> None:
    """Record the generation settings needed to reproduce this dataset."""
    manifest = {
        "seed": SEED,
        "schema": SCHEMA,
        "splits": {split: len(records) for split, records in records_by_split.items()},
        "query_families": sorted(FAMILIES),
        "fixtures": [path.name for path in fixture_paths],
    }
    path = DATA_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Generate, validate, and save every dataset artifact."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    records_by_split = {split: generate_split(split, rng) for split in SPLITS}
    fixture_paths = create_fixtures()
    validate(records_by_split, fixture_paths)

    for split, records in records_by_split.items():
        write_jsonl(DATA_DIR / f"{split}.jsonl", records)
    write_manifest(records_by_split, fixture_paths)

    print("Prepared deterministic English-to-SQL data:")
    for split, records in records_by_split.items():
        print(f"  {split:10} {len(records):3} examples")
    print(f"  fixtures   {len(fixture_paths):3} SQLite databases")
    print(f"  output       {DATA_DIR}")


if __name__ == "__main__":
    main()
