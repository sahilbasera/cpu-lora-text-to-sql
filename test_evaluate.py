"""Challenge the evaluator with known correct, incorrect, and forbidden queries."""

import sqlite3
import unittest

from evaluate import ROOT, execute_readonly, score_sql


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.paths = list((ROOT / "data" / "fixtures").glob("*.db"))
        self.sql = "SELECT * FROM orders WHERE country = 'France' AND amount > 100;"
        self.references = {path: execute_readonly(path, self.sql) for path in self.paths}

    def test_equivalent_sql_passes_without_exact_string_match(self):
        equivalent = "SELECT id, country, amount FROM orders WHERE amount > 100 AND country = 'France'"
        self.assertTrue(score_sql(equivalent, self.references)["correct"])

    def test_wrong_conditions_and_projection_fail(self):
        for wrong in (
            self.sql.replace(">", ">="), self.sql.replace("France", "Spain"),
            self.sql.replace(" AND ", " OR "),
            "SELECT * FROM orders WHERE amount > 100",
            self.sql.replace("*", "amount"),
            "SELECT * FROM orders WHERE 1 = 0",
        ):
            with self.subTest(sql=wrong):
                self.assertFalse(score_sql(wrong, self.references)["correct"])

    def test_forbidden_sql_is_rejected_and_database_unchanged(self):
        path = self.paths[0]
        before = execute_readonly(path, "SELECT * FROM orders")
        for forbidden in (
            "DELETE FROM orders", self.sql + " DROP TABLE orders;",
            "SELECT * FROM sqlite_master", "SELECT load_extension('missing')",
            "PRAGMA table_info(orders)", "```sql\n" + self.sql + "\n```",
        ):
            with self.subTest(sql=forbidden):
                with self.assertRaises((ValueError, sqlite3.Error)):
                    execute_readonly(path, forbidden)
        self.assertEqual(before, execute_readonly(path, "SELECT * FROM orders"))

    def test_duplicate_rows_are_not_silently_discarded(self):
        duplicate = self.sql.rstrip(";") + " UNION ALL " + self.sql
        self.assertFalse(score_sql(duplicate, self.references)["correct"])

    def test_expensive_query_is_interrupted(self):
        path = ROOT / "data" / "fixtures" / "orders_boundaries.db"
        with self.assertRaisesRegex(sqlite3.OperationalError, "interrupted"):
            execute_readonly(path, "SELECT DISTINCT a.id FROM orders a, orders b, orders c, orders d "
                             "WHERE a.amount + b.amount + c.amount + d.amount > 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
