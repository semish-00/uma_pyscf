"""Atomic writes and JSON reading."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.io import read_json, write_json_atomic, write_text_atomic


class AtomicJsonWriteTests(unittest.TestCase):
    def test_write_creates_parents_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "deeper" / "record.json"
            write_json_atomic(destination, {"value": 1})
            write_json_atomic(destination, {"value": 2})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_failed_serialization_leaves_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            write_json_atomic(destination, {"value": 1})
            with self.assertRaises(TypeError):
                write_json_atomic(destination, {"value": object()})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_failed_serialization_does_not_create_the_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fresh" / "record.json"
            with self.assertRaises(TypeError):
                write_json_atomic(destination, {"value": object()})
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_output_is_sorted_and_newline_terminated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            write_json_atomic(destination, {"b": 2, "a": 1})
            text = destination.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertLess(text.index('"a"'), text.index('"b"'))

    def test_non_ascii_content_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            write_json_atomic(destination, {"note": "Ångström"})
            self.assertEqual(read_json(destination), {"note": "Ångström"})

    def test_accepts_a_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            write_json_atomic(str(destination), {"value": 1})
            self.assertEqual(read_json(str(destination)), {"value": 1})


class AtomicTextWriteTests(unittest.TestCase):
    def test_write_creates_parents_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "note.txt"
            write_text_atomic(destination, "first\n")
            write_text_atomic(destination, "second\n")
            self.assertEqual(destination.read_text(encoding="utf-8"), "second\n")
            self.assertEqual(list(destination.parent.iterdir()), [destination])


class ReadJsonTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = {"case_id": "sih4_td_seed", "charge": 0, "atoms": [{"element": "Si"}]}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            write_json_atomic(destination, payload)
            self.assertEqual(read_json(destination), payload)

    def test_invalid_json_error_names_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "broken.json"
            destination.write_text('{"value": ', encoding="utf-8")
            with self.assertRaises(ValidationError) as caught:
                read_json(destination)
            self.assertIn("broken.json", str(caught.exception))

    def test_missing_file_raises_os_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_json(Path(directory) / "absent.json")


if __name__ == "__main__":
    unittest.main()
