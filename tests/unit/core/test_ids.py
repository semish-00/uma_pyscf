"""Record identifiers and content fingerprints."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import (
    CASE_ID_PATTERN,
    canonical_json_fingerprint,
    sha256_of_file,
    validate_record_id,
)


class RecordIdTests(unittest.TestCase):
    def test_valid_ids_are_returned_unchanged(self) -> None:
        for value in (
            "sih3_cation_singlet",
            "h2_wb97mv_def2tzvpd",
            "gecl4_bond1_x1p15",
            "ds_sigehcl_001",
            "0",
        ):
            with self.subTest(value=value):
                self.assertEqual(validate_record_id(value), value)
                self.assertIsNotNone(CASE_ID_PATTERN.fullmatch(value))

    def test_invalid_ids_are_rejected(self) -> None:
        for value in (
            "Bad-ID",
            "SiH3",
            "_leading_underscore",
            "-leading-hyphen",
            "",
            "has space",
            "trailing/slash",
            "dotted.id",
            "caseIdWithCaps",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    validate_record_id(value)

    def test_non_string_is_rejected(self) -> None:
        values: tuple[object, ...] = (None, 3, ["a"])
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    validate_record_id(value)  # type: ignore[arg-type]

    def test_error_names_the_offending_value(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            validate_record_id("Bad-ID")
        self.assertIn("Bad-ID", str(caught.exception))


class CanonicalFingerprintTests(unittest.TestCase):
    def test_digest_is_independent_of_key_order(self) -> None:
        first = {"charge": 0, "multiplicity": 1, "method": {"basis": "def2-TZVPD"}}
        second = {"method": {"basis": "def2-TZVPD"}, "multiplicity": 1, "charge": 0}
        self.assertEqual(canonical_json_fingerprint(first), canonical_json_fingerprint(second))

    def test_digest_changes_with_content(self) -> None:
        base = canonical_json_fingerprint({"charge": 0, "multiplicity": 1})
        self.assertNotEqual(base, canonical_json_fingerprint({"charge": 1, "multiplicity": 1}))
        self.assertNotEqual(base, canonical_json_fingerprint({"charge": 0, "multiplicity": 3}))

    def test_extra_bytes_change_the_digest(self) -> None:
        data = {"case_id": "sih4_td_seed"}
        without = canonical_json_fingerprint(data)
        with_bytes = canonical_json_fingerprint(data, b"5\n\nSi 0.0 0.0 0.0\n")
        self.assertNotEqual(without, with_bytes)
        self.assertNotEqual(with_bytes, canonical_json_fingerprint(data, b"other"))

    def test_empty_extra_bytes_do_not_change_the_digest(self) -> None:
        data = {"case_id": "sih4_td_seed"}
        self.assertEqual(canonical_json_fingerprint(data), canonical_json_fingerprint(data, b""))

    def test_composition_is_canonical_json_then_extra_bytes(self) -> None:
        data = {"b": [1, 2], "a": "x"}
        extra = b"structure bytes"
        expected = sha256(b'{"a":"x","b":[1,2]}' + extra).hexdigest()
        self.assertEqual(canonical_json_fingerprint(data, extra), expected)

    def test_digest_is_lowercase_hex_of_expected_length(self) -> None:
        digest = canonical_json_fingerprint({"a": 1})
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, digest.lower())

    def test_unserializable_input_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            canonical_json_fingerprint({"value": object()})


class FileDigestTests(unittest.TestCase):
    def test_matches_hashlib_over_the_same_bytes(self) -> None:
        payload = b"3\nH2 test structure\nH 0.0 0.0 0.0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structure.xyz"
            path.write_bytes(payload)
            self.assertEqual(sha256_of_file(path), sha256(payload).hexdigest())

    def test_empty_file_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_bytes(b"")
            self.assertEqual(sha256_of_file(path), sha256(b"").hexdigest())

    def test_large_file_is_read_in_chunks(self) -> None:
        payload = b"x" * (3 * (1 << 20) + 17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.bin"
            path.write_bytes(payload)
            self.assertEqual(sha256_of_file(path), sha256(payload).hexdigest())

    def test_missing_file_raises_os_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                sha256_of_file(Path(directory) / "absent.xyz")


if __name__ == "__main__":
    unittest.main()
