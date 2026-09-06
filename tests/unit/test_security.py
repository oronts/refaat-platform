from pathlib import Path

import pytest

from refaat.errors import ValidationError
from refaat.security import hash_api_key, safe_join, validate_outbound_url, verify_api_key


def test_api_key_hash_round_trip() -> None:
    digest = hash_api_key("refaat_secret", "pepper")

    assert verify_api_key("refaat_secret", digest, "pepper")
    assert not verify_api_key("refaat_wrong", digest, "pepper")
    assert not verify_api_key("refaat_secret", "broken", "pepper")


def test_safe_join_blocks_traversal(tmp_path: Path) -> None:
    assert safe_join(tmp_path, "folder/file.txt").is_relative_to(tmp_path)
    with pytest.raises(ValidationError):
        safe_join(tmp_path, "../secret")


def test_outbound_url_blocks_loopback() -> None:
    with pytest.raises(ValidationError):
        validate_outbound_url("http://127.0.0.1/private", allow_private=False)
    validate_outbound_url("http://127.0.0.1/private", allow_private=True)
