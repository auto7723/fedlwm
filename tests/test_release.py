from pathlib import Path

from fedlwm.audit import audit_release


def test_release_has_no_identity_or_machine_path_tokens():
    root = Path(__file__).resolve().parents[1]
    assert audit_release(root) == []
