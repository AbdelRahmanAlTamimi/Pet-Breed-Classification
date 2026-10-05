import pytest

from pet_breed_classification.data import load_records


def test_load_records_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="train.*val"):
        load_records("test")
