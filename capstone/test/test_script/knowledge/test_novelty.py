import pytest

from core.ingest.novelty import cliff_partition

G = 1.5
FLOOR = 0.55


def test_sharp_cliff_keeps_top_one():
    kept, novel = cliff_partition([0.82, 0.40, 0.35, 0.30], G, FLOOR)
    assert kept == [0]
    assert novel is False


def test_plateau_keeps_many_not_novel():
    ## two near-equal highs -> anchored to both, cliff after
    kept, novel = cliff_partition([0.80, 0.78, 0.35], G, FLOOR)
    assert set(kept) == {0, 1}
    assert novel is False


def test_flat_low_is_novel():
    ## uniformly low, no member clears the floor
    kept, novel = cliff_partition([0.30, 0.28, 0.27], G, FLOOR)
    assert novel is True


def test_flat_high_plateau_not_novel():
    ## uniformly high plateau -> kept-many, clears floor
    kept, novel = cliff_partition([0.90, 0.88, 0.87], G, FLOOR)
    assert len(kept) == 3
    assert novel is False


def test_floor_exact_equality_not_novel():
    kept, novel = cliff_partition([0.55, 0.20], G, FLOOR)
    assert novel is False


def test_single_element_above_floor():
    kept, novel = cliff_partition([0.70], G, FLOOR)
    assert kept == [0]
    assert novel is False


def test_single_element_below_floor():
    kept, novel = cliff_partition([0.40], G, FLOOR)
    assert kept == [0]
    assert novel is True


def test_empty_is_novel():
    kept, novel = cliff_partition([], G, FLOOR)
    assert kept == []
    assert novel is True


def test_unsorted_input_handled():
    ## caller may pass milvus hits in any order
    kept, novel = cliff_partition([0.30, 0.82, 0.35], G, FLOOR)
    assert kept == [1]
    assert novel is False


if __name__ == "__main__":
    pytest.main(["-v", __file__])
