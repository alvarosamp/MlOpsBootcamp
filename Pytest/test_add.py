import pytest
from Pytest.main import soma
def test_addition():
    assert soma(3,4) == 7, "Should be 7"
    assert soma(0,0) == 0, "Should be 0"
    assert soma(-1,-2) == -3, "Should be -3"
    