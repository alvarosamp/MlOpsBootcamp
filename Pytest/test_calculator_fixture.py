import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from calculator import add, subtract


@pytest.fixture
def calculator_setup():
    print("Setting up the envirometn")
    return {}

def test_add(calculator_setup):
    result = add(3, 5)
    assert result == 8, f"Expected 8 but got {result}"

def test_subtract(calculator_setup):
    result = subtract(10, 4)
    assert result == 6, f"Expected 6 but got {result}"
