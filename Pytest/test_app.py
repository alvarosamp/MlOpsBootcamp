import pytest
import requests

def test_app_running():
    url = 'http://127.0.0.1:5000'
    try:
        response = requests.get(url, timout = 5)
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"App is not running: {e}")
        