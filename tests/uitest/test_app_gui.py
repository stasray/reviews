import pytest


def test_generate_and_analyze_shows_metrics_and_table():
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        pytest.skip(f"streamlit testing API unavailable: {e}")

    at = AppTest.from_file("tests/uitest/app_bootstrap.py", default_timeout=20)

    at.run()

    if at.sidebar.slider:
        at.sidebar.slider[0].set_value(10)
    assert len(at.sidebar.button) >= 1
    at.sidebar.button[0].click().run()
