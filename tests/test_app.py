import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODEL_PATH, ROOT


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="lancez d'abord python -m src.train")
def test_app_shows_a_price():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value.endswith("$")