"""Streamlit AppTest suite to verify dashboard loading and tab switching."""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch
import os

@pytest.fixture(autouse=True)
def skip_if_no_models():
    """Skip tests if critical models aren't trained, to avoid noise."""
    import pathlib
    if not pathlib.Path("models/advanced").exists():
        pytest.skip("Models not trained, skipping dashboard tests.")

def test_app_tabs_load_without_crashing():
    """Verify that the dashboard loads and we can navigate tabs without exceptions."""
    # Ensure Streamlit runs headlessly
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Initialize the AppTest targeting our main Streamlit file
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    
    # Run the initial app load
    at.run()
    
    # Ensure no exceptions on the default tab (Executive Overview)
    assert not at.exception, f"App crashed on load: {at.exception}"
    
    # The sidebar typically has radio buttons for navigation. Let's find it.
    # In Streamlit AppTest, we can interact with widgets.
    if len(at.sidebar.radio) > 0:
        nav_radio = at.sidebar.radio[0]
        options = nav_radio.options
        
        # Click through all available tabs
        for option in options:
            nav_radio.set_value(option).run()
            assert not at.exception, f"App crashed when navigating to tab '{option}': {at.exception}"
