"""
Improved Streamlit UI testing with comprehensive mocking strategies.

This module provides better mocking approaches for Streamlit components
to handle context managers, multiple calls, and complex UI flows.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

# Create a mock session state that behaves like both a dict and has attributes
class MockSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class StreamlitUITester:
    """Comprehensive Streamlit UI testing helper."""
    
    def __init__(self):
        self.reset_mocks()
    
    def reset_mocks(self):
        """Reset all streamlit mocks for clean test state."""
        # Core streamlit mock
        self.mock_streamlit = Mock()
        self.mock_streamlit.session_state = MockSessionState()
        
        # Text and input components
        self.mock_streamlit.title = Mock()
        self.mock_streamlit.markdown = Mock()
        self.mock_streamlit.text_input = Mock(side_effect=self._text_input_handler)
        self.mock_streamlit.text_area = Mock(side_effect=self._text_area_handler)
        
        # Form components
        self.mock_form_context = MagicMock()
        self.mock_form_context.__enter__ = Mock(return_value=None)
        self.mock_form_context.__exit__ = Mock(return_value=None)
        self.mock_streamlit.form = Mock(return_value=self.mock_form_context)
        self.mock_streamlit.form_submit_button = Mock(return_value=False)
        
        # Layout components
        self.mock_col1 = MagicMock()
        self.mock_col1.__enter__ = Mock(return_value=None)
        self.mock_col1.__exit__ = Mock(return_value=None)
        self.mock_col2 = MagicMock()
        self.mock_col2.__enter__ = Mock(return_value=None)
        self.mock_col2.__exit__ = Mock(return_value=None)
        self.mock_streamlit.columns = Mock(return_value=[self.mock_col1, self.mock_col2])
        
        # Status components
        self.mock_streamlit.error = Mock()
        self.mock_streamlit.success = Mock()
        self.mock_streamlit.info = Mock()
        self.mock_streamlit.warning = Mock()
        
        # Interactive components
        self.mock_streamlit.button = Mock(return_value=False)
        
        # Loading components
        self.mock_spinner_context = MagicMock()
        self.mock_spinner_context.__enter__ = Mock(return_value=None)
        self.mock_spinner_context.__exit__ = Mock(return_value=None)
        self.mock_streamlit.spinner = Mock(return_value=self.mock_spinner_context)
        
        # App control
        self.mock_streamlit.rerun = Mock()
        self.mock_streamlit.stop = Mock()
        
        # Component return value handlers
        self._input_values = {}
        self._input_sequence = {}
    
    def set_input_value(self, key, value):
        """Set a specific input value by key."""
        self._input_values[key] = value
    
    def set_input_sequence(self, key, values):
        """Set a sequence of values for multiple calls to the same input."""
        self._input_sequence[key] = list(values)
    
    def _text_input_handler(self, label, **kwargs):
        """Handle text_input calls with proper return values."""
        key = kwargs.get('key', label)
        
        # Check for sequence first
        if key in self._input_sequence and self._input_sequence[key]:
            return self._input_sequence[key].pop(0)
        
        # Check for single value
        if key in self._input_values:
            return self._input_values[key]
        
        # Default empty string
        return ""
    
    def _text_area_handler(self, label, **kwargs):
        """Handle text_area calls with proper return values."""
        key = kwargs.get('key', label)
        
        # Check for sequence first
        if key in self._input_sequence and self._input_sequence[key]:
            return self._input_sequence[key].pop(0)
        
        # Check for single value
        if key in self._input_values:
            return self._input_values[key]
        
        # Default empty string
        return ""
    
    def set_button_return(self, button_text, value):
        """Set return value for a specific button."""
        # This is more complex since streamlit button doesn't have keys
        # We'll use a simple approach for now
        if button_text == "Login":
            self.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: value if text == button_text else False
        elif button_text == "Register":
            self.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: value if text == button_text else False
    
    def patch_streamlit(self):
        """Return patch context for streamlit module."""
        return patch.dict('sys.modules', {'streamlit': self.mock_streamlit})
    
    def assert_login_form_displayed(self):
        """Assert that login form was displayed."""
        self.mock_streamlit.title.assert_called_with("🔐 Login Required")
        self.mock_streamlit.markdown.assert_called_with("Please log in to access the AI Therapist.")
        self.mock_streamlit.form.assert_called_with("login_form")
    
    def assert_register_form_displayed(self):
        """Assert that register form was displayed."""
        self.mock_streamlit.form.assert_called_with("register_form")
    
    def assert_reset_form_displayed(self):
        """Assert that password reset form was displayed."""
        self.mock_streamlit.form.assert_called_with("reset_form")


@pytest.fixture
def streamlit_tester():
    """Fixture providing StreamlitUITester instance."""
    tester = StreamlitUITester()
    yield tester
    tester.reset_mocks()


# Example usage in tests
class TestImprovedAuthMiddleware:
    """Example of improved auth middleware testing."""
    
    @pytest.fixture
    def auth_middleware(self, streamlit_tester):
        """Create auth middleware with improved streamlit mocking."""
        with streamlit_tester.patch_streamlit():
            from auth.middleware import AuthMiddleware
            from auth.auth_service import AuthService
            
            # Mock auth service
            mock_auth_service = Mock(spec=AuthService)
            
            # Clear session state
            streamlit_tester.mock_streamlit.session_state.clear()
            
            middleware = AuthMiddleware(mock_auth_service)
            yield middleware, streamlit_tester
    
    def test_show_login_form_improved(self, auth_middleware):
        """Test login form with improved mocking."""
        middleware, tester = auth_middleware
        
        # Set up clean session state
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Assert form was displayed correctly
        tester.assert_login_form_displayed()
    
    def test_login_with_user_input(self, auth_middleware):
        """Test login flow with user input."""
        middleware, tester = auth_middleware
        
        # Set up user input
        tester.set_input_value("login_email", "test@example.com")
        tester.set_input_value("login_password", "SecurePass123")
        
        # Set up login button press
        tester.set_button_return("Login", True)
        
        # Mock successful login
        mock_result = Mock()
        mock_result.success = True
        mock_result.token = "jwt_token_123"
        mock_result.user = Mock()
        middleware.auth_service.login_user.return_value = mock_result
        
        # Clear session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form (which will process the login)
        middleware.show_login_form()
        
        # Assert login was attempted
        middleware.auth_service.login_user.assert_called_once_with("test@example.com", "SecurePass123")
        tester.mock_streamlit.success.assert_called_with("Login successful!")
        tester.mock_streamlit.rerun.assert_called_once()


# Additional utilities for component testing
def assert_form_called_with(mock_streamlit, form_name):
    """Assert that a specific form was called."""
    mock_streamlit.form.assert_called_with(form_name)


def assert_input_called_with(mock_streamlit, label, key=None):
    """Assert that text_input was called with specific parameters."""
    call_kwargs = {'label': label}
    if key:
        call_kwargs['key'] = key
    
    # Check if text_input was called with these parameters
    for call in mock_streamlit.text_input.call_args_list:
        if call.args[0] == label and (key is None or call.kwargs.get('key') == key):
            return True
    
    raise AssertionError(f"text_input not called with label={label}, key={key}")


def assert_button_clicked(mock_streamlit, button_text):
    """Assert that a specific button was clicked."""
    # Check form_submit_button calls
    for call in mock_streamlit.form_submit_button.call_args_list:
        if call.args[0] == button_text:
            return True
    
    # Check regular button calls
    for call in mock_streamlit.button.call_args_list:
        if call.args[0] == button_text:
            return True
    
    raise AssertionError(f"Button {button_text} was not clicked")