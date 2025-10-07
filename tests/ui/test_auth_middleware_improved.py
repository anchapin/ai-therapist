"""
Improved unit tests for auth/middleware.py using comprehensive streamlit testing utilities.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Import the improved streamlit testing utilities
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from streamlit_test_utils import StreamlitUITester, assert_form_called_with, assert_input_called_with, assert_button_clicked

# Import the modules with mocked streamlit
class TestAuthMiddlewareImproved:
    """Improved test cases for AuthMiddleware using better mocking."""
    
    @pytest.fixture
    def auth_middleware(self):
        """Create auth middleware with improved streamlit mocking."""
        tester = StreamlitUITester()
        
        with tester.patch_streamlit():
            from auth.middleware import AuthMiddleware
            from auth.auth_service import AuthService
            
            # Mock auth service
            mock_auth_service = Mock(spec=AuthService)
            
            # Clear session state
            tester.mock_streamlit.session_state.clear()
            
            middleware = AuthMiddleware(mock_auth_service)
            yield middleware, tester
    
    def test_show_login_form_basic(self, auth_middleware):
        """Test basic login form display."""
        middleware, tester = auth_middleware
        
        # Ensure clean session state
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Assert basic form elements
        tester.assert_login_form_displayed()
        
        # Assert input fields
        assert_input_called_with(tester.mock_streamlit, "Email", key="login_email")
        assert_input_called_with(tester.mock_streamlit, "Password", key="login_password")
    
    def test_login_success_flow(self, auth_middleware):
        """Test successful login flow."""
        middleware, tester = auth_middleware
        
        # Set up user input
        tester.set_input_value("login_email", "test@example.com")
        tester.set_input_value("login_password", "SecurePass123")
        
        # Set up login button press
        tester.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: text == "Login"
        
        # Mock successful login
        mock_result = Mock()
        mock_result.success = True
        mock_result.token = "jwt_token_123"
        mock_result.user = Mock()
        middleware.auth_service.login_user.return_value = mock_result
        
        # Clean session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Verify login attempt
        middleware.auth_service.login_user.assert_called_once_with(
            email="test@example.com", 
            password="SecurePass123",
            ip_address="streamlit_client",
            user_agent="streamlit_browser"
        )
        tester.mock_streamlit.success.assert_called_with("Login successful!")
        tester.mock_streamlit.rerun.assert_called_once()
    
    def test_login_failure_flow(self, auth_middleware):
        """Test failed login flow."""
        middleware, tester = auth_middleware
        
        # Set up user input
        tester.set_input_value("login_email", "test@example.com")
        tester.set_input_value("login_password", "WrongPassword")
        
        # Set up login button press
        tester.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: text == "Login"
        
        # Mock failed login
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Invalid credentials"
        middleware.auth_service.login_user.return_value = mock_result
        
        # Clean session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Verify login attempt
        middleware.auth_service.login_user.assert_called_once_with(
            email="test@example.com", 
            password="WrongPassword",
            ip_address="streamlit_client",
            user_agent="streamlit_browser"
        )
        tester.mock_streamlit.error.assert_called_with("Login failed: Invalid credentials")
    
    def test_login_validation_empty_fields(self, auth_middleware):
        """Test login with empty fields."""
        middleware, tester = auth_middleware
        
        # Set up empty user input
        tester.set_input_value("login_email", "")
        tester.set_input_value("login_password", "")
        
        # Set up login button press
        tester.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: text == "Login"
        
        # Clean session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Verify validation error
        tester.mock_streamlit.error.assert_called_with("Please enter both email and password.")
        
        # Login should not be attempted
        middleware.auth_service.login_user.assert_not_called()
    
    def test_register_button_flow(self, auth_middleware):
        """Test register button flow."""
        middleware, tester = auth_middleware
        
        # Set up register button press
        tester.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: text == "Register"
        
        # Clean session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Verify register flag is set
        assert tester.mock_streamlit.session_state.get('show_register') == True
        tester.mock_streamlit.rerun.assert_called_once()
    
    def test_password_reset_button_flow(self, auth_middleware):
        """Test password reset button flow."""
        middleware, tester = auth_middleware
        
        # Set up password reset button press
        tester.mock_streamlit.button.side_effect = lambda text, **kwargs: text == "Forgot Password?"
        
        # Clean session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Show login form
        middleware.show_login_form()
        
        # Verify reset flag is set
        assert tester.mock_streamlit.session_state.get('show_reset') == True
        tester.mock_streamlit.rerun.assert_called_once()
    
    def test_show_register_form_when_flagged(self, auth_middleware):
        """Test that register form is shown when flag is set."""
        middleware, tester = auth_middleware
        
        # Set register flag
        tester.mock_streamlit.session_state.show_register = True
        
        # Show login form (should show register form)
        middleware.show_login_form()
        
        # Verify register form was displayed
        tester.assert_register_form_displayed()
    
    def test_show_password_reset_form_when_flagged(self, auth_middleware):
        """Test that password reset form is shown when flag is set."""
        middleware, tester = auth_middleware
        
        # Set reset flag
        tester.mock_streamlit.session_state.show_reset = True
        
        # Show login form (should show reset form)
        middleware.show_login_form()
        
        # Verify reset form was displayed
        tester.assert_reset_form_displayed()
    
    def test_role_based_access_control(self, auth_middleware):
        """Test role-based access control decorator."""
        middleware, tester = auth_middleware
        
        # Mock authenticated user with therapist role
        mock_user = Mock()
        mock_user.role = "therapist"
        
        # Set up authentication
        tester.mock_streamlit.session_state.auth_token = "valid_token"
        middleware.auth_service.validate_token.return_value = mock_user
        
        # Create protected function
        @middleware.role_required(["therapist", "admin"])
        def therapist_function():
            return "therapist_content"
        
        # Should allow access
        result = therapist_function()
        assert result == "therapist_content"
    
    def test_role_based_access_denied(self, auth_middleware):
        """Test role-based access control denial."""
        middleware, tester = auth_middleware
        
        # Mock authenticated user with patient role
        mock_user = Mock()
        mock_user.role = "patient"
        
        # Set up authentication
        tester.mock_streamlit.session_state.auth_token = "valid_token"
        middleware.auth_service.validate_token.return_value = mock_user
        
        # Create protected function
        @middleware.role_required(["therapist", "admin"])
        def therapist_function():
            return "therapist_content"
        
        # Should deny access and show error message
        result = therapist_function()
        assert result is None
        tester.mock_streamlit.error.assert_called_with("Access denied. Insufficient permissions.")
    
    def test_session_state_persistence(self, auth_middleware):
        """Test that session state persists correctly."""
        middleware, tester = auth_middleware
        
        # Mock successful login
        mock_result = Mock()
        mock_result.success = True
        mock_result.token = "persisted_token"
        mock_result.user = Mock()
        middleware.auth_service.login_user.return_value = mock_result
        
        # Set up login
        tester.set_input_value("login_email", "test@example.com")
        tester.set_input_value("login_password", "SecurePass123")
        tester.mock_streamlit.form_submit_button.side_effect = lambda text, **kwargs: text == "Login"
        
        # Clean session state
        tester.mock_streamlit.session_state.clear()
        tester.mock_streamlit.session_state.show_reset = False
        tester.mock_streamlit.session_state.show_register = False
        
        # Perform login
        middleware.show_login_form()
        
        # Verify token was stored
        assert tester.mock_streamlit.session_state.auth_token == "persisted_token"
        assert tester.mock_streamlit.session_state.user == mock_result.user
    
    def test_logout_clears_session(self, auth_middleware):
        """Test that logout clears session state properly."""
        middleware, tester = auth_middleware
        
        # Set up authenticated session
        tester.mock_streamlit.session_state.auth_token = "some_token"
        tester.mock_streamlit.session_state.user = Mock()
        
        # Perform logout
        middleware.logout_user()
        
        # Verify session was cleared
        assert 'auth_token' not in tester.mock_streamlit.session_state
        assert 'user' not in tester.mock_streamlit.session_state
    
    def test_concurrent_form_handling(self, auth_middleware):
        """Test handling of multiple forms in same session."""
        middleware, tester = auth_middleware
        
        # Set both flags (edge case)
        tester.mock_streamlit.session_state.show_register = True
        tester.mock_streamlit.session_state.show_reset = True
        
        # Show login form
        middleware.show_login_form()
        
        # Should show both forms (implementation dependent)
        # At minimum, both form calls should be made
        form_calls = [call.args[0] for call in tester.mock_streamlit.form.call_args_list]
        assert "login_form" in form_calls
        # Depending on implementation order, either or both additional forms may appear