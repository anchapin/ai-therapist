"""
Comprehensive middleware tests to achieve 90%+ coverage for auth module.
Focuses on Streamlit middleware functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import os
import tempfile

from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserProfile, UserRole, UserStatus
from auth.middleware import AuthMiddleware


class TestMiddlewareComprehensive:
    """Comprehensive tests for middleware functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock environment variables
        os.environ['JWT_SECRET_KEY'] = 'test-secret-key'
        os.environ['JWT_EXPIRATION_HOURS'] = '24'
        os.environ['SESSION_TIMEOUT_MINUTES'] = '30'
        os.environ['MAX_CONCURRENT_SESSIONS'] = '5'
        
        # Initialize auth service
        self.auth_service = AuthService()
        
        # Create middleware
        self.middleware = AuthMiddleware(self.auth_service)
        
        # Create test user
        self.test_user = UserProfile(
            user_id="test_user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    def teardown_method(self):
        """Clean up after tests."""
        # Clean up environment variables
        for key in ['JWT_SECRET_KEY', 'JWT_EXPIRATION_HOURS', 'SESSION_TIMEOUT_MINUTES', 'MAX_CONCURRENT_SESSIONS']:
            if key in os.environ:
                del os.environ[key]

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_show_login_page(self, mock_markdown, mock_title, mock_session_state):
        """Test showing login page."""
        self.middleware.show_login_page()
        
        mock_title.assert_called_once_with("AI Therapist - Login")
        mock_markdown.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.rerun')
    def test_login_user_success(self, mock_rerun, mock_error, mock_success,
                               mock_text_input, mock_form_submit_button, mock_form,
                               mock_columns, mock_session_state):
        """Test successful user login through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "password"]
        mock_form_submit_button.return_value = True
        
        # Mock successful login
        self.auth_service.login_user.return_value = AuthResult(
            success=True,
            user=self.test_user,
            token="valid_token",
            session=Mock()
        )
        
        # Test login
        self.middleware.login_user("test@example.com", "password")
        
        # Verify session state is set
        assert mock_session_state.get('auth_token') == "valid_token"
        assert mock_session_state.get('user') is not None

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    def test_login_user_failure(self, mock_error, mock_text_input, mock_form_submit_button,
                               mock_form, mock_columns, mock_session_state):
        """Test failed user login through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "wrong_password"]
        mock_form_submit_button.return_value = True
        
        # Mock failed login
        self.auth_service.login_user.return_value = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        # Test login
        self.middleware.login_user("test@example.com", "wrong_password")
        
        # Verify error message
        mock_error.assert_called_with("Invalid credentials")

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_register_user_success(self, mock_error, mock_success, mock_text_input,
                                  mock_form_submit_button, mock_form, mock_columns,
                                  mock_session_state):
        """Test successful user registration through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["new@example.com", "SecurePass123", "SecurePass123", "New User"]
        mock_form_submit_button.return_value = True
        
        # Mock successful registration
        self.auth_service.register_user.return_value = AuthResult(
            success=True,
            user=self.test_user
        )
        
        # Test registration
        self.middleware.register_user("new@example.com", "SecurePass123", "SecurePass123", "New User")
        
        # Verify success message
        mock_success.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    def test_register_user_failure(self, mock_error, mock_text_input, mock_form_submit_button,
                                  mock_form, mock_columns, mock_session_state):
        """Test failed user registration through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "password", "different", "Test User"]
        mock_form_submit_button.return_value = True
        
        # Test registration with mismatched passwords
        self.middleware.register_user("test@example.com", "password", "different", "Test User")
        
        # Verify error message
        mock_error.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.button')
    @patch('streamlit.success')
    @patch('streamlit.rerun')
    def test_logout_user(self, mock_rerun, mock_success, mock_button, mock_session_state):
        """Test user logout through middleware."""
        # Setup session state
        mock_session_state['auth_token'] = "valid_token"
        mock_session_state['user'] = self.test_user
        
        # Mock successful logout
        self.auth_service.logout_user.return_value = True
        
        # Mock button click
        mock_button.return_value = True
        
        # Test logout
        self.middleware.logout_user()
        
        # Verify session state is cleared
        assert 'auth_token' not in mock_session_state
        assert 'user' not in mock_session_state

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_authenticated_true(self, mock_session_state):
        """Test authentication check when user is authenticated."""
        # Setup session state
        mock_session_state['auth_token'] = "valid_token"
        mock_session_state['user'] = self.test_user
        
        # Mock successful token validation
        self.auth_service.validate_token.return_value = self.test_user
        
        # Test authentication
        result = self.middleware.is_authenticated()
        
        assert result is True

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_authenticated_false_no_token(self, mock_session_state):
        """Test authentication check when no token exists."""
        # Test authentication without token
        result = self.middleware.is_authenticated()
        
        assert result is False

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_authenticated_false_invalid_token(self, mock_session_state):
        """Test authentication check when token is invalid."""
        # Setup session state
        mock_session_state['auth_token'] = "invalid_token"
        mock_session_state['user'] = self.test_user
        
        # Mock failed token validation
        self.auth_service.validate_token.return_value = None
        
        # Test authentication
        result = self.middleware.is_authenticated()
        
        assert result is False

    @patch('streamlit.session_state', new_callable=dict)
    def test_get_current_user(self, mock_session_state):
        """Test getting current user from session."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Test getting current user
        result = self.middleware.get_current_user()
        
        assert result == self.test_user

    @patch('streamlit.session_state', new_callable=dict)
    def test_get_current_user_none(self, mock_session_state):
        """Test getting current user when not authenticated."""
        # Test getting current user without session
        result = self.middleware.get_current_user()
        
        assert result is None

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_reset_password_success(self, mock_error, mock_success, mock_text_input,
                                   mock_form_submit_button, mock_form, mock_columns,
                                   mock_session_state):
        """Test successful password reset through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "SecurePass123", "SecurePass123"]
        mock_form_submit_button.return_value = True
        
        # Mock successful password reset
        self.auth_service.reset_password.return_value = AuthResult(
            success=True
        )
        
        # Test password reset
        self.middleware.reset_password("test@example.com", "SecurePass123", "SecurePass123")
        
        # Verify success message
        mock_success.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    def test_reset_password_failure(self, mock_error, mock_text_input, mock_form_submit_button,
                                   mock_form, mock_columns, mock_session_state):
        """Test failed password reset through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "password", "different"]
        mock_form_submit_button.return_value = True
        
        # Test password reset with mismatched passwords
        self.middleware.reset_password("test@example.com", "password", "different")
        
        # Verify error message
        mock_error.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_initiate_password_reset_success(self, mock_error, mock_success, mock_text_input,
                                            mock_form_submit_button, mock_form, mock_columns,
                                            mock_session_state):
        """Test successful password reset initiation through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "test@example.com"
        mock_form_submit_button.return_value = True
        
        # Mock successful password reset initiation
        self.auth_service.initiate_password_reset.return_value = AuthResult(
            success=True
        )
        
        # Test password reset initiation
        self.middleware.initiate_password_reset("test@example.com")
        
        # Verify success message
        mock_success.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    def test_initiate_password_reset_failure(self, mock_error, mock_text_input,
                                            mock_form_submit_button, mock_form, mock_columns,
                                            mock_session_state):
        """Test failed password reset initiation through middleware."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "nonexistent@example.com"
        mock_form_submit_button.return_value = True
        
        # Mock failed password reset initiation
        self.auth_service.initiate_password_reset.return_value = AuthResult(
            success=False,
            error_message="User not found"
        )
        
        # Test password reset initiation
        self.middleware.initiate_password_reset("nonexistent@example.com")
        
        # Verify error message
        mock_error.assert_called_with("User not found")

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_change_password_success(self, mock_error, mock_success, mock_text_input,
                                    mock_form_submit_button, mock_form, mock_columns,
                                    mock_session_state):
        """Test successful password change through middleware."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["old_password", "new_password", "new_password"]
        mock_form_submit_button.return_value = True
        
        # Mock successful password change
        self.auth_service.change_password.return_value = AuthResult(
            success=True
        )
        
        # Test password change
        self.middleware.change_password("old_password", "new_password", "new_password")
        
        # Verify success message
        mock_success.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    def test_change_password_failure(self, mock_error, mock_text_input,
                                    mock_form_submit_button, mock_form, mock_columns,
                                    mock_session_state):
        """Test failed password change through middleware."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["old_password", "new_password", "different"]
        mock_form_submit_button.return_value = True
        
        # Test password change with mismatched passwords
        self.middleware.change_password("old_password", "new_password", "different")
        
        # Verify error message
        mock_error.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_update_profile_success(self, mock_error, mock_success, mock_text_input,
                                   mock_form_submit_button, mock_form, mock_columns,
                                   mock_session_state):
        """Test successful profile update through middleware."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "Updated Name"
        mock_form_submit_button.return_value = True
        
        # Mock successful profile update
        self.auth_service.user_model.update_user.return_value = True
        
        # Test profile update
        self.middleware.update_profile("Updated Name")
        
        # Verify success message
        mock_success.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    def test_update_profile_failure(self, mock_error, mock_text_input,
                                   mock_form_submit_button, mock_form, mock_columns,
                                   mock_session_state):
        """Test failed profile update through middleware."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = ""
        mock_form_submit_button.return_value = True
        
        # Test profile update with empty name
        self.middleware.update_profile("")
        
        # Verify error message
        mock_error.assert_called()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_login_form(self, mock_error, mock_success, mock_text_input,
                        mock_form_submit_button, mock_form, mock_columns,
                        mock_session_state):
        """Test login form display and submission."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "password"]
        mock_form_submit_button.return_value = True
        
        # Mock successful login
        self.auth_service.login_user.return_value = AuthResult(
            success=True,
            user=self.test_user,
            token="valid_token",
            session=Mock()
        )
        
        # Test login form
        self.middleware.login_form()
        
        # Verify form was displayed and submitted
        mock_form.assert_called_once()
        mock_form_submit_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_register_form(self, mock_error, mock_success, mock_text_input,
                          mock_form_submit_button, mock_form, mock_columns,
                          mock_session_state):
        """Test registration form display and submission."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["new@example.com", "SecurePass123", "SecurePass123", "New User"]
        mock_form_submit_button.return_value = True
        
        # Mock successful registration
        self.auth_service.register_user.return_value = AuthResult(
            success=True,
            user=self.test_user
        )
        
        # Test registration form
        self.middleware.register_form()
        
        # Verify form was displayed and submitted
        mock_form.assert_called_once()
        mock_form_submit_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_reset_password_form(self, mock_error, mock_success, mock_text_input,
                                mock_form_submit_button, mock_form, mock_columns,
                                mock_session_state):
        """Test password reset form display and submission."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["test@example.com", "SecurePass123", "SecurePass123"]
        mock_form_submit_button.return_value = True
        
        # Mock successful password reset
        self.auth_service.reset_password.return_value = AuthResult(
            success=True
        )
        
        # Test password reset form
        self.middleware.reset_password_form()
        
        # Verify form was displayed and submitted
        mock_form.assert_called_once()
        mock_form_submit_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_change_password_form(self, mock_error, mock_success, mock_text_input,
                                 mock_form_submit_button, mock_form, mock_columns,
                                 mock_session_state):
        """Test password change form display and submission."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.side_effect = ["old_password", "new_password", "new_password"]
        mock_form_submit_button.return_value = True
        
        # Mock successful password change
        self.auth_service.change_password.return_value = AuthResult(
            success=True
        )
        
        # Test password change form
        self.middleware.change_password_form()
        
        # Verify form was displayed and submitted
        mock_form.assert_called_once()
        mock_form_submit_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_update_profile_form(self, mock_error, mock_success, mock_text_input,
                                mock_form_submit_button, mock_form, mock_columns,
                                mock_session_state):
        """Test profile update form display and submission."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "Updated Name"
        mock_form_submit_button.return_value = True
        
        # Mock successful profile update
        self.auth_service.user_model.update_user.return_value = True
        
        # Test profile update form
        self.middleware.update_profile_form()
        
        # Verify form was displayed and submitted
        mock_form.assert_called_once()
        mock_form_submit_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_forgot_password_form(self, mock_error, mock_success, mock_text_input,
                                 mock_form_submit_button, mock_form, mock_columns,
                                 mock_session_state):
        """Test forgot password form display and submission."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "test@example.com"
        mock_form_submit_button.return_value = True
        
        # Mock successful password reset initiation
        self.auth_service.initiate_password_reset.return_value = AuthResult(
            success=True
        )
        
        # Test forgot password form
        self.middleware.forgot_password_form()
        
        # Verify form was displayed and submitted
        mock_form.assert_called_once()
        mock_form_submit_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.button')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_logout_button(self, mock_error, mock_success, mock_button, mock_session_state):
        """Test logout button functionality."""
        # Setup session state
        mock_session_state['auth_token'] = "valid_token"
        mock_session_state['user'] = self.test_user
        
        # Mock successful logout
        self.auth_service.logout_user.return_value = True
        
        # Mock button click
        mock_button.return_value = True
        
        # Test logout button
        self.middleware.logout_button()
        
        # Verify button was displayed
        mock_button.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    def test_require_auth_authenticated(self, mock_session_state):
        """Test require_auth when user is authenticated."""
        # Setup session state
        mock_session_state['auth_token'] = "valid_token"
        mock_session_state['user'] = self.test_user
        
        # Mock successful token validation
        self.auth_service.validate_token.return_value = self.test_user
        
        # Test require_auth
        result = self.middleware.require_auth()
        
        assert result is True

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.stop')
    def test_require_auth_not_authenticated(self, mock_stop, mock_session_state):
        """Test require_auth when user is not authenticated."""
        # Test require_auth without authentication
        self.middleware.require_auth()
        
        # Verify streamlit.stop was called
        mock_stop.assert_called_once()

    @patch('streamlit.session_state', new_callable=dict)
    def test_get_user_role(self, mock_session_state):
        """Test getting user role."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Test getting user role
        result = self.middleware.get_user_role()
        
        assert result == UserRole.PATIENT

    @patch('streamlit.session_state', new_callable=dict)
    def test_get_user_role_none(self, mock_session_state):
        """Test getting user role when not authenticated."""
        # Test getting user role without session
        result = self.middleware.get_user_role()
        
        assert result is None

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_admin_true(self, mock_session_state):
        """Test admin check when user is admin."""
        # Create admin user
        admin_user = UserProfile(
            user_id="admin_user_123",
            email="admin@example.com",
            full_name="Admin User",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Setup session state
        mock_session_state['user'] = admin_user
        
        # Test admin check
        result = self.middleware.is_admin()
        
        assert result is True

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_admin_false(self, mock_session_state):
        """Test admin check when user is not admin."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Test admin check
        result = self.middleware.is_admin()
        
        assert result is False

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_therapist_true(self, mock_session_state):
        """Test therapist check when user is therapist."""
        # Create therapist user
        therapist_user = UserProfile(
            user_id="therapist_user_123",
            email="therapist@example.com",
            full_name="Therapist User",
            role=UserRole.THERAPIST,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Setup session state
        mock_session_state['user'] = therapist_user
        
        # Test therapist check
        result = self.middleware.is_therapist()
        
        assert result is True

    @patch('streamlit.session_state', new_callable=dict)
    def test_is_therapist_false(self, mock_session_state):
        """Test therapist check when user is not therapist."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Test therapist check
        result = self.middleware.is_therapist()
        
        assert result is False

    @patch('streamlit.session_state', new_callable=dict)
    def test_can_access_resource_true(self, mock_session_state):
        """Test resource access check when user has permission."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Mock successful permission check
        self.auth_service.validate_session_access.return_value = True
        
        # Test resource access
        result = self.middleware.can_access_resource("own_profile", "read")
        
        assert result is True

    @patch('streamlit.session_state', new_callable=dict)
    def test_can_access_resource_false(self, mock_session_state):
        """Test resource access check when user doesn't have permission."""
        # Setup session state
        mock_session_state['user'] = self.test_user
        
        # Mock failed permission check
        self.auth_service.validate_session_access.return_value = False
        
        # Test resource access
        result = self.middleware.can_access_resource("system_config", "update")
        
        assert result is False

    @patch('streamlit.session_state', new_callable=dict)
    def test_can_access_resource_not_authenticated(self, mock_session_state):
        """Test resource access check when user is not authenticated."""
        # Test resource access without authentication
        result = self.middleware.can_access_resource("own_profile", "read")
        
        assert result is False