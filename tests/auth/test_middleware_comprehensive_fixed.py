"""
Comprehensive test suite for auth/middleware.py with correct method names and proper mocking.
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import streamlit as st

from auth.middleware import AuthMiddleware
from auth.auth_service import AuthService, AuthResult
from auth.user_model import UserProfile, UserRole
from database.models import Session as UserSession


class TestMiddlewareComprehensiveFixed:
    """Test comprehensive middleware functionality with correct method names."""

    @pytest.fixture
    def mock_auth_service(self):
        """Create a mock auth service."""
        with mock.patch('auth.middleware.AuthService') as mock_service:
            yield mock_service

    @pytest.fixture
    def mock_streamlit(self):
        """Create comprehensive streamlit mocks."""
        with mock.patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {}
            
            # Mock streamlit components
            mock_st.title = mock.Mock()
            mock_st.markdown = mock.Mock()
            mock_st.subheader = mock.Mock()
            mock_st.form = mock.Mock()
            mock_st.text_input = mock.Mock(return_value="test")
            mock_st.text_input.side_effect = lambda label, **kwargs: f"test_{label.lower().replace(' ', '_')}"
            mock_st.columns = mock.Mock(return_value=[mock.Mock(), mock.Mock()])
            mock_st.form_submit_button = mock.Mock(return_value=False)
            mock_st.button = mock.Mock(return_value=False)
            mock_st.spinner = mock.Mock()
            mock_st.success = mock.Mock()
            mock_st.error = mock.Mock()
            mock_st.info = mock.Mock()
            mock_st.caption = mock.Mock()
            mock_st.selectbox = mock.Mock(return_value="Light")
            mock_st.sidebar = mock.Mock()
            mock_st.rerun = mock.Mock()
            
            # Mock form context manager
            mock_form_context = mock.MagicMock()
            mock_st.form.return_value.__enter__ = mock.Mock(return_value=mock_form_context)
            mock_st.form.return_value.__exit__ = mock.Mock(return_value=None)
            
            yield mock_st

    @pytest.fixture
    def middleware(self, mock_auth_service, mock_streamlit):
        """Create middleware instance with mocked dependencies."""
        return AuthMiddleware(mock_auth_service)

    @pytest.fixture
    def test_user(self):
        """Create a test user profile."""
        return UserProfile(
            user_id='test_user_123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status='active',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            failed_login_attempts=0,
            account_locked_until=None,
            password_reset_token=None,
            password_reset_expires=None,
            preferences={},
            medical_info={}
        )

    @pytest.fixture
    def test_session(self):
        """Create a test session."""
        return UserSession(
            session_id='test_session_123',
            user_id='test_user_123',
            token='test_token_123',
            ip_address='127.0.0.1',
            user_agent='test_agent',
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            is_active=True
        )

    def test_init(self, middleware, mock_auth_service):
        """Test middleware initialization."""
        assert middleware.auth_service == mock_auth_service

    def test_login_required_authenticated(self, middleware, mock_streamlit, test_user):
        """Test login_required decorator when user is authenticated."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Create test function
        @middleware.login_required
        def test_func():
            return "success"
        
        # Test decorator
        result = test_func()
        assert result == "success"
        middleware.auth_service.validate_token.assert_called_once_with('test_token')

    def test_login_required_not_authenticated(self, middleware, mock_streamlit):
        """Test login_required decorator when user is not authenticated."""
        # Setup unauthenticated state
        mock_streamlit.session_state = {}
        middleware.auth_service.validate_token.return_value = None
        
        # Create test function
        @middleware.login_required
        def test_func():
            return "success"
        
        # Test decorator
        result = test_func()
        assert result is None
        middleware.auth_service.validate_token.assert_called_once()

    def test_role_required_success(self, middleware, mock_streamlit, test_user):
        """Test role_required decorator when user has required role."""
        # Setup authenticated state with admin role
        mock_streamlit.session_state['auth_token'] = 'test_token'
        test_user.role = UserRole.ADMIN
        middleware.auth_service.validate_token.return_value = test_user
        
        # Create test function
        @middleware.role_required([UserRole.ADMIN])
        def test_func():
            return "success"
        
        # Test decorator
        result = test_func()
        assert result == "success"

    def test_role_required_wrong_role(self, middleware, mock_streamlit, test_user):
        """Test role_required decorator when user has wrong role."""
        # Setup authenticated state with patient role
        mock_streamlit.session_state['auth_token'] = 'test_token'
        test_user.role = UserRole.PATIENT
        middleware.auth_service.validate_token.return_value = test_user
        
        # Create test function
        @middleware.role_required([UserRole.ADMIN])
        def test_func():
            return "success"
        
        # Test decorator
        result = test_func()
        assert result is None
        mock_streamlit.error.assert_called_with("Access denied. Insufficient permissions.")

    def test_is_authenticated_true(self, middleware, mock_streamlit, test_user):
        """Test is_authenticated when user is authenticated."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Test authentication
        result = middleware.is_authenticated()
        assert result is True
        middleware.auth_service.validate_token.assert_called_once_with('test_token')

    def test_is_authenticated_false_no_token(self, middleware, mock_streamlit):
        """Test is_authenticated when no token exists."""
        # Setup unauthenticated state
        mock_streamlit.session_state = {}
        
        # Test authentication
        result = middleware.is_authenticated()
        assert result is False
        middleware.auth_service.validate_token.assert_not_called()

    def test_is_authenticated_false_invalid_token(self, middleware, mock_streamlit):
        """Test is_authenticated when token is invalid."""
        # Setup with invalid token
        mock_streamlit.session_state['auth_token'] = 'invalid_token'
        middleware.auth_service.validate_token.return_value = None
        
        # Test authentication
        result = middleware.is_authenticated()
        assert result is False
        middleware.auth_service.validate_token.assert_called_once_with('invalid_token')

    def test_get_current_user_success(self, middleware, mock_streamlit, test_user):
        """Test get_current_user when user is authenticated."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Test getting current user
        result = middleware.get_current_user()
        assert result == test_user
        middleware.auth_service.validate_token.assert_called_once_with('test_token')

    def test_get_current_user_none(self, middleware, mock_streamlit):
        """Test get_current_user when no token exists."""
        # Setup unauthenticated state
        mock_streamlit.session_state = {}
        
        # Test getting current user
        result = middleware.get_current_user()
        assert result is None
        middleware.auth_service.validate_token.assert_not_called()

    def test_login_user_success(self, middleware, mock_streamlit, test_user, test_session):
        """Test successful user login."""
        # Setup successful login result
        auth_result = AuthResult(
            success=True,
            user=test_user,
            token='test_token',
            session=test_session,
            error_message=None
        )
        middleware.auth_service.login_user.return_value = auth_result
        
        # Test login
        result = middleware.login_user('test@example.com', 'password')
        
        # Verify result
        assert result.success is True
        assert result.user == test_user
        assert result.token == 'test_token'
        
        # Verify session state is set
        assert mock_streamlit.session_state['auth_token'] == 'test_token'
        assert mock_streamlit.session_state['user'] == test_user
        assert mock_streamlit.session_state['auth_time'] == test_session.created_at
        
        # Verify auth service was called correctly
        middleware.auth_service.login_user.assert_called_once_with(
            email='test@example.com',
            password='password',
            ip_address='streamlit_client',
            user_agent='streamlit_browser'
        )

    def test_login_user_failure(self, middleware, mock_streamlit):
        """Test failed user login."""
        # Setup failed login result
        auth_result = AuthResult(
            success=False,
            user=None,
            token=None,
            session=None,
            error_message="Invalid credentials"
        )
        middleware.auth_service.login_user.return_value = auth_result
        
        # Test login
        result = middleware.login_user('test@example.com', 'wrong_password')
        
        # Verify result
        assert result.success is False
        assert result.error_message == "Invalid credentials"
        
        # Verify session state is not set
        assert 'auth_token' not in mock_streamlit.session_state
        assert 'user' not in mock_streamlit.session_state
        assert 'auth_time' not in mock_streamlit.session_state

    def test_logout_user_with_token(self, middleware, mock_streamlit):
        """Test logout user when token exists."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        mock_streamlit.session_state['user'] = 'test_user'
        mock_streamlit.session_state['auth_time'] = 'test_time'
        
        # Test logout
        middleware.logout_user()
        
        # Verify auth service was called
        middleware.auth_service.logout_user.assert_called_once_with('test_token')
        
        # Verify session state is cleared
        assert 'auth_token' not in mock_streamlit.session_state
        assert 'user' not in mock_streamlit.session_state
        assert 'auth_time' not in mock_streamlit.session_state

    def test_logout_user_no_token(self, middleware, mock_streamlit):
        """Test logout user when no token exists."""
        # Setup unauthenticated state
        mock_streamlit.session_state = {}
        
        # Test logout
        middleware.logout_user()
        
        # Verify auth service was not called
        middleware.auth_service.logout_user.assert_not_called()

    def test_show_login_form(self, middleware, mock_streamlit):
        """Test show_login_form displays correctly."""
        # Test showing login form
        middleware.show_login_form()
        
        # Verify components are called
        mock_streamlit.title.assert_called_with("🔐 Login Required")
        mock_streamlit.markdown.assert_called_with("Please log in to access the AI Therapist.")
        mock_streamlit.form.assert_called_with("login_form")

    def test_show_register_form(self, middleware, mock_streamlit):
        """Test show_register_form displays correctly."""
        # Test showing register form
        middleware.show_register_form()
        
        # Verify components are called
        mock_streamlit.subheader.assert_called_with("📝 Register New Account")
        mock_streamlit.form.assert_called_with("register_form")

    def test_show_password_reset_form(self, middleware, mock_streamlit):
        """Test show_password_reset_form displays correctly."""
        # Test showing password reset form
        middleware.show_password_reset_form()
        
        # Verify components are called
        mock_streamlit.subheader.assert_called_with("🔑 Reset Password")
        mock_streamlit.form.assert_called_with("reset_form")

    def test_show_user_menu(self, middleware, mock_streamlit, test_user):
        """Test show_user_menu displays correctly."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Test showing user menu
        middleware.show_user_menu()
        
        # Verify sidebar is used
        mock_streamlit.sidebar.assert_called_once()
        
        # Verify user info is displayed
        mock_streamlit.markdown.assert_called_with("---")
        mock_streamlit.subheader.assert_any_call(f"👤 {test_user.full_name}")

    def test_show_profile_settings(self, middleware, mock_streamlit, test_user):
        """Test show_profile_settings displays correctly."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Test showing profile settings
        middleware.show_profile_settings()
        
        # Verify components are called
        mock_streamlit.subheader.assert_called_with("👤 Profile Settings")
        mock_streamlit.form.assert_called_with("profile_form")

    def test_show_change_password_form(self, middleware, mock_streamlit, test_user):
        """Test show_change_password_form displays correctly."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Test showing change password form
        middleware.show_change_password_form()
        
        # Verify components are called
        mock_streamlit.subheader.assert_called_with("🔑 Change Password")
        mock_streamlit.form.assert_called_with("change_password_form")

    def test_get_client_ip(self, middleware):
        """Test _get_client_ip returns expected value."""
        result = middleware._get_client_ip()
        assert result == "streamlit_client"

    def test_get_user_agent(self, middleware):
        """Test _get_user_agent returns expected value."""
        result = middleware._get_user_agent()
        assert result == "streamlit_browser"

    def test_login_form_registration_flow(self, middleware, mock_streamlit, test_user, test_session):
        """Test login form registration flow."""
        # Setup form submission for registration
        mock_form_context = mock.MagicMock()
        mock_streamlit.form.return_value.__enter__ = mock.Mock(return_value=mock_form_context)
        
        # Mock form submit button to trigger registration
        def mock_form_submit_button(label, **kwargs):
            if label == "Register":
                return True
            return False
        mock_streamlit.form_submit_button.side_effect = mock_form_submit_button
        
        # Setup successful registration
        auth_result = AuthResult(
            success=True,
            user=test_user,
            token='test_token',
            session=test_session,
            error_message=None
        )
        middleware.auth_service.register_user.return_value = auth_result
        
        # Test login form with registration
        middleware.show_login_form()
        
        # Verify registration was initiated
        middleware.auth_service.register_user.assert_called_once()

    def test_login_form_password_reset_flow(self, middleware, mock_streamlit, test_user, test_session):
        """Test login form password reset flow."""
        # Setup button click for password reset
        mock_streamlit.button.return_value = True
        
        # Setup successful password reset
        auth_result = AuthResult(
            success=True,
            user=test_user,
            token='test_token',
            session=test_session,
            error_message=None
        )
        middleware.auth_service.initiate_password_reset.return_value = auth_result
        
        # Test login form with password reset
        middleware.show_login_form()
        
        # Verify password reset was initiated
        middleware.auth_service.initiate_password_reset.assert_called_once()

    def test_register_form_validation(self, middleware, mock_streamlit):
        """Test register form validation."""
        # Setup form submission with empty fields
        mock_form_context = mock.MagicMock()
        mock_streamlit.form.return_value.__enter__ = mock.Mock(return_value=mock_form_context)
        
        # Mock form submit button to return True
        mock_streamlit.form_submit_button.return_value = True
        
        # Mock text inputs to return empty strings
        mock_streamlit.text_input.return_value = ""
        
        # Test register form with empty fields
        middleware.show_register_form()
        
        # Verify error message is shown
        mock_streamlit.error.assert_called_with("Please fill in all fields.")

    def test_change_password_form_validation(self, middleware, mock_streamlit, test_user):
        """Test change password form validation."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Setup form submission with mismatched passwords
        mock_form_context = mock.MagicMock()
        mock_streamlit.form.return_value.__enter__ = mock.Mock(return_value=mock_form_context)
        
        # Mock form submit button to return True
        mock_streamlit.form_submit_button.return_value = True
        
        # Mock text inputs to return mismatched passwords
        def mock_text_input(label, **kwargs):
            if "New Password" in label:
                return "new_password"
            elif "Confirm" in label:
                return "different_password"
            return "old_password"
        mock_streamlit.text_input.side_effect = mock_text_input
        
        # Test change password form with mismatched passwords
        middleware.show_change_password_form()
        
        # Verify error message is shown
        mock_streamlit.error.assert_called_with("New passwords do not match.")

    def test_user_menu_profile_flow(self, middleware, mock_streamlit, test_user):
        """Test user menu profile settings flow."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        middleware.auth_service.validate_token.return_value = test_user
        
        # Setup profile button click
        def mock_button(label):
            if "Profile Settings" in label:
                return True
            return False
        mock_streamlit.button.side_effect = mock_button
        
        # Test user menu with profile click
        middleware.show_user_menu()
        
        # Verify profile settings is shown
        mock_streamlit.subheader.assert_any_call("👤 Profile Settings")

    def test_user_menu_logout_flow(self, middleware, mock_streamlit, test_user):
        """Test user menu logout flow."""
        # Setup authenticated state
        mock_streamlit.session_state['auth_token'] = 'test_token'
        mock_streamlit.session_state['user'] = test_user
        middleware.auth_service.validate_token.return_value = test_user
        
        # Setup logout button click
        def mock_button(label):
            if "Logout" in label:
                return True
            return False
        mock_streamlit.button.side_effect = mock_button
        
        # Test user menu with logout click
        middleware.show_user_menu()
        
        # Verify logout is called
        middleware.auth_service.logout_user.assert_called_once_with('test_token')
        mock_streamlit.success.assert_called_with("Logged out successfully!")