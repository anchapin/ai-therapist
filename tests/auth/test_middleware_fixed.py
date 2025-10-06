"""
Fixed tests for auth middleware to improve coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import streamlit as st

from auth.middleware import AuthMiddleware
from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserProfile, UserRole, UserStatus


class TestAuthMiddlewareFixed:
    """Fixed tests for AuthMiddleware to improve coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_auth_service = Mock(spec=AuthService)
        self.middleware = AuthMiddleware(self.mock_auth_service)
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
    def test_auth_middleware_init(self):
        """Test AuthMiddleware initialization."""
        assert self.middleware.auth_service == self.mock_auth_service

    @patch('auth.middleware.st')
    def test_login_required_decorator_authenticated(self, mock_st):
        """Test login_required decorator when user is authenticated."""
        # Setup
        mock_st.session_state = {'auth_token': 'valid_token'}
        self.mock_auth_service.validate_token.return_value = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test
        @self.middleware.login_required
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"

    @patch('auth.middleware.st')
    def test_login_required_decorator_not_authenticated(self, mock_st):
        """Test login_required decorator when user is not authenticated."""
        # Setup
        mock_st.session_state = {}
        
        # Test
        @self.middleware.login_required
        def test_function():
            return "success"
        
        result = test_function()
        assert result is None
        mock_st.session_state.get.assert_called_with('auth_token')

    @patch('auth.middleware.st')
    def test_role_required_decorator_success(self, mock_st):
        """Test role_required decorator with correct role."""
        # Setup
        mock_st.session_state = {'auth_token': 'valid_token'}
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.THERAPIST,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.mock_auth_service.validate_token.return_value = mock_user
        
        # Test
        @self.middleware.role_required([UserRole.THERAPIST])
        def admin_function():
            return "admin_success"
        
        result = admin_function()
        assert result == "admin_success"

    @patch('auth.middleware.st')
    def test_role_required_decorator_wrong_role(self, mock_st):
        """Test role_required decorator with wrong role."""
        # Setup
        mock_st.session_state = {'auth_token': 'valid_token'}
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.mock_auth_service.validate_token.return_value = mock_user
        
        # Test
        @self.middleware.role_required([UserRole.THERAPIST])
        def admin_function():
            return "admin_success"
        
        result = admin_function()
        assert result is None
        mock_st.error.assert_called_with("Access denied. Insufficient permissions.")

    @patch('auth.middleware.st')
    def test_is_authenticated_success(self, mock_st):
        """Test is_authenticated when user is valid."""
        # Setup
        mock_st.session_state = {'auth_token': 'valid_token'}
        self.mock_auth_service.validate_token.return_value = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test
        result = self.middleware.is_authenticated()
        assert result is True

    @patch('auth.middleware.st')
    def test_is_authenticated_no_token(self, mock_st):
        """Test is_authenticated when no token exists."""
        # Setup
        mock_st.session_state = {}
        
        # Test
        result = self.middleware.is_authenticated()
        assert result is False

    @patch('auth.middleware.st')
    def test_is_authenticated_invalid_token(self, mock_st):
        """Test is_authenticated when token is invalid."""
        # Setup
        mock_st.session_state = {'auth_token': 'invalid_token'}
        self.mock_auth_service.validate_token.return_value = None
        
        # Test
        result = self.middleware.is_authenticated()
        assert result is False

    @patch('auth.middleware.st')
    def test_get_current_user_success(self, mock_st):
        """Test get_current_user when user is valid."""
        # Setup
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_st.session_state = {'auth_token': 'valid_token'}
        self.mock_auth_service.validate_token.return_value = mock_user
        
        # Test
        result = self.middleware.get_current_user()
        assert result == mock_user

    @patch('auth.middleware.st')
    def test_get_current_user_no_token(self, mock_st):
        """Test get_current_user when no token exists."""
        # Setup
        mock_st.session_state = {}
        
        # Test
        result = self.middleware.get_current_user()
        assert result is None

    @patch('auth.middleware.st')
    def test_login_user_success(self, mock_st):
        """Test successful user login."""
        # Setup
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_session = AuthSession(
            session_id="session_123",
            user_id="test_user",
            created_at=datetime.now(),
            expires_at=datetime.now()
        )
        mock_result = AuthResult(
            success=True,
            user=mock_user,
            token="valid_token",
            session=mock_session
        )
        self.mock_auth_service.login_user.return_value = mock_result
        mock_st.session_state = {}
        
        # Test
        result = self.middleware.login_user("test@example.com", "password")
        
        # Verify
        assert result == mock_result
        assert mock_st.session_state.auth_token == "valid_token"
        assert mock_st.session_state.user == mock_user
        assert mock_st.session_state.auth_time == mock_session.created_at

    @patch('auth.middleware.st')
    def test_login_user_failure(self, mock_st):
        """Test failed user login."""
        # Setup
        mock_result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        self.mock_auth_service.login_user.return_value = mock_result
        mock_st.session_state = {}
        
        # Test
        result = self.middleware.login_user("test@example.com", "wrong_password")
        
        # Verify
        assert result == mock_result
        assert not hasattr(mock_st.session_state, 'auth_token')

    @patch('auth.middleware.st')
    def test_logout_user_with_token(self, mock_st):
        """Test logout user with existing token."""
        # Setup
        mock_session_state = {}
        mock_st.session_state = mock_session_state
        mock_session_state.auth_token = 'valid_token'
        mock_session_state.user = 'test_user'
        mock_session_state.auth_time = '2023-01-01'
        
        # Test
        self.middleware.logout_user()
        
        # Verify
        self.mock_auth_service.logout_user.assert_called_with('valid_token')
        assert 'auth_token' not in mock_st.session_state
        assert 'user' not in mock_st.session_state
        assert 'auth_time' not in mock_st.session_state

    @patch('auth.middleware.st')
    def test_logout_user_without_token(self, mock_st):
        """Test logout user without existing token."""
        # Setup
        mock_st.session_state = {}
        
        # Test
        self.middleware.logout_user()
        
        # Verify
        self.mock_auth_service.logout_user.assert_not_called()

    @patch('auth.middleware.st')
    def test_show_login_form(self, mock_st):
        """Test show_login_form method."""
        # Setup
        mock_st.session_state = {}
        
        # Create mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__ = MagicMock(return_value=None)
        mock_st.form.return_value = mock_form
        
        # Mock other streamlit functions
        mock_st.title = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.text_input = MagicMock(return_value="")
        mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        mock_st.form_submit_button = MagicMock(return_value=False)
        mock_st.button = MagicMock(return_value=False)
        
        # Test
        self.middleware.show_login_form()
        
        # Verify
        mock_st.title.assert_called_with("🔐 Login Required")
        mock_st.markdown.assert_called_with("Please log in to access the AI Therapist.")
        mock_st.form.assert_called_with("login_form")

    @patch('auth.middleware.st')
    def test_show_register_form(self, mock_st):
        """Test show_register_form method."""
        # Setup
        mock_st.session_state = {}
        
        # Create mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__ = MagicMock(return_value=None)
        mock_st.form.return_value = mock_form
        
        # Mock other streamlit functions
        mock_st.subheader = MagicMock()
        mock_st.text_input = MagicMock(return_value="")
        mock_st.caption = MagicMock()
        mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        mock_st.form_submit_button = MagicMock(return_value=False)
        
        # Test
        self.middleware.show_register_form()
        
        # Verify
        mock_st.subheader.assert_called_with("📝 Register New Account")
        mock_st.form.assert_called_with("register_form")

    @patch('auth.middleware.st')
    def test_show_password_reset_form(self, mock_st):
        """Test show_password_reset_form method."""
        # Setup
        mock_st.session_state = {}
        
        # Create mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__ = MagicMock(return_value=None)
        mock_st.form.return_value = mock_form
        
        # Mock other streamlit functions
        mock_st.subheader = MagicMock()
        mock_st.text_input = MagicMock(return_value="")
        mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        mock_st.form_submit_button = MagicMock(return_value=False)
        
        # Test
        self.middleware.show_password_reset_form()
        
        # Verify
        mock_st.subheader.assert_called_with("🔑 Reset Password")
        mock_st.form.assert_called_with("reset_form")

    @patch('auth.middleware.st')
    def test_show_user_menu(self, mock_st):
        """Test show_user_menu method."""
        # Setup
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_login=datetime.now()
        )
        mock_st.session_state = {}
        self.mock_auth_service.validate_token.return_value = mock_user
        
        # Create mock sidebar context manager
        mock_sidebar = MagicMock()
        mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
        mock_sidebar.__exit__ = MagicMock(return_value=None)
        mock_st.sidebar = mock_sidebar
        
        # Mock other streamlit functions
        mock_st.markdown = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.button = MagicMock(return_value=False)
        
        # Test
        self.middleware.show_user_menu()
        
        # Verify
        mock_st.sidebar.__enter__.assert_called_once()
        mock_st.markdown.assert_called_with("---")
        mock_st.subheader.assert_called_with("👤 Test User")

    @patch('auth.middleware.st')
    def test_show_profile_settings(self, mock_st):
        """Test show_profile_settings method."""
        # Setup
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_st.session_state = {}
        self.mock_auth_service.validate_token.return_value = mock_user
        
        # Create mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__ = MagicMock(return_value=None)
        mock_st.form.return_value = mock_form
        
        # Mock other streamlit functions
        mock_st.subheader = MagicMock()
        mock_st.text_input = MagicMock(return_value="Test User")
        mock_st.selectbox = MagicMock(return_value="Light")
        mock_st.form_submit_button = MagicMock(return_value=False)
        mock_st.info = MagicMock()
        
        # Test
        self.middleware.show_profile_settings()
        
        # Verify
        mock_st.subheader.assert_called_with("👤 Profile Settings")
        mock_st.form.assert_called_with("profile_form")

    @patch('auth.middleware.st')
    def test_show_change_password_form(self, mock_st):
        """Test show_change_password_form method."""
        # Setup
        mock_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_st.session_state = {}
        self.mock_auth_service.validate_token.return_value = mock_user
        
        # Create mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__ = MagicMock(return_value=None)
        mock_st.form.return_value = mock_form
        
        # Mock other streamlit functions
        mock_st.subheader = MagicMock()
        mock_st.text_input = MagicMock(return_value="")
        mock_st.form_submit_button = MagicMock(return_value=False)
        
        # Test
        self.middleware.show_change_password_form()
        
        # Verify
        mock_st.subheader.assert_called_with("🔑 Change Password")
        mock_st.form.assert_called_with("change_password_form")

    def test_get_client_ip(self):
        """Test _get_client_ip method."""
        result = self.middleware._get_client_ip()
        assert result == "streamlit_client"

    def test_get_user_agent(self):
        """Test _get_user_agent method."""
        result = self.middleware._get_user_agent()
        assert result == "streamlit_browser"