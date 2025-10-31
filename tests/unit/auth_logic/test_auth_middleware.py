"""
Unit tests for Streamlit authentication middleware.

Tests the actual AuthMiddleware class with comprehensive mocking of Streamlit components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

from auth.middleware import AuthMiddleware
from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserProfile, UserRole, UserStatus


class MockStreamlitSession:
    """Mock Streamlit session state that behaves like a dictionary."""

    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def __delitem__(self, key):
        del self.data[key]


class MockStreamlitColumn:
    """Mock Streamlit column for form layouts."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_st():
    """Create comprehensive Streamlit mock."""
    st_mock = MagicMock()

    # Mock session_state
    st_mock.session_state = MockStreamlitSession()

    # Mock UI components
    st_mock.title = Mock()
    st_mock.markdown = Mock()
    st_mock.text_input = Mock(return_value="")
    st_mock.form_submit_button = Mock(return_value=False)
    st_mock.spinner = Mock()
    st_mock.success = Mock()
    st_mock.error = Mock()
    st_mock.info = Mock()
    st_mock.button = Mock(return_value=False)
    st_mock.subheader = Mock()
    st_mock.caption = Mock()
    st_mock.selectbox = Mock(return_value="Light")
    st_mock.sidebar = Mock()
    st_mock.form = Mock()
    st_mock.columns = Mock(return_value=(MockStreamlitColumn(), MockStreamlitColumn()))
    st_mock.rerun = Mock()
    st_mock.text_input = Mock(return_value="")

    # Mock context managers
    st_mock.form.return_value.__enter__ = Mock(return_value=None)
    st_mock.form.return_value.__exit__ = Mock(return_value=None)
    st_mock.spinner.return_value.__enter__ = Mock(return_value=None)
    st_mock.spinner.return_value.__exit__ = Mock(return_value=None)
    st_mock.sidebar.return_value.__enter__ = Mock(return_value=None)
    st_mock.sidebar.return_value.__exit__ = Mock(return_value=None)

    return st_mock


@pytest.fixture
def auth_service():
    """Create mock auth service."""
    service = Mock(spec=AuthService)
    return service


@pytest.fixture
def auth_middleware(auth_service, mock_st):
    """Create auth middleware with mocked Streamlit."""
    with patch('auth.middleware.st', mock_st):
        middleware = AuthMiddleware(auth_service)
        return middleware


class TestAuthMiddlewareInitialization:
    """Test AuthMiddleware initialization."""

    def test_initialization(self, auth_service, mock_st):
        """Test middleware initialization."""
        with patch('auth.middleware.st', mock_st):
            middleware = AuthMiddleware(auth_service)

            assert middleware.auth_service == auth_service

    def test_initialization_with_none_service(self, mock_st):
        """Test initialization with None service raises error."""
        with patch('auth.middleware.st', mock_st):
            with pytest.raises(AttributeError):
                AuthMiddleware(None)


class TestAuthMiddlewareAuthentication:
    """Test authentication methods."""

    def test_is_authenticated_no_token(self, auth_middleware, auth_service, mock_st):
        """Test is_authenticated returns False when no token."""
        # Ensure no token in session
        assert 'auth_token' not in mock_st.session_state.data

        result = auth_middleware.is_authenticated()

        assert result is False
        auth_service.validate_token.assert_not_called()

    def test_is_authenticated_with_invalid_token(self, auth_middleware, auth_service, mock_st):
        """Test is_authenticated with invalid token."""
        mock_st.session_state.data['auth_token'] = 'invalid_token'
        auth_service.validate_token.return_value = None

        result = auth_middleware.is_authenticated()

        assert result is False
        auth_service.validate_token.assert_called_once_with('invalid_token')

    def test_is_authenticated_with_valid_token(self, auth_middleware, auth_service, mock_st):
        """Test is_authenticated with valid token."""
        mock_user = Mock(spec=UserProfile)
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        result = auth_middleware.is_authenticated()

        assert result is True
        auth_service.validate_token.assert_called_once_with('valid_token')

    def test_get_current_user_no_token(self, auth_middleware, auth_service, mock_st):
        """Test get_current_user returns None when no token."""
        result = auth_middleware.get_current_user()

        assert result is None
        auth_service.validate_token.assert_not_called()

    def test_get_current_user_with_token(self, auth_middleware, auth_service, mock_st):
        """Test get_current_user with valid token."""
        mock_user = Mock(spec=UserProfile)
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        result = auth_middleware.get_current_user()

        assert result == mock_user
        auth_service.validate_token.assert_called_once_with('valid_token')


class TestAuthMiddlewareLogin:
    """Test login functionality."""

    def test_login_user_success(self, auth_middleware, auth_service, mock_st):
        """Test successful user login."""
        # Setup mock response
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = 'user_123'
        mock_session = Mock(spec=AuthSession)
        mock_session.created_at = datetime.now()

        auth_result = AuthResult(
            success=True,
            user=mock_user,
            token='jwt_token_123',
            session=mock_session
        )

        auth_service.login_user.return_value = auth_result

        # Mock helper methods
        with patch.object(auth_middleware, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_middleware, '_get_user_agent', return_value='test_agent'):

            result = auth_middleware.login_user('test@example.com', 'password123')

        assert result.success is True
        assert result.user == mock_user
        assert result.token == 'jwt_token_123'

        # Verify session state updates
        assert mock_st.session_state.data['auth_token'] == 'jwt_token_123'
        assert mock_st.session_state.data['user'] == mock_user
        assert 'auth_time' in mock_st.session_state.data

        # Verify service call
        auth_service.login_user.assert_called_once_with(
            email='test@example.com',
            password='password123',
            ip_address='127.0.0.1',
            user_agent='test_agent'
        )

    def test_login_user_failure(self, auth_middleware, auth_service, mock_st):
        """Test failed user login."""
        auth_result = AuthResult(success=False, error_message='Invalid credentials')
        auth_service.login_user.return_value = auth_result

        with patch.object(auth_middleware, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_middleware, '_get_user_agent', return_value='test_agent'):

            result = auth_middleware.login_user('test@example.com', 'wrong_password')

        assert result.success is False
        assert result.error_message == 'Invalid credentials'

        # Verify session state not updated
        assert 'auth_token' not in mock_st.session_state.data
        assert 'user' not in mock_st.session_state.data

    def test_login_user_no_session(self, auth_middleware, auth_service, mock_st):
        """Test login success without session object."""
        mock_user = Mock(spec=UserProfile)
        auth_result = AuthResult(success=True, user=mock_user, token='jwt_token_123')
        auth_service.login_user.return_value = auth_result

        with patch.object(auth_middleware, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_middleware, '_get_user_agent', return_value='test_agent'):

            result = auth_middleware.login_user('test@example.com', 'password123')

        assert result.success is True
        assert mock_st.session_state.data['auth_time'] is None


class TestAuthMiddlewareLogout:
    """Test logout functionality."""

    def test_logout_user_with_token(self, auth_middleware, auth_service, mock_st):
        """Test logout when user has token."""
        mock_st.session_state.data.update({
            'auth_token': 'jwt_token_123',
            'user': Mock(spec=UserProfile),
            'auth_time': datetime.now()
        })

        auth_middleware.logout_user()

        # Verify service logout called
        auth_service.logout_user.assert_called_once_with('jwt_token_123')

        # Verify session state cleared
        assert 'auth_token' not in mock_st.session_state.data
        assert 'user' not in mock_st.session_state.data
        assert 'auth_time' not in mock_st.session_state.data

    def test_logout_user_no_token(self, auth_middleware, auth_service, mock_st):
        """Test logout when no token exists."""
        auth_middleware.logout_user()

        # Verify service not called
        auth_service.logout_user.assert_not_called()

        # Session state should remain unchanged
        assert len(mock_st.session_state.data) == 0


class TestAuthMiddlewareDecorators:
    """Test authentication decorators."""

    def test_login_required_authenticated(self, auth_middleware, auth_service, mock_st):
        """Test login_required decorator when user is authenticated."""
        mock_user = Mock(spec=UserProfile)
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        @auth_middleware.login_required
        def protected_function():
            return "protected content"

        with patch.object(auth_middleware, 'show_login_form') as mock_show_login:
            result = protected_function()

        assert result == "protected content"
        mock_show_login.assert_not_called()

    def test_login_required_not_authenticated(self, auth_middleware, auth_service, mock_st):
        """Test login_required decorator when user is not authenticated."""
        auth_service.validate_token.return_value = None

        @auth_middleware.login_required
        def protected_function():
            return "protected content"

        with patch.object(auth_middleware, 'show_login_form') as mock_show_login:
            result = protected_function()

        assert result is None
        mock_show_login.assert_called_once()

    def test_role_required_authenticated_correct_role(self, auth_middleware, auth_service, mock_st):
        """Test role_required decorator with correct role."""
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.THERAPIST
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        @auth_middleware.role_required([UserRole.THERAPIST, UserRole.ADMIN])
        def admin_function():
            return "admin content"

        result = admin_function()

        assert result == "admin content"

    def test_role_required_authenticated_wrong_role(self, auth_middleware, auth_service, mock_st):
        """Test role_required decorator with insufficient role."""
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.PATIENT
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        @auth_middleware.role_required([UserRole.THERAPIST])
        def therapist_function():
            return "therapist content"

        with patch('auth.middleware.st') as mock_st_patch:
            result = therapist_function()

        assert result is None
        mock_st_patch.error.assert_called_once_with("Access denied. Insufficient permissions.")

    def test_role_required_not_authenticated(self, auth_middleware, auth_service, mock_st):
        """Test role_required decorator when not authenticated."""
        auth_service.validate_token.return_value = None

        @auth_middleware.role_required([UserRole.ADMIN])
        def admin_function():
            return "admin content"

        with patch.object(auth_middleware, 'show_login_form') as mock_show_login:
            result = admin_function()

        assert result is None
        mock_show_login.assert_called_once()


class TestAuthMiddlewareHelperMethods:
    """Test helper methods."""

    def test_get_client_ip(self, auth_middleware):
        """Test _get_client_ip method."""
        result = auth_middleware._get_client_ip()

        assert result == "streamlit_client"

    def test_get_user_agent(self, auth_middleware):
        """Test _get_user_agent method."""
        result = auth_middleware._get_user_agent()

        assert result == "streamlit_browser"


class TestAuthMiddlewareUIForms:
    """Test UI form display methods."""

    def test_show_login_form_basic_display(self, auth_middleware, mock_st):
        """Test basic login form display."""
        auth_middleware.show_login_form()

        # Verify UI components called
        mock_st.title.assert_called_once_with("🔐 Login Required")
        mock_st.markdown.assert_called_with("Please log in to access the AI Therapist.")
        mock_st.form.assert_called_with("login_form")

    def test_show_login_form_successful_login(self, auth_middleware, auth_service, mock_st):
        """Test successful login through form."""
        # Setup form inputs
        mock_st.text_input.side_effect = ['test@example.com', 'password123']
        mock_st.form_submit_button.side_effect = [True, False]  # Login clicked, register not

        # Setup successful login
        mock_user = Mock(spec=UserProfile)
        auth_result = AuthResult(success=True, user=mock_user, token='jwt_token')
        auth_service.login_user.return_value = auth_result

        with patch.object(auth_middleware, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_middleware, '_get_user_agent', return_value='test_agent'):

            auth_middleware.show_login_form()

        # Verify login attempt
        auth_service.login_user.assert_called_once_with(
            email='test@example.com',
            password='password123',
            ip_address='127.0.0.1',
            user_agent='test_agent'
        )

        # Verify success message
        mock_st.success.assert_called_with("Login successful!")
        mock_st.rerun.assert_called()

    def test_show_login_form_failed_login(self, auth_middleware, auth_service, mock_st):
        """Test failed login through form."""
        # Setup form inputs
        mock_st.text_input.side_effect = ['test@example.com', 'wrong_password']
        mock_st.form_submit_button.side_effect = [True, False]  # Login clicked, register not

        # Setup failed login
        auth_result = AuthResult(success=False, error_message='Invalid credentials')
        auth_service.login_user.return_value = auth_result

        with patch.object(auth_middleware, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_middleware, '_get_user_agent', return_value='test_agent'):

            auth_middleware.show_login_form()

        # Verify error message
        mock_st.error.assert_called_with("Login failed: Invalid credentials")

    def test_show_login_form_register_button(self, auth_middleware, mock_st):
        """Test register button click."""
        mock_st.form_submit_button.side_effect = [False, True]  # Login not, register clicked

        auth_middleware.show_login_form()

        # Verify register state set
        assert mock_st.session_state.data.get('show_register') is True
        mock_st.rerun.assert_called()

    def test_show_login_form_empty_fields(self, auth_middleware, mock_st):
        """Test login with empty fields."""
        mock_st.text_input.side_effect = ['', '']  # Empty email and password
        mock_st.form_submit_button.side_effect = [True, False]  # Login clicked

        auth_middleware.show_login_form()

        # Verify error message for empty fields
        mock_st.error.assert_called_with("Please enter both email and password.")

    def test_show_register_form_success(self, auth_middleware, auth_service, mock_st):
        """Test successful registration."""
        # Setup register form state
        mock_st.session_state.data['show_register'] = True

        # Setup form inputs
        mock_st.text_input.side_effect = ['John Doe', 'new@example.com', 'password123', 'password123']

        # Setup successful registration
        mock_user = Mock(spec=UserProfile)
        auth_result = AuthResult(success=True, user=mock_user)
        auth_service.register_user.return_value = auth_result

        auth_middleware.show_login_form()

        # Verify registration call
        auth_service.register_user.assert_called_once_with(
            email='new@example.com',
            password='password123',
            full_name='John Doe'
        )

        # Verify success message
        mock_st.success.assert_called_with("Account created successfully! Please log in.")

    def test_show_register_form_validation_errors(self, auth_middleware, mock_st):
        """Test registration form validation."""
        mock_st.session_state.data['show_register'] = True

        # Test empty fields
        mock_st.text_input.side_effect = ['', 'email@test.com', 'pass123', 'pass123']
        mock_st.form_submit_button.side_effect = [True, False]  # Register clicked

        auth_middleware.show_login_form()

        mock_st.error.assert_called_with("Please fill in all fields.")

    def test_show_register_form_password_mismatch(self, auth_middleware, mock_st):
        """Test password mismatch validation."""
        mock_st.session_state.data['show_register'] = True

        # Passwords don't match
        mock_st.text_input.side_effect = ['John Doe', 'email@test.com', 'pass123', 'different']
        mock_st.form_submit_button.side_effect = [True, False]  # Register clicked

        auth_middleware.show_login_form()

        mock_st.error.assert_called_with("Passwords do not match.")

    def test_show_password_reset_form_success(self, auth_middleware, auth_service, mock_st):
        """Test successful password reset."""
        mock_st.session_state.data['show_reset'] = True
        mock_st.text_input.return_value = 'user@example.com'

        auth_result = AuthResult(success=True)
        auth_service.initiate_password_reset.return_value = auth_result

        auth_middleware.show_login_form()

        auth_service.initiate_password_reset.assert_called_once_with('user@example.com')
        mock_st.success.assert_called_with("Password reset link sent to your email.")

    def test_show_user_menu_authenticated(self, auth_middleware, auth_service, mock_st):
        """Test user menu display for authenticated user."""
        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = 'John Doe'
        mock_user.role = UserRole.PATIENT
        mock_user.last_login = datetime.now()

        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        auth_middleware.show_user_menu()

        # Verify sidebar created
        mock_st.sidebar.assert_called_once()

        # Verify user info displayed
        mock_st.subheader.assert_called_with('👤 John Doe')
        mock_st.caption.assert_any_call('Role: Patient')

    def test_show_user_menu_not_authenticated(self, auth_middleware, mock_st):
        """Test user menu when not authenticated."""
        auth_middleware.show_user_menu()

        # Should return early without displaying anything
        mock_st.sidebar.assert_not_called()

    def test_show_profile_settings_authenticated(self, auth_middleware, auth_service, mock_st):
        """Test profile settings display."""
        mock_st.session_state.data['show_profile'] = True

        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = 'John Doe'
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        auth_middleware.show_user_menu()

        # Verify profile form created
        mock_st.subheader.assert_called_with('👤 Profile Settings')

    def test_show_change_password_form_success(self, auth_middleware, auth_service, mock_st):
        """Test successful password change."""
        mock_st.session_state.data['show_change_password'] = True

        # Setup form inputs
        mock_st.text_input.side_effect = ['oldpass123', 'newpass123', 'newpass123']

        # Setup authenticated user
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = 'user_123'
        mock_st.session_state.data['auth_token'] = 'valid_token'
        auth_service.validate_token.return_value = mock_user

        # Setup successful password change
        auth_result = AuthResult(success=True)
        auth_service.change_password.return_value = auth_result

        auth_middleware.show_user_menu()

        # Verify password change call
        auth_service.change_password.assert_called_once_with('user_123', 'oldpass123', 'newpass123')

        # Verify success message
        mock_st.success.assert_called_with("Password changed successfully!")

    def test_show_change_password_form_validation(self, auth_middleware, mock_st):
        """Test password change validation."""
        mock_st.session_state.data['show_change_password'] = True

        # Empty fields
        mock_st.text_input.side_effect = ['', 'newpass', 'newpass']

        auth_middleware.show_user_menu()

        mock_st.error.assert_called_with("Please fill in all fields.")

    def test_show_change_password_form_mismatch(self, auth_middleware, mock_st):
        """Test password mismatch in change form."""
        mock_st.session_state.data['show_change_password'] = True

        # Passwords don't match
        mock_st.text_input.side_effect = ['oldpass', 'newpass', 'different']

        auth_middleware.show_user_menu()

        mock_st.error.assert_called_with("New passwords do not match.")

    def test_show_change_password_form_too_short(self, auth_middleware, mock_st):
        """Test password length validation."""
        mock_st.session_state.data['show_change_password'] = True

        # Password too short
        mock_st.text_input.side_effect = ['oldpass', 'short', 'short']

        auth_middleware.show_user_menu()

        mock_st.error.assert_called_with("New password must be at least 8 characters.")
