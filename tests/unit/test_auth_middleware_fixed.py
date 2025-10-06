"""
Fixed unit tests for auth/middleware.py with proper streamlit mocking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Create a comprehensive streamlit mock
mock_streamlit = Mock()
mock_streamlit.session_state = {}
mock_streamlit.title = Mock()
mock_streamlit.markdown = Mock()
mock_streamlit.text_input = Mock(return_value="")
mock_streamlit.text_area = Mock(return_value="")
mock_streamlit.form = Mock()
mock_streamlit.form_submit_button = Mock(return_value=False)
mock_streamlit.error = Mock()
mock_streamlit.success = Mock()
mock_streamlit.info = Mock()

# Mock streamlit module
with patch.dict('sys.modules', {'streamlit': mock_streamlit}):
    from auth.middleware import AuthMiddleware
    from auth.auth_service import AuthService, AuthResult
    from auth.user_model import UserProfile, UserRole


class TestAuthMiddleware:
    """Test AuthMiddleware functionality."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Create a mock auth service."""
        auth_service = Mock(spec=AuthService)
        return auth_service
    
    @pytest.fixture
    def auth_middleware(self, mock_auth_service):
        """Create auth middleware with mocked dependencies."""
        # Clear session state before each test
        mock_streamlit.session_state.clear()
        middleware = AuthMiddleware(mock_auth_service)
        return middleware
    
    def test_auth_middleware_initialization(self, mock_auth_service):
        """Test auth middleware initialization."""
        middleware = AuthMiddleware(mock_auth_service)
        assert middleware.auth_service == mock_auth_service
    
    def test_login_required_authenticated(self, auth_middleware):
        """Test login_required decorator when user is authenticated."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        @auth_middleware.login_required
        def protected_function():
            return "protected_content"
        
        result = protected_function()
        
        assert result == "protected_content"
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_login_required_not_authenticated(self, auth_middleware):
        """Test login_required decorator when user is not authenticated."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        @auth_middleware.login_required
        def protected_function():
            return "protected_content"
        
        result = protected_function()
        
        assert result is None
        mock_streamlit.title.assert_called_once_with("🔐 Login Required")
    
    def test_role_required_success(self, auth_middleware):
        """Test role_required decorator when user has required role."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.THERAPIST
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        @auth_middleware.role_required([UserRole.THERAPIST, UserRole.ADMIN])
        def protected_function():
            return "protected_content"
        
        result = protected_function()
        
        assert result == "protected_content"
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_role_required_insufficient_permissions(self, auth_middleware):
        """Test role_required decorator when user lacks required role."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.PATIENT
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        @auth_middleware.role_required([UserRole.THERAPIST, UserRole.ADMIN])
        def protected_function():
            return "protected_content"
        
        result = protected_function()
        
        assert result is None
        mock_streamlit.error.assert_called_once_with("Access denied. Insufficient permissions.")
    
    def test_is_authenticated_true(self, auth_middleware):
        """Test is_authenticated returns True for valid token."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        result = auth_middleware.is_authenticated()
        
        assert result is True
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_is_authenticated_false_no_token(self, auth_middleware):
        """Test is_authenticated returns False when no token."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        result = auth_middleware.is_authenticated()
        
        assert result is False
        auth_middleware.auth_service.validate_token.assert_not_called()
    
    def test_is_authenticated_false_invalid_token(self, auth_middleware):
        """Test is_authenticated returns False for invalid token."""
        # Mock streamlit session state with invalid token
        mock_streamlit.session_state = {'auth_token': 'invalid_token'}
        
        auth_middleware.auth_service.validate_token.return_value = None
        
        result = auth_middleware.is_authenticated()
        
        assert result is False
        auth_middleware.auth_service.validate_token.assert_called_once_with('invalid_token')
    
    def test_get_current_user_success(self, auth_middleware):
        """Test get_current_user returns user for valid token."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        result = auth_middleware.get_current_user()
        
        assert result == mock_user
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_get_current_user_no_token(self, auth_middleware):
        """Test get_current_user returns None when no token."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        result = auth_middleware.get_current_user()
        
        assert result is None
        auth_middleware.auth_service.validate_token.assert_not_called()
    
    def test_login_user_success(self, auth_middleware):
        """Test successful user login."""
        # Mock auth result
        mock_result = Mock(spec=AuthResult)
        mock_result.success = True
        mock_result.token = "jwt_token_123"
        mock_result.user = Mock(spec=UserProfile)
        mock_session = Mock()
        mock_session.created_at = datetime.now()
        mock_result.session = mock_session
        
        auth_middleware.auth_service.login_user.return_value = mock_result
        
        result = auth_middleware.login_user("test@example.com", "SecurePass123")
        
        assert result == mock_result
        assert mock_streamlit.session_state['auth_token'] == "jwt_token_123"
        assert mock_streamlit.session_state['user'] == mock_result.user
        assert mock_streamlit.session_state['auth_time'] == mock_session.created_at
        
        auth_middleware.auth_service.login_user.assert_called_once()
    
    def test_login_user_failure(self, auth_middleware):
        """Test failed user login."""
        # Mock auth result
        mock_result = Mock(spec=AuthResult)
        mock_result.success = False
        mock_result.token = None
        mock_result.user = None
        mock_result.session = None
        
        auth_middleware.auth_service.login_user.return_value = mock_result
        
        result = auth_middleware.login_user("test@example.com", "WrongPassword")
        
        assert result == mock_result
        assert 'auth_token' not in mock_streamlit.session_state
        assert 'user' not in mock_streamlit.session_state
        assert 'auth_time' not in mock_streamlit.session_state
        
        auth_middleware.auth_service.login_user.assert_called_once()
    
    def test_logout_user_success(self, auth_middleware):
        """Test successful user logout."""
        # Set up session state
        mock_streamlit.session_state = {
            'auth_token': 'valid_token',
            'user': Mock(spec=UserProfile),
            'auth_time': datetime.now()
        }
        
        result = auth_middleware.logout_user()
        
        assert result is None  # logout_user doesn't return anything
        assert 'auth_token' not in mock_streamlit.session_state
        assert 'user' not in mock_streamlit.session_state
        assert 'auth_time' not in mock_streamlit.session_state
        
        auth_middleware.auth_service.logout_user.assert_called_once_with('valid_token')
    
    def test_logout_user_no_token(self, auth_middleware):
        """Test logout when no token in session."""
        # Empty session state
        mock_streamlit.session_state = {}
        
        result = auth_middleware.logout_user()
        
        assert result is None  # logout_user doesn't return anything
        auth_middleware.auth_service.logout_user.assert_not_called()
    
    def test_show_login_form(self, auth_middleware):
        """Test login form display."""
        auth_middleware.show_login_form()
        
        mock_streamlit.title.assert_called_once_with("🔐 Login Required")
        mock_streamlit.markdown.assert_called_once_with("Please log in to access the AI Therapist.")
    
    def test_show_login_form_with_inputs(self, auth_middleware):
        """Test login form with user inputs."""
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        mock_streamlit.form.return_value = mock_form
        
        # Mock form inputs
        mock_streamlit.text_input.side_effect = ["test@example.com", "SecurePass123"]
        mock_streamlit.form_submit_button.return_value = True
        
        # Mock successful login
        mock_result = Mock(spec=AuthResult)
        mock_result.success = True
        mock_result.token = "jwt_token_123"
        mock_result.user = Mock(spec=UserProfile)
        mock_session = Mock()
        mock_session.created_at = datetime.now()
        mock_result.session = mock_session
        
        auth_middleware.auth_service.login_user.return_value = mock_result
        
        auth_middleware.show_login_form()
        
        # Verify form was created
        mock_streamlit.form.assert_called_once_with("login_form")
        
        # Verify inputs were requested
        assert mock_streamlit.text_input.call_count >= 2  # email and password
        
        # Verify login was attempted
        auth_middleware.auth_service.login_user.assert_called_once()
    
    def test_get_client_ip(self, auth_middleware):
        """Test client IP retrieval."""
        # This is a private method, but we can test it indirectly
        # through login_user which calls it
        mock_result = Mock(spec=AuthResult)
        mock_result.success = False
        auth_middleware.auth_service.login_user.return_value = mock_result
        
        auth_middleware.login_user("test@example.com", "password")
        
        # Verify login_user was called with ip_address and user_agent
        call_args = auth_middleware.auth_service.login_user.call_args
        assert 'ip_address' in call_args.kwargs
        assert 'user_agent' in call_args.kwargs
    
    def test_multiple_role_required(self, auth_middleware):
        """Test role_required decorator with multiple acceptable roles."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        # Test with ADMIN role
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.ADMIN
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        @auth_middleware.role_required([UserRole.THERAPIST, UserRole.ADMIN])
        def protected_function():
            return "protected_content"
        
        result = protected_function()
        
        assert result == "protected_content"
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_session_state_persistence(self, auth_middleware):
        """Test that session state persists across operations."""
        # Login user
        mock_result = Mock(spec=AuthResult)
        mock_result.success = True
        mock_result.token = "jwt_token_123"
        mock_result.user = Mock(spec=UserProfile)
        mock_session = Mock()
        mock_session.created_at = datetime.now()
        mock_result.session = mock_session
        
        auth_middleware.auth_service.login_user.return_value = mock_result
        
        auth_middleware.login_user("test@example.com", "password")
        
        # Verify user is authenticated
        assert auth_middleware.is_authenticated() is True
        
        # Verify we can get current user
        current_user = auth_middleware.get_current_user()
        assert current_user == mock_result.user
        
        # Logout and verify state is cleared
        auth_middleware.logout_user()
        assert auth_middleware.is_authenticated() is False
        assert auth_middleware.get_current_user() is None