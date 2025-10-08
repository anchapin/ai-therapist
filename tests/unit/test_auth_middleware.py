"""
Fixed unit tests for auth/middleware.py with proper session_state mocking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Mock streamlit to avoid import issues
with patch.dict('sys.modules', {'streamlit': Mock()}):
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
        middleware = AuthMiddleware(mock_auth_service)
        return middleware
    
    def test_auth_middleware_initialization(self, mock_auth_service):
        """Test auth middleware initialization."""
        middleware = AuthMiddleware(mock_auth_service)
        assert middleware.auth_service == mock_auth_service
    
    def test_is_authenticated_true(self, auth_middleware):
        """Test is_authenticated when user is authenticated."""
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = 'valid_token'
        mock_session_state.__contains__ = Mock(return_value=True)
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.is_authenticated()
            
            assert result is True
            auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
            mock_session_state.get.assert_called_once_with('auth_token')
    
    def test_is_authenticated_false_no_token(self, auth_middleware):
        """Test is_authenticated when no token is present."""
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = None
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.is_authenticated()
            
            assert result is False
            auth_middleware.auth_service.validate_token.assert_not_called()
            mock_session_state.get.assert_called_once_with('auth_token')
    
    def test_is_authenticated_false_invalid_token(self, auth_middleware):
        """Test is_authenticated when token is invalid."""
        auth_middleware.auth_service.validate_token.return_value = None
        
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = 'invalid_token'
        mock_session_state.__contains__ = Mock(return_value=True)
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.is_authenticated()
            
            assert result is False
            auth_middleware.auth_service.validate_token.assert_called_once_with('invalid_token')
            mock_session_state.get.assert_called_once_with('auth_token')
    
    def test_get_current_user_success(self, auth_middleware):
        """Test get_current_user when user is authenticated."""
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = 'valid_token'
        mock_session_state.__contains__ = Mock(return_value=True)
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.get_current_user()
            
            assert result == mock_user
            auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
            mock_session_state.get.assert_called_once_with('auth_token')
    
    def test_get_current_user_no_token(self, auth_middleware):
        """Test get_current_user when no token is present."""
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = None
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.get_current_user()
            
            assert result is None
            auth_middleware.auth_service.validate_token.assert_not_called()
            mock_session_state.get.assert_called_once_with('auth_token')
    
    def test_login_user_success(self, auth_middleware):
        """Test successful user login."""
        mock_user = Mock(spec=UserProfile)
        mock_session = Mock()
        mock_session.created_at = datetime.now()
        
        auth_result = AuthResult(
            success=True,
            user=mock_user,
            token="jwt_token_123",
            session=mock_session
        )
        
        auth_middleware.auth_service.login_user.return_value = auth_result
        
        # Mock the helper methods
        with patch.object(auth_middleware, '_get_client_ip', return_value="192.168.1.1"):
            with patch.object(auth_middleware, '_get_user_agent', return_value="Mozilla/5.0"):
                # Create a proper mock for session_state
                mock_session_state = Mock()
                
                with patch('auth.middleware.st') as mock_st:
                    mock_st.session_state = mock_session_state
                    result = auth_middleware.login_user("test@example.com", "SecurePass123")
        
        assert result == auth_result
        
        # Verify session state was set
        mock_session_state.__setitem__.assert_any_call('auth_token', "jwt_token_123")
        mock_session_state.__setitem__.assert_any_call('user', mock_user)
        mock_session_state.__setitem__.assert_any_call('auth_time', mock_session.created_at)
        
        auth_middleware.auth_service.login_user.assert_called_once_with(
            email="test@example.com",
            password="SecurePass123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
    
    def test_login_user_failure(self, auth_middleware):
        """Test failed user login."""
        auth_result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        auth_middleware.auth_service.login_user.return_value = auth_result
        
        # Mock the helper methods
        with patch.object(auth_middleware, '_get_client_ip', return_value="192.168.1.1"):
            with patch.object(auth_middleware, '_get_user_agent', return_value="Mozilla/5.0"):
                # Create a proper mock for session_state
                mock_session_state = Mock()
                
                with patch('auth.middleware.st') as mock_st:
                    mock_st.session_state = mock_session_state
                    result = auth_middleware.login_user("test@example.com", "wrongpassword")
        
        assert result == auth_result
        
        # Verify session state was not modified
        mock_session_state.__setitem__.assert_not_called()
        
        auth_middleware.auth_service.login_user.assert_called_once_with(
            email="test@example.com",
            password="wrongpassword",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
    
    def test_logout_user_success(self, auth_middleware):
        """Test successful user logout."""
        auth_middleware.auth_service.logout_user.return_value = True
        
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = 'jwt_token_123'
        mock_session_state.__contains__ = Mock(return_value=True)
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.logout_user()
            
            assert result is None  # logout_user doesn't return anything
            
            # Verify token was passed to logout_user
            auth_middleware.auth_service.logout_user.assert_called_once_with('jwt_token_123')
            
            # Verify session state keys were deleted
            mock_session_state.__delitem__.assert_any_call('auth_token')
            mock_session_state.__delitem__.assert_any_call('user')
            mock_session_state.__delitem__.assert_any_call('auth_time')
    
    def test_logout_user_no_token(self, auth_middleware):
        """Test logout when no token is present."""
        # Create a proper mock for session_state
        mock_session_state = Mock()
        mock_session_state.get.return_value = None
        mock_session_state.__contains__ = Mock(return_value=False)
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = mock_session_state
            result = auth_middleware.logout_user()
            
            assert result is None  # logout_user doesn't return anything
            
            # Verify logout_user was not called
            auth_middleware.auth_service.logout_user.assert_not_called()
            
            # Verify session state was not modified
            mock_session_state.__delitem__.assert_not_called()
    
    def test_get_client_ip(self, auth_middleware):
        """Test getting client IP address."""
        result = auth_middleware._get_client_ip()
        
        assert result == "streamlit_client"
    
    def test_get_user_agent(self, auth_middleware):
        """Test getting user agent."""
        result = auth_middleware._get_user_agent()
        
        assert result == "streamlit_browser"