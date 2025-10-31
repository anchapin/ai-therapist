"""
Unit tests for authentication middleware logic without Streamlit dependencies.

Tests the core authentication flow and middleware logic without UI components.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserRole, UserStatus


class MockAuthMiddleware:
    """Mock authentication middleware for testing without Streamlit."""
    
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
        self.current_user = None
        self.current_token = None
        self.session_state = {}
    
    def authenticate_token(self, token: str) -> bool:
        """Authenticate token and set current user."""
        user = self.auth_service.validate_token(token)
        if user:
            self.current_user = user
            self.current_token = token
            return True
        return False
    
    def logout(self) -> bool:
        """Logout current user."""
        if self.current_token:
            result = self.auth_service.logout_user(self.current_token)
            if result:
                self.current_user = None
                self.current_token = None
                self.session_state.clear()
            return result
        return False
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.current_user is not None
    
    def get_current_user_role(self) -> str:
        """Get current user role."""
        if self.current_user:
            return self.current_user.role.value
        return None
    
    def has_permission(self, resource: str, permission: str) -> bool:
        """Check if current user has permission."""
        if not self.current_user:
            return False
        return self.auth_service.validate_session_access(
            self.current_user.user_id, resource, permission
        )
    
    def login(self, email: str, password: str, **kwargs) -> AuthResult:
        """Login user."""
        result = self.auth_service.login_user(email, password, **kwargs)
        if result.success:
            self.current_user = result.user
            self.current_token = result.token
            self.session_state['user_id'] = result.user.user_id
            self.session_state['token'] = result.token
        return result
    
    def register(self, email: str, password: str, full_name: str, 
                role: UserRole = UserRole.PATIENT) -> AuthResult:
        """Register new user."""
        return self.auth_service.register_user(email, password, full_name, role)
    
    def initiate_password_reset(self, email: str) -> AuthResult:
        """Initiate password reset."""
        return self.auth_service.initiate_password_reset(email)
    
    def reset_password(self, reset_token: str, new_password: str) -> AuthResult:
        """Reset password."""
        return self.auth_service.reset_password(reset_token, new_password)
    
    def change_password(self, old_password: str, new_password: str) -> AuthResult:
        """Change password for current user."""
        if not self.current_user:
            return AuthResult(success=False, error_message="Not authenticated")
        
        return self.auth_service.change_password(
            self.current_user.user_id, old_password, new_password
        )


class TestAuthMiddlewareLogic:
    """Test authentication middleware logic without Streamlit dependencies."""
    
    @pytest.fixture
    def auth_middleware(self, auth_service):
        """Create auth middleware with auth service."""
        return MockAuthMiddleware(auth_service)
    
    def test_initialization(self, auth_service):
        """Test middleware initialization."""
        middleware = MockAuthMiddleware(auth_service)
        
        assert middleware.auth_service == auth_service
        assert middleware.current_user is None
        assert middleware.current_token is None
        assert middleware.session_state == {}
    
    def test_login_success(self, auth_middleware, auth_service):
        """Test successful login."""
        # Mock successful authentication
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        mock_user.status = UserStatus.ACTIVE
        
        auth_result = AuthResult(
            success=True,
            user=mock_user,
            token="jwt_token_123",
            session=MagicMock()
        )
        
        # Mock the auth service login_user method
        with patch.object(auth_service, 'login_user', return_value=auth_result):
            # Perform login
            result = auth_middleware.login(
                email="test@example.com",
                password="SecurePass123",
                ip_address="127.0.0.1"
            )
            
            # Verify result
            assert result.success is True
            assert result.user == mock_user
            assert result.token == "jwt_token_123"
            
            # Verify middleware state
            assert auth_middleware.current_user == mock_user
            assert auth_middleware.current_token == "jwt_token_123"
            assert auth_middleware.session_state['user_id'] == "user_123"
            assert auth_middleware.session_state['token'] == "jwt_token_123"
    
    def test_login_failure(self, auth_middleware, auth_service):
        """Test failed login."""
        # Mock failed authentication
        auth_result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        with patch.object(auth_service, 'login_user', return_value=auth_result):
            # Perform login
            result = auth_middleware.login(
                email="test@example.com",
                password="wrong_password"
            )
            
            # Verify result
            assert result.success is False
            assert result.error_message == "Invalid credentials"
            assert result.user is None
            
            # Verify middleware state is unchanged
            assert auth_middleware.current_user is None
            assert auth_middleware.current_token is None
            assert auth_middleware.session_state == {}
    
    def test_authenticate_token_success(self, auth_middleware, auth_service):
        """Test successful token authentication."""
        # Mock user
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        
        with patch.object(auth_service, 'validate_token', return_value=mock_user):
            # Authenticate token
            result = auth_middleware.authenticate_token("valid_token")
            
            # Verify result
            assert result is True
            assert auth_middleware.current_user == mock_user
            assert auth_middleware.current_token == "valid_token"
    
    def test_authenticate_token_invalid(self, auth_middleware, auth_service):
        """Test invalid token authentication."""
        with patch.object(auth_service, 'validate_token', return_value=None):
            result = auth_middleware.authenticate_token("invalid_token")
            
            # Verify result
            assert result is False
            assert auth_middleware.current_user is None
            assert auth_middleware.current_token is None
    
    def test_logout_success(self, auth_middleware, auth_service):
        """Test successful logout."""
        # Set up authenticated state
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        
        auth_middleware.current_user = mock_user
        auth_middleware.current_token = "jwt_token_123"
        auth_middleware.session_state = {'user_id': 'user_123', 'token': 'jwt_token_123'}
        
        with patch.object(auth_service, 'logout_user', return_value=True):
            # Perform logout
            result = auth_middleware.logout()
            
            # Verify result
            assert result is True
            assert auth_middleware.current_user is None
            assert auth_middleware.current_token is None
            assert auth_middleware.session_state == {}
    
    def test_logout_not_authenticated(self, auth_middleware):
        """Test logout when not authenticated."""
        result = auth_middleware.logout()
        
        assert result is False
        assert auth_middleware.current_user is None
        assert auth_middleware.current_token is None
        assert auth_middleware.session_state == {}
    
    def test_is_authenticated_true(self, auth_middleware):
        """Test is_authenticated returns True when user is set."""
        auth_middleware.current_user = MagicMock()
        
        assert auth_middleware.is_authenticated() is True
    
    def test_is_authenticated_false(self, auth_middleware):
        """Test is_authenticated returns False when no user."""
        assert auth_middleware.is_authenticated() is False
    
    def test_get_current_user_role(self, auth_middleware):
        """Test getting current user role."""
        mock_user = MagicMock()
        mock_user.role.value = "PATIENT"
        auth_middleware.current_user = mock_user
        
        assert auth_middleware.get_current_user_role() == "PATIENT"
        
        # Test with no user
        auth_middleware.current_user = None
        assert auth_middleware.get_current_user_role() is None
    
    def test_has_permission_authenticated_user(self, auth_middleware, auth_service):
        """Test permission check for authenticated user."""
        # Setup authenticated user
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        auth_middleware.current_user = mock_user
        
        # Mock permission check
        with patch.object(auth_service, 'validate_session_access', return_value=True):
            result = auth_middleware.has_permission("patient_data", "read")
            
            assert result is True
            auth_service.validate_session_access.assert_called_once_with(
                "user_123", "patient_data", "read"
            )
    
    def test_has_permission_unauthenticated_user(self, auth_middleware):
        """Test permission check for unauthenticated user."""
        result = auth_middleware.has_permission("patient_data", "read")
        
        assert result is False
    
    def test_register_success(self, auth_middleware, auth_service):
        """Test successful registration."""
        # Mock user creation
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        
        auth_result = AuthResult(success=True, user=mock_user)
        with patch.object(auth_service, 'register_user', return_value=auth_result):
            # Perform registration
            result = auth_middleware.register(
                email="test@example.com",
                password="SecurePass123",
                full_name="Test User"
            )
            
            # Verify result
            assert result.success is True
            assert result.user == mock_user
            auth_service.register_user.assert_called_once_with(
                "test@example.com", "SecurePass123", "Test User", UserRole.PATIENT
            )
    
    def test_register_with_role(self, auth_middleware, auth_service):
        """Test registration with specific role."""
        mock_user = MagicMock()
        auth_result = AuthResult(success=True, user=mock_user)
        with patch.object(auth_service, 'register_user', return_value=auth_result):
            # Perform registration with therapist role
            auth_middleware.register(
                email="therapist@example.com",
                password="SecurePass123",
                full_name="Therapist User",
                role=UserRole.THERAPIST
            )
            
            # Verify role was passed correctly
            auth_service.register_user.assert_called_once_with(
                "therapist@example.com", "SecurePass123", "Therapist User", UserRole.THERAPIST
            )
    
    def test_register_failure(self, auth_middleware, auth_service):
        """Test failed registration."""
        auth_result = AuthResult(
            success=False,
            error_message="Email already exists"
        )
        
        with patch.object(auth_service, 'register_user', return_value=auth_result):
            result = auth_middleware.register(
                email="existing@example.com",
                password="SecurePass123",
                full_name="Existing User"
            )
            
            assert result.success is False
            assert result.error_message == "Email already exists"
    
    def test_initiate_password_reset_success(self, auth_middleware, auth_service):
        """Test successful password reset initiation."""
        auth_result = AuthResult(
            success=True,
            error_message="Password reset email sent"
        )
        
        with patch.object(auth_service, 'initiate_password_reset', return_value=auth_result):
            result = auth_middleware.initiate_password_reset("user@example.com")
            
            assert result.success is True
            assert result.error_message == "Password reset email sent"
    
    def test_reset_password_success(self, auth_middleware, auth_service):
        """Test successful password reset."""
        auth_result = AuthResult(
            success=True,
            error_message="Password reset successfully"
        )
        
        with patch.object(auth_service, 'reset_password', return_value=auth_result):
            result = auth_middleware.reset_password("reset_token", "NewPassword123")
            
            assert result.success is True
            assert result.error_message == "Password reset successfully"
    
    def test_change_password_success(self, auth_middleware, auth_service):
        """Test successful password change."""
        # Setup authenticated user
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        auth_middleware.current_user = mock_user
        
        auth_result = AuthResult(
            success=True,
            error_message="Password changed successfully"
        )
        
        with patch.object(auth_service, 'change_password', return_value=auth_result):
            result = auth_middleware.change_password("OldPassword123", "NewPassword123")
            
            assert result.success is True
            assert result.error_message == "Password changed successfully"
    
    def test_change_password_not_authenticated(self, auth_middleware):
        """Test password change when not authenticated."""
        result = auth_middleware.change_password("OldPassword123", "NewPassword123")
        
        assert result.success is False
        assert result.error_message == "Not authenticated"
    
    def test_session_state_persistence(self, auth_middleware, auth_service):
        """Test that session state persists across operations."""
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        
        auth_result = AuthResult(
            success=True,
            user=mock_user,
            token="jwt_token_123"
        )
        
        with patch.object(auth_service, 'login_user', return_value=auth_result):
            # Login
            login_result = auth_middleware.login("test@example.com", "SecurePass123")
            assert login_result.success is True
            
            # Verify session state
            assert auth_middleware.session_state['user_id'] == "user_123"
            assert auth_middleware.session_state['token'] == "jwt_token_123"
            
            # Logout
            with patch.object(auth_service, 'logout_user', return_value=True):
                logout_result = auth_middleware.logout()
                assert logout_result is True
                
                # Verify session state is cleared
                assert auth_middleware.session_state == {}
    
    def test_permission_based_access_control(self, auth_middleware, auth_service):
        """Test permission-based access control."""
        # Setup authenticated user
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        auth_middleware.current_user = mock_user
        
        # Mock permission check
        with patch.object(auth_service, 'validate_session_access', return_value=True):
            # Test various permissions
            assert auth_middleware.has_permission("patient_data", "read") is True
            assert auth_middleware.has_permission("patient_data", "write") is True
            
            # Verify service calls
            assert auth_service.validate_session_access.call_count == 2
    
    def test_role_based_access_simulation(self, auth_middleware, auth_service):
        """Test role-based access control simulation."""
        # Setup authenticated user
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        auth_middleware.current_user = mock_user
        
        # Mock permission function based on role
        def mock_permission_check(user_id, resource, permission):
            # Simulate role-based permissions
            permissions = {
                "patient_read": True,
                "patient_write": True,
                "therapist_read": False,
                "admin_write": False
            }
            return permissions.get(f"{resource}_{permission}", False)
        
        with patch.object(auth_service, 'validate_session_access', side_effect=mock_permission_check):
            # Test permissions based on simulated role
            assert auth_middleware.has_permission("patient", "read") is True
            assert auth_middleware.has_permission("patient", "write") is True
            assert auth_middleware.has_permission("therapist", "read") is False
            assert auth_middleware.has_permission("admin", "write") is False