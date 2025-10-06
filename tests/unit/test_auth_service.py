"""
Comprehensive unit tests for auth/auth_service.py
"""

import pytest
import os
import jwt
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, ANY
from pathlib import Path
import tempfile
import shutil

# Mock the database modules to avoid import issues
with patch.dict('sys.modules', {
    'database.models': Mock(),
    'database.db_manager': Mock()
}):
    from auth.auth_service import AuthService, AuthSession, AuthResult
    from auth.user_model import UserRole, UserStatus, UserProfile


class TestAuthSession:
    """Test AuthSession functionality."""
    
    def test_auth_session_creation(self):
        """Test creating an authentication session."""
        now = datetime.now()
        expires_at = now + timedelta(hours=1)
        
        session = AuthSession(
            session_id="session_123",
            user_id="user_123",
            created_at=now,
            expires_at=expires_at,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            is_active=True
        )
        
        assert session.session_id == "session_123"
        assert session.user_id == "user_123"
        assert session.created_at == now
        assert session.expires_at == expires_at
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.is_active is True
    
    def test_auth_session_is_expired_true(self):
        """Test session expiration detection when expired."""
        past_time = datetime.now() - timedelta(hours=1)
        session = AuthSession(
            session_id="session_123",
            user_id="user_123",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=past_time
        )
        
        assert session.is_expired() is True
    
    def test_auth_session_is_expired_false(self):
        """Test session expiration detection when not expired."""
        future_time = datetime.now() + timedelta(hours=1)
        session = AuthSession(
            session_id="session_123",
            user_id="user_123",
            created_at=datetime.now(),
            expires_at=future_time
        )
        
        assert session.is_expired() is False
    
    def test_auth_session_to_dict(self):
        """Test converting session to dictionary."""
        now = datetime.now()
        expires_at = now + timedelta(hours=1)
        
        session = AuthSession(
            session_id="session_123",
            user_id="user_123",
            created_at=now,
            expires_at=expires_at,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            is_active=True
        )
        
        result = session.to_dict()
        
        assert isinstance(result, dict)
        assert result['session_id'] == "session_123"
        assert result['user_id'] == "user_123"
        assert result['created_at'] == now.isoformat()
        assert result['expires_at'] == expires_at.isoformat()
        assert result['ip_address'] == "192.168.1.1"
        assert result['user_agent'] == "Mozilla/5.0"
        assert result['is_active'] is True


class TestAuthResult:
    """Test AuthResult functionality."""
    
    def test_auth_result_success(self):
        """Test successful authentication result."""
        user = Mock(spec=UserProfile)
        session = Mock(spec=AuthSession)
        
        result = AuthResult(
            success=True,
            user=user,
            token="jwt_token_123",
            session=session
        )
        
        assert result.success is True
        assert result.user == user
        assert result.token == "jwt_token_123"
        assert result.session == session
        assert result.error_message is None
    
    def test_auth_result_failure(self):
        """Test failed authentication result."""
        result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        assert result.success is False
        assert result.user is None
        assert result.token is None
        assert result.session is None
        assert result.error_message == "Invalid credentials"


class TestAuthService:
    """Test AuthService functionality."""
    
    @pytest.fixture
    def mock_user_model(self):
        """Create a mock user model."""
        user_model = Mock()
        return user_model
    
    @pytest.fixture
    def mock_session_repo(self):
        """Create a mock session repository."""
        session_repo = Mock()
        return session_repo
    
    @pytest.fixture
    def auth_service(self, mock_user_model, mock_session_repo):
        """Create auth service with mocked dependencies."""
        with patch('auth.auth_service.SessionRepository', return_value=mock_session_repo):
            with patch('auth.auth_service.threading.Thread'):
                service = AuthService(user_model=mock_user_model)
                service.session_repo = mock_session_repo
                return service
    
    def test_auth_service_initialization(self, auth_service):
        """Test auth service initialization."""
        assert auth_service.jwt_secret is not None
        assert auth_service.jwt_algorithm == "HS256"
        assert auth_service.jwt_expiration_hours == 24
        assert auth_service.session_timeout_minutes == 30
        assert auth_service.max_concurrent_sessions == 5
        assert auth_service.user_model is not None
        assert auth_service.session_repo is not None
    
    def test_auth_service_custom_config(self):
        """Test auth service with custom configuration."""
        with patch.dict(os.environ, {
            'JWT_SECRET_KEY': 'custom-secret',
            'JWT_EXPIRATION_HOURS': '12',
            'SESSION_TIMEOUT_MINUTES': '60',
            'MAX_CONCURRENT_SESSIONS': '3'
        }):
            with patch('auth.auth_service.SessionRepository'):
                with patch('auth.auth_service.threading.Thread'):
                    service = AuthService()
                    
                    assert service.jwt_secret == 'custom-secret'
                    assert service.jwt_expiration_hours == 12
                    assert service.session_timeout_minutes == 60
                    assert service.max_concurrent_sessions == 3
    
    def test_register_user_success(self, auth_service, mock_user_model):
        """Test successful user registration."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.PATIENT
        
        mock_user_model.create_user.return_value = mock_user
        
        with patch.object(auth_service, '_filter_user_for_response', return_value=mock_user):
            result = auth_service.register_user(
                email="test@example.com",
                password="SecurePass123",
                full_name="Test User",
                role=UserRole.PATIENT
            )
        
        assert result.success is True
        assert result.user == mock_user
        assert result.error_message is None
        mock_user_model.create_user.assert_called_once()
    
    def test_register_user_invalid_email(self, auth_service, mock_user_model):
        """Test user registration with invalid email."""
        mock_user_model.create_user.side_effect = ValueError("Invalid email format")
        
        result = auth_service.register_user(
            email="invalid-email",
            password="SecurePass123",
            full_name="Test User"
        )
        
        assert result.success is False
        assert result.error_message == "Invalid email format"
        assert result.user is None
    
    def test_register_user_weak_password(self, auth_service, mock_user_model):
        """Test user registration with weak password."""
        mock_user_model.create_user.side_effect = ValueError("Password does not meet security requirements")
        
        result = auth_service.register_user(
            email="test@example.com",
            password="weak",
            full_name="Test User"
        )
        
        assert result.success is False
        assert result.error_message == "Password does not meet security requirements"
        assert result.user is None
    
    def test_register_user_existing_email(self, auth_service, mock_user_model):
        """Test user registration with existing email."""
        mock_user_model.create_user.side_effect = ValueError("User with this email already exists")
        
        result = auth_service.register_user(
            email="existing@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        assert result.success is False
        assert result.error_message == "User with this email already exists"
        assert result.user is None
    
    def test_register_user_non_patient_role(self, auth_service, mock_user_model):
        """Test user registration with non-patient role (should default to patient)."""
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.PATIENT
        
        mock_user_model.create_user.return_value = mock_user
        
        with patch.object(auth_service, '_filter_user_for_response', return_value=mock_user):
            result = auth_service.register_user(
                email="test@example.com",
                password="SecurePass123",
                full_name="Test User",
                role=UserRole.ADMIN  # This should be overridden to PATIENT
            )
        
        assert result.success is True
        # Verify that PATIENT role was used instead of ADMIN
        mock_user_model.create_user.assert_called_once_with(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User",
            role=UserRole.PATIENT
        )
    
    def test_login_user_success(self, auth_service, mock_user_model, mock_session_repo):
        """Test successful user login."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.status = UserStatus.ACTIVE
        mock_user.is_locked.return_value = False
        
        mock_session = Mock(spec=AuthSession)
        mock_session.session_id = "session_123"
        mock_session.created_at = datetime.now()
        mock_session.expires_at = datetime.now() + timedelta(hours=1)
        
        mock_user_model.authenticate_user.return_value = mock_user
        
        with patch.object(auth_service, '_create_session', return_value=mock_session):
            with patch.object(auth_service, '_generate_jwt_token', return_value="jwt_token_123"):
                with patch.object(auth_service, '_filter_user_for_response', return_value=mock_user):
                    result = auth_service.login_user(
                        email="test@example.com",
                        password="SecurePass123",
                        ip_address="192.168.1.1",
                        user_agent="Mozilla/5.0"
                    )
        
        assert result.success is True
        assert result.user == mock_user
        assert result.token == "jwt_token_123"
        assert result.session == mock_session
        assert result.error_message is None
        mock_user_model.authenticate_user.assert_called_once_with("test@example.com", "SecurePass123")
    
    def test_login_user_invalid_credentials(self, auth_service, mock_user_model):
        """Test login with invalid credentials."""
        mock_user_model.authenticate_user.return_value = None
        
        result = auth_service.login_user(
            email="test@example.com",
            password="wrongpassword"
        )
        
        assert result.success is False
        assert result.error_message == "Invalid credentials"
        assert result.user is None
        assert result.token is None
        assert result.session is None
    
    def test_login_user_inactive_account(self, auth_service, mock_user_model):
        """Test login with inactive account."""
        mock_user = Mock(spec=UserProfile)
        mock_user.status = UserStatus.INACTIVE
        
        mock_user_model.authenticate_user.return_value = mock_user
        
        result = auth_service.login_user(
            email="test@example.com",
            password="SecurePass123"
        )
        
        assert result.success is False
        assert result.error_message == "Account is not active"
        assert result.user is None
    
    def test_login_user_locked_account(self, auth_service, mock_user_model):
        """Test login with locked account."""
        mock_user = Mock(spec=UserProfile)
        mock_user.status = UserStatus.ACTIVE
        mock_user.is_locked.return_value = True
        
        mock_user_model.authenticate_user.return_value = mock_user
        
        result = auth_service.login_user(
            email="test@example.com",
            password="SecurePass123"
        )
        
        assert result.success is False
        assert result.error_message == "Account is temporarily locked"
        assert result.user is None
    
    def test_login_user_session_creation_failure(self, auth_service, mock_user_model):
        """Test login when session creation fails."""
        mock_user = Mock(spec=UserProfile)
        mock_user.status = UserStatus.ACTIVE
        mock_user.is_locked.return_value = False
        
        mock_user_model.authenticate_user.return_value = mock_user
        
        with patch.object(auth_service, '_create_session', return_value=None):
            result = auth_service.login_user(
                email="test@example.com",
                password="SecurePass123"
            )
        
        assert result.success is False
        assert result.error_message == "Failed to create session"
        assert result.user is None
    
    def test_validate_token_success(self, auth_service, mock_user_model):
        """Test successful token validation."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        mock_user.status = UserStatus.ACTIVE
        
        mock_user_model.get_user.return_value = mock_user
        
        # Create a valid token
        token = jwt.encode({
            'user_id': 'user_123',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_is_session_valid', return_value=True):
            result = auth_service.validate_token(token)
        
        assert result == mock_user
        mock_user_model.get_user.assert_called_once_with('user_123')
    
    def test_validate_token_expired(self, auth_service):
        """Test token validation with expired token."""
        # Create an expired token
        token = jwt.encode({
            'user_id': 'user_123',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(time.time()) - 7200,  # 2 hours ago
            'exp': int(time.time()) - 3600,   # 1 hour ago (expired)
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        result = auth_service.validate_token(token)
        
        assert result is None
    
    def test_validate_token_invalid_signature(self, auth_service):
        """Test token validation with invalid signature."""
        # Create token with different secret
        token = jwt.encode({
            'user_id': 'user_123',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, 'wrong-secret', algorithm=auth_service.jwt_algorithm)
        
        result = auth_service.validate_token(token)
        
        assert result is None
    
    def test_validate_token_user_not_found(self, auth_service, mock_user_model):
        """Test token validation when user is not found."""
        mock_user_model.get_user.return_value = None
        
        token = jwt.encode({
            'user_id': 'nonexistent_user',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_is_session_valid', return_value=True):
            result = auth_service.validate_token(token)
        
        assert result is None
    
    def test_validate_token_inactive_user(self, auth_service, mock_user_model):
        """Test token validation when user is inactive."""
        mock_user = Mock(spec=UserProfile)
        mock_user.status = UserStatus.INACTIVE
        
        mock_user_model.get_user.return_value = mock_user
        
        token = jwt.encode({
            'user_id': 'user_123',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_is_session_valid', return_value=True):
            result = auth_service.validate_token(token)
        
        assert result is None
    
    def test_validate_token_invalid_session(self, auth_service, mock_user_model):
        """Test token validation when session is invalid."""
        mock_user = Mock(spec=UserProfile)
        mock_user.status = UserStatus.ACTIVE
        
        mock_user_model.get_user.return_value = mock_user
        
        token = jwt.encode({
            'user_id': 'user_123',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_is_session_valid', return_value=False):
            result = auth_service.validate_token(token)
        
        assert result is None
    
    def test_logout_user_success(self, auth_service):
        """Test successful user logout."""
        token = jwt.encode({
            'user_id': 'user_123',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_invalidate_session') as mock_invalidate:
            result = auth_service.logout_user(token)
        
        assert result is True
        mock_invalidate.assert_called_once_with('session_123', 'user_123')
    
    def test_logout_user_invalid_token(self, auth_service):
        """Test logout with invalid token."""
        result = auth_service.logout_user("invalid_token")
        
        assert result is False
    
    def test_refresh_token_success(self, auth_service, mock_user_model, mock_session_repo):
        """Test successful token refresh."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        
        mock_db_session = Mock()
        mock_db_session.session_id = "session_123"
        mock_db_session.user_id = "user_123"
        mock_db_session.created_at = datetime.now()
        mock_db_session.expires_at = datetime.now() + timedelta(hours=1)
        mock_db_session.is_active = True
        
        mock_user_model.get_user.return_value = mock_user
        mock_session_repo.find_by_id.return_value = mock_db_session
        
        token = jwt.encode({
            'user_id': 'user_123',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_is_session_valid', return_value=True):
            with patch.object(auth_service, '_generate_jwt_token', return_value="new_jwt_token"):
                result = auth_service.refresh_token(token)
        
        assert result == "new_jwt_token"
    
    def test_refresh_token_invalid_user(self, auth_service, mock_user_model):
        """Test token refresh when user is invalid."""
        mock_user_model.get_user.return_value = None
        
        token = jwt.encode({
            'user_id': 'nonexistent_user',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        result = auth_service.refresh_token(token)
        
        assert result is None
    
    def test_refresh_token_invalid_session(self, auth_service, mock_user_model):
        """Test token refresh when session is invalid."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        
        mock_user_model.get_user.return_value = mock_user
        
        token = jwt.encode({
            'user_id': 'user_123',
            'session_id': 'session_123',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'iss': 'ai-therapist'
        }, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        with patch.object(auth_service, '_is_session_valid', return_value=False):
            result = auth_service.refresh_token(token)
        
        assert result is None
    
    def test_initiate_password_reset_success(self, auth_service, mock_user_model):
        """Test successful password reset initiation."""
        mock_user_model.initiate_password_reset.return_value = "reset_token_123"
        
        result = auth_service.initiate_password_reset("test@example.com")
        
        assert result.success is True
        assert result.error_message is None
        mock_user_model.initiate_password_reset.assert_called_once_with("test@example.com")
    
    def test_initiate_password_reset_user_not_found(self, auth_service, mock_user_model):
        """Test password reset initiation when user not found."""
        mock_user_model.initiate_password_reset.return_value = None
        
        result = auth_service.initiate_password_reset("nonexistent@example.com")
        
        assert result.success is False
        assert result.error_message == "User not found"
    
    def test_reset_password_success(self, auth_service, mock_user_model):
        """Test successful password reset."""
        mock_user_model.reset_password.return_value = True
        
        result = auth_service.reset_password("reset_token_123", "NewSecurePass123")
        
        assert result.success is True
        assert result.error_message is None
        mock_user_model.reset_password.assert_called_once_with("reset_token_123", "NewSecurePass123")
    
    def test_reset_password_invalid_token(self, auth_service, mock_user_model):
        """Test password reset with invalid token."""
        mock_user_model.reset_password.return_value = False
        
        result = auth_service.reset_password("invalid_token", "NewSecurePass123")
        
        assert result.success is False
        assert result.error_message == "Invalid or expired reset token"
    
    def test_change_password_success(self, auth_service, mock_user_model):
        """Test successful password change."""
        mock_user_model.change_password.return_value = True
        
        result = auth_service.change_password("user_123", "OldPass123", "NewPass123")
        
        assert result.success is True
        assert result.error_message is None
        mock_user_model.change_password.assert_called_once_with("user_123", "OldPass123", "NewPass123")
    
    def test_change_password_failure(self, auth_service, mock_user_model):
        """Test password change failure."""
        mock_user_model.change_password.return_value = False
        
        result = auth_service.change_password("user_123", "WrongOldPass", "NewPass123")
        
        assert result.success is False
        assert result.error_message == "Password change failed"
    
    def test_get_user_sessions(self, auth_service, mock_session_repo):
        """Test getting user sessions."""
        mock_db_session1 = Mock()
        mock_db_session1.session_id = "session_1"
        mock_db_session1.user_id = "user_123"
        mock_db_session1.created_at = datetime.now()
        mock_db_session1.expires_at = datetime.now() + timedelta(hours=1)
        mock_db_session1.ip_address = "192.168.1.1"
        mock_db_session1.user_agent = "Mozilla/5.0"
        mock_db_session1.is_active = True
        
        mock_db_session2 = Mock()
        mock_db_session2.session_id = "session_2"
        mock_db_session2.user_id = "user_123"
        mock_db_session2.created_at = datetime.now()
        mock_db_session2.expires_at = datetime.now() + timedelta(hours=1)
        mock_db_session2.ip_address = "192.168.1.2"
        mock_db_session2.user_agent = "Chrome/90.0"
        mock_db_session2.is_active = True
        
        mock_session_repo.find_by_user_id.return_value = [mock_db_session1, mock_db_session2]
        
        sessions = auth_service.get_user_sessions("user_123")
        
        assert len(sessions) == 2
        assert sessions[0].session_id == "session_1"
        assert sessions[1].session_id == "session_2"
        assert all(isinstance(session, AuthSession) for session in sessions)
        mock_session_repo.find_by_user_id.assert_called_once_with("user_123", active_only=True)
    
    def test_invalidate_user_sessions(self, auth_service, mock_session_repo):
        """Test invalidating user sessions."""
        mock_db_session1 = Mock()
        mock_db_session1.session_id = "session_1"
        mock_db_session1.is_active = True
        
        mock_db_session2 = Mock()
        mock_db_session2.session_id = "session_2"
        mock_db_session2.is_active = True
        
        mock_db_session3 = Mock()
        mock_db_session3.session_id = "session_3"
        mock_db_session3.is_active = True
        
        mock_session_repo.find_by_user_id.return_value = [mock_db_session1, mock_db_session2, mock_db_session3]
        mock_session_repo.save.return_value = True
        
        # Invalidate all sessions except session_2
        result = auth_service.invalidate_user_sessions("user_123", keep_current="session_2")
        
        assert result == 2  # Should invalidate 2 sessions
        assert mock_db_session1.is_active is False
        assert mock_db_session2.is_active is True  # Should remain active
        assert mock_db_session3.is_active is False
    
    def test_validate_session_access_success(self, auth_service, mock_user_model):
        """Test successful session access validation."""
        mock_user = Mock(spec=UserProfile)
        mock_user.can_access_resource.return_value = True
        
        mock_user_model.get_user.return_value = mock_user
        
        result = auth_service.validate_session_access("user_123", "therapy_sessions", "read")
        
        assert result is True
        mock_user.can_access_resource.assert_called_once_with("therapy_sessions", "read")
    
    def test_validate_session_access_user_not_found(self, auth_service, mock_user_model):
        """Test session access validation when user not found."""
        mock_user_model.get_user.return_value = None
        
        result = auth_service.validate_session_access("nonexistent_user", "therapy_sessions", "read")
        
        assert result is False
    
    def test_validate_session_access_permission_denied(self, auth_service, mock_user_model):
        """Test session access validation when permission denied."""
        mock_user = Mock(spec=UserProfile)
        mock_user.can_access_resource.return_value = False
        
        mock_user_model.get_user.return_value = mock_user
        
        result = auth_service.validate_session_access("user_123", "admin_panel", "write")
        
        assert result is False
    
    def test_create_session_success(self, auth_service, mock_session_repo):
        """Test successful session creation."""
        mock_db_session = Mock()
        mock_db_session.session_id = "session_123"
        mock_db_session.user_id = "user_123"
        mock_db_session.created_at = datetime.now()
        mock_db_session.expires_at = datetime.now() + timedelta(minutes=30)
        mock_db_session.ip_address = "192.168.1.1"
        mock_db_session.user_agent = "Mozilla/5.0"
        mock_db_session.is_active = True
        
        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = mock_db_session
            mock_session_repo.save.return_value = True
            
            session = auth_service._create_session(
                user_id="user_123",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
        
        assert session is not None
        assert isinstance(session, AuthSession)
        assert session.session_id == "session_123"
        assert session.user_id == "user_123"
    
    def test_create_session_concurrent_limit(self, auth_service, mock_session_repo):
        """Test session creation with concurrent session limit."""
        # Create 5 existing active sessions (the limit)
        existing_sessions = []
        for i in range(5):
            mock_session = Mock()
            mock_session.session_id = f"session_{i}"
            mock_session.created_at = datetime.now() - timedelta(minutes=i*10)
            mock_session.is_active = True
            existing_sessions.append(mock_session)
        
        mock_session_repo.find_by_user_id.return_value = existing_sessions
        
        # Mock the new session
        mock_new_session = Mock()
        mock_new_session.session_id = "session_new"
        mock_new_session.user_id = "user_123"
        mock_new_session.created_at = datetime.now()
        mock_new_session.expires_at = datetime.now() + timedelta(minutes=30)
        mock_new_session.ip_address = "192.168.1.1"
        mock_new_session.user_agent = "Mozilla/5.0"
        mock_new_session.is_active = True
        
        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = mock_new_session
            mock_session_repo.save.return_value = True
            
            session = auth_service._create_session(user_id="user_123")
        
        assert session is not None
        # Should have invalidated the oldest session
        oldest_session = min(existing_sessions, key=lambda s: s.created_at)
        assert oldest_session.is_active is False
    
    def test_create_session_save_failure(self, auth_service, mock_session_repo):
        """Test session creation when save fails."""
        mock_db_session = Mock()
        mock_db_session.session_id = "session_123"
        
        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = mock_db_session
            mock_session_repo.save.return_value = False
            
            session = auth_service._create_session(user_id="user_123")
        
        assert session is None
    
    def test_generate_jwt_token(self, auth_service):
        """Test JWT token generation."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        
        mock_session = Mock(spec=AuthSession)
        mock_session.session_id = "session_123"
        
        token = auth_service._generate_jwt_token(mock_user, mock_session)
        
        assert isinstance(token, str)
        
        # Decode and verify token contents
        payload = jwt.decode(token, auth_service.jwt_secret, algorithms=[auth_service.jwt_algorithm])
        assert payload['user_id'] == "user_123"
        assert payload['email'] == "test@example.com"
        assert payload['role'] == "patient"
        assert payload['session_id'] == "session_123"
        assert payload['iss'] == "ai-therapist"
        assert 'iat' in payload
        assert 'exp' in payload
    
    def test_generate_session_id(self, auth_service):
        """Test session ID generation."""
        session_id1 = auth_service._generate_session_id()
        session_id2 = auth_service._generate_session_id()
        
        assert isinstance(session_id1, str)
        assert isinstance(session_id2, str)
        assert session_id1 != session_id2
        assert session_id1.startswith("session_")
        assert session_id2.startswith("session_")
    
    def test_filter_user_for_response(self, auth_service):
        """Test user data filtering for response."""
        mock_user = Mock(spec=UserProfile)
        mock_user.to_dict.return_value = {
            'user_id': 'user_123',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'patient',
            'status': 'active'
        }
        
        result = auth_service._filter_user_for_response(mock_user, requesting_user_role='patient')
        
        assert isinstance(result, dict)
        mock_user.to_dict.assert_called_once_with(user_role='patient', include_sensitive=False)
    
    def test_get_auth_statistics(self, auth_service):
        """Test getting authentication statistics."""
        with patch('auth.auth_service.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_db.health_check.return_value = {
                'table_counts': {
                    'users': 10,
                    'sessions': 5
                }
            }
            mock_get_db.return_value = mock_db
            
            stats = auth_service.get_auth_statistics()
        
        assert isinstance(stats, dict)
        assert stats['total_users'] == 10
        assert stats['active_sessions'] == 5
        assert stats['total_sessions_created'] == 5