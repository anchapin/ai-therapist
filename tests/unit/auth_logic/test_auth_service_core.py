"""
Unit tests for core authentication logic without UI dependencies.

Tests the AuthService business logic in isolation without Streamlit
or other UI components.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import jwt
import secrets

from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserRole, UserStatus


class TestAuthServiceCore:
    """Test core authentication service functionality."""

    def test_initialization_with_defaults(self):
        """Test AuthService initialization with default configuration."""
        with patch('auth.auth_service.UserModel') as mock_user_model:
            mock_user_instance = MagicMock()
            mock_user_model.return_value = mock_user_instance
            
            auth_service = AuthService()
            
            assert auth_service.jwt_secret == "ai-therapist-jwt-secret-change-in-production"
            assert auth_service.jwt_algorithm == "HS256"
            assert auth_service.jwt_expiration_hours == 24
            assert auth_service.session_timeout_minutes == 30
            assert auth_service.max_concurrent_sessions == 5
            assert auth_service.user_model == mock_user_instance

    def test_initialization_with_custom_values(self):
        """Test AuthService initialization with custom configuration."""
        with patch.dict('os.environ', {
            'JWT_SECRET_KEY': 'custom-secret',
            'JWT_EXPIRATION_HOURS': '12',
            'SESSION_TIMEOUT_MINUTES': '60',
            'MAX_CONCURRENT_SESSIONS': '3'
        }):
            with patch('auth.auth_service.UserModel') as mock_user_model:
                mock_user_instance = MagicMock()
                mock_user_model.return_value = mock_user_instance
                
                auth_service = AuthService()
                
                assert auth_service.jwt_secret == 'custom-secret'
                assert auth_service.jwt_expiration_hours == 12
                assert auth_service.session_timeout_minutes == 60
                assert auth_service.max_concurrent_sessions == 3

    def test_user_registration_success(self, auth_service):
        """Test successful user registration."""
        # Mock user creation
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.PATIENT
        mock_user.status = UserStatus.ACTIVE
        
        auth_service.user_model.create_user.return_value = mock_user
        
        # Register user
        result = auth_service.register_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User",
            role=UserRole.PATIENT
        )
        
        # Verify result
        assert result.success is True
        assert result.user == mock_user
        assert result.error_message is None
        
        # Verify user model was called correctly
        auth_service.user_model.create_user.assert_called_once_with(
            "test@example.com", "SecurePass123", "Test User", UserRole.PATIENT
        )

    def test_user_registration_validation_error(self, auth_service):
        """Test user registration with validation error."""
        # Mock user model to raise validation error
        auth_service.user_model.create_user.side_effect = ValueError("Email already exists")
        
        result = auth_service.register_user(
            email="existing@example.com",
            password="SecurePass123",
            full_name="Existing User"
        )
        
        assert result.success is False
        assert result.error_message == "Email already exists"
        assert result.user is None

    def test_user_registration_system_error(self, auth_service):
        """Test user registration with system error."""
        # Mock user model to raise system error
        auth_service.user_model.create_user.side_effect = Exception("Database error")
        
        result = auth_service.register_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        assert result.success is False
        assert result.error_message == "Registration failed"
        assert result.user is None

    def test_user_login_success(self, auth_service):
        """Test successful user login."""
        # Mock user authentication
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.status = UserStatus.ACTIVE
        mock_user.is_locked.return_value = False
        
        auth_service.user_model.authenticate_user.return_value = mock_user
        
        # Mock session creation
        with patch.object(auth_service, '_create_session') as mock_create_session:
            mock_session = AuthSession(
                session_id="session_123",
                user_id="user_123",
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30)
            )
            mock_create_session.return_value = mock_session
            
            # Perform login
            result = auth_service.login_user(
                email="test@example.com",
                password="SecurePass123",
                ip_address="127.0.0.1",
                user_agent="test-agent"
            )
            
            # Verify result
            assert result.success is True
            assert result.user == mock_user
            assert result.session == mock_session
            assert result.token is not None
            assert result.error_message is None
            
            # Verify authentication was called
            auth_service.user_model.authenticate_user.assert_called_once_with(
                "test@example.com", "SecurePass123"
            )
            
            # Verify session was created
            mock_create_session.assert_called_once_with(
                "user_123", "127.0.0.1", "test-agent"
            )

    def test_user_login_invalid_credentials(self, auth_service):
        """Test user login with invalid credentials."""
        auth_service.user_model.authenticate_user.return_value = None
        
        result = auth_service.login_user(
            email="test@example.com",
            password="wrong_password"
        )
        
        assert result.success is False
        assert result.error_message == "Invalid credentials"
        assert result.user is None
        assert result.token is None

    def test_user_login_inactive_account(self, auth_service):
        """Test user login with inactive account."""
        mock_user = MagicMock()
        mock_user.status = UserStatus.INACTIVE
        
        auth_service.user_model.authenticate_user.return_value = mock_user
        
        result = auth_service.login_user(
            email="test@example.com",
            password="SecurePass123"
        )
        
        assert result.success is False
        assert result.error_message == "Account is not active"

    def test_user_login_locked_account(self, auth_service):
        """Test user login with locked account."""
        mock_user = MagicMock()
        mock_user.status = UserStatus.ACTIVE
        mock_user.is_locked.return_value = True
        
        auth_service.user_model.authenticate_user.return_value = mock_user
        
        result = auth_service.login_user(
            email="test@example.com",
            password="SecurePass123"
        )
        
        assert result.success is False
        assert result.error_message == "Account is temporarily locked"

    def test_jwt_token_generation(self, auth_service):
        """Test JWT token generation."""
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        
        mock_session = AuthSession(
            session_id="session_123",
            user_id="user_123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30)
        )
        
        token = auth_service._generate_jwt_token(mock_user, mock_session)
        
        # Verify token structure
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify payload
        payload = jwt.decode(token, auth_service.jwt_secret, algorithms=[auth_service.jwt_algorithm])
        
        assert payload['user_id'] == "user_123"
        assert payload['email'] == "test@example.com"
        assert payload['role'] == UserRole.PATIENT.value
        assert payload['session_id'] == "session_123"
        assert 'iat' in payload
        assert 'exp' in payload

    def test_token_validation_success(self, auth_service):
        """Test successful token validation."""
        # Create valid token
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        mock_user.status = UserStatus.ACTIVE
        
        auth_service.user_model.get_user.return_value = mock_user
        
        with patch.object(auth_service, '_is_session_valid', return_value=True):
            token = auth_service._generate_jwt_token(mock_user, MagicMock())
            
            # Validate token
            validated_user = auth_service.validate_token(token)
            
            assert validated_user == mock_user
            auth_service.user_model.get_user.assert_called_once_with("user_123")

    def test_token_validation_invalid_token(self, auth_service):
        """Test token validation with invalid token."""
        result = auth_service.validate_token("invalid_token")
        assert result is None

    def test_token_validation_expired_token(self, auth_service):
        """Test token validation with expired token."""
        # Create expired token
        expired_time = datetime.now() - timedelta(hours=1)
        payload = {
            'user_id': 'user_123',
            'email': 'test@example.com',
            'role': 'patient',
            'session_id': 'session_123',
            'iat': int(expired_time.timestamp()),
            'exp': int(expired_time.timestamp()),
            'iss': 'ai-therapist'
        }
        
        expired_token = jwt.encode(payload, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        result = auth_service.validate_token(expired_token)
        assert result is None

    def test_token_validation_invalid_session(self, auth_service):
        """Test token validation with invalid session."""
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.status = UserStatus.ACTIVE
        
        auth_service.user_model.get_user.return_value = mock_user
        
        with patch.object(auth_service, '_is_session_valid', return_value=False):
            token = auth_service._generate_jwt_token(mock_user, MagicMock())
            
            result = auth_service.validate_token(token)
            assert result is None

    def test_password_reset_initiation_success(self, auth_service):
        """Test successful password reset initiation."""
        auth_service.user_model.initiate_password_reset.return_value = "reset_token_123"
        
        result = auth_service.initiate_password_reset("test@example.com")
        
        assert result.success is True
        assert result.error_message is None
        auth_service.user_model.initiate_password_reset.assert_called_once_with("test@example.com")

    def test_password_reset_initiation_user_not_found(self, auth_service):
        """Test password reset initiation with non-existent user."""
        auth_service.user_model.initiate_password_reset.return_value = None
        
        result = auth_service.initiate_password_reset("nonexistent@example.com")
        
        assert result.success is False
        assert result.error_message == "User not found"

    def test_password_reset_completion_success(self, auth_service):
        """Test successful password reset completion."""
        auth_service.user_model.reset_password.return_value = True
        
        result = auth_service.reset_password("reset_token_123", "NewSecurePass123")
        
        assert result.success is True
        assert result.error_message is None
        auth_service.user_model.reset_password.assert_called_once_with("reset_token_123", "NewSecurePass123")

    def test_password_reset_completion_invalid_token(self, auth_service):
        """Test password reset completion with invalid token."""
        auth_service.user_model.reset_password.return_value = False
        
        result = auth_service.reset_password("invalid_token", "NewSecurePass123")
        
        assert result.success is False
        assert result.error_message == "Invalid or expired reset token"

    def test_password_change_success(self, auth_service):
        """Test successful password change."""
        auth_service.user_model.change_password.return_value = True
        
        result = auth_service.change_password("user_123", "OldPass123", "NewPass123")
        
        assert result.success is True
        assert result.error_message is None
        auth_service.user_model.change_password.assert_called_once_with("user_123", "OldPass123", "NewPass123")

    def test_password_change_failure(self, auth_service):
        """Test password change failure."""
        auth_service.user_model.change_password.return_value = False
        
        result = auth_service.change_password("user_123", "WrongOldPass", "NewPass123")
        
        assert result.success is False
        assert result.error_message == "Password change failed"

    def test_session_creation(self, auth_service):
        """Test session creation."""
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        mock_user.status = UserStatus.ACTIVE
        
        # Create session
        session = auth_service._create_session("user_123", "127.0.0.1", "test-agent")
        
        assert session is not None
        assert session.user_id == "user_123"
        assert session.ip_address == "127.0.0.1"
        assert session.user_agent == "test-agent"
        assert session.is_active is True
        assert session.expires_at > datetime.now()

    def test_session_validation(self, auth_service):
        """Test session validation."""
        # Create valid session
        mock_db_session = MagicMock()
        mock_db_session.session_id = "session_123"
        mock_db_session.user_id = "user_123"
        mock_db_session.is_active = True
        mock_db_session.is_expired.return_value = False
        
        auth_service.session_repo.find_by_id.return_value = mock_db_session
        
        result = auth_service._is_session_valid("session_123", "user_123")
        
        assert result is True
        auth_service.session_repo.find_by_id.assert_called_once_with("session_123")

    def test_session_validation_invalid_session(self, auth_service):
        """Test session validation with invalid session."""
        auth_service.session_repo.find_by_id.return_value = None
        
        result = auth_service._is_session_valid("invalid_session", "user_123")
        
        assert result is False

    def test_logout_user_success(self, auth_service):
        """Test successful user logout."""
        # Create mock session
        mock_session = AuthSession(
            session_id="session_123",
            user_id="user_123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30)
        )
        
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.PATIENT
        
        token = auth_service._generate_jwt_token(mock_user, mock_session)
        
        with patch.object(auth_service, '_invalidate_session') as mock_invalidate:
            result = auth_service.logout_user(token)
            
            assert result is True
            mock_invalidate.assert_called_once_with("session_123", "user_123")

    def test_get_user_sessions(self, auth_service):
        """Test getting user sessions."""
        mock_db_sessions = [
            MagicMock(
                session_id="session_1",
                user_id="user_123",
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30),
                ip_address="127.0.0.1",
                user_agent="test-agent",
                is_active=True
            )
        ]
        
        auth_service.session_repo.find_by_user_id.return_value = mock_db_sessions
        
        sessions = auth_service.get_user_sessions("user_123")
        
        assert len(sessions) == 1
        assert sessions[0].session_id == "session_1"
        assert sessions[0].user_id == "user_123"

    def test_concurrent_session_limit(self, auth_service):
        """Test concurrent session limit enforcement."""
        # Create mock existing sessions
        existing_sessions = [MagicMock(is_active=True) for _ in range(5)]
        auth_service.session_repo.find_by_user_id.return_value = existing_sessions
        
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        
        with patch.object(auth_service, '_invalidate_session') as mock_invalidate:
            # Should invalidate oldest session when creating new one
            session = auth_service._create_session("user_123")
            
            # Should have invalidated the oldest session
            mock_invalidate.assert_called_once()

    def test_validate_session_access(self, auth_service):
        """Test session access validation."""
        mock_user = MagicMock()
        mock_user.can_access_resource.return_value = True
        
        auth_service.user_model.get_user.return_value = mock_user
        
        result = auth_service.validate_session_access("user_123", "resource", "read")
        
        assert result is True
        auth_service.user_model.get_user.assert_called_once_with("user_123")
        mock_user.can_access_resource.assert_called_once_with("resource", "read")