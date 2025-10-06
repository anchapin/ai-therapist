"""
Final corrected test file to boost auth module coverage to 90%+.
This file fixes the API issues from the previous test file.
"""

import pytest
import os
import sys
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import auth module components
from auth.auth_service import AuthService, AuthSession, AuthResult
from auth.user_model import UserModel, UserProfile, UserRole, UserStatus
from auth.middleware import AuthMiddleware


class TestAuthServiceCorrected:
    """Corrected tests for AuthService with proper API usage."""

    def setup_method(self):
        """Set up test environment."""
        # Mock the database imports to avoid threading issues
        with patch('auth.auth_service.SessionRepository'):
            self.auth_service = AuthService()
            self.auth_service.session_repo = Mock()

    def test_init_with_custom_user_model(self):
        """Test AuthService initialization with custom user model."""
        custom_user_model = Mock()
        with patch('auth.auth_service.SessionRepository'):
            auth_service = AuthService(user_model=custom_user_model)
            assert auth_service.user_model == custom_user_model

    def test_init_with_environment_variables(self):
        """Test AuthService initialization with environment variables."""
        with patch.dict(os.environ, {
            'JWT_SECRET_KEY': 'test-secret',
            'JWT_EXPIRATION_HOURS': '12',
            'SESSION_TIMEOUT_MINUTES': '60',
            'MAX_CONCURRENT_SESSIONS': '3'
        }):
            with patch('auth.auth_service.SessionRepository'):
                auth_service = AuthService()
                assert auth_service.jwt_secret == 'test-secret'
                assert auth_service.jwt_expiration_hours == 12
                assert auth_service.session_timeout_minutes == 60
                assert auth_service.max_concurrent_sessions == 3

    def test_register_user_with_patient_role(self):
        """Test user registration with patient role."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.email = 'test@example.com'
        user_profile.full_name = 'Test User'
        user_profile.role = UserRole.PATIENT
        user_profile.status = UserStatus.ACTIVE
        user_profile.created_at = datetime.now()
        user_profile.updated_at = datetime.now()
        user_profile.last_login = None
        user_profile.login_attempts = 0
        user_profile.account_locked_until = None
        user_profile.password_reset_token = None
        user_profile.password_reset_expires = None
        user_profile.preferences = {}
        user_profile.medical_info = {}

        self.auth_service.user_model.create_user.return_value = user_profile

        result = self.auth_service.register_user(
            email='test@example.com',
            password='SecurePass123',
            full_name='Test User'
        )

        assert result.success is True
        assert result.user == user_profile
        self.auth_service.user_model.create_user.assert_called_once()

    def test_register_user_with_admin_role_forced_to_patient(self):
        """Test that admin role is forced to patient role in registration."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.email = 'test@example.com'
        user_profile.full_name = 'Test User'
        user_profile.role = UserRole.PATIENT
        user_profile.status = UserStatus.ACTIVE
        user_profile.created_at = datetime.now()
        user_profile.updated_at = datetime.now()
        user_profile.last_login = None
        user_profile.login_attempts = 0
        user_profile.account_locked_until = None
        user_profile.password_reset_token = None
        user_profile.password_reset_expires = None
        user_profile.preferences = {}
        user_profile.medical_info = {}

        self.auth_service.user_model.create_user.return_value = user_profile

        result = self.auth_service.register_user(
            email='test@example.com',
            password='SecurePass123',
            full_name='Test User',
            role=UserRole.ADMIN  # This should be forced to PATIENT
        )

        assert result.success is True
        # Verify that PATIENT role was used instead of ADMIN
        call_args = self.auth_service.user_model.create_user.call_args
        assert call_args[1]['role'] == UserRole.PATIENT

    def test_register_user_with_validation_error(self):
        """Test user registration with validation error."""
        self.auth_service.user_model.create_user.side_effect = ValueError("Invalid email")

        result = self.auth_service.register_user(
            email='invalid-email',
            password='SecurePass123',
            full_name='Test User'
        )

        assert result.success is False
        assert result.error_message == "Invalid email"

    def test_register_user_with_exception(self):
        """Test user registration with unexpected exception."""
        self.auth_service.user_model.create_user.side_effect = Exception("Database error")

        result = self.auth_service.register_user(
            email='test@example.com',
            password='SecurePass123',
            full_name='Test User'
        )

        assert result.success is False
        assert result.error_message == "Registration failed"

    def test_login_user_success(self):
        """Test successful user login."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.email = 'test@example.com'
        user_profile.status = UserStatus.ACTIVE
        user_profile.is_locked.return_value = False

        session = Mock()
        session.session_id = 'session123'
        session.created_at = datetime.now()

        self.auth_service.user_model.authenticate_user.return_value = user_profile
        self.auth_service._create_session.return_value = session
        self.auth_service._generate_jwt_token.return_value = 'jwt_token'

        result = self.auth_service.login_user(
            email='test@example.com',
            password='password',
            ip_address='127.0.0.1',
            user_agent='test-browser'
        )

        assert result.success is True
        assert result.token == 'jwt_token'
        assert result.session == session
        self.auth_service.user_model.authenticate_user.assert_called_once_with('test@example.com', 'password')
        self.auth_service._create_session.assert_called_once_with('user123', '127.0.0.1', 'test-browser')

    def test_login_user_invalid_credentials(self):
        """Test login with invalid credentials."""
        self.auth_service.user_model.authenticate_user.return_value = None

        result = self.auth_service.login_user(
            email='test@example.com',
            password='wrong-password'
        )

        assert result.success is False
        assert result.error_message == "Invalid credentials"

    def test_login_user_inactive_account(self):
        """Test login with inactive account."""
        user_profile = Mock()
        user_profile.status = UserStatus.INACTIVE

        self.auth_service.user_model.authenticate_user.return_value = user_profile

        result = self.auth_service.login_user(
            email='test@example.com',
            password='password'
        )

        assert result.success is False
        assert result.error_message == "Account is not active"

    def test_login_user_locked_account(self):
        """Test login with locked account."""
        user_profile = Mock()
        user_profile.status = UserStatus.ACTIVE
        user_profile.is_locked.return_value = True

        self.auth_service.user_model.authenticate_user.return_value = user_profile

        result = self.auth_service.login_user(
            email='test@example.com',
            password='password'
        )

        assert result.success is False
        assert result.error_message == "Account is temporarily locked"

    def test_login_user_session_creation_failure(self):
        """Test login with session creation failure."""
        user_profile = Mock()
        user_profile.status = UserStatus.ACTIVE
        user_profile.is_locked.return_value = False

        self.auth_service.user_model.authenticate_user.return_value = user_profile
        self.auth_service._create_session.return_value = None

        result = self.auth_service.login_user(
            email='test@example.com',
            password='password'
        )

        assert result.success is False
        assert result.error_message == "Failed to create session"

    def test_login_user_with_exception(self):
        """Test login with unexpected exception."""
        self.auth_service.user_model.authenticate_user.side_effect = Exception("Database error")

        result = self.auth_service.login_user(
            email='test@example.com',
            password='password'
        )

        assert result.success is False
        assert result.error_message == "Login failed"

    def test_validate_token_success(self):
        """Test successful token validation."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.status = UserStatus.ACTIVE

        self.auth_service.user_model.get_user.return_value = user_profile
        self.auth_service._is_session_valid.return_value = True

        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'session_id': 'session123',
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp())
            }

            result = self.auth_service.validate_token('valid_token')

            assert result == user_profile
            self.auth_service.user_model.get_user.assert_called_once_with('user123')
            self.auth_service._is_session_valid.assert_called_once_with('session123', 'user123')

    def test_validate_token_expired(self):
        """Test token validation with expired token."""
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'exp': int((datetime.now() - timedelta(hours=1)).timestamp())
            }

            result = self.auth_service.validate_token('expired_token')

            assert result is None

    def test_validate_token_invalid_signature(self):
        """Test token validation with invalid signature."""
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.InvalidTokenError("Invalid signature")

            result = self.auth_service.validate_token('invalid_token')

            assert result is None

    def test_validate_token_expired_signature(self):
        """Test token validation with expired signature."""
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token has expired")

            result = self.auth_service.validate_token('expired_token')

            assert result is None

    def test_validate_token_inactive_user(self):
        """Test token validation with inactive user."""
        user_profile = Mock()
        user_profile.status = UserStatus.INACTIVE

        self.auth_service.user_model.get_user.return_value = user_profile

        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp())
            }

            result = self.auth_service.validate_token('valid_token')

            assert result is None

    def test_validate_token_invalid_session(self):
        """Test token validation with invalid session."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.status = UserStatus.ACTIVE

        self.auth_service.user_model.get_user.return_value = user_profile
        self.auth_service._is_session_valid.return_value = False

        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'session_id': 'session123',
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp())
            }

            result = self.auth_service.validate_token('valid_token')

            assert result is None

    def test_logout_user_success(self):
        """Test successful user logout."""
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'session_id': 'session123'
            }

            result = self.auth_service.logout_user('valid_token')

            assert result is True
            self.auth_service._invalidate_session.assert_called_once_with('session123', 'user123')

    def test_logout_user_with_exception(self):
        """Test logout with exception."""
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Decode error")

            result = self.auth_service.logout_user('invalid_token')

            assert result is False

    def test_refresh_token_success(self):
        """Test successful token refresh."""
        user_profile = Mock()
        user_profile.user_id = 'user123'

        db_session = Mock()
        db_session.session_id = 'session123'
        db_session.user_id = 'user123'
        db_session.created_at = datetime.now()
        db_session.expires_at = datetime.now() + timedelta(hours=1)
        db_session.ip_address = '127.0.0.1'
        db_session.user_agent = 'test-browser'
        db_session.is_active = True
        db_session.is_expired.return_value = False

        self.auth_service.user_model.get_user.return_value = user_profile
        self.auth_service.session_repo.find_by_id.return_value = db_session
        self.auth_service._is_session_valid.return_value = True
        self.auth_service._generate_jwt_token.return_value = 'new_jwt_token'

        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'session_id': 'session123'
            }

            result = self.auth_service.refresh_token('valid_token')

            assert result == 'new_jwt_token'

    def test_refresh_token_invalid_session(self):
        """Test token refresh with invalid session."""
        user_profile = Mock()
        user_profile.user_id = 'user123'

        self.auth_service.user_model.get_user.return_value = user_profile
        self.auth_service._is_session_valid.return_value = False

        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'session_id': 'session123'
            }

            result = self.auth_service.refresh_token('valid_token')

            assert result is None

    def test_refresh_token_inactive_session(self):
        """Test token refresh with inactive session."""
        user_profile = Mock()
        user_profile.user_id = 'user123'

        db_session = Mock()
        db_session.is_active = False

        self.auth_service.user_model.get_user.return_value = user_profile
        self.auth_service.session_repo.find_by_id.return_value = db_session
        self.auth_service._is_session_valid.return_value = True

        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'user_id': 'user123',
                'session_id': 'session123'
            }

            result = self.auth_service.refresh_token('valid_token')

            assert result is None

    def test_refresh_token_with_exception(self):
        """Test token refresh with exception."""
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Decode error")

            result = self.auth_service.refresh_token('invalid_token')

            assert result is None

    def test_initiate_password_reset_success(self):
        """Test successful password reset initiation."""
        self.auth_service.user_model.initiate_password_reset.return_value = 'reset_token'

        result = self.auth_service.initiate_password_reset('test@example.com')

        assert result.success is True
        self.auth_service.user_model.initiate_password_reset.assert_called_once_with('test@example.com')

    def test_initiate_password_reset_user_not_found(self):
        """Test password reset initiation with user not found."""
        self.auth_service.user_model.initiate_password_reset.return_value = None

        result = self.auth_service.initiate_password_reset('nonexistent@example.com')

        assert result.success is False
        assert result.error_message == "User not found"

    def test_initiate_password_reset_with_exception(self):
        """Test password reset initiation with exception."""
        self.auth_service.user_model.initiate_password_reset.side_effect = Exception("Database error")

        result = self.auth_service.initiate_password_reset('test@example.com')

        assert result.success is False
        assert result.error_message == "Password reset failed"

    def test_reset_password_success(self):
        """Test successful password reset."""
        self.auth_service.user_model.reset_password.return_value = True

        result = self.auth_service.reset_password('valid_token', 'NewPassword123')

        assert result.success is True
        self.auth_service.user_model.reset_password.assert_called_once_with('valid_token', 'NewPassword123')

    def test_reset_password_invalid_token(self):
        """Test password reset with invalid token."""
        self.auth_service.user_model.reset_password.return_value = False

        result = self.auth_service.reset_password('invalid_token', 'NewPassword123')

        assert result.success is False
        assert result.error_message == "Invalid or expired reset token"

    def test_reset_password_with_exception(self):
        """Test password reset with exception."""
        self.auth_service.user_model.reset_password.side_effect = Exception("Database error")

        result = self.auth_service.reset_password('valid_token', 'NewPassword123')

        assert result.success is False
        assert result.error_message == "Password reset failed"

    def test_change_password_success(self):
        """Test successful password change."""
        self.auth_service.user_model.change_password.return_value = True

        result = self.auth_service.change_password('user123', 'OldPassword123', 'NewPassword123')

        assert result.success is True
        self.auth_service.user_model.change_password.assert_called_once_with('user123', 'OldPassword123', 'NewPassword123')

    def test_change_password_failure(self):
        """Test password change failure."""
        self.auth_service.user_model.change_password.return_value = False

        result = self.auth_service.change_password('user123', 'WrongOldPassword', 'NewPassword123')

        assert result.success is False
        assert result.error_message == "Password change failed"

    def test_change_password_with_exception(self):
        """Test password change with exception."""
        self.auth_service.user_model.change_password.side_effect = Exception("Database error")

        result = self.auth_service.change_password('user123', 'OldPassword123', 'NewPassword123')

        assert result.success is False
        assert result.error_message == "Password change failed"

    def test_get_user_sessions(self):
        """Test getting user sessions."""
        db_sessions = [
            Mock(
                session_id='session1',
                user_id='user123',
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1),
                ip_address='127.0.0.1',
                user_agent='browser1',
                is_active=True
            ),
            Mock(
                session_id='session2',
                user_id='user123',
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1),
                ip_address='127.0.0.1',
                user_agent='browser2',
                is_active=True
            )
        ]

        self.auth_service.session_repo.find_by_user_id.return_value = db_sessions

        sessions = self.auth_service.get_user_sessions('user123')

        assert len(sessions) == 2
        assert sessions[0].session_id == 'session1'
        assert sessions[1].session_id == 'session2'
        self.auth_service.session_repo.find_by_user_id.assert_called_once_with('user123', active_only=True)

    def test_invalidate_user_sessions(self):
        """Test invalidating user sessions."""
        db_sessions = [
            Mock(session_id='session1', is_active=True),
            Mock(session_id='session2', is_active=True),
            Mock(session_id='session3', is_active=True)
        ]

        self.auth_service.session_repo.find_by_user_id.return_value = db_sessions
        self.auth_service.session_repo.save.return_value = True

        invalidated = self.auth_service.invalidate_user_sessions('user123', keep_current='session2')

        assert invalidated == 2
        # Verify session1 and session3 were invalidated
        assert db_sessions[0].is_active is False
        assert db_sessions[1].is_active is True  # session2 kept active
        assert db_sessions[2].is_active is False

    def test_validate_session_access_success(self):
        """Test successful session access validation."""
        user_profile = Mock()
        user_profile.can_access_resource.return_value = True

        self.auth_service.user_model.get_user.return_value = user_profile

        result = self.auth_service.validate_session_access('user123', 'therapy_sessions', 'read')

        assert result is True
        user_profile.can_access_resource.assert_called_once_with('therapy_sessions', 'read')

    def test_validate_session_access_user_not_found(self):
        """Test session access validation with user not found."""
        self.auth_service.user_model.get_user.return_value = None

        result = self.auth_service.validate_session_access('nonexistent', 'therapy_sessions', 'read')

        assert result is False

    def test_create_session_success(self):
        """Test successful session creation."""
        db_session = Mock()
        db_session.session_id = 'session123'
        db_session.user_id = 'user123'
        db_session.created_at = datetime.now()
        db_session.expires_at = datetime.now() + timedelta(minutes=30)
        db_session.ip_address = '127.0.0.1'
        db_session.user_agent = 'test-browser'
        db_session.is_active = True

        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = db_session
            self.auth_service.session_repo.save.return_value = True

            session = self.auth_service._create_session('user123', '127.0.0.1', 'test-browser')

            assert session is not None
            assert session.session_id == 'session123'
            assert session.user_id == 'user123'

    def test_create_session_with_concurrent_limit(self):
        """Test session creation with concurrent session limit."""
        existing_sessions = [
            Mock(session_id='old_session', is_active=True, created_at=datetime.now() - timedelta(hours=1))
        ]

        self.auth_service.max_concurrent_sessions = 1
        self.auth_service.session_repo.find_by_user_id.return_value = existing_sessions

        db_session = Mock()
        db_session.session_id = 'new_session'
        db_session.user_id = 'user123'
        db_session.created_at = datetime.now()
        db_session.expires_at = datetime.now() + timedelta(minutes=30)
        db_session.ip_address = '127.0.0.1'
        db_session.user_agent = 'test-browser'
        db_session.is_active = True

        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = db_session
            self.auth_service.session_repo.save.return_value = True

            session = self.auth_service._create_session('user123', '127.0.0.1', 'test-browser')

            assert session is not None
            assert session.session_id == 'new_session'
            # Verify old session was invalidated
            assert existing_sessions[0].is_active is False

    def test_create_session_save_failure(self):
        """Test session creation with save failure."""
        db_session = Mock()
        db_session.session_id = 'session123'

        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = db_session
            self.auth_service.session_repo.save.return_value = False

            session = self.auth_service._create_session('user123')

            assert session is None

    def test_create_session_with_exception(self):
        """Test session creation with exception."""
        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.side_effect = Exception("Database error")

            session = self.auth_service._create_session('user123')

            assert session is None

    def test_invalidate_session(self):
        """Test session invalidation."""
        db_session = Mock()
        db_session.is_active = True

        self.auth_service.session_repo.find_by_id.return_value = db_session
        self.auth_service.session_repo.save.return_value = True

        self.auth_service._invalidate_session('session123', 'user123')

        assert db_session.is_active is False
        self.auth_service.session_repo.save.assert_called_once_with(db_session)

    def test_invalidate_session_not_found(self):
        """Test invalidating non-existent session."""
        self.auth_service.session_repo.find_by_id.return_value = None

        # Should not raise exception
        self.auth_service._invalidate_session('nonexistent', 'user123')

    def test_is_session_valid_success(self):
        """Test valid session check."""
        db_session = Mock()
        db_session.is_active = True
        db_session.user_id = 'user123'
        db_session.is_expired.return_value = False

        self.auth_service.session_repo.find_by_id.return_value = db_session

        result = self.auth_service._is_session_valid('session123', 'user123')

        assert result is True

    def test_is_session_valid_not_active(self):
        """Test session validity check with inactive session."""
        db_session = Mock()
        db_session.is_active = False
        db_session.user_id = 'user123'

        self.auth_service.session_repo.find_by_id.return_value = db_session

        result = self.auth_service._is_session_valid('session123', 'user123')

        assert result is False

    def test_is_session_valid_wrong_user(self):
        """Test session validity check with wrong user."""
        db_session = Mock()
        db_session.is_active = True
        db_session.user_id = 'other_user'

        self.auth_service.session_repo.find_by_id.return_value = db_session

        result = self.auth_service._is_session_valid('session123', 'user123')

        assert result is False

    def test_is_session_valid_expired(self):
        """Test session validity check with expired session."""
        db_session = Mock()
        db_session.is_active = True
        db_session.user_id = 'user123'
        db_session.is_expired.return_value = True

        self.auth_service.session_repo.find_by_id.return_value = db_session

        result = self.auth_service._is_session_valid('session123', 'user123')

        assert result is False

    def test_is_session_valid_not_found(self):
        """Test session validity check with non-existent session."""
        self.auth_service.session_repo.find_by_id.return_value = None

        result = self.auth_service._is_session_valid('nonexistent', 'user123')

        assert result is False

    def test_generate_jwt_token(self):
        """Test JWT token generation."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.email = 'test@example.com'
        user_profile.role = UserRole.PATIENT

        session = Mock()
        session.session_id = 'session123'

        with patch('jwt.encode') as mock_encode:
            mock_encode.return_value = 'jwt_token'

            token = self.auth_service._generate_jwt_token(user_profile, session)

            assert token == 'jwt_token'
            mock_encode.assert_called_once()

    def test_generate_session_id(self):
        """Test session ID generation."""
        session_id = self.auth_service._generate_session_id()

        assert session_id.startswith('session_')
        assert len(session_id) > 10

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        with patch('auth.auth_service.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_db.health_check.return_value = {
                'table_counts': {'sessions': 10}
            }
            mock_db.cleanup_expired_data.return_value = 5
            mock_get_db.return_value = mock_db

            self.auth_service._cleanup_expired_sessions()

            mock_db.cleanup_expired_data.assert_called_once()

    def test_get_auth_statistics(self):
        """Test getting authentication statistics."""
        with patch('auth.auth_service.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_db.health_check.return_value = {
                'table_counts': {
                    'users': 100,
                    'sessions': 25
                }
            }
            mock_get_db.return_value = mock_db

            stats = self.auth_service.get_auth_statistics()

            assert stats['total_users'] == 100
            assert stats['active_sessions'] == 25
            assert stats['total_sessions_created'] == 25

    def test_filter_user_for_response_with_user(self):
        """Test filtering user for response."""
        user_profile = Mock()
        user_profile.to_dict.return_value = {'user_id': 'user123', 'email': 'test@example.com'}

        result = self.auth_service._filter_user_for_response(user_profile, requesting_user_role='patient')

        assert result == {'user_id': 'user123', 'email': 'test@example.com'}
        user_profile.to_dict.assert_called_once_with(user_role='patient', include_sensitive=False)

    def test_filter_user_for_response_without_user(self):
        """Test filtering None user for response."""
        result = self.auth_service._filter_user_for_response(None)

        assert result is None


class TestAuthSessionCorrected:
    """Corrected tests for AuthSession."""

    def test_auth_session_init(self):
        """Test AuthSession initialization."""
        now = datetime.now()
        expires_at = now + timedelta(hours=1)

        session = AuthSession(
            session_id='session123',
            user_id='user123',
            created_at=now,
            expires_at=expires_at,
            ip_address='127.0.0.1',
            user_agent='test-browser',
            is_active=True
        )

        assert session.session_id == 'session123'
        assert session.user_id == 'user123'
        assert session.created_at == now
        assert session.expires_at == expires_at
        assert session.ip_address == '127.0.0.1'
        assert session.user_agent == 'test-browser'
        assert session.is_active is True

    def test_auth_session_is_expired_true(self):
        """Test AuthSession is_expired when expired."""
        past_time = datetime.now() - timedelta(hours=1)

        session = AuthSession(
            session_id='session123',
            user_id='user123',
            created_at=past_time - timedelta(hours=2),
            expires_at=past_time,  # Expired 1 hour ago
            is_active=True
        )

        assert session.is_expired() is True

    def test_auth_session_is_expired_false(self):
        """Test AuthSession is_expired when not expired."""
        future_time = datetime.now() + timedelta(hours=1)

        session = AuthSession(
            session_id='session123',
            user_id='user123',
            created_at=datetime.now(),
            expires_at=future_time,  # Expires in 1 hour
            is_active=True
        )

        assert session.is_expired() is False

    def test_auth_session_to_dict(self):
        """Test AuthSession to_dict conversion."""
        now = datetime.now()
        expires_at = now + timedelta(hours=1)

        session = AuthSession(
            session_id='session123',
            user_id='user123',
            created_at=now,
            expires_at=expires_at,
            ip_address='127.0.0.1',
            user_agent='test-browser',
            is_active=True
        )

        result = session.to_dict()

        assert result['session_id'] == 'session123'
        assert result['user_id'] == 'user123'
        assert result['ip_address'] == '127.0.0.1'
        assert result['user_agent'] == 'test-browser'
        assert result['is_active'] is True
        assert 'created_at' in result
        assert 'expires_at' in result


class TestAuthResultCorrected:
    """Corrected tests for AuthResult."""

    def test_auth_result_init_success(self):
        """Test AuthResult initialization for success."""
        user_profile = Mock()
        user_profile.user_id = 'user123'
        user_profile.email = 'test@example.com'
        user_profile.role = UserRole.PATIENT
        user_profile.status = UserStatus.ACTIVE
        user_profile.created_at = datetime.now()
        user_profile.updated_at = datetime.now()
        user_profile.last_login = None
        user_profile.login_attempts = 0
        user_profile.account_locked_until = None
        user_profile.password_reset_token = None
        user_profile.password_reset_expires = None
        user_profile.preferences = {}
        user_profile.medical_info = {}

        session = Mock()
        session.session_id = 'session123'

        result = AuthResult(
            success=True,
            user=user_profile,
            token='jwt_token',
            session=session
        )

        assert result.success is True
        assert result.user == user_profile
        assert result.token == 'jwt_token'
        assert result.session == session
        assert result.error_message is None

    def test_auth_result_init_failure(self):
        """Test AuthResult initialization for failure."""
        result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )

        assert result.success is False
        assert result.error_message == "Invalid credentials"
        assert result.user is None
        assert result.token is None
        assert result.session is None


class TestMiddlewareCorrected:
    """Corrected tests for AuthMiddleware."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_auth_service = Mock()
        self.middleware = AuthMiddleware(self.mock_auth_service)

    def test_init(self):
        """Test AuthMiddleware initialization."""
        assert self.middleware.auth_service == self.mock_auth_service

    def test_login_required_decorator_authenticated(self):
        """Test login_required decorator when authenticated."""
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = 'valid_token'
            self.mock_auth_service.validate_token.return_value = Mock()

            @self.middleware.login_required
            def test_func():
                return "success"

            result = test_func()

            assert result == "success"

    def test_login_required_decorator_not_authenticated(self):
        """Test login_required decorator when not authenticated."""
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = None
            self.mock_auth_service.validate_token.return_value = None

            @self.middleware.login_required
            def test_func():
                return "success"

            result = test_func()

            assert result is None

    def test_role_required_decorator_with_correct_role(self):
        """Test role_required decorator with correct role."""
        user_profile = Mock()
        user_profile.role = UserRole.THERAPIST

        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = 'valid_token'
            self.mock_auth_service.validate_token.return_value = user_profile

            @self.middleware.role_required([UserRole.THERAPIST])
            def test_func():
                return "success"

            result = test_func()

            assert result == "success"

    def test_role_required_decorator_with_wrong_role(self):
        """Test role_required decorator with wrong role."""
        user_profile = Mock()
        user_profile.role = UserRole.PATIENT

        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = 'valid_token'
            self.mock_auth_service.validate_token.return_value = user_profile

            @self.middleware.role_required([UserRole.THERAPIST])
            def test_func():
                return "success"

            result = test_func()

            assert result is None

    def test_is_authenticated_true(self):
        """Test is_authenticated when user is authenticated."""
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = 'valid_token'
            self.mock_auth_service.validate_token.return_value = Mock()

            result = self.middleware.is_authenticated()

            assert result is True

    def test_is_authenticated_false(self):
        """Test is_authenticated when user is not authenticated."""
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = None

            result = self.middleware.is_authenticated()

            assert result is False

    def test_get_current_user(self):
        """Test get_current_user."""
        user_profile = Mock()
        user_profile.user_id = 'user123'

        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = 'valid_token'
            self.mock_auth_service.validate_token.return_value = user_profile

            result = self.middleware.get_current_user()

            assert result == user_profile

    def test_get_current_user_no_token(self):
        """Test get_current_user when no token."""
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.get.return_value = None

            result = self.middleware.get_current_user()

            assert result is None

    def test_login_user_success(self):
        """Test successful user login."""
        auth_result = Mock()
        auth_result.success = True
        auth_result.token = 'jwt_token'
        auth_result.user = Mock()
        auth_result.session = Mock()
        auth_result.session.created_at = datetime.now()

        self.mock_auth_service.login_user.return_value = auth_result

        with patch('auth.middleware.st') as mock_st:
            result = self.middleware.login_user('test@example.com', 'password')

            assert result == auth_result
            assert mock_st.session_state.auth_token == 'jwt_token'
            assert mock_st.session_state.user == auth_result.user
            assert mock_st.session_state.auth_time == auth_result.session.created_at

    def test_login_user_failure(self):
        """Test failed user login."""
        auth_result = Mock()
        auth_result.success = False

        self.mock_auth_service.login_user.return_value = auth_result

        with patch('auth.middleware.st') as mock_st:
            result = self.middleware.login_user('test@example.com', 'wrong_password')

            assert result == auth_result
            # Verify session state was not updated
            assert not hasattr(mock_st.session_state, 'auth_token')

    def test_logout_user(self):
        """Test user logout."""
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state.auth_token = 'jwt_token'
            mock_st.session_state.user = Mock()
            mock_st.session_state.auth_time = datetime.now()

            self.middleware.logout_user()

            self.mock_auth_service.logout_user.assert_called_once_with('jwt_token')
            assert not hasattr(mock_st.session_state, 'auth_token')
            assert not hasattr(mock_st.session_state, 'user')
            assert not hasattr(mock_st.session_state, 'auth_time')

    def test_logout_user_no_token(self):
        """Test logout when no token exists."""
        with patch('auth.middleware.st') as mock_st:
            # No token in session state
            mock_st.session_state = {}

            self.middleware.logout_user()

            # Should not raise exception
            self.mock_auth_service.logout_user.assert_not_called()

    def test_get_client_ip(self):
        """Test getting client IP."""
        ip = self.middleware._get_client_ip()
        assert ip == "streamlit_client"

    def test_get_user_agent(self):
        """Test getting user agent."""
        user_agent = self.middleware._get_user_agent()
        assert user_agent == "streamlit_browser"


class TestUserProfileCorrected:
    """Corrected tests for UserProfile."""

    def test_user_profile_init_with_patient_role(self):
        """Test UserProfile initialization with PATIENT role."""
        now = datetime.now()

        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )

        assert profile.user_id == 'user123'
        assert profile.email == 'test@example.com'
        assert profile.full_name == 'Test User'
        assert profile.role == UserRole.PATIENT
        assert profile.status == UserStatus.ACTIVE
        assert profile.created_at == now
        assert profile.updated_at == now

    def test_user_profile_to_dict_with_pii_protection(self):
        """Test UserProfile to_dict with PII protection."""
        now = datetime.now()

        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            medical_info={
                'condition': 'Anxiety',
                'medication': 'Sertraline',
                'allergies': 'Penicillin'
            }
        )

        # Test with patient role (limited access)
        result = profile.to_dict(user_role='patient', include_sensitive=False)

        assert 'password_reset_token' not in result
        assert 'password_reset_expires' not in result
        # Medical info should be sanitized for patient
        assert '_sanitized' in result['medical_info']

    def test_user_profile_to_dict_with_admin_role(self):
        """Test UserProfile to_dict with admin role."""
        now = datetime.now()

        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            medical_info={
                'condition': 'Anxiety',
                'medication': 'Sertraline',
                'allergies': 'Penicillin'
            }
        )

        # Test with admin role (full access)
        result = profile.to_dict(user_role='admin', include_sensitive=True)

        assert 'password_reset_token' not in result
        assert 'password_reset_expires' not in result
        # Medical info should be visible to admin
        assert 'condition' in result['medical_info']
        assert 'medication' in result['medical_info']

    def test_user_profile_is_locked_true(self):
        """Test UserProfile is_locked when account is locked."""
        future_time = datetime.now() + timedelta(minutes=30)

        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.LOCKED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=future_time
        )

        assert profile.is_locked() is True

    def test_user_profile_is_locked_false(self):
        """Test UserProfile is_locked when account is not locked."""
        past_time = datetime.now() - timedelta(minutes=30)

        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=past_time
        )

        assert profile.is_locked() is False

    def test_user_profile_is_locked_none(self):
        """Test UserProfile is_locked when account_locked_until is None."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=None
        )

        assert profile.is_locked() is False

    def test_user_profile_increment_login_attempts(self):
        """Test UserProfile increment_login_attempts."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=3
        )

        profile.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)

        assert profile.login_attempts == 4
        assert profile.status == UserStatus.ACTIVE  # Not locked yet

    def test_user_profile_increment_login_attempts_lock(self):
        """Test UserProfile increment_login_attempts with lock."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=4
        )

        profile.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)

        assert profile.login_attempts == 5
        assert profile.status == UserStatus.LOCKED
        assert profile.account_locked_until is not None

    def test_user_profile_reset_login_attempts(self):
        """Test UserProfile reset_login_attempts."""
        future_time = datetime.now() + timedelta(minutes=30)

        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.LOCKED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=5,
            account_locked_until=future_time
        )

        profile.reset_login_attempts()

        assert profile.login_attempts == 0
        assert profile.status == UserStatus.ACTIVE
        assert profile.account_locked_until is None

    def test_user_profile_can_access_resource_success(self):
        """Test UserProfile can_access_resource with permission."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        result = profile.can_access_resource('own_profile', 'read')

        assert result is True

    def test_user_profile_can_access_resource_failure(self):
        """Test UserProfile can_access_resource without permission."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        result = profile.can_access_resource('all_profiles', 'delete')

        assert result is False

    def test_user_profile_mask_email(self):
        """Test UserProfile _mask_email method."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        masked = profile._mask_email('test@example.com')

        assert masked == 't***t@example.com'

    def test_user_profile_mask_email_short(self):
        """Test UserProfile _mask_email method with short local part."""
        profile = UserProfile(
            user_id='user123',
            email='ab@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        masked = profile._mask_email('ab@example.com')

        assert masked == '**@example.com'

    def test_user_profile_sanitize_medical_info_patient(self):
        """Test UserProfile _sanitize_medical_info for patient role."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        medical_info = {
            'condition': 'Anxiety',
            'medication': 'Sertraline',
            'allergies': 'Penicillin',
            'treatment_history': 'Long term therapy'
        }

        sanitized = profile._sanitize_medical_info(medical_info, 'patient')

        assert 'insurance_provider' not in sanitized  # Not in original
        assert 'condition' not in sanitized  # Not visible to patient
        assert 'medication' not in sanitized  # Not visible to patient
        assert 'allergies' not in sanitized  # Not visible to patient
        assert 'treatment_history' not in sanitized  # Not visible to patient
        assert '_sanitized' in sanitized

    def test_user_profile_sanitize_medical_info_therapist(self):
        """Test UserProfile _sanitize_medical_info for therapist role."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        medical_info = {
            'condition': 'Anxiety',
            'medication': 'Sertraline',
            'allergies': 'Penicillin',
            'treatment_history': 'Long term therapy'
        }

        sanitized = profile._sanitize_medical_info(medical_info, 'therapist')

        assert 'condition' in sanitized  # Visible to therapist
        assert 'medication' in sanitized  # Visible to therapist
        assert 'allergies' in sanitized  # Visible to therapist
        assert 'treatment_history' not in sanitized  # Not visible to therapist
        assert '_sanitized' in sanitized  # Some fields were hidden

    def test_user_profile_sanitize_medical_info_admin(self):
        """Test UserProfile _sanitize_medical_info for admin role."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        medical_info = {
            'condition': 'Anxiety',
            'medication': 'Sertraline',
            'allergies': 'Penicillin',
            'treatment_history': 'Long term therapy'
        }

        sanitized = profile._sanitize_medical_info(medical_info, 'admin')

        assert 'condition' in sanitized  # Visible to admin
        assert 'medication' in sanitized  # Visible to admin
        assert 'allergies' in sanitized  # Visible to admin
        assert 'treatment_history' in sanitized  # Visible to admin
        assert '_sanitized' not in sanitized  # All fields visible

    def test_user_profile_is_owner_request(self):
        """Test UserProfile _is_owner_request method."""
        profile = UserProfile(
            user_id='user123',
            email='test@example.com',
            full_name='Test User',
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # This method always returns False in the current implementation
        result = profile._is_owner_request('patient')
        assert result is False


# Import jwt for the tests
import jwt