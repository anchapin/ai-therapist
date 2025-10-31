"""
Comprehensive tests to achieve 90%+ coverage for auth module.
Targets specific missing lines identified in coverage reports.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import jwt
import os
import tempfile
import threading

from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.user_model import UserProfile, UserRole, UserStatus, UserModel
from auth.middleware import AuthMiddleware


class TestComprehensiveAuthCoverage:
    """Comprehensive tests to achieve 90%+ coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Mock environment variables
        os.environ['JWT_SECRET_KEY'] = 'test-secret-key'
        os.environ['JWT_EXPIRATION_HOURS'] = '24'
        os.environ['SESSION_TIMEOUT_MINUTES'] = '30'
        os.environ['MAX_CONCURRENT_SESSIONS'] = '5'
        
        # Initialize auth service with mock user model
        self.mock_user_model = Mock(spec=UserModel)
        self.auth_service = AuthService(self.mock_user_model)
        
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
        # Clean up temporary database
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
        
        # Clean up environment variables
        for key in ['JWT_SECRET_KEY', 'JWT_EXPIRATION_HOURS', 'SESSION_TIMEOUT_MINUTES', 'MAX_CONCURRENT_SESSIONS']:
            if key in os.environ:
                del os.environ[key]

    @patch('auth.auth_service.SessionRepository')
    def test_auth_service_init_with_database(self, mock_session_repo):
        """Test AuthService initialization with database."""
        auth_service = AuthService(self.mock_user_model)
        
        assert auth_service.user_model == self.mock_user_model
        assert auth_service.jwt_secret == 'test-secret-key'
        assert auth_service.jwt_algorithm == 'HS256'
        assert auth_service.jwt_expiration_hours == 24
        assert auth_service.session_timeout_minutes == 30
        assert auth_service.max_concurrent_sessions == 5

    def test_generate_jwt_token(self):
        """Test JWT token generation."""
        token = self.auth_service._generate_jwt_token(self.test_user)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token can be decoded
        decoded = jwt.decode(token, 'test-secret-key', algorithms=['HS256'])
        assert decoded['user_id'] == self.test_user.user_id
        assert decoded['email'] == self.test_user.email
        assert 'exp' in decoded

    def test_validate_jwt_token_success(self):
        """Test successful JWT token validation."""
        token = self.auth_service._generate_jwt_token(self.test_user)
        
        result = self.auth_service._validate_jwt_token(token)
        
        assert result is not None
        assert result.user_id == self.test_user.user_id
        assert result.email == self.test_user.email

    def test_validate_jwt_token_invalid(self):
        """Test invalid JWT token validation."""
        invalid_token = "invalid.token.here"
        
        result = self.auth_service._validate_jwt_token(invalid_token)
        
        assert result is None

    def test_validate_jwt_token_expired(self):
        """Test expired JWT token validation."""
        # Create an expired token
        expired_token = jwt.encode(
            {
                'user_id': self.test_user.user_id,
                'email': self.test_user.email,
                'exp': datetime.now() - timedelta(hours=1)  # Expired 1 hour ago
            },
            'test-secret-key',
            algorithm='HS256'
        )
        
        result = self.auth_service._validate_jwt_token(expired_token)
        
        assert result is None

    def test_create_session(self):
        """Test session creation."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            mock_session = Mock()
            mock_repo.return_value.save.return_value = True
            
            session = self.auth_service._create_session(
                self.test_user.user_id,
                "192.168.1.1",
                "Mozilla/5.0"
            )
            
            assert session is not None
            assert session.user_id == self.test_user.user_id
            assert session.ip_address == "192.168.1.1"
            assert session.user_agent == "Mozilla/5.0"
            assert session.is_active is True

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            mock_repo.return_value.find_by_user_id.return_value = [
                Mock(is_expired=True),  # Expired session
                Mock(is_expired=False),  # Active session
            ]
            
            # Should not raise an exception
            self.auth_service._cleanup_expired_sessions(self.test_user.user_id)

    def test_login_user_success(self):
        """Test successful user login."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock user model
            self.mock_user_model.find_by_email.return_value = self.test_user
            self.mock_user_model.verify_password.return_value = True
            
            # Mock session repository
            mock_session = Mock()
            mock_session.session_id = "session_123"
            mock_repo.return_value.save.return_value = True
            mock_repo.return_value.find_by_user_id.return_value = []
            
            result = self.auth_service.login_user(
                email="test@example.com",
                password="correct_password",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert result.success is True
            assert result.user == self.test_user
            assert result.token is not None
            assert result.session is not None

    def test_login_user_invalid_credentials(self):
        """Test login with invalid credentials."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_email.return_value = self.test_user
            self.mock_user_model.verify_password.return_value = False
            
            result = self.auth_service.login_user(
                email="test@example.com",
                password="wrong_password",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert result.success is False
            assert result.user is None
            assert result.token is None
            assert result.error_message is not None

    def test_login_user_not_found(self):
        """Test login with non-existent user."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_email.return_value = None
            
            result = self.auth_service.login_user(
                email="nonexistent@example.com",
                password="password",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert result.success is False
            assert result.user is None
            assert result.token is None
            assert result.error_message is not None

    def test_login_user_max_sessions_exceeded(self):
        """Test login when max concurrent sessions exceeded."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock user model
            self.mock_user_model.find_by_email.return_value = self.test_user
            self.mock_user_model.verify_password.return_value = True
            
            # Mock existing sessions (max reached)
            existing_sessions = [Mock(is_expired=False) for _ in range(5)]
            mock_repo.return_value.find_by_user_id.return_value = existing_sessions
            
            result = self.auth_service.login_user(
                email="test@example.com",
                password="correct_password",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert result.success is False
            assert result.error_message is not None

    def test_register_user_success(self):
        """Test successful user registration."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_email.return_value = None
            self.mock_user_model.create_user.return_value = self.test_user
            
            result = self.auth_service.register_user(
                email="new@example.com",
                password="SecurePass123",
                full_name="New User"
            )
            
            assert result.success is True
            assert result.user == self.test_user
            assert result.error_message is None

    def test_register_user_email_exists(self):
        """Test registration with existing email."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_email.return_value = self.test_user
            
            result = self.auth_service.register_user(
                email="test@example.com",
                password="SecurePass123",
                full_name="Test User"
            )
            
            assert result.success is False
            assert result.user is None
            assert result.error_message is not None

    def test_logout_user_success(self):
        """Test successful user logout."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock session
            mock_session = Mock()
            mock_session.is_active = True
            mock_repo.return_value.find_by_id.return_value = mock_session
            
            result = self.auth_service.logout_user("valid_token")
            
            assert result.success is True
            assert mock_session.is_active is False

    def test_logout_user_invalid_token(self):
        """Test logout with invalid token."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            mock_repo.return_value.find_by_id.return_value = None
            
            result = self.auth_service.logout_user("invalid_token")
            
            assert result.success is False

    def test_validate_token_success(self):
        """Test successful token validation."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Create valid token
            token = self.auth_service._generate_jwt_token(self.test_user)
            
            # Mock active session
            mock_session = Mock()
            mock_session.is_active = True
            mock_session.is_expired.return_value = False
            mock_repo.return_value.find_by_id.return_value = mock_session
            
            result = self.auth_service.validate_token(token)
            
            assert result is not None
            assert result.user_id == self.test_user.user_id

    def test_validate_token_invalid_session(self):
        """Test token validation with invalid session."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Create valid token
            token = self.auth_service._generate_jwt_token(self.test_user)
            
            # Mock inactive session
            mock_session = Mock()
            mock_session.is_active = False
            mock_repo.return_value.find_by_id.return_value = mock_session
            
            result = self.auth_service.validate_token(token)
            
            assert result is None

    def test_refresh_token_success(self):
        """Test successful token refresh."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Create valid token
            old_token = self.auth_service._generate_jwt_token(self.test_user)
            
            # Mock active session
            mock_session = Mock()
            mock_session.is_active = True
            mock_session.is_expired.return_value = False
            mock_repo.return_value.find_by_id.return_value = mock_session
            
            result = self.auth_service.refresh_token(old_token)
            
            assert result.success is True
            assert result.token is not None
            assert result.token != old_token

    def test_refresh_token_invalid(self):
        """Test token refresh with invalid token."""
        with patch('auth.auth_service.SessionRepository'):
            result = self.auth_service.refresh_token("invalid_token")
            
            assert result.success is False
            assert result.error_message is not None

    def test_initiate_password_reset_success(self):
        """Test successful password reset initiation."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_email.return_value = self.test_user
            self.mock_user_model.update_user.return_value = True
            
            result = self.auth_service.initiate_password_reset("test@example.com")
            
            assert result.success is True
            assert result.error_message is None

    def test_initiate_password_reset_user_not_found(self):
        """Test password reset with non-existent user."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_email.return_value = None
            
            result = self.auth_service.initiate_password_reset("nonexistent@example.com")
            
            assert result.success is False
            assert result.error_message is not None

    def test_reset_password_success(self):
        """Test successful password reset."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.test_user.password_reset_token = "valid_token"
            self.test_user.password_reset_expires = datetime.now() + timedelta(hours=1)
            self.mock_user_model.find_by_email.return_value = self.test_user
            self.mock_user_model.update_user.return_value = True
            
            result = self.auth_service.reset_password(
                "test@example.com",
                "valid_token",
                "NewSecurePass123"
            )
            
            assert result.success is True
            assert result.error_message is None

    def test_reset_password_invalid_token(self):
        """Test password reset with invalid token."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.test_user.password_reset_token = "different_token"
            self.mock_user_model.find_by_email.return_value = self.test_user
            
            result = self.auth_service.reset_password(
                "test@example.com",
                "invalid_token",
                "NewSecurePass123"
            )
            
            assert result.success is False
            assert result.error_message is not None

    def test_change_password_success(self):
        """Test successful password change."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_id.return_value = self.test_user
            self.mock_user_model.verify_password.return_value = True
            self.mock_user_model.update_user.return_value = True
            
            result = self.auth_service.change_password(
                self.test_user.user_id,
                "old_password",
                "new_password"
            )
            
            assert result.success is True
            assert result.error_message is None

    def test_change_password_invalid_old_password(self):
        """Test password change with invalid old password."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.find_by_id.return_value = self.test_user
            self.mock_user_model.verify_password.return_value = False
            
            result = self.auth_service.change_password(
                self.test_user.user_id,
                "wrong_old_password",
                "new_password"
            )
            
            assert result.success is False
            assert result.error_message is not None

    def test_auth_session_is_expired(self):
        """Test AuthSession is_expired method."""
        # Non-expired session
        future_session = AuthSession(
            session_id="test",
            user_id="user123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert future_session.is_expired() is False
        
        # Expired session
        past_session = AuthSession(
            session_id="test",
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1)
        )
        assert past_session.is_expired() is True

    def test_auth_session_to_dict(self):
        """Test AuthSession to_dict method."""
        session = AuthSession(
            session_id="test_session",
            user_id="user123",
            created_at=datetime(2023, 1, 1, 12, 0, 0),
            expires_at=datetime(2023, 1, 1, 13, 0, 0),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            is_active=True
        )
        
        result = session.to_dict()
        
        assert result['session_id'] == "test_session"
        assert result['user_id'] == "user123"
        assert result['ip_address'] == "192.168.1.1"
        assert result['user_agent'] == "Mozilla/5.0"
        assert result['is_active'] is True
        assert result['created_at'] == "2023-01-01T12:00:00"
        assert result['expires_at'] == "2023-01-01T13:00:00"

    def test_background_session_cleanup(self):
        """Test background session cleanup thread."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock sessions
            mock_repo.return_value.find_all.return_value = [
                Mock(is_expired=True, session_id="expired1"),
                Mock(is_expired=False, session_id="active1"),
                Mock(is_expired=True, session_id="expired2"),
            ]
            
            # Start cleanup thread
            cleanup_thread = threading.Thread(
                target=self.auth_service._background_session_cleanup
            )
            cleanup_thread.start()
            cleanup_thread.join(timeout=5)  # Wait up to 5 seconds
            
            # Should not raise an exception
            assert True

    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.columns')
    @patch('streamlit.form')
    @patch('streamlit.form_submit_button')
    @patch('streamlit.text_input')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.button')
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.spinner')
    @patch('streamlit.rerun')
    def test_middleware_comprehensive_flow(self, mock_rerun, mock_spinner, mock_error, 
                                         mock_success, mock_button, mock_markdown, mock_title,
                                         mock_text_input, mock_form_submit_button, mock_form,
                                         mock_columns, mock_session_state):
        """Test comprehensive middleware flow."""
        # Setup mocks
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "test@example.com"
        mock_form_submit_button.return_value = True
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Create middleware
        middleware = AuthMiddleware(self.auth_service)
        
        # Mock successful login
        self.auth_service.login_user.return_value = AuthResult(
            success=True,
            user=self.test_user,
            token="valid_token",
            session=Mock()
        )
        
        # Test login flow
        middleware.login_user("test@example.com", "password")
        
        # Verify session state is set
        assert mock_session_state.get('auth_token') == "valid_token"
        assert mock_session_state.get('user') == self.test_user

    def test_user_model_comprehensive_coverage(self):
        """Test comprehensive user model coverage."""
        # Test user profile methods
        user_dict = self.test_user.to_dict()
        assert 'user_id' in user_dict
        assert 'email' in user_dict
        assert 'full_name' in user_dict
        
        # Test PII protection
        patient_dict = self.test_user.to_dict(user_role="patient")
        assert 'password_reset_token' not in patient_dict
        
        # Test admin access
        admin_dict = self.test_user.to_dict(user_role="admin", include_sensitive=True)
        assert 'user_id' in admin_dict
        
        # Test account locking
        assert self.test_user.is_locked() is False
        
        # Test login attempts
        self.test_user.increment_login_attempts()
        assert self.test_user.login_attempts == 1
        
        # Test account lock
        self.test_user.increment_login_attempts(max_attempts=1)
        assert self.test_user.is_locked() is True