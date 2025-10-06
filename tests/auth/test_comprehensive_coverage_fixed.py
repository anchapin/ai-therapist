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
        # Create a test session
        test_session = AuthSession(
            session_id="session_123",
            user_id=self.test_user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        token = self.auth_service._generate_jwt_token(self.test_user, test_session)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token can be decoded
        decoded = jwt.decode(token, 'test-secret-key', algorithms=['HS256'])
        assert decoded['user_id'] == self.test_user.user_id
        assert decoded['email'] == self.test_user.email
        assert 'exp' in decoded

    def test_validate_token_success(self):
        """Test successful token validation."""
        # Create a test session
        test_session = AuthSession(
            session_id="session_123",
            user_id=self.test_user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        token = self.auth_service._generate_jwt_token(self.test_user, test_session)
        
        with patch.object(self.auth_service, '_is_session_valid', return_value=True):
            with patch.object(self.mock_user_model, 'get_user', return_value=self.test_user):
                result = self.auth_service.validate_token(token)
                
                assert result is not None
                assert result.user_id == self.test_user.user_id

    def test_validate_token_invalid(self):
        """Test invalid token validation."""
        invalid_token = "invalid.token.here"
        
        result = self.auth_service.validate_token(invalid_token)
        
        assert result is None

    def test_validate_token_expired(self):
        """Test expired token validation."""
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
        
        result = self.auth_service.validate_token(expired_token)
        
        assert result is None

    @patch('auth.auth_service.SessionRepository')
    def test_create_session(self, mock_repo):
        """Test session creation."""
        mock_session = Mock()
        mock_session.session_id = "session_123"
        mock_session.user_id = self.test_user.user_id
        mock_session.created_at = datetime.now()
        mock_session.expires_at = datetime.now() + timedelta(hours=1)
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        
        mock_repo.return_value.save.return_value = True
        mock_repo.return_value.find_by_user_id.return_value = []
        
        with patch('auth.auth_service.Session') as mock_session_class:
            mock_session_class.create.return_value = mock_session
            
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
            # Mock database manager
            with patch('auth.auth_service.get_database_manager') as mock_db_manager:
                mock_db = Mock()
                mock_db.cleanup_expired_data.return_value = 5
                mock_db_manager.return_value = mock_db
                
                # Should not raise an exception
                self.auth_service._cleanup_expired_sessions()

    def test_login_user_success(self):
        """Test successful user login."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock user model
            self.mock_user_model.authenticate_user.return_value = self.test_user
            
            # Mock session creation
            test_session = AuthSession(
                session_id="session_123",
                user_id=self.test_user.user_id,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            with patch.object(self.auth_service, '_create_session', return_value=test_session):
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
            self.mock_user_model.authenticate_user.return_value = None
            
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

    def test_login_user_account_not_active(self):
        """Test login with inactive account."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            inactive_user = UserProfile(
                user_id="test_user_123",
                email="test@example.com",
                full_name="Test User",
                role=UserRole.PATIENT,
                status=UserStatus.INACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.mock_user_model.authenticate_user.return_value = inactive_user
            
            result = self.auth_service.login_user(
                email="test@example.com",
                password="correct_password",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert result.success is False
            assert result.error_message == "Account is not active"

    def test_login_user_account_locked(self):
        """Test login with locked account."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            locked_user = UserProfile(
                user_id="test_user_123",
                email="test@example.com",
                full_name="Test User",
                role=UserRole.PATIENT,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                account_locked_until=datetime.now() + timedelta(hours=1)
            )
            self.mock_user_model.authenticate_user.return_value = locked_user
            
            result = self.auth_service.login_user(
                email="test@example.com",
                password="correct_password",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert result.success is False
            assert result.error_message == "Account is temporarily locked"

    def test_register_user_success(self):
        """Test successful user registration."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
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
            self.mock_user_model.create_user.side_effect = ValueError("User with this email already exists")
            
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
            # Create valid token
            test_session = AuthSession(
                session_id="session_123",
                user_id=self.test_user.user_id,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            token = self.auth_service._generate_jwt_token(self.test_user, test_session)
            
            # Mock session
            mock_session = Mock()
            mock_session.is_active = True
            mock_repo.return_value.find_by_id.return_value = mock_session
            
            result = self.auth_service.logout_user(token)
            
            assert result is True

    def test_logout_user_invalid_token(self):
        """Test logout with invalid token."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            mock_repo.return_value.find_by_id.return_value = None
            
            result = self.auth_service.logout_user("invalid_token")
            
            assert result is False

    def test_validate_token_invalid_session(self):
        """Test token validation with invalid session."""
        # Create valid token
        test_session = AuthSession(
            session_id="session_123",
            user_id=self.test_user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        token = self.auth_service._generate_jwt_token(self.test_user, test_session)
        
        with patch.object(self.auth_service, '_is_session_valid', return_value=False):
            with patch.object(self.mock_user_model, 'get_user', return_value=self.test_user):
                result = self.auth_service.validate_token(token)
                
                assert result is None

    def test_refresh_token_success(self):
        """Test successful token refresh."""
        # Create valid token
        test_session = AuthSession(
            session_id="session_123",
            user_id=self.test_user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        old_token = self.auth_service._generate_jwt_token(self.test_user, test_session)
        
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock active session
            mock_session = Mock()
            mock_session.is_active = True
            mock_session.is_expired.return_value = False
            mock_session.session_id = "session_123"
            mock_session.user_id = self.test_user.user_id
            mock_session.created_at = datetime.now()
            mock_session.expires_at = datetime.now() + timedelta(hours=1)
            mock_session.ip_address = "192.168.1.1"
            mock_session.user_agent = "Mozilla/5.0"
            
            mock_repo.return_value.find_by_id.return_value = mock_session
            mock_repo.return_value.save.return_value = True
            
            with patch.object(self.mock_user_model, 'get_user', return_value=self.test_user):
                with patch.object(self.auth_service, '_is_session_valid', return_value=True):
                    result = self.auth_service.refresh_token(old_token)
                    
                    assert result is not None
                    assert result != old_token

    def test_refresh_token_invalid(self):
        """Test token refresh with invalid token."""
        with patch('auth.auth_service.SessionRepository'):
            result = self.auth_service.refresh_token("invalid_token")
            
            assert result is None

    def test_initiate_password_reset_success(self):
        """Test successful password reset initiation."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.initiate_password_reset.return_value = "reset_token_123"
            
            result = self.auth_service.initiate_password_reset("test@example.com")
            
            assert result.success is True
            assert result.error_message is None

    def test_initiate_password_reset_user_not_found(self):
        """Test password reset with non-existent user."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.initiate_password_reset.return_value = None
            
            result = self.auth_service.initiate_password_reset("nonexistent@example.com")
            
            assert result.success is False
            assert result.error_message is not None

    def test_reset_password_success(self):
        """Test successful password reset."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.reset_password.return_value = True
            
            result = self.auth_service.reset_password(
                "valid_token",
                "NewSecurePass123"
            )
            
            assert result.success is True
            assert result.error_message is None

    def test_reset_password_invalid_token(self):
        """Test password reset with invalid token."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.reset_password.return_value = False
            
            result = self.auth_service.reset_password(
                "invalid_token",
                "NewSecurePass123"
            )
            
            assert result.success is False
            assert result.error_message is not None

    def test_change_password_success(self):
        """Test successful password change."""
        with patch('auth.auth_service.SessionRepository'):
            # Mock user model
            self.mock_user_model.change_password.return_value = True
            
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
            self.mock_user_model.change_password.return_value = False
            
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

    def test_background_cleanup(self):
        """Test background cleanup thread."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock sessions
            mock_repo.return_value.find_all.return_value = [
                Mock(is_expired=True, session_id="expired1"),
                Mock(is_expired=False, session_id="active1"),
                Mock(is_expired=True, session_id="expired2"),
            ]
            
            # Mock database manager
            with patch('auth.auth_service.get_database_manager') as mock_db_manager:
                mock_db = Mock()
                mock_db.cleanup_expired_data.return_value = 2
                mock_db_manager.return_value = mock_db
                
                # Should not raise an exception
                self.auth_service._cleanup_expired_sessions()

    def test_get_user_sessions(self):
        """Test getting user sessions."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock database sessions
            mock_db_session = Mock()
            mock_db_session.session_id = "session_123"
            mock_db_session.user_id = self.test_user.user_id
            mock_db_session.created_at = datetime.now()
            mock_db_session.expires_at = datetime.now() + timedelta(hours=1)
            mock_db_session.ip_address = "192.168.1.1"
            mock_db_session.user_agent = "Mozilla/5.0"
            mock_db_session.is_active = True
            
            mock_repo.return_value.find_by_user_id.return_value = [mock_db_session]
            
            sessions = self.auth_service.get_user_sessions(self.test_user.user_id)
            
            assert len(sessions) == 1
            assert sessions[0].session_id == "session_123"
            assert sessions[0].user_id == self.test_user.user_id

    def test_invalidate_user_sessions(self):
        """Test invalidating user sessions."""
        with patch('auth.auth_service.SessionRepository') as mock_repo:
            # Mock database sessions
            mock_session1 = Mock()
            mock_session1.session_id = "session_1"
            mock_session1.is_active = True
            
            mock_session2 = Mock()
            mock_session2.session_id = "session_2"
            mock_session2.is_active = True
            
            mock_repo.return_value.find_by_user_id.return_value = [mock_session1, mock_session2]
            mock_repo.return_value.save.return_value = True
            
            invalidated = self.auth_service.invalidate_user_sessions(
                self.test_user.user_id,
                keep_current="session_2"
            )
            
            assert invalidated == 1
            assert mock_session1.is_active is False
            assert mock_session2.is_active is True

    def test_validate_session_access(self):
        """Test session access validation."""
        # Mock user with permissions
        mock_user = Mock()
        mock_user.can_access_resource.return_value = True
        
        with patch.object(self.mock_user_model, 'get_user', return_value=mock_user):
            result = self.auth_service.validate_session_access(
                self.test_user.user_id,
                "therapy_sessions",
                "read"
            )
            
            assert result is True

    def test_get_auth_statistics(self):
        """Test getting authentication statistics."""
        with patch('auth.auth_service.get_database_manager') as mock_db_manager:
            mock_db = Mock()
            mock_db.health_check.return_value = {
                'table_counts': {
                    'users': 10,
                    'sessions': 5
                }
            }
            mock_db_manager.return_value = mock_db
            
            stats = self.auth_service.get_auth_statistics()
            
            assert stats['total_users'] == 10
            assert stats['active_sessions'] == 5
            assert stats['total_sessions_created'] == 5

    def test_filter_user_for_response(self):
        """Test filtering user data for response."""
        # Test with admin role
        filtered = self.auth_service._filter_user_for_response(
            self.test_user,
            requesting_user_role='admin'
        )
        
        assert 'user_id' in filtered
        assert 'email' in filtered
        
        # Test with patient role
        filtered = self.auth_service._filter_user_for_response(
            self.test_user,
            requesting_user_role='patient'
        )
        
        assert 'user_id' in filtered
        assert 'email' in filtered

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
        
        # Test reset login attempts
        self.test_user.reset_login_attempts()
        assert self.test_user.login_attempts == 0
        assert self.test_user.status == UserStatus.ACTIVE

    def test_user_model_role_permissions(self):
        """Test user role permissions."""
        # Test patient permissions
        patient_permissions = self.test_user._get_role_permissions()
        assert 'own_profile' in patient_permissions
        assert 'read' in patient_permissions['own_profile']
        
        # Test can access resource
        assert self.test_user.can_access_resource('own_profile', 'read') is True
        assert self.test_user.can_access_resource('system_config', 'update') is False

    def test_user_model_medical_info_sanitization(self):
        """Test medical information sanitization."""
        medical_info = {
            'condition': 'Anxiety',
            'medication': 'Sertraline',
            'allergies': 'Penicillin',
            'treatment_history': 'Long-term therapy'
        }
        
        # Test patient view
        patient_view = self.test_user._sanitize_medical_info(medical_info, 'patient')
        assert 'condition' not in patient_view
        assert 'medication' not in patient_view
        assert '_sanitized' in patient_view
        
        # Test therapist view
        therapist_view = self.test_user._sanitize_medical_info(medical_info, 'therapist')
        assert 'condition' in therapist_view
        assert 'medication' in therapist_view
        assert 'treatment_history' not in therapist_view
        
        # Test admin view
        admin_view = self.test_user._sanitize_medical_info(medical_info, 'admin')
        assert 'condition' in admin_view
        assert 'medication' in admin_view
        assert 'treatment_history' in admin_view

    def test_user_model_email_masking(self):
        """Test email masking for privacy."""
        # Short email
        short_email = "ab@example.com"
        masked_short = self.test_user._mask_email(short_email)
        assert masked_short == "**@example.com"
        
        # Long email
        long_email = "username@example.com"
        masked_long = self.test_user._mask_email(long_email)
        assert masked_long == "u******e@example.com"

    def test_user_model_create_user_validation(self):
        """Test user creation validation."""
        user_model = UserModel()
        
        # Test invalid email
        with pytest.raises(ValueError, match="Invalid email format"):
            user_model.create_user("invalid-email", "Password123", "Test User")
        
        # Test weak password
        with pytest.raises(ValueError, match="Password does not meet security requirements"):
            user_model.create_user("test@example.com", "weak", "Test User")

    def test_user_model_authenticate_user(self):
        """Test user authentication."""
        user_model = UserModel()
        
        # Test non-existent user
        result = user_model.authenticate_user("nonexistent@example.com", "password")
        assert result is None
        
        # Test with mock user
        with patch.object(user_model.user_repo, 'find_by_email') as mock_find:
            mock_user = Mock()
            mock_user.status = UserStatus.ACTIVE
            mock_user.is_locked.return_value = False
            mock_user.reset_login_attempts = Mock()
            mock_user.last_login = datetime.now()
            mock_user.updated_at = datetime.now()
            
            mock_find.return_value = mock_user
            
            with patch.object(user_model, '_verify_password', return_value=True):
                result = user_model.authenticate_user("test@example.com", "password")
                assert result is not None

    def test_user_model_change_password_validation(self):
        """Test password change validation."""
        user_model = UserModel()
        
        # Test non-existent user
        result = user_model.change_password("nonexistent", "old", "new")
        assert result is False
        
        # Test with mock user
        with patch.object(user_model.user_repo, 'find_by_id') as mock_find:
            mock_user = Mock()
            mock_find.return_value = mock_user
            
            with patch.object(user_model, '_verify_password', return_value=False):
                result = user_model.change_password("user123", "wrong_old", "new")
                assert result is False
            
            with patch.object(user_model, '_verify_password', return_value=True):
                with patch.object(user_model, '_validate_password', return_value=False):
                    result = user_model.change_password("user123", "correct_old", "weak")
                    assert result is False