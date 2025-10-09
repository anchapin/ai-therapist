"""
Comprehensive unit tests to complete 50% coverage target for auth/auth_service.py.
Focuses on core authentication, session management, and security functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import with robust error handling and mock patches
import sys
import os
import jwt
import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Set environment variables early to avoid import issues
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key"
os.environ["JWT_EXPIRATION_HOURS"] = "24"
os.environ["SESSION_TIMEOUT_MINUTES"] = "30"
os.environ["MAX_CONCURRENT_SESSIONS"] = "5"

# Import path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Patch problematic imports before importing auth_service
import importlib
sys.modules['database.db_manager'] = Mock()
sys.modules['database.models'] = Mock()
sys.modules['..database.db_manager'] = Mock()
sys.modules['..database.models'] = Mock()

try:
    from auth.auth_service import AuthService, AuthSession, AuthResult
    from auth.user_model import UserModel, UserProfile, UserRole, UserStatus
    AUTH_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Auth import error: {e}")
    AUTH_MODULE_AVAILABLE = False

# Import with mocking if module not available
if not AUTH_MODULE_AVAILABLE:
    class MockAuthService:
        def __init__(self, *args, **kwargs):
            pass
    class MockAuthSession:
        def __init__(self, *args, **kwargs):
            pass
    class MockAuthResult:
        def __init__(self, *args, **kwargs):
            pass
    
    AuthService = MockAuthService
    AuthSession = MockAuthSession
    AuthResult = MockAuthResult
    
    # Create mock classes for missing dependencies
    class MockUserModel:
        pass
    class MockUserProfile:
        def __init__(self, *args, **kwargs):
            self.user_id = "test"
            self.email = "test@example.com"
            self.full_name = "Test User"
    
    class MockUserRole:
        PATIENT = "patient"
        ADMIN = "admin"
        THERAPIST = "therapist"
    
    class MockUserStatus:
        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"
    
    UserModel = MockUserModel
    UserProfile = MockUserProfile
    UserRole = MockUserRole
    UserStatus = MockUserStatus


class TestAuthService50PercentCoverage:
    """Comprehensive unit tests to reach 50% coverage for auth_service.py."""
    
    @pytest.fixture(autouse=True)
    def patch_database_imports(self):
        """Patch database imports to avoid relative import issues."""
        with patch.dict('sys.modules', {
            'database.db_manager': Mock(),
            'database.models': Mock()
        }):
            yield
    
    @pytest.fixture
    def mock_user_model(self):
        """Create a mock user model with all required methods."""
        return Mock(spec=UserModel,
            get_user_by_id=Mock(return_value=None),
            validate_password=Mock(return_value=True),
            update_password=Mock(return_value=True),
            get_user_by_email=Mock(return_value=None),
            verify_password=Mock(return_value=True),
            create_user=Mock(return_value=None),
            generate_password_reset_token=Mock(return_value="reset_token_123"),
            send_password_reset_email=Mock(return_value=True)
        )
    
    @pytest.fixture
    def auth_service(self, mock_user_model):
        """Create an AuthService with mocked dependencies."""
        return AuthService(mock_user_model)
    
    @pytest.fixture
    def valid_user_profile(self):
        """Create a valid user profile for testing."""
        return UserProfile(
            user_id="user123",
            email="user@example.com",
            full_name="John Doe",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_change_password_success(self, auth_service, valid_user_profile):
        """Test successful password change."""
        current_password = "oldpassword123"
        new_password = "newpassword456"
        
        # Mock user lookup and password change
        auth_service.user_model.change_password.return_value = True
        
        result = auth_service.change_password("user123", current_password, new_password)
        
        assert result.success is True
        assert result.user is None  # Actual implementation doesn't return user
        assert result.token is None
        assert result.session is None
        assert result.error_message is None
        
        # Verify the method was called
        auth_service.user_model.change_password.assert_called_once_with("user123", current_password, new_password)
    
    def test_change_password_invalid_current(self, auth_service, valid_user_profile):
        """Test password change with invalid current password."""
        current_password = "wrongpassword"
        new_password = "newpassword456"
        
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        auth_service.user_model.validate_password.return_value = False  # Wrong current password
        
        result = auth_service.change_password("user123", current_password, new_password)
        
        assert result.success is False
        assert result.error_message == "Current password is incorrect"
        assert result.user is None
        
        # Should not update password
        auth_service.user_model.update_password.assert_not_called()
    
    def test_change_password_invalid_new_password(self, auth_service, valid_user_profile):
        """Test password change with invalid new password."""
        current_password = "oldpassword123"
        new_password = "weak"  # Too short
        
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        auth_service.user_model.validate_password.return_value = True  # Current password valid
        auth_service.user_model.validate_password.side_effect = [True, False]  # New password invalid
        
        result = auth_service.change_password("user123", current_password, new_password)
        
        assert result.success is False
        assert result.error_message == "New password is invalid"
        assert result.user is None
        
        # Should not update password
        auth_service.user_model.update_password.assert_not_called()
    
    def test_change_password_user_not_found(self, auth_service):
        """Test password change for nonexistent user."""
        current_password = "oldpassword123"
        new_password = "newpassword456"
        
        auth_service.user_model.get_user_by_id.return_value = None
        
        result = auth_service.change_password("nonexistent_user", current_password, new_password)
        
        assert result.success is False
        assert result.error_message == "User not found"
        assert result.user is None
    
    def test_change_password_database_error(self, auth_service, valid_user_profile):
        """Test password change with database error."""
        current_password = "oldpassword123"
        new_password = "newpassword456"
        
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        auth_service.user_model.validate_password.return_value = True
        auth_service.user_model.update_password.side_effect = Exception("Database error")
        
        result = auth_service.change_password("user123", current_password, new_password)
        
        assert result.success is False
        assert result.error_message == "Password change failed"
        assert result.user is None
    
    def test_get_auth_statistics_empty(self, auth_service):
        """Test authentication statistics with no sessions."""
        # Mock empty repository
        auth_service.session_repo.find_by_user_id.return_value = []
        
        stats = auth_service.get_auth_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'expired_sessions_today' in stats
        assert 'failed_login_attempts_today' in stats
        assert 'successful_logins_today' in stats
        assert 'password_resets_today' in stats
        assert 'registration_requests_today' in stats
        assert 'average_session_duration' in stats
        
        # Should be zero when empty
        assert stats['total_sessions'] == 0
        assert stats['active_sessions'] == 0
        assert stats['expired_sessions_today'] == 0
    
    def test_get_auth_statistics_with_sessions(self, auth_service):
        """Test authentication statistics with active sessions."""
        # Mock session data
        active_sessions = [
            AuthSession(
                session_id="sess1",
                user_id="user1",
                created_at=datetime.now() - timedelta(hours=2),
                expires_at=datetime.now() + timedelta(hours=1),
                is_active=True
            ),
            AuthSession(
                session_id="sess2",
                user_id="user2",
                created_at=datetime.now() - timedelta(hours=5),
                expires_at=datetime.now() + timedelta(hours=3),
                is_active=True
            ),
            AuthSession(
                session_id="sess3",
                user_id="user3",
                created_at=datetime.now() - timedelta(hours=24),
                expires_at=datetime.now() - timedelta(hours=1),  # Expired
                is_active=False
            )
        ]
        
        auth_service.session_repo.find_by_user_id.return_value = active_sessions[:2]
        
        stats = auth_service.get_auth_statistics()
        
        assert stats['total_sessions'] >= 2
        assert stats['active_sessions'] >= 2
        assert isinstance(stats['failed_login_attempts_today'], int)
        assert isinstance(stats['successful_logins_today'], int)
    
    def test_get_user_sessions_single_session(self, auth_service, valid_user_profile):
        """Test getting user sessions for a single active session."""
        session = AuthSession(
            session_id="sess1",
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_user_id.return_value = [session]
        
        sessions = auth_service.get_user_sessions("user123")
        
        assert isinstance(sessions, list)
        assert len(sessions) == 1
        assert sessions[0].session_id == "sess1"
        assert sessions[0].user_id == "user123"
        assert sessions[0].is_active is True
    
    def test_get_user_sessions_multiple_sessions(self, auth_service, valid_user_profile):
        """Test getting user sessions for multiple sessions."""
        sessions = [
            AuthSession(
                session_id="sess1",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=3),
                expires_at=datetime.now() + timedelta(hours=2),
                is_active=True
            ),
            AuthSession(
                session_id="sess2",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1),
                is_active=True
            ),
            AuthSession(
                session_id="sess3",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=24),
                expires_at=datetime.now() - timedelta(hours=1),
                is_active=False
            )
        ]
        
        auth_service.session_repo.find_by_user_id.return_value = sessions
        
        all_sessions = auth_service.get_user_sessions("user123")
        active_sessions = auth_service.get_user_sessions("user123", active_only=True)
        
        assert len(all_sessions) == 3
        assert len(active_sessions) == 2
        
        # Check active sessions
        for session in active_sessions:
            assert session.is_active is True
            assert session.user_id == "user123"
    
    def test_get_user_sessions_no_sessions(self, auth_service):
        """Test getting user sessions with no sessions."""
        auth_service.session_repo.find_by_user_id.return_value = []
        
        sessions = auth_service.get_user_sessions("nonexistent_user")
        
        assert isinstance(sessions, list)
        assert len(sessions) == 0
    
    def test_get_user_sessions_active_only_filter(self, auth_service):
        """Test filtering active-only sessions."""
        sessions = [
            AuthSession(
                session_id="sess1",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1),
                is_active=True
            ),
            AuthSession(
                session_id="sess2",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=2),
                expires_at=datetime.now() - timedelta(hours=1),
                is_active=False
            )
        ]
        
        auth_service.session_repo.find_by_user_id.return_value = sessions
        
        all_sessions = auth_service.get_user_sessions("user123")
        active_sessions = auth_service.get_user_sessions("user123", active_only=True)
        
        assert len(all_sessions) == 2
        assert len(active_sessions) == 1
        assert active_sessions[0].is_active is True
        assert active_sessions[0].session_id == "sess1"
    
    def test_initiate_password_reset_success(self, auth_service, valid_user_profile):
        """Test successful password reset initiation."""
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.generate_password_reset_token.return_value = "reset_token_123"
        
        result = auth_service.initiate_password_reset("user@example.com")
        
        assert result.success is True
        assert result.error_message is None
        assert "reset_token_123" in result.message or result.message is not None
        
        # Verify method calls
        auth_service.user_model.get_user_by_email.assert_called_once_with("user@example.com")
        auth_service.user_model.generate_password_reset_token.assert_called_once_with(valid_user_profile.user_id)
    
    def test_initiate_password_reset_user_not_found(self, auth_service):
        """Test password reset for nonexistent user."""
        auth_service.user_model.get_user_by_email.return_value = None
        
        result = auth_service.initiate_password_reset("nonexistent@example.com")
        
        assert result.success is False
        assert result.error_message == "User not found"
        assert result.message is None
    
    def test_initiate_password_reset_token_generation_error(self, auth_service, valid_user_profile):
        """Test password reset with token generation error."""
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.generate_password_reset_token.side_effect = Exception("Token error")
        
        result = auth_service.initiate_password_reset("user@example.com")
        
        assert result.success is False
        assert result.error_message == "Failed to generate reset token"
        assert result.message is None
    
    def test_initiate_password_reset_email_error(self, auth_service, valid_user_profile):
        """Test password reset with email sending error."""
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.generate_password_reset_token.return_value = "reset_token_123"
        auth_service.user_model.send_password_reset_email.side_effect = Exception("Email error")
        
        result = auth_service.initiate_password_reset("user@example.com")
        
        assert result.success is False
        assert "Password reset email failed" in result.error_message
        assert result.message is None
    
    def test_invalidate_user_sessions_success(self, auth_service, valid_user_profile):
        """Test successful session invalidation."""
        sessions = [
            AuthSession(
                session_id="sess1",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1),
                is_active=True
            ),
            AuthSession(
                session_id="sess2",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=2),
                expires_at=datetime.now() + timedelta(hours=2),
                is_active=True
            )
        ]
        
        auth_service.session_repo.find_by_user_id.return_value = sessions
        auth_service.session_repo.save.side_effect = lambda session: True
        
        result = auth_service.invalidate_user_sessions("user123")
        
        assert result.success is True
        assert result.error_message is None
        assert result.invalidated_sessions == 2
        
        # Verify all sessions were marked inactive
        for session in sessions:
            assert session.is_active is False
        
        # Verify sessions were saved
        assert auth_service.session_repo.save.call_count == 2
    
    def test_invalidate_user_sessions_no_sessions(self, auth_service):
        """Test session invalidation with no sessions."""
        auth_service.session_repo.find_by_user_id.return_value = []
        
        result = auth_service.invalidate_user_sessions("user123")
        
        assert result.success is True
        assert result.error_message is None
        assert result.invalidated_sessions == 0
    
    def test_invalidate_user_sessions_save_error(self, auth_service, valid_user_profile):
        """Test session invalidation with database error."""
        session = AuthSession(
            session_id="sess1",
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_user_id.return_value = [session]
        auth_service.session_repo.save.side_effect = Exception("Database error")
        
        result = auth_service.invalidate_user_sessions("user123")
        
        assert result.success is False
        assert result.error_message == "Failed to invalidate sessions"
        assert result.invalidated_sessions == 0
    
    def test_login_user_success(self, auth_service, valid_user_profile):
        """Test successful user login."""
        email = "user@example.com"
        password = "password123"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.verify_password.return_value = True
        
        result = auth_service.login_user(email, password, ip_address, user_agent)
        
        assert result.success is True
        assert result.user == valid_user_profile
        assert result.token is not None
        assert result.session is not None
        assert result.error_message is None
        
        # Verify JWT token was created
        decoded_token = jwt.decode(
            result.token,
            auth_service.jwt_secret,
            algorithms=[auth_service.jwt_algorithm]
        )
        assert decoded_token['user_id'] == valid_user_profile.user_id
        assert decoded_token['email'] == valid_user_profile.email
        
        # Verify session was created
        assert result.session.user_id == valid_user_profile.user_id
        assert result.session.is_active is True
        assert result.session.ip_address == ip_address
        assert result.session.user_agent == user_agent
    
    def test_login_user_invalid_credentials(self, auth_service, valid_user_profile):
        """Test login with invalid credentials."""
        email = "user@example.com"
        password = "wrongpassword"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.verify_password.return_value = False
        
        result = auth_service.login_user(email, password, ip_address, user_agent)
        
        assert result.success is False
        assert result.user is None
        assert result.token is None
        assert result.session is None
        assert result.error_message == "Invalid email or password"
    
    def test_login_user_invalid_user(self, auth_service):
        """Test login for nonexistent user."""
        email = "nonexistent@example.com"
        password = "password123"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        auth_service.user_model.get_user_by_email.return_value = None
        
        result = auth_service.login_user(email, password, ip_address, user_agent)
        
        assert result.success is False
        assert result.user is None
        assert result.token is None
        assert result.session is None
        assert result.error_message == "Invalid email or password"
    
    def test_login_user_inactive_user(self, auth_service):
        """Test login for inactive user."""
        inactive_user = UserProfile(
            user_id="user123",
            email="user@example.com",
            full_name="John Doe",
            role=UserRole.PATIENT,
            status=UserStatus.INACTIVE,
            created_at=datetime.now(),
            last_login=None
        )
        
        email = "user@example.com"
        password = "password123"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        auth_service.user_model.get_user_by_email.return_value = inactive_user
        auth_service.user_model.verify_password.return_value = True
        
        result = auth_service.login_user(email, password, ip_address, user_agent)
        
        assert result.success is False
        assert result.user is None
        assert result.token is None
        assert result.session is None
        assert result.error_message == "Account is not active"
    
    def test_login_user_max_sessions_exceeded(self, auth_service, valid_user_profile):
        """Test login when user exceeds max concurrent sessions."""
        # Mock existing sessions at max limit
        existing_sessions = []
        for i in range(5):  # MAX_CONCURRENT_SESSIONS = 5
            session = AuthSession(
                session_id=f"sess{i}",
                user_id="user123",
                created_at=datetime.now() - timedelta(hours=i),
                expires_at=datetime.now() + timedelta(hours=1),
                is_active=True
            )
            existing_sessions.append(session)
        
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.verify_password.return_value = True
        auth_service.session_repo.find_by_user_id.return_value = existing_sessions
        
        email = "user@example.com"
        password = "password123"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        result = auth_service.login_user(email, password, ip_address, user_agent)
        
        # Should still allow login but may invalidate oldest sessions
        assert result.success is True
        assert result.user == valid_user_profile
        assert result.token is not None
        assert result.session is not None
    
    def test_login_user_jwt_error(self, auth_service, valid_user_profile):
        """Test login with JWT token creation error."""
        email = "user@example.com"
        password = "password123"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        auth_service.user_model.get_user_by_email.return_value = valid_user_profile
        auth_service.user_model.verify_password.return_value = True
        
        # Mock JWT error
        with patch.object(auth_service, '_create_jwt_token') as mock_jwt:
            mock_jwt.side_effect = Exception("JWT error")
            
            result = auth_service.login_user(email, password, ip_address, user_agent)
            
            assert result.success is False
            assert result.error_message == "Authentication failed"
            assert result.user is None
            assert result.token is None
            assert result.session is None
    
    def test_logout_user_success(self, auth_service, valid_user_profile):
        """Test successful user logout."""
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.session_repo.save.side_effect = lambda session: True
        
        result = auth_service.logout_user(session_id)
        
        assert result.success is True
        assert result.error_message is None
        
        # Session should be marked inactive
        assert session.is_active is False
        
        # Session should be saved
        auth_service.session_repo.find_by_id.assert_called_once_with(session_id)
        auth_service.session_repo.save.assert_called_once()
    
    def test_logout_user_session_not_found(self, auth_service):
        """Test logout with nonexistent session."""
        session_id = "nonexistent_session"
        
        auth_service.session_repo.find_by_id.return_value = None
        
        result = auth_service.logout_user(session_id)
        
        assert result.success is False
        assert result.error_message == "Session not found"
    
    def test_logout_user_database_error(self, auth_service, valid_user_profile):
        """Test logout with database error."""
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.session_repo.save.side_effect = Exception("Database error")
        
        result = auth_service.logout_user(session_id)
        
        assert result.success is False
        assert result.error_message == "Logout failed"
    
    def test_refresh_token_success(self, auth_service, valid_user_profile):
        """Test successful token refresh."""
        old_token = jwt.encode(
            {
                'user_id': valid_user_profile.user_id,
                'email': valid_user_profile.email,
                'exp': datetime.utcnow() + timedelta(hours=23)  # About to expire
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id=valid_user_profile.user_id,
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.session_repo.save.side_effect = lambda session: True
        
        result = auth_service.refresh_token(old_token, session_id)
        
        assert result.success is True
        assert result.token is not None
        assert result.token != old_token  # Should be different token
        assert result.error_message is None
        
        # Verify new token
        decoded_token = jwt.decode(
            result.token,
            auth_service.jwt_secret,
            algorithms=[auth_service.jwt_algorithm]
        )
        assert decoded_token['user_id'] == valid_user_profile.user_id
        assert decoded_token['email'] == valid_user_profile.email
        
        # Verify session was updated
        assert session.is_active is True
        assert session.expires_at > session.created_at
    
    def test_refresh_token_invalid_token(self, auth_service):
        """Test refresh token with invalid token."""
        invalid_token = "invalid.jwt.token"
        session_id = "sess123"
        
        result = auth_service.refresh_token(invalid_token, session_id)
        
        assert result.success is False
        assert result.token is None
        assert result.error_message == "Invalid token"
    
    def test_refresh_token_expired_token(self, auth_service):
        """Test refresh token with expired token."""
        expired_token = jwt.encode(
            {
                'user_id': 'user123',
                'email': 'user@example.com',
                'exp': datetime.utcnow() - timedelta(hours=1)  # Expired
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        session_id = "sess123"
        
        result = auth_service.refresh_token(expired_token, session_id)
        
        assert result.success is False
        assert result.token is None
        assert result.error_message == "Token has expired"
    
    def test_refresh_token_session_not_found(self, auth_service):
        """Test refresh token with nonexistent session."""
        valid_token = jwt.encode(
            {
                'user_id': 'user123',
                'email': 'user@example.com',
                'exp': datetime.utcnow() + timedelta(hours=23)
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        session_id = "nonexistent_session"
        auth_service.session_repo.find_by_id.return_value = None
        
        result = auth_service.refresh_token(valid_token, session_id)
        
        assert result.success is False
        assert result.token is None
        assert result.error_message == "Session not found"
    
    def test_refresh_token_user_not_found(self, auth_service):
        """Test refresh token with user not found."""
        valid_token = jwt.encode(
            {
                'user_id': 'nonexistent_user',
                'email': 'user@example.com',
                'exp': datetime.utcnow() + timedelta(hours=23)
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="nonexistent_user",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.user_model.get_user_by_id.return_value = None
        
        result = auth_service.refresh_token(valid_token, session_id)
        
        assert result.success is False
        assert result.token is None
        assert result.error_message == "User not found"
    
    def test_validate_session_access_valid(self, auth_service, valid_user_profile):
        """Test valid session access validation."""
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        
        result = auth_service.validate_session_access(session_id)
        
        assert result.success is True
        assert result.session is not None
        assert result.user is not None
        assert result.error_message is None
        assert result.session.user_id == "user123"
        assert result.user.user_id == "user123"
    
    def test_validate_session_access_session_expired(self, auth_service, valid_user_profile):
        """Test session validation with expired session."""
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),  # Expired
            is_active=True
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        
        result = auth_service.validate_session_access(session_id)
        
        assert result.success is False
        assert result.session is not None
        assert result.user is not None
        assert result.error_message == "Session has expired"
        assert result.session.session_id == "sess123"
    
    def test_validate_session_access_session_inactive(self, auth_service, valid_user_profile):
        """Test session validation with inactive session."""
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=False  # Inactive
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.user_model.get_user_by_id.return_value = valid_user_profile
        
        result = auth_service.validate_session_access(session_id)
        
        assert result.success is False
        assert result.session is not None
        assert result.user is not None
        assert result.error_message == "Session is not active"
    
    def test_validate_session_access_session_not_found(self, auth_service):
        """Test session validation with nonexistent session."""
        session_id = "nonexistent_session"
        
        auth_service.session_repo.find_by_id.return_value = None
        
        result = auth_service.validate_session_access(session_id)
        
        assert result.success is False
        assert result.session is None
        assert result.user is None
        assert result.error_message == "Session not found"
    
    def test_validate_session_access_user_not_found(self, auth_service):
        """Test session validation with user not found."""
        session_id = "sess123"
        session = AuthSession(
            session_id=session_id,
            user_id="nonexistent_user",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        auth_service.session_repo.find_by_id.return_value = session
        auth_service.user_model.get_user_by_id.return_value = None
        
        result = auth_service.validate_session_access(session_id)
        
        assert result.success is False
        assert result.session is not None
        assert result.user is None
        assert result.error_message == "User not found"
    
    def test_validate_token_valid(self, auth_service, valid_user_profile):
        """Test valid token validation."""
        valid_token = jwt.encode(
            {
                'user_id': valid_user_profile.user_id,
                'email': valid_user_profile.email,
                'exp': datetime.utcnow() + timedelta(hours=23)
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        result = auth_service.validate_token(valid_token)
        
        assert result.success is True
        assert result.user is not None
        assert result.error_message is None
        assert result.user.user_id == valid_user_profile.user_id
        assert result.user.email == valid_user_profile.email
    
    def test_validate_token_invalid(self, auth_service):
        """Test invalid token validation."""
        invalid_token = "invalid.jwt.token"
        
        result = auth_service.validate_token(invalid_token)
        
        assert result.success is False
        assert result.user is None
        assert result.error_message == "Invalid token"
    
    def test_validate_token_expired(self, auth_service):
        """Test expired token validation."""
        expired_token = jwt.encode(
            {
                'user_id': 'user123',
                'email': 'user@example.com',
                'exp': datetime.utcnow() - timedelta(hours=1)  # Expired
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        result = auth_service.validate_token(expired_token)
        
        assert result.success is False
        assert result.user is None
        assert result.error_message == "Token has expired"
    
    def test_validate_token_user_not_found(self, auth_service):
        """Test token validation with user not found."""
        token = jwt.encode(
            {
                'user_id': 'nonexistent_user',
                'email': 'user@example.com',
                'exp': datetime.utcnow() + timedelta(hours=23)
            },
            auth_service.jwt_secret,
            algorithm=auth_service.jwt_algorithm
        )
        
        auth_service.user_model.get_user_by_id.return_value = None
        
        result = auth_service.validate_token(token)
        
        assert result.success is False
        assert result.user is None
        assert result.error_message == "User not found"
    
    def test_background_cleanup(self, auth_service):
        """Test background cleanup thread functionality."""
        # This test verifies that the cleanup thread starts and runs
        # without errors
        cleanup_thread = auth_service.cleanup_thread
        assert cleanup_thread is not None
        assert cleanup_thread.is_alive()
        assert cleanup_thread.daemon
        
        # Give cleanup thread a moment to start
        time.sleep(0.1)
        
        # Thread should still be running
        assert cleanup_thread.is_alive()
    
    def test_jwt_configuration(self, auth_service):
        """Test JWT configuration."""
        assert hasattr(auth_service, 'jwt_secret')
        assert hasattr(auth_service, 'jwt_algorithm')
        assert hasattr(auth_service, 'jwt_expiration_hours')
        
        assert auth_service.jwt_secret is not None
        assert auth_service.jwt_algorithm == "HS256"
        assert isinstance(auth_service.jwt_expiration_hours, int)
        assert auth_service.jwt_expiration_hours > 0
    
    def test_session_configuration(self, auth_service):
        """Test session configuration."""
        assert hasattr(auth_service, 'session_timeout_minutes')
        assert hasattr(auth_service, 'max_concurrent_sessions')
        
        assert isinstance(auth_service.session_timeout_minutes, int)
        assert isinstance(auth_service.max_concurrent_sessions, int)
        assert auth_service.session_timeout_minutes > 0
        assert auth_service.max_concurrent_sessions > 0
    
    def test_auth_session_creation(self):
        """Test AuthSession dataclass functionality."""
        session_id = "test_session"
        user_id = "test_user"
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=24)
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"
        
        session = AuthSession(
            session_id=session_id,
            user_id=user_id,
            created_at=created_at,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )
        
        # Test properties
        assert session.session_id == session_id
        assert session.user_id == user_id
        assert session.created_at == created_at
        assert session.expires_at == expires_at
        assert session.ip_address == ip_address
        assert session.user_agent == user_agent
        assert session.is_active is True
        
        # Test methods
        assert session.is_expired() is False
        assert session.to_dict() is not None
        assert isinstance(session.to_dict(), dict)
        
        # Test to_dict serialization
        session_dict = session.to_dict()
        assert session_dict['session_id'] == session_id
        assert session_dict['user_id'] == user_id
        assert 'created_at' in session_dict
        assert 'expires_at' in session_dict
    
    def test_auth_session_expired(self):
        """Test AuthSession expiration logic."""
        # Non-expired session
        future_session = AuthSession(
            session_id="future_session",
            user_id="test_user",
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        assert future_session.is_expired() is False
        
        # Expired session
        expired_session = AuthSession(
            session_id="expired_session",
            user_id="test_user",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
            is_active=False
        )
        assert expired_session.is_expired() is True
    
    def test_auth_result_creation(self):
        """Test AuthResult dataclass functionality."""
        user_profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            last_login=None
        )
        
        token = "test.jwt.token"
        session = AuthSession(
            session_id="sess123",
            user_id="test_user",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        result = AuthResult(
            success=True,
            user=user_profile,
            token=token,
            session=session,
            error_message=None
        )
        
        # Test properties
        assert result.success is True
        assert result.user == user_profile
        assert result.token == token
        assert result.session == session
        assert result.error_message is None
        
        # Test failure result
        failure_result = AuthResult(
            success=False,
            user=None,
            token=None,
            session=None,
            error_message="Authentication failed"
        )
        
        assert failure_result.success is False
        assert failure_result.user is None
        assert failure_result.token is None
        assert failure_result.session is None
        assert failure_result.error_message == "Authentication failed"
    
    def test_service_initialization(self, auth_service):
        """Test service initialization."""
        assert hasattr(auth_service, 'user_model')
        assert hasattr(auth_service, 'session_repo')
        assert hasattr(auth_service, 'cleanup_thread')
        assert hasattr(auth_service, 'jwt_secret')
        assert hasattr(auth_service, 'jwt_algorithm')
        assert hasattr(auth_service, 'jwt_expiration_hours')
        assert hasattr(auth_service, 'session_timeout_minutes')
        assert hasattr(auth_service, 'max_concurrent_sessions')
        
        # Mock dependencies should be initialized
        assert auth_service.user_model is not None
        assert auth_service.session_repo is not None
        
        # Cleanup thread should be running
        cleanup_thread = auth_service.cleanup_thread
        assert cleanup_thread is not None
        assert cleanup_thread.daemon is True
    
    def test_service_configuration_from_env(self, auth_service):
        """Test that service configuration is loaded from environment."""
        # Test JWT configuration
        assert auth_service.jwt_secret is not None
        assert auth_service.jwt_algorithm == "HS256"
        assert isinstance(auth_service.jwt_expiration_hours, int)
        assert auth_service.jwt_expiration_hours > 0
        
        # Test session configuration
        assert isinstance(auth_service.session_timeout_minutes, int)
        assert isinstance(auth_service.max_concurrent_sessions, int)
        assert auth_service.session_timeout_minutes > 0
        assert auth_service.max_concurrent_sessions > 0
    
    def test_mock_repository_fallback(self):
        """Test that mock repository works when database is not available."""
        # This test verifies that the fallback repository is created
        # when database models are not available
        from auth.auth_service import SessionRepository
        
        repo = SessionRepository()
        assert repo is not None
        assert hasattr(repo, 'sessions')
        assert hasattr(repo, 'save')
        assert hasattr(repo, 'find_by_id')
        assert hasattr(repo, 'find_by_user_id')
        
        # Test basic operations
        test_session = AuthSession(
            session_id="test",
            user_id="test",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            is_active=True
        )
        
        repo.save(test_session)
        assert test_session.session_id in repo.sessions
        
        retrieved = repo.find_by_id("test")
        assert retrieved is not None
        assert retrieved.session_id == "test"
        
        user_sessions = repo.find_by_user_id("test")
        assert len(user_sessions) == 1
        assert user_sessions[0].user_id == "test"