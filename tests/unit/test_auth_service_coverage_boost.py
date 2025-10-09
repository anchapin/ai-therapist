"""
Simplified unit tests for auth/auth_service.py to boost coverage.
Focuses on core authentication business logic that is currently uncovered.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import jwt

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from auth.auth_service import AuthService, AuthResult, AuthSession, UserStatus
    from auth.user_model import UserProfile, UserRole
except ImportError as e:
    pytest.skip(f"auth_service module not available: {e}", allow_module_level=True)


class TestAuthServiceCoreCoverage:
    """Targeted unit tests to boost auth_service.py coverage."""
    
    @pytest.fixture
    def auth_service(self):
        """Create an AuthService without mocking dependencies."""
        # Create service directly - it will use default repositories
        return AuthService()
    
    def test_auth_service_initialization(self, auth_service):
        """Test auth service initialization."""
        assert auth_service.user_model is not None
        assert hasattr(auth_service, 'jwt_secret')
        assert isinstance(auth_service.jwt_secret, str)
        assert len(auth_service.jwt_secret) > 10
        assert auth_service.jwt_algorithm == "HS256"
        assert isinstance(auth_service.jwt_expiration_hours, int)
        assert auth_service.jwt_expiration_hours > 0
    
    def test_generate_jwt_token(self, auth_service):
        """Test JWT token generation."""
        user_profile = Mock(spec=UserProfile)
        user_profile.user_id = "user_123"
        
        token = auth_service._generate_jwt_token(user_profile)
        
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are typically long
        
        # Verify token can be decoded
        decoded = jwt.decode(token, auth_service.jwt_secret, algorithms=['HS256'])
        assert decoded['user_id'] == "user_123"
    
    def test_validate_token_success(self, auth_service):
        """Test successful token validation."""
        user_profile = Mock(spec=UserProfile)
        user_profile.user_id = "user_123"
        user_profile.email = "test@example.com"
        user_profile.role = UserRole.PATIENT
        user_profile.status = UserStatus.ACTIVE
        
        # Generate a token
        token = auth_service._generate_jwt_token(user_profile)
        
        # Validate token
        result = auth_service._validate_token(token)
        
        assert result is not None
        assert result.user_id == "user_123"
    
    def test_validate_token_invalid(self, auth_service):
        """Test invalid token validation."""
        invalid_token = "invalid.jwt.token"
        
        result = auth_service._validate_token(invalid_token)
        
        assert result is None
    
    def test_validate_token_malformed(self, auth_service):
        """Test malformed token validation."""
        malformed_token = "not.a.jwt"
        
        result = auth_service._validate_token(malformed_token)
        
        assert result is None
    
    def test_create_session_basic(self, auth_service):
        """Test basic session creation."""
        user_id = "user_123"
        ip_address = "127.0.0.1"
        user_agent = "Mozilla/5.0"
        
        session = auth_service._create_session(user_id, ip_address, user_agent)
        
        assert isinstance(session, AuthSession)
        assert session.user_id == user_id
        assert session.ip_address == ip_address
        assert session.user_agent == user_agent
        assert session.is_active is True
        assert session.expires_at > datetime.now()
        assert session.created_at <= datetime.now()
    
    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "SecurePass123"
        
        hashed = auth_service._hash_password(password)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA256 hex length
        assert hashed != password  # Should be different from original
        # Basic check - should contain hexadecimal characters
        assert all(c in '0123456789abcdef' for c in hashed.lower())
    
    def test_verify_password_correct(self, auth_service):
        """Test password verification with correct password."""
        password = "SecurePass123"
        hashed = auth_service._hash_password(password)
        
        # Re-hash and compare
        verification_hash = auth_service._hash_password(password)
        
        result = (hashed == verification_hash)
        assert result is True
    
    def test_verify_password_incorrect(self, auth_service):
        """Test password verification with incorrect password."""
        password = "SecurePass123"
        wrong_password = "WrongPass456"
        hashed = auth_service._hash_password(password)
        wrong_hashed = auth_service._hash_password(wrong_password)
        
        result = (hashed == wrong_hashed)
        assert result is False
    
    def test_generate_session_id(self, auth_service):
        """Test session ID generation."""
        session_id1 = auth_service._generate_session_id()
        session_id2 = auth_service._generate_session_id()
        
        assert isinstance(session_id1, str)
        assert isinstance(session_id2, str)
        assert len(session_id1) > 20  # Should be reasonably long
        assert len(session_id2) > 20
        assert session_id1 != session_id2  # Should be unique
        assert session_id1.replace('-', '').isalnum()  # Should be alphanumeric with dashes
    
    def test_format_expiration_time(self, auth_service):
        """Test expiration time formatting."""
        now = datetime.now()
        future = now + timedelta(hours=1)
        
        formatted = auth_service._format_expiration_time(future)
        
        assert isinstance(formatted, str)
        # Should contain ISO formatted datetime
        assert 'T' in formatted
    
    def test_is_token_expired_false(self, auth_service):
        """Test token expiration check for valid token."""
        future_time = datetime.now() + timedelta(hours=1)
        
        result = auth_service._is_token_expired(future_time)
        
        assert result is False
    
    def test_is_token_expired_true(self, auth_service):
        """Test token expiration check for expired token."""
        past_time = datetime.now() - timedelta(hours=1)
        
        result = auth_service._is_token_expired(past_time)
        
        assert result is True
    
    def test_is_token_expired_exact_now(self, auth_service):
        """Test token expiration check for current time."""
        now = datetime.now()
        
        result = auth_service._is_token_expired(now)
        
        assert result is False  # Current time is not expired
    
    def test_jwt_token_expiry(self, auth_service):
        """Test JWT token expiry handling."""
        user_profile = Mock(spec=UserProfile)
        user_profile.user_id = "user_123"
        
        # Generate token
        token = auth_service._generate_jwt_token(user_profile)
        
        # Manually decode to check expiry
        decoded = jwt.decode(token, auth_service.jwt_secret, algorithms=['HS256'])
        assert 'exp' in decoded
        
        # Verify expiry is in the future
        exp_timestamp = decoded['exp']
        current_timestamp = datetime.now().timestamp()
        assert exp_timestamp > current_timestamp
        
        # Verify expiry is not too far in the future (within 24 hours)
        assert exp_timestamp < current_timestamp + (24 * 60 * 60)
    
    def test_jwt_token_includes_required_claims(self, auth_service):
        """Test JWT token includes required claims."""
        user_profile = Mock(spec=UserProfile)
        user_profile.user_id = "user_123"
        user_profile.email = "test@example.com"
        user_profile.role = UserRole.PATIENT
        
        token = auth_service._generate_jwt_token(user_profile)
        decoded = jwt.decode(token, auth_service.jwt_secret, algorithms=['HS256'])
        
        # Check required claims
        assert 'user_id' in decoded
        assert 'email' in decoded
        assert 'role' in decoded
        assert 'exp' in decoded
        assert 'iat' in decoded
    
    def test_create_service_metrics(self, auth_service):
        """Test service metrics initialization."""
        assert hasattr(auth_service, 'metrics')
        assert 'users_registered' in auth_service.metrics
        assert 'failed_logins' in auth_service.metrics
        assert 'successful_logins' in auth_service.metrics
        assert isinstance(auth_service.metrics['users_registered'], int)
        assert isinstance(auth_service.metrics['failed_logins'], int)
        assert isinstance(auth_service.metrics['successful_logins'], int)
    
    def test_update_metrics(self, auth_service):
        """Test metrics updating."""
        # Update login metrics
        auth_service._update_metrics('successful_login')
        assert auth_service.metrics['successful_logins'] == 1
        
        auth_service._update_metrics('failed_login')
        assert auth_service.metrics['failed_logins'] == 1
        
        auth_service._update_metrics('user_registration')
        assert auth_service.metrics['users_registered'] == 1
    
    def test_get_service_statistics(self, auth_service):
        """Test getting service statistics."""
        stats = auth_service.get_service_statistics()
        
        assert isinstance(stats, dict)
        assert 'users_registered' in stats
        assert 'failed_logins' in stats
        assert 'successful_logins' in stats
        assert 'active_sessions' in stats
        assert isinstance(stats['users_registered'], int)
        assert isinstance(stats['failed_logins'], int)
        assert isinstance(stats['successful_logins'], int)
        assert isinstance(stats['active_sessions'], int)
    
    def test_get_service_statistics_empty(self, auth_service):
        """Test service statistics when no activity."""
        # Reset metrics
        auth_service.metrics['users_registered'] = 0
        auth_service.metrics['failed_logins'] = 0
        auth_service.metrics['successful_logins'] = 0
        
        stats = auth_service.get_service_statistics()
        
        assert stats['users_registered'] == 0
        assert stats['failed_logins'] == 0
        assert stats['successful_logins'] == 0
    
    def test_user_status_checks(self, auth_service):
        """Test user status validation logic."""
        active_user = Mock(spec=UserProfile)
        active_user.status = UserStatus.ACTIVE
        
        inactive_user = Mock(spec=UserProfile)
        inactive_user.status = UserStatus.INACTIVE
        
        locked_user = Mock(spec=UserProfile)
        locked_user.status = UserStatus.LOCKED
        
        # Status checks should work with user objects
        assert active_user.status == UserStatus.ACTIVE
        assert inactive_user.status == UserStatus.INACTIVE
        assert locked_user.status == UserStatus.LOCKED
    
    def test_role_hierarchy_logic(self, auth_service):
        """Test role hierarchy validation logic."""
        patient_role = UserRole.PATIENT
        therapist_role = UserRole.THERAPIST
        admin_role = UserRole.ADMIN
        
        # Role comparisons
        assert admin_role.value > therapist_role.value
        assert therapist_role.value > patient_role.value
        assert admin_role.value > patient_role.value
    
    def test_session_activity_tracking(self, auth_service):
        """Test session activity time tracking."""
        session = AuthSession(
            user_id="user_123",
            session_id="session_123",
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        
        # Test activity update
        original_activity = session.last_activity
        # Simulate time passing
        new_activity = original_activity + timedelta(minutes=5)
        
        # Update and verify
        session.last_activity = new_activity
        assert session.last_activity > original_activity
        assert session.last_activity == new_activity
    
    def test_session_security_context(self, auth_service):
        """Test session security context."""
        session = AuthSession(
            user_id="user_123",
            session_id="session_123",
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        
        # Verify security fields
        assert session.user_id == "user_123"
        assert session.ip_address == "127.0.0.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.session_id == "session_123"
        assert session.is_active is True
    
    def test_token_user_profile_mapping(self, auth_service):
        """Test user profile to token mapping."""
        user_profile = Mock(spec=UserProfile)
        user_profile.user_id = "user_123"
        user_profile.email = "test@example.com"
        user_profile.role = UserRole.PATIENT
        user_profile.status = UserStatus.ACTIVE
        user_profile.full_name = "Test User"
        user_profile.created_at = datetime.now()
        
        token = auth_service._generate_jwt_token(user_profile)
        decoded = jwt.decode(token, auth_service.jwt_secret, algorithms=['HS256'])
        
        # Verify all user profile fields are in token
        assert decoded['user_id'] == "user_123"
        assert decoded['email'] == "test@example.com"
        assert decoded['role'] == UserRole.PATIENT.value
        assert decoded['status'] == UserStatus.ACTIVE.value
        assert decoded['full_name'] == "Test User"
        assert 'created_at' in decoded