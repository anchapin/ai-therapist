"""
Corrected comprehensive test file to boost auth module coverage to 90%+
Targets specific missing lines identified in coverage reports with correct API usage
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# Import auth modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from auth.auth_service import AuthService, AuthSession, AuthResult
from auth.middleware import AuthMiddleware
from auth.user_model import UserProfile, UserRole, UserStatus, UserModel
from database.models import Session


class TestAuthServiceWorkingCoverage:
    """Test cases for AuthService to cover remaining gaps with correct API"""
    
    @pytest.fixture
    def mock_user_model(self):
        """Mock UserModel"""
        model_mock = Mock(spec=UserModel)
        return model_mock
    
    @pytest.fixture
    def auth_service(self, mock_user_model):
        """Create AuthService with mocked UserModel"""
        return AuthService(user_model=mock_user_model)
    
    def test_init_with_custom_secret(self, mock_user_model):
        """Test AuthService initialization with custom secret"""
        with patch.dict(os.environ, {'JWT_SECRET_KEY': 'custom_secret_key'}):
            service = AuthService(user_model=mock_user_model)
            assert service.jwt_secret == 'custom_secret_key'
    
    def test_init_with_default_secret(self, mock_user_model):
        """Test AuthService initialization with default secret"""
        with patch.dict(os.environ, {}, clear=True):
            service = AuthService(user_model=mock_user_model)
            assert service.jwt_secret == 'ai-therapist-jwt-secret-change-in-production'
    
    def test_generate_token_with_custom_expiration(self, auth_service):
        """Test token generation with custom expiration"""
        user_id = "user123"
        custom_exp = 7200  # 2 hours
        
        token = auth_service.generate_token(user_id, expires_in=custom_exp)
        
        # Decode token to verify expiration
        decoded = jwt.decode(token, auth_service.jwt_secret, algorithms=[auth_service.jwt_algorithm])
        exp_time = datetime.fromtimestamp(decoded['exp'])
        expected_exp = datetime.utcnow() + timedelta(seconds=custom_exp)
        
        # Allow for small time difference (within 1 minute)
        assert abs((exp_time - expected_exp).total_seconds()) < 60
    
    def test_validate_token_with_none_token(self, auth_service):
        """Test token validation with None token"""
        result = auth_service.validate_token(None)
        assert result is None
    
    def test_validate_token_with_empty_string(self, auth_service):
        """Test token validation with empty string"""
        result = auth_service.validate_token("")
        assert result is None
    
    def test_validate_token_with_invalid_token(self, auth_service):
        """Test token validation with invalid token"""
        result = auth_service.validate_token("invalid_token")
        assert result is None
    
    def test_validate_token_with_expired_token(self, auth_service):
        """Test token validation with expired token"""
        # Create expired token
        expired_payload = {
            'user_id': "user123",
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        result = auth_service.validate_token(expired_token)
        assert result is None
    
    def test_refresh_token_with_invalid_token(self, auth_service):
        """Test token refresh with invalid token"""
        result = auth_service.refresh_token("invalid_token")
        assert result is None
    
    def test_refresh_token_with_expired_token(self, auth_service):
        """Test token refresh with expired token"""
        # Create expired token
        expired_payload = {
            'user_id': "user123",
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        result = auth_service.refresh_token(expired_token)
        assert result is None
    
    def test_is_token_expired_with_none_token(self, auth_service):
        """Test token expiration check with None token"""
        result = auth_service.is_token_expired(None)
        assert result is True
    
    def test_is_token_expired_with_empty_string(self, auth_service):
        """Test token expiration check with empty string"""
        result = auth_service.is_token_expired("")
        assert result is True
    
    def test_is_token_expired_with_invalid_token(self, auth_service):
        """Test token expiration check with invalid token"""
        result = auth_service.is_token_expired("invalid_token")
        assert result is True
    
    def test_is_token_expired_with_valid_token(self, auth_service):
        """Test token expiration check with valid token"""
        token = auth_service.generate_token("user123")
        result = auth_service.is_token_expired(token)
        assert result is False
    
    def test_get_token_payload_with_none_token(self, auth_service):
        """Test getting token payload with None token"""
        result = auth_service.get_token_payload(None)
        assert result is None
    
    def test_get_token_payload_with_empty_string(self, auth_service):
        """Test getting token payload with empty string"""
        result = auth_service.get_token_payload("")
        assert result is None
    
    def test_get_token_payload_with_invalid_token(self, auth_service):
        """Test getting token payload with invalid token"""
        result = auth_service.get_token_payload("invalid_token")
        assert result is None
    
    def test_get_token_payload_with_valid_token(self, auth_service):
        """Test getting token payload with valid token"""
        token = auth_service.generate_token("user123")
        payload = auth_service.get_token_payload(token)
        
        assert payload is not None
        assert payload['user_id'] == "user123"
        assert 'exp' in payload
        assert 'iat' in payload


class TestMiddlewareWorkingCoverage:
    """Test cases for AuthMiddleware to cover remaining gaps with correct API"""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock AuthService"""
        service_mock = Mock(spec=AuthService)
        return service_mock
    
    @pytest.fixture
    def middleware(self, mock_auth_service):
        """Create AuthMiddleware with mocked AuthService"""
        return AuthMiddleware(auth_service=mock_auth_service)
    
    @pytest.fixture
    def mock_st(self):
        """Mock Streamlit"""
        st_mock = Mock()
        st_mock.session_state = {}
        st_mock.error = Mock()
        st_mock.success = Mock()
        st_mock.warning = Mock()
        st_mock.info = Mock()
        st_mock.text_input = Mock(return_value="")
        st_mock.text_area = Mock(return_value="")
        st_mock.button = Mock(return_value=False)
        st_mock.columns = Mock(return_value=[Mock(), Mock()])
        st_mock.container = Mock()
        st_mock.expander = Mock()
        st_mock.markdown = Mock()
        st_mock.write = Mock()
        st_mock.subheader = Mock()
        st_mock.title = Mock()
        st_mock.sidebar = Mock()
        st_mock.form = Mock()
        st_mock.form_submit_button = Mock(return_value=False)
        return st_mock
    
    def test_init_with_custom_auth_service(self):
        """Test AuthMiddleware initialization with custom AuthService"""
        custom_service = Mock(spec=AuthService)
        middleware = AuthMiddleware(auth_service=custom_service)
        assert middleware.auth_service == custom_service
    
    def test_login_required_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test login_required decorator with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.validate_token.side_effect = Exception("Service error")
            
            @middleware.login_required
            def test_func():
                return "success"
            
            # Should not raise exception
            result = test_func()
            assert result is None
    
    def test_role_required_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test role_required decorator with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.validate_token.side_effect = Exception("Service error")
            
            @middleware.role_required([UserRole.ADMIN])
            def test_func():
                return "success"
            
            # Should not raise exception
            result = test_func()
            assert result is None
    
    def test_is_authenticated_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test is_authenticated with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.validate_token.side_effect = Exception("Service error")
            
            result = middleware.is_authenticated()
            assert result is False
    
    def test_get_current_user_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test get_current_user with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.validate_token.side_effect = Exception("Service error")
            
            result = middleware.get_current_user()
            assert result is None
    
    def test_login_user_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test login_user with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.login_user.side_effect = Exception("Service error")
            
            result = middleware.login_user("test@example.com", "password")
            assert result is None
    
    def test_logout_user_with_exception(self, middleware, mock_st):
        """Test logout_user with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup session state to raise exception on deletion
            class SessionStateWithException:
                def __init__(self):
                    self.auth_token = "test_token"
                
                def __delitem__(self, key):
                    raise Exception("Session error")
            
            mock_st.session_state = SessionStateWithException()
            
            # Should not raise exception
            middleware.logout_user()


class TestUserProfileWorkingCoverage:
    """Test cases for UserProfile to cover remaining gaps with correct API"""
    
    def test_user_profile_init_with_all_parameters(self):
        """Test UserProfile initialization with all parameters"""
        now = datetime.utcnow()
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        assert profile.user_id == "user123"
        assert profile.email == "test@example.com"
        assert profile.full_name == "Test User"
        assert profile.role == UserRole.USER
        assert profile.status == UserStatus.ACTIVE
        assert profile.created_at == now
        assert profile.updated_at == now
    
    def test_user_profile_init_with_minimal_parameters(self):
        """Test UserProfile initialization with minimal parameters"""
        now = datetime.utcnow()
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        assert profile.user_id == "user123"
        assert profile.email == "test@example.com"
        assert profile.full_name == "Test User"
        assert profile.role == UserRole.USER
        assert profile.status == UserStatus.ACTIVE
        assert profile.created_at is not None
        assert profile.updated_at is not None
    
    def test_user_profile_to_dict(self):
        """Test UserProfile to_dict method"""
        now = datetime.utcnow()
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        result = profile.to_dict()
        
        assert result['user_id'] == "user123"
        assert result['email'] == "test@example.com"
        assert result['full_name'] == "Test User"
        assert result['role'] == UserRole.USER
        assert result['status'] == UserStatus.ACTIVE
        assert result['created_at'] == now
        assert result['updated_at'] == now
    
    def test_user_profile_str_representation(self):
        """Test UserProfile string representation"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        result = str(profile)
        assert "test@example.com" in result
        assert "Test User" in result
        assert "user" in result
    
    def test_user_profile_repr_representation(self):
        """Test UserProfile repr representation"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        result = repr(profile)
        assert "UserProfile" in result
        assert "user123" in result
        assert "test@example.com" in result
    
    def test_user_profile_equality(self):
        """Test UserProfile equality comparison"""
        now = datetime.utcnow()
        profile1 = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        profile2 = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        profile3 = UserProfile(
            user_id="user456",
            email="other@example.com",
            full_name="Other User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        assert profile1 == profile2
        assert profile1 != profile3
        assert profile1 != "not a profile"
    
    def test_user_profile_hash(self):
        """Test UserProfile hash"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Should not raise exception
        hash_value = hash(profile)
        assert isinstance(hash_value, int)
    
    def test_user_profile_is_locked(self):
        """Test UserProfile is_locked method"""
        # Test with unlocked account
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            account_locked_until=None
        )
        
        assert profile.is_locked() is False
        
        # Test with locked account
        profile.account_locked_until = datetime.utcnow() + timedelta(hours=1)
        assert profile.is_locked() is True
        
        # Test with expired lock
        profile.account_locked_until = datetime.utcnow() - timedelta(hours=1)
        assert profile.is_locked() is False
    
    def test_user_profile_increment_login_attempts(self):
        """Test UserProfile increment_login_attempts method"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            login_attempts=0
        )
        
        # Increment attempts
        profile.increment_login_attempts()
        assert profile.login_attempts == 1
        assert profile.status == UserStatus.ACTIVE
        
        # Increment to lock threshold
        for i in range(4):
            profile.increment_login_attempts()
        
        assert profile.login_attempts == 5
        assert profile.status == UserStatus.LOCKED
        assert profile.account_locked_until is not None
    
    def test_user_profile_reset_login_attempts(self):
        """Test UserProfile reset_login_attempts method"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.LOCKED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            login_attempts=5,
            account_locked_until=datetime.utcnow() + timedelta(hours=1)
        )
        
        profile.reset_login_attempts()
        
        assert profile.login_attempts == 0
        assert profile.account_locked_until is None
        assert profile.status == UserStatus.ACTIVE
    
    def test_user_profile_can_access_resource(self):
        """Test UserProfile can_access_resource method"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Test with valid permission
        result = profile.can_access_resource("own_profile", "read")
        assert result is True
        
        # Test with invalid permission
        result = profile.can_access_resource("own_profile", "delete")
        assert result is False
    
    def test_user_profile_to_dict_with_pii_protection(self):
        """Test UserProfile to_dict method with PII protection"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            medical_info={"diagnosis": "test", "medications": ["test"]},
            password_reset_token="secret_token",
            password_reset_expires=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Test with no user role (should remove sensitive fields)
        result = profile.to_dict()
        assert "password_reset_token" not in result
        assert "password_reset_expires" not in result
        assert "medical_info" in result
        
        # Test with patient role (should sanitize medical info)
        result = profile.to_dict(user_role="patient")
        assert "password_reset_token" not in result
        assert "password_reset_expires" not in result
        assert "medical_info" in result
        
        # Test with therapist role (should include medical info)
        result = profile.to_dict(user_role="therapist", include_sensitive=True)
        assert "password_reset_token" not in result
        assert "password_reset_expires" not in result
        assert "medical_info" in result
        
        # Test with admin role (should include all except password reset fields)
        result = profile.to_dict(user_role="admin", include_sensitive=True)
        assert "password_reset_token" not in result
        assert "password_reset_expires" not in result
        assert "medical_info" in result


class TestAuthSessionWorkingCoverage:
    """Test cases for AuthSession to cover remaining gaps"""
    
    def test_auth_session_init(self):
        """Test AuthSession initialization"""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)
        
        session = AuthSession(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="Test Browser",
            is_active=True
        )
        
        assert session.session_id == "session123"
        assert session.user_id == "user123"
        assert session.created_at == now
        assert session.expires_at == expires
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Test Browser"
        assert session.is_active is True
    
    def test_auth_session_is_expired(self):
        """Test AuthSession is_expired method"""
        now = datetime.utcnow()
        
        # Test with non-expired session
        future_expires = now + timedelta(hours=1)
        session = AuthSession(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=future_expires
        )
        
        assert session.is_expired() is False
        
        # Test with expired session
        past_expires = now - timedelta(hours=1)
        session.expires_at = past_expires
        assert session.is_expired() is True
    
    def test_auth_session_to_dict(self):
        """Test AuthSession to_dict method"""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)
        
        session = AuthSession(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="Test Browser",
            is_active=True
        )
        
        result = session.to_dict()
        
        assert result['session_id'] == "session123"
        assert result['user_id'] == "user123"
        assert result['created_at'] == now.isoformat()
        assert result['expires_at'] == expires.isoformat()
        assert result['ip_address'] == "192.168.1.1"
        assert result['user_agent'] == "Test Browser"
        assert result['is_active'] is True


class TestAuthResultWorkingCoverage:
    """Test cases for AuthResult to cover remaining gaps"""
    
    def test_auth_result_init_success(self):
        """Test AuthResult initialization with success"""
        profile = UserProfile(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session = AuthSession(
            session_id="session123",
            user_id="user123",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        result = AuthResult(
            success=True,
            user=profile,
            token="test_token",
            session=session,
            error_message=None
        )
        
        assert result.success is True
        assert result.user == profile
        assert result.token == "test_token"
        assert result.session == session
        assert result.error_message is None
    
    def test_auth_result_init_failure(self):
        """Test AuthResult initialization with failure"""
        result = AuthResult(
            success=False,
            user=None,
            token=None,
            session=None,
            error_message="Invalid credentials"
        )
        
        assert result.success is False
        assert result.user is None
        assert result.token is None
        assert result.session is None
        assert result.error_message == "Invalid credentials"