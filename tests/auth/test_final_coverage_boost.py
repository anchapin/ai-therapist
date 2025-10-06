"""
Final comprehensive test file to boost auth module coverage to 90%+
Targets specific missing lines identified in coverage reports
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

from auth.auth_service import AuthService
from auth.middleware import AuthMiddleware
from auth.user_model import UserProfile
from database.models import Session


class TestAuthServiceFinalCoverage:
    """Test cases for AuthService to cover remaining gaps"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection"""
        db_mock = Mock()
        db_mock.cursor.return_value.__enter__ = Mock()
        db_mock.cursor.return_value.__exit__ = Mock()
        return db_mock
    
    @pytest.fixture
    def auth_service(self, mock_db):
        """Create AuthService with mocked database"""
        with patch('database.db_manager.get_db', return_value=mock_db):
            return AuthService()
    
    def test_init_with_custom_secret(self, mock_db):
        """Test AuthService initialization with custom secret"""
        with patch('database.db_manager.get_db', return_value=mock_db):
            service = AuthService(secret_key='custom_secret_key')
            assert service.secret_key == 'custom_secret_key'
    
    def test_init_with_default_secret(self, mock_db):
        """Test AuthService initialization with default secret"""
        with patch('database.db_manager.get_db', return_value=mock_db):
            service = AuthService()
            assert service.secret_key == 'default_secret_key_change_in_production'
    
    def test_register_user_duplicate_email(self, auth_service, mock_db):
        """Test user registration with duplicate email"""
        # Setup mock to indicate existing user
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = (1,)  # User exists
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.register_user("test@example.com", "password123", "Test User")
        assert result is False
    
    def test_register_user_database_error(self, auth_service, mock_db):
        """Test user registration with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.register_user("test@example.com", "password123", "Test User")
        assert result is False
    
    def test_authenticate_user_invalid_credentials(self, auth_service, mock_db):
        """Test authentication with invalid credentials"""
        # Setup mock to return no user
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = None
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.authenticate_user("test@example.com", "wrongpassword")
        assert result is None
    
    def test_authenticate_user_database_error(self, auth_service, mock_db):
        """Test authentication with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.authenticate_user("test@example.com", "password123")
        assert result is None
    
    def test_generate_token_with_custom_expiration(self, auth_service):
        """Test token generation with custom expiration"""
        user_id = 123
        custom_exp = 7200  # 2 hours
        
        token = auth_service.generate_token(user_id, expires_in=custom_exp)
        
        # Decode token to verify expiration
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=['HS256'])
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
            'user_id': 123,
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, auth_service.secret_key, algorithm='HS256')
        
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
            'user_id': 123,
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, auth_service.secret_key, algorithm='HS256')
        
        result = auth_service.refresh_token(expired_token)
        assert result is None
    
    def test_create_session_with_database_error(self, auth_service, mock_db):
        """Test session creation with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.create_session(123, "test_token")
        assert result is None
    
    def test_validate_session_with_database_error(self, auth_service, mock_db):
        """Test session validation with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.validate_session("test_token")
        assert result is None
    
    def test_validate_session_with_none_token(self, auth_service):
        """Test session validation with None token"""
        result = auth_service.validate_session(None)
        assert result is None
    
    def test_validate_session_with_empty_string(self, auth_service):
        """Test session validation with empty string"""
        result = auth_service.validate_session("")
        assert result is None
    
    def test_cleanup_expired_sessions_with_database_error(self, auth_service, mock_db):
        """Test session cleanup with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        # Should not raise exception
        auth_service.cleanup_expired_sessions()
    
    def test_get_user_sessions_with_database_error(self, auth_service, mock_db):
        """Test getting user sessions with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.get_user_sessions(123)
        assert result == []
    
    def test_revoke_session_with_database_error(self, auth_service, mock_db):
        """Test session revocation with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.revoke_session("test_token")
        assert result is False
    
    def test_revoke_all_user_sessions_with_database_error(self, auth_service, mock_db):
        """Test revoking all user sessions with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.revoke_all_user_sessions(123)
        assert result is False
    
    def test_change_password_with_wrong_current_password(self, auth_service, mock_db):
        """Test password change with wrong current password"""
        # Setup mock to return user with different password
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = (
            123,
            "test@example.com",
            generate_password_hash("correct_password"),
            "Test User",
            "user"
        )
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.change_password(123, "wrong_password", "new_password")
        assert result is False
    
    def test_change_password_with_database_error(self, auth_service, mock_db):
        """Test password change with database error"""
        # Setup mock to raise exception on update
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = (
            123,
            "test@example.com",
            generate_password_hash("current_password"),
            "Test User",
            "user"
        )
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.change_password(123, "current_password", "new_password")
        assert result is False
    
    def test_request_password_reset_with_nonexistent_email(self, auth_service, mock_db):
        """Test password reset request for non-existent email"""
        # Setup mock to return no user
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = None
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.request_password_reset("nonexistent@example.com")
        assert result is False
    
    def test_request_password_reset_with_database_error(self, auth_service, mock_db):
        """Test password reset request with database error"""
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.request_password_reset("test@example.com")
        assert result is False
    
    def test_reset_password_with_invalid_token(self, auth_service, mock_db):
        """Test password reset with invalid token"""
        result = auth_service.reset_password("invalid_token", "new_password")
        assert result is False
    
    def test_reset_password_with_expired_token(self, auth_service, mock_db):
        """Test password reset with expired token"""
        # Create expired reset token
        expired_token = auth_service.generate_reset_token(123, expires_in=-3600)  # Expired 1 hour ago
        
        result = auth_service.reset_password(expired_token, "new_password")
        assert result is False
    
    def test_reset_password_with_database_error(self, auth_service, mock_db):
        """Test password reset with database error"""
        # Create valid reset token
        reset_token = auth_service.generate_reset_token(123)
        
        # Setup mock to raise exception
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = (123,)
        cursor_mock.execute.side_effect = Exception("Database error")
        mock_db.cursor.return_value = cursor_mock
        
        result = auth_service.reset_password(reset_token, "new_password")
        assert result is False
    
    def test_generate_reset_token_with_custom_expiration(self, auth_service):
        """Test reset token generation with custom expiration"""
        user_id = 123
        custom_exp = 3600  # 1 hour
        
        token = auth_service.generate_reset_token(user_id, expires_in=custom_exp)
        
        # Decode token to verify expiration
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=['HS256'])
        exp_time = datetime.fromtimestamp(decoded['exp'])
        expected_exp = datetime.utcnow() + timedelta(seconds=custom_exp)
        
        # Allow for small time difference (within 1 minute)
        assert abs((exp_time - expected_exp).total_seconds()) < 60
    
    def test_validate_reset_token_with_none_token(self, auth_service):
        """Test reset token validation with None token"""
        result = auth_service.validate_reset_token(None)
        assert result is None
    
    def test_validate_reset_token_with_empty_string(self, auth_service):
        """Test reset token validation with empty string"""
        result = auth_service.validate_reset_token("")
        assert result is None
    
    def test_validate_reset_token_with_invalid_token(self, auth_service):
        """Test reset token validation with invalid token"""
        result = auth_service.validate_reset_token("invalid_token")
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
        token = auth_service.generate_token(123)
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
        token = auth_service.generate_token(123)
        payload = auth_service.get_token_payload(token)
        
        assert payload is not None
        assert payload['user_id'] == 123
        assert 'exp' in payload
        assert 'iat' in payload


class TestMiddlewareFinalCoverage:
    """Test cases for AuthMiddleware to cover remaining gaps"""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock AuthService"""
        service_mock = Mock()
        return service_mock
    
    @pytest.fixture
    def middleware(self, mock_auth_service):
        """Create AuthMiddleware with mocked AuthService"""
        with patch('auth.middleware.AuthService', return_value=mock_auth_service):
            return AuthMiddleware()
    
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
        custom_service = Mock()
        middleware = AuthMiddleware(auth_service=custom_service)
        assert middleware.auth_service == custom_service
    
    def test_init_with_default_auth_service(self):
        """Test AuthMiddleware initialization with default AuthService"""
        with patch('auth.middleware.AuthService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            middleware = AuthMiddleware()
            assert middleware.auth_service == mock_service
    
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
            
            @middleware.role_required("admin")
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
            mock_auth_service.authenticate_user.side_effect = Exception("Service error")
            
            result = middleware.login_user("test@example.com", "password")
            assert result is False
    
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
    
    def test_show_login_form_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test show_login_form with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.authenticate_user.side_effect = Exception("Service error")
            
            # Setup form to return True for submit button
            mock_st.button.return_value = True
            
            # Should not raise exception
            middleware.show_login_form()
    
    def test_show_register_form_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test show_register_form with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.register_user.side_effect = Exception("Service error")
            
            # Setup form to return True for submit button
            mock_st.button.return_value = True
            
            # Should not raise exception
            middleware.show_register_form()
    
    def test_show_password_reset_form_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test show_password_reset_form with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.request_password_reset.side_effect = Exception("Service error")
            
            # Setup form to return True for submit button
            mock_st.button.return_value = True
            
            # Should not raise exception
            middleware.show_password_reset_form()
    
    def test_show_user_menu_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test show_user_menu with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.validate_token.side_effect = Exception("Service error")
            
            # Should not raise exception
            middleware.show_user_menu()
    
    def test_show_profile_settings_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test show_profile_settings with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.change_password.side_effect = Exception("Service error")
            
            # Setup form to return True for submit button
            mock_st.button.return_value = True
            
            # Should not raise exception
            middleware.show_profile_settings()
    
    def test_show_change_password_form_with_exception(self, middleware, mock_st, mock_auth_service):
        """Test show_change_password_form with exception"""
        with patch('auth.middleware.st', mock_st):
            # Setup mock to raise exception
            mock_auth_service.change_password.side_effect = Exception("Service error")
            
            # Setup form to return True for submit button
            mock_st.button.return_value = True
            
            # Should not raise exception
            middleware.show_change_password_form()


class TestUserModelFinalCoverage:
    """Test cases for UserProfile to cover remaining gaps"""
    
    def test_user_profile_init_with_all_parameters(self):
        """Test UserProfile initialization with all parameters"""
        now = datetime.utcnow()
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user",
            status="active",
            created_at=now,
            updated_at=now
        )
        
        assert profile.user_id == 123
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name == "Test User"
        assert profile.role == "user"
        assert profile.status == "active"
        assert profile.created_at == now
        assert profile.updated_at == now
    
    def test_user_profile_init_with_minimal_parameters(self):
        """Test UserProfile initialization with minimal parameters"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        assert profile.user_id == 123
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name == "Test User"
        assert profile.role == "user"
        assert profile.status == "active"  # Default value
        assert profile.created_at is not None
        assert profile.updated_at is not None
    
    def test_user_profile_to_dict(self):
        """Test UserProfile to_dict method"""
        now = datetime.utcnow()
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user",
            status="active",
            created_at=now,
            updated_at=now
        )
        
        result = profile.to_dict()
        
        assert result['user_id'] == 123
        assert result['email'] == "test@example.com"
        assert result['password_hash'] == "hashed_password"
        assert result['full_name'] == "Test User"
        assert result['role'] == "user"
        assert result['status'] == "active"
        assert result['created_at'] == now
        assert result['updated_at'] == now
    
    def test_user_profile_from_dict(self):
        """Test UserProfile from_dict method"""
        now = datetime.utcnow()
        data = {
            'user_id': 123,
            'email': "test@example.com",
            'password_hash': "hashed_password",
            'full_name': "Test User",
            'role': "user",
            'status': "active",
            'created_at': now,
            'updated_at': now
        }
        
        profile = UserProfile.from_dict(data)
        
        assert profile.user_id == 123
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name == "Test User"
        assert profile.role == "user"
        assert profile.status == "active"
        assert profile.created_at == now
        assert profile.updated_at == now
    
    def test_user_profile_from_dict_with_minimal_data(self):
        """Test UserProfile from_dict method with minimal data"""
        data = {
            'user_id': 123,
            'email': "test@example.com",
            'password_hash': "hashed_password",
            'full_name': "Test User",
            'role': "user"
        }
        
        profile = UserProfile.from_dict(data)
        
        assert profile.user_id == 123
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name == "Test User"
        assert profile.role == "user"
        assert profile.status == "active"  # Default value
        assert profile.created_at is not None
        assert profile.updated_at is not None
    
    def test_user_profile_str_representation(self):
        """Test UserProfile string representation"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = str(profile)
        assert "test@example.com" in result
        assert "Test User" in result
        assert "user" in result
    
    def test_user_profile_repr_representation(self):
        """Test UserProfile repr representation"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = repr(profile)
        assert "UserProfile" in result
        assert "123" in result
        assert "test@example.com" in result
    
    def test_user_profile_equality(self):
        """Test UserProfile equality comparison"""
        profile1 = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        profile2 = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        profile3 = UserProfile(
            user_id=456,
            email="other@example.com",
            password_hash="hashed_password",
            full_name="Other User",
            role="user"
        )
        
        assert profile1 == profile2
        assert profile1 != profile3
        assert profile1 != "not a profile"
    
    def test_user_profile_hash(self):
        """Test UserProfile hash"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        # Should not raise exception
        hash_value = hash(profile)
        assert isinstance(hash_value, int)
    
    def test_user_profile_set_password(self):
        """Test UserProfile set_password method"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="",
            full_name="Test User",
            role="user"
        )
        
        profile.set_password("password123")
        
        # Password should be hashed
        assert profile.password_hash != "password123"
        assert check_password_hash(profile.password_hash, "password123")
    
    def test_user_profile_check_password_with_correct_password(self):
        """Test UserProfile check_password with correct password"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="",
            full_name="Test User",
            role="user"
        )
        
        profile.set_password("password123")
        
        result = profile.check_password("password123")
        assert result is True
    
    def test_user_profile_check_password_with_wrong_password(self):
        """Test UserProfile check_password with wrong password"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="",
            full_name="Test User",
            role="user"
        )
        
        profile.set_password("password123")
        
        result = profile.check_password("wrongpassword")
        assert result is False
    
    def test_user_profile_check_password_with_no_password(self):
        """Test UserProfile check_password with no password set"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="",
            full_name="Test User",
            role="user"
        )
        
        result = profile.check_password("password123")
        assert result is False
    
    def test_user_profile_update_profile(self):
        """Test UserProfile update_profile method"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        original_updated_at = profile.updated_at
        
        # Wait a bit to ensure timestamp difference
        time.sleep(0.01)
        
        profile.update_profile(
            full_name="Updated Name",
            email="updated@example.com",
            role="admin"
        )
        
        assert profile.full_name == "Updated Name"
        assert profile.email == "updated@example.com"
        assert profile.role == "admin"
        assert profile.updated_at > original_updated_at
    
    def test_user_profile_update_profile_with_no_changes(self):
        """Test UserProfile update_profile with no changes"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        original_updated_at = profile.updated_at
        
        # Wait a bit to ensure timestamp difference
        time.sleep(0.01)
        
        profile.update_profile()
        
        # Updated at should still change
        assert profile.updated_at > original_updated_at
    
    def test_user_profile_is_admin_with_admin_role(self):
        """Test UserProfile is_admin with admin role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="admin"
        )
        
        result = profile.is_admin()
        assert result is True
    
    def test_user_profile_is_admin_with_user_role(self):
        """Test UserProfile is_admin with user role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.is_admin()
        assert result is False
    
    def test_user_profile_is_active_with_active_status(self):
        """Test UserProfile is_active with active status"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user",
            status="active"
        )
        
        result = profile.is_active()
        assert result is True
    
    def test_user_profile_is_active_with_inactive_status(self):
        """Test UserProfile is_active with inactive status"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user",
            status="inactive"
        )
        
        result = profile.is_active()
        assert result is False
    
    def test_user_profile_activate(self):
        """Test UserProfile activate method"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user",
            status="inactive"
        )
        
        profile.activate()
        
        assert profile.status == "active"
    
    def test_user_profile_deactivate(self):
        """Test UserProfile deactivate method"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user",
            status="active"
        )
        
        profile.deactivate()
        
        assert profile.status == "inactive"
    
    def test_user_profile_has_permission_with_admin_role(self):
        """Test UserProfile has_permission with admin role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="admin"
        )
        
        result = profile.has_permission("any_permission")
        assert result is True
    
    def test_user_profile_has_permission_with_user_role_and_matching_permission(self):
        """Test UserProfile has_permission with user role and matching permission"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.has_permission("read_profile")
        assert result is True
    
    def test_user_profile_has_permission_with_user_role_and_non_matching_permission(self):
        """Test UserProfile has_permission with user role and non-matching permission"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.has_permission("admin_permission")
        assert result is False
    
    def test_user_profile_get_permissions_with_admin_role(self):
        """Test UserProfile get_permissions with admin role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="admin"
        )
        
        result = profile.get_permissions()
        assert "all" in result
    
    def test_user_profile_get_permissions_with_user_role(self):
        """Test UserProfile get_permissions with user role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.get_permissions()
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_user_profile_validate_with_valid_profile(self):
        """Test UserProfile validate with valid profile"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.validate()
        assert result is True
    
    def test_user_profile_validate_with_invalid_email(self):
        """Test UserProfile validate with invalid email"""
        profile = UserProfile(
            user_id=123,
            email="invalid_email",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.validate()
        assert result is False
    
    def test_user_profile_validate_with_empty_name(self):
        """Test UserProfile validate with empty name"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="",
            role="user"
        )
        
        result = profile.validate()
        assert result is False
    
    def test_user_profile_validate_with_empty_password_hash(self):
        """Test UserProfile validate with empty password hash"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="",
            full_name="Test User",
            role="user"
        )
        
        result = profile.validate()
        assert result is False
    
    def test_user_profile_validate_with_invalid_role(self):
        """Test UserProfile validate with invalid role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="invalid_role"
        )
        
        result = profile.validate()
        assert result is False
    
    def test_user_profile_get_validation_errors_with_valid_profile(self):
        """Test UserProfile get_validation_errors with valid profile"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.get_validation_errors()
        assert len(result) == 0
    
    def test_user_profile_get_validation_errors_with_invalid_email(self):
        """Test UserProfile get_validation_errors with invalid email"""
        profile = UserProfile(
            user_id=123,
            email="invalid_email",
            password_hash="hashed_password",
            full_name="Test User",
            role="user"
        )
        
        result = profile.get_validation_errors()
        assert len(result) > 0
        assert any("email" in error.lower() for error in result)
    
    def test_user_profile_get_validation_errors_with_empty_name(self):
        """Test UserProfile get_validation_errors with empty name"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="",
            role="user"
        )
        
        result = profile.get_validation_errors()
        assert len(result) > 0
        assert any("name" in error.lower() for error in result)
    
    def test_user_profile_get_validation_errors_with_empty_password_hash(self):
        """Test UserProfile get_validation_errors with empty password hash"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="",
            full_name="Test User",
            role="user"
        )
        
        result = profile.get_validation_errors()
        assert len(result) > 0
        assert any("password" in error.lower() for error in result)
    
    def test_user_profile_get_validation_errors_with_invalid_role(self):
        """Test UserProfile get_validation_errors with invalid role"""
        profile = UserProfile(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role="invalid_role"
        )
        
        result = profile.get_validation_errors()
        assert len(result) > 0
        assert any("role" in error.lower() for error in result)