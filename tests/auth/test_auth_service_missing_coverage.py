"""
Tests for auth service missing coverage lines
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import jwt

# Import the auth service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from auth.auth_service import AuthService, AuthSession, AuthResult


class TestAuthServiceMissingCoverage:
    """Tests for missing coverage lines in auth service"""
    
    @pytest.fixture
    def mock_user_model(self):
        """Mock user model"""
        with patch('auth.auth_service.UserModel') as mock:
            yield mock
    
    @pytest.fixture
    def mock_jwt(self):
        """Mock JWT"""
        with patch('auth.auth_service.jwt') as mock:
            yield mock
    
    @pytest.fixture
    def mock_threading(self):
        """Mock threading"""
        with patch('auth.auth_service.threading') as mock:
            yield mock
    
    def test_auth_session_is_expired(self):
        """Test AuthSession.is_expired method"""
        # Test expired session
        expired_time = datetime.utcnow() - timedelta(hours=1)
        session = AuthSession(
            session_id="test_session",
            user_id="test_user",
            created_at=expired_time,
            expires_at=expired_time,
            last_accessed=expired_time,
            ip_address="127.0.0.1",
            user_agent="test_agent",
            is_active=True
        )
        
        assert session.is_expired() is True
        
        # Test active session
        future_time = datetime.utcnow() + timedelta(hours=1)
        active_session = AuthSession(
            session_id="test_session",
            user_id="test_user",
            created_at=datetime.utcnow(),
            expires_at=future_time,
            last_accessed=datetime.utcnow(),
            ip_address="127.0.0.1",
            user_agent="test_agent",
            is_active=True
        )
        
        assert active_session.is_expired() is False
    
    def test_auth_session_to_dict(self):
        """Test AuthSession.to_dict method"""
        session = AuthSession(
            session_id="test_session",
            user_id="test_user",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            last_accessed=datetime.utcnow(),
            ip_address="127.0.0.1",
            user_agent="test_agent",
            is_active=True
        )
        
        result = session.to_dict()
        
        assert result["session_id"] == "test_session"
        assert result["user_id"] == "test_user"
        assert result["ip_address"] == "127.0.0.1"
        assert result["user_agent"] == "test_agent"
        assert result["is_active"] is True
        assert "created_at" in result
        assert "expires_at" in result
        assert "last_accessed" in result
    
    def test_register_user_email_exists(self, mock_user_model):
        """Test register_user when email already exists"""
        mock_user_model.return_value.get_user_by_email.return_value = {"id": "existing_user"}
        
        auth_service = AuthService()
        
        result = auth_service.register_user("existing@example.com", "password", "Test User")
        
        assert result.success is False
        assert "already exists" in result.error
        mock_user_model.return_value.get_user_by_email.assert_called_with("existing@example.com")
    
    def test_register_user_creation_failure(self, mock_user_model):
        """Test register_user when user creation fails"""
        mock_user_model.return_value.get_user_by_email.return_value = None
        mock_user_model.return_value.create_user.return_value = None
        
        auth_service = AuthService()
        
        result = auth_service.register_user("new@example.com", "password", "Test User")
        
        assert result.success is False
        assert "Failed to create user" in result.error
        mock_user_model.return_value.create_user.assert_called()
    
    def test_login_user_invalid_credentials(self, mock_user_model):
        """Test login_user with invalid credentials"""
        mock_user_model.return_value.authenticate_user.return_value = None
        
        auth_service = AuthService()
        
        result = auth_service.login_user("test@example.com", "wrong_password")
        
        assert result.success is False
        assert "Invalid credentials" in result.error
        mock_user_model.return_value.authenticate_user.assert_called_with("test@example.com", "wrong_password")
    
    def test_login_user_account_locked(self, mock_user_model):
        """Test login_user with locked account"""
        locked_user = {"id": "locked_user", "is_locked": True}
        mock_user_model.return_value.authenticate_user.return_value = locked_user
        
        auth_service = AuthService()
        
        result = auth_service.login_user("locked@example.com", "password")
        
        assert result.success is False
        assert "account is locked" in result.error.lower()
    
    def test_validate_token_invalid_token(self, mock_jwt):
        """Test validate_token with invalid token"""
        mock_jwt.decode.side_effect = jwt.InvalidTokenError("Invalid token")
        
        auth_service = AuthService()
        
        result = auth_service.validate_token("invalid_token")
        
        assert result is None
        mock_jwt.decode.assert_called()
    
    def test_validate_token_expired_token(self, mock_jwt):
        """Test validate_token with expired token"""
        mock_jwt.decode.side_effect = jwt.ExpiredSignatureError("Token expired")
        
        auth_service = AuthService()
        
        result = auth_service.validate_token("expired_token")
        
        assert result is None
        mock_jwt.decode.assert_called()
    
    def test_validate_token_missing_user_id(self, mock_jwt):
        """Test validate_token with missing user_id"""
        mock_jwt.decode.return_value = {"exp": int(time.time()) + 3600}  # No user_id
        
        auth_service = AuthService()
        
        result = auth_service.validate_token("token_without_user_id")
        
        assert result is None
    
    def test_refresh_token_invalid_token(self, mock_jwt):
        """Test refresh_token with invalid token"""
        mock_jwt.decode.side_effect = jwt.InvalidTokenError("Invalid token")
        
        auth_service = AuthService()
        
        result = auth_service.refresh_token("invalid_token")
        
        assert result.success is False
        assert "Invalid refresh token" in result.error
    
    def test_refresh_token_session_not_found(self, mock_jwt):
        """Test refresh_token when session not found"""
        mock_jwt.decode.return_value = {"session_id": "nonexistent_session"}
        
        auth_service = AuthService()
        
        result = auth_service.refresh_token("token_with_nonexistent_session")
        
        assert result.success is False
        assert "Session not found" in result.error
    
    def test_initiate_password_reset_user_not_found(self, mock_user_model):
        """Test initiate_password_reset when user not found"""
        mock_user_model.return_value.get_user_by_email.return_value = None
        
        auth_service = AuthService()
        
        result = auth_service.initiate_password_reset("nonexistent@example.com")
        
        assert result.success is False
        assert "User not found" in result.error
    
    def test_reset_password_invalid_token(self, mock_user_model):
        """Test reset_password with invalid token"""
        mock_user_model.return_value.get_user_by_reset_token.return_value = None
        
        auth_service = AuthService()
        
        result = auth_service.reset_password("invalid_token", "new_password")
        
        assert result.success is False
        assert "Invalid or expired reset token" in result.error
    
    def test_change_password_incorrect_current_password(self, mock_user_model):
        """Test change_password with incorrect current password"""
        user = {"id": "test_user", "email": "test@example.com"}
        mock_user_model.return_value.get_user.return_value = user
        mock_user_model.return_value.change_password.return_value = False
        
        auth_service = AuthService()
        
        result = auth_service.change_password("test_user", "wrong_current_password", "new_password")
        
        assert result.success is False
        assert "Current password is incorrect" in result.error
    
    def test_create_session_concurrent_session_limit(self, mock_user_model):
        """Test _create_session with concurrent session limit"""
        user = {"id": "test_user", "email": "test@example.com"}
        mock_user_model.return_value.get_user.return_value = user
        
        # Mock existing sessions (at limit)
        existing_sessions = [
            Mock(is_expired=lambda: False, is_active=True)
            for _ in range(5)  # Assuming limit is 5
        ]
        
        auth_service = AuthService()
        auth_service.sessions = {"test_user": existing_sessions}
        
        with patch.object(auth_service, '_generate_session_id', return_value="new_session_id"):
            session = auth_service._create_session(
                user_id="test_user",
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )
        
        assert session is not None
        assert len([s for s in auth_service.sessions["test_user"] if s.is_active]) <= 5
    
    def test_create_session_cleanup_expired(self, mock_user_model):
        """Test _create_session cleanup of expired sessions"""
        user = {"id": "test_user", "email": "test@example.com"}
        mock_user_model.return_value.get_user.return_value = user
        
        # Mock expired sessions
        expired_session = Mock(is_expired=lambda: True, is_active=True)
        active_session = Mock(is_expired=lambda: False, is_active=True)
        
        auth_service = AuthService()
        auth_service.sessions = {"test_user": [expired_session, active_session]}
        
        with patch.object(auth_service, '_generate_session_id', return_value="new_session_id"):
            session = auth_service._create_session(
                user_id="test_user",
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )
        
        # Expired session should be removed
        assert expired_session not in auth_service.sessions["test_user"]
        assert active_session in auth_service.sessions["test_user"]
    
    def test_invalidate_session(self):
        """Test _invalidate_session method"""
        session = Mock(is_active=True)
        
        auth_service = AuthService()
        auth_service._invalidate_session(session)
        
        assert session.is_active is False
    
    def test_is_session_valid(self):
        """Test _is_session_valid method"""
        # Valid session
        valid_session = Mock(is_expired=lambda: False, is_active=True)
        
        # Expired session
        expired_session = Mock(is_expired=lambda: True, is_active=True)
        
        # Inactive session
        inactive_session = Mock(is_expired=lambda: False, is_active=False)
        
        auth_service = AuthService()
        
        assert auth_service._is_session_valid(valid_session) is True
        assert auth_service._is_session_valid(expired_session) is False
        assert auth_service._is_session_valid(inactive_session) is False
    
    def test_generate_session_id(self):
        """Test _generate_session_id method"""
        auth_service = AuthService()
        
        session_id1 = auth_service._generate_session_id()
        session_id2 = auth_service._generate_session_id()
        
        assert session_id1 != session_id2
        assert len(session_id1) > 0
        assert len(session_id2) > 0
    
    def test_background_cleanup(self, mock_threading):
        """Test _background_cleanup method"""
        auth_service = AuthService()
        
        with patch.object(auth_service, '_cleanup_expired_sessions') as mock_cleanup:
            auth_service._background_cleanup()
        
        mock_cleanup.assert_called_once()
        mock_threading.Timer.assert_called_once()
    
    def test_cleanup_expired_sessions(self):
        """Test _cleanup_expired_sessions method"""
        # Mock expired sessions
        expired_session1 = Mock(is_expired=lambda: True, is_active=True)
        expired_session2 = Mock(is_expired=lambda: True, is_active=True)
        active_session = Mock(is_expired=lambda: False, is_active=True)
        
        auth_service = AuthService()
        auth_service.sessions = {
            "user1": [expired_session1, active_session],
            "user2": [expired_session2]
        }
        
        auth_service._cleanup_expired_sessions()
        
        # Expired sessions should be removed
        assert expired_session1 not in auth_service.sessions["user1"]
        assert active_session in auth_service.sessions["user1"]
        assert expired_session2 not in auth_service.sessions["user2"]
    
    def test_get_auth_statistics(self):
        """Test get_auth_statistics method"""
        # Mock sessions
        active_session = Mock(is_expired=lambda: False, is_active=True)
        expired_session = Mock(is_expired=lambda: True, is_active=True)
        inactive_session = Mock(is_expired=lambda: False, is_active=False)
        
        auth_service = AuthService()
        auth_service.sessions = {
            "user1": [active_session, expired_session],
            "user2": [inactive_session]
        }
        
        stats = auth_service.get_auth_statistics()
        
        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "expired_sessions" in stats
        assert "users_with_sessions" in stats
        assert stats["total_sessions"] == 3
        assert stats["active_sessions"] == 1
        assert stats["expired_sessions"] == 1
        assert stats["users_with_sessions"] == 2