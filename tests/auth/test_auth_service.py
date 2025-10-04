"""
Tests for authentication service functionality.

Tests cover user registration, login, JWT tokens, session management,
and password reset functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from auth.auth_service import AuthService, AuthResult
from auth.user_model import UserModel, UserRole, UserStatus


class TestAuthService:
    """Test cases for AuthService."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp:
            yield temp

    @pytest.fixture
    def user_model(self, temp_dir):
        """Create user model with temporary storage."""
        with patch.dict(os.environ, {'AUTH_DATA_DIR': temp_dir}):
            model = UserModel()
            yield model

    @pytest.fixture
    def auth_service(self, user_model):
        """Create auth service with user model."""
        service = AuthService(user_model)
        yield service

    def test_user_registration_success(self, auth_service):
        """Test successful user registration."""
        result = auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        assert result.success == True
        assert result.user is not None
        assert result.user.email == "test@example.com"
        assert result.user.full_name == "Test User"
        assert result.user.role == UserRole.PATIENT

    def test_user_registration_duplicate_email(self, auth_service):
        """Test registration with duplicate email fails."""
        # Register first user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Try to register again with same email
        result = auth_service.register_user(
            email="test@example.com",
            password="TestPass456",
            full_name="Another User"
        )

        assert result.success == False
        assert "already exists" in result.error_message

    def test_user_registration_weak_password(self, auth_service):
        """Test registration with weak password fails."""
        result = auth_service.register_user(
            email="test@example.com",
            password="weak",
            full_name="Test User"
        )

        assert result.success == False
        assert "password" in result.error_message.lower()

    def test_user_login_success(self, auth_service):
        """Test successful user login."""
        # Register user first
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Login
        result = auth_service.login_user("test@example.com", "TestPass123")

        assert result.success == True
        assert result.user is not None
        assert result.token is not None
        assert result.session is not None
        assert result.user.email == "test@example.com"

    def test_user_login_wrong_password(self, auth_service):
        """Test login with wrong password fails."""
        # Register user first
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Try login with wrong password
        result = auth_service.login_user("test@example.com", "WrongPass123")

        assert result.success == False
        assert "Invalid credentials" in result.error_message

    def test_user_login_nonexistent_user(self, auth_service):
        """Test login with nonexistent user fails."""
        result = auth_service.login_user("nonexistent@example.com", "TestPass123")

        assert result.success == False
        assert "Invalid credentials" in result.error_message

    def test_token_validation_success(self, auth_service):
        """Test successful token validation."""
        # Register and login user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )
        login_result = auth_service.login_user("test@example.com", "TestPass123")

        # Validate token
        user = auth_service.validate_token(login_result.token)

        assert user is not None
        assert user.email == "test@example.com"

    def test_token_validation_expired(self, auth_service):
        """Test expired token validation fails."""
        with patch('auth.auth_service.datetime') as mock_datetime:
            # Set current time
            now = datetime.now()
            mock_datetime.now.return_value = now

            # Register and login user
            auth_service.register_user(
                email="test@example.com",
                password="TestPass123",
                full_name="Test User"
            )
            login_result = auth_service.login_user("test@example.com", "TestPass123")

            # Move time forward past token expiration
            future_time = now + timedelta(hours=25)
            mock_datetime.now.return_value = future_time

            # Validate token (should fail)
            user = auth_service.validate_token(login_result.token)
            assert user is None

    def test_token_validation_invalid(self, auth_service):
        """Test invalid token validation fails."""
        user = auth_service.validate_token("invalid.token.here")
        assert user is None

    def test_logout_user(self, auth_service):
        """Test user logout."""
        # Register and login user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )
        login_result = auth_service.login_user("test@example.com", "TestPass123")

        # Logout
        success = auth_service.logout_user(login_result.token)
        assert success == True

        # Token should no longer be valid
        user = auth_service.validate_token(login_result.token)
        assert user is None

    def test_password_reset_initiate(self, auth_service):
        """Test password reset initiation."""
        # Register user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Initiate password reset
        result = auth_service.initiate_password_reset("test@example.com")

        assert result.success == True

    def test_password_reset_complete(self, auth_service):
        """Test complete password reset."""
        # Register user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Initiate password reset
        reset_result = auth_service.initiate_password_reset("test@example.com")
        assert reset_result.success == True

        # For testing, we'll need to access the reset token
        # In real implementation, this would be sent via email
        user = auth_service.user_model.get_user_by_email("test@example.com")
        reset_token = user.password_reset_token

        # Reset password
        result = auth_service.reset_password(reset_token, "NewPass123")

        assert result.success == True

        # Should be able to login with new password
        login_result = auth_service.login_user("test@example.com", "NewPass123")
        assert login_result.success == True

    def test_change_password(self, auth_service):
        """Test password change."""
        # Register user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Login to get user
        login_result = auth_service.login_user("test@example.com", "TestPass123")
        user_id = login_result.user.user_id

        # Change password
        result = auth_service.change_password(user_id, "TestPass123", "NewPass123")

        assert result.success == True

        # Should be able to login with new password
        login_result = auth_service.login_user("test@example.com", "NewPass123")
        assert login_result.success == True

        # Old password should no longer work
        login_result = auth_service.login_user("test@example.com", "TestPass123")
        assert login_result.success == False

    def test_session_management(self, auth_service):
        """Test session management."""
        # Register and login user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )
        login_result = auth_service.login_user("test@example.com", "TestPass123")

        user_id = login_result.user.user_id

        # Check user has sessions
        sessions = auth_service.get_user_sessions(user_id)
        assert len(sessions) > 0

        # Invalidate all user sessions
        count = auth_service.invalidate_user_sessions(user_id)
        assert count > 0

        # User should no longer have active sessions
        sessions = auth_service.get_user_sessions(user_id)
        assert len(sessions) == 0

        # Token should no longer be valid
        user = auth_service.validate_token(login_result.token)
        assert user is None

    def test_concurrent_session_limit(self, auth_service):
        """Test concurrent session limits."""
        # Register user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Login multiple times to reach limit
        results = []
        for i in range(auth_service.max_concurrent_sessions + 2):
            result = auth_service.login_user("test@example.com", "TestPass123")
            results.append(result)

        # First few should succeed
        successful_logins = sum(1 for r in results if r.success)
        assert successful_logins == auth_service.max_concurrent_sessions

    def test_access_validation(self, auth_service):
        """Test resource access validation."""
        # Register user
        auth_service.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        user = auth_service.user_model.get_user_by_email("test@example.com")

        # Test access permissions
        assert auth_service.validate_session_access(user.user_id, "own_profile", "read")
        assert auth_service.validate_session_access(user.user_id, "therapy_sessions", "read")
        assert not auth_service.validate_session_access(user.user_id, "admin_panel", "read")