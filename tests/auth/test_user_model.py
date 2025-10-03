"""
Tests for user model functionality.

Tests cover user creation, authentication, password management,
and role-based access control.
"""

import pytest
import os
import tempfile
from unittest.mock import patch

from auth.user_model import UserModel, UserRole, UserStatus


class TestUserModel:
    """Test cases for UserModel."""

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

    def test_create_user_success(self, user_model):
        """Test successful user creation."""
        user = user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User",
            role=UserRole.PATIENT
        )

        assert user.user_id is not None
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == UserRole.PATIENT
        assert user.status == UserStatus.ACTIVE

    def test_create_user_duplicate_email(self, user_model):
        """Test creating user with duplicate email fails."""
        # Create first user
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Try to create second user with same email
        with pytest.raises(ValueError, match="already exists"):
            user_model.create_user(
                email="test@example.com",
                password="TestPass456",
                full_name="Another User"
            )

    def test_create_user_weak_password(self, user_model):
        """Test creating user with weak password fails."""
        with pytest.raises(ValueError, match="password"):
            user_model.create_user(
                email="test@example.com",
                password="weak",
                full_name="Test User"
            )

    def test_authenticate_user_success(self, user_model):
        """Test successful user authentication."""
        # Create user
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Authenticate
        user = user_model.authenticate_user("test@example.com", "TestPass123")

        assert user is not None
        assert user.email == "test@example.com"
        assert user.last_login is not None

    def test_authenticate_user_wrong_password(self, user_model):
        """Test authentication with wrong password fails."""
        # Create user
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Try wrong password
        user = user_model.authenticate_user("test@example.com", "WrongPass123")

        assert user is None

    def test_authenticate_user_nonexistent(self, user_model):
        """Test authentication of nonexistent user fails."""
        user = user_model.authenticate_user("nonexistent@example.com", "TestPass123")
        assert user is None

    def test_get_user_by_email(self, user_model):
        """Test getting user by email."""
        # Create user
        created_user = user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Get by email
        retrieved_user = user_model.get_user_by_email("test@example.com")

        assert retrieved_user is not None
        assert retrieved_user.user_id == created_user.user_id
        assert retrieved_user.email == "test@example.com"

    def test_get_user_by_email_case_insensitive(self, user_model):
        """Test email lookup is case insensitive."""
        # Create user
        user_model.create_user(
            email="Test@Example.Com",
            password="TestPass123",
            full_name="Test User"
        )

        # Get with different case
        user = user_model.get_user_by_email("test@example.com")
        assert user is not None

        user = user_model.get_user_by_email("TEST@EXAMPLE.COM")
        assert user is not None

    def test_update_user(self, user_model):
        """Test user profile update."""
        # Create user
        user = user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Update profile
        success = user_model.update_user(user.user_id, {
            'full_name': 'Updated Name',
            'preferences': {'theme': 'dark'}
        })

        assert success == True

        # Verify changes
        updated_user = user_model.get_user(user.user_id)
        assert updated_user.full_name == 'Updated Name'
        assert updated_user.preferences['theme'] == 'dark'

    def test_change_password_success(self, user_model):
        """Test successful password change."""
        # Create user
        user = user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Change password
        success = user_model.change_password(user.user_id, "TestPass123", "NewPass123")

        assert success == True

        # Should authenticate with new password
        auth_user = user_model.authenticate_user("test@example.com", "NewPass123")
        assert auth_user is not None

        # Old password should fail
        auth_user = user_model.authenticate_user("test@example.com", "TestPass123")
        assert auth_user is None

    def test_change_password_wrong_old(self, user_model):
        """Test password change with wrong old password fails."""
        # Create user
        user = user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Try to change with wrong old password
        success = user_model.change_password(user.user_id, "WrongPass123", "NewPass123")

        assert success == False

    def test_initiate_password_reset(self, user_model):
        """Test password reset initiation."""
        # Create user
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Initiate reset
        token = user_model.initiate_password_reset("test@example.com")

        assert token is not None

        # Check user has reset token
        user = user_model.get_user_by_email("test@example.com")
        assert user.password_reset_token is not None
        assert user.password_reset_expires is not None

    def test_reset_password_success(self, user_model):
        """Test successful password reset."""
        # Create user and initiate reset
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )
        token = user_model.initiate_password_reset("test@example.com")

        # Reset password
        success = user_model.reset_password(token, "NewPass123")

        assert success == True

        # Should authenticate with new password
        user = user_model.authenticate_user("test@example.com", "NewPass123")
        assert user is not None

        # Reset token should be cleared
        user = user_model.get_user_by_email("test@example.com")
        assert user.password_reset_token is None

    def test_reset_password_invalid_token(self, user_model):
        """Test password reset with invalid token fails."""
        success = user_model.reset_password("invalid_token", "NewPass123")
        assert success == False

    def test_account_lockout(self, user_model):
        """Test account lockout after failed attempts."""
        # Create user
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Simulate multiple failed login attempts
        for _ in range(5):
            user_model.authenticate_user("test@example.com", "WrongPass123")

        # Check user is locked
        user = user_model.get_user_by_email("test@example.com")
        assert user.status == UserStatus.LOCKED
        assert user.is_locked()

    def test_role_based_permissions(self, user_model):
        """Test role-based access permissions."""
        # Create users with different roles
        patient = user_model.create_user(
            email="patient@example.com",
            password="TestPass123",
            full_name="Patient User",
            role=UserRole.PATIENT
        )

        therapist = user_model.create_user(
            email="therapist@example.com",
            password="TestPass123",
            full_name="Therapist User",
            role=UserRole.THERAPIST
        )

        admin = user_model.create_user(
            email="admin@example.com",
            password="TestPass123",
            full_name="Admin User",
            role=UserRole.ADMIN
        )

        # Test patient permissions
        assert patient.can_access_resource("own_profile", "read")
        assert patient.can_access_resource("therapy_sessions", "read")
        assert not patient.can_access_resource("admin_panel", "read")

        # Test therapist permissions
        assert therapist.can_access_resource("own_profile", "read")
        assert therapist.can_access_resource("therapy_sessions", "update")
        assert therapist.can_access_resource("assigned_patients", "read")
        assert not therapist.can_access_resource("system_config", "read")

        # Test admin permissions
        assert admin.can_access_resource("all_profiles", "read")
        assert admin.can_access_resource("system_config", "update")
        assert admin.can_access_resource("audit_logs", "read")

    def test_deactivate_user(self, user_model):
        """Test user deactivation."""
        # Create user
        user = user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )

        # Deactivate
        success = user_model.deactivate_user(user.user_id)

        assert success == True

        # Check user is inactive
        user = user_model.get_user(user.user_id)
        assert user.status == UserStatus.INACTIVE

    def test_cleanup_expired_reset_tokens(self, user_model):
        """Test cleanup of expired password reset tokens."""
        from datetime import datetime, timedelta

        # Create user and initiate reset
        user_model.create_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )
        user_model.initiate_password_reset("test@example.com")

        # Manually expire the token
        user = user_model.get_user_by_email("test@example.com")
        user.password_reset_expires = datetime.now() - timedelta(hours=1)

        # Run cleanup
        user_model.cleanup_expired_data()

        # Token should be cleared
        user = user_model.get_user_by_email("test@example.com")
        assert user.password_reset_token is None
        assert user.password_reset_expires is None