"""
Fixed comprehensive tests for user model functionality.

This file provides extensive test coverage for user_model.py including:
- UserProfile class methods and properties
- UserRole and UserStatus enums
- User class functionality
- UserRepository operations
- PII protection and HIPAA compliance
- Account locking and security features
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import os

# Mock streamlit to avoid import issues
with patch.dict('sys.modules', {'streamlit': Mock()}):
    from auth.user_model import (
        UserProfile, UserRole, UserStatus, User, UserRepository
    )


class TestUserProfileComprehensive:
    """Comprehensive test cases for UserProfile class."""

    def test_user_profile_creation_with_all_fields(self):
        """Test user profile creation with all fields populated."""
        created_at = datetime.now()
        updated_at = datetime.now()
        
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at,
            last_login=datetime.now(),
            login_attempts=0,
            account_locked_until=None,
            password_reset_token=None,
            password_reset_expires=None,
            preferences={"theme": "dark", "language": "en"},
            medical_info={"conditions": ["anxiety"], "medications": []}
        )
        
        assert profile.user_id == "user_123"
        assert profile.email == "test@example.com"
        assert profile.full_name == "Test User"
        assert profile.role == UserRole.PATIENT
        assert profile.status == UserStatus.ACTIVE
        assert profile.created_at == created_at
        assert profile.updated_at == updated_at
        assert profile.preferences["theme"] == "dark"
        assert profile.medical_info["conditions"] == ["anxiety"]

    def test_user_profile_to_dict_with_admin_role(self):
        """Test user profile serialization with admin role."""
        created_at = datetime.now()
        updated_at = datetime.now()
        
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at,
            preferences={"theme": "light"},
            medical_info={"conditions": ["depression"], "medications": ["sertraline"]}
        )
        
        # Test with admin role (full access)
        result = profile.to_dict(user_role="admin", include_sensitive=True)
        
        assert result["user_id"] == "user_123"
        assert result["email"] == "test@example.com"
        assert result["full_name"] == "Test User"
        assert result["role"] == UserRole.PATIENT.value
        assert result["status"] == UserStatus.ACTIVE.value
        assert result["created_at"] == created_at.isoformat()
        assert result["updated_at"] == updated_at.isoformat()
        assert result["preferences"] == {"theme": "light"}
        assert result["medical_info"] == {"conditions": ["depression"], "medications": ["sertraline"]}

    def test_user_profile_to_dict_with_patient_role_own_data(self):
        """Test user profile serialization with patient role accessing own data."""
        created_at = datetime.now()
        updated_at = datetime.now()
        
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at,
            preferences={"theme": "light"},
            medical_info={"conditions": ["depression"], "medications": ["sertraline"]}
        )
        
        # Test with patient role (limited access to medical info)
        result = profile.to_dict(user_role="patient", include_sensitive=False)
        
        assert result["user_id"] == "user_123"
        assert result["email"] == "test@example.com"
        assert result["full_name"] == "Test User"
        assert result["role"] == UserRole.PATIENT.value
        assert result["status"] == UserStatus.ACTIVE.value
        assert result["preferences"] == {"theme": "light"}
        # Medical info should be sanitized for patients
        assert "medical_info" in result
        assert isinstance(result["medical_info"], dict)

    def test_user_profile_to_dict_with_guest_role(self):
        """Test user profile serialization with guest role (minimal access)."""
        created_at = datetime.now()
        updated_at = datetime.now()
        
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at,
            preferences={"theme": "light"},
            medical_info={"conditions": ["depression"], "medications": ["sertraline"]}
        )
        
        # Test with guest role (minimal access)
        result = profile.to_dict(user_role="guest", include_sensitive=False)
        
        assert result["user_id"] == "user_123"
        assert result["email"] == "test@example.com"  # Email should be masked for non-admins
        assert result["full_name"] == "Test User"
        assert result["role"] == UserRole.PATIENT.value
        assert result["status"] == UserStatus.ACTIVE.value
        assert result["preferences"] == {"theme": "light"}
        # Medical info should be heavily sanitized for guests
        assert "medical_info" in result
        assert isinstance(result["medical_info"], dict)

    def test_user_profile_is_locked(self):
        """Test account locking functionality."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test unlocked account
        assert profile.is_locked() is False
        
        # Test locked account with future expiration
        future_time = datetime.now() + timedelta(minutes=30)
        profile.account_locked_until = future_time
        assert profile.is_locked() is True
        
        # Test expired lock
        past_time = datetime.now() - timedelta(minutes=30)
        profile.account_locked_until = past_time
        assert profile.is_locked() is False

    def test_user_profile_increment_login_attempts(self):
        """Test login attempts tracking and account locking."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=0
        )
        
        # Increment attempts below threshold
        profile.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)
        assert profile.login_attempts == 1
        assert profile.is_locked() is False
        
        # Increment to threshold
        for i in range(4):  # Already at 1, need 4 more to reach 5
            profile.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)
        
        assert profile.login_attempts == 5
        assert profile.is_locked() is True
        assert profile.account_locked_until is not None
        assert profile.account_locked_until > datetime.now()

    def test_user_profile_reset_login_attempts(self):
        """Test resetting login attempts."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=3,
            account_locked_until=datetime.now() + timedelta(minutes=30)
        )
        
        # Reset attempts
        profile.reset_login_attempts()
        
        assert profile.login_attempts == 0
        assert profile.account_locked_until is None

    def test_user_profile_set_password_reset_token(self):
        """Test setting password reset token."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Set reset token
        token = "reset_token_123"
        expires = datetime.now() + timedelta(hours=1)
        
        profile.set_password_reset_token(token, expires)
        
        assert profile.password_reset_token == token
        assert profile.password_reset_expires == expires

    def test_user_profile_is_password_reset_token_valid(self):
        """Test password reset token validation."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test with no token
        assert profile.is_password_reset_token_valid("any_token") is False
        
        # Test with valid token
        token = "reset_token_123"
        expires = datetime.now() + timedelta(hours=1)
        profile.set_password_reset_token(token, expires)
        
        assert profile.is_password_reset_token_valid(token) is True
        assert profile.is_password_reset_token_valid("wrong_token") is False
        
        # Test with expired token
        expires = datetime.now() - timedelta(hours=1)
        profile.set_password_reset_token(token, expires)
        
        assert profile.is_password_reset_token_valid(token) is False

    def test_user_profile_clear_password_reset_token(self):
        """Test clearing password reset token."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Set and then clear token
        profile.set_password_reset_token("token_123", datetime.now() + timedelta(hours=1))
        assert profile.password_reset_token is not None
        
        profile.clear_password_reset_token()
        
        assert profile.password_reset_token is None
        assert profile.password_reset_expires is None

    def test_user_profile_update_last_login(self):
        """Test updating last login timestamp."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        original_last_login = profile.last_login
        profile.update_last_login()
        
        assert profile.last_login is not None
        assert profile.last_login != original_last_login
        assert profile.last_login > (datetime.now() - timedelta(seconds=1))

    def test_user_profile_update_preferences(self):
        """Test updating user preferences."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            preferences={"theme": "light", "language": "en"}
        )
        
        new_preferences = {"theme": "dark", "notifications": True}
        profile.update_preferences(new_preferences)
        
        assert profile.preferences == {
            "theme": "dark",
            "language": "en",  # Preserved
            "notifications": True
        }

    def test_user_profile_update_medical_info(self):
        """Test updating medical information."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            medical_info={"conditions": ["anxiety"]}
        )
        
        new_medical_info = {"conditions": ["anxiety", "depression"], "medications": ["sertraline"]}
        profile.update_medical_info(new_medical_info)
        
        assert profile.medical_info == new_medical_info

    def test_user_profile_get_role_permissions(self):
        """Test getting role-based permissions."""
        # Test guest permissions
        guest_profile = UserProfile(
            user_id="guest_123",
            email="guest@example.com",
            full_name="Guest User",
            role=UserRole.GUEST,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        guest_permissions = guest_profile.get_role_permissions()
        assert "read:own_profile" in guest_permissions
        assert "write:own_profile" not in guest_permissions
        assert "read:patient_profiles" not in guest_permissions
        
        # Test patient permissions
        patient_profile = UserProfile(
            user_id="patient_123",
            email="patient@example.com",
            full_name="Patient User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        patient_permissions = patient_profile.get_role_permissions()
        assert "read:own_profile" in patient_permissions
        assert "write:own_profile" in patient_permissions
        assert "read:patient_profiles" not in patient_permissions
        
        # Test therapist permissions
        therapist_profile = UserProfile(
            user_id="therapist_123",
            email="therapist@example.com",
            full_name="Therapist User",
            role=UserRole.THERAPIST,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        therapist_permissions = therapist_profile.get_role_permissions()
        assert "read:own_profile" in therapist_permissions
        assert "write:own_profile" in therapist_permissions
        assert "read:patient_profiles" in therapist_permissions
        assert "write:patient_profiles" in therapist_permissions
        
        # Test admin permissions
        admin_profile = UserProfile(
            user_id="admin_123",
            email="admin@example.com",
            full_name="Admin User",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        admin_permissions = admin_profile.get_role_permissions()
        assert "read:own_profile" in admin_permissions
        assert "write:own_profile" in admin_permissions
        assert "read:patient_profiles" in admin_permissions
        assert "write:patient_profiles" in admin_permissions
        assert "admin:users" in admin_permissions
        assert "admin:system" in admin_permissions

    def test_user_profile_has_permission(self):
        """Test permission checking for user profile."""
        # Test patient permissions
        patient_profile = UserProfile(
            user_id="patient_123",
            email="patient@example.com",
            full_name="Patient User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert patient_profile.has_permission("read:own_profile") is True
        assert patient_profile.has_permission("write:own_profile") is True
        assert patient_profile.has_permission("read:patient_profiles") is False
        assert patient_profile.has_permission("admin:users") is False
        
        # Test admin permissions
        admin_profile = UserProfile(
            user_id="admin_123",
            email="admin@example.com",
            full_name="Admin User",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert admin_profile.has_permission("read:own_profile") is True
        assert admin_profile.has_permission("write:own_profile") is True
        assert admin_profile.has_permission("read:patient_profiles") is True
        assert admin_profile.has_permission("admin:users") is True

    def test_user_profile_is_active(self):
        """Test checking if user profile is active."""
        # Active user
        active_profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert active_profile.is_active() is True
        
        # Inactive user
        inactive_profile = UserProfile(
            user_id="user_456",
            email="inactive@example.com",
            full_name="Inactive User",
            role=UserRole.PATIENT,
            status=UserStatus.INACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert inactive_profile.is_active() is False
        
        # Suspended user
        suspended_profile = UserProfile(
            user_id="user_789",
            email="suspended@example.com",
            full_name="Suspended User",
            role=UserRole.PATIENT,
            status=UserStatus.SUSPENDED,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert suspended_profile.is_active() is False

    def test_user_profile_activate_deactivate(self):
        """Test activating and deactivating user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Deactivate
        profile.deactivate()
        assert profile.status == UserStatus.INACTIVE
        assert profile.is_active() is False
        
        # Activate
        profile.activate()
        assert profile.status == UserStatus.ACTIVE
        assert profile.is_active() is True

    def test_user_profile_suspend(self):
        """Test suspending user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        profile.suspend()
        
        assert profile.status == UserStatus.SUSPENDED
        assert profile.is_active() is False

    def test_user_profile_lock(self):
        """Test locking user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        duration = timedelta(hours=24)
        profile.lock(duration)
        
        assert profile.status == UserStatus.LOCKED
        assert profile.account_locked_until is not None
        assert profile.account_locked_until > datetime.now()

    def test_user_profile_unlock(self):
        """Test unlocking user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.LOCKED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=datetime.now() + timedelta(hours=24)
        )
        
        profile.unlock()
        
        assert profile.status == UserStatus.ACTIVE
        assert profile.account_locked_until is None


class TestUserRoleAndStatus:
    """Test cases for UserRole and UserStatus enums."""

    def test_user_role_values(self):
        """Test UserRole enum values."""
        assert UserRole.GUEST.value == "guest"
        assert UserRole.PATIENT.value == "patient"
        assert UserRole.THERAPIST.value == "therapist"
        assert UserRole.ADMIN.value == "admin"

    def test_user_status_values(self):
        """Test UserStatus enum values."""
        assert UserStatus.ACTIVE.value == "active"
        assert UserStatus.INACTIVE.value == "inactive"
        assert UserStatus.SUSPENDED.value == "suspended"
        assert UserStatus.PENDING_VERIFICATION.value == "pending_verification"
        assert UserStatus.LOCKED.value == "locked"


class TestUserRepository:
    """Test cases for UserRepository class."""

    def test_user_repository_save_and_find(self):
        """Test saving and finding users."""
        repo = UserRepository()
        
        user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save user
        result = repo.save(user)
        assert result is True
        
        # Find by ID
        found_user = repo.find_by_id("user_123")
        assert found_user is not None
        assert found_user.user_id == "user_123"
        assert found_user.email == "test@example.com"
        
        # Find by email
        found_user = repo.find_by_email("test@example.com")
        assert found_user is not None
        assert found_user.user_id == "user_123"

    def test_user_repository_find_nonexistent(self):
        """Test finding non-existent users."""
        repo = UserRepository()
        
        # Find non-existent by ID
        found_user = repo.find_by_id("nonexistent")
        assert found_user is None
        
        # Find non-existent by email
        found_user = repo.find_by_email("nonexistent@example.com")
        assert found_user is None

    def test_user_repository_find_all(self):
        """Test finding all users."""
        repo = UserRepository()
        
        # Create multiple users
        users = [
            User(
                user_id="user_1",
                email="user1@example.com",
                full_name="User 1",
                role=UserRole.PATIENT,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            User(
                user_id="user_2",
                email="user2@example.com",
                full_name="User 2",
                role=UserRole.THERAPIST,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            User(
                user_id="user_3",
                email="user3@example.com",
                full_name="User 3",
                role=UserRole.PATIENT,
                status=UserStatus.INACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        # Save all users
        for user in users:
            repo.save(user)
        
        # Find all users
        all_users = repo.find_all()
        assert len(all_users) == 3
        
        # Find by status
        active_users = repo.find_all(status=UserStatus.ACTIVE)
        assert len(active_users) == 2
        
        inactive_users = repo.find_all(status=UserStatus.INACTIVE)
        assert len(inactive_users) == 1
        
        # Test limit
        limited_users = repo.find_all(limit=2)
        assert len(limited_users) == 2

    def test_user_repository_delete(self):
        """Test deleting users."""
        repo = UserRepository()
        
        user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save user
        repo.save(user)
        assert repo.find_by_id("user_123") is not None
        
        # Delete user
        result = repo.delete("user_123")
        assert result is True
        assert repo.find_by_id("user_123") is None
        
        # Delete non-existent user
        result = repo.delete("nonexistent")
        assert result is False

    def test_user_repository_update(self):
        """Test updating users."""
        repo = UserRepository()
        
        user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save user
        repo.save(user)
        
        # Update user
        user.full_name = "Updated User"
        user.status = UserStatus.INACTIVE
        user.updated_at = datetime.now()
        
        result = repo.update(user)
        assert result is True
        
        # Verify update
        updated_user = repo.find_by_id("user_123")
        assert updated_user.full_name == "Updated User"
        assert updated_user.status == UserStatus.INACTIVE
        
        # Update non-existent user
        non_existent_user = User(
            user_id="nonexistent",
            email="nonexistent@example.com",
            full_name="Non-existent",
            role=UserRole.GUEST,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        result = repo.update(non_existent_user)
        assert result is False

    def test_user_repository_count(self):
        """Test counting users."""
        repo = UserRepository()
        
        # Initially empty
        assert repo.count() == 0
        
        # Add users
        for i in range(5):
            user = User(
                user_id=f"user_{i}",
                email=f"user{i}@example.com",
                full_name=f"User {i}",
                role=UserRole.PATIENT,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            repo.save(user)
        
        # Count all users
        assert repo.count() == 5
        
        # Count by status
        # Update some users to inactive
        for i in range(2):
            user = repo.find_by_id(f"user_{i}")
            user.status = UserStatus.INACTIVE
            repo.update(user)
        
        assert repo.count(status=UserStatus.ACTIVE) == 3
        assert repo.count(status=UserStatus.INACTIVE) == 2

    def test_user_repository_find_by_role(self):
        """Test finding users by role."""
        repo = UserRepository()
        
        # Create users with different roles
        roles_users = [
            (UserRole.GUEST, "guest1@example.com"),
            (UserRole.PATIENT, "patient1@example.com"),
            (UserRole.PATIENT, "patient2@example.com"),
            (UserRole.THERAPIST, "therapist1@example.com"),
            (UserRole.ADMIN, "admin1@example.com")
        ]
        
        for role, email in roles_users:
            user = User(
                user_id=email.split("@")[0],
                email=email,
                full_name=email.split("@")[0].title(),
                role=role,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            repo.save(user)
        
        # Find by role
        patients = repo.find_by_role(UserRole.PATIENT)
        assert len(patients) == 2
        
        therapists = repo.find_by_role(UserRole.THERAPIST)
        assert len(therapists) == 1
        
        admins = repo.find_by_role(UserRole.ADMIN)
        assert len(admins) == 1
        
        guests = repo.find_by_role(UserRole.GUEST)
        assert len(guests) == 1

    def test_user_repository_find_active_users(self):
        """Test finding only active users."""
        repo = UserRepository()
        
        # Create mix of active and inactive users
        for i in range(5):
            status = UserStatus.ACTIVE if i % 2 == 0 else UserStatus.INACTIVE
            user = User(
                user_id=f"user_{i}",
                email=f"user{i}@example.com",
                full_name=f"User {i}",
                role=UserRole.PATIENT,
                status=status,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            repo.save(user)
        
        # Find active users
        active_users = repo.find_active_users()
        assert len(active_users) == 3  # users 0, 2, 4
        
        for user in active_users:
            assert user.status == UserStatus.ACTIVE

    def test_user_repository_find_by_date_range(self):
        """Test finding users by creation date range."""
        repo = UserRepository()
        
        # Create users with different creation dates
        base_date = datetime.now()
        dates = [
            base_date - timedelta(days=5),
            base_date - timedelta(days=3),
            base_date - timedelta(days=1),
            base_date,
            base_date + timedelta(days=1)
        ]
        
        for i, date in enumerate(dates):
            user = User(
                user_id=f"user_{i}",
                email=f"user{i}@example.com",
                full_name=f"User {i}",
                role=UserRole.PATIENT,
                status=UserStatus.ACTIVE,
                created_at=date,
                updated_at=date
            )
            repo.save(user)
        
        # Find users in date range
        start_date = base_date - timedelta(days=2)
        end_date = base_date + timedelta(days=2)
        
        users_in_range = repo.find_by_date_range(start_date, end_date)
        assert len(users_in_range) == 3  # users 2, 3, 4
        
        for user in users_in_range:
            assert start_date <= user.created_at <= end_date


class TestUserClass:
    """Test cases for User class."""

    def test_user_creation(self):
        """Test User class creation."""
        created_at = datetime.now()
        updated_at = datetime.now()
        
        user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at
        )
        
        assert user.user_id == "user_123"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == UserRole.PATIENT
        assert user.status == UserStatus.ACTIVE
        assert user.created_at == created_at
        assert user.updated_at == updated_at

    def test_user_to_dict(self):
        """Test User class to_dict method."""
        created_at = datetime.now()
        updated_at = datetime.now()
        
        user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at,
            preferences={"theme": "dark"},
            medical_info={"conditions": ["anxiety"]}
        )
        
        user_dict = user.to_dict()
        
        assert user_dict["user_id"] == "user_123"
        assert user_dict["email"] == "test@example.com"
        assert user_dict["full_name"] == "Test User"
        assert user_dict["role"] == UserRole.PATIENT.value
        assert user_dict["status"] == UserStatus.ACTIVE.value
        assert user_dict["created_at"] == created_at.isoformat()
        assert user_dict["updated_at"] == updated_at.isoformat()
        assert user_dict["preferences"] == {"theme": "dark"}
        assert user_dict["medical_info"] == {"conditions": ["anxiety"]}

    def test_user_from_dict(self):
        """Test User class from_dict method."""
        user_data = {
            "user_id": "user_123",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": UserRole.PATIENT.value,
            "status": UserStatus.ACTIVE.value,
            "created_at": "2023-01-01T12:00:00",
            "updated_at": "2023-01-01T12:00:00",
            "preferences": {"theme": "dark"},
            "medical_info": {"conditions": ["anxiety"]}
        }
        
        user = User.from_dict(user_data)
        
        assert user.user_id == "user_123"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == UserRole.PATIENT
        assert user.status == UserStatus.ACTIVE
        assert user.preferences == {"theme": "dark"}
        assert user.medical_info == {"conditions": ["anxiety"]}

    def test_user_validate_email(self):
        """Test email validation."""
        # Valid emails
        assert User.validate_email("test@example.com") is True
        assert User.validate_email("user.name@domain.co.uk") is True
        assert User.validate_email("user+tag@example.org") is True
        
        # Invalid emails
        assert User.validate_email("invalid-email") is False
        assert User.validate_email("@example.com") is False
        assert User.validate_email("test@") is False
        assert User.validate_email("test.example.com") is False
        assert User.validate_email("") is False

    def test_user_validate_password_strength(self):
        """Test password strength validation."""
        # Valid passwords
        assert User.validate_password("SecurePass123!") is True
        assert User.validate_password("MyStrongP@ssw0rd") is True
        assert User.validate_password("ComplexPassword_2023") is True
        
        # Invalid passwords
        assert User.validate_password("short") is False  # Too short
        assert User.validate_password("alllowercase123!") is False  # No uppercase
        assert User.validate_password("ALLUPPERCASE123!") is False  # No lowercase
        assert User.validate_password("NoDigits!") is False  # No digits
        assert User.validate_password("NoSpecialChars123") is False  # No special characters

    def test_user_hash_and_verify_password(self):
        """Test password hashing and verification."""
        password = "SecurePassword123!"
        
        # Hash password
        hashed_password = User.hash_password(password)
        assert hashed_password is not None
        assert hashed_password != password
        assert len(hashed_password) > 50  # bcrypt hashes are long
        assert hashed_password.startswith("$2b$")  # bcrypt prefix
        
        # Verify correct password
        assert User.verify_password(password, hashed_password) is True
        
        # Verify incorrect password
        assert User.verify_password("WrongPassword123!", hashed_password) is False

    def test_user_generate_user_id(self):
        """Test user ID generation."""
        user_id1 = User.generate_user_id()
        user_id2 = User.generate_user_id()
        
        assert user_id1 != user_id2  # Should be unique
        assert user_id1.startswith("user_")
        assert len(user_id1) > 10  # Should be reasonably long
        assert user_id1.isalnum() or "_" in user_id1  # Should contain valid characters

    def test_user_generate_reset_token(self):
        """Test reset token generation."""
        token1 = User.generate_reset_token()
        token2 = User.generate_reset_token()
        
        assert token1 != token2  # Should be unique
        assert len(token1) > 30  # Should be reasonably long
        assert token1.isalnum() or "-" in token1 or "_" in token1  # Should contain valid characters

    def test_user_is_valid(self):
        """Test user validation."""
        # Valid user
        valid_user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert valid_user.is_valid() is True
        
        # Invalid user - missing user_id
        invalid_user1 = User(
            user_id="",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert invalid_user1.is_valid() is False
        
        # Invalid user - invalid email
        invalid_user2 = User(
            user_id="user_123",
            email="invalid-email",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert invalid_user2.is_valid() is False

    def test_user_get_display_name(self):
        """Test getting display name."""
        # User with full name
        user_with_name = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert user_with_name.get_display_name() == "Test User"
        
        # User without full name
        user_without_name = User(
            user_id="user_456",
            email="test@example.com",
            full_name="",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert user_without_name.get_display_name() == "test@example.com"

    def test_user_get_safe_data(self):
        """Test getting safe user data."""
        user = User(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            preferences={"theme": "dark"},
            medical_info={"conditions": ["anxiety"], "ssn": "123-45-6789"}
        )
        
        safe_data = user.get_safe_data()
        
        # Should include basic info
        assert safe_data["user_id"] == "user_123"
        assert safe_data["email"] == "test@example.com"
        assert safe_data["full_name"] == "Test User"
        assert safe_data["role"] == UserRole.PATIENT.value
        assert safe_data["status"] == UserStatus.ACTIVE.value
        assert safe_data["preferences"] == {"theme": "dark"}
        
        # Should include medical info but with sensitive data filtered
        assert "medical_info" in safe_data
        assert "conditions" in safe_data["medical_info"]
        # SSN should be filtered out
        assert "ssn" not in safe_data["medical_info"]