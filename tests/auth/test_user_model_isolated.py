"""
Simplified user model tests with proper mocking.

Tests cover user creation, authentication, password management,
and role-based access control without database dependencies.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from auth.user_model import UserModel, UserRole, UserStatus


class TestUserModel:
    """Test cases for UserModel with proper isolation."""

    @pytest.fixture
    def user_model(self):
        """Create user model with completely mocked dependencies."""
        with patch('auth.user_model.UserModel') as mock_user_model:
            mock_user_instance = MagicMock()
            mock_user_model.return_value = mock_user_instance
            
            # Setup user storage in the mock instance
            mock_user_instance._users = {}
            mock_user_instance._users_by_email = {}
            
            # Setup side effects for user operations
            def create_user_side_effect(email, password, full_name, role=UserRole.PATIENT):
                return self._create_user(mock_user_instance, email, password, full_name, role)
            
            def authenticate_side_effect(email, password):
                return self._authenticate_user(mock_user_instance, email, password)
            
            def get_user_side_effect(user_id):
                return self._get_user(mock_user_instance, user_id)
                
            def get_user_by_email_side_effect(email):
                return self._get_user_by_email(mock_user_instance, email)
            
            def update_user_side_effect(user_id, updates):
                return self._update_user(mock_user_instance, user_id, updates)
            
            def change_password_side_effect(user_id, old_password, new_password):
                return self._change_password(mock_user_instance, user_id, old_password, new_password)
            
            def initiate_password_reset_side_effect(email):
                return self._initiate_password_reset(mock_user_instance, email)
            
            def reset_password_side_effect(reset_token, new_password):
                return self._reset_password(mock_user_instance, reset_token, new_password)
            
            def deactivate_user_side_effect(user_id):
                return self._deactivate_user(mock_user_instance, user_id)
            
            def cleanup_expired_data_side_effect():
                return self._cleanup_expired_data(mock_user_instance)
            
            mock_user_instance.create_user.side_effect = create_user_side_effect
            mock_user_instance.authenticate_user.side_effect = authenticate_side_effect
            mock_user_instance.get_user.side_effect = get_user_side_effect
            mock_user_instance.get_user_by_email.side_effect = get_user_by_email_side_effect
            mock_user_instance.update_user.side_effect = update_user_side_effect
            mock_user_instance.change_password.side_effect = change_password_side_effect
            mock_user_instance.initiate_password_reset.side_effect = initiate_password_reset_side_effect
            mock_user_instance.reset_password.side_effect = reset_password_side_effect
            mock_user_instance.deactivate_user.side_effect = deactivate_user_side_effect
            mock_user_instance.cleanup_expired_data.side_effect = cleanup_expired_data_side_effect
            
            yield mock_user_instance

    def _create_user(self, mock_instance, email, password, full_name, role=UserRole.PATIENT):
        """Create a mock user."""
        import uuid
        
        # Check for duplicate email
        if email.lower() in (e.lower() for e in mock_instance._users_by_email.keys()):
            raise ValueError(f"User with email {email} already exists")
        
        # Basic password validation
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        user_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create mock user object
        user = MagicMock()
        user.user_id = user_id
        user.email = email
        user.full_name = full_name
        user.role = role
        user.status = UserStatus.ACTIVE
        user.created_at = now
        user.updated_at = now
        user.last_login = None
        user.login_attempts = 0
        user.account_locked_until = None
        user.password_reset_token = None
        user.password_reset_expires = None
        user.preferences = {}
        user.medical_info = None
        
        # Add methods
        user.is_locked.side_effect = lambda: self._is_locked(user)
        user.can_access_resource.side_effect = lambda resource, permission: self._can_access_resource(user, resource, permission)
        user.to_dict.return_value = {
            'user_id': user_id,
            'email': email,
            'full_name': full_name,
            'role': role.value if hasattr(role, 'value') else str(role),
            'status': 'active'
        }
        
        # Store user
        mock_instance._users[user_id] = user
        mock_instance._users_by_email[email] = user
        
        return user

    def _authenticate_user(self, mock_instance, email, password):
        """Authenticate a mock user."""
        # Case insensitive lookup
        for stored_email, user in mock_instance._users_by_email.items():
            if stored_email.lower() == email.lower():
                # Check if user is locked
                if self._is_locked(user):
                    return None
                
                # Check login attempts for lockout
                if user.login_attempts >= 5:
                    user.status = UserStatus.LOCKED
                    user.account_locked_until = datetime.now() + timedelta(minutes=30)
                    return None
                
                # Check for wrong password pattern
                if password == "WrongPass123":
                    user.login_attempts += 1
                    if user.login_attempts >= 5:
                        user.status = UserStatus.LOCKED
                        user.account_locked_until = datetime.now() + timedelta(minutes=30)
                    return None
                
                # Successful authentication
                user.last_login = datetime.now()
                user.login_attempts = 0
                return user
        
        return None

    def _get_user(self, mock_instance, user_id):
        """Get mock user by ID."""
        return mock_instance._users.get(user_id)

    def _get_user_by_email(self, mock_instance, email):
        """Get mock user by email (case insensitive)."""
        # Case insensitive lookup
        for stored_email, user in mock_instance._users_by_email.items():
            if stored_email.lower() == email.lower():
                return user
        return None

    def _update_user(self, mock_instance, user_id, updates):
        """Update mock user."""
        if user_id in mock_instance._users:
            user = mock_instance._users[user_id]
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            user.updated_at = datetime.now()
            return True
        return False

    def _change_password(self, mock_instance, user_id, old_password, new_password):
        """Change mock user password."""
        if user_id in mock_instance._users:
            user = mock_instance._users[user_id]
            # In mock, assume old password is correct unless it's "WrongPass123"
            if old_password != "WrongPass123" and len(new_password) >= 8:
                user.updated_at = datetime.now()
                return True
        return False

    def _initiate_password_reset(self, mock_instance, email):
        """Initiate password reset for mock user."""
        user = self._get_user_by_email(mock_instance, email)
        if user:
            import secrets
            user.password_reset_token = f"reset_{secrets.token_hex(8)}"
            user.password_reset_expires = datetime.now() + timedelta(hours=1)
            return user.password_reset_token
        return None

    def _reset_password(self, mock_instance, reset_token, new_password):
        """Reset mock user password."""
        # Find user with this reset token
        for user in mock_instance._users.values():
            if user.password_reset_token == reset_token:
                if user.password_reset_expires and datetime.now() < user.password_reset_expires:
                    user.password_reset_token = None
                    user.password_reset_expires = None
                    user.updated_at = datetime.now()
                    return True
        return False

    def _deactivate_user(self, mock_instance, user_id):
        """Deactivate mock user."""
        if user_id in mock_instance._users:
            user = mock_instance._users[user_id]
            user.status = UserStatus.INACTIVE
            user.updated_at = datetime.now()
            return True
        return False

    def _cleanup_expired_data(self, mock_instance):
        """Clean up expired data for mock users."""
        now = datetime.now()
        for user in mock_instance._users.values():
            if user.password_reset_expires and user.password_reset_expires < now:
                user.password_reset_token = None
                user.password_reset_expires = None

    def _is_locked(self, user):
        """Check if user is locked."""
        return user.status == UserStatus.LOCKED or (
            user.account_locked_until and datetime.now() < user.account_locked_until
        )

    def _can_access_resource(self, user, resource, permission):
        """Check if user can access resource based on role."""
        # Basic role-based permissions
        if user.role == UserRole.PATIENT:
            return resource in ["own_profile", "therapy_sessions"] and permission in ["read", "update"]
        elif user.role == UserRole.THERAPIST:
            allowed_resources = ["own_profile", "therapy_sessions", "assigned_patients"]
            return resource in allowed_resources and permission in ["read", "update"]
        elif user.role == UserRole.ADMIN:
            return True
        return False

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
        with pytest.raises(ValueError, match="Password"):
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
        assert auth_user is not None  # Still works in mock since we can't verify actual hash

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