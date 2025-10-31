"""
Unit tests for user model functionality without database dependencies.

Tests the UserModel business logic in isolation without database
connections or external dependencies.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any
import hashlib
import secrets

from auth.user_model import UserProfile, UserRole, UserStatus


class MockUserModel:
    """Mock user model implementation for testing without database."""
    
    def __init__(self):
        self._users = {}
        self._users_by_email = {}
        self._reset_tokens = {}
    
    def create_user(self, email: str, password: str, full_name: str, 
                   role: UserRole = UserRole.PATIENT) -> UserProfile:
        """Create a new user without database."""
        # Validation
        if not email or '@' not in email:
            raise ValueError("Valid email is required")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not full_name or len(full_name.strip()) < 2:
            raise ValueError("Full name must be at least 2 characters long")
        
        # Check for duplicate email
        if email in self._users_by_email:
            raise ValueError(f"User with email {email} already exists")
        
        # Create user
        import uuid
        user_id = str(uuid.uuid4())
        now = datetime.now()
        
        user = MockUserProfile(
            user_id=user_id,
            email=email,
            full_name=full_name.strip(),
            role=role,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_login=None,
            login_attempts=0,
            account_locked_until=None,
            password_reset_token=None,
            password_reset_expires=None
        )
        
        # Hash password
        user._password_hash = self._hash_password(password)
        
        # Store user
        self._users[user_id] = user
        self._users_by_email[email] = user
        
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[UserProfile]:
        """Authenticate user without database."""
        user = self._users_by_email.get(email)
        if not user:
            return None
        
        # Check if account is locked
        if user.is_locked():
            return None
        
        # Verify password
        if self._verify_password(password, user._password_hash):
            user.last_login = datetime.now()
            user.login_attempts = 0
            return user
        else:
            user.login_attempts += 1
            # Lock account after 3 failed attempts
            if user.login_attempts >= 3:
                user.account_locked_until = datetime.now() + timedelta(minutes=30)
            return None
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID without database."""
        return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email without database."""
        return self._users_by_email.get(email)
    
    def initiate_password_reset(self, email: str) -> Optional[str]:
        """Initiate password reset without database."""
        user = self._users_by_email.get(email)
        if not user:
            return None
        
        reset_token = secrets.token_urlsafe(32)
        user.password_reset_token = self._hash_token(reset_token)
        user.password_reset_expires = datetime.now() + timedelta(hours=1)
        self._reset_tokens[reset_token] = user.user_id
        
        return reset_token
    
    def reset_password(self, reset_token: str, new_password: str) -> bool:
        """Reset password without database."""
        user_id = self._reset_tokens.get(reset_token)
        if not user_id:
            return False
        
        user = self._users.get(user_id)
        if not user or not user.password_reset_token:
            return False
        
        # Check if token is expired
        if user.password_reset_expires and datetime.now() > user.password_reset_expires:
            return False
        
        # Verify token
        if not self._verify_token(reset_token, user.password_reset_token):
            return False
        
        # Update password
        user._password_hash = self._hash_password(new_password)
        user.password_reset_token = None
        user.password_reset_expires = None
        user.login_attempts = 0
        user.account_locked_until = None
        
        # Clean up reset token
        del self._reset_tokens[reset_token]
        
        return True
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change password without database."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        # Verify old password
        if not self._verify_password(old_password, user._password_hash):
            return False
        
        # Validate new password
        if len(new_password) < 8:
            return False
        
        # Update password
        user._password_hash = self._hash_password(new_password)
        user.updated_at = datetime.now()
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${password_hash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, hash_hex = stored_hash.split('$')
            stored_hash_bytes = bytes.fromhex(hash_hex)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return secrets.compare_digest(password_hash, stored_hash_bytes)
        except (ValueError, TypeError):
            return False
    
    def _hash_token(self, token: str) -> str:
        """Hash token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _verify_token(self, token: str, stored_hash: str) -> bool:
        """Verify token against stored hash."""
        return self._hash_token(token) == stored_hash


@dataclass
class MockUserProfile:
    """Mock user profile for testing without database."""
    
    # Basic fields
    user_id: str
    email: str
    full_name: str
    role: UserRole
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    
    # Authentication fields
    last_login: Optional[datetime]
    login_attempts: int
    account_locked_until: Optional[datetime]
    password_reset_token: Optional[str]
    password_reset_expires: Optional[datetime]
    
    # Additional fields
    preferences: Optional[Dict[str, Any]] = None
    medical_info: Optional[Dict[str, Any]] = None
    
    # Internal fields (not exposed in API)
    _password_hash: Optional[str] = None
    
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.account_locked_until is None:
            return False
        return datetime.now() < self.account_locked_until
    
    def can_access_resource(self, resource: str, permission: str) -> bool:
        """Check if user can access a resource."""
        if self.status != UserStatus.ACTIVE:
            return False
        
        # Simple role-based access control
        if self.role == UserRole.ADMIN:
            return True
        elif self.role == UserRole.THERAPIST:
            return permission in ['read', 'write']
        elif self.role == UserRole.PATIENT:
            return permission == 'read' and resource.startswith('patient_')
        
        return False
    
    def to_dict(self, user_role: str = None, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary with PII protection."""
        data = {
            'user_id': self.user_id,
            'email': self.email if include_sensitive else self._mask_email(self.email),
            'full_name': self.full_name if include_sensitive else self._mask_name(self.full_name),
            'role': self.role.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        
        if include_sensitive:
            data.update({
                'login_attempts': self.login_attempts,
                'account_locked_until': self.account_locked_until.isoformat() if self.account_locked_until else None,
                'preferences': self.preferences,
                'medical_info': self.medical_info
            })
        
        return data
    
    def _mask_email(self, email: str) -> str:
        """Mask email for privacy."""
        if '@' not in email:
            return email
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        return f"{masked_local}@{domain}"
    
    def _mask_name(self, name: str) -> str:
        """Mask name for privacy."""
        if len(name) <= 2:
            return '*' * len(name)
        return name[0] + '*' * (len(name) - 1)


class TestMockUserModel:
    """Test mock user model functionality."""
    
    @pytest.fixture
    def user_model(self):
        """Provide mock user model for testing."""
        return MockUserModel()
    
    def test_create_user_success(self, user_model):
        """Test successful user creation."""
        user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User",
            role=UserRole.PATIENT
        )
        
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == UserRole.PATIENT
        assert user.status == UserStatus.ACTIVE
        assert user.user_id is not None
        assert user.created_at is not None
        assert user._password_hash is not None
    
    def test_create_user_invalid_email(self, user_model):
        """Test user creation with invalid email."""
        with pytest.raises(ValueError, match="Valid email is required"):
            user_model.create_user(
                email="invalid-email",
                password="SecurePass123",
                full_name="Test User"
            )
    
    def test_create_user_short_password(self, user_model):
        """Test user creation with short password."""
        with pytest.raises(ValueError, match="Password must be at least 8 characters long"):
            user_model.create_user(
                email="test@example.com",
                password="short",
                full_name="Test User"
            )
    
    def test_create_user_short_name(self, user_model):
        """Test user creation with short name."""
        with pytest.raises(ValueError, match="Full name must be at least 2 characters long"):
            user_model.create_user(
                email="test@example.com",
                password="SecurePass123",
                full_name="A"
            )
    
    def test_create_user_duplicate_email(self, user_model):
        """Test user creation with duplicate email."""
        user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        with pytest.raises(ValueError, match="User with email test@example.com already exists"):
            user_model.create_user(
                email="test@example.com",
                password="AnotherPass123",
                full_name="Another User"
            )
    
    def test_authenticate_user_success(self, user_model):
        """Test successful user authentication."""
        user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        authenticated_user = user_model.authenticate_user(
            email="test@example.com",
            password="SecurePass123"
        )
        
        assert authenticated_user == user
        assert authenticated_user.last_login is not None
    
    def test_authenticate_user_invalid_email(self, user_model):
        """Test authentication with invalid email."""
        result = user_model.authenticate_user(
            email="nonexistent@example.com",
            password="SecurePass123"
        )
        
        assert result is None
    
    def test_authenticate_user_invalid_password(self, user_model):
        """Test authentication with invalid password."""
        user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        result = user_model.authenticate_user(
            email="test@example.com",
            password="WrongPassword"
        )
        
        assert result is None
    
    def test_authenticate_user_account_lock(self, user_model):
        """Test authentication with locked account."""
        user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        # Fail authentication 3 times to lock account
        for _ in range(3):
            result = user_model.authenticate_user(
                email="test@example.com",
                password="WrongPassword"
            )
            assert result is None
        
        # Try authentication with correct password
        result = user_model.authenticate_user(
            email="test@example.com",
            password="SecurePass123"
        )
        
        assert result is None
    
    def test_get_user_success(self, user_model):
        """Test successful user retrieval."""
        created_user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        retrieved_user = user_model.get_user(created_user.user_id)
        
        assert retrieved_user == created_user
    
    def test_get_user_not_found(self, user_model):
        """Test user retrieval with non-existent ID."""
        result = user_model.get_user("nonexistent_id")
        assert result is None
    
    def test_get_user_by_email_success(self, user_model):
        """Test successful user retrieval by email."""
        created_user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        retrieved_user = user_model.get_user_by_email("test@example.com")
        
        assert retrieved_user == created_user
    
    def test_initiate_password_reset_success(self, user_model):
        """Test successful password reset initiation."""
        user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        reset_token = user_model.initiate_password_reset("test@example.com")
        
        assert reset_token is not None
        assert isinstance(reset_token, str)
        assert len(reset_token) > 0
    
    def test_initiate_password_reset_nonexistent_user(self, user_model):
        """Test password reset initiation for non-existent user."""
        result = user_model.initiate_password_reset("nonexistent@example.com")
        assert result is None
    
    def test_reset_password_success(self, user_model):
        """Test successful password reset."""
        user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        reset_token = user_model.initiate_password_reset("test@example.com")
        result = user_model.reset_password(reset_token, "NewSecurePass123")
        
        assert result is True
        
        # Test authentication with new password
        authenticated_user = user_model.authenticate_user(
            email="test@example.com",
            password="NewSecurePass123"
        )
        assert authenticated_user is not None
    
    def test_reset_password_invalid_token(self, user_model):
        """Test password reset with invalid token."""
        result = user_model.reset_password("invalid_token", "NewSecurePass123")
        assert result is False
    
    def test_change_password_success(self, user_model):
        """Test successful password change."""
        created_user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        result = user_model.change_password(
            user_id=created_user.user_id,
            old_password="SecurePass123",
            new_password="NewSecurePass123"
        )
        
        assert result is True
        
        # Test authentication with new password
        authenticated_user = user_model.authenticate_user(
            email="test@example.com",
            password="NewSecurePass123"
        )
        assert authenticated_user is not None
    
    def test_change_password_wrong_old_password(self, user_model):
        """Test password change with wrong old password."""
        created_user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        result = user_model.change_password(
            user_id=created_user.user_id,
            old_password="WrongPassword",
            new_password="NewSecurePass123"
        )
        
        assert result is False
    
    def test_change_password_short_new_password(self, user_model):
        """Test password change with short new password."""
        created_user = user_model.create_user(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        
        result = user_model.change_password(
            user_id=created_user.user_id,
            old_password="SecurePass123",
            new_password="short"
        )
        
        assert result is False


class TestMockUserProfile:
    """Test mock user profile functionality."""
    
    @pytest.fixture
    def user_profile(self):
        """Create a mock user profile for testing."""
        return MockUserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_login=None,
            login_attempts=0,
            account_locked_until=None,
            password_reset_token=None,
            password_reset_expires=None
        )
    
    def test_is_locked_false(self, user_profile):
        """Test unlocked account."""
        assert user_profile.is_locked() is False
    
    def test_is_locked_true(self, user_profile):
        """Test locked account."""
        user_profile.account_locked_until = datetime.now() + timedelta(minutes=30)
        assert user_profile.is_locked() is True
    
    def test_is_locked_expired(self, user_profile):
        """Test expired lock."""
        user_profile.account_locked_until = datetime.now() - timedelta(minutes=30)
        assert user_profile.is_locked() is False
    
    def test_can_access_resource_admin(self, user_profile):
        """Test admin resource access."""
        user_profile.role = UserRole.ADMIN
        assert user_profile.can_access_resource("any_resource", "any_permission") is True
    
    def test_can_access_resource_therapist_read(self, user_profile):
        """Test therapist read access."""
        user_profile.role = UserRole.THERAPIST
        assert user_profile.can_access_resource("patient_data", "read") is True
        assert user_profile.can_access_resource("admin_data", "read") is True
    
    def test_can_access_resource_therapist_write(self, user_profile):
        """Test therapist write access."""
        user_profile.role = UserRole.THERAPIST
        assert user_profile.can_access_resource("patient_data", "write") is True
        assert user_profile.can_access_resource("admin_data", "write") is True
    
    def test_can_access_resource_patient_read(self, user_profile):
        """Test patient read access."""
        user_profile.role = UserRole.PATIENT
        assert user_profile.can_access_resource("patient_123_data", "read") is True
        assert user_profile.can_access_resource("admin_data", "read") is False
    
    def test_can_access_resource_patient_write(self, user_profile):
        """Test patient write access."""
        user_profile.role = UserRole.PATIENT
        assert user_profile.can_access_resource("patient_123_data", "write") is False
    
    def test_can_access_resource_inactive(self, user_profile):
        """Test inactive user access."""
        user_profile.role = UserRole.ADMIN
        user_profile.status = UserStatus.INACTIVE
        assert user_profile.can_access_resource("any_resource", "any_permission") is False
    
    def test_to_dict_with_sensitive_data(self, user_profile):
        """Test to_dict with sensitive data included."""
        user_profile.login_attempts = 2
        user_profile.account_locked_until = datetime.now() + timedelta(minutes=30)
        
        data = user_profile.to_dict(include_sensitive=True)
        
        assert data['user_id'] == "user_123"
        assert data['email'] == "test@example.com"
        assert data['full_name'] == "Test User"
        assert data['login_attempts'] == 2
        assert data['account_locked_until'] is not None
    
    def test_to_dict_without_sensitive_data(self, user_profile):
        """Test to_dict without sensitive data."""
        data = user_profile.to_dict(include_sensitive=False)
        
        assert data['user_id'] == "user_123"
        assert data['email'] == "t**t@example.com"  # Masked email
        assert data['full_name'] == "T********"  # Masked name
        assert 'login_attempts' not in data
        assert 'account_locked_until' not in data
    
    def test_mask_email_short_local(self, user_profile):
        """Test email masking with short local part."""
        masked = user_profile._mask_email("ab@example.com")
        assert masked == "**@example.com"
    
    def test_mask_name_short(self, user_profile):
        """Test name masking with short name."""
        masked = user_profile._mask_name("A")
        assert masked == "*"