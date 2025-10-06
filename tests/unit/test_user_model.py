"""
Comprehensive unit tests for auth/user_model.py
"""

import pytest
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, ANY
from pathlib import Path
import bcrypt

# Mock the database modules to avoid import issues
with patch.dict('sys.modules', {
    'database.models': Mock(),
    'database.db_manager': Mock()
}):
    from auth.user_model import (
        UserModel, UserProfile, UserRole, UserStatus
    )


class TestUserRole:
    """Test UserRole enum functionality."""
    
    def test_user_role_values(self):
        """Test user role enum values."""
        assert UserRole.PATIENT.value == "patient"
        assert UserRole.THERAPIST.value == "therapist"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.GUEST.value == "guest"
    
    def test_user_role_comparison(self):
        """Test user role comparison."""
        assert UserRole.PATIENT == UserRole.PATIENT
        assert UserRole.PATIENT != UserRole.THERAPIST
        assert UserRole.PATIENT != "patient"


class TestUserStatus:
    """Test UserStatus enum functionality."""
    
    def test_user_status_values(self):
        """Test user status enum values."""
        assert UserStatus.ACTIVE.value == "active"
        assert UserStatus.INACTIVE.value == "inactive"
        assert UserStatus.SUSPENDED.value == "suspended"
        assert UserStatus.PENDING_VERIFICATION.value == "pending_verification"
        assert UserStatus.LOCKED.value == "locked"
    
    def test_user_status_comparison(self):
        """Test user status comparison."""
        assert UserStatus.ACTIVE == UserStatus.ACTIVE
        assert UserStatus.ACTIVE != UserStatus.INACTIVE
        assert UserStatus.ACTIVE != "active"


class TestUserProfile:
    """Test UserProfile functionality."""
    
    def test_user_profile_creation(self):
        """Test creating a user profile."""
        now = datetime.now()
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_login=now + timedelta(hours=1),
            login_attempts=0,
            account_locked_until=None,
            password_reset_token=None,
            password_reset_expires=None,
            preferences={"theme": "light"},
            medical_info={"condition": "anxiety"}
        )
        
        assert profile.user_id == "user_123"
        assert profile.email == "test@example.com"
        assert profile.full_name == "Test User"
        assert profile.role == UserRole.PATIENT
        assert profile.status == UserStatus.ACTIVE
        assert profile.created_at == now
        assert profile.updated_at == now
        assert profile.last_login == now + timedelta(hours=1)
        assert profile.login_attempts == 0
        assert profile.account_locked_until is None
        assert profile.password_reset_token is None
        assert profile.password_reset_expires is None
        assert profile.preferences == {"theme": "light"}
        assert profile.medical_info == {"condition": "anxiety"}
    
    def test_user_profile_to_dict_no_role(self):
        """Test converting profile to dictionary without role filtering."""
        now = datetime.now()
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            password_reset_token="reset_token",
            password_reset_expires=now + timedelta(hours=24),
            preferences={"theme": "light"},
            medical_info={"condition": "anxiety", "medication": "sertraline"}
        )
        
        result = profile.to_dict()
        
        assert isinstance(result, dict)
        assert result['user_id'] == "user_123"
        assert result['email'] == "test@example.com"
        assert result['full_name'] == "Test User"
        assert result['role'] == UserRole.PATIENT
        assert result['status'] == UserStatus.ACTIVE
        assert result['preferences'] == {"theme": "light"}
        assert result['medical_info'] == {"condition": "anxiety", "medication": "sertraline"}
        # Sensitive fields should be removed
        assert 'password_reset_token' not in result
        assert 'password_reset_expires' not in result
    
    def test_user_profile_to_dict_with_patient_role(self):
        """Test converting profile to dictionary with patient role filtering."""
        now = datetime.now()
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            medical_info={
                "condition": "anxiety",
                "medication": "sertraline",
                "treatment_history": ["CBT", "medication"],
                "diagnoses": ["Generalized Anxiety Disorder"]
            }
        )
        
        result = profile.to_dict(user_role='patient')
        
        assert isinstance(result, dict)
        assert result['user_id'] == "user_123"
        # Medical info should be sanitized for patients
        assert 'medical_info' in result
        assert '_sanitized' in result['medical_info']
        assert '_visible_fields' in result['medical_info']
        # Should only show insurance_provider and emergency_contact if present
        assert "condition" not in result['medical_info']
        assert "medication" not in result['medical_info']
    
    def test_user_profile_to_dict_with_therapist_role(self):
        """Test converting profile to dictionary with therapist role filtering."""
        now = datetime.now()
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            medical_info={
                "condition": "anxiety",
                "medication": "sertraline",
                "allergies": ["penicillin"],
                "treatment_history": ["CBT", "medication"],
                "diagnoses": ["Generalized Anxiety Disorder"]
            }
        )
        
        result = profile.to_dict(user_role='therapist')
        
        assert isinstance(result, dict)
        # Medical info should show more for therapists
        assert 'medical_info' in result
        assert "condition" in result['medical_info']
        assert "medication" in result['medical_info']
        assert "allergies" in result['medical_info']
        # But not treatment history or diagnoses unless include_sensitive=True
        assert "treatment_history" not in result['medical_info']
        assert "diagnoses" not in result['medical_info']
    
    def test_user_profile_to_dict_with_admin_role(self):
        """Test converting profile to dictionary with admin role filtering."""
        now = datetime.now()
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            medical_info={
                "condition": "anxiety",
                "medication": "sertraline",
                "allergies": ["penicillin"],
                "treatment_history": ["CBT", "medication"],
                "diagnoses": ["Generalized Anxiety Disorder"]
            }
        )
        
        result = profile.to_dict(user_role='admin')
        
        assert isinstance(result, dict)
        # Medical info should show most for admins
        assert 'medical_info' in result
        assert "condition" in result['medical_info']
        assert "medication" in result['medical_info']
        assert "allergies" in result['medical_info']
        assert "treatment_history" in result['medical_info']
        assert "diagnoses" in result['medical_info']
    
    def test_user_profile_to_dict_with_sensitive_info(self):
        """Test converting profile to dictionary with sensitive info included."""
        now = datetime.now()
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            medical_info={
                "condition": "anxiety",
                "treatment_history": ["CBT", "medication"]
            }
        )
        
        result = profile.to_dict(user_role='therapist', include_sensitive=True)
        
        assert isinstance(result, dict)
        # With include_sensitive=True, therapists should see more
        assert 'medical_info' in result
        assert "condition" in result['medical_info']
        # Still should be sanitized unless it's admin - treatment_history is admin-only
        assert "treatment_history" not in result['medical_info']
        # Should indicate sanitization occurred
        assert result['medical_info'].get('_sanitized') is True
    
    def test_user_profile_to_dict_email_masking(self):
        """Test email masking in to_dict."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test with non-admin role
        result = profile.to_dict(user_role='patient')
        assert result['email'] != "test@example.com"  # Should be masked
        assert "@" in result['email']  # Should still be recognizable as email
        
        # Test with admin role
        result = profile.to_dict(user_role='admin')
        assert result['email'] == "test@example.com"  # Should not be masked
    
    def test_user_profile_is_locked_true(self):
        """Test account lock detection when locked."""
        future_time = datetime.now() + timedelta(hours=1)
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=future_time
        )
        
        assert profile.is_locked() is True
    
    def test_user_profile_is_locked_false(self):
        """Test account lock detection when not locked."""
        past_time = datetime.now() - timedelta(hours=1)
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=past_time
        )
        
        assert profile.is_locked() is False
    
    def test_user_profile_is_locked_none(self):
        """Test account lock detection when never locked."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=None
        )
        
        assert profile.is_locked() is False
    
    def test_user_profile_increment_login_attempts_no_lock(self):
        """Test incrementing login attempts without reaching lock threshold."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=3
        )
        
        profile.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)
        
        assert profile.login_attempts == 4
        assert profile.status == UserStatus.ACTIVE
        assert profile.account_locked_until is None
    
    def test_user_profile_increment_login_attempts_with_lock(self):
        """Test incrementing login attempts reaching lock threshold."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=4
        )
        
        profile.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)
        
        assert profile.login_attempts == 5
        assert profile.status == UserStatus.LOCKED
        assert profile.account_locked_until is not None
        assert profile.account_locked_until > datetime.now()
    
    def test_user_profile_reset_login_attempts(self):
        """Test resetting login attempts."""
        future_time = datetime.now() + timedelta(hours=1)
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.LOCKED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=5,
            account_locked_until=future_time
        )
        
        profile.reset_login_attempts()
        
        assert profile.login_attempts == 0
        assert profile.account_locked_until is None
        assert profile.status == UserStatus.ACTIVE
    
    def test_user_profile_can_access_resource_success(self):
        """Test successful resource access check."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        result = profile.can_access_resource("own_profile", "read")
        
        assert result is True
    
    def test_user_profile_can_access_resource_failure(self):
        """Test failed resource access check."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        result = profile.can_access_resource("admin_panel", "write")
        
        assert result is False
    
    def test_sanitize_medical_info_patient_role(self):
        """Test medical info sanitization for patient role."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        medical_info = {
            "condition": "anxiety",
            "medication": "sertraline",
            "insurance_provider": "Health Insurance Co",
            "emergency_contact": "John Doe - 555-0123"
        }
        
        result = profile._sanitize_medical_info(medical_info, 'patient')
        
        assert "insurance_provider" in result
        assert "emergency_contact" in result
        assert "condition" not in result
        assert "medication" not in result
        assert result["_sanitized"] is True
        assert "_visible_fields" in result
    
    def test_sanitize_medical_info_therapist_role(self):
        """Test medical info sanitization for therapist role."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        medical_info = {
            "condition": "anxiety",
            "medication": "sertraline",
            "allergies": ["penicillin"],
            "insurance_provider": "Health Insurance Co",
            "emergency_contact": "John Doe - 555-0123",
            "treatment_history": ["CBT", "medication"]
        }
        
        result = profile._sanitize_medical_info(medical_info, 'therapist')
        
        assert "condition" in result
        assert "medication" in result
        assert "allergies" in result
        assert "insurance_provider" in result
        assert "emergency_contact" in result
        assert "treatment_history" not in result
        assert result["_sanitized"] is True
    
    def test_sanitize_medical_info_admin_role(self):
        """Test medical info sanitization for admin role."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        medical_info = {
            "condition": "anxiety",
            "medication": "sertraline",
            "allergies": ["penicillin"],
            "treatment_history": ["CBT", "medication"],
            "diagnoses": ["Generalized Anxiety Disorder"]
        }
        
        result = profile._sanitize_medical_info(medical_info, 'admin')
        
        assert "condition" in result
        assert "medication" in result
        assert "allergies" in result
        assert "treatment_history" in result
        assert "diagnoses" in result
        assert "_sanitized" not in result  # Admins see everything
    
    def test_mask_email(self):
        """Test email masking functionality."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test normal email
        masked = profile._mask_email("test@example.com")
        assert masked == "t**t@example.com"
        
        # Test short local part
        masked = profile._mask_email("ab@example.com")
        assert masked == "**@example.com"
        
        # Test single character local part
        masked = profile._mask_email("a@example.com")
        assert masked == "*@example.com"
    
    def test_is_owner_request(self):
        """Test owner request check (always returns False in current implementation)."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        result = profile._is_owner_request('patient')
        assert result is False


class TestUserModel:
    """Test UserModel functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_user_repo(self):
        """Create a mock user repository."""
        return Mock()
    
    @pytest.fixture
    def user_model(self, mock_user_repo, temp_data_dir):
        """Create user model with mocked dependencies."""
        with patch('auth.user_model.UserRepository', return_value=mock_user_repo):
            model = UserModel(data_dir=temp_data_dir)
            model.user_repo = mock_user_repo
            return model
    
    def test_user_model_initialization(self, user_model):
        """Test user model initialization."""
        assert user_model.user_repo is not None
        assert user_model.data_dir is not None
        assert user_model.users_file.name == "users.json"
        assert user_model.passwords_file.name == "passwords.json"
    
    def test_user_model_custom_data_dir(self):
        """Test user model with custom data directory."""
        custom_dir = "/tmp/custom_auth_data"
        with patch('auth.user_model.UserRepository'):
            model = UserModel(data_dir=custom_dir)
            assert str(model.data_dir) == custom_dir
    
    def test_create_user_success(self, user_model, mock_user_repo):
        """Test successful user creation."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        mock_db_user.email = "test@example.com"
        mock_db_user.full_name = "Test User"
        mock_db_user.role = UserRole.PATIENT
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.created_at = datetime.now()
        mock_db_user.updated_at = datetime.now()
        mock_db_user.last_login = None
        mock_db_user.login_attempts = 0
        mock_db_user.account_locked_until = None
        mock_db_user.password_reset_token = None
        mock_db_user.password_reset_expires = None
        mock_db_user.preferences = {}
        mock_db_user.medical_info = {}
        
        with patch('auth.user_model.User.create', return_value=mock_db_user):
            mock_user_repo.save.return_value = True
            user_model.get_user_by_email = Mock(return_value=None)  # No existing user
            
            result = user_model.create_user(
                email="test@example.com",
                password="SecurePass123",
                full_name="Test User",
                role=UserRole.PATIENT
            )
        
        assert isinstance(result, UserProfile)
        assert result.email == "test@example.com"
        assert result.full_name == "Test User"
        assert result.role == UserRole.PATIENT
        assert result.status == UserStatus.ACTIVE
        mock_user_repo.save.assert_called_once()
    
    def test_create_user_invalid_email(self, user_model):
        """Test user creation with invalid email."""
        with pytest.raises(ValueError, match="Invalid email format"):
            user_model.create_user(
                email="invalid-email",
                password="SecurePass123",
                full_name="Test User"
            )
    
    def test_create_user_weak_password(self, user_model):
        """Test user creation with weak password."""
        with pytest.raises(ValueError, match="Password does not meet security requirements"):
            user_model.create_user(
                email="test@example.com",
                password="weak",
                full_name="Test User"
            )
    
    def test_create_user_existing_email(self, user_model):
        """Test user creation with existing email."""
        mock_existing_user = Mock()
        user_model.get_user_by_email = Mock(return_value=mock_existing_user)
        
        with pytest.raises(ValueError, match="User with this email already exists"):
            user_model.create_user(
                email="existing@example.com",
                password="SecurePass123",
                full_name="Test User"
            )
    
    def test_create_user_save_failure(self, user_model, mock_user_repo):
        """Test user creation when save fails."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        
        with patch('auth.user_model.User.create', return_value=mock_db_user):
            mock_user_repo.save.return_value = False
            user_model.get_user_by_email = Mock(return_value=None)  # No existing user
            
            with pytest.raises(ValueError, match="Failed to create user account"):
                user_model.create_user(
                    email="test@example.com",
                    password="SecurePass123",
                    full_name="Test User"
                )
    
    def test_authenticate_user_success(self, user_model, mock_user_repo):
        """Test successful user authentication."""
        mock_db_user = Mock()
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.is_locked.return_value = False
        mock_db_user.password_hash = bcrypt.hashpw("SecurePass123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        mock_db_user.user_id = "user_123"
        mock_db_user.email = "test@example.com"
        mock_db_user.full_name = "Test User"
        mock_db_user.role = UserRole.PATIENT
        mock_db_user.created_at = datetime.now()
        mock_db_user.updated_at = datetime.now()
        mock_db_user.last_login = None
        mock_db_user.login_attempts = 0
        mock_db_user.account_locked_until = None
        mock_db_user.password_reset_token = None
        mock_db_user.password_reset_expires = None
        mock_db_user.preferences = {}
        mock_db_user.medical_info = {}
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        result = user_model.authenticate_user("test@example.com", "SecurePass123")
        
        assert isinstance(result, UserProfile)
        assert result.email == "test@example.com"
        assert result.status == UserStatus.ACTIVE
        mock_db_user.reset_login_attempts.assert_called_once()
        mock_db_user.last_login = ANY
        mock_user_repo.save.assert_called()
    
    def test_authenticate_user_not_found(self, user_model, mock_user_repo):
        """Test authentication when user not found."""
        mock_user_repo.find_by_email.return_value = None
        
        result = user_model.authenticate_user("nonexistent@example.com", "password")
        
        assert result is None
    
    def test_authenticate_user_inactive(self, user_model, mock_user_repo):
        """Test authentication when user is inactive."""
        mock_db_user = Mock()
        mock_db_user.status = UserStatus.INACTIVE
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        
        result = user_model.authenticate_user("inactive@example.com", "password")
        
        assert result is None
    
    def test_authenticate_user_locked(self, user_model, mock_user_repo):
        """Test authentication when user is locked."""
        mock_db_user = Mock()
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.is_locked.return_value = True
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        
        result = user_model.authenticate_user("locked@example.com", "password")
        
        assert result is None
    
    def test_authenticate_user_wrong_password(self, user_model, mock_user_repo):
        """Test authentication with wrong password."""
        mock_db_user = Mock()
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.is_locked.return_value = False
        mock_db_user.password_hash = bcrypt.hashpw("CorrectPassword".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        mock_db_user.login_attempts = 0
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        result = user_model.authenticate_user("test@example.com", "WrongPassword")
        
        assert result is None
        mock_db_user.increment_login_attempts.assert_called_once()
        mock_user_repo.save.assert_called()
    
    def test_get_user_success(self, user_model, mock_user_repo):
        """Test successful user retrieval."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        mock_db_user.email = "test@example.com"
        mock_db_user.full_name = "Test User"
        mock_db_user.role = UserRole.PATIENT
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.created_at = datetime.now()
        mock_db_user.updated_at = datetime.now()
        mock_db_user.last_login = None
        mock_db_user.login_attempts = 0
        mock_db_user.account_locked_until = None
        mock_db_user.password_reset_token = None
        mock_db_user.password_reset_expires = None
        mock_db_user.preferences = {}
        mock_db_user.medical_info = {}
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        
        result = user_model.get_user("user_123")
        
        assert isinstance(result, UserProfile)
        assert result.user_id == "user_123"
        assert result.email == "test@example.com"
    
    def test_get_user_not_found(self, user_model, mock_user_repo):
        """Test user retrieval when not found."""
        mock_user_repo.find_by_id.return_value = None
        
        result = user_model.get_user("nonexistent_user")
        
        assert result is None
    
    def test_get_user_by_email_success(self, user_model, mock_user_repo):
        """Test successful user retrieval by email."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        mock_db_user.email = "test@example.com"
        mock_db_user.full_name = "Test User"
        mock_db_user.role = UserRole.PATIENT
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.created_at = datetime.now()
        mock_db_user.updated_at = datetime.now()
        mock_db_user.last_login = None
        mock_db_user.login_attempts = 0
        mock_db_user.account_locked_until = None
        mock_db_user.password_reset_token = None
        mock_db_user.password_reset_expires = None
        mock_db_user.preferences = {}
        mock_db_user.medical_info = {}
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        
        result = user_model.get_user_by_email("test@example.com")
        
        assert isinstance(result, UserProfile)
        assert result.email == "test@example.com"
    
    def test_get_user_by_email_not_found(self, user_model, mock_user_repo):
        """Test user retrieval by email when not found."""
        mock_user_repo.find_by_email.return_value = None
        
        result = user_model.get_user_by_email("nonexistent@example.com")
        
        assert result is None
    
    def test_update_user_success(self, user_model, mock_user_repo):
        """Test successful user update."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        mock_db_user.full_name = "Old Name"
        mock_db_user.preferences = {"theme": "light"}
        mock_db_user.medical_info = {}
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        result = user_model.update_user("user_123", {
            "full_name": "New Name",
            "preferences": {"theme": "dark"}
        })
        
        assert result is True
        assert mock_db_user.full_name == "New Name"
        assert mock_db_user.preferences == {"theme": "dark"}
        mock_user_repo.save.assert_called_once()
    
    def test_update_user_not_found(self, user_model, mock_user_repo):
        """Test user update when user not found."""
        mock_user_repo.find_by_id.return_value = None
        
        result = user_model.update_user("nonexistent_user", {"full_name": "New Name"})
        
        assert result is False
    
    def test_update_user_disallowed_fields(self, user_model, mock_user_repo):
        """Test user update with disallowed fields."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        mock_db_user.email = "test@example.com"
        mock_db_user.role = UserRole.PATIENT
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        result = user_model.update_user("user_123", {
            "email": "new@example.com",  # Disallowed
            "role": UserRole.ADMIN,      # Disallowed
            "full_name": "New Name"      # Allowed
        })
        
        assert result is True
        assert mock_db_user.email == "test@example.com"  # Should not change
        assert mock_db_user.role == UserRole.PATIENT      # Should not change
        assert mock_db_user.full_name == "New Name"      # Should change
    
    def test_change_password_success(self, user_model, mock_user_repo):
        """Test successful password change."""
        old_hash = bcrypt.hashpw("OldPassword123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        mock_db_user = Mock()
        mock_db_user.password_hash = old_hash
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        result = user_model.change_password("user_123", "OldPassword123", "NewPassword123")
        
        assert result is True
        assert mock_db_user.password_hash != old_hash  # Should be new hash
        mock_user_repo.save.assert_called_once()
    
    def test_change_password_wrong_old_password(self, user_model, mock_user_repo):
        """Test password change with wrong old password."""
        old_hash = bcrypt.hashpw("CorrectPassword".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        mock_db_user = Mock()
        mock_db_user.password_hash = old_hash
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        
        result = user_model.change_password("user_123", "WrongPassword", "NewPassword123")
        
        assert result is False
    
    def test_change_password_weak_new_password(self, user_model, mock_user_repo):
        """Test password change with weak new password."""
        old_hash = bcrypt.hashpw("OldPassword123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        mock_db_user = Mock()
        mock_db_user.password_hash = old_hash
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        
        result = user_model.change_password("user_123", "OldPassword123", "weak")
        
        assert result is False
    
    def test_change_password_user_not_found(self, user_model, mock_user_repo):
        """Test password change when user not found."""
        mock_user_repo.find_by_id.return_value = None
        
        result = user_model.change_password("nonexistent_user", "OldPassword123", "NewPassword123")
        
        assert result is False
    
    def test_initiate_password_reset_success(self, user_model, mock_user_repo):
        """Test successful password reset initiation."""
        mock_db_user = Mock()
        mock_db_user.user_id = "user_123"
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        with patch.object(user_model, '_generate_reset_token', return_value="reset_token_123"):
            result = user_model.initiate_password_reset("test@example.com")
        
        assert result == "reset_token_123"
        assert mock_db_user.password_reset_token == "reset_token_123"
        assert mock_db_user.password_reset_expires is not None
        mock_user_repo.save.assert_called_once()
    
    def test_initiate_password_reset_user_not_found(self, user_model, mock_user_repo):
        """Test password reset initiation when user not found."""
        mock_user_repo.find_by_email.return_value = None
        
        result = user_model.initiate_password_reset("nonexistent@example.com")
        
        assert result is None
    
    def test_reset_password_not_implemented(self, user_model):
        """Test password reset with token (not implemented)."""
        result = user_model.reset_password("reset_token_123", "NewPassword123")
        
        assert result is False
    
    def test_deactivate_user_success(self, user_model, mock_user_repo):
        """Test successful user deactivation."""
        mock_db_user = Mock()
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = True
        
        result = user_model.deactivate_user("user_123")
        
        assert result is True
        assert mock_db_user.status == UserStatus.INACTIVE
        mock_user_repo.save.assert_called_once()
    
    def test_deactivate_user_not_found(self, user_model, mock_user_repo):
        """Test user deactivation when user not found."""
        mock_user_repo.find_by_id.return_value = None
        
        result = user_model.deactivate_user("nonexistent_user")
        
        assert result is False
    
    def test_hash_password(self, user_model):
        """Test password hashing."""
        password = "SecurePass123"
        hashed = user_model._hash_password(password)
        
        assert isinstance(hashed, str)
        assert hashed != password
        assert bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def test_verify_password(self, user_model):
        """Test password verification."""
        password = "SecurePass123"
        hashed = user_model._hash_password(password)
        
        assert user_model._verify_password(password, hashed) is True
        assert user_model._verify_password("WrongPassword", hashed) is False
    
    def test_generate_user_id(self, user_model):
        """Test user ID generation."""
        # Mock the users property to return an empty dict
        user_model._users = {}
        
        user_id1 = user_model._generate_user_id()
        user_id2 = user_model._generate_user_id()
        
        assert isinstance(user_id1, str)
        assert isinstance(user_id2, str)
        assert user_id1 != user_id2
        assert user_id1.startswith("user_")
        assert user_id2.startswith("user_")
    
    def test_generate_reset_token(self, user_model):
        """Test reset token generation."""
        token1 = user_model._generate_reset_token()
        token2 = user_model._generate_reset_token()
        
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert token1 != token2
        assert len(token1) > 20  # Should be reasonably long
    
    def test_validate_email_valid(self, user_model):
        """Test email validation with valid emails."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org",
            "user123@test-domain.com"
        ]
        
        for email in valid_emails:
            assert user_model._validate_email(email) is True
    
    def test_validate_email_invalid(self, user_model):
        """Test email validation with invalid emails."""
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test.example.com",
            "test@.com",
            "test@com.",
            ""
        ]
        
        for email in invalid_emails:
            assert user_model._validate_email(email) is False
    
    def test_validate_password_valid(self, user_model):
        """Test password validation with valid passwords."""
        valid_passwords = [
            "SecurePass123",
            "MyPassword1",
            "Test123456",
            "Password123!"
        ]
        
        for password in valid_passwords:
            assert user_model._validate_password(password) is True
    
    def test_validate_password_invalid(self, user_model):
        """Test password validation with invalid passwords."""
        invalid_passwords = [
            "short",           # Too short
            "nouppercase1",    # No uppercase
            "NOLOWERCASE1",    # No lowercase
            "NoNumbers!",      # No numbers
            "12345678",        # No letters
            ""                 # Empty
        ]
        
        for password in invalid_passwords:
            assert user_model._validate_password(password) is False
    
    def test_get_all_users(self, user_model, mock_user_repo):
        """Test getting all users."""
        mock_db_user1 = Mock()
        mock_db_user1.user_id = "user_1"
        mock_db_user1.email = "user1@example.com"
        mock_db_user1.full_name = "User One"
        mock_db_user1.role = UserRole.PATIENT
        mock_db_user1.status = UserStatus.ACTIVE
        mock_db_user1.created_at = datetime.now()
        mock_db_user1.updated_at = datetime.now()
        mock_db_user1.last_login = None
        mock_db_user1.login_attempts = 0
        mock_db_user1.account_locked_until = None
        mock_db_user1.password_reset_token = None
        mock_db_user1.password_reset_expires = None
        mock_db_user1.preferences = {}
        mock_db_user1.medical_info = {}
        
        mock_db_user2 = Mock()
        mock_db_user2.user_id = "user_2"
        mock_db_user2.email = "user2@example.com"
        mock_db_user2.full_name = "User Two"
        mock_db_user2.role = UserRole.THERAPIST
        mock_db_user2.status = UserStatus.ACTIVE
        mock_db_user2.created_at = datetime.now()
        mock_db_user2.updated_at = datetime.now()
        mock_db_user2.last_login = None
        mock_db_user2.login_attempts = 0
        mock_db_user2.account_locked_until = None
        mock_db_user2.password_reset_token = None
        mock_db_user2.password_reset_expires = None
        mock_db_user2.preferences = {}
        mock_db_user2.medical_info = {}
        
        mock_user_repo.find_all.return_value = [mock_db_user1, mock_db_user2]
        
        result = user_model.get_all_users()
        
        assert len(result) == 2
        assert all(isinstance(user, UserProfile) for user in result)
        assert result[0].user_id == "user_1"
        assert result[1].user_id == "user_2"
    
    def test_users_property(self, user_model, mock_user_repo):
        """Test users property for backward compatibility."""
        mock_db_user1 = Mock()
        mock_db_user1.user_id = "user_1"
        mock_db_user2 = Mock()
        mock_db_user2.user_id = "user_2"
        
        mock_user_repo.find_all.return_value = [mock_db_user1, mock_db_user2]
        
        result = user_model.users
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "user_1" in result
        assert "user_2" in result
        assert result["user_1"] == mock_db_user1
        assert result["user_2"] == mock_db_user2
    
    def test_password_hashes_property(self, user_model, mock_user_repo):
        """Test password_hashes property for backward compatibility."""
        mock_db_user1 = Mock()
        mock_db_user1.user_id = "user_1"
        mock_db_user1.password_hash = "hash1"
        mock_db_user2 = Mock()
        mock_db_user2.user_id = "user_2"
        mock_db_user2.password_hash = "hash2"
        
        mock_user_repo.find_all.return_value = [mock_db_user1, mock_db_user2]
        
        result = user_model.password_hashes
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert result["user_1"] == "hash1"
        assert result["user_2"] == "hash2"
    
    def test_cleanup_expired_data(self, user_model):
        """Test cleanup of expired data."""
        # This method currently just logs a message
        result = user_model.cleanup_expired_data()
        assert result is None  # Method doesn't return anything