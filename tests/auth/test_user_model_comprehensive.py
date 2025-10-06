"""
Comprehensive tests for user model functionality.

This file provides extensive test coverage for user_model.py including:
- User profile creation and validation
- Password hashing and verification
- Role-based permissions
- User statistics and analytics
- Database operations and edge cases
- Error handling and security features
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
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role=UserRole.PATIENT,
            is_active=True,
            created_at=datetime.now(),
            last_login=datetime.now(),
            preferences={"theme": "dark", "language": "en"},
            metadata={"source": "web"}
        )
        
        assert profile.user_id == "user_123"
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name == "Test User"
        assert profile.role == UserRole.PATIENT
        assert profile.is_active is True
        assert profile.preferences["theme"] == "dark"
        assert profile.metadata["source"] == "web"

    def test_user_profile_creation_with_minimal_fields(self):
        """Test user profile creation with minimal required fields."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password"
        )
        
        assert profile.user_id == "user_123"
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name is None
        assert profile.role == UserRole.GUEST  # Default value
        assert profile.is_active is True  # Default value
        assert profile.created_at is not None
        assert profile.last_login is None
        assert profile.preferences == {}
        assert profile.metadata == {}

    def test_user_profile_to_dict_with_all_fields(self):
        """Test user profile serialization to dictionary."""
        created_at = datetime.now()
        last_login = datetime.now()
        
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role=UserRole.THERAPIST,
            is_active=True,
            created_at=created_at,
            last_login=last_login,
            preferences={"theme": "light"},
            metadata={"department": "therapy"}
        )
        
        result = profile.to_dict()
        
        assert result["user_id"] == "user_123"
        assert result["email"] == "test@example.com"
        assert result["password_hash"] == "hashed_password"
        assert result["full_name"] == "Test User"
        assert result["role"] == UserRole.THERAPIST.value
        assert result["is_active"] is True
        assert result["created_at"] == created_at.isoformat()
        assert result["last_login"] == last_login.isoformat()
        assert result["preferences"] == {"theme": "light"}
        assert result["metadata"] == {"department": "therapy"}

    def test_user_profile_to_dict_with_none_values(self):
        """Test user profile serialization with None values."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name=None,
            last_login=None
        )
        
        result = profile.to_dict()
        
        assert result["full_name"] is None
        assert result["last_login"] is None

    def test_user_profile_from_dict_with_all_fields(self):
        """Test user profile deserialization from dictionary."""
        data = {
            "user_id": "user_123",
            "email": "test@example.com",
            "password_hash": "hashed_password",
            "full_name": "Test User",
            "role": UserRole.THERAPIST.value,
            "is_active": True,
            "created_at": "2023-01-01T12:00:00",
            "last_login": "2023-01-02T12:00:00",
            "preferences": {"theme": "dark"},
            "metadata": {"source": "mobile"}
        }
        
        profile = UserProfile.from_dict(data)
        
        assert profile.user_id == "user_123"
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name == "Test User"
        assert profile.role == UserRole.THERAPIST
        assert profile.is_active is True
        assert profile.created_at.isoformat() == "2023-01-01T12:00:00"
        assert profile.last_login.isoformat() == "2023-01-02T12:00:00"
        assert profile.preferences == {"theme": "dark"}
        assert profile.metadata == {"source": "mobile"}

    def test_user_profile_from_dict_with_missing_fields(self):
        """Test user profile deserialization with missing fields."""
        data = {
            "user_id": "user_123",
            "email": "test@example.com",
            "password_hash": "hashed_password"
        }
        
        profile = UserProfile.from_dict(data)
        
        assert profile.user_id == "user_123"
        assert profile.email == "test@example.com"
        assert profile.password_hash == "hashed_password"
        assert profile.full_name is None
        assert profile.role == UserRole.GUEST  # Default
        assert profile.is_active is True  # Default
        assert profile.created_at is not None
        assert profile.last_login is None
        assert profile.preferences == {}
        assert profile.metadata == {}

    def test_user_profile_update_last_login(self):
        """Test updating last login timestamp."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password"
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
            password_hash="hashed_password",
            preferences={"theme": "light"}
        )
        
        new_preferences = {"theme": "dark", "language": "es", "notifications": True}
        profile.update_preferences(new_preferences)
        
        assert profile.preferences == new_preferences

    def test_user_profile_merge_preferences(self):
        """Test merging preferences with existing ones."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            preferences={"theme": "light", "language": "en"}
        )
        
        new_preferences = {"theme": "dark", "notifications": True}
        profile.update_preferences(new_preferences)
        
        assert profile.preferences == {
            "theme": "dark",
            "language": "en",  # Preserved
            "notifications": True
        }

    def test_user_profile_add_metadata(self):
        """Test adding metadata to user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            metadata={"source": "web"}
        )
        
        profile.add_metadata("device", "mobile")
        profile.add_metadata("version", "1.0")
        
        assert profile.metadata == {
            "source": "web",
            "device": "mobile",
            "version": "1.0"
        }

    def test_user_profile_add_metadata_overwrite(self):
        """Test overwriting existing metadata."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            metadata={"source": "web"}
        )
        
        profile.add_metadata("source", "mobile")
        
        assert profile.metadata == {"source": "mobile"}

    def test_user_profile_deactivate(self):
        """Test deactivating user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True
        )
        
        profile.deactivate()
        
        assert profile.is_active is False

    def test_user_profile_activate(self):
        """Test activating user profile."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            is_active=False
        )
        
        profile.activate()
        
        assert profile.is_active is True

    def test_user_profile_role_promotion_and_demotion(self):
        """Test role promotion and demotion."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            role=UserRole.GUEST
        )
        
        # Promote to patient
        profile.promote_role(UserRole.PATIENT)
        assert profile.role == UserRole.PATIENT
        
        # Promote to therapist
        profile.promote_role(UserRole.THERAPIST)
        assert profile.role == UserRole.THERAPIST
        
        # Demote to patient
        profile.demote_role(UserRole.PATIENT)
        assert profile.role == UserRole.PATIENT

    def test_user_profile_has_permission_with_different_roles(self):
        """Test permission checking for different roles."""
        # Test guest permissions
        guest_profile = UserProfile(
            user_id="guest_123",
            email="guest@example.com",
            password_hash="hashed_password",
            role=UserRole.GUEST
        )
        
        assert guest_profile.has_permission("read:own_profile") is True
        assert guest_profile.has_permission("read:all_profiles") is False
        assert guest_profile.has_permission("write:profile") is False
        
        # Test patient permissions
        patient_profile = UserProfile(
            user_id="patient_123",
            email="patient@example.com",
            password_hash="hashed_password",
            role=UserRole.PATIENT
        )
        
        assert patient_profile.has_permission("read:own_profile") is True
        assert patient_profile.has_permission("write:own_profile") is True
        assert patient_profile.has_permission("read:all_profiles") is False
        
        # Test therapist permissions
        therapist_profile = UserProfile(
            user_id="therapist_123",
            email="therapist@example.com",
            password_hash="hashed_password",
            role=UserRole.THERAPIST
        )
        
        assert therapist_profile.has_permission("read:own_profile") is True
        assert therapist_profile.has_permission("write:own_profile") is True
        assert therapist_profile.has_permission("read:patient_profiles") is True
        assert therapist_profile.has_permission("read:all_profiles") is False
        
        # Test admin permissions
        admin_profile = UserProfile(
            user_id="admin_123",
            email="admin@example.com",
            password_hash="hashed_password",
            role=UserRole.ADMIN
        )
        
        assert admin_profile.has_permission("read:own_profile") is True
        assert admin_profile.has_permission("write:own_profile") is True
        assert admin_profile.has_permission("read:all_profiles") is True
        assert admin_profile.has_permission("write:all_profiles") is True

    def test_user_profile_is_valid_with_complete_data(self):
        """Test profile validation with complete data."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User"
        )
        
        assert profile.is_valid() is True

    def test_user_profile_is_valid_missing_fields(self):
        """Test profile validation with missing fields."""
        # Missing user_id
        profile1 = UserProfile(
            user_id="",
            email="test@example.com",
            password_hash="hashed_password"
        )
        assert profile1.is_valid() is False
        
        # Missing email
        profile2 = UserProfile(
            user_id="user_123",
            email="",
            password_hash="hashed_password"
        )
        assert profile2.is_valid() is False
        
        # Missing password_hash
        profile3 = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash=""
        )
        assert profile3.is_valid() is False

    def test_user_profile_is_valid_invalid_email(self):
        """Test profile validation with invalid email."""
        profile = UserProfile(
            user_id="user_123",
            email="invalid-email",
            password_hash="hashed_password"
        )
        
        assert profile.is_valid() is False

    def test_user_profile_get_safe_data(self):
        """Test getting safe user data without sensitive information."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role=UserRole.PATIENT,
            preferences={"theme": "dark"},
            metadata={"source": "web"}
        )
        
        safe_data = profile.get_safe_data()
        
        assert "password_hash" not in safe_data
        assert safe_data["user_id"] == "user_123"
        assert safe_data["email"] == "test@example.com"
        assert safe_data["full_name"] == "Test User"
        assert safe_data["role"] == UserRole.PATIENT.value
        assert safe_data["preferences"] == {"theme": "dark"}
        assert safe_data["metadata"] == {"source": "web"}

    def test_user_profile_get_safe_data_with_sensitive_metadata(self):
        """Test getting safe data filters sensitive metadata."""
        profile = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            metadata={
                "source": "web",
                "ssn": "123-45-6789",
                "credit_card": "4111-1111-1111-1111",
                "api_key": "secret_key"
            }
        )
        
        safe_data = profile.get_safe_data()
        
        assert "ssn" not in safe_data["metadata"]
        assert "credit_card" not in safe_data["metadata"]
        assert "api_key" not in safe_data["metadata"]
        assert "source" in safe_data["metadata"]


class TestUserSessionComprehensive:
    """Comprehensive test cases for UserSession class."""

    def test_user_session_creation_with_all_fields(self):
        """Test user session creation with all fields."""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=24)
        
        session = UserSession(
            session_id="session_123",
            user_id="user_123",
            token="jwt_token_123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            created_at=created_at,
            expires_at=expires_at,
            is_active=True,
            metadata={"device": "mobile"}
        )
        
        assert session.session_id == "session_123"
        assert session.user_id == "user_123"
        assert session.token == "jwt_token_123"
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.created_at == created_at
        assert session.expires_at == expires_at
        assert session.is_active is True
        assert session.metadata["device"] == "mobile"

    def test_user_session_creation_with_minimal_fields(self):
        """Test user session creation with minimal fields."""
        session = UserSession(
            session_id="session_123",
            user_id="user_123",
            token="jwt_token_123"
        )
        
        assert session.session_id == "session_123"
        assert session.user_id == "user_123"
        assert session.token == "jwt_token_123"
        assert session.ip_address is None
        assert session.user_agent is None
        assert session.created_at is not None
        assert session.expires_at is not None
        assert session.is_active is True  # Default
        assert session.metadata == {}

    def test_user_session_is_expired(self):
        """Test session expiration checking."""
        # Create expired session
        past_time = datetime.now() - timedelta(hours=1)
        expired_session = UserSession(
            session_id="session_123",
            user_id="user_123",
            token="jwt_token_123",
            created_at=past_time,
            expires_at=past_time + timedelta(minutes=30)  # Expired 30 minutes ago
        )
        
        assert expired_session.is_expired() is True
        
        # Create valid session
        future_time = datetime.now() + timedelta(hours=1)
        valid_session = UserSession(
            session_id="session_456",
            user_id="user_456",
            token="jwt_token_456",
            expires_at=future_time
        )
        
        assert valid_session.is_expired() is False

    def test_user_session_extend(self):
        """Test extending session expiration."""
        original_expires_at = datetime.now() + timedelta(hours=1)
        session = UserSession(
            session_id="session_123",
            user_id="user_123",
            token="jwt_token_123",
            expires_at=original_expires_at
        )
        
        session.extend(timedelta(hours=24))
        
        assert session.expires_at == original_expires_at + timedelta(hours=24)

    def test_user_session_invalidate(self):
        """Test session invalidation."""
        session = UserSession(
            session_id="session_123",
            user_id="user_123",
            token="jwt_token_123",
            is_active=True
        )
        
        session.invalidate()
        
        assert session.is_active is False

    def test_user_session_to_dict_and_from_dict(self):
        """Test session serialization and deserialization."""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=24)
        
        session = UserSession(
            session_id="session_123",
            user_id="user_123",
            token="jwt_token_123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            created_at=created_at,
            expires_at=expires_at,
            is_active=True,
            metadata={"device": "mobile"}
        )
        
        # Test to_dict
        session_dict = session.to_dict()
        assert session_dict["session_id"] == "session_123"
        assert session_dict["user_id"] == "user_123"
        assert session_dict["token"] == "jwt_token_123"
        assert session_dict["ip_address"] == "192.168.1.1"
        assert session_dict["user_agent"] == "Mozilla/5.0"
        assert session_dict["created_at"] == created_at.isoformat()
        assert session_dict["expires_at"] == expires_at.isoformat()
        assert session_dict["is_active"] is True
        assert session_dict["metadata"] == {"device": "mobile"}
        
        # Test from_dict
        restored_session = UserSession.from_dict(session_dict)
        assert restored_session.session_id == session.session_id
        assert restored_session.user_id == session.user_id
        assert restored_session.token == session.token
        assert restored_session.ip_address == session.ip_address
        assert restored_session.user_agent == session.user_agent
        assert restored_session.created_at == session.created_at
        assert restored_session.expires_at == session.expires_at
        assert restored_session.is_active == session.is_active
        assert restored_session.metadata == session.metadata


class TestUserStatisticsComprehensive:
    """Comprehensive test cases for UserStatistics class."""

    def test_user_statistics_creation_with_all_fields(self):
        """Test user statistics creation with all fields."""
        stats = UserStatistics(
            user_id="user_123",
            total_sessions=10,
            total_duration=timedelta(hours=5),
            average_session_duration=timedelta(minutes=30),
            last_session_date=datetime.now(),
            session_count_by_month={"2023-01": 5, "2023-02": 5},
            preferred_session_time="morning",
            completion_rate=0.85,
            engagement_score=7.5,
            metadata={"device_preference": "mobile"}
        )
        
        assert stats.user_id == "user_123"
        assert stats.total_sessions == 10
        assert stats.total_duration == timedelta(hours=5)
        assert stats.average_session_duration == timedelta(minutes=30)
        assert stats.session_count_by_month == {"2023-01": 5, "2023-02": 5}
        assert stats.preferred_session_time == "morning"
        assert stats.completion_rate == 0.85
        assert stats.engagement_score == 7.5
        assert stats.metadata["device_preference"] == "mobile"

    def test_user_statistics_creation_with_minimal_fields(self):
        """Test user statistics creation with minimal fields."""
        stats = UserStatistics(user_id="user_123")
        
        assert stats.user_id == "user_123"
        assert stats.total_sessions == 0
        assert stats.total_duration == timedelta(0)
        assert stats.average_session_duration == timedelta(0)
        assert stats.last_session_date is None
        assert stats.session_count_by_month == {}
        assert stats.preferred_session_time is None
        assert stats.completion_rate == 0.0
        assert stats.engagement_score == 0.0
        assert stats.metadata == {}

    def test_user_statistics_record_session(self):
        """Test recording a new session."""
        stats = UserStatistics(user_id="user_123")
        session_duration = timedelta(minutes=45)
        session_date = datetime.now()
        
        stats.record_session(session_duration, session_date)
        
        assert stats.total_sessions == 1
        assert stats.total_duration == session_duration
        assert stats.average_session_duration == session_duration
        assert stats.last_session_date == session_date
        
        # Record another session
        session_duration_2 = timedelta(minutes=30)
        session_date_2 = datetime.now()
        
        stats.record_session(session_duration_2, session_date_2)
        
        assert stats.total_sessions == 2
        assert stats.total_duration == session_duration + session_duration_2
        assert stats.average_session_duration == (session_duration + session_duration_2) / 2
        assert stats.last_session_date == session_date_2

    def test_user_statistics_get_monthly_session_count(self):
        """Test getting monthly session count."""
        stats = UserStatistics(
            user_id="user_123",
            session_count_by_month={
                "2023-01": 5,
                "2023-02": 8,
                "2023-03": 3
            }
        )
        
        assert stats.get_monthly_session_count("2023-01") == 5
        assert stats.get_monthly_session_count("2023-02") == 8
        assert stats.get_monthly_session_count("2023-03") == 3
        assert stats.get_monthly_session_count("2023-04") == 0  # Not present

    def test_user_statistics_get_total_session_hours(self):
        """Test getting total session hours."""
        stats = UserStatistics(
            user_id="user_123",
            total_duration=timedelta(hours=5, minutes=30)
        )
        
        assert stats.get_total_session_hours() == 5.5

    def test_user_statistics_update_engagement_score(self):
        """Test updating engagement score."""
        stats = UserStatistics(user_id="user_123")
        
        stats.update_engagement_score(8.5)
        assert stats.engagement_score == 8.5
        
        stats.update_engagement_score(7.0)
        assert stats.engagement_score == 7.0

    def test_user_statistics_to_dict_and_from_dict(self):
        """Test statistics serialization and deserialization."""
        stats = UserStatistics(
            user_id="user_123",
            total_sessions=10,
            total_duration=timedelta(hours=5),
            average_session_duration=timedelta(minutes=30),
            last_session_date=datetime.now(),
            session_count_by_month={"2023-01": 5, "2023-02": 5},
            preferred_session_time="morning",
            completion_rate=0.85,
            engagement_score=7.5,
            metadata={"device_preference": "mobile"}
        )
        
        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["user_id"] == "user_123"
        assert stats_dict["total_sessions"] == 10
        assert stats_dict["total_duration"] == "5:00:00"
        assert stats_dict["average_session_duration"] == "0:30:00"
        assert stats_dict["session_count_by_month"] == {"2023-01": 5, "2023-02": 5}
        assert stats_dict["preferred_session_time"] == "morning"
        assert stats_dict["completion_rate"] == 0.85
        assert stats_dict["engagement_score"] == 7.5
        assert stats_dict["metadata"] == {"device_preference": "mobile"}
        
        # Test from_dict
        restored_stats = UserStatistics.from_dict(stats_dict)
        assert restored_stats.user_id == stats.user_id
        assert restored_stats.total_sessions == stats.total_sessions
        assert restored_stats.total_duration == stats.total_duration
        assert restored_stats.average_session_duration == stats.average_session_duration
        assert restored_stats.session_count_by_month == stats.session_count_by_month
        assert restored_stats.preferred_session_time == stats.preferred_session_time
        assert restored_stats.completion_rate == stats.completion_rate
        assert restored_stats.engagement_score == stats.engagement_score
        assert restored_stats.metadata == stats.metadata


class TestPasswordManagerComprehensive:
    """Comprehensive test cases for PasswordManager class."""

    def test_password_manager_hash_password(self):
        """Test password hashing."""
        password = "SecurePassword123!"
        password_manager = PasswordManager()
        
        hashed_password = password_manager.hash_password(password)
        
        assert hashed_password is not None
        assert hashed_password != password
        assert len(hashed_password) > 50  # bcrypt hashes are long
        assert hashed_password.startswith("$2b$")  # bcrypt prefix

    def test_password_manager_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "SecurePassword123!"
        password_manager = PasswordManager()
        
        hashed_password = password_manager.hash_password(password)
        is_valid = password_manager.verify_password(password, hashed_password)
        
        assert is_valid is True

    def test_password_manager_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "SecurePassword123!"
        wrong_password = "WrongPassword123!"
        password_manager = PasswordManager()
        
        hashed_password = password_manager.hash_password(password)
        is_valid = password_manager.verify_password(wrong_password, hashed_password)
        
        assert is_valid is False

    def test_password_manager_validate_password_strong(self):
        """Test password validation with strong password."""
        password_manager = PasswordManager()
        
        # Strong password
        is_valid, errors = password_manager.validate_password("SecurePass123!")
        assert is_valid is True
        assert len(errors) == 0

    def test_password_manager_validate_password_weak(self):
        """Test password validation with weak passwords."""
        password_manager = PasswordManager()
        
        # Too short
        is_valid, errors = password_manager.validate_password("short")
        assert is_valid is False
        assert any("at least 8 characters" in error for error in errors)
        
        # No uppercase
        is_valid, errors = password_manager.validate_password("lowercase123!")
        assert is_valid is False
        assert any("uppercase letter" in error for error in errors)
        
        # No lowercase
        is_valid, errors = password_manager.validate_password("UPPERCASE123!")
        assert is_valid is False
        assert any("lowercase letter" in error for error in errors)
        
        # No digit
        is_valid, errors = password_manager.validate_password("NoDigits!")
        assert is_valid is False
        assert any("digit" in error for error in errors)
        
        # No special character
        is_valid, errors = password_manager.validate_password("NoSpecialChars123")
        assert is_valid is False
        assert any("special character" in error for error in errors)

    def test_password_manager_generate_random_password(self):
        """Test random password generation."""
        password_manager = PasswordManager()
        
        password = password_manager.generate_random_password()
        
        assert len(password) >= 12
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    def test_password_manager_generate_random_password_custom_length(self):
        """Test random password generation with custom length."""
        password_manager = PasswordManager()
        
        password = password_manager.generate_random_password(16)
        
        assert len(password) == 16

    def test_password_manager_check_password_breached(self):
        """Test checking if password is breached."""
        password_manager = PasswordManager()
        
        # Common breached password
        is_breached = password_manager.check_password_breached("password123")
        assert is_breached is True
        
        # Strong unique password (should not be breached)
        is_breached = password_manager.check_password_breached("UniqueSecurePass123!@#")
        assert is_breached is False


class TestPermissionManagerComprehensive:
    """Comprehensive test cases for PermissionManager class."""

    def test_permission_manager_has_permission_guest(self):
        """Test permission checking for guest role."""
        permission_manager = PermissionManager()
        
        assert permission_manager.has_permission(UserRole.GUEST, "read:own_profile") is True
        assert permission_manager.has_permission(UserRole.GUEST, "write:own_profile") is False
        assert permission_manager.has_permission(UserRole.GUEST, "read:patient_profiles") is False
        assert permission_manager.has_permission(UserRole.GUEST, "admin:users") is False

    def test_permission_manager_has_permission_patient(self):
        """Test permission checking for patient role."""
        permission_manager = PermissionManager()
        
        assert permission_manager.has_permission(UserRole.PATIENT, "read:own_profile") is True
        assert permission_manager.has_permission(UserRole.PATIENT, "write:own_profile") is True
        assert permission_manager.has_permission(UserRole.PATIENT, "read:patient_profiles") is False
        assert permission_manager.has_permission(UserRole.PATIENT, "admin:users") is False

    def test_permission_manager_has_permission_therapist(self):
        """Test permission checking for therapist role."""
        permission_manager = PermissionManager()
        
        assert permission_manager.has_permission(UserRole.THERAPIST, "read:own_profile") is True
        assert permission_manager.has_permission(UserRole.THERAPIST, "write:own_profile") is True
        assert permission_manager.has_permission(UserRole.THERAPIST, "read:patient_profiles") is True
        assert permission_manager.has_permission(UserRole.THERAPIST, "write:patient_profiles") is True
        assert permission_manager.has_permission(UserRole.THERAPIST, "admin:users") is False

    def test_permission_manager_has_permission_admin(self):
        """Test permission checking for admin role."""
        permission_manager = PermissionManager()
        
        assert permission_manager.has_permission(UserRole.ADMIN, "read:own_profile") is True
        assert permission_manager.has_permission(UserRole.ADMIN, "write:own_profile") is True
        assert permission_manager.has_permission(UserRole.ADMIN, "read:patient_profiles") is True
        assert permission_manager.has_permission(UserRole.ADMIN, "write:patient_profiles") is True
        assert permission_manager.has_permission(UserRole.ADMIN, "admin:users") is True
        assert permission_manager.has_permission(UserRole.ADMIN, "admin:system") is True

    def test_permission_manager_get_permissions_for_role(self):
        """Test getting all permissions for a role."""
        permission_manager = PermissionManager()
        
        guest_permissions = permission_manager.get_permissions_for_role(UserRole.GUEST)
        assert "read:own_profile" in guest_permissions
        assert "write:own_profile" not in guest_permissions
        
        admin_permissions = permission_manager.get_permissions_for_role(UserRole.ADMIN)
        assert "read:own_profile" in admin_permissions
        assert "write:own_profile" in admin_permissions
        assert "admin:users" in admin_permissions
        assert "admin:system" in admin_permissions

    def test_permission_manager_can_access_resource(self):
        """Test resource access checking."""
        permission_manager = PermissionManager()
        
        # Patient accessing own profile
        assert permission_manager.can_access_resource(
            UserRole.PATIENT, "profile", "user_123", "user_123"
        ) is True
        
        # Patient accessing another user's profile
        assert permission_manager.can_access_resource(
            UserRole.PATIENT, "profile", "user_123", "user_456"
        ) is False
        
        # Therapist accessing patient profile
        assert permission_manager.can_access_resource(
            UserRole.THERAPIST, "profile", "therapist_123", "patient_456"
        ) is True
        
        # Admin accessing any profile
        assert permission_manager.can_access_resource(
            UserRole.ADMIN, "profile", "admin_123", "user_456"
        ) is True


class TestUserAnalyticsComprehensive:
    """Comprehensive test cases for UserAnalytics class."""

    def test_user_analytics_get_user_summary(self):
        """Test getting user summary analytics."""
        # Mock user DAO
        mock_user_dao = Mock()
        mock_user = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role=UserRole.PATIENT,
            created_at=datetime.now() - timedelta(days=30)
        )
        mock_user_dao.get_user_by_id.return_value = mock_user
        
        # Mock statistics DAO
        mock_stats_dao = Mock()
        mock_stats = UserStatistics(
            user_id="user_123",
            total_sessions=10,
            total_duration=timedelta(hours=5),
            average_session_duration=timedelta(minutes=30),
            last_session_date=datetime.now() - timedelta(days=1),
            completion_rate=0.85,
            engagement_score=7.5
        )
        mock_stats_dao.get_user_statistics.return_value = mock_stats
        
        # Mock session DAO
        mock_session_dao = Mock()
        recent_sessions = [
            UserSession(
                session_id="session_1",
                user_id="user_123",
                token="token_1",
                created_at=datetime.now() - timedelta(days=1)
            ),
            UserSession(
                session_id="session_2",
                user_id="user_123",
                token="token_2",
                created_at=datetime.now() - timedelta(days=3)
            )
        ]
        mock_session_dao.get_user_sessions.return_value = recent_sessions
        
        analytics = UserAnalytics(mock_user_dao, mock_stats_dao, mock_session_dao)
        summary = analytics.get_user_summary("user_123")
        
        assert summary["user_id"] == "user_123"
        assert summary["email"] == "test@example.com"
        assert summary["full_name"] == "Test User"
        assert summary["role"] == UserRole.PATIENT.value
        assert summary["total_sessions"] == 10
        assert summary["total_hours"] == 5.0
        assert summary["average_session_duration_minutes"] == 30.0
        assert summary["completion_rate"] == 0.85
        assert summary["engagement_score"] == 7.5
        assert summary["recent_sessions_count"] == 2
        assert summary["account_age_days"] == 30

    def test_user_analytics_get_user_summary_user_not_found(self):
        """Test getting user summary when user doesn't exist."""
        # Mock user DAO
        mock_user_dao = Mock()
        mock_user_dao.get_user_by_id.return_value = None
        
        # Mock statistics DAO
        mock_stats_dao = Mock()
        mock_stats_dao.get_user_statistics.return_value = None
        
        # Mock session DAO
        mock_session_dao = Mock()
        mock_session_dao.get_user_sessions.return_value = []
        
        analytics = UserAnalytics(mock_user_dao, mock_stats_dao, mock_session_dao)
        summary = analytics.get_user_summary("nonexistent_user")
        
        assert summary is None

    def test_user_analytics_get_user_engagement_trends(self):
        """Test getting user engagement trends."""
        # Mock statistics DAO
        mock_stats_dao = Mock()
        mock_stats = UserStatistics(
            user_id="user_123",
            session_count_by_month={
                "2023-01": 5,
                "2023-02": 8,
                "2023-03": 12,
                "2023-04": 10
            },
            engagement_score=7.5
        )
        mock_stats_dao.get_user_statistics.return_value = mock_stats
        
        analytics = UserAnalytics(Mock(), mock_stats_dao, Mock())
        trends = analytics.get_user_engagement_trends("user_123")
        
        assert trends["user_id"] == "user_123"
        assert trends["current_engagement_score"] == 7.5
        assert "monthly_sessions" in trends
        assert trends["monthly_sessions"]["2023-01"] == 5
        assert trends["monthly_sessions"]["2023-02"] == 8
        assert trends["monthly_sessions"]["2023-03"] == 12
        assert trends["monthly_sessions"]["2023-04"] == 10
        assert "trend_direction" in trends
        assert "average_sessions_per_month" in trends

    def test_user_analytics_get_user_activity_patterns(self):
        """Test getting user activity patterns."""
        # Mock session DAO
        mock_session_dao = Mock()
        
        # Create sessions at different times
        morning_session = UserSession(
            session_id="session_1",
            user_id="user_123",
            token="token_1",
            created_at=datetime.now().replace(hour=9, minute=0)
        )
        
        evening_session = UserSession(
            session_id="session_2",
            user_id="user_123",
            token="token_2",
            created_at=datetime.now().replace(hour=18, minute=0)
        )
        
        mock_session_dao.get_user_sessions.return_value = [morning_session, evening_session]
        
        analytics = UserAnalytics(Mock(), Mock(), mock_session_dao)
        patterns = analytics.get_user_activity_patterns("user_123")
        
        assert patterns["user_id"] == "user_123"
        assert "preferred_time_of_day" in patterns
        assert "session_frequency" in patterns
        assert "most_active_day" in patterns
        assert patterns["total_sessions_analyzed"] == 2

    def test_user_analytics_get_user_completion_analytics(self):
        """Test getting user completion analytics."""
        # Mock statistics DAO
        mock_stats_dao = Mock()
        mock_stats = UserStatistics(
            user_id="user_123",
            completion_rate=0.85,
            total_sessions=20
        )
        mock_stats_dao.get_user_statistics.return_value = mock_stats
        
        analytics = UserAnalytics(Mock(), mock_stats_dao, Mock())
        completion = analytics.get_user_completion_analytics("user_123")
        
        assert completion["user_id"] == "user_123"
        assert completion["completion_rate"] == 0.85
        assert completion["total_sessions"] == 20
        assert completion["completed_sessions"] == 17  # 20 * 0.85
        assert completion["incomplete_sessions"] == 3  # 20 - 17
        assert completion["completion_grade"] == "Excellent"  # Based on rate

    def test_user_analytics_compare_user_with_peers(self):
        """Test comparing user with peers."""
        # Mock user DAO
        mock_user_dao = Mock()
        mock_user = UserProfile(
            user_id="user_123",
            email="test@example.com",
            password_hash="hashed_password",
            role=UserRole.PATIENT
        )
        mock_user_dao.get_user_by_id.return_value = mock_user
        
        # Mock statistics DAO
        mock_stats_dao = Mock()
        user_stats = UserStatistics(
            user_id="user_123",
            total_sessions=10,
            engagement_score=7.5,
            completion_rate=0.85
        )
        mock_stats_dao.get_user_statistics.return_value = user_stats
        
        # Mock peer statistics
        peer_stats = [
            UserStatistics(user_id="peer_1", total_sessions=8, engagement_score=6.0, completion_rate=0.75),
            UserStatistics(user_id="peer_2", total_sessions=12, engagement_score=8.0, completion_rate=0.90),
            UserStatistics(user_id="peer_3", total_sessions=15, engagement_score=7.0, completion_rate=0.80)
        ]
        mock_stats_dao.get_peer_statistics.return_value = peer_stats
        
        analytics = UserAnalytics(mock_user_dao, mock_stats_dao, Mock())
        comparison = analytics.compare_user_with_peers("user_123")
        
        assert comparison["user_id"] == "user_123"
        assert comparison["user_stats"]["total_sessions"] == 10
        assert comparison["user_stats"]["engagement_score"] == 7.5
        assert comparison["user_stats"]["completion_rate"] == 0.85
        assert comparison["peer_averages"]["total_sessions"] == 11.67  # (8+12+15)/3
        assert comparison["peer_averages"]["engagement_score"] == 7.0  # (6.0+8.0+7.0)/3
        assert comparison["peer_averages"]["completion_rate"] == 0.82  # (0.75+0.90+0.80)/3
        assert comparison["percentiles"]["total_sessions"] == 33  # User is below average
        assert comparison["percentiles"]["engagement_score"] == 67  # User is above average
        assert comparison["percentiles"]["completion_rate"] == 67  # User is above average