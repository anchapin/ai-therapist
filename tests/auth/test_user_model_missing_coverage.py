"""
Tests for user model missing coverage lines
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the user model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from auth.user_model import UserProfile, UserModel, UserRole, UserStatus


class TestUserModelMissingCoverage:
    """Tests for missing coverage lines in user model"""
    
    def test_user_profile_to_dict_with_sensitive_data(self):
        """Test UserProfile.to_dict with sensitive data masking"""
        profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            login_attempts=0,
            locked_until=None,
            medical_info={"ssn": "123-45-6789", "diagnosis": "anxiety"},
            preferences={"theme": "dark"},
            security_settings={"2fa_enabled": True}
        )
        
        # Test with owner request (should show sensitive data)
        result = profile.to_dict(is_owner=True)
        assert result["email"] == "test@example.com"
        assert result["medical_info"]["ssn"] == "123-45-6789"
        
        # Test with non-owner request (should mask sensitive data)
        result = profile.to_dict(is_owner=False)
        assert "@" in result["email"]  # Email should be masked
        assert result["medical_info"]["ssn"] != "123-45-6789"  # SSN should be masked
    
    def test_user_profile_sanitize_medical_info(self):
        """Test UserProfile._sanitize_medical_info method"""
        profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            login_attempts=0,
            locked_until=None,
            medical_info={
                "ssn": "123-45-6789",
                "diagnosis": "anxiety",
                "medications": ["prozac", "xanax"],
                "notes": "Patient has history of depression"
            },
            preferences={},
            security_settings={}
        )
        
        # Test with owner request
        result = profile._sanitize_medical_info(is_owner=True)
        assert result["ssn"] == "123-45-6789"
        assert result["diagnosis"] == "anxiety"
        
        # Test with non-owner request
        result = profile._sanitize_medical_info(is_owner=False)
        assert result["ssn"] != "123-45-6789"  # Should be masked
        assert len(result["medications"]) == 0  # Should be empty for privacy
        assert result["notes"] == "[REDACTED]"  # Should be redacted
    
    def test_user_profile_mask_email(self):
        """Test UserProfile._mask_email method"""
        profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            login_attempts=0,
            locked_until=None,
            medical_info={},
            preferences={},
            security_settings={}
        )
        
        # Test email masking
        result = profile._mask_email()
        assert "@" in result
        assert "test" not in result  # Username should be partially masked
        assert "example.com" in result  # Domain should remain visible
    
    def test_user_model_migrate_legacy_data(self):
        """Test UserModel._migrate_legacy_data method"""
        # Create a temporary file with legacy data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            legacy_data = {
                "users": {
                    "legacy_user1": {
                        "email": "legacy1@example.com",
                        "password_hash": "old_hash",
                        "full_name": "Legacy User 1",
                        "role": "user"
                    },
                    "legacy_user2": {
                        "email": "legacy2@example.com",
                        "password_hash": "old_hash2",
                        "full_name": "Legacy User 2",
                        "role": "admin"
                    }
                }
            }
            json.dump(legacy_data, f)
            temp_file = f.name
        
        try:
            with patch('auth.user_model.DATA_FILE', temp_file):
                user_model = UserModel()
                
                # Check if legacy data was migrated
                users = user_model.get_all_users()
                assert len(users) >= 2
                
                # Check if users have the new structure
                user1 = user_model.get_user_by_email("legacy1@example.com")
                assert user1 is not None
                assert user1["role"] == "user"
                
                user2 = user_model.get_user_by_email("legacy2@example.com")
                assert user2 is not None
                assert user2["role"] == "admin"
        finally:
            os.unlink(temp_file)
    
    def test_user_model_create_user_missing_fields(self):
        """Test UserModel.create_user with missing required fields"""
        user_model = UserModel()
        
        # Test with missing email
        result = user_model.create_user("", "password", "Test User")
        assert result is None
        
        # Test with missing password
        result = user_model.create_user("test@example.com", "", "Test User")
        assert result is None
        
        # Test with missing name
        result = user_model.create_user("test@example.com", "password", "")
        assert result is None
    
    def test_user_model_change_password_incorrect_current(self):
        """Test UserModel.change_password with incorrect current password"""
        user_model = UserModel()
        
        # Create a user first
        user = user_model.create_user("test@example.com", "current_password", "Test User")
        assert user is not None
        
        # Try to change with incorrect current password
        result = user_model.change_password(user["id"], "wrong_current_password", "new_password")
        assert result is False
    
    def test_user_model_initiate_password_reset_user_not_found(self):
        """Test UserModel.initiate_password_reset when user not found"""
        user_model = UserModel()
        
        result = user_model.initiate_password_reset("nonexistent@example.com")
        assert result is None
    
    def test_user_model_get_all_users_with_filters(self):
        """Test UserModel.get_all_users with filters"""
        user_model = UserModel()
        
        # Create users with different roles
        user1 = user_model.create_user("user1@example.com", "password", "User 1")
        user2 = user_model.create_user("admin@example.com", "password", "Admin User")
        
        # Update user2 to admin role
        user_model.update_user(user2["id"], role="admin")
        
        # Get all users
        all_users = user_model.get_all_users()
        assert len(all_users) >= 2
        
        # Get users by role
        admin_users = user_model.get_all_users(role="admin")
        assert any(user["email"] == "admin@example.com" for user in admin_users)
        
        # Get active users
        active_users = user_model.get_all_users(status="active")
        assert len(active_users) >= 2
    
    def test_user_model_cleanup_expired_data(self):
        """Test UserModel.cleanup_expired_data method"""
        user_model = UserModel()
        
        # Create a user with expired reset token
        user = user_model.create_user("cleanup@example.com", "password", "Cleanup User")
        
        # Manually add expired reset token
        user_model.users[user["id"]]["reset_token"] = "expired_token"
        user_model.users[user["id"]]["reset_token_expires"] = datetime.utcnow() - timedelta(hours=1)
        user_model._save_data()
        
        # Run cleanup
        user_model.cleanup_expired_data()
        
        # Check if expired token was removed
        updated_user = user_model.get_user(user["id"])
        assert "reset_token" not in updated_user or updated_user.get("reset_token") is None
    
    def test_user_model_edge_cases(self):
        """Test UserModel edge cases"""
        user_model = UserModel()
        
        # Test get_user with non-existent ID
        result = user_model.get_user("non_existent_id")
        assert result is None
        
        # Test get_user_by_email with non-existent email
        result = user_model.get_user_by_email("nonexistent@example.com")
        assert result is None
        
        # Test update_user with non-existent ID
        result = user_model.update_user("non_existent_id", full_name="Updated Name")
        assert result is None
        
        # Test reset_password with invalid token
        result = user_model.reset_password("invalid_token", "new_password")
        assert result is False
        
        # Test deactivate_user with non-existent ID
        result = user_model.deactivate_user("non_existent_id")
        assert result is False
    
    def test_user_profile_edge_cases(self):
        """Test UserProfile edge cases"""
        profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            login_attempts=0,
            locked_until=None,
            medical_info={},
            preferences={},
            security_settings={}
        )
        
        # Test is_locked with no lock
        assert profile.is_locked() is False
        
        # Test is_locked with expired lock
        profile.locked_until = datetime.utcnow() - timedelta(hours=1)
        assert profile.is_locked() is False
        
        # Test is_locked with active lock
        profile.locked_until = datetime.utcnow() + timedelta(hours=1)
        assert profile.is_locked() is True
        
        # Test can_access_resource with owner
        assert profile.can_access_resource("test_user", "read") is True
        
        # Test can_access_resource with non-owner
        assert profile.can_access_resource("other_user", "read") is False
        
        # Test _is_owner_request with matching user
        assert profile._is_owner_request("test_user") is True
        
        # Test _is_owner_request with non-matching user
        assert profile._is_owner_request("other_user") is False
    
    def test_user_model_validation_edge_cases(self):
        """Test UserModel validation edge cases"""
        user_model = UserModel()
        
        # Test _validate_email with invalid emails
        assert user_model._validate_email("") is False
        assert user_model._validate_email("invalid-email") is False
        assert user_model._validate_email("no-at-symbol.com") is False
        assert user_model._validate_email("@missing-local.com") is False
        
        # Test _validate_password with weak passwords
        assert user_model._validate_password("") is False
        assert user_model._validate_password("123") is False  # Too short
        assert user_model._validate_password("password") is False  # Too common
        
        # Test _validate_password with strong passwords
        assert user_model._validate_password("StrongP@ssw0rd!") is True
        assert user_model._validate_password("MySecurePass123") is True
    
    def test_user_model_concurrent_access(self):
        """Test UserModel with concurrent access"""
        user_model = UserModel()
        
        # Create a user
        user = user_model.create_user("concurrent@example.com", "password", "Concurrent User")
        assert user is not None
        
        # Simulate concurrent updates
        def update_user(user_id, suffix):
            return user_model.update_user(user_id, full_name=f"Updated Name {suffix}")
        
        # Create multiple threads to update the same user
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_user, args=(user["id"], i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check if user was updated (at least one update should have succeeded)
        updated_user = user_model.get_user(user["id"])
        assert "Updated Name" in updated_user["full_name"]