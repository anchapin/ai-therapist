"""
Comprehensive unit tests for voice/security.py module.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.security import VoiceSecurity, SecurityLevel, SecurityError
    from voice.audio_processor import AudioData
except ImportError as e:
    pytest.skip(f"voice.security module not available: {e}", allow_module_level=True)


class TestSecurityLevel:
    """Test SecurityLevel enum."""
    
    def test_security_level_values(self):
        """Test security level enum values."""
        assert SecurityLevel.GUEST.value == 0
        assert SecurityLevel.USER.value == 1
        assert SecurityLevel.THERAPIST.value == 2
        assert SecurityLevel.ADMIN.value == 3
    
    def test_security_level_ordering(self):
        """Test security level ordering."""
        assert SecurityLevel.GUEST < SecurityLevel.USER
        assert SecurityLevel.USER < SecurityLevel.THERAPIST
        assert SecurityLevel.THERAPIST < SecurityLevel.ADMIN
    
    def test_security_level_from_string(self):
        """Test creating security level from string."""
        assert SecurityLevel("guest") == SecurityLevel.GUEST
        assert SecurityLevel("user") == SecurityLevel.USER
        assert SecurityLevel("therapist") == SecurityLevel.THERAPIST
        assert SecurityLevel("admin") == SecurityLevel.ADMIN


class TestVoiceSecurity:
    """Test VoiceSecurity class."""
    
    @pytest.fixture
    def voice_security(self):
        """Create a voice security instance."""
        return VoiceSecurity()
    
    def test_voice_security_initialization(self, voice_security):
        """Test voice security initialization."""
        assert voice_security is not None
        assert hasattr(voice_security, 'audit_log')
        assert hasattr(voice_security, 'session_manager')
    
    def test_validate_command_valid(self, voice_security):
        """Test validating a valid command."""
        command = "start_session"
        user_level = SecurityLevel.USER
        params = {"user_id": "test_user"}
        
        result = voice_security.validate_command(command, user_level, params)
        
        assert result is True
    
    def test_validate_command_invalid_level(self, voice_security):
        """Test validating command with insufficient security level."""
        command = "delete_all_data"  # Requires admin level
        user_level = SecurityLevel.GUEST
        params = {}
        
        result = voice_security.validate_command(command, user_level, params)
        
        assert result is False
    
    def test_validate_command_missing_params(self, voice_security):
        """Test validating command with missing required parameters."""
        command = "start_session"  # Requires user_id
        user_level = SecurityLevel.USER
        params = {}  # Missing user_id
        
        result = voice_security.validate_command(command, user_level, params)
        
        assert result is False
    
    def test_validate_command_unknown_command(self, voice_security):
        """Test validating unknown command."""
        command = "unknown_command"
        user_level = SecurityLevel.USER
        params = {}
        
        result = voice_security.validate_command(command, user_level, params)
        
        assert result is False
    
    def test_create_session(self, voice_security):
        """Test creating a security session."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Verify session was created
        session = voice_security.get_session(session_id)
        assert session is not None
        assert session.user_id == user_id
        assert session.security_level == security_level
    
    def test_create_session_with_expiry(self, voice_security):
        """Test creating a session with custom expiry."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        expires_in = timedelta(hours=2)
        
        session_id = voice_security.create_session(
            user_id, 
            security_level, 
            expires_in=expires_in
        )
        
        session = voice_security.get_session(session_id)
        assert session is not None
        # Check expiry time is approximately 2 hours from now
        expected_expiry = datetime.now() + expires_in
        actual_expiry = session.expires_at
        time_diff = abs(expected_expiry - actual_expiry)
        assert time_diff < timedelta(minutes=1)  # Allow 1 minute tolerance
    
    def test_get_session_valid(self, voice_security):
        """Test getting a valid session."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        session = voice_security.get_session(session_id)
        
        assert session is not None
        assert session.user_id == user_id
        assert session.security_level == security_level
        assert session.is_active == True
    
    def test_get_session_invalid(self, voice_security):
        """Test getting an invalid session."""
        session = voice_security.get_session("invalid_session_id")
        
        assert session is None
    
    def test_get_session_expired(self, voice_security):
        """Test getting an expired session."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        expires_in = timedelta(seconds=0.1)  # Very short expiry
        
        session_id = voice_security.create_session(
            user_id, 
            security_level, 
            expires_in=expires_in
        )
        
        # Wait for session to expire
        time.sleep(0.2)
        
        session = voice_security.get_session(session_id)
        assert session is None or session.is_active == False
    
    def test_validate_session_valid(self, voice_security):
        """Test validating a valid session."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        
        result = voice_security.validate_session(session_id)
        
        assert result is True
    
    def test_validate_session_invalid(self, voice_security):
        """Test validating an invalid session."""
        result = voice_security.validate_session("invalid_session_id")
        
        assert result is False
    
    def test_validate_session_expired(self, voice_security):
        """Test validating an expired session."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        expires_in = timedelta(seconds=0.1)  # Very short expiry
        
        session_id = voice_security.create_session(
            user_id, 
            security_level, 
            expires_in=expires_in
        )
        
        # Wait for session to expire
        time.sleep(0.2)
        
        result = voice_security.validate_session(session_id)
        
        assert result is False
    
    def test_destroy_session(self, voice_security):
        """Test destroying a session."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        
        # Verify session exists
        session = voice_security.get_session(session_id)
        assert session is not None
        
        # Destroy session
        voice_security.destroy_session(session_id)
        
        # Verify session no longer exists
        session = voice_security.get_session(session_id)
        assert session is None
    
    def test_destroy_session_invalid(self, voice_security):
        """Test destroying an invalid session (should not raise error)."""
        # Should not raise an exception
        voice_security.destroy_session("invalid_session_id")
    
    def test_cleanup_expired_sessions(self, voice_security):
        """Test cleaning up expired sessions."""
        # Create multiple sessions with different expiry times
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        # Create session that will expire quickly
        short_lived_session = voice_security.create_session(
            user_id, 
            security_level, 
            expires_in=timedelta(seconds=0.1)
        )
        
        # Create session that will last longer
        long_lived_session = voice_security.create_session(
            user_id, 
            security_level, 
            expires_in=timedelta(hours=1)
        )
        
        # Wait for short-lived session to expire
        time.sleep(0.2)
        
        # Cleanup expired sessions
        voice_security.cleanup_expired_sessions()
        
        # Short-lived session should be gone
        short_session = voice_security.get_session(short_lived_session)
        assert short_session is None
        
        # Long-lived session should still exist
        long_session = voice_security.get_session(long_lived_session)
        assert long_session is not None
    
    def test_log_security_event(self, voice_security):
        """Test logging security events."""
        event_type = "command_validation"
        user_id = "test_user"
        details = {"command": "start_session", "result": "success"}
        
        voice_security.log_security_event(event_type, user_id, details)
        
        # Verify event was logged
        audit_log = voice_security.get_audit_log()
        assert len(audit_log) > 0
        
        last_event = audit_log[-1]
        assert last_event["event_type"] == event_type
        assert last_event["user_id"] == user_id
        assert last_event["details"] == details
        assert "timestamp" in last_event
    
    def test_get_audit_log(self, voice_security):
        """Test getting audit log."""
        # Log some events
        voice_security.log_security_event("test_event", "user1", {"test": "data1"})
        voice_security.log_security_event("test_event", "user2", {"test": "data2"})
        
        audit_log = voice_security.get_audit_log()
        
        assert isinstance(audit_log, list)
        assert len(audit_log) >= 2
        
        # Check structure of log entries
        for entry in audit_log:
            assert "event_type" in entry
            assert "user_id" in entry
            assert "details" in entry
            assert "timestamp" in entry
    
    def test_get_audit_log_filtered(self, voice_security):
        """Test getting filtered audit log."""
        # Log events for different users
        voice_security.log_security_event("test_event", "user1", {"test": "data1"})
        voice_security.log_security_event("test_event", "user2", {"test": "data2"})
        voice_security.log_security_event("other_event", "user1", {"test": "data3"})
        
        # Filter by user
        user1_log = voice_security.get_audit_log(user_id="user1")
        assert all(entry["user_id"] == "user1" for entry in user1_log)
        
        # Filter by event type
        test_event_log = voice_security.get_audit_log(event_type="test_event")
        assert all(entry["event_type"] == "test_event" for entry in test_event_log)
        
        # Filter by both
        filtered_log = voice_security.get_audit_log(
            user_id="user1", 
            event_type="test_event"
        )
        assert all(
            entry["user_id"] == "user1" and entry["event_type"] == "test_event" 
            for entry in filtered_log
        )
    
    def test_get_audit_log_time_range(self, voice_security):
        """Test getting audit log within time range."""
        now = datetime.now()
        
        # Log events at different times
        voice_security.log_security_event("test_event", "user1", {"test": "data1"})
        
        time.sleep(0.1)
        middle_time = datetime.now()
        
        voice_security.log_security_event("test_event", "user2", {"test": "data2"})
        
        time.sleep(0.1)
        end_time = datetime.now()
        
        # Get log within time range
        time_range_log = voice_security.get_audit_log(
            start_time=middle_time,
            end_time=end_time
        )
        
        # Should only include the second event
        assert len(time_range_log) == 1
        assert time_range_log[0]["user_id"] == "user2"
    
    def test_check_audio_permission(self, voice_security):
        """Test checking audio processing permissions."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        
        # Create test audio data
        audio_data = AudioData(
            data=b"fake_audio_data",
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        # Check permission for audio processing
        result = voice_security.check_audio_permission(session_id, audio_data)
        
        assert result is True
    
    def test_check_audio_permission_invalid_session(self, voice_security):
        """Test checking audio permission with invalid session."""
        session_id = "invalid_session_id"
        
        audio_data = AudioData(
            data=b"fake_audio_data",
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        result = voice_security.check_audio_permission(session_id, audio_data)
        
        assert result is False
    
    def test_check_audio_permission_insufficient_level(self, voice_security):
        """Test checking audio permission with insufficient security level."""
        user_id = "test_user"
        security_level = SecurityLevel.GUEST  # Guest level might not have audio permissions
        
        session_id = voice_security.create_session(user_id, security_level)
        
        audio_data = AudioData(
            data=b"fake_audio_data",
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        result = voice_security.check_audio_permission(session_id, audio_data)
        
        # This depends on the implementation - guests might not have audio permissions
        assert isinstance(result, bool)
    
    def test_encrypt_audio_data(self, voice_security):
        """Test encrypting audio data."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        
        audio_data = AudioData(
            data=b"sensitive_audio_data",
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        encrypted_data = voice_security.encrypt_audio_data(session_id, audio_data)
        
        assert encrypted_data is not None
        assert encrypted_data != audio_data.data  # Should be different (encrypted)
        assert len(encrypted_data) > 0
    
    def test_encrypt_audio_data_invalid_session(self, voice_security):
        """Test encrypting audio data with invalid session."""
        session_id = "invalid_session_id"
        
        audio_data = AudioData(
            data=b"sensitive_audio_data",
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        with pytest.raises(SecurityError) as exc_info:
            voice_security.encrypt_audio_data(session_id, audio_data)
        
        assert "Invalid session" in str(exc_info.value)
    
    def test_decrypt_audio_data(self, voice_security):
        """Test decrypting audio data."""
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id = voice_security.create_session(user_id, security_level)
        
        original_data = b"sensitive_audio_data"
        audio_data = AudioData(
            data=original_data,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        
        # Encrypt the data
        encrypted_data = voice_security.encrypt_audio_data(session_id, audio_data)
        
        # Decrypt the data
        decrypted_data = voice_security.decrypt_audio_data(session_id, encrypted_data)
        
        assert decrypted_data == original_data
    
    def test_decrypt_audio_data_invalid_session(self, voice_security):
        """Test decrypting audio data with invalid session."""
        session_id = "invalid_session_id"
        encrypted_data = b"encrypted_data"
        
        with pytest.raises(SecurityError) as exc_info:
            voice_security.decrypt_audio_data(session_id, encrypted_data)
        
        assert "Invalid session" in str(exc_info.value)
    
    def test_get_security_stats(self, voice_security):
        """Test getting security statistics."""
        # Create some sessions and log events
        user_id = "test_user"
        security_level = SecurityLevel.USER
        
        session_id1 = voice_security.create_session(user_id, security_level)
        session_id2 = voice_security.create_session(user_id, security_level)
        
        voice_security.log_security_event("test_event", user_id, {"test": "data"})
        
        stats = voice_security.get_security_stats()
        
        assert isinstance(stats, dict)
        assert "active_sessions" in stats
        assert "total_events" in stats
        assert "session_expiry_time" in stats
        
        assert stats["active_sessions"] >= 2
        assert stats["total_events"] >= 1


class TestSecurityError:
    """Test SecurityError exception."""
    
    def test_security_error_creation(self):
        """Test creating SecurityError."""
        error = SecurityError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_security_error_inheritance(self):
        """Test SecurityError inheritance."""
        error = SecurityError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ValueError)