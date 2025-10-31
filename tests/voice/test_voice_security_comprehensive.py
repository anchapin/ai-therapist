"""
Comprehensive security tests for voice/security.py
Target: 80%+ coverage with focus on security-critical paths
"""

import pytest
import hashlib
import time
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import security module directly to avoid torch initialization issues
import importlib.util
spec = importlib.util.spec_from_file_location("voice.security", "voice/security.py")
voice_security_module = importlib.util.module_from_spec(spec)
sys.modules['voice.security'] = voice_security_module
spec.loader.exec_module(voice_security_module)

VoiceSecurity = voice_security_module.VoiceSecurity
SecurityConfig = voice_security_module.SecurityConfig
SecurityError = voice_security_module.SecurityError
ConsentRecord = voice_security_module.ConsentRecord
AuditLogEntry = voice_security_module.AuditLogEntry
AuditLogger = voice_security_module.AuditLogger
ConsentManager = voice_security_module.ConsentManager
AccessManager = voice_security_module.AccessManager
EmergencyProtocolManager = voice_security_module.EmergencyProtocolManager
DataRetentionManager = voice_security_module.DataRetentionManager


class TestVoiceSecurityInitialization:
    """Test VoiceSecurity initialization and configuration"""
    
    def test_initialization_default_config(self):
        """Test initialization with default config"""
        security = VoiceSecurity()
        assert security.initialized is True
        assert security.session_timeout_minutes == 30
        assert security.encryption_key_rotation_days == 90
        
    def test_initialization_custom_config(self):
        """Test initialization with custom config"""
        config = SecurityConfig(
            encryption_enabled=True,
            consent_required=True,
            session_timeout_minutes=60,
            encryption_key_rotation_days=30
        )
        security = VoiceSecurity(config)
        assert security.session_timeout_minutes == 60
        assert security.encryption_key_rotation_days == 30
        
    def test_initialize_method(self):
        """Test initialize method returns True"""
        security = VoiceSecurity()
        result = security.initialize()
        assert result is True
        

class TestEncryption:
    """Test encryption and decryption functionality"""
    
    def test_encrypt_data_success(self):
        """Test successful data encryption"""
        security = VoiceSecurity()
        data = b"sensitive_data"
        user_id = "user123"
        
        encrypted = security.encrypt_data(data, user_id)
        assert encrypted != data
        assert isinstance(encrypted, bytes)
        
    def test_encrypt_data_none_input(self):
        """Test encryption with None data raises TypeError"""
        security = VoiceSecurity()
        with pytest.raises(TypeError, match="Data cannot be None"):
            security.encrypt_data(None, "user123")
            
    def test_encrypt_data_invalid_type(self):
        """Test encryption with non-bytes data raises TypeError"""
        security = VoiceSecurity()
        with pytest.raises(TypeError, match="Data must be bytes"):
            security.encrypt_data("string_data", "user123")
            
    def test_encrypt_data_invalid_user_id(self):
        """Test encryption with invalid user_id raises TypeError"""
        security = VoiceSecurity()
        with pytest.raises(TypeError, match="User ID must be string"):
            security.encrypt_data(b"data", 123)
            
    def test_encrypt_data_when_disabled(self):
        """Test encryption returns plain data when disabled"""
        config = SecurityConfig(encryption_enabled=False)
        security = VoiceSecurity(config)
        data = b"sensitive_data"
        
        result = security.encrypt_data(data, "user123")
        assert result == data
        
    def test_encrypt_data_without_master_key(self):
        """Test encryption raises SecurityError when master key unavailable"""
        config = SecurityConfig(encryption_enabled=True)
        security = VoiceSecurity(config)
        security.master_key = None
        
        with pytest.raises(SecurityError, match="Encryption key not available"):
            security.encrypt_data(b"data", "user123")
            
    def test_decrypt_data_success(self):
        """Test successful data decryption"""
        security = VoiceSecurity()
        data = b"sensitive_data"
        user_id = "user123"
        
        encrypted = security.encrypt_data(data, user_id)
        decrypted = security.decrypt_data(encrypted, user_id)
        assert decrypted == data
        
    def test_decrypt_data_none_input(self):
        """Test decryption with None data raises TypeError"""
        security = VoiceSecurity()
        with pytest.raises(TypeError, match="Encrypted data cannot be None"):
            security.decrypt_data(None, "user123")
            
    def test_decrypt_data_invalid_type(self):
        """Test decryption with non-bytes data raises TypeError"""
        security = VoiceSecurity()
        with pytest.raises(TypeError, match="Encrypted data must be bytes"):
            security.decrypt_data("string_data", "user123")
            
    def test_decrypt_data_invalid_user_id(self):
        """Test decryption with invalid user_id raises TypeError"""
        security = VoiceSecurity()
        with pytest.raises(TypeError, match="User ID must be string"):
            security.decrypt_data(b"data", 123)
            
    def test_decrypt_data_when_disabled(self):
        """Test decryption returns data when encryption disabled"""
        config = SecurityConfig(encryption_enabled=False)
        security = VoiceSecurity(config)
        data = b"encrypted_data"
        
        result = security.decrypt_data(data, "user123")
        assert result == data
        
    def test_decrypt_data_key_rotation_expired(self):
        """Test decryption fails when encryption key has expired"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock decryption path
        
        # Set key creation time to be older than rotation days
        security.key_creation_time = datetime.now() - timedelta(days=100)
        security.encryption_key_rotation_days = 90
        
        with pytest.raises(SecurityError, match="Encryption key has expired"):
            security.decrypt_data(b"mock_encrypted_data", "user123")
            
    def test_decrypt_mock_data_unauthorized_user(self):
        """Test decryption fails for unauthorized user"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock decryption
        
        # Encrypt with one user
        encrypted = b"mock_encrypted_sensitive_voice_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        # Try to decrypt with different user
        with pytest.raises(ValueError, match="not authorized to decrypt"):
            security.decrypt_data(encrypted, "user456")
            

class TestVoiceTranscriptionFiltering:
    """Test voice transcription filtering and PII detection"""
    
    def test_filter_voice_transcription_basic(self):
        """Test basic transcription filtering"""
        security = VoiceSecurity()
        result = security.filter_voice_transcription(
            "Hello, my name is John",
            "user123",
            "session456"
        )
        
        assert "original_transcription" in result
        assert "filtered_transcription" in result
        assert "user_id" in result
        assert result["user_id"] == "user123"
        assert result["session_id"] == "session456"
        
    def test_filter_voice_transcription_crisis_detection(self):
        """Test crisis keyword detection in transcription"""
        security = VoiceSecurity()
        result = security.filter_voice_transcription(
            "I want to kill myself",
            "user123",
            "session456"
        )
        
        assert result["crisis_detected"] is True
        
    def test_filter_voice_transcription_no_crisis(self):
        """Test non-crisis transcription"""
        security = VoiceSecurity()
        result = security.filter_voice_transcription(
            "I'm feeling much better today",
            "user123",
            "session456"
        )
        
        assert result["crisis_detected"] is False
        
    def test_filter_voice_transcription_error_handling(self):
        """Test error handling in transcription filtering"""
        security = VoiceSecurity()
        
        # Mock an exception in the filtering process using the module instance
        with patch.object(security, '_detect_crisis_keywords', side_effect=Exception("Test error")):
            result = security.filter_voice_transcription(
                "test transcription",
                "user123",
                "session456"
            )
            
            assert "error" in result
            assert result["filtered_transcription"] == "[TRANSCRIPTION FILTERING ERROR]"
            
    def test_detect_crisis_keywords_suicide(self):
        """Test crisis keyword detection for suicide"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("I want to commit suicide") is True
        
    def test_detect_crisis_keywords_self_harm(self):
        """Test crisis keyword detection for self harm"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("I want to harm myself") is True
        
    def test_detect_crisis_keywords_end_life(self):
        """Test crisis keyword detection for end life"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("I want to end my life") is True
        
    def test_detect_crisis_keywords_normal_text(self):
        """Test no crisis detection in normal text"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("I'm having a good day") is False
        
    def test_get_user_role_admin(self):
        """Test user role detection for admin"""
        security = VoiceSecurity()
        assert security._get_user_role("admin_user") == "admin"
        
    def test_get_user_role_therapist(self):
        """Test user role detection for therapist"""
        security = VoiceSecurity()
        assert security._get_user_role("therapist_user") == "therapist"
        
    def test_get_user_role_patient(self):
        """Test user role detection for patient"""
        security = VoiceSecurity()
        assert security._get_user_role("patient_user") == "patient"
        
    def test_get_user_role_guest(self):
        """Test user role detection for guest"""
        security = VoiceSecurity()
        assert security._get_user_role("unknown_user") == "guest"
        

class TestAuditLogging:
    """Test audit logging functionality"""
    
    def test_audit_logger_log_event(self):
        """Test basic event logging"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="VOICE_INPUT",
            session_id="session123",
            user_id="user456",
            details={"action": "test"}
        )
        
        assert "event_id" in result
        assert result["event_type"] == "VOICE_INPUT"
        assert result["user_id"] == "user456"
        assert result["session_id"] == "session123"
        
    def test_audit_logger_phi_access_event(self):
        """Test PHI access event logging with HIPAA fields"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="PHI_ACCESS",
            user_id="user123",
            details={"resource": "patient_data"}
        )
        
        assert result["event_type"] == "PHI_ACCESS"
        assert result["details"]["action"] == "access"
        assert result["details"]["purpose"] == "treatment"
        
    def test_audit_logger_phi_modification_event(self):
        """Test PHI modification event logging"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="PHI_MODIFICATION",
            user_id="user123",
            details={"resource": "patient_record"}
        )
        
        assert result["details"]["action"] == "modify"
        
    def test_audit_logger_phi_disclosure_event(self):
        """Test PHI disclosure event logging"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="PHI_DISCLOSURE",
            user_id="user123"
        )
        
        assert result["details"]["action"] == "disclose"
        
    def test_audit_logger_phi_deletion_event(self):
        """Test PHI deletion event logging"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="PHI_DELETION",
            user_id="user123"
        )
        
        assert result["details"]["action"] == "delete"
        
    def test_audit_logger_get_session_logs(self):
        """Test retrieving session logs"""
        security = VoiceSecurity()
        session_id = "session789"
        
        # Log some events
        security.audit_logger.log_event("EVENT1", session_id=session_id, user_id="user123")
        security.audit_logger.log_event("EVENT2", session_id=session_id, user_id="user123")
        
        logs = security.audit_logger.get_session_logs(session_id)
        assert len(logs) == 2
        
    def test_audit_logger_get_session_logs_empty(self):
        """Test retrieving logs for nonexistent session"""
        security = VoiceSecurity()
        logs = security.audit_logger.get_session_logs("nonexistent_session")
        assert len(logs) == 0
        
    def test_audit_logger_get_session_logs_test_session(self):
        """Test mock logs for test session"""
        security = VoiceSecurity()
        logs = security.audit_logger.get_session_logs("test_session_123")
        assert len(logs) == 5
        
    def test_audit_logger_get_logs_in_date_range(self):
        """Test retrieving logs within date range"""
        security = VoiceSecurity()
        
        # Log some events
        security.audit_logger.log_event("EVENT1", user_id="user123")
        time.sleep(0.1)
        security.audit_logger.log_event("EVENT2", user_id="user123")
        
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        logs = security.audit_logger.get_logs_in_date_range(start_date, end_date)
        assert len(logs) >= 2
        
    def test_audit_logger_get_user_logs(self):
        """Test retrieving all logs for a user"""
        security = VoiceSecurity()
        user_id = "user999"
        
        # Log events for specific user
        security.audit_logger.log_event("EVENT1", user_id=user_id)
        security.audit_logger.log_event("EVENT2", user_id=user_id)
        security.audit_logger.log_event("EVENT3", user_id="other_user")
        
        user_logs = security.audit_logger.get_user_logs(user_id)
        assert len(user_logs) >= 2
        assert all(log["user_id"] == user_id for log in user_logs)
        

class TestConsentManagement:
    """Test consent management functionality"""
    
    def test_consent_record_consent(self):
        """Test recording user consent"""
        security = VoiceSecurity()
        result = security.consent_manager.record_consent(
            user_id="user123",
            consent_type="voice_recording",
            granted=True,
            version="1.0"
        )
        
        assert result["user_id"] == "user123"
        assert result["consent_type"] == "voice_recording"
        assert result["granted"] is True
        
    def test_consent_has_consent_granted(self):
        """Test checking granted consent"""
        security = VoiceSecurity()
        security.consent_manager.record_consent(
            user_id="user123",
            consent_type="voice_recording",
            granted=True
        )
        
        assert security.consent_manager.has_consent("user123", "voice_recording") is True
        
    def test_consent_has_consent_not_granted(self):
        """Test checking non-granted consent"""
        security = VoiceSecurity()
        security.consent_manager.record_consent(
            user_id="user123",
            consent_type="voice_recording",
            granted=False
        )
        
        assert security.consent_manager.has_consent("user123", "voice_recording") is False
        
    def test_consent_has_consent_nonexistent(self):
        """Test checking nonexistent consent"""
        security = VoiceSecurity()
        assert security.consent_manager.has_consent("user999", "nonexistent") is False
        
    def test_consent_withdraw_consent(self):
        """Test withdrawing consent"""
        security = VoiceSecurity()
        
        # First grant consent
        security.consent_manager.record_consent(
            user_id="user123",
            consent_type="voice_recording",
            granted=True
        )
        
        # Then withdraw it
        security.consent_manager.withdraw_consent("user123", "voice_recording")
        
        assert security.consent_manager.has_consent("user123", "voice_recording") is False
        
    def test_consent_withdraw_nonexistent(self):
        """Test withdrawing nonexistent consent"""
        security = VoiceSecurity()
        # Should not raise an error
        security.consent_manager.withdraw_consent("user999", "nonexistent")
        

class TestAccessControl:
    """Test access control functionality"""
    
    def test_access_grant_access(self):
        """Test granting access to resource"""
        security = VoiceSecurity()
        security.access_manager.grant_access("user123", "resource456", "read")
        
        assert security.access_manager.has_access("user123", "resource456", "read") is True
        
    def test_access_has_access_denied(self):
        """Test access denied for non-granted permission"""
        security = VoiceSecurity()
        assert security.access_manager.has_access("user123", "resource456", "write") is False
        
    def test_access_has_access_multiple_permissions(self):
        """Test multiple permissions for same resource"""
        security = VoiceSecurity()
        security.access_manager.grant_access("user123", "resource456", "read")
        security.access_manager.grant_access("user123", "resource456", "write")
        
        assert security.access_manager.has_access("user123", "resource456", "read") is True
        assert security.access_manager.has_access("user123", "resource456", "write") is True
        
    def test_access_revoke_access(self):
        """Test revoking access to resource"""
        security = VoiceSecurity()
        
        # First grant access
        security.access_manager.grant_access("user123", "resource456", "read")
        assert security.access_manager.has_access("user123", "resource456", "read") is True
        
        # Then revoke it
        security.access_manager.revoke_access("user123", "resource456", "read")
        assert security.access_manager.has_access("user123", "resource456", "read") is False
        
    def test_access_revoke_nonexistent(self):
        """Test revoking nonexistent access"""
        security = VoiceSecurity()
        # Should not raise an error
        security.access_manager.revoke_access("user999", "resource999", "read")
        

class TestDataRetention:
    """Test data retention functionality"""
    
    def test_retention_apply_policy(self):
        """Test applying data retention policy"""
        config = SecurityConfig(data_retention_days=1)
        security = VoiceSecurity(config)
        
        # Create old log entry
        old_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
        old_log = {
            "event_id": "old_event",
            "timestamp": old_timestamp,
            "event_type": "TEST",
            "user_id": "user123",
            "details": {}
        }
        security.audit_logger.logs.append(old_log)
        
        # Apply retention policy
        removed_count = security.retention_manager.apply_retention_policy()
        
        assert removed_count >= 1
        
    def test_retention_cleanup_expired_data(self):
        """Test cleanup expired data method"""
        security = VoiceSecurity()
        # Should not raise an error
        security.retention_manager.cleanup_expired_data()
        

class TestSecurityMetrics:
    """Test security metrics and reporting"""
    
    def test_get_security_metrics(self):
        """Test retrieving security metrics"""
        security = VoiceSecurity()
        
        # Log some events
        security.audit_logger.log_event("EVENT1", user_id="user123")
        security.audit_logger.log_event("EVENT2", user_id="user456")
        
        metrics = security.get_security_metrics()
        
        assert "total_events" in metrics
        assert "unique_users" in metrics
        assert "security_incidents" in metrics
        assert "compliance_score" in metrics
        assert metrics["total_events"] >= 2
        
    def test_get_penetration_testing_scope(self):
        """Test penetration testing scope configuration"""
        security = VoiceSecurity()
        scope = security.get_penetration_testing_scope()
        
        assert "target_systems" in scope
        assert "test_scenarios" in scope
        assert "excluded_areas" in scope
        assert "authorization_requirements" in scope
        assert "voice_api" in scope["target_systems"]
        

class TestSecurityProperties:
    """Test security configuration properties"""
    
    def test_property_encryption_enabled(self):
        """Test encryption_enabled property"""
        config = SecurityConfig(encryption_enabled=True)
        security = VoiceSecurity(config)
        assert security.encryption_enabled is True
        
    def test_property_consent_required(self):
        """Test consent_required property"""
        config = SecurityConfig(consent_required=True)
        security = VoiceSecurity(config)
        assert security.consent_required is True
        
    def test_property_privacy_mode(self):
        """Test privacy_mode property"""
        config = SecurityConfig(privacy_mode=True)
        security = VoiceSecurity(config)
        assert security.privacy_mode is True
        
    def test_property_audit_logging_enabled(self):
        """Test audit_logging_enabled property"""
        config = SecurityConfig(audit_logging_enabled=True)
        security = VoiceSecurity(config)
        assert security.audit_logging_enabled is True
        
    def test_property_data_retention_days(self):
        """Test data_retention_days property"""
        config = SecurityConfig(data_retention_days=60)
        security = VoiceSecurity(config)
        assert security.data_retention_days == 60
        

class TestEmergencyProtocols:
    """Test emergency protocol management"""
    
    def test_trigger_emergency_protocol(self):
        """Test triggering emergency protocol"""
        security = VoiceSecurity()
        # Should not raise an error
        security.emergency_manager.trigger_emergency_protocol(
            incident_type="CRISIS_DETECTED",
            details={"severity": "high"}
        )
        

class TestCompatibilityMethods:
    """Test compatibility methods for tests"""
    
    def test_check_consent_status(self):
        """Test _check_consent_status compatibility method"""
        security = VoiceSecurity()
        result = security._check_consent_status("user123", "session456")
        assert result is True
        
    def test_verify_security_requirements(self):
        """Test _verify_security_requirements compatibility method"""
        security = VoiceSecurity()
        result = security._verify_security_requirements("user123", "session456")
        assert result is True
        
    @pytest.mark.asyncio
    async def test_process_audio(self):
        """Test process_audio compatibility method"""
        security = VoiceSecurity()
        audio_data = b"audio_bytes"
        result = await security.process_audio(audio_data)
        assert result == audio_data
        

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_encrypt_empty_data(self):
        """Test encrypting empty data"""
        security = VoiceSecurity()
        encrypted = security.encrypt_data(b"", "user123")
        assert isinstance(encrypted, bytes)
        
    def test_decrypt_empty_data_when_disabled(self):
        """Test decrypting empty data when encryption disabled"""
        config = SecurityConfig(encryption_enabled=False)
        security = VoiceSecurity(config)
        result = security.decrypt_data(b"", "user123")
        assert result == b""
        
    def test_filter_empty_transcription(self):
        """Test filtering empty transcription"""
        security = VoiceSecurity()
        result = security.filter_voice_transcription("", "user123", "session456")
        assert "original_transcription" in result
        assert result["original_transcription"] == ""
        
    def test_audit_log_with_none_details(self):
        """Test audit logging with None details"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="TEST_EVENT",
            details=None
        )
        assert result["details"] == {}
        
    def test_consent_record_with_details(self):
        """Test recording consent with additional details"""
        security = VoiceSecurity()
        details = {"ip_address": "192.168.1.1", "device": "mobile"}
        result = security.consent_manager.record_consent(
            user_id="user123",
            consent_type="data_processing",
            granted=True,
            details=details
        )
        assert result["details"] == details
        

class TestSecurityIntegration:
    """Test integration between security components"""
    
    def test_encrypt_decrypt_with_audit_logging(self):
        """Test encryption/decryption creates audit logs"""
        security = VoiceSecurity()
        user_id = "user123"
        data = b"sensitive_data"
        
        initial_log_count = len(security.audit_logger.logs)
        
        # Encrypt data
        encrypted = security.encrypt_data(data, user_id)
        
        # Should have logged encryption event
        assert len(security.audit_logger.logs) > initial_log_count
        
    def test_consent_with_audit_logging(self):
        """Test consent management creates audit logs"""
        security = VoiceSecurity()
        
        initial_log_count = len(security.audit_logger.logs)
        
        # Record consent
        security.consent_manager.record_consent(
            user_id="user123",
            consent_type="voice_recording",
            granted=True
        )
        
        # Should have logged consent event
        assert len(security.audit_logger.logs) > initial_log_count
        
    def test_access_control_with_audit_logging(self):
        """Test access control creates audit logs"""
        security = VoiceSecurity()
        
        initial_log_count = len(security.audit_logger.logs)
        
        # Grant and check access
        security.access_manager.grant_access("user123", "resource456", "read")
        security.access_manager.has_access("user123", "resource456", "read")
        
        # Should have logged access events
        assert len(security.audit_logger.logs) > initial_log_count


class TestDecryptionEdgeCases:
    """Test decryption edge cases and complex scenarios"""
    
    def test_decrypt_mock_encrypted_empty_data(self):
        """Test decrypting mock encrypted empty data"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock path
        
        encrypted = b"mock_encrypted_empty_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        result = security.decrypt_data(encrypted, "user123")
        assert result == b""
        
    def test_decrypt_mock_encrypted_single_byte(self):
        """Test decrypting mock encrypted single byte"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock path
        
        # Create a mock encrypted single byte in the new format
        # Format: "mock_encrypted_" + xored_byte + entropy
        user_id = "user123"
        user_seed = hashlib.sha256(f"{user_id}_1".encode()).digest()
        deterministic_sources = [
            user_seed[0], user_seed[1], user_seed[2], user_seed[3],
            hashlib.sha256(user_seed).digest()[0]
        ]
        
        original_byte = 42
        xored_byte = original_byte
        for rand_byte in deterministic_sources:
            xored_byte ^= rand_byte
            
        encrypted = b"mock_encrypted_" + bytes([xored_byte]) + b"0" * 50
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = user_id
        
        result = security.decrypt_data(encrypted, user_id)
        assert result == bytes([original_byte])
        
    def test_decrypt_mock_encrypted_multibyte(self):
        """Test decrypting mock encrypted multi-byte data"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock path
        
        user_id = "user123"
        original_data = b"test_data_12345"
        
        # Create mock encrypted multi-byte format
        # Format: "mock_encrypted_" + length(4) + xored_data + key + entropy(64)
        import secrets
        xor_key = secrets.token_bytes(len(original_data))
        xored_data = bytes([original_data[i] ^ xor_key[i] for i in range(len(original_data))])
        
        length_bytes = len(original_data).to_bytes(4, 'big')
        entropy = secrets.token_bytes(64)
        
        encrypted = b"mock_encrypted_" + length_bytes + xored_data + xor_key + entropy
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = user_id
        
        result = security.decrypt_data(encrypted, user_id)
        assert result == original_data
        
    def test_decrypt_mock_fallback_format(self):
        """Test decrypting with fallback old format"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock path
        
        # Create encrypted data that doesn't match new format (too short)
        encrypted = b"mock_encrypted_short"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        # Should fall back to old format handling
        result = security.decrypt_data(encrypted, "user123")
        assert isinstance(result, bytes)


class TestEncryptionInitialization:
    """Test encryption initialization paths"""
    
    def test_initialize_encryption_with_mock(self):
        """Test encryption initialization in mock mode"""
        config = SecurityConfig(encryption_enabled=True)
        security = VoiceSecurity(config)
        
        # When CRYPTOGRAPHY_AVAILABLE is False, master_key should be None
        if not voice_security_module.CRYPTOGRAPHY_AVAILABLE:
            assert security.master_key is None
            
    def test_get_current_time(self):
        """Test _get_current_time helper method"""
        security = VoiceSecurity()
        current_time = security._get_current_time()
        assert isinstance(current_time, datetime)


class TestAuditLoggerEdgeCases:
    """Test audit logger edge cases"""
    
    def test_audit_logger_custom_phi_event(self):
        """Test PHI event with custom action"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="PHI_CUSTOM_ACTION",
            user_id="user123",
            details={"resource": "patient_data"}
        )
        
        assert result["event_type"] == "PHI_CUSTOM_ACTION"
        assert result["details"]["action"] == "custom_action"
        
    def test_audit_logger_phi_event_with_purpose(self):
        """Test PHI event with custom purpose"""
        security = VoiceSecurity()
        result = security.audit_logger.log_event(
            event_type="PHI_ACCESS",
            user_id="user123",
            details={"purpose": "diagnosis"}
        )
        
        assert result["details"]["purpose"] == "diagnosis"
        
    def test_audit_logger_get_logs_in_date_range_regular(self):
        """Test retrieving logs with regular timestamps"""
        security = VoiceSecurity()
        
        # Log an event (creates regular timestamp)
        security.audit_logger.log_event(
            event_type="TEST_EVENT",
            user_id="user123",
            details={"test": "data"}
        )
        
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        logs = security.audit_logger.get_logs_in_date_range(start_date, end_date)
        assert len(logs) >= 1


class TestConsentManagerEdgeCases:
    """Test consent manager edge cases"""
    
    def test_consent_record_multiple_versions(self):
        """Test recording consent with multiple versions"""
        security = VoiceSecurity()
        
        # Record v1.0
        security.consent_manager.record_consent(
            user_id="user123",
            consent_type="data_processing",
            granted=True,
            version="1.0"
        )
        
        # Record v2.0 (should overwrite)
        security.consent_manager.record_consent(
            user_id="user123",
            consent_type="data_processing",
            granted=False,
            version="2.0"
        )
        
        # Should have latest version
        assert security.consent_manager.has_consent("user123", "data_processing") is False


class TestAccessManagerEdgeCases:
    """Test access manager edge cases"""
    
    def test_access_grant_multiple_resources(self):
        """Test granting access to multiple resources"""
        security = VoiceSecurity()
        
        security.access_manager.grant_access("user123", "resource1", "read")
        security.access_manager.grant_access("user123", "resource2", "write")
        
        assert security.access_manager.has_access("user123", "resource1", "read")
        assert security.access_manager.has_access("user123", "resource2", "write")
        assert not security.access_manager.has_access("user123", "resource1", "write")
        
    def test_access_revoke_specific_permission(self):
        """Test revoking specific permission while keeping others"""
        security = VoiceSecurity()
        
        # Grant multiple permissions
        security.access_manager.grant_access("user123", "resource1", "read")
        security.access_manager.grant_access("user123", "resource1", "write")
        
        # Revoke one
        security.access_manager.revoke_access("user123", "resource1", "write")
        
        # Read should still work
        assert security.access_manager.has_access("user123", "resource1", "read")
        assert not security.access_manager.has_access("user123", "resource1", "write")


class TestDataRetentionEdgeCases:
    """Test data retention edge cases"""
    
    def test_retention_apply_policy_with_session_cache(self):
        """Test retention policy on session cache"""
        config = SecurityConfig(data_retention_days=1)
        security = VoiceSecurity(config)
        
        # Create old log in session cache
        old_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
        old_log = {
            "event_id": "old_event",
            "timestamp": old_timestamp,
            "event_type": "TEST",
            "user_id": "user123",
            "details": {}
        }
        
        session_id = "test_session"
        security.audit_logger.session_logs_cache[session_id] = [old_log]
        
        removed_count = security.retention_manager.apply_retention_policy()
        assert removed_count >= 1


class TestTranscriptionPIIDetection:
    """Test transcription PII detection scenarios"""
    
    def test_filter_transcription_with_pii_detection_failure(self):
        """Test graceful handling when PII detection fails"""
        security = VoiceSecurity()
        
        # The test should handle PII detection failures gracefully
        result = security.filter_voice_transcription(
            "My SSN is 123-45-6789",
            "user123",
            "session456"
        )
        
        assert "filtered_transcription" in result
        assert "pii_detected" in result


class TestAnonymization:
    """Test data anonymization functionality"""
    
    def test_anonymize_data_enabled(self):
        """Test data anonymization when enabled"""
        config = SecurityConfig(anonymization_enabled=True)
        security = VoiceSecurity(config)
        
        data = {
            "user_id": "user123",
            "session_id": "session456",
            "content": "test data"
        }
        
        anonymized = security.anonymize_data(data)
        
        assert anonymized["user_id"] != data["user_id"]
        assert anonymized["session_id"] != data["session_id"]
        assert anonymized["user_id"].startswith("user_")
        assert anonymized["session_id"].startswith("session_")
        assert anonymized["content"] == data["content"]
        
    def test_anonymize_data_disabled(self):
        """Test data anonymization when disabled"""
        config = SecurityConfig(anonymization_enabled=False)
        security = VoiceSecurity(config)
        
        data = {
            "user_id": "user123",
            "session_id": "session456"
        }
        
        anonymized = security.anonymize_data(data)
        assert anonymized == data
        
    def test_anonymize_data_with_privacy_mode(self):
        """Test data anonymization in privacy mode removes audio"""
        config = SecurityConfig(privacy_mode=True, anonymization_enabled=True)
        security = VoiceSecurity(config)
        
        data = {
            "user_id": "user123",
            "audio_data": b"audio_bytes"
        }
        
        anonymized = security.anonymize_data(data)
        assert "audio_data" not in anonymized
        assert "user_id" in anonymized


class TestAudioEncryptionDecryption:
    """Test audio-specific encryption/decryption"""
    
    def test_decrypt_audio_data_mock_encrypted(self):
        """Test decrypting mock encrypted audio data"""
        security = VoiceSecurity()
        
        encrypted = b"mock_encrypted_test_audio_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        result = security.decrypt_audio_data(encrypted, "user123")
        assert result == b"test_audio_data"
        
    def test_decrypt_audio_data_unauthorized(self):
        """Test decrypting audio with wrong user"""
        security = VoiceSecurity()
        
        encrypted = b"mock_encrypted_test_audio_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        with pytest.raises(ValueError, match="not authorized"):
            security.decrypt_audio_data(encrypted, "user456")


class TestSecurityConfigNoAttribute:
    """Test security configuration fallback behavior"""
    
    def test_encryption_enabled_default_no_config(self):
        """Test encryption_enabled defaults to True with no config attributes"""
        security = VoiceSecurity(None)
        assert security.encryption_enabled is True
        
    def test_consent_required_default_no_config(self):
        """Test consent_required defaults to True with no config"""
        security = VoiceSecurity(None)
        assert security.consent_required is True
        
    def test_privacy_mode_default_no_config(self):
        """Test privacy_mode defaults to False with no config"""
        security = VoiceSecurity(None)
        assert security.privacy_mode is False
        
    def test_audit_logging_default_no_config(self):
        """Test audit_logging_enabled defaults to True"""
        security = VoiceSecurity(None)
        assert security.audit_logging_enabled is True


class TestInternalStorage:
    """Test internal storage structures"""
    
    def test_test_backups_storage(self):
        """Test test_backups internal storage"""
        security = VoiceSecurity()
        
        backup_data = {"key": "value"}
        security._test_backups["test_id"] = backup_data
        
        assert "test_id" in security._test_backups
        assert security._test_backups["test_id"] == backup_data


class TestSecurityLogging:
    """Test security event logging"""
    
    def test_log_security_event_when_enabled(self):
        """Test logging security event when enabled"""
        config = SecurityConfig(audit_logging_enabled=True)
        security = VoiceSecurity(config)
        
        initial_count = len(security.audit_logger.logs)
        
        security._log_security_event(
            event_type="test_event",
            user_id="user123",
            action="test_action",
            resource="test_resource",
            result="success",
            details={"info": "test"}
        )
        
        # Should have logged event
        assert len(security.audit_logger.logs) > initial_count


class TestCrisisKeywordDetection:
    """Test all crisis keyword variations"""
    
    def test_detect_crisis_want_to_die(self):
        """Test crisis detection for 'want to die'"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("I want to die") is True
        
    def test_detect_crisis_emergency(self):
        """Test crisis detection for 'emergency'"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("This is an emergency") is True
        
    def test_detect_crisis_case_insensitive(self):
        """Test crisis detection is case insensitive"""
        security = VoiceSecurity()
        assert security._detect_crisis_keywords("I WANT TO KILL MYSELF") is True
        assert security._detect_crisis_keywords("Suicide") is True


class TestDecryptionFallbackPaths:
    """Test decryption fallback and old format handling"""
    
    def test_decrypt_old_format_with_parts(self):
        """Test decrypting old format with underscore parts"""
        security = VoiceSecurity()
        security.master_key = None  # Force mock path
        
        # Create old format: mock_encrypted_data_length_actualdata
        original_data = b"test123"
        length_bytes = len(original_data).to_bytes(4, 'big')
        encrypted = b"mock_encrypted_data_" + length_bytes + b"_" + original_data
        
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        result = security.decrypt_data(encrypted, "user123")
        assert isinstance(result, bytes)
        
    def test_decrypt_single_byte_invalid_format(self):
        """Test decryption with invalid single byte format"""
        security = VoiceSecurity()
        security.master_key = None
        
        # Too short for single byte format
        encrypted = b"mock_encrypted_"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        # Should fall back to old format
        result = security.decrypt_data(encrypted, "user123")
        assert isinstance(result, bytes)
        
    def test_decrypt_multibyte_invalid_length(self):
        """Test decryption with invalid multi-byte format"""
        security = VoiceSecurity()
        security.master_key = None
        
        # Create invalid format (length mismatch)
        encrypted = b"mock_encrypted_" + b"\x00\x00\x00\x10" + b"short"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        # Should fall back to old format
        result = security.decrypt_data(encrypted, "user123")
        assert isinstance(result, bytes)


class TestMockEncryptedDataVariants:
    """Test various mock encrypted data formats"""
    
    def test_decrypt_mock_encrypted_sensitive_data(self):
        """Test decrypting b'mock_encrypted_sensitive_data'"""
        security = VoiceSecurity()
        security.master_key = None
        
        encrypted = b"mock_encrypted_sensitive_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        result = security.decrypt_data(encrypted, "user123")
        assert result == b"sensitive_data"
        
    def test_decrypt_different_mock_user(self):
        """Test that different users can't decrypt each other's data"""
        security = VoiceSecurity()
        security.master_key = None
        
        encrypted = b"mock_encrypted_sensitive_voice_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user_original"
        
        with pytest.raises(ValueError, match="not authorized"):
            security.decrypt_data(encrypted, "user_different")


class TestEncryptionAuditTrail:
    """Test encryption operations create proper audit trails"""
    
    def test_encryption_creates_audit_log(self):
        """Test that encryption creates an audit log entry"""
        security = VoiceSecurity()
        data = b"test_data"
        user_id = "user123"
        
        initial_count = len(security.audit_logger.logs)
        security.encrypt_data(data, user_id)
        
        # Should have created audit log
        assert len(security.audit_logger.logs) > initial_count
        
        # Check log details
        last_log = security.audit_logger.logs[-1]
        assert last_log["event_type"] == "data_encryption"
        
    def test_decryption_creates_audit_log(self):
        """Test that decryption creates an audit log entry"""
        security = VoiceSecurity()
        security.master_key = None  # Use mock mode
        
        encrypted = b"mock_encrypted_sensitive_voice_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        initial_count = len(security.audit_logger.logs)
        security.decrypt_data(encrypted, "user123")
        
        # Should have created audit log
        assert len(security.audit_logger.logs) > initial_count


class TestAccessManagerMultipleScenarios:
    """Test complex access manager scenarios"""
    
    def test_access_check_for_nonexistent_user(self):
        """Test access check for user that was never granted access"""
        security = VoiceSecurity()
        
        has_access = security.access_manager.has_access("new_user", "resource1", "read")
        assert has_access is False
        
    def test_access_check_for_different_permission(self):
        """Test access check for permission that wasn't granted"""
        security = VoiceSecurity()
        
        security.access_manager.grant_access("user123", "resource1", "read")
        has_access = security.access_manager.has_access("user123", "resource1", "execute")
        
        assert has_access is False


class TestConsentWithdrawal:
    """Test consent withdrawal scenarios"""
    
    def test_withdraw_consent_updates_timestamp(self):
        """Test that withdrawing consent updates timestamp"""
        security = VoiceSecurity()
        
        # Grant consent
        security.consent_manager.record_consent("user123", "data_processing", True)
        
        # Get initial consent record
        initial_consent = security.consent_manager.consents["user123"]["data_processing"]
        initial_timestamp = initial_consent.timestamp
        
        # Wait a bit and withdraw
        time.sleep(0.1)
        security.consent_manager.withdraw_consent("user123", "data_processing")
        
        # Check that timestamp was updated
        updated_consent = security.consent_manager.consents["user123"]["data_processing"]
        assert updated_consent.timestamp > initial_timestamp
        assert updated_consent.granted is False


class TestEmergencyManager:
    """Test emergency protocol manager"""
    
    def test_trigger_emergency_with_details(self):
        """Test triggering emergency protocol with details"""
        security = VoiceSecurity()
        
        # Should not raise an error
        security.emergency_manager.trigger_emergency_protocol(
            incident_type="PATIENT_CRISIS",
            details={
                "severity": "critical",
                "user_id": "patient123",
                "reason": "suicidal ideation"
            }
        )


class TestAdditionalDecryptionPaths:
    """Test additional decryption code paths"""
    
    def test_decrypt_authorized_user_same_user(self):
        """Test decryption when user is the same as encrypted"""
        security = VoiceSecurity()
        security.master_key = None
        
        # Simulate encrypted data tracking
        encrypted = b"mock_encrypted_sensitive_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        # Decrypt with same user
        result = security.decrypt_data(encrypted, "user123")
        assert result == b"sensitive_data"


class TestAuditLogSessionCache:
    """Test audit log session cache functionality"""
    
    def test_session_logs_multiple_sessions(self):
        """Test session logs across multiple sessions"""
        security = VoiceSecurity()
        
        # Log events to different sessions
        security.audit_logger.log_event("EVENT1", session_id="session1", user_id="user1")
        security.audit_logger.log_event("EVENT2", session_id="session2", user_id="user2")
        security.audit_logger.log_event("EVENT3", session_id="session1", user_id="user1")
        
        # Check session1 has 2 events
        session1_logs = security.audit_logger.get_session_logs("session1")
        assert len(session1_logs) == 2
        
        # Check session2 has 1 event
        session2_logs = security.audit_logger.get_session_logs("session2")
        assert len(session2_logs) == 1


class TestDataRetentionWithBothCaches:
    """Test data retention with both session cache and main logs"""
    
    def test_retention_removes_from_both_lists(self):
        """Test retention policy removes from both session cache and main logs"""
        config = SecurityConfig(data_retention_days=1)
        security = VoiceSecurity(config)
        
        # Create old logs in both places
        old_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
        old_log1 = {
            "event_id": "old1",
            "timestamp": old_timestamp,
            "event_type": "TEST",
            "user_id": "user123",
            "details": {}
        }
        old_log2 = {
            "event_id": "old2",
            "timestamp": old_timestamp,
            "event_type": "TEST",
            "user_id": "user456",
            "details": {}
        }
        
        # Add to session cache
        security.audit_logger.session_logs_cache["old_session"] = [old_log1]
        # Add to main logs
        security.audit_logger.logs.append(old_log2)
        
        initial_total = len(security.audit_logger.logs) + len(security.audit_logger.session_logs_cache.get("old_session", []))
        
        # Apply retention
        removed = security.retention_manager.apply_retention_policy()
        
        # Should have removed at least 2
        assert removed >= 2


class TestAccessManagerUserWithoutRecords:
    """Test access manager with users without access records"""
    
    def test_has_access_user_without_any_records(self):
        """Test has_access for user with no access records at all"""
        security = VoiceSecurity()
        
        # Don't grant any access
        result = security.access_manager.has_access("new_user", "new_resource", "read")
        
        assert result is False


class TestConfigPropertyEdgeCases:
    """Test config property edge cases"""
    
    def test_anonymization_disabled_config(self):
        """Test anonymize_data when config has anonymization_enabled=False"""
        config = SecurityConfig(anonymization_enabled=False)
        security = VoiceSecurity(config)
        
        data = {"user_id": "user123", "data": "test"}
        result = security.anonymize_data(data)
        
        # Should return data unchanged
        assert result == data


class TestDecryptionUserAuthorizationChecks:
    """Test user authorization checks in decryption"""
    
    def test_decrypt_sensitive_voice_data_wrong_user(self):
        """Test decryption fails when wrong user tries to decrypt"""
        security = VoiceSecurity()
        security.master_key = None
        
        encrypted = b"mock_encrypted_sensitive_voice_data"
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "original_user"
        
        with pytest.raises(ValueError, match="not authorized"):
            security.decrypt_data(encrypted, "wrong_user")


class TestComplexDecryptionScenarios:
    """Test complex decryption scenarios"""
    
    def test_decrypt_with_invalid_multibyte_format_very_short(self):
        """Test decryption with very short invalid multi-byte format"""
        security = VoiceSecurity()
        security.master_key = None
        
        # Create data that's too short for the expected format
        encrypted = b"mock_encrypted_" + b"\x00\x00\x00"  # Only 3 bytes, need 4 for length
        data_hash = hashlib.sha256(encrypted).hexdigest()
        security.encrypted_data_tracking[data_hash] = "user123"
        
        # Should fall back gracefully
        result = security.decrypt_data(encrypted, "user123")
        assert isinstance(result, bytes)
