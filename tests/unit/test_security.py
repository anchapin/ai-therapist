#!/usr/bin/env python3
"""
Unit tests for security service to reach 90%+ coverage.
"""

import pytest
import asyncio
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import test utilities for safe module loading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.test_utils import (
    setup_voice_module_mocks,
    get_security_module
)

# Set up mocks
setup_voice_module_mocks(project_root)

# Import modules safely
security_module = get_security_module(project_root)

# Extract classes from the module
VoiceSecurity = security_module.VoiceSecurity
SecurityConfig = security_module.SecurityConfig
ConsentRecord = security_module.ConsentRecord
AuditLogEntry = security_module.AuditLogEntry
SecurityError = security_module.SecurityError


class TestSecurityConfig:
    """Tests for SecurityConfig class."""

    def test_security_config_creation(self):
        """Test security config creation with default values."""
        config = SecurityConfig()

        assert config.encryption_enabled == True
        assert config.consent_required == True
        assert config.privacy_mode == False
        assert config.hipaa_compliance_enabled == True
        assert config.data_retention_days == 30
        assert config.audit_logging_enabled == True
        assert config.session_timeout_minutes == 30
        assert config.max_login_attempts == 3
        assert config.encryption_key_rotation_days == 90
        assert config.backup_encryption_enabled == True
        assert config.anonymization_enabled == True

    def test_security_config_custom_values(self):
        """Test security config creation with custom values."""
        config = SecurityConfig(
            encryption_enabled=False,
            consent_required=False,
            privacy_mode=True,
            data_retention_days=60,
            session_timeout_minutes=60
        )

        assert config.encryption_enabled == False
        assert config.consent_required == False
        assert config.privacy_mode == True
        assert config.data_retention_days == 60
        assert config.session_timeout_minutes == 60


class TestConsentRecord:
    """Tests for ConsentRecord class."""

    def test_consent_record_creation(self):
        """Test consent record creation."""
        timestamp = datetime.now()
        record = ConsentRecord(
            user_id="user123",
            consent_type="voice_processing",
            granted=True,
            timestamp=timestamp,
            version="1.0"
        )

        assert record.user_id == "user123"
        assert record.consent_type == "voice_processing"
        assert record.granted == True
        assert record.timestamp == timestamp
        assert record.version == "1.0"
        assert record.details is None

    def test_consent_record_with_details(self):
        """Test consent record creation with details."""
        timestamp = datetime.now()
        details = {"purpose": "therapy", "duration": "session"}
        record = ConsentRecord(
            user_id="user123",
            consent_type="voice_processing",
            granted=True,
            timestamp=timestamp,
            version="1.0",
            details=details
        )

        assert record.details == details


class TestAuditLogEntry:
    """Tests for AuditLogEntry class."""

    def test_audit_log_entry_creation(self):
        """Test audit log entry creation."""
        timestamp = datetime.now()
        entry = AuditLogEntry(
            timestamp=timestamp,
            event_type="VOICE_DATA_ACCESS",
            user_id="user123",
            session_id="session456",
            details={"action": "encrypt"}
        )

        assert entry.timestamp == timestamp
        assert entry.event_type == "VOICE_DATA_ACCESS"
        assert entry.user_id == "user123"
        assert entry.session_id == "session456"
        assert entry.details == {"action": "encrypt"}
        assert entry.severity == "INFO"  # Default value

    def test_audit_log_entry_with_severity(self):
        """Test audit log entry creation with custom severity."""
        timestamp = datetime.now()
        entry = AuditLogEntry(
            timestamp=timestamp,
            event_type="SECURITY_BREACH",
            user_id="user123",
            session_id="session456",
            details={"action": "unauthorized_access"},
            severity="CRITICAL"
        )

        assert entry.severity == "CRITICAL"


class TestVoiceSecurity:
    """Tests for VoiceSecurity class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock security configuration."""
        return SecurityConfig(
            encryption_enabled=True,
            consent_required=True,
            privacy_mode=False,
            hipaa_compliance_enabled=True,
            data_retention_days=30,
            audit_logging_enabled=True,
            session_timeout_minutes=30
        )

    @pytest.fixture
    def voice_security(self, mock_config):
        """Create voice security instance for testing."""
        return VoiceSecurity(mock_config)

    def test_initialization(self, voice_security):
        """Test voice security initialization."""
        assert voice_security.original_config is not None
        assert voice_security.logger is not None
        assert voice_security.initialized == True

    def test_initialization_without_config(self):
        """Test voice security initialization without config."""
        security = VoiceSecurity()
        assert security.original_config is not None
        assert security.encryption_enabled == True  # Default value

    def test_get_current_time(self, voice_security):
        """Test getting current time."""
        current_time = voice_security._get_current_time()
        assert isinstance(current_time, datetime)
        # Should be very recent (within 1 second)
        assert abs((datetime.now() - current_time).total_seconds()) < 1.0

    def test_encrypt_decrypt_data(self, voice_security):
        """Test basic data encryption and decryption."""
        test_data = b"Hello, this is sensitive voice data!"
        user_id = "test_user"

        # Encrypt the data
        encrypted_data = voice_security.encrypt_data(test_data, user_id)
        assert encrypted_data != test_data
        assert isinstance(encrypted_data, bytes)

        # Decrypt the data
        decrypted_data = voice_security.decrypt_data(encrypted_data, user_id)
        assert decrypted_data == test_data

    def test_encrypt_decrypt_audio_data(self, voice_security):
        """Test audio data encryption and decryption."""
        audio_data = b"raw_audio_bytes_here"
        user_id = "test_user"

        # Encrypt audio data
        encrypted_audio = voice_security.encrypt_audio_data(audio_data, user_id)
        assert encrypted_audio != audio_data
        assert isinstance(encrypted_audio, bytes)

        # Decrypt audio data
        decrypted_audio = voice_security.decrypt_audio_data(encrypted_audio, user_id)
        assert decrypted_audio == audio_data

    def test_encrypt_empty_data(self, voice_security):
        """Test encryption of empty data."""
        empty_data = b""
        user_id = "test_user"

        encrypted = voice_security.encrypt_data(empty_data, user_id)
        decrypted = voice_security.decrypt_data(encrypted, user_id)
        assert decrypted == empty_data

    def test_anonymization(self, voice_security):
        """Test data anonymization."""
        personal_data = {
            "user_id": "john_doe_123",
            "name": "John Doe",
            "email": "john@example.com",
            "voice_data": b"audio_bytes"
        }

        # Anonymize data
        anonymized_data = voice_security.anonymize_data(personal_data)

        # Check that personal identifiers are removed/anonymized
        assert anonymized_data != personal_data
        assert "voice_data" in anonymized_data  # Non-personal data remains

    def test_consent_management(self, voice_security):
        """Test consent management functionality."""
        user_id = "test_user"
        consent_type = "voice_processing"

        # Record consent
        success = voice_security.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            details={"purpose": "therapy"}
        )
        assert success == True

        # Check consent status
        has_consent = voice_security.has_consent(user_id, consent_type)
        assert has_consent == True

    def test_consent_revocation(self, voice_security):
        """Test consent revocation."""
        user_id = "test_user"
        consent_type = "voice_processing"

        # Grant consent first
        voice_security.record_consent(user_id, consent_type, True)
        assert voice_security.has_consent(user_id, consent_type) == True

        # Revoke consent
        voice_security.withdraw_consent(user_id, consent_type)
        assert voice_security.has_consent(user_id, consent_type) == False

    def test_consent_without_record(self, voice_security):
        """Test checking consent for non-existent record."""
        has_consent = voice_security.has_consent("nonexistent_user", "voice_processing")
        assert has_consent == False

    def test_access_control(self, voice_security):
        """Test access control functionality."""
        user_id = "test_user"
        resource = "voice_data"
        permission = "read"

        # Grant access
        voice_security.access_manager.grant_access(user_id, resource, permission)

        # Check access
        has_access = voice_security.access_manager.has_access(user_id, resource, permission)
        assert has_access == True

        # Check non-existent permission
        has_no_access = voice_security.access_manager.has_access(user_id, resource, "delete")
        assert has_no_access == False

    def test_audit_logging(self, voice_security):
        """Test audit logging functionality."""
        event_type = "VOICE_DATA_ACCESS"
        session_id = "test_session"
        user_id = "test_user"
        action = "data_encryption"
        details = {"file": "voice.wav", "size": 1024}

        # Log an event
        voice_security.audit_logger.log_event(
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            action=action,
            details=details
        )

        # Check that logs were created
        session_logs = voice_security.audit_logger.get_session_logs(session_id)
        assert len(session_logs) > 0

        # Check user logs
        user_logs = voice_security.audit_logger.get_user_logs(user_id)
        assert len(user_logs) > 0

        # Verify log content
        log_entry = session_logs[0]
        assert log_entry["event_type"] == event_type
        assert log_entry["user_id"] == user_id
        assert log_entry["action"] == action

    def test_cleanup(self, voice_security):
        """Test cleanup functionality."""
        # Cleanup should work without errors
        voice_security.cleanup()
        assert True  # If we get here, cleanup didn't crash

    def test_apply_retention_policy(self, voice_security):
        """Test retention policy application."""
        # Apply retention policy
        result = voice_security.apply_retention_policy()
        assert isinstance(result, int)
        assert result >= 0  # Should return count of items processed

    def test_cleanup_expired_sessions(self, voice_security):
        """Test expired session cleanup."""
        # Cleanup expired sessions
        voice_security.cleanup_expired_sessions()
        assert True  # If we get here, cleanup didn't crash

    def test_security_scan(self, voice_security):
        """Test security scanning."""
        # Perform security scan
        scan_results = voice_security.perform_security_scan()
        assert isinstance(scan_results, dict)
        assert "vulnerabilities" in scan_results
        assert "compliance_status" in scan_results

    def test_security_incident_reporting(self, voice_security):
        """Test security incident reporting."""
        incident_type = "UNAUTHORIZED_ACCESS"
        details = {"user": "unknown", "resource": "voice_data"}

        # Report incident
        incident_id = voice_security.report_security_incident(incident_type, details)
        assert incident_id is not None

        # Get incident details
        incident_details = voice_security.get_incident_details(incident_id)
        assert incident_details is not None
        assert incident_details["incident_type"] == incident_type

    def test_compliance_reporting(self, voice_security):
        """Test compliance reporting."""
        # Generate compliance report
        report = voice_security.generate_compliance_report()
        assert isinstance(report, dict)
        assert "hipaa_compliance" in report
        assert "data_protection" in report
        assert "audit_trail" in report

    def test_backup_and_restore(self, voice_security):
        """Test backup and restore functionality."""
        test_data = {"user_records": 5, "consent_records": 10}

        # Create backup
        backup_id = voice_security.backup_secure_data(test_data)
        assert backup_id is not None

        # Restore from backup
        restored_data = voice_security.restore_secure_data(backup_id)
        assert restored_data == test_data

    def test_security_metrics(self, voice_security):
        """Test security metrics generation."""
        # Get security metrics
        metrics = voice_security.get_security_metrics()
        assert isinstance(metrics, dict)
        assert "encryption_status" in metrics
        assert "audit_events" in metrics
        assert "active_sessions" in metrics

    def test_property_accessors(self, voice_security):
        """Test property accessor methods."""
        # Test config property
        config = voice_security.config
        assert config is not None
        assert config.encryption_enabled == True

        # Test individual properties
        assert voice_security.encryption_enabled == True
        assert voice_security.consent_required == True
        assert voice_security.privacy_mode == False
        assert voice_security.audit_logging_enabled == True
        assert voice_security.data_retention_days == 30

    def test_enable_privacy_mode(self, voice_security):
        """Test privacy mode enabling."""
        # Enable privacy mode
        voice_security.enable_privacy_mode()
        assert voice_security.privacy_mode == True

    def test_penetration_testing_scope(self, voice_security):
        """Test penetration testing scope."""
        # Get penetration testing scope
        scope = voice_security.get_penetration_testing_scope()
        assert isinstance(scope, dict)
        assert "endpoints" in scope
        assert "test_methods" in scope

    def test_consent_status_checking(self, voice_security):
        """Test consent status checking."""
        user_id = "test_user"

        # Check without consent
        has_consent = voice_security._check_consent_status(user_id=user_id)
        assert isinstance(has_consent, bool)

        # Record consent and check again
        voice_security.record_consent(user_id, "voice_processing", True)
        has_consent = voice_security._check_consent_status(user_id=user_id)
        assert has_consent == True

    def test_security_verification(self, voice_security):
        """Test security requirement verification."""
        user_id = "test_user"

        # Verify security requirements
        is_secure = voice_security._verify_security_requirements(user_id=user_id)
        assert isinstance(is_secure, bool)

    @pytest.mark.asyncio
    async def test_audio_processing(self, voice_security):
        """Test async audio processing."""
        mock_audio_data = b"test_audio_data"

        # Process audio data
        result = await voice_security.process_audio(mock_audio_data)
        # Result depends on implementation, just check it doesn't crash
        assert result is not None

    def test_security_with_disabled_encryption(self):
        """Test security operations with encryption disabled."""
        config = SecurityConfig(encryption_enabled=False)
        security = VoiceSecurity(config)

        test_data = b"Test data"
        user_id = "test_user"

        # Should still work even with encryption disabled
        encrypted = security.encrypt_data(test_data, user_id)
        decrypted = security.decrypt_data(encrypted, user_id)
        assert decrypted == test_data

    def test_error_handling_in_encryption(self, voice_security):
        """Test error handling in encryption operations."""
        user_id = "test_user"

        # Test with invalid data - should handle gracefully
        with pytest.raises(Exception):
            voice_security.decrypt_data(b"invalid_encrypted_data", user_id)