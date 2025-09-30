"""
Security and compliance tests for voice features.

Tests SPEECH_PRD.md requirements:
- HIPAA compliance testing
- Data encryption testing
- Consent management testing
- Audit logging testing
- Privacy mode testing
- Security penetration testing
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import json
import tempfile
import os
from datetime import datetime, timedelta

from voice.security import VoiceSecurity, SecurityConfig, AuditLogger, ConsentManager
from voice.config import VoiceConfig


class TestSecurityCompliance:
    """Test security and compliance features."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = VoiceConfig()
        config.encryption_enabled = True
        config.consent_required = True
        config.privacy_mode = True
        config.audit_logging_enabled = True
        config.data_retention_days = 30
        return config

    @pytest.fixture
    def security(self, config):
        """Create VoiceSecurity instance for testing."""
        return VoiceSecurity(config)

    @pytest.fixture
    def encrypted_audio_data(self):
        """Generate encrypted audio data for testing."""
        import cryptography.fernet
        key = cryptography.fernet.Fernet.generate_key()
        cipher = cryptography.fernet.Fernet(key)
        audio_data = b'test_audio_data'
        return cipher.encrypt(audio_data)

    def test_security_initialization(self, security, config):
        """Test security initialization."""
        assert security.config == config
        assert security.encryption_enabled == config.encryption_enabled
        assert security.consent_required == config.consent_required
        assert security.privacy_mode == config.privacy_mode
        assert security.audit_logging_enabled == config.audit_logging_enabled

    def test_audit_logging_functionality(self, security):
        """Test audit logging functionality."""
        # Test log creation
        event_type = "VOICE_INPUT"
        session_id = "test_session_123"
        user_id = "test_user_456"
        details = {"duration": 5.2, "provider": "openai"}

        log_entry = security.audit_logger.log_event(
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            details=details
        )

        assert log_entry['event_type'] == event_type
        assert log_entry['session_id'] == session_id
        assert log_entry['user_id'] == user_id
        assert log_entry['details'] == details
        assert 'timestamp' in log_entry
        assert 'event_id' in log_entry

    def test_audit_log_retrieval(self, security):
        """Test audit log retrieval."""
        # Create test logs
        session_id = "test_session_123"
        for i in range(5):
            security.audit_logger.log_event(
                event_type="VOICE_INPUT",
                session_id=session_id,
                user_id="test_user",
                details={"iteration": i}
            )

        # Retrieve logs
        logs = security.audit_logger.get_session_logs(session_id)
        assert len(logs) == 5

        # Test date filtering
        from_date = datetime.now() - timedelta(hours=1)
        filtered_logs = security.audit_logger.get_logs_in_date_range(from_date, datetime.now())
        assert len(filtered_logs) == 5

    def test_consent_management(self, security):
        """Test consent management."""
        user_id = "test_user_789"
        consent_type = "VOICE_DATA_PROCESSING"

        # Test consent recording
        consent_record = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version="1.0"
        )

        assert consent_record['user_id'] == user_id
        assert consent_record['consent_type'] == consent_type
        assert consent_record['granted'] == True
        assert consent_record['version'] == "1.0"

        # Test consent verification
        has_consent = security.consent_manager.has_consent(user_id, consent_type)
        assert has_consent == True

        # Test consent withdrawal
        security.consent_manager.withdraw_consent(user_id, consent_type)
        has_consent = security.consent_manager.has_consent(user_id, consent_type)
        assert has_consent == False

    def test_data_encryption(self, security):
        """Test data encryption."""
        test_data = b"sensitive_voice_data"
        user_id = "test_user_123"

        # Encrypt data
        encrypted_data = security.encrypt_data(test_data, user_id)
        assert encrypted_data != test_data
        assert isinstance(encrypted_data, bytes)

        # Decrypt data
        decrypted_data = security.decrypt_data(encrypted_data, user_id)
        assert decrypted_data == test_data

        # Test with different user (should fail)
        with pytest.raises(Exception):
            security.decrypt_data(encrypted_data, "different_user")

    def test_audio_data_encryption(self, security, encrypted_audio_data):
        """Test audio data encryption."""
        user_id = "test_user_456"

        # Encrypt audio
        encrypted_audio = security.encrypt_audio_data(encrypted_audio_data, user_id)
        assert encrypted_audio != encrypted_audio_data

        # Decrypt audio
        decrypted_audio = security.decrypt_audio_data(encrypted_audio, user_id)
        assert decrypted_audio == encrypted_audio_data

    def test_privacy_mode_functionality(self, security):
        """Test privacy mode functionality."""
        # Enable privacy mode
        security.enable_privacy_mode()

        # Test data anonymization
        test_data = {
            'user_id': 'specific_user_123',
            'session_id': 'specific_session_456',
            'audio_data': b'sensitive_audio',
            'transcript': 'I feel anxious about work'
        }

        anonymized_data = security.anonymize_data(test_data)
        assert anonymized_data['user_id'] != 'specific_user_123'
        assert anonymized_data['session_id'] != 'specific_session_456'
        assert 'audio_data' not in anonymized_data  # Should be removed in privacy mode
        assert len(anonymized_data['transcript']) > 0  # Transcript should be preserved but potentially masked

    def test_data_retention_policy(self, security):
        """Test data retention policy."""
        # Create old audit logs
        old_date = datetime.now() - timedelta(days=45)  # Older than 30 days
        with patch('voice.security.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_date
            security.audit_logger.log_event(
                event_type="VOICE_INPUT",
                session_id="old_session",
                user_id="old_user",
                details={"test": "old_data"}
            )

        # Create recent logs
        recent_date = datetime.now() - timedelta(days=15)
        with patch('voice.security.datetime') as mock_datetime:
            mock_datetime.now.return_value = recent_date
            security.audit_logger.log_event(
                event_type="VOICE_INPUT",
                session_id="recent_session",
                user_id="recent_user",
                details={"test": "recent_data"}
            )

        # Apply retention policy
        removed_count = security.apply_retention_policy()
        assert removed_count >= 1  # Should remove old logs

    def test_security_audit_trail(self, security):
        """Test security audit trail."""
        # Perform security-sensitive operations
        user_id = "test_user_123"

        # Record consent
        security.consent_manager.record_consent(
            user_id=user_id,
            consent_type="VOICE_DATA_PROCESSING",
            granted=True
        )

        # Encrypt data
        test_data = b"sensitive_data"
        security.encrypt_data(test_data, user_id)

        # Generate audit trail
        audit_trail = security.get_security_audit_trail(user_id)

        assert len(audit_trail) >= 2
        assert all('event_type' in entry for entry in audit_trail)
        assert all('timestamp' in entry for entry in audit_trail)

    def test_access_control(self, security):
        """Test access control mechanisms."""
        user_id = "test_user_123"
        resource_id = "voice_session_456"

        # Grant access
        security.access_manager.grant_access(user_id, resource_id, "read")

        # Check access
        has_access = security.access_manager.has_access(user_id, resource_id, "read")
        assert has_access == True

        # Check denied access
        has_write_access = security.access_manager.has_access(user_id, resource_id, "write")
        assert has_write_access == False

        # Revoke access
        security.access_manager.revoke_access(user_id, resource_id, "read")
        has_access = security.access_manager.has_access(user_id, resource_id, "read")
        assert has_access == False

    def test_vulnerability_scanning(self, security):
        """Test vulnerability scanning capabilities."""
        # Mock security scan
        scan_results = security.perform_security_scan()

        assert 'vulnerabilities' in scan_results
        assert 'compliance_status' in scan_results
        assert 'security_score' in scan_results
        assert 'recommendations' in scan_results

        # Verify all critical areas are checked
        critical_areas = [
            'encryption',
            'authentication',
            'authorization',
            'audit_logging',
            'data_retention',
            'privacy_protection'
        ]

        for area in critical_areas:
            assert area in scan_results['compliance_status']

    def test_incident_response(self, security):
        """Test incident response procedures."""
        # Simulate security incident
        incident_type = "UNAUTHORIZED_ACCESS"
        incident_details = {
            'user_id': 'malicious_user',
            'resource': 'voice_data',
            'timestamp': datetime.now().isoformat(),
            'severity': 'HIGH'
        }

        # Report incident
        incident_id = security.report_security_incident(incident_type, incident_details)
        assert incident_id is not None

        # Get incident details
        incident = security.get_incident_details(incident_id)
        assert incident['incident_type'] == incident_type
        assert incident['status'] in ['OPEN', 'INVESTIGATING', 'RESOLVED']

    def test_compliance_reporting(self, security):
        """Test compliance reporting."""
        # Generate compliance report
        report = security.generate_compliance_report()

        assert 'hipaa_compliance' in report
        assert 'data_protection' in report
        assert 'audit_trail' in report
        assert 'consent_management' in report
        assert 'security_measures' in report

        # Verify HIPAA compliance sections
        hipaa_sections = [
            'privacy_rule',
            'security_rule',
            'breach_notification',
            'data_encryption',
            'access_controls',
            'audit_controls'
        ]

        for section in hipaa_sections:
            assert section in report['hipaa_compliance']

    def test_backup_and_recovery(self, security):
        """Test backup and recovery procedures."""
        # Create test data
        test_data = {
            'user_id': 'test_user_123',
            'voice_data': b'encrypted_audio',
            'metadata': {'session_id': 'test_session_456'}
        }

        # Backup data
        backup_id = security.backup_secure_data(test_data)
        assert backup_id is not None

        # Restore data
        restored_data = security.restore_secure_data(backup_id)
        assert restored_data == test_data

    def test_penetration_testing_preparation(self, security):
        """Test preparation for penetration testing."""
        # Get penetration testing scope
        scope = security.get_penetration_testing_scope()

        assert 'target_systems' in scope
        assert 'test_scenarios' in scope
        assert 'excluded_areas' in scope
        assert 'authorization_requirements' in scope

        # Verify common penetration test scenarios
        test_scenarios = scope['test_scenarios']
        assert 'sql_injection' in test_scenarios
        assert 'xss_attacks' in test_scenarios
        assert 'authentication_bypass' in test_scenarios
        assert 'data_exfiltration' in test_scenarios

    def test_security_metrics(self, security):
        """Test security metrics collection."""
        # Perform some security operations
        for i in range(5):
            security.audit_logger.log_event(
                event_type="VOICE_INPUT",
                session_id=f"session_{i}",
                user_id="test_user",
                details={"iteration": i}
            )

        # Get security metrics
        metrics = security.get_security_metrics()

        assert 'total_events' in metrics
        assert 'unique_users' in metrics
        assert 'security_incidents' in metrics
        assert 'compliance_score' in metrics
        assert 'data_encryption_rate' in metrics

    def test_cleanup(self, security):
        """Test security cleanup."""
        # Create test data
        for i in range(10):
            security.audit_logger.log_event(
                event_type="TEST_EVENT",
                session_id=f"test_session_{i}",
                user_id="test_user",
                details={"test": f"data_{i}"}
            )

        # Cleanup
        security.cleanup()

        # Verify cleanup completed
        assert True  # If no exception, cleanup was successful