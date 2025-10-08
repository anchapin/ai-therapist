"""
Security testing fixtures for consistent security feature testing.

Provides fixtures for encryption, PII protection, authentication, and 
audit logging to ensure reliable and isolated security testing.
"""

import pytest
from unittest.mock import MagicMock, patch
from cryptography.fernet import Fernet
import json
import tempfile
import os
from datetime import datetime, timedelta


@pytest.fixture
def mock_encryption_service():
    """Mock encryption service for testing."""
    with patch('security.pii_protection.EncryptionService') as mock_encryption_class:
        mock_encryption = MagicMock()
        mock_encryption_class.return_value = mock_encryption
        
        # Mock encryption/decryption
        mock_encryption.encrypt_text.return_value = "encrypted_text_123"
        mock_encryption.decrypt_text.return_value = "original_text"
        mock_encryption.encrypt_data.return_value = b"encrypted_data_123"
        mock_encryption.decrypt_data.return_value = b"original_data"
        
        # Mock key management
        mock_encryption.generate_key.return_value = Fernet.generate_key().decode()
        mock_encryption.rotate_key.return_value = "new_encryption_key_123"
        
        # Mock batch operations
        mock_encryption.encrypt_dict_values.return_value = {
            'field1': 'encrypted_value1',
            'field2': 'encrypted_value2'
        }
        mock_encryption.decrypt_dict_values.return_value = {
            'field1': 'original_value1',
            'field2': 'original_value2'
        }
        
        # Mock health check
        mock_encryption.health_check.return_value = {
            'status': 'healthy',
            'key_rotation_enabled': True,
            'last_rotation': datetime.now().isoformat(),
            'algorithm': 'Fernet'
        }
        
        yield mock_encryption


@pytest.fixture
def real_encryption_service():
    """Real encryption service for testing (with temp key)."""
    from security.pii_protection import EncryptionService
    
    # Create temporary key for testing
    temp_key = Fernet.generate_key().decode()
    
    with patch.dict('os.environ', {'ENCRYPTION_KEY': temp_key}):
        service = EncryptionService()
        yield service


@pytest.fixture
def mock_pii_detector():
    """Mock PII detector for testing."""
    with patch('security.pii_protection.PIIDetector') as mock_detector_class:
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        
        # Mock PII detection
        mock_detector.detect_pii.return_value = {
            'has_pii': True,
            'pii_types': ['email', 'phone'],
            'locations': [(0, 15), (20, 30)],
            'confidence': 0.95
        }
        
        # Mock PII masking
        mock_detector.mask_pii.return_value = "My email is ****@****.com and phone is ***-***-1234"
        
        # Mock redaction
        mock_detector.redact_pii.return_value = "My email is [REDACTED] and phone is [REDACTED]"
        
        # Mock PII validation
        mock_detector.validate_pii_handling.return_value = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Mock configuration
        mock_detector.add_pii_pattern.return_value = True
        mock_detector.remove_pii_pattern.return_value = True
        
        yield mock_detector


@pytest.fixture
def sample_pii_data():
    """Sample data containing PII for testing."""
    return {
        'patient_name': 'John Doe',
        'email': 'john.doe@email.com',
        'phone': '555-123-4567',
        'ssn': '123-45-6789',
        'address': '123 Main St, Anytown, USA 12345',
        'medical_record': 'Patient has anxiety symptoms',
        'notes': 'Contact patient at john.doe@email.com for follow-up',
        'metadata': {
            'created_at': '2024-01-01T00:00:00',
            'therapist_id': 'therapist_123'
        }
    }


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing."""
    with patch('security.pii_protection.AuditLogger') as mock_audit_class:
        mock_audit = MagicMock()
        mock_audit_class.return_value = mock_audit
        
        # Mock logging methods
        mock_audit.log_access.return_value = True
        mock_audit.log_modification.return_value = True
        mock_audit.log_deletion.return_value = True
        mock_audit.log_export.return_value = True
        mock_audit.log_security_event.return_value = True
        mock_audit.log_authentication.return_value = True
        
        # Mock retrieval methods
        mock_audit.get_access_logs.return_value = [
            {
                'timestamp': datetime.now().isoformat(),
                'user_id': 'user_123',
                'action': 'access',
                'resource': 'patient_data',
                'ip_address': '127.0.0.1'
            }
        ]
        
        mock_audit.get_security_events.return_value = [
            {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'login_attempt',
                'user_id': 'user_123',
                'success': True,
                'ip_address': '127.0.0.1'
            }
        ]
        
        # Mock analytics
        mock_audit.get_access_statistics.return_value = {
            'total_accesses': 100,
            'unique_users': 10,
            'most_accessed_resource': 'patient_data',
            'access_by_hour': {9: 15, 10: 20, 11: 25}
        }
        
        # Mock compliance checking
        mock_audit.check_compliance.return_value = {
            'compliant': True,
            'violations': [],
            'last_check': datetime.now().isoformat()
        }
        
        # Mock log retention
        mock_audit.cleanup_old_logs.return_value = 50  # Cleaned up 50 old logs
        
        yield mock_audit


@pytest.fixture
def mock_access_control():
    """Mock access control service for testing."""
    with patch('security.pii_protection.AccessControl') as mock_ac_class:
        mock_ac = MagicMock()
        mock_ac_class.return_value = mock_ac
        
        # Mock permission checking
        mock_ac.has_permission.return_value = True
        mock_ac.has_role.return_value = True
        mock_ac.can_access_resource.return_value = True
        
        # Mock role management
        mock_ac.assign_role.return_value = True
        mock_ac.remove_role.return_value = True
        mock_ac.get_user_roles.return_value = ['therapist', 'admin']
        
        # Mock resource management
        mock_ac.create_resource.return_value = True
        mock_ac.update_resource.return_value = True
        mock_ac.delete_resource.return_value = True
        mock_ac.get_resource_permissions.return_value = ['read', 'write', 'delete']
        
        # Mock user management
        mock_ac.create_user.return_value = {'user_id': 'user_123', 'status': 'active'}
        mock_ac.update_user.return_value = True
        mock_ac.deactivate_user.return_value = True
        mock_ac.get_user_permissions.return_value = ['read:patient_data', 'write:notes']
        
        # Mock session management
        mock_ac.create_session.return_value = {'session_id': 'session_123', 'expires_at': '2024-01-01T01:00:00'}
        mock_ac.validate_session.return_value = True
        mock_ac.invalidate_session.return_value = True
        
        # Mock audit trail
        mock_ac.get_access_history.return_value = [
            {
                'timestamp': datetime.now().isoformat(),
                'user_id': 'user_123',
                'resource': 'patient_456',
                'action': 'read',
                'granted': True
            }
        ]
        
        yield mock_ac


@pytest.fixture
def mock_consent_manager():
    """Mock consent manager for testing."""
    with patch('security.pii_protection.ConsentManager') as mock_consent_class:
        mock_consent = MagicMock()
        mock_consent_class.return_value = mock_consent
        
        # Mock consent management
        mock_consent.create_consent.return_value = {
            'consent_id': 'consent_123',
            'patient_id': 'patient_456',
            'status': 'active',
            'created_at': datetime.now().isoformat()
        }
        
        # Mock consent validation
        mock_consent.has_consent.return_value = True
        mock_consent.get_consent_status.return_value = {
            'status': 'active',
            'expires_at': '2024-12-31T23:59:59',
            'purposes': ['treatment', 'research']
        }
        
        # Mock consent updates
        mock_consent.update_consent.return_value = True
        mock_consent.revoke_consent.return_value = True
        mock_consent.renew_consent.return_value = True
        
        # Mock consent retrieval
        mock_consent.get_patient_consents.return_value = [
            {
                'consent_id': 'consent_123',
                'purpose': 'treatment',
                'status': 'active',
                'granted_at': '2024-01-01T00:00:00'
            }
        ]
        
        # Mock compliance checking
        mock_consent.check_consent_compliance.return_value = {
            'compliant': True,
            'missing_consents': [],
            'expired_consents': []
        }
        
        yield mock_consent


@pytest.fixture
def security_test_environment(mock_encryption_service, mock_pii_detector, 
                             mock_audit_logger, mock_access_control, mock_consent_manager):
    """Complete security test environment with all services mocked."""
    return {
        'encryption': mock_encryption_service,
        'pii_detector': mock_pii_detector,
        'audit_logger': mock_audit_logger,
        'access_control': mock_access_control,
        'consent_manager': mock_consent_manager
    }


@pytest.fixture
def hipaa_compliance_config():
    """HIPAA compliance configuration for testing."""
    return {
        'audit_retention_days': 2555,  # 7 years
        'data_encryption_required': True,
        'access_log_all_views': True,
        'session_timeout_minutes': 15,
        'password_complexity_required': True,
        'multi_factor_auth_required': False,
        'business_associate_agreements': True,
        'incident_reporting_required': True,
        'patient_rights_access': True,
        'data_backup_required': True,
        'disaster_recovery_required': True,
        'employee_training_required': True,
        'sanction_policy_required': True
    }


@pytest.fixture
def gdpr_compliance_config():
    """GDPR compliance configuration for testing."""
    return {
        'data_minimization': True,
        'purpose_limitation': True,
        'retention_limitation': True,
        'accuracy_rights': True,
        'storage_limitation': True,
        'security_measures': True,
        'accountability_required': True,
        'data_subject_rights': True,
        'consent_required': True,
        'breach_notification_hours': 72,
        'data_protection_officer_required': False,
        'data_protection_impact_assessment': False,
        'international_transfers': False
    }


@pytest.fixture
def security_violation_examples():
    """Examples of security violations for testing."""
    return [
        {
            'type': 'unauthorized_access',
            'severity': 'high',
            'description': 'User accessed patient data without permission',
            'user_id': 'user_123',
            'resource': 'patient_456',
            'timestamp': datetime.now().isoformat()
        },
        {
            'type': 'data_exfiltration',
            'severity': 'critical',
            'description': 'Large amount of data downloaded suspiciously',
            'user_id': 'user_789',
            'data_volume': 5000,  # records
            'timestamp': datetime.now().isoformat()
        },
        {
            'type': 'consent_violation',
            'severity': 'medium',
            'description': 'Patient data used without proper consent',
            'patient_id': 'patient_456',
            'purpose': 'research',
            'timestamp': datetime.now().isoformat()
        },
        {
            'type': 'encryption_failure',
            'severity': 'high',
            'description': 'Data stored without encryption',
            'resource': 'backup_123',
            'timestamp': datetime.now().isoformat()
        }
    ]


@pytest.fixture
def security_compliance_report():
    """Sample security compliance report for testing."""
    return {
        'report_date': datetime.now().isoformat(),
        'hipaa_compliance': {
            'overall_score': 95,
            'violations': [],
            'recommendations': ['Enable MFA for admin accounts'],
            'last_audit': '2024-01-01T00:00:00'
        },
        'gdpr_compliance': {
            'overall_score': 92,
            'violations': [],
            'recommendations': ['Update privacy policy'],
            'last_audit': '2024-01-01T00:00:00'
        },
        'security_metrics': {
            'total_violations': 0,
            'critical_violations': 0,
            'high_violations': 0,
            'medium_violations': 0,
            'low_violations': 0,
            'days_since_last_violation': 45
        },
        'access_patterns': {
            'total_accesses': 1250,
            'unauthorized_attempts': 3,
            'denied_accesses': 12,
            'suspicious_activities': 1
        }
    }


@pytest.fixture
def temporary_security_logs():
    """Create temporary directory for security log testing."""
    temp_dir = tempfile.mkdtemp(prefix="security_logs_test_")
    
    # Create sample log files
    log_files = {
        'access.log': os.path.join(temp_dir, 'access.log'),
        'security.log': os.path.join(temp_dir, 'security.log'),
        'audit.log': os.path.join(temp_dir, 'audit.log'),
        'errors.log': os.path.join(temp_dir, 'errors.log')
    }
    
    # Write sample log entries
    sample_logs = {
        'access.log': '2024-01-01 10:00:00 INFO user_123 accessed patient_456\n',
        'security.log': '2024-01-01 10:01:00 WARNING Failed login attempt for user_789\n',
        'audit.log': '2024-01-01 10:02:00 INFO Data modified by user_123\n',
        'errors.log': '2024-01-01 10:03:00 ERROR Encryption service unavailable\n'
    }
    
    for log_file, content in sample_logs.items():
        with open(log_files[log_file], 'w') as f:
            f.write(content)
    
    yield log_files
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)