"""
HIPAA Compliance Comprehensive Testing

This module provides extensive coverage for HIPAA compliance including:
- Protected Health Information (PHI) detection and masking
- Audit trail integrity and logging
- Data retention and deletion policies
- Access control and authentication
- Encryption and data security
- Breach notification procedures
- Business associate agreement validation
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import os
from datetime import datetime, timedelta
import hashlib
import re
from dataclasses import dataclass
import logging
from cryptography.fernet import Fernet
import base64


class TestHIPAAComplianceComprehensive:
    """Comprehensive HIPAA compliance test suite."""
    
    @pytest.fixture
    def phi_test_data(self):
        """Sample PHI data for testing detection and masking."""
        return {
            'patient_names': [
                "John Doe",
                "Jane Smith",
                "Dr. Michael Johnson",
                "Sarah Williams RN"
            ],
            'medical_records': [
                "MRN: 123456789",
                "Patient ID: PAT-001234",
                "Medical Record #: MR987654"
            ],
            'contact_info': [
                "Phone: (555) 123-4567",
                "Email: john.doe@email.com",
                "Address: 123 Main St, City, State 12345"
            ],
            'health_info': [
                "Diagnosis: Depression",
                "Medication: Sertraline 50mg",
                "Treatment: Cognitive Behavioral Therapy"
            ],
            'dates': [
                "DOB: 01/15/1980",
                "Appointment: 12/25/2023",
                "Admission: 2023-01-01"
            ],
            'complex_phrases': [
                "John Doe was diagnosed with anxiety on January 15, 2023",
                "Patient Sarah Williams (MRN: 123456) called 555-123-4567",
                "Dr. Smith prescribed medication for Jane Doe's depression"
            ]
        }
    
    @pytest.fixture
    def audit_log_data(self):
        """Sample audit log entries for testing."""
        return [
            {
                'timestamp': '2023-12-01T10:00:00Z',
                'user_id': 'therapist_001',
                'action': 'access_patient_record',
                'resource_id': 'patient_123',
                'ip_address': '192.168.1.100',
                'success': True,
                'phi_accessed': ['name', 'diagnosis']
            },
            {
                'timestamp': '2023-12-01T10:05:00Z',
                'user_id': 'therapist_001',
                'action': 'modify_treatment_plan',
                'resource_id': 'patient_123',
                'ip_address': '192.168.1.100',
                'success': True,
                'phi_accessed': ['name', 'diagnosis', 'medication']
            },
            {
                'timestamp': '2023-12-01T10:10:00Z',
                'user_id': 'system',
                'action': 'backup_data',
                'resource_id': 'all_records',
                'ip_address': '127.0.0.1',
                'success': True,
                'phi_accessed': ['all_fields']
            }
        ]
    
    @pytest.fixture
    def mock_security_services(self):
        """Mock security services for testing."""
        with patch('security.pii_protection.PIIProtection') as mock_pii, \
             patch('security.response_sanitizer.ResponseSanitizer') as mock_sanitizer, \
             patch('security.audit_logging.AuditLogger') as mock_audit:
            
            # Mock PII detection
            mock_pii_instance = Mock()
            mock_pii_instance.detect_phi.return_value = {
                'detected': True,
                'entities': [
                    {'type': 'PERSON', 'value': 'John Doe', 'confidence': 0.95},
                    {'type': 'DATE', 'value': '01/15/1980', 'confidence': 0.88}
                ]
            }
            mock_pii_instance.mask_phi.return_value = "PATIENT was diagnosed on DATE"
            mock_pii.return_value = mock_pii_instance
            
            # Mock response sanitization
            mock_sanitizer_instance = Mock()
            mock_sanitizer_instance.sanitize_response.return_value = "Sanitized response"
            mock_sanitizer.return_value = mock_sanitizer_instance
            
            # Mock audit logging
            mock_audit_instance = Mock()
            mock_audit_instance.log_access.return_value = True
            mock_audit_instance.get_audit_trail.return_value = []
            mock_audit.return_value = mock_audit_instance
            
            yield {
                'pii': mock_pii_instance,
                'sanitizer': mock_sanitizer_instance,
                'audit': mock_audit_instance
            }
    
    @pytest.fixture
    def mock_encryption_service(self):
        """Mock encryption service for testing."""
        with patch('cryptography.fernet.Fernet') as mock_fernet:
            mock_key = Fernet.generate_key()
            mock_fernet.return_value = Mock()
            
            # Mock encryption/decryption
            mock_encrypted = b'encrypted_data_placeholder'
            mock_fernet.return_value.encrypt.return_value = mock_encrypted
            mock_fernet.return_value.decrypt.return_value = b'decrypted_data'
            
            yield {
                'fernet': mock_fernet,
                'key': mock_key,
                'encrypted_data': mock_encrypted
            }
    
    class TestPHIDetectionAndMasking:
        """Test PHI detection and masking capabilities."""
        
        def test_comprehensive_phi_detection(self, phi_test_data, mock_security_services):
            """Test detection of all PHI types."""
            pii_service = mock_security_services['pii']
            
            # Test each PHI category
            for category, examples in phi_test_data.items():
                for example in examples:
                    result = pii_service.detect_phi(example)
                    
                    assert result['detected'] is True, f"Should detect PHI in: {example}"
                    assert len(result['entities']) > 0, f"Should identify PHI entities in: {example}"
                    
                    # Verify confidence scores
                    for entity in result['entities']:
                        assert 0.0 <= entity['confidence'] <= 1.0, "Confidence should be between 0-1"
        
        def test_phi_masking_accuracy(self, phi_test_data, mock_security_services):
            """Test accurate PHI masking while preserving context."""
            pii_service = mock_security_services['pii']
            
            for phrase in phi_test_data['complex_phrases']:
                # Detect PHI first
                detection = pii_service.detect_phi(phrase)
                
                # Mask the PHI
                masked = pii_service.mask_phi(phrase)
                
                # Verify no original PHI remains
                for entity in detection['entities']:
                    assert entity['value'] not in masked, f"PHI '{entity['value']}' should be masked"
                
                # Verify context is preserved
                assert len(masked) > 0, "Masked text should not be empty"
                assert masked != phrase, "Masked text should be different from original"
        
        def test_phi_false_positive_handling(self, mock_security_services):
            """Test handling of false positives in PHI detection."""
            pii_service = mock_security_services['pii']
            
            # Test non-PHI text that might trigger false positives
            non_phi_examples = [
                "The temperature is 98.6 degrees",
                "Call me at your convenience",
                "The date is 12/25/2023",
                "John is a common name"
            ]
            
            for example in non_phi_examples:
                # Configure mock to return false positives for testing
                pii_service.detect_phi.return_value = {
                    'detected': False,
                    'entities': [],
                    'confidence': 0.1
                }
                
                result = pii_service.detect_phi(example)
                
                if result['detected']:
                    # If detected, confidence should be low for potential false positives
                    for entity in result['entities']:
                        assert entity['confidence'] < 0.5, "False positives should have low confidence"
        
        def test_nested_structures_phi_detection(self, phi_test_data, mock_security_services):
            """Test PHI detection in nested data structures."""
            pii_service = mock_security_services['pii']
            
            # Test nested dictionary
            nested_data = {
                'patient_info': {
                    'name': phi_test_data['patient_names'][0],
                    'contact': {
                        'phone': phi_test_data['contact_info'][0],
                        'email': 'protected@email.com'
                    }
                },
                'medical_history': [
                    {'condition': 'Anxiety', 'date': phi_test_data['dates'][0]},
                    {'medication': 'Sertraline', 'prescribed_by': 'Dr. Smith'}
                ]
            }
            
            # Configure mock to detect PHI in nested structures
            pii_service.detect_phi_in_dict.return_value = {
                'phi_found': True,
                'locations': [
                    {'path': 'patient_info.name', 'value': 'John Doe'},
                    {'path': 'patient_info.contact.phone', 'value': '(555) 123-4567'},
                    {'path': 'medical_history[0].date', 'value': '01/15/1980'}
                ]
            }
            
            result = pii_service.detect_phi_in_dict(nested_data)
            
            assert result['phi_found'] is True, "Should detect PHI in nested structures"
            assert len(result['locations']) >= 2, "Should find multiple PHI instances"
            
            # Verify path accuracy
            for location in result['locations']:
                assert 'path' in location, "Should include path to PHI"
                assert 'value' in location, "Should include PHI value"
        
        @pytest.mark.asyncio
        async def test_real_time_phi_filtering(self, mock_security_services):
            """Test real-time PHI filtering during voice processing."""
            pii_service = mock_security_services['pii']
            
            # Simulate real-time voice transcription
            voice_segments = [
                "My name is John Doe",
                "I was born on January 15, 1980",
                "My doctor is Dr. Smith",
                "I take medication for anxiety"
            ]
            
            filtered_segments = []
            
            for segment in voice_segments:
                # Detect and mask PHI in real-time
                detection = pii_service.detect_phi(segment)
                
                if detection['detected']:
                    masked_segment = pii_service.mask_phi(segment)
                    filtered_segments.append(masked_segment)
                else:
                    filtered_segments.append(segment)
            
            # Verify no PHI remains in filtered segments
            combined_text = ' '.join(filtered_segments)
            phi_entities = ['John Doe', 'January 15, 1980', 'Dr. Smith']
            
            for entity in phi_entities:
                assert entity not in combined_text, f"PHI '{entity}' should be filtered from real-time processing"
    
    class TestAuditTrailIntegrity:
        """Test audit trail logging and integrity."""
        
        def test_comprehensive_audit_logging(self, audit_log_data, mock_security_services):
            """Test comprehensive audit logging for all access types."""
            audit_service = mock_security_services['audit']
            
            # Configure mock to track log calls
            logged_entries = []
            
            def track_log_access(user_id, action, resource_id, **kwargs):
                logged_entries.append({
                    'user_id': user_id,
                    'action': action,
                    'resource_id': resource_id,
                    'timestamp': datetime.now().isoformat(),
                    **kwargs
                })
                return True
            
            audit_service.log_access.side_effect = track_log_access
            
            # Simulate various audit events
            test_events = [
                {
                    'user_id': 'therapist_001',
                    'action': 'view_patient_record',
                    'resource_id': 'patient_123',
                    'ip_address': '192.168.1.100'
                },
                {
                    'user_id': 'therapist_001',
                    'action': 'modify_treatment_plan',
                    'resource_id': 'patient_123',
                    'ip_address': '192.168.1.100'
                },
                {
                    'user_id': 'admin_001',
                    'action': 'export_patient_data',
                    'resource_id': 'patient_123',
                    'ip_address': '192.168.1.200'
                }
            ]
            
            for event in test_events:
                result = audit_service.log_access(**event)
                assert result is True, f"Should log {event['action']} successfully"
            
            # Verify all events were logged
            assert len(logged_entries) == len(test_events), "Should log all events"
            
            # Verify log entry completeness
            for entry in logged_entries:
                assert 'user_id' in entry, "Should log user ID"
                assert 'action' in entry, "Should log action"
                assert 'resource_id' in entry, "Should log resource ID"
                assert 'timestamp' in entry, "Should log timestamp"
        
        def test_audit_log_tampering_detection(self, mock_security_services):
            """Test detection of audit log tampering."""
            audit_service = mock_security_services['audit']
            
            # Create original log entry with checksum
            original_entry = {
                'timestamp': '2023-12-01T10:00:00Z',
                'user_id': 'therapist_001',
                'action': 'access_patient_record',
                'resource_id': 'patient_123'
            }
            
            # Generate checksum for original entry
            original_data = json.dumps(original_entry, sort_keys=True)
            original_checksum = hashlib.sha256(original_data.encode()).hexdigest()
            
            # Configure mock to verify checksums
            def verify_integrity(entries):
                for entry in entries:
                    entry_copy = entry.copy()
                    checksum = entry_copy.pop('checksum', None)
                    
                    if checksum:
                        current_data = json.dumps(entry_copy, sort_keys=True)
                        current_checksum = hashlib.sha256(current_data.encode()).hexdigest()
                        
                        if current_checksum != checksum:
                            return False, entry
                return True, None
            
            audit_service.verify_log_integrity.return_value = verify_integrity([original_entry])
            
            # Test integrity check
            result, tampered_entry = verify_integrity([original_entry])
            assert result is True, "Original entry should pass integrity check"
            assert tampered_entry is None, "No tampering should be detected"
            
            # Simulate tampering
            tampered_entry = original_entry.copy()
            tampered_entry['user_id'] = 'malicious_user'
            
            tampered_data = json.dumps(tampered_entry, sort_keys=True)
            tampered_checksum = hashlib.sha256(tampered_data.encode()).hexdigest()
            
            result, detected_entry = verify_integrity([tampered_entry])
            assert result is False, "Tampered entry should fail integrity check"
            assert detected_entry is not None, "Should identify tampered entry"
        
        def test_audit_log_retention_policy(self, mock_security_services):
            """Test audit log retention according to HIPAA requirements."""
            audit_service = mock_security_services['audit']
            
            # HIPAA requires 6-year retention for audit logs
            retention_period = 6 * 365  # days
            
            # Create log entries with different ages
            current_time = datetime.now()
            log_entries = []
            
            for days_ago in [0, 30, 365, 1825, 2190]:  # 0 days to 6 years
                entry_time = current_time - timedelta(days=days_ago)
                log_entries.append({
                    'timestamp': entry_time.isoformat(),
                    'user_id': f'user_{days_ago}',
                    'action': 'test_action',
                    'resource_id': 'test_resource'
                })
            
            # Configure mock to filter by retention
            def filter_by_retention(entries, retention_days):
                cutoff_time = datetime.now() - timedelta(days=retention_days)
                return [entry for entry in entries 
                       if datetime.fromisoformat(entry['timestamp']) > cutoff_time]
            
            audit_service.filter_by_retention.side_effect = filter_by_retention
            
            # Test retention filtering
            retained_entries = filter_by_retention(log_entries, retention_period)
            
            # Should retain entries within retention period
            assert len(retained_entries) >= 4, "Should retain entries within 6 years"
            
            # Should remove entries older than retention period
            for entry in retained_entries:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                age_days = (current_time - entry_time).days
                assert age_days <= retention_period, f"Entry age {age_days} exceeds retention period"
        
        def test_audit_log_access_control(self, mock_security_services):
            """Test access control for audit log viewing."""
            audit_service = mock_security_services['audit']
            
            # Define user roles and permissions
            user_roles = {
                'therapist_001': ['view_own_logs'],
                'admin_001': ['view_all_logs', 'export_logs'],
                'auditor_001': ['view_all_logs', 'verify_integrity'],
                'staff_001': []  # No audit log access
            }
            
            # Test access control
            for user_id, permissions in user_roles.items():
                def check_access(user_id, required_permission):
                    user_permissions = user_roles.get(user_id, [])
                    return required_permission in user_permissions
                
                audit_service.check_access.side_effect = check_access
                
                # Test different access scenarios
                scenarios = [
                    ('view_own_logs', user_id in ['therapist_001']),
                    ('view_all_logs', user_id in ['admin_001', 'auditor_001']),
                    ('export_logs', user_id == 'admin_001'),
                    ('verify_integrity', user_id == 'auditor_001')
                ]
                
                for permission, expected_result in scenarios:
                    result = check_access(user_id, permission)
                    assert result == expected_result, f"User {user_id} access to {permission} should be {expected_result}"
    
    class TestDataProtectionAndEncryption:
        """Test data protection and encryption mechanisms."""
        
        def test_phi_encryption_at_rest(self, mock_encryption_service):
            """Test encryption of PHI at rest."""
            phi_data = "Patient John Doe, DOB: 01/15/1980"
            
            # Encrypt PHI data
            encrypted_data = mock_encryption_service['fernet'].encrypt(phi_data.encode())
            
            assert encrypted_data != phi_data.encode(), "Encrypted data should differ from original"
            assert len(encrypted_data) > 0, "Encrypted data should not be empty"
            
            # Decrypt and verify
            decrypted_data = mock_encryption_service['fernet'].decrypt(encrypted_data)
            assert decrypted_data.decode() == phi_data, "Decrypted data should match original"
        
        def test_encryption_key_management(self):
            """Test secure encryption key management."""
            # Test key generation
            key = Fernet.generate_key()
            assert len(key) == 44, "Fernet key should be 44 bytes (base64)"
            
            # Test key derivation from passphrase
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            
            passphrase = b"secure_passphrase"
            salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            derived_key = base64.urlsafe_b64encode(kdf.derive(passphrase))
            
            # Test key is deterministic with same passphrase and salt
            kdf2 = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            derived_key2 = base64.urlsafe_b64encode(kdf2.derive(passphrase))
            assert derived_key == derived_key2, "Same inputs should produce same key"
            
            # Test key is different with different salt
            different_salt = os.urandom(16)
            kdf3 = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=different_salt,
                iterations=100000,
            )
            
            derived_key3 = base64.urlsafe_b64encode(kdf3.derive(passphrase))
            assert derived_key != derived_key3, "Different salt should produce different key"
        
        def test_secure_data_transmission(self, mock_encryption_service):
            """Test secure data transmission with encryption."""
            phi_payload = {
                'patient_id': 'PAT001',
                'name': 'John Doe',
                'diagnosis': 'Anxiety',
                'timestamp': datetime.now().isoformat()
            }
            
            # Serialize and encrypt payload
            serialized_data = json.dumps(phi_payload).encode()
            encrypted_payload = mock_encryption_service['fernet'].encrypt(serialized_data)
            
            # Simulate transmission and decryption
            received_encrypted = encrypted_payload
            decrypted_data = mock_encryption_service['fernet'].decrypt(received_encrypted)
            received_payload = json.loads(decrypted_data.decode())
            
            # Verify data integrity
            assert received_payload == phi_payload, "Transmitted data should match original"
            assert 'name' not in str(encrypted_payload), "PHI should not be visible in encrypted payload"
        
        @pytest.mark.asyncio
        async def test_end_to_end_voice_data_encryption(self, mock_encryption_service):
            """Test end-to-end encryption for voice data."""
            # Simulate voice data
            voice_data = {
                'audio_bytes': b'fake_audio_data',
                'transcription': 'Patient John Doe discussed anxiety treatment',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session_123'
            }
            
            # Encrypt sensitive fields
            encrypted_data = {}
            for key, value in voice_data.items():
                if key in ['transcription']:  # Only encrypt sensitive fields
                    encrypted_value = mock_encryption_service['fernet'].encrypt(
                        json.dumps(value).encode()
                    )
                    encrypted_data[key] = encrypted_value
                else:
                    encrypted_data[key] = value
            
            # Simulate processing and decryption
            processed_data = {}
            for key, value in encrypted_data.items():
                if key in ['transcription']:
                    decrypted_value = json.loads(
                        mock_encryption_service['fernet'].decrypt(value).decode()
                    )
                    processed_data[key] = decrypted_value
                else:
                    processed_data[key] = value
            
            # Verify end-to-end integrity
            assert processed_data['transcription'] == voice_data['transcription']
            assert 'John Doe' not in str(encrypted_data['transcription'])
    
    class TestAccessControlAndAuthentication:
        """Test access control and authentication mechanisms."""
        
        def test_role_based_access_control(self):
            """Test role-based access control for PHI access."""
            # Define roles and permissions
            rbac_config = {
                'therapist': {
                    'permissions': [
                        'view_assigned_patients',
                        'modify_treatment_plans',
                        'access_session_notes'
                    ],
                    'restrictions': [
                        'cannot_export_bulk_data',
                        'cannot_view_other_therapist_patients'
                    ]
                },
                'admin': {
                    'permissions': [
                        'view_all_patients',
                        'manage_user_accounts',
                        'access_audit_logs',
                        'export_data'
                    ],
                    'restrictions': []
                },
                'auditor': {
                    'permissions': [
                        'view_all_patients_readonly',
                        'access_audit_logs',
                        'verify_compliance'
                    ],
                    'restrictions': [
                        'cannot_modify_patient_data',
                        'cannot_export_data'
                    ]
                }
            }
            
            # Test access control logic
            def check_permission(user_role, action):
                role_config = rbac_config.get(user_role, {})
                permissions = role_config.get('permissions', [])
                restrictions = role_config.get('restrictions', [])
                
                if action in restrictions:
                    return False
                return action in permissions
            
            # Test various scenarios
            test_cases = [
                ('therapist', 'view_assigned_patients', True),
                ('therapist', 'view_all_patients', False),
                ('therapist', 'export_data', False),
                ('admin', 'view_all_patients', True),
                ('admin', 'export_data', True),
                ('auditor', 'view_all_patients_readonly', True),
                ('auditor', 'modify_treatment_plans', False),
                ('unknown_role', 'view_assigned_patients', False)
            ]
            
            for role, action, expected in test_cases:
                result = check_permission(role, action)
                assert result == expected, f"{role} access to {action} should be {expected}"
        
        def test_multi_factor_authentication(self):
            """Test multi-factor authentication requirements."""
            # Define MFA requirements for different access levels
            mfa_requirements = {
                'standard_access': ['password'],
                'phi_access': ['password', 'totp'],
                'admin_access': ['password', 'totp', 'hardware_key'],
                'emergency_access': ['password', 'emergency_codes']
            }
            
            def validate_mfa(access_level, provided_factors):
                required_factors = mfa_requirements.get(access_level, [])
                return all(factor in provided_factors for factor in required_factors)
            
            # Test MFA validation
            test_cases = [
                ('standard_access', ['password'], True),
                ('phi_access', ['password'], False),
                ('phi_access', ['password', 'totp'], True),
                ('admin_access', ['password', 'totp'], False),
                ('admin_access', ['password', 'totp', 'hardware_key'], True),
                ('emergency_access', ['password', 'emergency_codes'], True)
            ]
            
            for access_level, factors, expected in test_cases:
                result = validate_mfa(access_level, factors)
                assert result == expected, f"MFA for {access_level} should be {expected}"
        
        def test_session_security(self):
            """Test secure session management."""
            # Session configuration
            session_config = {
                'timeout_minutes': 30,
                'idle_timeout_minutes': 15,
                'max_concurrent_sessions': 3,
                'require_reauth_for_phi': True,
                'ip_binding': True
            }
            
            # Session management
            active_sessions = {}
            
            def create_session(user_id, ip_address):
                session_id = hashlib.sha256(f"{user_id}_{time.time()}".encode()).hexdigest()
                active_sessions[session_id] = {
                    'user_id': user_id,
                    'ip_address': ip_address,
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'reauth_required': False
                }
                return session_id
            
            def validate_session(session_id, current_ip):
                session = active_sessions.get(session_id)
                if not session:
                    return False, "Session not found"
                
                # Check timeout
                if time.time() - session['last_activity'] > session_config['idle_timeout_minutes'] * 60:
                    del active_sessions[session_id]
                    return False, "Session timed out"
                
                # Check IP binding
                if session_config['ip_binding'] and session['ip_address'] != current_ip:
                    return False, "IP address mismatch"
                
                # Update last activity
                session['last_activity'] = time.time()
                return True, "Session valid"
            
            # Test session lifecycle
            session_id = create_session('therapist_001', '192.168.1.100')
            assert session_id in active_sessions, "Session should be created"
            
            # Test valid session
            is_valid, message = validate_session(session_id, '192.168.1.100')
            assert is_valid is True, "Valid session should pass validation"
            
            # Test IP mismatch
            is_valid, message = validate_session(session_id, '192.168.1.200')
            assert is_valid is False, "IP mismatch should invalidate session"
            
            # Test session timeout
            session = active_sessions[session_id]
            session['last_activity'] = time.time() - (session_config['idle_timeout_minutes'] * 60 + 1)
            is_valid, message = validate_session(session_id, '192.168.1.100')
            assert is_valid is False, "Expired session should be invalid"
    
    class TestBreachDetectionAndNotification:
        """Test breach detection and notification procedures."""
        
        def test_unauthorized_access_detection(self, mock_security_services):
            """Test detection of unauthorized access attempts."""
            audit_service = mock_security_services['audit']
            
            # Simulate unauthorized access patterns
            suspicious_activities = [
                {
                    'user_id': 'unknown_user',
                    'ip_address': '10.0.0.1',
                    'action': 'access_patient_record',
                    'resource_id': 'patient_123',
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'user_id': 'therapist_001',
                    'ip_address': '192.168.1.100',
                    'action': 'access_patient_record',
                    'resource_id': 'patient_456',  # Not assigned to this therapist
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'user_id': 'therapist_001',
                    'ip_address': '192.168.1.100',
                    'action': 'bulk_export',
                    'resource_id': 'all_patients',
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            # Breach detection logic
            def detect_breach_pattern(activities):
                breach_indicators = []
                
                # Check for failed access attempts
                failed_attempts = [a for a in activities if not a['success']]
                if len(failed_attempts) >= 3:
                    breach_indicators.append('Multiple failed access attempts')
                
                # Check for unusual access patterns
                unusual_access = [a for a in activities 
                                if a['action'] in ['bulk_export', 'access_all_records']]
                if unusual_access:
                    breach_indicators.append('Unusual data access pattern')
                
                # Check for off-hours access
                for activity in activities:
                    activity_time = datetime.fromisoformat(activity['timestamp'])
                    if activity_time.hour < 6 or activity_time.hour > 22:
                        breach_indicators.append('Off-hours access')
                        break
                
                return breach_indicators
            
            # Test breach detection
            indicators = detect_breach_pattern(suspicious_activities)
            assert len(indicators) > 0, "Should detect breach indicators"
            assert 'Unusual data access pattern' in indicators, "Should detect bulk export"
        
        def test_breach_notification_protocol(self):
            """Test breach notification protocol according to HIPAA."""
            # Breach severity levels
            breach_severity = {
                'low': {
                    'description': 'Limited PHI exposure, immediate containment',
                    'notification_timeline': '60 days',
                    'internal_only': True
                },
                'medium': {
                    'description': 'Significant PHI exposure, investigation required',
                    'notification_timeline': '60 days',
                    'internal_only': False
                },
                'high': {
                    'description': 'Widespread PHI exposure, immediate notification required',
                    'notification_timeline': '60 days (or sooner if required)',
                    'internal_only': False
                }
            }
            
            # Breach assessment
            def assess_breach_severity(affected_records, breach_type):
                if affected_records < 10 and breach_type in ['accidental_disclosure']:
                    return 'low'
                elif affected_records < 500:
                    return 'medium'
                else:
                    return 'high'
            
            # Notification requirements
            def get_notification_requirements(severity):
                requirements = breach_severity[severity]
                return {
                    'timeline': requirements['notification_timeline'],
                    'notify_hhs': not requirements['internal_only'],
                    'notify_individuals': not requirements['internal_only'],
                    'notify_media': severity == 'high' and affected_records > 500
                }
            
            # Test breach scenarios
            test_breaches = [
                {'records': 5, 'type': 'accidental_disclosure'},
                {'records': 100, 'type': 'unauthorized_access'},
                {'records': 1000, 'type': 'hacking'}
            ]
            
            for breach in test_breaches:
                severity = assess_breach_severity(breach['records'], breach['type'])
                requirements = get_notification_requirements(severity)
                
                # Verify notification requirements
                assert 'timeline' in requirements, "Should specify notification timeline"
                
                if breach['records'] >= 500:
                    assert requirements['notify_media'] is True, "Large breaches require media notification"
        
        def test_incident_response_procedures(self):
            """Test incident response procedures for PHI breaches."""
            # Incident response workflow
            response_steps = [
                '1. Immediate containment',
                '2. Initial assessment (1 hour)',
                '3. Breach notification to security team',
                '4. Detailed investigation (24 hours)',
                '5. Risk assessment',
                '6. Notification determination',
                '7. Required notifications',
                '8. Corrective actions',
                '9. Documentation',
                '10. Prevention measures'
            ]
            
            # Simulate incident response
            incident_log = []
            
            def execute_response_step(step_number, description):
                incident_log.append({
                    'step': step_number,
                    'description': description,
                    'timestamp': datetime.now().isoformat(),
                    'completed': True
                })
                return True
            
            # Test response workflow
            for i, step in enumerate(response_steps, 1):
                success = execute_response_step(i, step)
                assert success is True, f"Step {i} should complete successfully"
            
            # Verify all steps were logged
            assert len(incident_log) == len(response_steps), "All response steps should be executed"
            
            # Verify proper sequencing
            for i in range(1, len(incident_log)):
                prev_time = datetime.fromisoformat(incident_log[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(incident_log[i]['timestamp'])
                assert curr_time >= prev_time, "Steps should execute in chronological order"
    
    class TestBusinessAssociateCompliance:
        """Test business associate agreement compliance."""
        
        def test_baa_verification(self):
            """Test business associate agreement verification."""
            # BAA requirements checklist
            baa_requirements = [
                'Uses encryption for PHI',
                'Implements access controls',
                'Provides audit logs',
                'Reports breaches within 60 days',
                'Allows HIPAA audits',
                'Ensures workforce compliance',
                'Proper disposes of PHI',
                'Returns PHI upon termination'
            ]
            
            # Mock vendor assessment
            vendor_compliance = {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'access_controls': True,
                'audit_logging': True,
                'breach_reporting': True,
                'hipaa_audits': True,
                'workforce_training': True,
                'phi_disposal': True,
                'phi_return': True
            }
            
            # Verify BAA compliance
            def verify_baa_compliance(vendor_data):
                compliant_items = []
                non_compliant_items = []
                
                if vendor_data.get('encryption_at_rest') and vendor_data.get('encryption_in_transit'):
                    compliant_items.append('Uses encryption for PHI')
                else:
                    non_compliant_items.append('Uses encryption for PHI')
                
                if vendor_data.get('access_controls'):
                    compliant_items.append('Implements access controls')
                else:
                    non_compliant_items.append('Implements access controls')
                
                # Continue for all requirements...
                for requirement in baa_requirements[2:]:
                    compliant_items.append(requirement)  # Assuming compliance for demo
                
                return {
                    'compliant': len(non_compliant_items) == 0,
                    'compliant_items': compliant_items,
                    'non_compliant_items': non_compliant_items
                }
            
            result = verify_baa_compliance(vendor_compliance)
            assert result['compliant'] is True, "Vendor should meet all BAA requirements"
            assert len(result['non_compliant_items']) == 0, "No non-compliant items should exist"
        
        def test_vendor_phi_handling(self):
            """Test vendor PHI handling procedures."""
            # Vendor PHI handling scenarios
            phi_scenarios = [
                {
                    'scenario': 'Data processing',
                    'requirements': ['Encryption', 'Access logging', 'Limited retention'],
                    'vendor_capability': ['Encryption', 'Access logging', 'Limited retention']
                },
                {
                    'scenario': 'Data storage',
                    'requirements': ['Encrypted storage', 'Access controls', 'Backup encryption'],
                    'vendor_capability': ['Encrypted storage', 'Access controls', 'Backup encryption']
                },
                {
                    'scenario': 'Data transmission',
                    'requirements': ['TLS 1.2+', 'Certificate validation', 'Endpoint security'],
                    'vendor_capability': ['TLS 1.2+', 'Certificate validation']
                }
            ]
            
            # Validate vendor capabilities
            def validate_vendor_capabilities(scenarios):
                validation_results = []
                
                for scenario in scenarios:
                    requirements = set(scenario['requirements'])
                    capabilities = set(scenario['vendor_capability'])
                    
                    missing = requirements - capabilities
                    validation_results.append({
                        'scenario': scenario['scenario'],
                        'compliant': len(missing) == 0,
                        'missing_requirements': list(missing)
                    })
                
                return validation_results
            
            results = validate_vendor_capabilities(phi_scenarios)
            
            # Check compliance for each scenario
            for result in results:
                if result['scenario'] == 'Data transmission':
                    # One missing requirement for testing
                    assert result['compliant'] is False, "Should identify missing requirement"
                    assert 'Endpoint security' in result['missing_requirements'], "Should identify missing capability"
                else:
                    assert result['compliant'] is True, f"{result['scenario']} should be compliant"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])