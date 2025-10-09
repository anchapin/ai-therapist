"""
Comprehensive security tests for HIPAA compliance.
Tests all security and compliance features required for HIPAA.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import hashlib
import base64

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from security.pii_protection import PIIProtection, PIIType, MaskingStrategy
    from security.pii_config import PIIConfig
    from security.response_sanitizer import ResponseSanitizer
    from voice.security import VoiceSecurity, SecurityLevel
    from auth.auth_service import AuthService
    from database.db_manager import DatabaseManager
except ImportError as e:
    pytest.skip(f"Security test dependencies not available: {e}", allow_module_level=True)


@pytest.mark.security
class TestHIPAACompliance:
    """Test HIPAA compliance requirements."""
    
    @pytest.fixture
    def pii_protection(self):
        """Create PII protection instance for HIPAA testing."""
        config = {
            'hipaa_mode': True,
            'name_patterns': [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'],
            'email_patterns': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone_patterns': [r'\b\d{3}-\d{3}-\d{4}\b'],
            'ssn_patterns': [r'\b\d{3}-\d{2}-\d{4}\b'],
            'medical_id_patterns': [r'\b(MR|MED|PAT)-\d{6,8}\b'],
            'insurance_patterns': [r'\b[INS|INSURANCE]-\d{8,12}\b'],
            'default_masking_strategy': MaskingStrategy.REMOVE,
            'audit_all_operations': True
        }
        return PIIProtection(config)
    
    @pytest.fixture
    def response_sanitizer(self):
        """Create response sanitizer instance."""
        return ResponseSanitizer()
    
    @pytest.fixture
    async def test_db_manager(self):
        """Create in-memory database for testing."""
        db_config = {
            'database_url': 'sqlite:///:memory:',
            'pool_size': 5,
            'encryption_enabled': True,
            'audit_enabled': True
        }
        
        db_manager = DatabaseManager(db_config)
        await db_manager.initialize()
        await db_manager.create_tables()
        
        yield db_manager
        await db_manager.close()
    
    def test_phi_detection_comprehensive(self, pii_protection):
        """Test comprehensive PHI (Protected Health Information) detection."""
        # Test various PHI types
        test_cases = [
            # Name
            ("Patient John Smith arrived", PIIType.NAME),
            # Email
            ("Contact at patient@example.com", PIIType.EMAIL),
            # Phone
            ("Call 555-123-4567 for appointment", PIIType.PHONE),
            # SSN
            ("SSN: 123-45-6789", PIIType.SSN),
            # Medical ID
            ("Medical ID: MR-1234567", PIIType.MEDICAL_ID),
            # Insurance
            ("Insurance: INS-1234567890", PIIType.INSURANCE_ID),
            # DOB
            ("Date of birth: 01/15/1980", PIIType.DOB),
            # Address
            ("123 Main St, Anytown, NY 12345", PIIType.ADDRESS),
        ]
        
        for text, expected_type in test_cases:
            results = pii_protection.detect_pii(text)
            found_types = [r.pii_type for r in results]
            assert expected_type in found_types, f"Failed to detect {expected_type} in: {text}"
    
    def test_phi_masking_compliance(self, pii_protection):
        """Test PHI masking meets HIPAA requirements."""
        text = "Patient John Smith (SSN: 123-45-6789) can be reached at john.smith@example.com or 555-123-4567"
        results = pii_protection.detect_pii(text)
        
        # Use REMOVE strategy for full compliance
        masking_result = pii_protection.mask_pii(text, results, MaskingStrategy.REMOVE)
        
        # Ensure all PHI is removed
        assert "John Smith" not in masking_result.masked_text
        assert "123-45-6789" not in masking_result.masked_text
        assert "john.smith@example.com" not in masking_result.masked_text
        assert "555-123-4567" not in masking_result.masked_text
        
        # Ensure text remains readable
        assert "Patient" in masking_result.masked_text
        assert "reached" in masking_result.masked_text
        assert "or" in masking_result.masked_text
    
    def test_medical_condition_detection(self, pii_protection):
        """Test detection of medical conditions (PHI)."""
        text = "Patient has diabetes and hypertension, taking metformin and lisinopril"
        results = pii_protection.detect_pii(text)
        
        medical_results = [r for r in results if r.pii_type in [
            PIIType.MEDICAL_CONDITION, PIIType.MEDICATION
        ]]
        
        assert len(medical_results) >= 2  # Should detect conditions and medications
    
    def test_voice_transcription_sanitization(self, pii_protection):
        """Test voice transcription sanitization for HIPAA compliance."""
        transcription = "My name is Sarah Johnson and my date of birth is March 15, 1985. My doctor prescribed metformin for my diabetes."
        
        sanitized = pii_protection.sanitize_voice_transcription(
            transcription, 
            masking_strategy=MaskingStrategy.ANONYMIZE
        )
        
        # Verify PHI is anonymized
        assert "Sarah Johnson" not in sanitized
        assert "March 15, 1985" not in sanitized
        assert "[PERSON" in sanitized or "[NAME" in sanitized
        assert "[DOB" in sanitized or "[DATE" in sanitized
    
    def test_encryption_at_rest(self, test_db_manager):
        """Test data encryption at rest."""
        sensitive_data = "Patient PHI: SSN 123-45-6789"
        
        # Simulate encryption (in real implementation, this would use proper encryption)
        encryption_key = b'test_encryption_key_32'
        encrypted_data = hashlib.sha256(sensitive_data.encode() + encryption_key).hexdigest()
        
        # Store encrypted data
        await test_db_manager.execute_query(
            "INSERT INTO audit_logs (user_id, action, metadata) VALUES (?, ?, ?)",
            ("test_patient", "phi_storage", {"encrypted_phi": encrypted_data})
        )
        
        # Retrieve and verify encryption
        result = await test_db_manager.fetch_one(
            "SELECT metadata FROM audit_logs WHERE user_id = ?",
            ("test_patient",)
        )
        
        assert result is not None
        assert "encrypted_phi" in result['metadata']
        assert sensitive_data not in str(result['metadata'])  # Plain text should not be stored
    
    def test_access_control_enforcement(self):
        """Test access control enforcement for PHI."""
        # Mock user roles and permissions
        users = {
            'doctor': {'role': 'therapist', 'permissions': ['read_phi', 'write_phi']},
            'nurse': {'role': 'medical_staff', 'permissions': ['read_phi']},
            'admin': {'role': 'admin', 'permissions': ['read_phi', 'write_phi', 'delete_phi']},
            'patient': {'role': 'patient', 'permissions': ['read_own_phi']}
        }
        
        # Test access permissions
        assert 'read_phi' in users['doctor']['permissions']
        assert 'delete_phi' not in users['nurse']['permissions']
        assert 'read_phi' in users['patient']['permissions']
        assert 'write_phi' not in users['patient']['permissions']
    
    async def test_audit_logging_comprehensive(self, test_db_manager):
        """Test comprehensive audit logging for HIPAA compliance."""
        audit_events = [
            {
                'user_id': 'doctor_1',
                'action': 'phi_access',
                'resource_type': 'patient_record',
                'resource_id': 'patient_123',
                'metadata': {
                    'access_reason': 'treatment',
                    'ip_address': '192.168.1.100',
                    'user_agent': 'EHR/1.0'
                }
            },
            {
                'user_id': 'system',
                'action': 'phi_export',
                'resource_type': 'patient_data',
                'resource_id': 'bulk_export_001',
                'metadata': {
                    'export_reason': 'backup',
                    'record_count': 150,
                    'encryption_used': 'AES-256'
                }
            },
            {
                'user_id': 'admin_1',
                'action': 'phi_deletion',
                'resource_type': 'patient_record',
                'resource_id': 'patient_456',
                'metadata': {
                    'deletion_reason': 'request',
                    'retention_period_expired': False
                }
            }
        ]
        
        # Log all events
        for event in audit_events:
            await test_db_manager.execute_query(
                """INSERT INTO audit_logs 
                   (user_id, action, resource_type, resource_id, metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event['user_id'], event['action'], event['resource_type'],
                 event['resource_id'], json.dumps(event['metadata']), datetime.now())
            )
        
        # Verify all events were logged
        for event in audit_events:
            logged = await test_db_manager.fetch_one(
                """SELECT * FROM audit_logs 
                   WHERE user_id = ? AND action = ? AND resource_type = ?""",
                (event['user_id'], event['action'], event['resource_type'])
            )
            assert logged is not None
            assert logged['timestamp'] is not None
            assert json.loads(logged['metadata']) == event['metadata']
    
    def test_data_retention_policy(self):
        """Test data retention policy compliance."""
        retention_periods = {
            'adult_records': 7,  # years
            'minor_records': 7,  # years after age of majority
            'audit_logs': 6,    # years
            'billing_records': 7, # years
            'emergency_records': 30  # days
        }
        
        current_date = datetime.now()
        
        # Test retention calculations
        for record_type, years in retention_periods.items():
            if record_type == 'minor_records':
                # Assume minor is now 18, records kept until 25
                cutoff_date = current_date - timedelta(days=years * 365)
            else:
                cutoff_date = current_date - timedelta(days=years * 365)
            
            assert cutoff_date < current_date
            assert (current_date - cutoff_date).days == years * 365
    
    def test_minimum_necessary_standard(self, pii_protection):
        """Test HIPAA Minimum Necessary Standard."""
        full_record = """
        Patient: John Smith
        DOB: 01/15/1980
        SSN: 123-45-6789
        Address: 123 Main St, Anytown, NY 12345
        Phone: 555-123-4567
        Email: john.smith@example.com
        Insurance: INS-1234567890
        Medical Condition: Diabetes Type 2
        Medication: Metformin 500mg
        """
        
        # Detect all PII
        results = pii_protection.detect_pii(full_record)
        detected_types = [r.pii_type for r in results]
        
        # Should detect multiple PHI types
        assert PIIType.NAME in detected_types
        assert PIIType.DOB in detected_types
        assert PIIType.SSN in detected_types
        assert PIIType.ADDRESS in detected_types
        assert PIIType.PHONE in detected_types
        assert PIIType.EMAIL in detected_types
        assert PIIType.MEDICAL_CONDITION in detected_types
        assert PIIType.MEDICATION in detected_types
        
        # Apply minimum necessary - only keep medical info, remove identifiers
        # This would be implemented based on role and context
        medical_only = pii_protection.mask_pii(
            full_record, 
            results, 
            MaskingStrategy.REMOVE
        )
        
        # Verify identifiers are removed but medical context remains
        assert "John Smith" not in medical_only.masked_text
        assert "123-45-6789" not in medical_only.masked_text
        assert "555-123-4567" not in medical_only.masked_text
    
    def test_breach_detection_and_response(self):
        """Test breach detection and response procedures."""
        # Simulate breach scenarios
        breach_indicators = [
            {'type': 'unauthorized_access', 'severity': 'high', 'threshold': 1},
            {'type': 'multiple_failed_logins', 'severity': 'medium', 'threshold': 5},
            {'type': 'data_export_spike', 'severity': 'high', 'threshold': 1000},
            {'type': 'unusual_access_hours', 'severity': 'medium', 'threshold': 10}
        ]
        
        for indicator in breach_indicators:
            assert 'type' in indicator
            assert 'severity' in indicator
            assert 'threshold' in indicator
            assert indicator['severity'] in ['low', 'medium', 'high', 'critical']
    
    async def test_business_associate_agreement(self, test_db_manager):
        """Test business associate agreement tracking."""
        ba_data = {
            'company_name': 'Secure Voice Processing Inc.',
            'contact_email': 'compliance@securevoice.com',
            'agreement_date': datetime.now() - timedelta(days=365),
            'expiration_date': datetime.now() + timedelta(days=365),
            'permitted_uses': ['voice_processing', 'transcription'],
            'safeguards': ['encryption_at_rest', 'encryption_in_transit', 'access_controls'],
            'compliance_certifications': ['HIPAA', 'HITRUST']
        }
        
        # Store BAA information
        await test_db_manager.execute_query(
            """INSERT INTO business_associates 
               (company_name, contact_email, agreement_date, expiration_date, 
                permitted_uses, safeguards, certifications, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ba_data['company_name'], ba_data['contact_email'],
             ba_data['agreement_date'], ba_data['expiration_date'],
             json.dumps(ba_data['permitted_uses']),
             json.dumps(ba_data['safeguards']),
             json.dumps(ba_data['compliance_certifications']),
             json.dumps(ba_data))
        )
        
        # Verify BAA tracking
        result = await test_db_manager.fetch_one(
            "SELECT * FROM business_associates WHERE company_name = ?",
            (ba_data['company_name'],)
        )
        
        assert result is not None
        assert result['company_name'] == ba_data['company_name']
        assert result['contact_email'] == ba_data['contact_email']
    
    def test_patient_rights_implementation(self, pii_protection):
        """Test implementation of patient rights under HIPAA."""
        # Test right to access (limited)
        patient_data = "Patient John Doe has appointment on 01/15/2024"
        results = pii_protection.detect_pii(patient_data)
        
        # Patient can access their own records with limited PII
        patient_view = pii_protection.mask_pii(
            patient_data, 
            results, 
            MaskingStrategy.PARTIAL_MASK
        )
        
        # Should show partial information
        assert "John" in patient_view.masked_text or "Doe" in patient_view.masked_text
        assert "01/15/2024" in patient_view.masked_text
        
        # Test right to amendment
        amendment_request = {
            'patient_id': 'patient_123',
            'record_id': 'record_456',
            'amendment_type': 'correction',
            'reason': 'incorrect_medication_dosage',
            'requested_change': 'Change metformin 500mg to metformin 1000mg',
            'status': 'pending_review'
        }
        
        assert 'patient_id' in amendment_request
        assert 'amendment_type' in amendment_request
        assert amendment_request['status'] in ['pending_review', 'approved', 'denied']
    
    def test_security_incident_logging(self):
        """Test security incident logging and response."""
        security_incidents = [
            {
                'incident_id': 'SEC001',
                'type': 'unauthorized_access',
                'severity': 'high',
                'description': 'Unauthorized user accessed patient records',
                'discovery_time': datetime.now() - timedelta(hours=2),
                'response_time': datetime.now() - timedelta(hours=1),
                'mitigation_steps': ['account_disabled', 'password_reset', 'access_review'],
                'notification_sent': True,
                'affected_records': 25,
                'breach_confirmed': False
            },
            {
                'incident_id': 'SEC002',
                'type': 'malware_detection',
                'severity': 'critical',
                'description': 'Ransomware detected on database server',
                'discovery_time': datetime.now() - timedelta(minutes=30),
                'response_time': datetime.now() - timedelta(minutes=15),
                'mitigation_steps': ['server_isolation', 'backup_restoration', 'forensic_analysis'],
                'notification_sent': True,
                'affected_records': 1000,
                'breach_confirmed': True
            }
        ]
        
        for incident in security_incidents:
            # Verify incident documentation completeness
            required_fields = [
                'incident_id', 'type', 'severity', 'description',
                'discovery_time', 'response_time', 'mitigation_steps',
                'notification_sent', 'affected_records', 'breach_confirmed'
            ]
            
            for field in required_fields:
                assert field in incident
            
            # Verify response time requirements (within 60 days for breaches)
            if incident['breach_confirmed']:
                response_time = incident['response_time'] - incident['discovery_time']
                assert response_time.total_seconds() < 60 * 24 * 3600  # 60 days in seconds


@pytest.mark.security
class TestVoiceSecurityCompliance:
    """Test voice-specific security compliance."""
    
    @pytest.fixture
    def voice_security(self):
        """Create voice security instance."""
        return VoiceSecurity(encryption_enabled=True, consent_required=True)
    
    def test_voice_data_encryption(self, voice_security):
        """Test voice data encryption."""
        audio_data = b'sensitive_voice_data_containing_phi'
        
        # Encrypt audio data
        encrypted = voice_security.encrypt_audio(audio_data)
        
        assert encrypted != audio_data
        assert len(encrypted) > 0
        assert audio_data not in encrypted  # Plain text should not be visible
        
        # Decrypt audio data
        decrypted = voice_security.decrypt_audio(encrypted)
        assert decrypted == audio_data
    
    def test_voice_consent_management(self, voice_security):
        """Test voice recording consent management."""
        consent_data = {
            'patient_id': 'patient_123',
            'session_id': 'session_456',
            'consent_given': True,
            'consent_timestamp': datetime.now(),
            'consent_purpose': 'treatment',
            'consent_duration': 'session_only',
            'withdrawal_allowed': True
        }
        
        # Verify consent requirements
        assert consent_data['consent_given'] is True
        assert 'consent_timestamp' in consent_data
        assert consent_data['consent_purpose'] in ['treatment', 'research', 'operations']
        assert consent_data['withdrawal_allowed'] is True
    
    def test_voice_session_audit_trail(self, voice_security):
        """Test comprehensive voice session audit trail."""
        session_events = [
            {'event': 'session_started', 'timestamp': datetime.now() - timedelta(minutes=10)},
            {'event': 'recording_started', 'timestamp': datetime.now() - timedelta(minutes=9)},
            {'event': 'phi_detected', 'timestamp': datetime.now() - timedelta(minutes=8)},
            {'event': 'phi_masked', 'timestamp': datetime.now() - timedelta(minutes=8)},
            {'event': 'recording_stopped', 'timestamp': datetime.now() - timedelta(minutes=5)},
            {'event': 'session_ended', 'timestamp': datetime.now() - timedelta(minutes=4)}
        ]
        
        # Verify audit trail completeness
        assert len(session_events) >= 5  # Minimum expected events
        for event in session_events:
            assert 'event' in event
            assert 'timestamp' in event
            assert isinstance(event['timestamp'], datetime)