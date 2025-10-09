"""
Simplified unit tests for security/pii_protection.py to boost coverage.
Focuses on core PII detection and protection functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import re
from datetime import datetime
from typing import Dict, List, Any

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from security.pii_protection import PIIDetector
    from security.pii_protection import PIIType, PIIDetectionResult
    from security.pii_protection import PIIProtectionConfig
except ImportError as e:
    pytest.skip(f"pii_protection module not available: {e}", allow_module_level=True)


class TestPIIProtectionCoverage:
    """Targeted unit tests to boost pii_protection.py coverage."""
    
    @pytest.fixture
    def pii_detector(self):
        """Create a PII detector instance."""
        return PIIDetector()
    
    def test_pii_detector_initialization(self, pii_detector):
        """Test PII detector initialization."""
        assert hasattr(pii_detector, 'patterns')
        assert hasattr(pii_detector, 'name_patterns')
        assert hasattr(pii_detector, 'logger')
        
        # Check that key patterns are loaded
        assert PIIType.EMAIL in pii_detector.patterns
        assert PIIType.PHONE in pii_detector.patterns
        assert PIIType.SSN in pii_detector.patterns
        assert PIIType.CREDIT_CARD in pii_detector.patterns
        assert PIIType.ADDRESS in pii_detector.patterns
        assert PIIType.MEDICAL_CONDITION in pii_detector.patterns
    
    def test_detect_email_pii(self, pii_detector):
        """Test email PII detection."""
        email_cases = [
            "user@example.com",
            " john.doe@company.org ",
            "test.email+tag@domain.co.uk",
            "CONTACT: user@example.com",
        ]
        
        for email in email_cases:
            results = pii_detector.detect_pii(email)
            email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
            assert len(email_results) >= 1
            
            # Check detection result structure
            result = email_results[0]
            assert isinstance(result, PIIDetectionResult)
            assert result.pii_type == PIIType.EMAIL
            assert result.confidence == 0.9  # High confidence for regex matches
            assert email in result.value
            assert result.start_pos <= email.index("@")
    
    def test_detect_phone_pii(self, pii_detector):
        """Test phone PII detection."""
        phone_cases = [
            "555-123-4567",
            "(555) 123-4567",
            "+1-555-123-4567",
            "5551234567",
            "Phone: 555-123-4567",
        ]
        
        for phone in phone_cases:
            results = pii_detector.detect_pii(phone)
            phone_results = [r for r in results if r.pii_type == PIIType.PHONE]
            assert len(phone_results) >= 1
            
            result = phone_results[0]
            assert result.pii_type == PIIType.PHONE
            assert result.confidence == 0.9
    
    def test_detect_ssn_pii(self, pii_detector):
        """Test SSN PII detection."""
        ssn_cases = [
            "123-45-6789",
            "123456789",
            " 123-45-6789 ",
            "SSN: 123-45-6789",
        ]
        
        for ssn in ssn_cases:
            results = pii_detector.detect_pii(ssn)
            ssn_results = [r for r in results if r.pii_type == PIIType.SSN]
            assert len(ssn_results) >= 1
            
            result = ssn_results[0]
            assert result.pii_type == PIIType.SSN
            assert result.confidence == 0.9
            assert re.search(r'\d{3}[-]?\d{2}[-]?\d{4}', result.value)
    
    def test_detect_credit_card_pii(self, pii_detector):
        """Test credit card PII detection."""
        credit_card_cases = [
            "4111111111111111",
            " 4111 1111 1111 1111 ",
            "Visa: 4111111111111111",
            "4111-1111-1111-1111",
            "5555555555554444",  # MasterCard
        ]
        
        for card in credit_card_cases:
            results = pii_detector.detect_pii(card)
            card_results = [r for r in results if r.pii_type == PIIType.CREDIT_CARD]
            assert len(card_results) >= 1
            
            result = card_results[0]
            assert result.pii_type == PIIType.CREDIT_CARD
            assert result.confidence == 0.9
    
    def test_detect_dob_pii(self, pii_detector):
        """Test date of birth PII detection."""
        dob_cases = [
            "01/15/1985",
            "1985-01-15",
            "Born: 01/15/1985",
            "DOB: 01-15-1985",
        ]
        
        for dob in dob_cases:
            results = pii_detector.detect_pii(dob)
            dob_results = [r for r in results if r.pii_type == PIIType.DOB]
            assert len(dob_results) >= 1
            
            result = dob_results[0]
            assert result.pii_type == PIIType.DOB
            assert result.confidence == 0.9
    
    def test_detect_medical_id_pii(self, pii_detector):
        """Test medical ID PII detection."""
        medical_id_cases = [
            "MRN: PAT12345",
            "Patient ID: MED-67890",
            "Medical Record: REC-24680",
            "MRN12345",
            "Patient: PAT-24680",
        ]
        
        for medical_id in medical_id_cases:
            results = pii_detector.detect_pii(medical_id)
            medical_results = [r for r in results if r.pii_type == PIIType.MEDICAL_ID]
            assert len(medical_results) >= 1
            
            result = medical_results[0]
            assert result.pii_type == PIIType.MEDICAL_ID
            assert result.confidence == 0.9
    
    def test_detect_medical_condition_pii(self, pii_detector):
        """Test medical condition PII detection."""
        condition_cases = [
            "Patient has depression",
            "Diagnosis: hypertension",
            "Medication: sertraline",
            "Allergy: penicillin",
            "History includes diabetes",
            "Patient reports anxiety",
        ]
        
        for condition in condition_cases:
            results = pii_detector.detect_pii(condition)
            condition_results = [r for r in results if r.pii_type == PIIType.MEDICAL_CONDITION]
            assert len(condition_results) >= 1
            
            result = condition_results[0]
            assert result.pii_type == PIIType.MEDICAL_CONDITION
            assert result.confidence == 0.9
    
    def test_detect_medication_pii(self, pii_detector):
        """Test medication PII detection."""
        medication_cases = [
            "Prozac 20mg",
            "Patient takes sertraline",
            "Prescribed zoloft",
            "Medication: lexapro",
            "Fluoxetine treatment",
        ]
        
        for medication in medication_cases:
            results = pii_detector.detect_pii(medication)
            medication_results = [r for r in results if r.pii_type == PIIType.MEDICATION]
            assert len(medication_results) >= 1
            
            result = medication_results[0]
            assert result.pii_type == PIIType.MEDICATION
            assert result.confidence == 0.9
    
    def test_detect_treatment_pii(self, pii_detector):
        """Test treatment PII detection."""
        treatment_cases = [
            "Patient receives therapy",
            "CBT treatment plan",
            "Cognitive behavioral therapy",
            "Psychotherapy sessions",
            "Medication management",
        ]
        
        for treatment in treatment_cases:
            results = pii_detector.detect_pii(treatment)
            treatment_results = [r for r in results if r.pii_type == PIIType.TREATMENT]
            assert len(treatment_results) >= 1
            
            result = treatment_results[0]
            assert result.pii_type == PIIType.TREATMENT
            assert result.confidence == 0.9
    
    def test_detect_ip_address_pii(self, pii_detector):
        """Test IP address PII detection."""
        ip_cases = [
            "192.168.1.1",
            " 10.0.0.1 ",
            "Server: 172.16.0.1",
            "Localhost: 127.0.0.1",
        ]
        
        for ip in ip_cases:
            results = pii_detector.detect_pii(ip)
            ip_results = [r for r in results if r.pii_type == PIIType.IP_ADDRESS]
            assert len(ip_results) >= 1
            
            result = ip_results[0]
            assert result.pii_type == PIIType.IP_ADDRESS
            assert result.confidence == 0.9
    
    def test_detect_address_pii(self, pii_detector):
        """Test address PII detection."""
        address_cases = [
            "123 Main St, Anytown, CA 12345",
            "P.O. Box 456, City, State 12345",
            "1234 Oak Avenue, Apt 5B, Springfield, IL 62704",
            "15 Broadway, New York, NY 10001",
        ]
        
        for address in address_cases:
            results = pii_detector.detect_pii(address)
            address_results = [r for r in results if r.pii_type == PIIType.ADDRESS]
            assert len(address_results) >= 1
            
            result = address_results[0]
            assert result.pii_type == PIIType.ADDRESS
            assert result.confidence == 0.9
    
    def test_detect_names_pii(self, pii_detector):
        """Test name PII detection."""
        name_cases = [
            "Dr. John Smith",
            "Mr. Robert Johnson",
            "Mary Jane Williams",
            "Dr. Sarah Brown Davis",
            "John David Anderson",
        ]
        
        for name in name_cases:
            results = pii_detector.detect_pii(name)
            name_results = [r for r in results if r.pii_type == PIIType.NAME]
            assert len(name_results) >= 1
            
            result = name_results[0]
            assert result.pii_type == PIIType.NAME
            assert result.confidence == 0.7  # Lower confidence for names
    
    def test_detect_no_pii(self, pii_detector):
        """Test cases with no PII."""
        non_pii_cases = [
            "Hello world",
            "This is a test message",
            "No sensitive information here",
            "Regular business communication",
            "Generic product information",
            "Normal conversation",
            "Weather report",
            "Sports scores",
        ]
        
        for non_pii in non_pii_cases:
            results = pii_detector.detect_pii(non_pii)
            # Should have very few or no PII results
            high_confidence_results = [r for r in results if r.confidence >= 0.9]
            assert len(high_confidence_results) == 0
    
    def test_detect_multiple_pii_types(self, pii_detector):
        """Test detecting multiple PII types in one text."""
        text = """
        Contact John Doe (john.doe@example.com) or call 555-123-4567.
        SSN: 123-45-6789. Credit card: 4111111111111111.
        Born: 01/15/1985. Address: 123 Main St, Anytown, CA 12345.
        """
        
        results = pii_detector.detect_pii(text)
        
        # Should detect multiple PII instances
        assert len(results) >= 8  # At least 8 different PII instances
        
        # Check specific PII types
        pii_types = set(r.pii_type for r in results)
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
        assert PIIType.SSN in pii_types
        assert PIIType.CREDIT_CARD in pii_types
        assert PIIType.DOB in pii_types
        assert PIIType.ADDRESS in pii_types
        assert PIIType.NAME in pii_types
    
    def test_detect_context_aware_pii(self, pii_detector):
        """Test context-aware PII detection."""
        text = "I want to die"
        
        # Without context
        results_no_context = pii_detector.detect_pii(text)
        
        # With voice transcription context
        results_with_context = pii_detector.detect_pii(text, context="voice_transcription")
        
        # Should have crisis detection in voice context
        voice_results = [r for r in results_with_context if r.pii_type == PIIType.VOICE_TRANSCRIPTION]
        assert len(voice_results) >= 1
        assert voice_results[0].context == "crisis_voice_content"
        assert voice_results[0].confidence == 0.95
    
    def test_pii_detection_result_structure(self, pii_detector):
        """Test PII detection result structure."""
        text = "john.doe@example.com"
        results = pii_detector.detect_pii(text)
        
        # Find email result
        email_result = next(r for r in results if r.pii_type == PIIType.EMAIL)
        
        # Check all required fields
        assert hasattr(email_result, 'pii_type')
        assert hasattr(email_result, 'value')
        assert hasattr(email_result, 'start_pos')
        assert hasattr(email_result, 'end_pos')
        assert hasattr(email_result, 'confidence')
        assert hasattr(email_result, 'context')
        assert hasattr(email_result, 'metadata')
        
        # Check field types
        assert isinstance(email_result.pii_type, PIIType)
        assert isinstance(email_result.value, str)
        assert isinstance(email_result.start_pos, int)
        assert isinstance(email_result.end_pos, int)
        assert isinstance(email_result.confidence, float)
        assert email_result.confidence == 0.9
    
    def test_detect_in_dict(self, pii_detector):
        """Test PII detection in dictionary structures."""
        data = {
            "patient_name": "John Doe",
            "contact_email": "john.doe@example.com",
            "phone_number": "555-123-4567",
            "medical_condition": "depression",
            "notes": "Regular meeting notes",
            "metadata": {
                "ssn": "123-45-6789",
                "insurance_id": "INS-24680"
            }
        }
        
        results = pii_detector.detect_in_dict(data)
        
        # Should find PII in nested structure
        assert len(results) >= 6  # At least 6 PII instances
        
        # Check that results include field paths
        field_paths = [path for path, result in results]
        assert "patient_name" in field_paths
        assert "contact_email" in field_paths
        assert "phone_number" in field_paths
        assert "medical_condition" in field_paths
        assert "metadata.ssn" in field_paths
        assert "metadata.insurance_id" in field_paths
    
    def test_empty_text_handling(self, pii_detector):
        """Test handling of empty or invalid text."""
        empty_cases = [
            "",
            None,
            "   ",  # Only whitespace
            123,  # Not a string
            [],  # Not a string
        ]
        
        for empty_case in empty_cases:
            if empty_case is None or not isinstance(empty_case, str):
                results = pii_detector.detect_pii(empty_case)
                assert results == []
            elif isinstance(empty_case, str):
                results = pii_detector.detect_pii(empty_case)
                if empty_case.strip():
                    # Has content, check if PII found
                    assert isinstance(results, list)
                else:
                    # Only whitespace, no PII
                    assert results == []
    
    def test_large_text_performance(self, pii_detector):
        """Test PII detection performance with large text."""
        import time
        
        # Create large text with some PII
        large_text = "Normal text " * 1000 + " john.doe@example.com " + " more text " * 1000
        
        start_time = time.time()
        results = pii_detector.detect_pii(large_text)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        
        # Should find the email
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        assert len(email_results) == 1
    
    def test_unicode_text_handling(self, pii_detector):
        """Test PII detection with Unicode text."""
        unicode_cases = [
            " josé.gonzález@ejemplo.com",  # Spanish
            "jean.dupont@exemple.fr",  # French
            "müller@example.de",  # German
            "北京@例子.公司",  # Chinese
        ]
        
        for unicode_email in unicode_cases:
            results = pii_detector.detect_pii(unicode_email)
            # Email pattern should work with Unicode in local part
            assert isinstance(results, list)
    
    def test_pii_detection_statistics(self, pii_detector):
        """Test PII detection statistics."""
        text = """
        Contact: john.doe@example.com or 555-123-4567.
        Medical records: depression treatment, Prozac prescription.
        Address: 123 Main St, Anytown, CA 12345.
        """
        
        results = pii_detector.detect_pii(text)
        
        # Test statistics calculation
        pii_types = {}
        for result in results:
            if result.pii_type in pii_types:
                pii_types[result.pii_type] += 1
            else:
                pii_types[result.pii_type] = 1
        
        # Should have multiple PII types
        assert len(pii_types) >= 4
        assert all(count >= 1 for count in pii_types.values())
        
        # Most common should be detected
        assert sum(pii_types.values()) == len(results)