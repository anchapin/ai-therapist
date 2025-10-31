"""
Comprehensive unit tests for security/pii_protection.py module.
Tests PII detection, masking, and anonymization functionality.
"""

import pytest
import re
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from security.pii_protection import (
        PIIProtection, PIIType, MaskingStrategy, PIIDetectionResult,
        PIIMaskingResult, PIIAnonymizationResult
    )
except ImportError as e:
    pytest.skip(f"security.pii_protection module not available: {e}", allow_module_level=True)


class TestPIIProtection:
    """Test PIIProtection core functionality."""
    
    @pytest.fixture
    def pii_protection(self):
        """Create a PIIProtection instance with default configuration."""
        return PIIProtection()
    
    @pytest.fixture
    def custom_pii_protection(self):
        """Create a PIIProtection instance with custom configuration."""
        config = {
            'name_patterns': [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'],
            'email_patterns': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone_patterns': [r'\b\d{3}-\d{3}-\d{4}\b'],
            'ssn_patterns': [r'\b\d{3}-\d{2}-\d{4}\b'],
            'default_masking_strategy': MaskingStrategy.PARTIAL_MASK
        }
        return PIIProtection(config)
    
    def test_pii_protection_initialization(self, pii_protection):
        """Test PII protection initialization."""
        assert pii_protection.config is not None
        assert hasattr(pii_protection, 'name_patterns')
        assert hasattr(pii_protection, 'email_patterns')
        assert hasattr(pii_protection, 'phone_patterns')
        assert hasattr(pii_protection, 'ssn_patterns')
    
    def test_pii_protection_custom_config(self, custom_pii_protection):
        """Test PII protection with custom configuration."""
        assert custom_pii_protection.config['default_masking_strategy'] == MaskingStrategy.PARTIAL_MASK
        assert len(custom_pii_protection.name_patterns) == 1
    
    def test_detect_pii_email(self, pii_protection):
        """Test PII detection for email addresses."""
        text = "Contact me at john.doe@example.com for more information"
        results = pii_protection.detect_pii(text)
        
        assert len(results) > 0
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        assert len(email_results) == 1
        assert email_results[0].value == "john.doe@example.com"
        assert "john.doe@example.com" in email_results[0].context
    
    def test_detect_pii_phone(self, pii_protection):
        """Test PII detection for phone numbers."""
        text = "Call me at 555-123-4567 tomorrow"
        results = pii_protection.detect_pii(text)
        
        phone_results = [r for r in results if r.pii_type == PIIType.PHONE]
        assert len(phone_results) == 1
        assert phone_results[0].value == "555-123-4567"
    
    def test_detect_pii_ssn(self, pii_protection):
        """Test PII detection for Social Security Numbers."""
        text = "My SSN is 123-45-6789 please keep it confidential"
        results = pii_protection.detect_pii(text)
        
        ssn_results = [r for r in results if r.pii_type == PIIType.SSN]
        assert len(ssn_results) == 1
        assert ssn_results[0].value == "123-45-6789"
    
    def test_detect_pii_multiple_types(self, pii_protection):
        """Test PII detection for multiple types in one text."""
        text = "John Smith's email is john.smith@example.com and his phone is 555-123-4567"
        results = pii_protection.detect_pii(text)
        
        # Should detect name, email, and phone
        detected_types = {r.pii_type for r in results}
        assert PIIType.NAME in detected_types
        assert PIIType.EMAIL in detected_types
        assert PIIType.PHONE in detected_types
    
    def test_detect_pii_no_pii(self, pii_protection):
        """Test PII detection when no PII is present."""
        text = "This is a simple sentence with no personal information"
        results = pii_protection.detect_pii(text)
        
        assert len(results) == 0
    
    def test_detect_pii_empty_text(self, pii_protection):
        """Test PII detection with empty text."""
        results = pii_protection.detect_pii("")
        assert len(results) == 0
    
    def test_detect_pii_voice_transcription(self, pii_protection):
        """Test PII detection specifically for voice transcriptions."""
        text = "My name is Sarah Johnson and you can reach me at sarah.j@company.com"
        results = pii_protection.detect_pii(text, content_type=PIIType.VOICE_TRANSCRIPTION)
        
        assert len(results) >= 2  # Should detect name and email
        voice_results = [r for r in results if r.content_type == PIIType.VOICE_TRANSCRIPTION]
        assert len(voice_results) >= 2
    
    def test_mask_pii_full_mask(self, pii_protection):
        """Test PII masking with full mask strategy."""
        text = "Email me at john.doe@example.com"
        results = pii_protection.detect_pii(text)
        
        masking_result = pii_protection.mask_pii(text, results, MaskingStrategy.FULL_MASK)
        
        assert "[REDACTED:EMAIL]" in masking_result.masked_text
        assert "john.doe@example.com" not in masking_result.masked_text
        assert len(masking_result.masking_results) == 1
    
    def test_mask_pii_partial_mask(self, pii_protection):
        """Test PII masking with partial mask strategy."""
        text = "Email me at john.doe@example.com"
        results = pii_protection.detect_pii(text)
        
        masking_result = pii_protection.mask_pii(text, results, MaskingStrategy.PARTIAL_MASK)
        
        # Should show first and last parts with middle masked
        assert "john" in masking_result.masked_text or "com" in masking_result.masked_text
        assert "john.doe@example.com" not in masking_result.masked_text
    
    def test_mask_pii_hash_mask(self, pii_protection):
        """Test PII masking with hash strategy."""
        text = "Email me at john.doe@example.com"
        results = pii_protection.detect_pii(text)
        
        masking_result = pii_protection.mask_pii(text, results, MaskingStrategy.HASH_MASK)
        
        # Should contain hash of the email
        assert "[HASHED:EMAIL:" in masking_result.masked_text
        assert "john.doe@example.com" not in masking_result.masked_text
    
    def test_mask_pii_remove_strategy(self, pii_protection):
        """Test PII masking with remove strategy."""
        text = "Email me at john.doe@example.com for details"
        results = pii_protection.detect_pii(text)
        
        masking_result = pii_protection.mask_pii(text, results, MaskingStrategy.REMOVE)
        
        assert "john.doe@example.com" not in masking_result.masked_text
        assert "Email me at for details" in masking_result.masked_text
    
    def test_mask_pii_anonymize_strategy(self, pii_protection):
        """Test PII masking with anonymize strategy."""
        text = "John Smith's email is john.smith@example.com"
        results = pii_protection.detect_pii(text)
        
        masking_result = pii_protection.mask_pii(text, results, MaskingStrategy.ANONYMIZE)
        
        assert "[PERSON_1]" in masking_result.masked_text
        assert "[EMAIL_1]" in masking_result.masked_text
        assert "John Smith" not in masking_result.masked_text
        assert "john.smith@example.com" not in masking_result.masked_text
    
    def test_mask_pii_no_detection_results(self, pii_protection):
        """Test PII masking with no detection results."""
        text = "This is a clean sentence"
        results = []
        
        masking_result = pii_protection.mask_pii(text, results)
        
        assert masking_result.masked_text == text
        assert len(masking_result.masking_results) == 0
    
    def test_anonymize_pii(self, pii_protection):
        """Test PII anonymization."""
        text = "John Smith and Jane Doe work at company.com"
        results = pii_protection.detect_pii(text)
        
        anonymization_result = pii_protection.anonymize_pii(text, results)
        
        # Should replace names with placeholders
        assert "[PERSON_1]" in anonymization_result.anonymized_text
        assert "[PERSON_2]" in anonymization_result.anonymized_text
        assert "John Smith" not in anonymization_result.anonymized_text
        assert "Jane Doe" not in anonymization_result.anonymized_text
        assert len(anonymization_result.anonymization_map) > 0
    
    def test_anonymize_pii_with_mapping(self, pii_protection):
        """Test PII anonymization with mapping preservation."""
        text = "John Smith's email is john.smith@example.com"
        results = pii_protection.detect_pii(text)
        
        anonymization_result = pii_protection.anonymize_pii(text, results)
        
        # Check that mapping preserves original values
        assert "John Smith" in anonymization_result.anonymization_map.values()
        assert "john.smith@example.com" in anonymization_result.anonymization_map.values()
    
    def test_sanitize_voice_transcription(self, pii_protection):
        """Test sanitizing voice transcriptions."""
        transcription = "My name is Robert Johnson and my number is 555-987-6543"
        
        sanitized = pii_protection.sanitize_voice_transcription(transcription)
        
        # Should detect and mask PII
        assert "[REDACTED" in sanitized or "[PERSON" in sanitized
        assert "Robert Johnson" not in sanitized or "555-987-6543" not in sanitized
    
    def test_sanitize_voice_transcription_custom_strategy(self, pii_protection):
        """Test sanitizing voice transcriptions with custom strategy."""
        transcription = "Contact Sarah at sarah@example.com"
        
        sanitized = pii_protection.sanitize_voice_transcription(
            transcription, 
            masking_strategy=MaskingStrategy.ANONYMIZE
        )
        
        assert "[PERSON" in sanitized
        assert "[EMAIL" in sanitized
        assert "Sarah" not in sanitized
        assert "sarah@example.com" not in sanitized
    
    def test_is_pii_detected(self, pii_protection):
        """Test checking if PII is detected in text."""
        pii_text = "My email is user@domain.com"
        clean_text = "This is a clean sentence"
        
        assert pii_protection.is_pii_detected(pii_text) is True
        assert pii_protection.is_pii_detected(clean_text) is False
    
    def test_get_pii_summary(self, pii_protection):
        """Test getting PII detection summary."""
        text = "John Smith's email is john.smith@example.com and phone is 555-123-4567"
        results = pii_protection.detect_pii(text)
        
        summary = pii_protection.get_pii_summary(results)
        
        assert 'total_count' in summary
        assert 'by_type' in summary
        assert summary['total_count'] >= 3
        assert PIIType.NAME in summary['by_type']
        assert PIIType.EMAIL in summary['by_type']
        assert PIIType.PHONE in summary['by_type']
    
    def test_validate_pii_patterns(self, pii_protection):
        """Test validation of PII patterns."""
        # Test valid email pattern
        assert pii_protection._test_pattern(
            pii_protection.email_patterns[0], 
            "test@example.com"
        ) is True
        
        # Test invalid email pattern
        assert pii_protection._test_pattern(
            pii_protection.email_patterns[0], 
            "not-an-email"
        ) is False
    
    def test_add_custom_pattern(self, pii_protection):
        """Test adding custom PII patterns."""
        custom_pattern = r'\bCUSTOM-\d{4}\b'
        pii_protection.add_custom_pattern(PIIType.MEDICAL_ID, custom_pattern)
        
        text = "The medical ID is CUSTOM-1234"
        results = pii_protection.detect_pii(text)
        
        medical_results = [r for r in results if r.pii_type == PIIType.MEDICAL_ID]
        assert len(medical_results) == 1
        assert medical_results[0].value == "CUSTOM-1234"
    
    def test_remove_pii_type_patterns(self, pii_protection):
        """Test removing patterns for a specific PII type."""
        # Remove email patterns
        pii_protection.remove_pii_type_patterns(PIIType.EMAIL)
        
        text = "Email me at john.doe@example.com"
        results = pii_protection.detect_pii(text)
        
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        assert len(email_results) == 0
    
    def test_get_supported_pii_types(self, pii_protection):
        """Test getting supported PII types."""
        supported_types = pii_protection.get_supported_pii_types()
        
        assert PIIType.EMAIL in supported_types
        assert PIIType.PHONE in supported_types
        assert PIIType.SSN in supported_types
        assert PIIType.NAME in supported_types
        assert len(supported_types) >= 4


class TestPIIDetectionResult:
    """Test PIIDetectionResult functionality."""
    
    def test_pii_detection_result_creation(self):
        """Test PII detection result creation."""
        result = PIIDetectionResult(
            pii_type=PIIType.EMAIL,
            value="test@example.com",
            confidence=0.95,
            start_pos=10,
            end_pos=26,
            context="Email me at test@example.com"
        )
        
        assert result.pii_type == PIIType.EMAIL
        assert result.value == "test@example.com"
        assert result.confidence == 0.95
        assert result.start_pos == 10
        assert result.end_pos == 26
        assert result.context == "Email me at test@example.com"
    
    def test_pii_detection_result_defaults(self):
        """Test PII detection result with default values."""
        result = PIIDetectionResult(
            pii_type=PIIType.PHONE,
            value="555-123-4567"
        )
        
        assert result.confidence == 0.5  # Default confidence
        assert result.start_pos is None
        assert result.end_pos is None
        assert result.context is None
    
    def test_pii_detection_result_to_dict(self):
        """Test converting PII detection result to dictionary."""
        result = PIIDetectionResult(
            pii_type=PIIType.EMAIL,
            value="test@example.com",
            confidence=0.95
        )
        
        data = result.to_dict()
        
        assert 'pii_type' in data
        assert 'value' in data
        assert 'confidence' in data
        assert data['pii_type'] == PIIType.EMAIL.value
        assert data['value'] == "test@example.com"
        assert data['confidence'] == 0.95


class TestPIIMaskingResult:
    """Test PIIMaskingResult functionality."""
    
    def test_pii_masking_result_creation(self):
        """Test PII masking result creation."""
        result = PIIMaskingResult(
            original_value="test@example.com",
            masked_value="[REDACTED:EMAIL]",
            masking_strategy=MaskingStrategy.FULL_MASK,
            pii_type=PIIType.EMAIL
        )
        
        assert result.original_value == "test@example.com"
        assert result.masked_value == "[REDACTED:EMAIL]"
        assert result.masking_strategy == MaskingStrategy.FULL_MASK
        assert result.pii_type == PIIType.EMAIL
    
    def test_pii_masking_result_to_dict(self):
        """Test converting PII masking result to dictionary."""
        result = PIIMaskingResult(
            original_value="test@example.com",
            masked_value="[REDACTED:EMAIL]",
            masking_strategy=MaskingStrategy.FULL_MASK,
            pii_type=PIIType.EMAIL
        )
        
        data = result.to_dict()
        
        assert 'original_value' in data
        assert 'masked_value' in data
        assert 'masking_strategy' in data
        assert 'pii_type' in data
        assert data['original_value'] == "test@example.com"
        assert data['masked_value'] == "[REDACTED:EMAIL]"


class TestPIIAnonymizationResult:
    """Test PIIAnonymizationResult functionality."""
    
    def test_pii_anonymization_result_creation(self):
        """Test PII anonymization result creation."""
        anonymization_map = {"[PERSON_1]": "John Smith", "[EMAIL_1]": "john@example.com"}
        result = PIIAnonymizationResult(
            anonymized_text="[PERSON_1]'s email is [EMAIL_1]",
            anonymization_map=anonymization_map,
            total_replacements=2
        )
        
        assert result.anonymized_text == "[PERSON_1]'s email is [EMAIL_1]"
        assert result.anonymization_map == anonymization_map
        assert result.total_replacements == 2
    
    def test_pii_anonymization_result_to_dict(self):
        """Test converting PII anonymization result to dictionary."""
        anonymization_map = {"[PERSON_1]": "John Smith"}
        result = PIIAnonymizationResult(
            anonymized_text="[PERSON_1] is here",
            anonymization_map=anonymization_map,
            total_replacements=1
        )
        
        data = result.to_dict()
        
        assert 'anonymized_text' in data
        assert 'anonymization_map' in data
        assert 'total_replacements' in data
        assert data['anonymized_text'] == "[PERSON_1] is here"
        assert data['total_replacements'] == 1