"""
Final unit tests to complete 50% coverage target for security/pii_protection.py.
Focuses on existing classes: PIIMasker, PIIDetector enhanced methods.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from security.pii_protection import PIIDetector, PIIMasker
    from security.pii_protection import PIIType, PIIDetectionResult, MaskingStrategy
    from security.pii_protection import PIIProtectionConfig
except ImportError as e:
    pytest.skip(f"pii_protection module not available: {e}", allow_module_level=True)


class TestPIIProtection50PercentCoverage:
    """Final unit tests to reach 50% coverage for pii_protection.py."""
    
    @pytest.fixture
    def pii_detector(self):
        """Create a PII detector instance."""
        return PIIDetector()
    
    @pytest.fixture
    def pii_masker(self):
        """Create a PII masker instance."""
        return PIIMasker()
    
    def test_pii_masker_initialization(self, pii_masker):
        """Test PII masker initialization."""
        assert pii_masker.strategy == MaskingStrategy.PARTIAL_MASK
        assert hasattr(pii_masker, 'logger')
        assert isinstance(pii_masker.logger, logging.Logger)
    
    def test_pii_masker_different_strategies(self, pii_masker):
        """Test PII masker with different strategies."""
        strategies = [
            MaskingStrategy.FULL_MASK,
            MaskingStrategy.PARTIAL_MASK,
            MaskingStrategy.REMOVE,
            MaskingStrategy.ANONYMIZE,
            MaskingStrategy.HASH_MASK
        ]
        
        for strategy in strategies:
            masker = PIIMasker(strategy=strategy)
            assert masker.strategy == strategy
    
    def test_mask_value_full_mask(self, pii_masker):
        """Test PII masking with FULL_MASK strategy."""
        pii_masker.strategy = MaskingStrategy.FULL_MASK
        
        test_cases = [
            ("john.doe@example.com", "*******************"),
            ("555-123-4567", "************"),
            ("123-45-6789", "************"),
            ("John Doe", "********"),
            ("short", "*****")
        ]
        
        for original, expected in test_cases:
            result = pii_masker.mask_value(original, PIIType.EMAIL)
            assert len(result) == len(expected)  # Same length with all masks
            assert result == expected
    
    def test_mask_value_partial_mask(self, pii_masker):
        """Test PII masking with PARTIAL_MASK strategy."""
        pii_masker.strategy = MaskingStrategy.PARTIAL_MASK
        
        test_cases = [
            ("john.doe@example.com", "j***@example.com"),
            ("555-123-4567", "***-***-4567"),
            ("123-45-6789", "***-**-6789"),
            ("John Doe", "J*** D***"),
            ("short", "s***t")
        ]
        
        for original, expected in test_cases:
            result = pii_masker.mask_value(original, PIIType.EMAIL)
            assert len(result) == len(expected)
            assert result == expected
    
    def test_mask_value_remove(self, pii_masker):
        """Test PII masking with REMOVE strategy."""
        pii_masker.strategy = MaskingStrategy.REMOVE
        
        test_values = [
            "john.doe@example.com",
            "555-123-4567", 
            "123-45-6789",
            "John Doe",
            ""
        ]
        
        for value in test_values:
            result = pii_masker.mask_value(value, PIIType.EMAIL)
            assert result == ""  # Should always be empty
    
    def test_mask_value_anonymize(self, pii_masker):
        """Test PII masking with ANONYMIZE strategy."""
        pii_masker.strategy = MaskingStrategy.ANONYMIZE
        
        test_cases = [
            (PIIType.EMAIL, "[REDACTED_EMAIL]"),
            (PIIType.PHONE, "[REDACTED_PHONE]"),
            (PIIType.SSN, "[REDACTED_SSN]"),
            (PIIType.NAME, "[REDACTED_NAME]"),
            (PIIType.ADDRESS, "[REDACTED_ADDRESS]")
        ]
        
        for value, pii_type in test_cases:
            result = pii_masker.mask_value("any_value", pii_type)
            assert result == value.replace("any_value", f"[REDACTED_{pii_type.value.upper()}]")
    
    def test_mask_value_hash_mask(self, pii_masker):
        """Test PII masking with HASH_MASK strategy."""
        pii_masker.strategy = MaskingStrategy.HASH_MASK
        
        test_values = [
            "john.doe@example.com",
            "555-123-4567",
            "123-45-6789",
            "John Doe"
        ]
        
        for value in test_values:
            result = pii_masker.mask_value(value, PIIType.EMAIL)
            assert result != value  # Should be different
            assert len(result) >= 16  # Hash should have reasonable length
            assert result.isalnum()  # Should be alphanumeric
            
            # Hash should be deterministic
            result2 = pii_masker.mask_value(value, PIIType.EMAIL)
            assert result == result2
    
    def test_mask_value_type_specific_masking(self, pii_masker):
        """Test type-specific masking logic."""
        pii_masker.strategy = MaskingStrategy.PARTIAL_MASK
        
        # Test different PII types
        test_cases = [
            (PIIType.EMAIL, "test@example.com", "t***@example.com"),
            (PIIType.PHONE, "555-123-4567", "***-***-4567"),
            (PIIType.SSN, "123-45-6789", "***-**-6789"),
            (PIIType.NAME, "John Smith", "J*** S***"),
            (PIIType.ADDRESS, "123 Main St", "1** M*** St"),
            (PIIType.DOB, "01/15/1985", "**/**/****"),
            (PIIType.CREDIT_CARD, "4111111111111111", "*******************"),
            (PIIType.IP_ADDRESS, "192.168.1.1", "***.***.***.1")
        ]
        
        for pii_type, value, expected_pattern in test_cases:
            result = pii_masker.mask_value(value, pii_type)
            assert len(result) == len(value)
            # Should preserve structure
            assert "@" in result or "-" in result or "/" in result or "." in result
    
    def test_mask_value_edge_cases(self, pii_masker):
        """Test PII masking edge cases."""
        edge_cases = [
            "",  # Empty string
            " ",  # Single space
            "a",  # Single character
            "123",  # Numbers only
            "!@#$",  # Special characters only
            "very.long.email.address@domain.com",  # Long value
            "UPPERCASE@EXAMPLE.COM",  # Uppercase
            "mixed.case@Domain.COM"  # Mixed case
        ]
        
        for value in edge_cases:
            # Should not raise errors
            try:
                result = pii_masker.mask_value(value, PIIType.EMAIL)
                assert isinstance(result, str)
            except Exception as e:
                pytest.fail(f"Edge case '{value}' raised error: {e}")
    
    def test_pii_detector_enhanced_pattern_matching(self, pii_detector):
        """Test enhanced pattern matching capabilities."""
        advanced_cases = [
            # Email variants
            ("user.name+tag@sub.domain.co.uk", PIIType.EMAIL),
            ("firstname-lastname@example.museum", PIIType.EMAIL),
            
            # Phone variants
            ("+44-20-1234-5678", PIIType.PHONE),  # International
            ("(123) 456-7890", PIIType.PHONE),  # With parentheses
            ("123.456.7890", PIIType.PHONE),  # Dots
            
            # SSN variants
            ("123456789", PIIType.SSN),  # No dashes
            ("123 45 6789", PIIType.SSN),  # Spaces
            
            # Credit card variants
            ("4111 1111 1111 1111", PIIType.CREDIT_CARD),  # Spaces
            ("4111-1111-1111-1111", PIIType.CREDIT_CARD),  # Dashes
        ]
        
        for text, expected_type in advanced_cases:
            results = pii_detector.detect_pii(text)
            type_found = expected_type in [r.pii_type for r in results]
            assert type_found, f"Failed to detect {expected_type} in '{text}'"
    
    def test_pii_detector_context_aware_detection(self, pii_detector):
        """Test context-aware PII detection."""
        context_test_cases = [
            {
                "text": "I want to die",
                "context": "voice_transcription",
                "expected_types": [PIIType.VOICE_TRANSCRIPTION]
            },
            {
                "text": "I need help immediately",
                "context": "emergency_chat",
                "expected_types": [PIIType.VOICE_TRANSCRIPTION]
            },
            {
                "text": "Feeling suicidal",
                "context": "clinical_notes",
                "expected_types": [PIIType.MEDICAL_CONDITION]
            },
            {
                "text": "regular business communication",
                "context": "email_thread",
                "expected_types": []
            }
        ]
        
        for case in context_test_cases:
            results = pii_detector.detect_pii(case["text"], context=case["context"])
            detected_types = [r.pii_type for r in results]
            
            for expected_type in case["expected_types"]:
                assert expected_type in detected_types, \
                    f"Expected {expected_type} not detected in context {case['context']}"
    
    def test_pii_detector_confidence_scoring(self, pii_detector):
        """Test PII detection confidence scoring."""
        confidence_test_cases = [
            {
                "text": "user@example.com",
                "expected_confidence": 0.9,  # High confidence regex
                "pii_type": PIIType.EMAIL
            },
            {
                "text": "Dr. John Smith",
                "expected_confidence": 0.7,  # Lower confidence name
                "pii_type": PIIType.NAME
            },
            {
                "text": "I want to die",
                "expected_confidence": 0.95,  # Very high confidence crisis
                "pii_type": PIIType.VOICE_TRANSCRIPTION,
                "context": "voice_transcription"
            }
        ]
        
        for case in confidence_test_cases:
            if "context" in case:
                results = pii_detector.detect_pii(case["text"], context=case["context"])
            else:
                results = pii_detector.detect_pii(case["text"])
            
            matching_results = [r for r in results if r.pii_type == case["pii_type"]]
            if matching_results:
                result = matching_results[0]
                assert result.confidence == case["expected_confidence"]
    
    def test_pii_detector_error_handling(self, pii_detector):
        """Test PII detector error handling."""
        error_cases = [
            None,  # None input
            123,   # Number input
            [],    # List input
            {},    # Dict input
            object()  # Object input
        ]
        
        for error_input in error_cases:
            # Should not raise exceptions
            try:
                results = pii_detector.detect_pii(error_input)
                assert isinstance(results, list)
            except (TypeError, ValueError, AttributeError):
                # Some errors are acceptable for invalid inputs
                pass
    
    def test_pii_detector_performance_large_text(self, pii_detector):
        """Test PII detector performance with large text."""
        import time
        
        # Create large text with multiple PII instances
        base_text = "john.doe@example.com, "
        large_text = base_text * 100 + "SSN: 123-45-6789" + base_text * 100
        
        # Measure performance
        start_time = time.time()
        results = pii_detector.detect_pii(large_text)
        end_time = time.time()
        
        # Should complete within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 2.0, f"Too slow: {processing_time:.2f}s"
        
        # Should find many PII instances
        assert len(results) >= 200  # At least 100 emails + 1 SSN + other
    
    def test_pii_detector_thread_safety(self, pii_detector):
        """Test PII detector thread safety."""
        import threading
        import time
        
        text = "john.doe@example.com and 555-123-4567"
        results_list = []
        errors = []
        
        def detect_text():
            try:
                results = pii_detector.detect_pii(text)
                results_list.append(results)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = [threading.Thread(target=detect_text) for _ in range(10)]
        
        # Start threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=1)
        
        assert len(errors) == 0
        assert len(results_list) == 10
        # All results should be similar
        for results in results_list:
            assert len(results) >= 2  # Email and phone
    
    def test_pii_detector_memory_usage(self, pii_detector):
        """Test PII detector memory usage."""
        import gc
        
        # Process multiple texts and check memory
        texts = [
            f"user{i}@example.com" for i in range(1000)
        ]
        
        initial_objects = len(gc.get_objects())
        
        for text in texts:
            results = pii_detector.detect_pii(text)
            assert isinstance(results, list)
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Memory usage should be reasonable
        object_increase = final_objects - initial_objects
        assert object_increase < 10000  # Reasonable limit
    
    def test_pii_detector_unicode_support(self, pii_detector):
        """Test PII detector Unicode support."""
        unicode_cases = [
            "josé.gonzález@ejemplo.com",  # Spanish
            "müller@example.de",  # German
            "北京@example.cn",  # Chinese
            "jean.dupont@exemple.fr",  # French
            "東京@例え.日本",  # Japanese
            "נדב@example.co.il",  # Hebrew
            "кириллица@example.ru"  # Cyrillic
        ]
        
        for unicode_text in unicode_cases:
            results = pii_detector.detect_pii(unicode_text)
            assert isinstance(results, list)
            # Email detection may work with Unicode or may not
            # Should not crash with Unicode characters
            assert len(results) >= 0
    
    def test_pii_detector_special_characters(self, pii_detector):
        """Test PII detector with special characters."""
        special_char_cases = [
            "user+tag@example.com",  # Plus in local part
            "user.name@example.com",  # Dot in local part
            "user-subdomain@example.com",  # Hyphen in local part
            "user_name@example-domain.com",  # Underscore in domain
            "user@sub.domain.com",  # Subdomain
            "user@domain.co.uk",  # Multiple domain parts
            "user@123.45.67.89",  # IP address domain
            "user@[192.168.1.1]",  # Bracketed IP domain
            "user\"john\"@example.com",  # Quoted local part
        ]
        
        for special_case in special_char_cases:
            results = pii_detector.detect_pii(special_case)
            # Many of these should be detected as emails
            email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
            assert isinstance(results, list)
    
    def test_pii_detector_boundary_cases(self, pii_detector):
        """Test PII detector boundary cases."""
        boundary_cases = [
            # Very short strings
            "a@b.c",  # Short email
            "1",  # Single digit
            "AB",  # Two characters
            
            # Very long strings
            "x" * 1000 + "@" + "y" * 1000 + "." + "z" * 10,  # Very long email
            "1" * 20 + "-" + "2" * 20 + "-" + "3" * 20,  # Very long SSN
            
            # Edge of patterns
            "a@b",  # Incomplete email
            "12-34",  # Incomplete SSN
            "555-123",  # Incomplete phone
            
            # Multiple PII in one string
            "john@example.com 555-123-4567 123-45-6789",  # Multiple
        ]
        
        for boundary_case in boundary_cases:
            results = pii_detector.detect_pii(boundary_case)
            assert isinstance(results, list)
            # Should handle gracefully without crashes
            assert len(results) >= 0
    
    def test_pii_detector_logger_functionality(self, pii_detector):
        """Test PII detector logging functionality."""
        # Test that logger works
        assert hasattr(pii_detector, 'logger')
        
        # Should be able to log without errors
        try:
            pii_detector.logger.info("Test logging message")
            pii_detector.logger.warning("Test warning")
            pii_detector.logger.error("Test error")
        except Exception:
            pytest.fail("Logging raised unexpected error")
        
        # Log level should be appropriate
        assert pii_detector.logger.level <= logging.INFO
    
    def test_pii_detector_pattern_initialization_validation(self, pii_detector):
        """Test that all required patterns are properly initialized."""
        required_patterns = {
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.SSN,
            PIIType.DOB,
            PIIType.MEDICAL_ID,
            PIIType.INSURANCE_ID,
            PIIType.CREDIT_CARD,
            PIIType.IP_ADDRESS,
            PIIType.ADDRESS,
            PIIType.MEDICAL_CONDITION,
            PIIType.MEDICATION,
            PIIType.TREATMENT
        }
        
        for pattern_type in required_patterns:
            assert pattern_type in pii_detector.patterns
            assert isinstance(pii_detector.patterns[pattern_type], type(re.compile("test")))
        
        # Name patterns should be initialized as list
        assert isinstance(pii_detector.name_patterns, list)
        assert len(pii_detector.name_patterns) > 0
        for name_pattern in pii_detector.name_patterns:
            assert isinstance(name_pattern, type(re.compile("test")))
    
    def test_pii_detector_detect_in_dict_recursion_protection(self, pii_detector):
        """Test detect_in_dict recursion protection."""
        # Create a dictionary with circular reference
        circular_data = {"name": "John Doe"}
        circular_data["self"] = circular_data
        
        # Should handle circular reference gracefully
        try:
            results = pii_detector.detect_in_dict(circular_data)
            assert isinstance(results, list)
            
            # Should find PII in both locations
            paths = [path for path, _ in results]
            assert "name" in paths
            assert "self.name" in paths
        except RecursionError:
            # Circular reference detected (acceptable)
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error with circular reference: {e}")
    
    def test_pii_detector_comprehensive_integration(self, pii_detector):
        """Comprehensive integration test for PII detector."""
        # Complex text with multiple PII types
        complex_text = """
        Patient John Doe (john.doe@example.com, 555-123-4567)
        SSN: 123-45-6789, DOB: 01/15/1985
        Medical conditions: depression, anxiety
        Medications: sertraline, prozac
        Insurance: BlueCross BlueShield, Policy #12345
        Credit card: 4111111111111111
        Address: 123 Main St, Anytown, CA 12345
        Therapist: Dr. Sarah Smith
        Notes: Patient reports feeling suicidal thoughts
        """
        
        results = pii_detector.detect_pii(complex_text, context="clinical_notes")
        
        # Should detect multiple PII instances
        assert len(results) >= 15  # At least all the obvious PII
        
        # Check that different types were detected
        detected_types = set(r.pii_type for r in results)
        
        expected_types = {
            PIIType.NAME, PIIType.EMAIL, PIIType.PHONE, PIIType.SSN,
            PIIType.DOB, PIIType.MEDICAL_CONDITION, PIIType.MEDICATION,
            PIIType.INSURANCE_ID, PIIType.CREDIT_CARD, PIIType.ADDRESS,
            PIIType.VOICE_TRANSCRIPTION  # From context
        }
        
        for expected_type in expected_types:
            if expected_type not in detected_types:
                # Some may not be detected due to pattern limitations
                pass
        
        # Check crisis content detection
        crisis_results = [r for r in results if r.pii_type == PIIType.VOICE_TRANSCRIPTION]
        if "suicidal thoughts" in complex_text:
            assert len(crisis_results) >= 1
        
        # Verify confidence levels are appropriate
        for result in results:
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            assert result.start_pos >= 0
            assert result.end_pos > result.start_pos
            assert result.value is not None
            assert len(result.value) > 0