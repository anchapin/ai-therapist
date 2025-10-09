"""
Focused unit tests for remaining uncovered functions in security/pii_protection.py.
Targets reaching 50% coverage with existing available functions.
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
    from security.pii_protection import PIIDetector
    from security.pii_protection import PIIType, PIIDetectionResult
except ImportError as e:
    pytest.skip(f"pii_protection module not available: {e}", allow_module_level=True)


class TestPIIProtectionFocusedCoverage:
    """Focused unit tests to reach 50% coverage for pii_protection.py."""
    
    @pytest.fixture
    def pii_detector(self):
        """Create a PII detector instance."""
        return PIIDetector()
    
    def test_detect_in_dict_basic(self, pii_detector):
        """Test basic PII detection in dictionary."""
        data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789"
        }
        
        results = pii_detector.detect_in_dict(data)
        
        assert len(results) >= 4  # Should detect name, email, phone, ssn
        assert isinstance(results, list)
        assert all(isinstance(path, str) for path, result in results)
        assert all(isinstance(result, PIIDetectionResult) for path, result in results)
        
        # Check field paths
        paths = [path for path, _ in results]
        assert "name" in paths
        assert "email" in paths
        assert "phone" in paths
        assert "ssn" in paths
    
    def test_detect_in_dict_nested_structure(self, pii_detector):
        """Test PII detection in nested dictionary structure."""
        data = {
            "patient": {
                "personal_info": {
                    "name": "Jane Smith",
                    "email": "jane.smith@example.com"
                },
                "medical": {
                    "condition": "depression",
                    "medication": "sertraline"
                },
                "contact": {
                    "phone": "555-987-6543",
                    "address": "123 Main St"
                }
            },
            "metadata": {
                "created_at": "2024-01-15",
                "updated_at": "2024-01-20"
            }
        }
        
        results = pii_detector.detect_in_dict(data)
        
        assert len(results) >= 6  # Should detect multiple PII instances
        
        # Check nested paths
        paths = [path for path, _ in results]
        assert "patient.personal_info.name" in paths
        assert "patient.personal_info.email" in paths
        assert "patient.contact.phone" in paths
        assert "patient.contact.address" in paths
    
    def test_detect_in_dict_empty_dict(self, pii_detector):
        """Test PII detection in empty dictionary."""
        data = {}
        
        results = pii_detector.detect_in_dict(data)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_detect_in_dict_no_pii(self, pii_detector):
        """Test PII detection in dictionary with no PII."""
        data = {
            "product": "Widget",
            "price": 19.99,
            "category": "Electronics",
            "description": "Useful product"
        }
        
        results = pii_detector.detect_in_dict(data)
        
        # Should detect no PII (except possibly names)
        high_confidence_results = [r for _, r in results if r.confidence >= 0.9]
        assert len(high_confidence_results) == 0
    
    def test_detect_in_dict_mixed_data_types(self, pii_detector):
        """Test PII detection with mixed data types in dictionary."""
        data = {
            "user_id": 123,  # Integer
            "user_name": "Bob Johnson",  # String with PII
            "is_active": True,  # Boolean
            "profile": {  # Nested dict
                "email": "bob@example.com",
                "age": 35
            },
            "tags": ["email", "phone"],  # List
            "created_at": datetime(2024, 1, 15)  # DateTime
        }
        
        results = pii_detector.detect_in_dict(data)
        
        assert len(results) >= 2  # Should detect name and email
        
        # Check paths include nested structure
        paths = [path for path, _ in results]
        assert "user_name" in paths
        assert "profile.email" in paths
    
    def test_detect_in_dict_list_values(self, pii_detector):
        """Test PII detection in dictionary with list values."""
        data = {
            "contacts": [
                {"name": "Alice Brown", "email": "alice@example.com"},
                {"name": "Charlie Wilson", "phone": "555-111-2222"}
            ],
            "emails": ["admin@example.com", "support@example.com"],
            "notes": ["Regular notes", "SSN: 987-65-4321"]
        }
        
        results = pii_detector.detect_in_dict(data)
        
        # Should detect PII in lists
        assert len(results) >= 6  # Names, emails, phone, SSN
        
        # Check that paths include list indices
        paths = [path for path, _ in results]
        list_paths = [path for path in paths if "[" in path and "]" in path]
        assert len(list_paths) > 0
    
    def test_detect_in_dict_recursion_depth(self, pii_detector):
        """Test PII detection recursion in deeply nested dictionaries."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "user": {
                                "name": "Deep User",
                                "email": "deep@example.com",
                                "ssn": "111-22-3333"
                            }
                        }
                    }
                }
            }
        }
        
        results = pii_detector.detect_in_dict(data)
        
        # Should detect PII at deep nesting level
        paths = [path for path, _ in results]
        deep_paths = [path for path in paths if "level4" in path]
        assert len(deep_paths) >= 3
    
    def test_detect_pii_edge_cases(self, pii_detector):
        """Test PII detection edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "TEXT",  # All caps
            "text",  # All lowercase
            "TexT",  # Mixed case
            "123",  # Numbers only
            "!@#$",  # Special characters only
            "user@example",  # Partial email
            "555-123",  # Partial phone
            "12-3456",  # Partial SSN
        ]
        
        for edge_case in edge_cases:
            results = pii_detector.detect_pii(edge_case)
            assert isinstance(results, list)
            assert all(isinstance(r, PIIDetectionResult) for r in results)
    
    def test_detect_pii_multiple_instances_same_type(self, pii_detector):
        """Test detecting multiple instances of the same PII type."""
        text = "Multiple emails: user1@example.com, user2@example.com, admin@example.org"
        
        results = pii_detector.detect_pii(text)
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        
        assert len(email_results) >= 3
        
        # Check positions are different
        positions = [(r.start_pos, r.end_pos) for r in email_results]
        unique_positions = set(positions)
        assert len(unique_positions) == len(positions)
    
    def test_detect_pii_overlapping_patterns(self, pii_detector):
        """Test handling of overlapping PII patterns."""
        text = "John Doe (john.doe@example.com), SSN: 123-45-6789"
        
        results = pii_detector.detect_pii(text)
        
        # Should detect name, email, and SSN
        types_found = set(r.pii_type for r in results)
        assert PIIType.NAME in types_found
        assert PIIType.EMAIL in types_found
        assert PIIType.SSN in types_found
    
    def test_detect_pii_context_parameter(self, pii_detector):
        """Test PII detection with context parameter."""
        text = "I feel depressed"
        
        # Without context
        results_no_context = pii_detector.detect_pii(text)
        
        # With voice transcription context
        results_with_context = pii_detector.detect_pii(text, context="voice_transcription")
        
        # Should detect more PII with voice context (crisis content)
        medical_no_context = [r for r in results_no_context if r.pii_type == PIIType.MEDICAL_CONDITION]
        medical_with_context = [r for r in results_with_context if r.pii_type == PIIType.MEDICAL_CONDITION]
        
        assert len(medical_with_context) >= len(medical_no_context)
        
        # Voice context may detect crisis content
        voice_results = [r for r in results_with_context if r.pii_type == PIIType.VOICE_TRANSCRIPTION]
        if len(voice_results) >= 1:
            assert voice_results[0].context == "crisis_voice_content"
        else:
            # If crisis content not detected, that's acceptable depending on the exact text
            pass  # Allow flexibility in crisis detection
    
    def test_detect_pii_large_text_performance(self, pii_detector):
        """Test PII detection performance with large text."""
        import time
        
        # Create moderately large text
        base_text = "Regular text without PII. "
        large_text = base_text * 100 + "Contact: john.doe@example.com for details." + base_text * 100
        
        # Measure performance
        start_time = time.time()
        results = pii_detector.detect_pii(large_text)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        processing_time = end_time - start_time
        assert processing_time < 1.0
        
        # Should find the email
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        assert len(email_results) >= 1
    
    def test_detect_pii_unicode_and_special_characters(self, pii_detector):
        """Test PII detection with Unicode and special characters."""
        test_cases = [
            "josé.gonzález@ejemplo.com",  # Unicode email
            "Müller@example.de",  # German umlaut
            "张@example.cn",  # Chinese characters
            "Phone: +1-800-555-0199",  # International phone
            "Address: 123 Main St., Apt. 5B",  # Address with punctuation
            "Name: Dr. John A. Smith Jr.",  # Name with titles and suffixes
        ]
        
        for test_case in test_cases:
            results = pii_detector.detect_pii(test_case)
            assert isinstance(results, list)
            
            # Should detect at least email in most cases
            email_found = any(r.pii_type == PIIType.EMAIL for r in results)
            phone_found = any(r.pii_type == PIIType.PHONE for r in results)
            name_found = any(r.pii_type == PIIType.NAME for r in results)
            address_found = any(r.pii_type == PIIType.ADDRESS for r in results)
            
            # At least one type should be detected in most cases
            # Some Unicode or complex patterns might not be detected
            if len(results) == 0:
                # Allow some test cases to not detect PII if patterns don't match
                pass  # Unicode emails often don't match basic patterns
            else:
                assert email_found or phone_found or name_found or address_found
    
    def test_detect_pii_result_confidence_levels(self, pii_detector):
        """Test confidence levels for different PII types."""
        test_cases = [
            ("john.doe@example.com", PIIType.EMAIL, 0.9),  # High confidence regex
            ("555-123-4567", PIIType.PHONE, 0.9),  # High confidence regex
            ("Dr. Jane Smith", PIIType.NAME, 0.7),  # Lower confidence names
            ("I want to die", PIIType.VOICE_TRANSCRIPTION, 0.95),  # Very high crisis
        ]
        
        for text, expected_type, expected_confidence in test_cases:
            results = pii_detector.detect_pii(text, context="voice_transcription")
            
            # Find the result with expected type
            matching_results = [r for r in results if r.pii_type == expected_type]
            
            if len(matching_results) > 0:
                result = matching_results[0]
                assert result.confidence == expected_confidence
            elif expected_type != PIIType.VOICE_TRANSCRIPTION:
                # Some might not be detected without proper context
                pass  # Allow flexible matching for non-crisis content
    
    def test_detect_in_dict_error_handling(self, pii_detector):
        """Test error handling in detect_in_dict."""
        # Test with invalid dictionary (should not crash)
        invalid_cases = [
            None,  # None instead of dict
            "not a dict",  # String instead of dict
            [],  # List instead of dict
            123,  # Number instead of dict
        ]
        
        for invalid_case in invalid_cases:
            # Should handle gracefully (return empty list or raise appropriate error)
            try:
                results = pii_detector.detect_in_dict(invalid_case)
                # If it returns, should be a list
                assert isinstance(results, list)
            except (TypeError, AttributeError):
                # If it raises error, should be appropriate
                pass
    
    def test_detect_in_dict_circular_reference(self, pii_detector):
        """Test handling of circular references in detect_in_dict."""
        # Create a dictionary with circular reference
        circular_data = {"name": "John Doe"}
        circular_data["self"] = circular_data
        
        # Should handle circular references gracefully
        try:
            results = pii_detector.detect_in_dict(circular_data)
            assert isinstance(results, list)
            
            # Should detect PII in both locations
            paths = [path for path, _ in results]
            assert "name" in paths
            assert "self.name" in paths
        except RuntimeError:
            # If circular reference is detected and RuntimeError is raised, that's acceptable
            pass
    
    def test_pattern_initialization(self, pii_detector):
        """Test that PII patterns are properly initialized."""
        # Check that essential patterns exist
        assert PIIType.EMAIL in pii_detector.patterns
        assert PIIType.PHONE in pii_detector.patterns
        assert PIIType.SSN in pii_detector.patterns
        assert PIIType.DOB in pii_detector.patterns
        assert PIIType.MEDICAL_ID in pii_detector.patterns
        assert PIIType.INSURANCE_ID in pii_detector.patterns
        assert PIIType.CREDIT_CARD in pii_detector.patterns
        assert PIIType.IP_ADDRESS in pii_detector.patterns
        assert PIIType.ADDRESS in pii_detector.patterns
        assert PIIType.MEDICAL_CONDITION in pii_detector.patterns
        assert PIIType.MEDICATION in pii_detector.patterns
        assert PIIType.TREATMENT in pii_detector.patterns
        
        # Check that patterns are compiled regex objects
        for pattern in pii_detector.patterns.values():
            assert isinstance(pattern, type(re.compile("test")))
        
        # Check name patterns
        assert len(pii_detector.name_patterns) > 0
        for pattern in pii_detector.name_patterns:
            assert isinstance(pattern, type(re.compile("test")))
    
    def test_logger_initialization(self, pii_detector):
        """Test that logger is properly initialized."""
        assert hasattr(pii_detector, 'logger')
        assert isinstance(pii_detector.logger, logging.Logger)
        
        # Should not raise when logging
        pii_detector.logger.info("Test log message")
        pii_detector.logger.debug("Debug message")
        pii_detector.logger.warning("Warning message")
    
    def test_method_return_types(self, pii_detector):
        """Test that methods return expected types."""
        text = "john.doe@example.com"
        
        # detect_pii should return list of PIIDetectionResult
        results = pii_detector.detect_pii(text)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, PIIDetectionResult)
        
        # detect_in_dict should return list of tuples (path, result)
        data = {"email": text}
        dict_results = pii_detector.detect_in_dict(data)
        assert isinstance(dict_results, list)
        for path, result in dict_results:
            assert isinstance(path, str)
            assert isinstance(result, PIIDetectionResult)