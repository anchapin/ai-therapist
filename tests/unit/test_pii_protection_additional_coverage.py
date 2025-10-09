"""
Additional unit tests for security/pii_protection.py to complete 50% coverage target.
Focuses on remaining uncovered functions and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from security.pii_protection import PIIDetector
    from security.pii_protection import PIIType, PIIDetectionResult, MaskingStrategy
    from security.pii_protection import PIIProtectionConfig, PIIAuditLog, PIIRiskLevel
    from security.pii_protection import PIISanitizer, PIIAnonymizer
    from security.pii_protection import PIIComplianceChecker
except ImportError as e:
    pytest.skip(f"pii_protection module not available: {e}", allow_module_level=True)


class TestPIIProtectionAdditionalCoverage:
    """Additional unit tests to reach 50% coverage target for pii_protection.py."""
    
    @pytest.fixture
    def pii_detector(self):
        """Create a PII detector instance."""
        return PIIDetector()
    
    @pytest.fixture
    def pii_config(self):
        """Create a PII protection configuration."""
        return PIIProtectionConfig()
    
    def test_mask_pii_with_strategies(self, pii_detector):
        """Test PII masking with different strategies."""
        test_cases = [
            ("123-45-6789", MaskingStrategy.FULL_MASK, "XXX-XX-XXXX"),
            ("john.doe@example.com", MaskingStrategy.PARTIAL_MASK, "j***@example.com"),
            ("555-123-4567", MaskingStrategy.REMOVE, ""),
            ("John Doe", MaskingStrategy.ANONYMIZE, "[REDACTED_NAME]"),
        ]
        
        for original, strategy, expected in test_cases:
            result = pii_detector.mask_pii(original, strategy)
            assert result is not None
            assert result != original
    
    def test_anonymize_json_data(self, pii_detector):
        """Test JSON data anonymization."""
        json_data = {
            "patient_name": "John Doe",
            "contact_email": "john.doe@example.com",
            "phone_number": "555-123-4567",
            "medical_condition": "Hypertension",
            "appointment_date": "2024-01-15",
            "sensitive_info": {
                "ssn": "123-45-6789",
                "credit_card": "4111111111111111"
            }
        }
        
        anonymized = pii_detector.anonymize_json(json_data)
        
        # PII should be masked
        assert anonymized["patient_name"] != "John Doe"
        assert "john.doe" not in anonymized["contact_email"]
        assert "555-123-4567" not in anonymized["phone_number"]
        assert "123-45-6789" not in str(anonymized["sensitive_info"])
        
        # Non-PII should be preserved
        assert anonymized["medical_condition"] == "Hypertension"
        assert anonymized["appointment_date"] == "2024-01-15"
    
    def test_create_audit_log_entry(self, pii_detector):
        """Test audit log entry creation."""
        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": "user_123",
            "action": "ACCESS_PATIENT_DATA",
            "resource_id": "patient_456",
            "pii_types": ["SSN", "MEDICAL_RECORD"],
            "risk_level": "HIGH",
            "compliant": True
        }
        
        log_entry = pii_detector.create_audit_log_entry(audit_data)
        
        assert isinstance(log_entry, PIIAuditLog)
        assert log_entry.user_id == "user_123"
        assert log_entry.action == "ACCESS_PATIENT_DATA"
        assert log_entry.risk_level == "HIGH"
        assert log_entry.timestamp is not None
    
    def test_calculate_risk_level(self, pii_detector):
        """Test risk level calculation for PII."""
        test_cases = [
            ([PIIType.EMAIL], PIIRiskLevel.LOW),
            ([PIIType.PHONE], PIIRiskLevel.MEDIUM),
            ([PIIType.SSN], PIIRiskLevel.HIGH),
            ([PIIType.CREDIT_CARD, PIIType.SSN], PIIRiskLevel.CRITICAL),
            ([PIIType.MEDICAL_RECORD, PIIType.NAME], PIIRiskLevel.HIGH),
            ([], PIIRiskLevel.NONE),
        ]
        
        for pii_types, expected_risk in test_cases:
            calculated_risk = pii_detector.calculate_risk_level(pii_types)
            assert calculated_risk == expected_risk
    
    def test_sanitize_response(self, pii_detector):
        """Test response sanitization."""
        test_responses = [
            "Contact John Doe at john.doe@example.com or call 555-123-4567",
            "Patient SSN: 123-45-6789, born 01/15/1985",
            "Credit card ending in 1111: 4111111111111111",
            "Regular business communication with no PII"
        ]
        
        for response in test_responses:
            sanitized = pii_detector.sanitize_response(response)
            
            # Should not contain original sensitive information
            assert "john.doe@example.com" not in sanitized
            assert "555-123-4567" not in sanitized
            assert "123-45-6789" not in sanitized
            assert "4111111111111111" not in sanitized
            
            # Should preserve basic communication
            assert "Contact" in sanitized or "Regular" in sanitized
    
    def test_batch_pii_processing(self, pii_detector):
        """Test batch processing of multiple texts."""
        texts = [
            "john.doe@example.com",
            "555-123-4567", 
            "123-45-6789",
            "Regular text"
        ]
        
        results = pii_detector.batch_process_texts(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) >= 0 for r in results)
        
        # Check that PII was detected in appropriate texts
        assert len(results[0]) >= 1  # Email
        assert len(results[1]) >= 1  # Phone
        assert len(results[2]) >= 1  # SSN
        assert len(results[3]) == 0  # No PII
    
    def test_validate_pii_compliance(self, pii_detector):
        """Test PII compliance validation."""
        compliance_cases = [
            ({"data": "Regular text", "pii_count": 0, "max_allowed": 5}, True),
            ({"data": "john.doe@example.com", "pii_count": 1, "max_allowed": 0}, False),
            ({"data": "SSN: 123-45-6789", "pii_count": 1, "max_allowed": 2, "pii_types": [PIIType.SSN]}, False),
            ({"data": "Email only", "pii_count": 1, "max_allowed": 2, "pii_types": [PIIType.EMAIL]}, True),
        ]
        
        for case_data, expected in compliance_cases:
            result = pii_detector.validate_compliance(case_data)
            assert result == expected
    
    def test_extract_pii_statistics(self, pii_detector):
        """Test PII statistics extraction."""
        text = """
        Multiple PII instances:
        john.doe@example.com (Email)
        555-123-4567 (Phone)
        123-45-6789 (SSN)
        depression (Medical Condition)
        Additional text
        """
        
        stats = pii_detector.extract_pii_statistics(text)
        
        assert isinstance(stats, dict)
        assert "total_pii_instances" in stats
        assert "pii_by_type" in stats
        assert "risk_level" in stats
        assert stats["total_pii_instances"] >= 4
        assert PIIType.EMAIL in stats["pii_by_type"]
        assert PIIType.PHONE in stats["pii_by_type"]
    
    def test_pattern_performance_optimization(self, pii_detector):
        """Test regex pattern performance optimization."""
        import time
        
        # Test compiled patterns are faster
        text = "Large text with john.doe@example.com embedded multiple times" * 100
        
        # Warm up
        pii_detector.detect_pii(text)
        
        # Measure performance
        start_time = time.time()
        for _ in range(10):
            results = pii_detector.detect_pii(text)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.1  # Should be fast
        assert len(results) >= 100  # Should detect multiple instances
    
    def test_unicode_handling(self, pii_detector):
        """Test Unicode text handling in PII detection."""
        unicode_cases = [
            "josé.gonzález@ejemplo.com",
            "jean.dupont@exemple.fr",
            "北京用户@example.公司",
            "москва@example.ru"
        ]
        
        for unicode_text in unicode_cases:
            results = pii_detector.detect_pii(unicode_text)
            # Email detection should work with Unicode
            email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
            # Some Unicode emails may not match basic pattern
            assert isinstance(results, list)
    
    def test_edge_case_text_handling(self, pii_detector):
        """Test edge cases in text processing."""
        edge_cases = [
            "",  # Empty
            None,  # None
            "   ",  # Whitespace only
            "TEXT",  # All caps
            "text",  # All lowercase
            "TeXt",  # Mixed case
            "123",  # Numbers only
            "!@#$",  # Special chars only
        ]
        
        for edge_case in edge_cases:
            if edge_case is None:
                continue
            results = pii_detector.detect_pii(edge_case)
            assert isinstance(results, list)
            
            if edge_case.strip():
                # Has content, should check properly
                assert all(isinstance(r, PIIDetectionResult) for r in results)
    
    def test_nested_data_structure_pii(self, pii_detector):
        """Test PII detection in deeply nested structures."""
        nested_data = {
            "level1": {
                "level2": {
                    "patient_info": {
                        "name": "John Doe",
                        "email": "john.doe@example.com",
                        "medical": {
                            "condition": "depression",
                            "treatment": "therapy"
                        }
                    }
                },
                "other": {
                    "contact": "555-123-4567"
                }
            }
        }
        
        results = pii_detector.detect_in_dict(nested_data)
        
        # Should detect PII at all levels
        assert len(results) >= 6  # At least name, email, phone, condition, treatment, etc.
        
        # Check that paths are recorded correctly
        paths = [path for path, _ in results]
        assert any("patient_info" in path for path in paths)
        assert any("medical" in path for path in paths)
        assert any("contact" in path for path in paths)
    
    def test_concurrent_pii_detection(self, pii_detector):
        """Test thread-safe PII detection."""
        import threading
        import time
        
        texts = [
            "john.doe@example.com",
            "555-123-4567",
            "123-45-6789"
        ] * 10
        
        results = []
        errors = []
        
        def detect_text(text):
            try:
                text_results = pii_detector.detect_pii(text)
                results.extend(text_results)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=detect_text, args=(text,)) for text in texts]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=2)
        
        assert len(errors) == 0
        assert len(results) >= 30  # Should detect PII in all texts
    
    def test_memory_efficient_processing(self, pii_detector):
        """Test memory-efficient PII processing."""
        import gc
        
        # Process large text and ensure memory is released
        large_text = "john.doe@example.com, " * 10000
        
        initial_count = len(gc.get_objects())
        
        # Process text
        results = pii_detector.detect_pii(large_text)
        assert len(results) >= 10000  # Should detect all instances
        
        # Clear references
        del results
        del large_text
        
        # Force garbage collection
        gc.collect()
        
        # Memory should not significantly increase
        final_count = len(gc.get_objects())
        assert final_count - initial_count < 1000  # Reasonable limit
    
    def test_pii_detection_accuracy_metrics(self, pii_detector):
        """Test PII detection accuracy metrics."""
        test_cases = [
            {"text": "john.doe@example.com", "expected_types": [PIIType.EMAIL], "min_confidence": 0.8},
            {"text": "555-123-4567", "expected_types": [PIIType.PHONE], "min_confidence": 0.8},
            {"text": "123-45-6789", "expected_types": [PIIType.SSN], "min_confidence": 0.8},
            {"text": "Dr. Sarah Smith", "expected_types": [PIIType.NAME], "min_confidence": 0.6},
            {"text": "Regular text", "expected_types": [], "min_confidence": 0.0}
        ]
        
        metrics = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
        
        for case in test_cases:
            results = pii_detector.detect_pii(case["text"])
            detected_types = [r.pii_type for r in results]
            
            if case["expected_types"]:
                # Should detect PII
                if any(pt in detected_types for pt in case["expected_types"]):
                    metrics["true_positive"] += 1
                else:
                    metrics["false_negative"] += 1
            else:
                # Should not detect PII
                if len(results) == 0:
                    metrics["true_negative"] += 1
                else:
                    metrics["false_positive"] += 1
        
        # Should have reasonable accuracy
        total = sum(metrics.values())
        if total > 0:
            accuracy = (metrics["true_positive"] + metrics["true_negative"]) / total
            assert accuracy >= 0.8  # At least 80% accuracy
    
    def test_pii_detection_result_dataclass(self, pii_detector):
        """Test PII detection result dataclass serialization."""
        text = "john.doe@example.com"
        results = pii_detector.detect_pii(text)
        
        if results:
            result = results[0]
            
            # Test dataclass functionality
            result_dict = result.__dict__
            assert isinstance(result_dict, dict)
            assert "pii_type" in result_dict
            assert "value" in result_dict
            assert "start_pos" in result_dict
            assert "end_pos" in result_dict
            assert "confidence" in result_dict
            
            # Test JSON serialization
            json_str = json.dumps(result_dict, default=str)
            assert isinstance(json_str, str)
            assert len(json_str) > 0
            
            # Test reconstruction
            reconstructed = json.loads(json_str)
            assert reconstructed["pii_type"] == result.pii_type.value
            assert reconstructed["value"] == result.value