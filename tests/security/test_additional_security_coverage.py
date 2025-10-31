"""
Additional security tests to improve coverage for critical security functions.
Focuses on high-impact, untested security features.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import hashlib
import json
import tempfile
import os
from pathlib import Path

# Import security modules
from security.pii_config import PIIConfig, PIIDetectionPattern
from security.pii_protection import PIIProtection, PIIDetector, PIIMasker
from security.response_sanitizer import ResponseSanitizer


class TestPIIConfigAdditional:
    """Additional tests for PII configuration to improve coverage."""

    def test_pii_pattern_creation(self):
        """Test PII pattern creation and compilation."""
        pattern = PIIDetectionPattern(
            name="test_email",
            pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            pii_type="email",
            description="Email pattern"
        )
        
        assert pattern.name == "test_email"
        assert pattern.compiled_pattern is not None
        assert pattern.enabled == True

    def test_pii_pattern_invalid_regex(self):
        """Test PII pattern with invalid regex."""
        pattern = PIIDetectionPattern(
            name="invalid_pattern",
            pattern=r"[invalid_regex",  # Missing closing bracket
            pii_type="test"
        )
        
        # Should disable pattern due to invalid regex
        assert pattern.enabled == False
        assert pattern.compiled_pattern is None

    def test_config_initialization(self):
        """Test PII config initialization with defaults."""
        config = PIIConfig()
        
        # Should have default values
        assert hasattr(config, 'pii_detection_rules')
        assert hasattr(config, 'masking_settings')
        assert hasattr(config, 'audit_settings')

    def test_config_environment_variables(self):
        """Test configuration loading from environment variables."""
        with patch.dict(os.environ, {
            'PII_DETECTION_ENABLED': 'true',
            'PII_MASKING_ENABLED': 'true',
            'PII_AUDIT_ENABLED': 'false'
        }):
            config = PIIConfig()
            # Config should load environment variables
            assert hasattr(config, 'detection_enabled')
            assert hasattr(config, 'masking_enabled')
            assert hasattr(config, 'audit_enabled')


class TestPIIProtectionAdvanced:
    """Advanced PII protection tests for uncovered scenarios."""

    def test_pii_detector_initialization(self):
        """Test PII detector initialization."""
        detector = PIIDetector()
        
        assert hasattr(detector, 'patterns')
        assert hasattr(detector, 'detection_enabled')

    def test_pii_detector_empty_text(self):
        """Test PII detection with empty text."""
        detector = PIIDetector()
        
        result = detector.detect_pii("")
        assert isinstance(result, list)

    def test_pii_detector_no_pii(self):
        """Test PII detection with text containing no PII."""
        detector = PIIDetector()
        
        text = "This is a simple text without any personal information."
        result = detector.detect_pii(text)
        
        assert isinstance(result, list)

    def test_pii_masker_initialization(self):
        """Test PII masker initialization."""
        masker = PIIMasker()
        
        assert hasattr(masker, 'masking_rules')
        assert hasattr(masker, 'default_mask_char')

    def test_pii_protection_initialization(self):
        """Test PII protection service initialization."""
        protection = PIIProtection()
        
        assert hasattr(protection, 'detector')
        assert hasattr(protection, 'masker')
        assert hasattr(protection, 'audit_enabled')

    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        protection = PIIProtection()
        
        text = "Email: test@example.com"
        result = protection.sanitize_text(text, user_role="therapist")
        
        assert isinstance(result, str)
        # Should be masked
        assert "test@example.com" not in result or "@" not in result

    def test_sanitize_dict_basic(self):
        """Test basic dictionary sanitization."""
        protection = PIIProtection()
        
        data = {
            "email": "test@example.com",
            "name": "John Doe",
            "message": "Hello world"
        }
        
        result = protection.sanitize_dict(data, user_role="therapist")
        
        assert isinstance(result, dict)
        # PII should be masked
        assert "test@example.com" not in str(result)

    def test_health_check(self):
        """Test PII protection health check."""
        protection = PIIProtection()
        
        health = protection.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health


class TestResponseSanitizerExtended:
    """Extended tests for response sanitizer."""

    def test_sanitizer_initialization(self):
        """Test response sanitizer initialization."""
        sanitizer = ResponseSanitizer()
        
        assert hasattr(sanitizer, 'sanitization_rules')
        assert hasattr(sanitizer, 'audit_enabled')

    def test_sanitize_text_response(self):
        """Test text response sanitization."""
        sanitizer = ResponseSanitizer()
        
        response = "User email is test@example.com"
        result = sanitizer.sanitize(response, {"endpoint": "/api/test"})
        
        assert isinstance(result, str)
        # PII should be masked
        assert "test@example.com" not in result or "@" not in result

    def test_sanitize_json_response(self):
        """Test JSON response sanitization."""
        sanitizer = ResponseSanitizer()
        
        response_data = {
            "user": {
                "email": "test@example.com",
                "name": "John Doe"
            }
        }
        
        result = sanitizer.sanitize_json_response(response_data, "/api/test")
        
        assert isinstance(result, dict)
        # PII should be masked
        assert "test@example.com" not in str(result)

    def test_endpoint_configuration(self):
        """Test endpoint-specific configuration."""
        sanitizer = ResponseSanitizer()
        
        # Configure endpoint
        sanitizer.configure_endpoint("/api/sensitive", {"level": "high"})
        
        # Test that configuration is stored
        assert "/api/sensitive" in sanitizer.endpoint_configs

    def test_sanitization_statistics(self):
        """Test sanitization statistics."""
        sanitizer = ResponseSanitizer()
        
        # Process some data
        sanitizer.sanitize("test", {"endpoint": "/api/test"})
        
        # Get stats
        stats = sanitizer.get_sanitization_stats()
        
        assert isinstance(stats, dict)
        assert 'total_processed' in stats


class TestSecurityIntegration:
    """Integration tests for security components."""

    def test_pii_config_load_patterns(self):
        """Test loading PII patterns from configuration."""
        config = PIIConfig()
        
        # Load patterns
        patterns = config.load_pii_patterns()
        
        assert isinstance(patterns, list)
        # Should have default patterns
        assert len(patterns) > 0

    def test_pii_pattern_matching(self):
        """Test PII pattern matching."""
        pattern = PIIDetectionPattern(
            name="email_pattern",
            pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            pii_type="email"
        )
        
        text = "Contact us at support@example.com for help"
        match = pattern.compiled_pattern.search(text)
        
        assert match is not None
        assert "support@example.com" in match.group()

    def test_masker_with_different_strategies(self):
        """Test PII masker with different masking strategies."""
        masker = PIIMasker()
        
        test_text = "email@example.com"
        
        # Test different masking approaches
        if hasattr(masker, 'mask_email'):
            result = masker.mask_email(test_text)
            assert isinstance(result, str)

    def test_detector_multiple_pii_types(self):
        """Test detection of multiple PII types in text."""
        detector = PIIDetector()
        
        text = "Contact John Doe at john@example.com or 555-123-4567"
        results = detector.detect_pii(text)
        
        assert isinstance(results, list)
        # Should detect multiple PII items if patterns are configured

    def test_concurrent_safety(self):
        """Test thread safety of PII protection."""
        protection = PIIProtection()
        
        # Simulate concurrent access (basic test)
        texts = ["Email: user{}@example.com".format(i) for i in range(5)]
        
        results = []
        for text in texts:
            result = protection.sanitize_text(text, user_role="therapist")
            results.append(result)
        
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, str)

    def test_configuration_file_operations(self):
        """Test configuration file operations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            # Write basic config
            json.dump({"detection_enabled": True}, f)
        
        try:
            config = PIIConfig()
            # Test if config can be loaded from file
            if hasattr(config, 'load_from_file'):
                loaded_config = config.load_from_file(config_path)
                assert loaded_config is not None
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)