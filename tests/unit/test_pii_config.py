"""
Unit tests for PII Configuration module.

Tests PII detection patterns, rules, and configuration management
with comprehensive coverage of all methods and edge cases.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, mock_open

# Set environment variables for testing
os.environ['PII_DETECT_NAMES'] = 'true'
os.environ['PII_DETECT_EMAILS'] = 'true'
os.environ['PII_DETECT_PHONES'] = 'true'
os.environ['PII_DETECT_ADDRESSES'] = 'true'
os.environ['PII_DETECT_SSN'] = 'true'
os.environ['PII_DETECT_DOB'] = 'true'
os.environ['PII_DETECT_MEDICAL_CONDITIONS'] = 'true'
os.environ['PII_DETECT_MEDICATIONS'] = 'true'
os.environ['PII_DETECT_TREATMENTS'] = 'true'
os.environ['PII_DETECT_MEDICAL_IDS'] = 'true'
os.environ['PII_DETECT_CREDIT_CARDS'] = 'true'
os.environ['PII_DETECT_BANK_ACCOUNTS'] = 'true'
os.environ['PII_DETECT_INSURANCE_IDS'] = 'true'
os.environ['PII_DETECT_IP_ADDRESSES'] = 'true'
os.environ['PII_DETECT_LOCATION'] = 'true'
os.environ['PII_DETECT_VOICE_TRANSCRIPTIONS'] = 'true'
os.environ['PII_DETECT_CRISIS_KEYWORDS'] = 'true'
os.environ['PII_CONTEXT_AWARE'] = 'true'
os.environ['PII_FUZZY_MATCHING'] = 'false'
os.environ['PII_WHITELIST_WORDS'] = 'test,example,demo'
os.environ['PII_BLACKLIST_WORDS'] = 'secret,confidential,private'

from security.pii_config import (
    PIIDetectionPattern, PIIDetectionRules, PIIConfig
)


class TestPIIDetectionPattern:
    """Test PIIDetectionPattern dataclass."""
    
    def test_pattern_creation(self):
        """Test PIIDetectionPattern creation."""
        pattern = PIIDetectionPattern(
            name="test_pattern",
            pattern=r'\btest\b',
            pii_type="test_type",
            confidence=0.8,
            description="Test pattern",
            enabled=True
        )
        
        assert pattern.name == "test_pattern"
        assert pattern.pattern == r'\btest\b'
        assert pattern.pii_type == "test_type"
        assert pattern.confidence == 0.8
        assert pattern.description == "Test pattern"
        assert pattern.enabled is True
        assert pattern.compiled_pattern is not None
    
    def test_pattern_defaults(self):
        """Test PIIDetectionPattern with default values."""
        pattern = PIIDetectionPattern(
            name="test_pattern",
            pattern=r'\btest\b',
            pii_type="test_type"
        )
        
        assert pattern.confidence == 0.9
        assert pattern.description is None
        assert pattern.enabled is True
    
    def test_pattern_compilation_success(self):
        """Test successful regex compilation."""
        pattern = PIIDetectionPattern(
            name="email_pattern",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            pii_type="email"
        )
        
        assert pattern.compiled_pattern is not None
        assert pattern.enabled is True
    
    def test_pattern_compilation_failure(self):
        """Test regex compilation failure."""
        pattern = PIIDetectionPattern(
            name="invalid_pattern",
            pattern=r'[invalid_regex',
            pii_type="test_type"
        )
        
        assert pattern.compiled_pattern is None
        assert pattern.enabled is False
    
    def test_pattern_empty_regex(self):
        """Test pattern with empty regex."""
        pattern = PIIDetectionPattern(
            name="empty_pattern",
            pattern="",
            pii_type="test_type"
        )
        
        assert pattern.compiled_pattern is None
        assert pattern.enabled is False


class TestPIIDetectionRules:
    """Test PIIDetectionRules class."""
    
    def test_detection_rules_defaults(self):
        """Test PIIDetectionRules default values."""
        rules = PIIDetectionRules()
        
        assert rules.names_enabled is True
        assert rules.emails_enabled is True
        assert rules.phones_enabled is True
        assert rules.addresses_enabled is True
        assert rules.ssn_enabled is True
        assert rules.dob_enabled is True
        assert rules.medical_conditions_enabled is True
        assert rules.medications_enabled is True
        assert rules.treatments_enabled is True
        assert rules.medical_ids_enabled is True
        assert rules.credit_cards_enabled is True
        assert rules.bank_accounts_enabled is True
        assert rules.insurance_ids_enabled is True
        assert rules.ip_addresses_enabled is True
        assert rules.location_data_enabled is True
        assert rules.voice_transcriptions_enabled is True
        assert rules.crisis_keywords_enabled is True
        assert rules.context_aware_detection is True
        assert rules.fuzzy_matching is False
        assert rules.whitelist_words == ['test', 'example', 'demo']
        assert rules.blacklist_words == ['secret', 'confidential', 'private']
        assert len(rules.custom_patterns) > 0
    
    def test_load_default_patterns(self):
        """Test loading of default patterns."""
        rules = PIIDetectionRules()
        
        # Check that default patterns are loaded
        pattern_names = [p.name for p in rules.custom_patterns]
        assert "full_name_title" in pattern_names
        assert "full_name" in pattern_names
        assert "mental_health_conditions" in pattern_names
        assert "physical_conditions" in pattern_names
        assert "antidepressants" in pattern_names
        assert "anxiolytics" in pattern_names
        assert "therapy_types" in pattern_names
        assert "suicide_keywords" in pattern_names
        assert "harm_keywords" in pattern_names
        assert "us_address_full" in pattern_names
        assert "routing_account" in pattern_names
        assert "coordinates" in pattern_names
    
    def test_load_env_config(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            'PII_DETECT_NAMES': 'false',
            'PII_DETECT_EMAILS': 'false',
            'PII_DETECT_PHONES': 'false',
            'PII_DETECT_ADDRESSES': 'false',
            'PII_DETECT_SSN': 'false',
            'PII_DETECT_DOB': 'false',
            'PII_DETECT_MEDICAL_CONDITIONS': 'false',
            'PII_DETECT_MEDICATIONS': 'false',
            'PII_DETECT_TREATMENTS': 'false',
            'PII_DETECT_MEDICAL_IDS': 'false',
            'PII_DETECT_CREDIT_CARDS': 'false',
            'PII_DETECT_BANK_ACCOUNTS': 'false',
            'PII_DETECT_INSURANCE_IDS': 'false',
            'PII_DETECT_IP_ADDRESSES': 'false',
            'PII_DETECT_LOCATION': 'false',
            'PII_DETECT_VOICE_TRANSCRIPTIONS': 'false',
            'PII_DETECT_CRISIS_KEYWORDS': 'false',
            'PII_CONTEXT_AWARE': 'false',
            'PII_FUZZY_MATCHING': 'true',
            'PII_WHITELIST_WORDS': 'allowed,permitted',
            'PII_BLACKLIST_WORDS': 'forbidden,restricted'
        }):
            rules = PIIDetectionRules()
            
            assert rules.names_enabled is False
            assert rules.emails_enabled is False
            assert rules.phones_enabled is False
            assert rules.addresses_enabled is False
            assert rules.ssn_enabled is False
            assert rules.dob_enabled is False
            assert rules.medical_conditions_enabled is False
            assert rules.medications_enabled is False
            assert rules.treatments_enabled is False
            assert rules.medical_ids_enabled is False
            assert rules.credit_cards_enabled is False
            assert rules.bank_accounts_enabled is False
            assert rules.insurance_ids_enabled is False
            assert rules.ip_addresses_enabled is False
            assert rules.location_data_enabled is False
            assert rules.voice_transcriptions_enabled is False
            assert rules.crisis_keywords_enabled is False
            assert rules.context_aware_detection is False
            assert rules.fuzzy_matching is True
            assert rules.whitelist_words == ['allowed', 'permitted']
            assert rules.blacklist_words == ['forbidden', 'restricted']
    
    def test_load_custom_patterns_from_env(self):
        """Test loading custom patterns from environment."""
        custom_patterns = [
            {
                "name": "custom_email",
                "pattern": r'\bcustom@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "pii_type": "email",
                "confidence": 0.95,
                "description": "Custom email pattern"
            },
            {
                "name": "custom_phone",
                "pattern": r'\b\d{3}-\d{3}-\d{4}\b',
                "pii_type": "phone",
                "confidence": 0.9
            }
        ]
        
        with patch.dict(os.environ, {
            'PII_CUSTOM_PATTERNS': json.dumps(custom_patterns)
        }):
            rules = PIIDetectionRules()
            
            # Check that custom patterns are loaded
            pattern_names = [p.name for p in rules.custom_patterns]
            assert "custom_email" in pattern_names
            assert "custom_phone" in pattern_names
            
            # Check pattern details
            custom_email = next(p for p in rules.custom_patterns if p.name == "custom_email")
            assert custom_email.pii_type == "email"
            assert custom_email.confidence == 0.95
            assert custom_email.description == "Custom email pattern"
    
    def test_load_invalid_custom_patterns(self):
        """Test handling of invalid custom patterns."""
        with patch.dict(os.environ, {
            'PII_CUSTOM_PATTERNS': 'invalid_json'
        }):
            # Should not raise exception, just log warning
            rules = PIIDetectionRules()
            assert rules.custom_patterns is not None
    
    def test_get_enabled_patterns_all_enabled(self):
        """Test getting all enabled patterns when all categories are enabled."""
        rules = PIIDetectionRules()
        
        enabled_patterns = rules.get_enabled_patterns()
        
        # Should have patterns from all enabled categories
        pii_types = set(p.pii_type for p in enabled_patterns)
        assert "name" in pii_types
        assert "medical_condition" in pii_types
        assert "medication" in pii_types
        assert "treatment" in pii_types
        assert "voice_transcription" in pii_types
        
        # All patterns should be enabled
        assert all(p.enabled for p in enabled_patterns)
    
    def test_get_enabled_patterns_selective_enabled(self):
        """Test getting enabled patterns with selective categories."""
        rules = PIIDetectionRules()
        rules.names_enabled = False
        rules.medical_conditions_enabled = False
        rules.voice_transcriptions_enabled = False
        
        enabled_patterns = rules.get_enabled_patterns()
        
        # Should not have patterns from disabled categories
        pii_types = set(p.pii_type for p in enabled_patterns)
        assert "name" not in pii_types
        assert "medical_condition" not in pii_types
        assert "voice_transcription" not in pii_types
        
        # Should still have patterns from other enabled categories
        assert "medication" in pii_types
        assert "treatment" in pii_types
    
    def test_get_enabled_patterns_with_disabled_patterns(self):
        """Test getting enabled patterns with individual patterns disabled."""
        rules = PIIDetectionRules()
        
        # Disable some individual patterns
        for pattern in rules.custom_patterns:
            if pattern.name == "full_name_title":
                pattern.enabled = False
        
        enabled_patterns = rules.get_enabled_patterns()
        
        # Should not include disabled patterns
        pattern_names = [p.name for p in enabled_patterns]
        assert "full_name_title" not in pattern_names
        
        # Should include other patterns
        assert "full_name" in pattern_names
    
    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        rules = PIIDetectionRules()
        initial_count = len(rules.custom_patterns)
        
        new_pattern = PIIDetectionPattern(
            name="test_pattern",
            pattern=r'\btest\b',
            pii_type="test_type"
        )
        
        rules.add_custom_pattern(new_pattern)
        
        assert len(rules.custom_patterns) == initial_count + 1
        assert new_pattern in rules.custom_patterns
    
    def test_remove_custom_pattern(self):
        """Test removing custom pattern."""
        rules = PIIDetectionRules()
        
        # Add a pattern first
        new_pattern = PIIDetectionPattern(
            name="test_pattern",
            pattern=r'\btest\b',
            pii_type="test_type"
        )
        rules.add_custom_pattern(new_pattern)
        
        initial_count = len(rules.custom_patterns)
        
        # Remove the pattern
        rules.remove_custom_pattern("test_pattern")
        
        assert len(rules.custom_patterns) == initial_count - 1
        assert new_pattern not in rules.custom_patterns
        assert "test_pattern" not in [p.name for p in rules.custom_patterns]
    
    def test_to_dict(self):
        """Test converting rules to dictionary."""
        rules = PIIDetectionRules()
        
        rules_dict = rules.to_dict()
        
        # Check all required fields
        assert "names_enabled" in rules_dict
        assert "emails_enabled" in rules_dict
        assert "phones_enabled" in rules_dict
        assert "addresses_enabled" in rules_dict
        assert "ssn_enabled" in rules_dict
        assert "dob_enabled" in rules_dict
        assert "medical_conditions_enabled" in rules_dict
        assert "medications_enabled" in rules_dict
        assert "treatments_enabled" in rules_dict
        assert "medical_ids_enabled" in rules_dict
        assert "credit_cards_enabled" in rules_dict
        assert "bank_accounts_enabled" in rules_dict
        assert "insurance_ids_enabled" in rules_dict
        assert "ip_addresses_enabled" in rules_dict
        assert "location_data_enabled" in rules_dict
        assert "voice_transcriptions_enabled" in rules_dict
        assert "crisis_keywords_enabled" in rules_dict
        assert "context_aware_detection" in rules_dict
        assert "fuzzy_matching" in rules_dict
        assert "custom_patterns_count" in rules_dict
        assert "whitelist_words" in rules_dict
        assert "blacklist_words" in rules_dict
        
        # Check values
        assert rules_dict["names_enabled"] is True
        assert rules_dict["custom_patterns_count"] == len(rules.custom_patterns)
        assert rules_dict["whitelist_words"] == ['test', 'example', 'demo']
        assert rules_dict["blacklist_words"] == ['secret', 'confidential', 'private']


class TestPIIConfig:
    """Test PIIConfig class."""
    
    def test_config_initialization_no_file(self):
        """Test PIIConfig initialization without config file."""
        config = PIIConfig()
        
        assert config.config_file is None
        assert config.detection_rules is not None
        assert isinstance(config.detection_rules, PIIDetectionRules)
    
    def test_config_initialization_with_file(self):
        """Test PIIConfig initialization with config file."""
        config_file = "/tmp/test_pii_config.json"
        config = PIIConfig(config_file=config_file)
        
        assert config.config_file == config_file
        assert config.detection_rules is not None
    
    def test_config_initialization_with_env_file(self):
        """Test PIIConfig initialization with environment file."""
        with patch.dict(os.environ, {
            'PII_CONFIG_FILE': '/tmp/env_config.json'
        }):
            config = PIIConfig()
            
            assert config.config_file == '/tmp/env_config.json'
    
    def test_load_from_file_success(self):
        """Test successful loading from file."""
        config_data = {
            "names_enabled": False,
            "emails_enabled": False,
            "medical_conditions_enabled": False,
            "custom_patterns": [
                {
                    "name": "file_pattern",
                    "pattern": r'\bfiletest\b',
                    "pii_type": "test_type",
                    "confidence": 0.8,
                    "description": "Pattern from file"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = PIIConfig(config_file=config_file)
            config.load_from_file()
            
            assert config.detection_rules.names_enabled is False
            assert config.detection_rules.emails_enabled is False
            assert config.detection_rules.medical_conditions_enabled is False
            
            # Check custom pattern from file
            pattern_names = [p.name for p in config.detection_rules.custom_patterns]
            assert "file_pattern" in pattern_names
            
            file_pattern = next(p for p in config.detection_rules.custom_patterns if p.name == "file_pattern")
            assert file_pattern.pii_type == "test_type"
            assert file_pattern.confidence == 0.8
            assert file_pattern.description == "Pattern from file"
        finally:
            os.unlink(config_file)
    
    def test_load_from_file_not_exists(self):
        """Test loading from non-existent file."""
        config = PIIConfig(config_file="/tmp/nonexistent.json")
        
        # Should not raise exception
        config.load_from_file()
        
        # Should still have default rules
        assert config.detection_rules.names_enabled is True
    
    def test_load_from_file_invalid_json(self):
        """Test loading from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_file = f.name
        
        try:
            config = PIIConfig(config_file=config_file)
            
            # Should not raise exception
            config.load_from_file()
            
            # Should still have default rules
            assert config.detection_rules.names_enabled is True
        finally:
            os.unlink(config_file)
    
    def test_save_to_file_success(self):
        """Test successful saving to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config = PIIConfig()
            
            # Modify some rules
            config.detection_rules.names_enabled = False
            config.detection_rules.emails_enabled = False
            
            # Add custom pattern
            custom_pattern = PIIDetectionPattern(
                name="save_test",
                pattern=r'\bsavetest\b',
                pii_type="test_type",
                confidence=0.85
            )
            config.detection_rules.add_custom_pattern(custom_pattern)
            
            # Save to file
            config.save_to_file(config_file)
            
            # Verify file contents
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["names_enabled"] is False
            assert saved_data["emails_enabled"] is False
            assert "custom_patterns" in saved_data
            
            # Check custom pattern
            saved_patterns = saved_data["custom_patterns"]
            save_test_pattern = next(p for p in saved_patterns if p["name"] == "save_test")
            assert save_test_pattern["pii_type"] == "test_type"
            assert save_test_pattern["confidence"] == 0.85
        finally:
            os.unlink(config_file)
    
    def test_save_to_file_no_path(self):
        """Test saving to file without specifying path."""
        config = PIIConfig()  # No config_file specified
        
        with pytest.raises(ValueError, match="No configuration file path specified"):
            config.save_to_file()
    
    def test_save_to_file_io_error(self):
        """Test handling of IO error during save."""
        config = PIIConfig()
        
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                config.save_to_file("/tmp/test_config.json")
    
    def test_get_detection_rules(self):
        """Test getting detection rules."""
        config = PIIConfig()
        
        rules = config.get_detection_rules()
        
        assert rules is config.detection_rules
        assert isinstance(rules, PIIDetectionRules)
    
    def test_update_detection_rules_valid(self):
        """Test updating detection rules with valid updates."""
        config = PIIConfig()
        
        updates = {
            "names_enabled": False,
            "emails_enabled": False,
            "medical_conditions_enabled": False
        }
        
        config.update_detection_rules(updates)
        
        assert config.detection_rules.names_enabled is False
        assert config.detection_rules.emails_enabled is False
        assert config.detection_rules.medical_conditions_enabled is False
    
    def test_update_detection_rules_invalid(self):
        """Test updating detection rules with invalid updates."""
        config = PIIConfig()
        
        updates = {
            "invalid_field": True,
            "names_enabled": False
        }
        
        config.update_detection_rules(updates)
        
        # Valid update should be applied
        assert config.detection_rules.names_enabled is False
        
        # Invalid field should be ignored (no exception raised)
        assert not hasattr(config.detection_rules, "invalid_field")
    
    def test_health_check(self):
        """Test health check functionality."""
        config = PIIConfig()
        
        # Add some custom patterns
        custom_pattern = PIIDetectionPattern(
            name="health_test",
            pattern=r'\bhealthtest\b',
            pii_type="test_type"
        )
        config.detection_rules.add_custom_pattern(custom_pattern)
        
        health = config.health_check()
        
        assert health["status"] == "healthy"
        assert "config_file" in health
        assert "detection_rules" in health
        assert "features" in health
        
        # Check detection rules section
        detection_rules = health["detection_rules"]
        assert "enabled_categories" in detection_rules
        assert "custom_patterns" in detection_rules
        assert "enabled_patterns" in detection_rules
        assert "whitelist_words" in detection_rules
        assert "blacklist_words" in detection_rules
        
        assert detection_rules["enabled_categories"] > 0
        assert detection_rules["custom_patterns"] > 0
        assert detection_rules["enabled_patterns"] > 0
        assert detection_rules["whitelist_words"] == 3
        assert detection_rules["blacklist_words"] == 3
        
        # Check features section
        features = health["features"]
        assert "context_aware_detection" in features
        assert "fuzzy_matching" in features
        assert features["context_aware_detection"] is True
        assert features["fuzzy_matching"] is False


class TestPIIConfigIntegration:
    """Integration tests for PII Configuration."""
    
    def test_end_to_end_config_workflow(self):
        """Test end-to-end configuration workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # Create config with custom settings
            config = PIIConfig(config_file=config_file)
            
            # Update rules
            config.update_detection_rules({
                "names_enabled": False,
                "emails_enabled": False,
                "whitelist_words": ["allowed1", "allowed2"],
                "blacklist_words": ["blocked1", "blocked2"]
            })
            
            # Add custom pattern
            custom_pattern = PIIDetectionPattern(
                name="integration_test",
                pattern=r'\bintegrationtest\b',
                pii_type="test_type",
                confidence=0.95,
                description="Integration test pattern"
            )
            config.detection_rules.add_custom_pattern(custom_pattern)
            
            # Save configuration
            config.save_to_file()
            
            # Create new config instance and load
            new_config = PIIConfig(config_file=config_file)
            new_config.load_from_file()
            
            # Verify loaded configuration
            assert new_config.detection_rules.names_enabled is False
            assert new_config.detection_rules.emails_enabled is False
            assert new_config.detection_rules.whitelist_words == ["allowed1", "allowed2"]
            assert new_config.detection_rules.blacklist_words == ["blocked1", "blocked2"]
            
            # Verify custom pattern
            pattern_names = [p.name for p in new_config.detection_rules.custom_patterns]
            assert "integration_test" in pattern_names
            
            loaded_pattern = next(p for p in new_config.detection_rules.custom_patterns 
                                if p.name == "integration_test")
            assert loaded_pattern.pii_type == "test_type"
            assert loaded_pattern.confidence == 0.95
            assert loaded_pattern.description == "Integration test pattern"
            
            # Test enabled patterns filtering
            enabled_patterns = new_config.get_enabled_patterns()
            pattern_types = set(p.pii_type for p in enabled_patterns)
            
            # Names should be disabled
            assert "name" not in pattern_types
            
            # Custom pattern should be enabled
            assert "test_type" in pattern_types
            
            # Test health check
            health = new_config.health_check()
            assert health["status"] == "healthy"
            assert health["detection_rules"]["enabled_categories"] < 7  # Some categories disabled
            
        finally:
            os.unlink(config_file)
    
    def test_pattern_matching_functionality(self):
        """Test that loaded patterns actually work for matching."""
        config = PIIConfig()
        
        # Get enabled patterns
        enabled_patterns = config.get_enabled_patterns()
        
        # Find email pattern
        email_patterns = [p for p in enabled_patterns if p.pii_type == "email"]
        assert len(email_patterns) > 0
        
        email_pattern = email_patterns[0]
        assert email_pattern.compiled_pattern is not None
        
        # Test pattern matching
        test_text = "Contact me at test@example.com"
        match = email_pattern.compiled_pattern.search(test_text)
        assert match is not None
        assert match.group() == "test@example.com"
    
    def test_environment_override_workflow(self):
        """Test that environment variables properly override defaults."""
        with patch.dict(os.environ, {
            'PII_DETECT_NAMES': 'false',
            'PII_DETECT_EMAILS': 'false',
            'PII_WHITELIST_WORDS': 'env_allowed1,env_allowed2',
            'PII_BLACKLIST_WORDS': 'env_blocked1,env_blocked2',
            'PII_CONTEXT_AWARE': 'false',
            'PII_FUZZY_MATCHING': 'true'
        }):
            config = PIIConfig()
            
            assert config.detection_rules.names_enabled is False
            assert config.detection_rules.emails_enabled is False
            assert config.detection_rules.whitelist_words == ['env_allowed1', 'env_allowed2']
            assert config.detection_rules.blacklist_words == ['env_blocked1', 'env_blocked2']
            assert config.detection_rules.context_aware_detection is False
            assert config.detection_rules.fuzzy_matching is True
            
            # Verify enabled patterns reflect the changes
            enabled_patterns = config.get_enabled_patterns()
            pattern_types = set(p.pii_type for p in enabled_patterns)
            
            # Names should be disabled
            assert "name" not in pattern_types
            
            # Other patterns should still be enabled
            assert "medical_condition" in pattern_types
            assert "medication" in pattern_types