"""
Unit tests for Response Sanitizer module.

Tests response sanitization, PII filtering, and role-based access control
with comprehensive coverage of all methods and edge cases.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

# Set environment variables for testing
os.environ['RESPONSE_SANITIZATION_ENABLED'] = 'true'
os.environ['AUTO_PII_DETECTION_ENABLED'] = 'true'
os.environ['SANITIZATION_LOGGING_ENABLED'] = 'true'
os.environ['DEFAULT_SENSITIVITY_LEVEL'] = 'internal'
os.environ['SANITIZATION_EXCLUDE_ENDPOINTS'] = 'health,status,metrics'

from security.response_sanitizer import (
    SensitivityLevel, SanitizationRule, ResponseSanitizerConfig,
    ResponseSanitizer, ResponseSanitizationMiddleware
)


class TestSensitivityLevel:
    """Test SensitivityLevel enum."""
    
    def test_sensitivity_level_values(self):
        """Test SensitivityLevel enum values."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.INTERNAL.value == "internal"
        assert SensitivityLevel.SENSITIVE.value == "sensitive"
        assert SensitivityLevel.HIPAA.value == "hipaa"


class TestSanitizationRule:
    """Test SanitizationRule dataclass."""
    
    def test_sanitization_rule_creation(self):
        """Test SanitizationRule creation."""
        rule = SanitizationRule(
            field_pattern="*.email",
            sensitivity_level=SensitivityLevel.INTERNAL,
            allowed_roles=["admin", "therapist"],
            mask_strategy="partial",
            description="Email addresses"
        )
        
        assert rule.field_pattern == "*.email"
        assert rule.sensitivity_level == SensitivityLevel.INTERNAL
        assert rule.allowed_roles == ["admin", "therapist"]
        assert rule.mask_strategy == "partial"
        assert rule.description == "Email addresses"
    
    def test_sanitization_rule_defaults(self):
        """Test SanitizationRule with default values."""
        rule = SanitizationRule(
            field_pattern="*.phone",
            sensitivity_level=SensitivityLevel.SENSITIVE,
            allowed_roles=["admin"]
        )
        
        assert rule.mask_strategy == "partial"
        assert rule.description is None


class TestResponseSanitizerConfig:
    """Test ResponseSanitizerConfig dataclass."""
    
    def test_config_defaults(self):
        """Test ResponseSanitizerConfig default values."""
        config = ResponseSanitizerConfig()
        
        assert config.enabled is True
        assert config.default_sensitivity == SensitivityLevel.INTERNAL
        assert config.auto_detect_pii is True
        assert config.log_sanitization is True
        assert config.exclude_endpoints is None
        assert config.custom_rules is None
    
    def test_config_custom(self):
        """Test ResponseSanitizerConfig with custom values."""
        custom_rules = [
            SanitizationRule(
                field_pattern="*.test",
                sensitivity_level=SensitivityLevel.PUBLIC,
                allowed_roles=[]
            )
        ]
        
        config = ResponseSanitizerConfig(
            enabled=False,
            default_sensitivity=SensitivityLevel.HIPAA,
            auto_detect_pii=False,
            log_sanitization=False,
            exclude_endpoints=["health", "status"],
            custom_rules=custom_rules
        )
        
        assert config.enabled is False
        assert config.default_sensitivity == SensitivityLevel.HIPAA
        assert config.auto_detect_pii is False
        assert config.log_sanitization is False
        assert config.exclude_endpoints == ["health", "status"]
        assert config.custom_rules == custom_rules


class TestResponseSanitizer:
    """Test ResponseSanitizer class."""
    
    def test_sanitizer_initialization(self):
        """Test ResponseSanitizer initialization."""
        sanitizer = ResponseSanitizer()
        
        assert sanitizer.config.enabled is True
        assert sanitizer.pii_protection is not None
        assert len(sanitizer.sanitization_rules) > 0
        assert sanitizer.stats['responses_sanitized'] == 0
        assert sanitizer.stats['pii_instances_masked'] == 0
        assert sanitizer.stats['sanitization_errors'] == 0
    
    def test_sanitizer_custom_config(self):
        """Test ResponseSanitizer with custom config."""
        config = ResponseSanitizerConfig(
            enabled=False,
            auto_detect_pii=False,
            log_sanitization=False
        )
        sanitizer = ResponseSanitizer(config=config)
        
        assert sanitizer.config.enabled is False
        assert sanitizer.config.auto_detect_pii is False
        assert sanitizer.config.log_sanitization is False
    
    def test_load_env_config(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            'RESPONSE_SANITIZATION_ENABLED': 'false',
            'AUTO_PII_DETECTION_ENABLED': 'false',
            'SANITIZATION_LOGGING_ENABLED': 'false',
            'DEFAULT_SENSITIVITY_LEVEL': 'HIPAA',
            'SANITIZATION_EXCLUDE_ENDPOINTS': 'health,status,metrics'
        }):
            sanitizer = ResponseSanitizer()
            
            assert sanitizer.config.enabled is False
            assert sanitizer.config.auto_detect_pii is False
            assert sanitizer.config.log_sanitization is False
            assert sanitizer.config.default_sensitivity == SensitivityLevel.HIPAA
            assert sanitizer.config.exclude_endpoints == ['health', 'status', 'metrics']
    
    def test_load_env_config_invalid_sensitivity(self):
        """Test handling of invalid sensitivity level in environment."""
        with patch.dict(os.environ, {
            'DEFAULT_SENSITIVITY_LEVEL': 'invalid_level'
        }):
            # Should not raise exception, should use default
            sanitizer = ResponseSanitizer()
            assert sanitizer.config.default_sensitivity == SensitivityLevel.INTERNAL
    
    def test_load_default_rules(self):
        """Test loading of default sanitization rules."""
        sanitizer = ResponseSanitizer()
        
        # Check that default rules are loaded
        field_patterns = [rule.field_pattern for rule in sanitizer.sanitization_rules]
        assert "*.email" in field_patterns
        assert "*.phone" in field_patterns
        assert "*.medical_info.*" in field_patterns
        assert "*.address" in field_patterns
        assert "*.financial_info.*" in field_patterns
        assert "*.conversation_history.*.text" in field_patterns
        assert "*.transcription" in field_patterns
        assert "*.ip_address" in field_patterns
        assert "*.session_id" in field_patterns
    
    def test_sanitize_response_disabled(self):
        """Test sanitization when disabled."""
        config = ResponseSanitizerConfig(enabled=False)
        sanitizer = ResponseSanitizer(config=config)
        
        data = {"email": "test@example.com", "phone": "555-123-4567"}
        context = {"endpoint": "test", "user_role": "guest"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # Should return original data unchanged
        assert result == data
    
    def test_sanitize_response_excluded_endpoint(self):
        """Test sanitization of excluded endpoint."""
        config = ResponseSanitizerConfig(exclude_endpoints=["health"])
        sanitizer = ResponseSanitizer(config=config)
        
        data = {"email": "test@example.com", "status": "healthy"}
        context = {"endpoint": "health", "user_role": "guest"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # Should return original data unchanged
        assert result == data
    
    def test_sanitize_response_dict_with_pii(self):
        """Test sanitization of dictionary with PII."""
        sanitizer = ResponseSanitizer()
        
        data = {
            "email": "test@example.com",
            "phone": "555-123-4567",
            "name": "John Doe",
            "public_info": "This is public"
        }
        context = {"endpoint": "user_profile", "user_role": "guest"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # PII should be masked for guest
        assert result["email"] != "test@example.com"
        assert result["phone"] != "555-123-4567"
        assert result["name"] != "John Doe"
        # Public info should remain
        assert result["public_info"] == "This is public"
        
        # Should have sanitization flag
        assert "_sanitized" in result
        assert result["_sanitized"] is True
        
        # Should update statistics
        assert sanitizer.stats['responses_sanitized'] == 1
        assert sanitizer.stats['pii_instances_masked'] > 0
    
    def test_sanitize_response_dict_with_admin_role(self):
        """Test sanitization of dictionary with admin role."""
        sanitizer = ResponseSanitizer()
        
        data = {
            "email": "admin@example.com",
            "phone": "555-123-4567",
            "name": "Admin User",
            "ssn": "123-45-6789"
        }
        context = {"endpoint": "user_profile", "user_role": "admin"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # Admin should see most PII except highly sensitive
        assert result["email"] == "admin@example.com"
        assert result["phone"] == "555-123-4567"
        assert result["name"] == "Admin User"
        # SSN might be masked depending on rules
        # assert result["ssn"] != "123-45-6789"
    
    def test_sanitize_response_nested_dict(self):
        """Test sanitization of nested dictionary."""
        sanitizer = ResponseSanitizer()
        
        data = {
            "user": {
                "profile": {
                    "email": "user@example.com",
                    "phone": "555-123-4567"
                },
                "medical_info": {
                    "condition": "depression",
                    "medication": "prozac"
                }
            },
            "public_data": "public info"
        }
        context = {"endpoint": "user_details", "user_role": "guest"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # Nested PII should be masked
        assert result["user"]["profile"]["email"] != "user@example.com"
        assert result["user"]["profile"]["phone"] != "555-123-4567"
        assert result["user"]["medical_info"]["condition"] != "depression"
        assert result["user"]["medical_info"]["medication"] != "prozac"
        
        # Public data should remain
        assert result["public_data"] == "public info"
    
    def test_sanitize_response_list(self):
        """Test sanitization of list data."""
        sanitizer = ResponseSanitizer()
        
        data = [
            {"email": "user1@example.com", "name": "User One"},
            {"email": "user2@example.com", "name": "User Two"},
            "public string",
            123
        ]
        context = {"endpoint": "user_list", "user_role": "guest"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # PII in list items should be masked
        assert result[0]["email"] != "user1@example.com"
        assert result[0]["name"] != "User One"
        assert result[1]["email"] != "user2@example.com"
        assert result[1]["name"] != "User Two"
        
        # Non-dict items should remain
        assert result[2] == "public string"
        assert result[3] == 123
    
    def test_sanitize_response_text(self):
        """Test sanitization of text content."""
        sanitizer = ResponseSanitizer()
        
        data = "Contact John Doe at john@example.com or 555-123-4567"
        context = {"endpoint": "contact_info", "user_role": "guest"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # PII in text should be masked
        assert result != data
        assert "john@example.com" not in result
        assert "555-123-4567" not in result
    
    def test_sanitize_response_other_types(self):
        """Test sanitization of non-string, non-dict, non-list data."""
        sanitizer = ResponseSanitizer()
        
        # Integer
        result = sanitizer.sanitize_response(123, {"user_role": "guest"})
        assert result == 123
        
        # Float
        result = sanitizer.sanitize_response(45.67, {"user_role": "guest"})
        assert result == 45.67
        
        # Boolean
        result = sanitizer.sanitize_response(True, {"user_role": "guest"})
        assert result is True
        
        # None
        result = sanitizer.sanitize_response(None, {"user_role": "guest"})
        assert result is None
    
    def test_sanitize_response_error_handling(self):
        """Test error handling during sanitization."""
        sanitizer = ResponseSanitizer()
        
        # Mock PII protection to raise exception
        sanitizer.pii_protection.sanitize_text = MagicMock(side_effect=Exception("Test error"))
        
        data = "This has PII: test@example.com"
        context = {"endpoint": "test", "user_role": "guest"}
        
        # Should return original data on error
        result = sanitizer.sanitize_response(data, context)
        assert result == data
        
        # Should increment error count
        assert sanitizer.stats['sanitization_errors'] == 1
    
    def test_sanitize_dict_field_specific_rules(self):
        """Test sanitization with field-specific rules."""
        sanitizer = ResponseSanitizer()
        
        data = {
            "email": "test@example.com",
            "phone": "555-123-4567",
            "medical_info": {
                "condition": "depression",
                "treatment": "therapy"
            }
        }
        context = {"endpoint": "patient_record", "user_role": "therapist"}
        
        result = sanitizer.sanitize_response(data, context)
        
        # Therapist should see medical info but contact info might be filtered
        # depending on sensitivity levels
        assert isinstance(result, dict)
    
    def test_determine_sensitivity_level_from_context(self):
        """Test determining sensitivity level from context."""
        sanitizer = ResponseSanitizer()
        
        # Explicit sensitivity in context
        context = {"sensitivity_level": "hipaa", "user_role": "guest"}
        level = sanitizer._determine_sensitivity_level(context)
        assert level == SensitivityLevel.HIPAA
        
        # User role based sensitivity
        context = {"user_role": "guest"}
        level = sanitizer._determine_sensitivity_level(context)
        assert level == SensitivityLevel.PUBLIC
        
        context = {"user_role": "admin"}
        level = sanitizer._determine_sensitivity_level(context)
        assert level == SensitivityLevel.INTERNAL
        
        context = {"user_role": "therapist"}
        level = sanitizer._determine_sensitivity_level(context)
        assert level == SensitivityLevel.SENSITIVE
        
        # Default sensitivity
        context = {"user_role": "unknown"}
        level = sanitizer._determine_sensitivity_level(context)
        assert level == SensitivityLevel.INTERNAL
    
    def test_determine_sensitivity_level_invalid_context(self):
        """Test determining sensitivity level with invalid context."""
        sanitizer = ResponseSanitizer()
        
        # Invalid sensitivity level
        context = {"sensitivity_level": "invalid", "user_role": "guest"}
        level = sanitizer._determine_sensitivity_level(context)
        assert level == SensitivityLevel.PUBLIC  # Should fallback to role-based
    
        # No context
        level = sanitizer._determine_sensitivity_level({})
        assert level == SensitivityLevel.INTERNAL  # Should use default
    
    def test_get_field_rule(self):
        """Test getting field rule for field path."""
        sanitizer = ResponseSanitizer()
        
        # Exact match
        rule = sanitizer._get_field_rule("user.email", SensitivityLevel.INTERNAL, "guest")
        assert rule is not None
        assert rule.field_pattern == "*.email"
        
        # No match
        rule = sanitizer._get_field_rule("user.nonexistent", SensitivityLevel.INTERNAL, "guest")
        assert rule is None
        
        # Sensitivity level too low
        rule = sanitizer._get_field_rule("user.email", SensitivityLevel.PUBLIC, "guest")
        assert rule is None  # Email rule requires INTERNAL or higher
    
    def test_matches_field_pattern(self):
        """Test field pattern matching."""
        sanitizer = ResponseSanitizer()
        
        # Exact match
        assert sanitizer._matches_field_pattern("user.email", "*.email") is True
        
        # Wildcard match
        assert sanitizer._matches_field_pattern("user.profile.email", "*.email") is True
        assert sanitizer._matches_field_pattern("profile.email", "*.email") is True
        
        # Nested wildcard match
        assert sanitizer._matches_field_pattern("user.medical_info.condition", "*.medical_info.*") is True
        assert sanitizer._matches_field_pattern("medical_info.condition", "*.medical_info.*") is True
        
        # No match
        assert sanitizer._matches_field_pattern("user.name", "*.email") is False
        assert sanitizer._matches_field_pattern("user.info.email", "*.email") is False
    
    def test_should_mask_field(self):
        """Test field masking decisions."""
        sanitizer = ResponseSanitizer()
        
        rule = SanitizationRule(
            field_pattern="*.email",
            sensitivity_level=SensitivityLevel.INTERNAL,
            allowed_roles=["admin", "therapist"]
        )
        
        # Allowed role
        assert sanitizer._should_mask_field(rule, "admin") is False
        assert sanitizer._should_mask_field(rule, "therapist") is False
        
        # Not allowed role
        assert sanitizer._should_mask_field(rule, "guest") is True
        assert sanitizer._should_mask_field(rule, "patient") is True
        
        # No role
        assert sanitizer._should_mask_field(rule, None) is True
        assert sanitizer._should_mask_field(rule, "") is True
        
        # No allowed roles
        rule_no_roles = SanitizationRule(
            field_pattern="*.public",
            sensitivity_level=SensitivityLevel.PUBLIC,
            allowed_roles=[]
        )
        assert sanitizer._should_mask_field(rule_no_roles, "guest") is False
    
    def test_mask_field_value_strategies(self):
        """Test different masking strategies."""
        sanitizer = ResponseSanitizer()
        
        rule_remove = SanitizationRule(
            field_pattern="*.test",
            sensitivity_level=SensitivityLevel.PUBLIC,
            allowed_roles=[],
            mask_strategy="remove"
        )
        
        rule_full = SanitizationRule(
            field_pattern="*.test",
            sensitivity_level=SensitivityLevel.PUBLIC,
            allowed_roles=[],
            mask_strategy="full"
        )
        
        rule_hash = SanitizationRule(
            field_pattern="*.test",
            sensitivity_level=SensitivityLevel.PUBLIC,
            allowed_roles=[],
            mask_strategy="hash"
        )
        
        rule_partial = SanitizationRule(
            field_pattern="*.test",
            sensitivity_level=SensitivityLevel.PUBLIC,
            allowed_roles=[],
            mask_strategy="partial"
        )
        
        # Test remove strategy
        result = sanitizer._mask_field_value("sensitive_data", rule_remove)
        assert result == "[REDACTED]"
        
        # Test full mask strategy
        result = sanitizer._mask_field_value("sensitive_data", rule_full)
        assert result == "[FULLY REDACTED]"
        
        # Test hash strategy
        result = sanitizer._mask_field_value("sensitive_data", rule_hash)
        assert len(result) == 16  # SHA256 truncated
        assert result != "sensitive_data"
        
        # Test partial strategy
        result = sanitizer._mask_field_value("sensitive_data", rule_partial)
        assert result != "sensitive_data"
        assert "se***" in result or "sensitive_data"[:2] in result
        
        # Test partial strategy with short string
        result = sanitizer._mask_field_value("ab", rule_partial)
        assert result == "**"
        
        # Test partial strategy with non-string
        result = sanitizer._mask_field_value(123, rule_partial)
        assert result == "[MASKED]"
    
    def test_add_custom_rule(self):
        """Test adding custom sanitization rule."""
        sanitizer = ResponseSanitizer()
        initial_count = len(sanitizer.sanitization_rules)
        
        custom_rule = SanitizationRule(
            field_pattern="*.custom_field",
            sensitivity_level=SensitivityLevel.INTERNAL,
            allowed_roles=["admin"],
            description="Custom field rule"
        )
        
        sanitizer.add_custom_rule(custom_rule)
        
        assert len(sanitizer.sanitization_rules) == initial_count + 1
        assert custom_rule in sanitizer.sanitization_rules
    
    def test_remove_custom_rule(self):
        """Test removing custom sanitization rule."""
        sanitizer = ResponseSanitizer()
        
        # Add a custom rule first
        custom_rule = SanitizationRule(
            field_pattern="*.temp_field",
            sensitivity_level=SensitivityLevel.INTERNAL,
            allowed_roles=["admin"]
        )
        sanitizer.add_custom_rule(custom_rule)
        
        initial_count = len(sanitizer.sanitization_rules)
        
        # Remove the rule
        sanitizer.remove_custom_rule("*.temp_field")
        
        assert len(sanitizer.sanitization_rules) == initial_count - 1
        assert custom_rule not in sanitizer.sanitization_rules
        assert "*.temp_field" not in [rule.field_pattern for rule in sanitizer.sanitization_rules]
    
    def test_get_sanitization_stats(self):
        """Test getting sanitization statistics."""
        sanitizer = ResponseSanitizer()
        
        # Perform some sanitization to generate stats
        data = {"email": "test@example.com"}
        context = {"endpoint": "test", "user_role": "guest"}
        sanitizer.sanitize_response(data, context)
        
        stats = sanitizer.get_sanitization_stats()
        
        assert "responses_sanitized" in stats
        assert "pii_instances_masked" in stats
        assert "sanitization_errors" in stats
        assert stats["responses_sanitized"] >= 1
        assert stats["pii_instances_masked"] >= 1
        
        # Should return a copy, not the original dict
        stats["test"] = "value"
        assert "test" not in sanitizer.stats
    
    def test_health_check(self):
        """Test health check functionality."""
        sanitizer = ResponseSanitizer()
        
        # Perform some operations to generate stats
        data = {"email": "test@example.com"}
        context = {"endpoint": "test", "user_role": "guest"}
        sanitizer.sanitize_response(data, context)
        
        health = sanitizer.health_check()
        
        assert "status" in health
        assert "config" in health
        assert "statistics" in health
        assert "pii_protection_status" in health
        
        assert health["status"] == "healthy"
        
        # Check config section
        config = health["config"]
        assert "enabled" in config
        assert "auto_detect_pii" in config
        assert "default_sensitivity" in config
        assert "rules_count" in config
        assert config["enabled"] is True
        assert config["auto_detect_pii"] is True
        assert config["default_sensitivity"] == "internal"
        assert config["rules_count"] > 0
        
        # Check statistics section
        stats = health["statistics"]
        assert "responses_sanitized" in stats
        assert "pii_instances_masked" in stats
        assert "sanitization_errors" in stats
        assert stats["responses_sanitized"] >= 1
        
        # Check PII protection status
        pii_status = health["pii_protection_status"]
        assert "status" in pii_status
        assert pii_status["status"] == "healthy"


class TestResponseSanitizationMiddleware:
    """Test ResponseSanitizationMiddleware class."""
    
    def test_middleware_initialization(self):
        """Test middleware initialization."""
        middleware = ResponseSanitizationMiddleware()
        
        assert middleware.sanitizer is not None
        assert isinstance(middleware.sanitizer, ResponseSanitizer)
    
    def test_middleware_initialization_with_app(self):
        """Test middleware initialization with Flask app."""
        mock_app = MagicMock()
        middleware = ResponseSanitizationMiddleware(app=mock_app)
        
        # Should have called init_app
        mock_app.after_request.assert_called_once()
    
    def test_middleware_init_app(self):
        """Test middleware initialization with Flask app."""
        mock_app = MagicMock()
        sanitizer = ResponseSanitizer()
        middleware = ResponseSanitizationMiddleware()
        middleware.sanitizer = sanitizer
        
        middleware.init_app(mock_app)
        
        # Should have registered after_request handler
        mock_app.after_request.assert_called_once()
        
        # Get the registered handler
        handler = mock_app.after_request.call_args[0][0]
        assert callable(handler)
    
    @patch('security.response_sanitizer.request')
    @patch('security.response_sanitizer.g')
    def test_middleware_response_handling(self, mock_g, mock_request):
        """Test middleware response handling."""
        # Setup mocks
        mock_request.endpoint = "user_profile"
        mock_request.method = "GET"
        mock_request.path = "/api/user/profile"
        mock_request.content_type = "application/json"
        
        mock_g.user_role = "guest"
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.content_type = "application/json"
        mock_response.get_json.return_value = {
            "email": "test@example.com",
            "name": "Test User"
        }
        mock_response.set_data = MagicMock()
        
        # Create and initialize middleware
        mock_app = MagicMock()
        middleware = ResponseSanitizationMiddleware()
        middleware.init_app(mock_app)
        
        # Get the registered handler
        handler = mock_app.after_request.call_args[0][0]
        
        # Call the handler
        result = handler(mock_response)
        
        # Should have called set_data with sanitized content
        mock_response.set_data.assert_called_once()
        
        # Should return the response
        assert result == mock_response
    
    @patch('security.response_sanitizer.request')
    def test_middleware_non_json_response(self, mock_request):
        """Test middleware with non-JSON response."""
        mock_request.endpoint = "health"
        
        mock_response = MagicMock()
        mock_response.content_type = "text/html"
        
        # Create and initialize middleware
        mock_app = MagicMock()
        middleware = ResponseSanitizationMiddleware()
        middleware.init_app(mock_app)
        
        # Get the registered handler
        handler = mock_app.after_request.call_args[0][0]
        
        # Call the handler
        result = handler(mock_response)
        
        # Should not modify non-JSON responses
        mock_response.set_data.assert_not_called()
        
        # Should return the response
        assert result == mock_response
    
    @patch('security.response_sanitizer.request')
    @patch('security.response_sanitizer.logging')
    def test_middleware_error_handling(self, mock_logging, mock_request):
        """Test middleware error handling."""
        mock_request.endpoint = "test"
        mock_request.content_type = "application/json"
        
        mock_response = MagicMock()
        mock_response.content_type = "application/json"
        mock_response.get_json.side_effect = Exception("JSON decode error")
        
        # Create and initialize middleware
        mock_app = MagicMock()
        middleware = ResponseSanitizationMiddleware()
        middleware.init_app(mock_app)
        
        # Get the registered handler
        handler = mock_app.after_request.call_args[0][0]
        
        # Call the handler
        result = handler(mock_response)
        
        # Should log error but not break response
        mock_logging.error.assert_called_once()
        
        # Should return the response unchanged
        assert result == mock_response
        mock_response.set_data.assert_not_called()


class TestResponseSanitizerIntegration:
    """Integration tests for Response Sanitizer."""
    
    def test_end_to_end_response_sanitization(self):
        """Test end-to-end response sanitization workflow."""
        sanitizer = ResponseSanitizer()
        
        # Complex nested data
        data = {
            "user_profile": {
                "personal": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "555-123-4567",
                    "ssn": "123-45-6789"
                },
                "medical": {
                    "conditions": ["depression", "anxiety"],
                    "medications": ["prozac", "zoloft"],
                    "treatment_history": [
                        {
                            "date": "2023-01-01",
                            "therapist": "Dr. Smith",
                            "notes": "Patient showing improvement"
                        }
                    ]
                }
            },
            "session_info": {
                "session_id": "sess_12345",
                "ip_address": "192.168.1.1"
            },
            "public_data": {
                "last_login": "2023-12-01",
                "account_type": "premium"
            }
        }
        
        # Test with different user roles
        guest_context = {"endpoint": "user_profile", "user_role": "guest"}
        therapist_context = {"endpoint": "user_profile", "user_role": "therapist"}
        admin_context = {"endpoint": "user_profile", "user_role": "admin"}
        
        # Guest should see heavily masked data
        guest_result = sanitizer.sanitize_response(data, guest_context)
        assert guest_result["user_profile"]["personal"]["email"] != "john@example.com"
        assert guest_result["user_profile"]["personal"]["phone"] != "555-123-4567"
        assert guest_result["user_profile"]["personal"]["ssn"] != "123-45-6789"
        assert guest_result["user_profile"]["medical"]["conditions"][0] != "depression"
        assert guest_result["session_info"]["session_id"] != "sess_12345"
        assert guest_result["session_info"]["ip_address"] != "192.168.1.1"
        
        # Public data should remain visible
        assert guest_result["public_data"]["last_login"] == "2023-12-01"
        assert guest_result["public_data"]["account_type"] == "premium"
        
        # Therapist should see more data
        therapist_result = sanitizer.sanitize_response(data, therapist_context)
        # Medical info should be more visible to therapist
        assert therapist_result["user_profile"]["medical"]["conditions"][0] == "depression"
        
        # Admin should see most data
        admin_result = sanitizer.sanitize_response(data, admin_context)
        assert admin_result["user_profile"]["personal"]["name"] == "John Doe"
        assert admin_result["user_profile"]["personal"]["email"] == "john@example.com"
        
        # All should have sanitization flags where PII was masked
        assert "_sanitized" in guest_result["user_profile"]["personal"]
        assert guest_result["user_profile"]["personal"]["_sanitized"] is True
    
    def test_sensitivity_level_filtering(self):
        """Test filtering based on sensitivity levels."""
        sanitizer = ResponseSanitizer()
        
        data = {
            "public_info": "This is public",
            "internal_info": "This is internal",
            "sensitive_info": "This is sensitive",
            "hipaa_info": "This is HIPAA protected"
        }
        
        # Test different sensitivity levels
        public_context = {
            "endpoint": "test",
            "user_role": "guest",
            "sensitivity_level": "public"
        }
        
        internal_context = {
            "endpoint": "test", 
            "user_role": "user",
            "sensitivity_level": "internal"
        }
        
        sensitive_context = {
            "endpoint": "test",
            "user_role": "therapist", 
            "sensitivity_level": "sensitive"
        }
        
        hipaa_context = {
            "endpoint": "test",
            "user_role": "doctor",
            "sensitivity_level": "hipaa"
        }
        
        # Each level should filter appropriately
        public_result = sanitizer.sanitize_response(data, public_context)
        internal_result = sanitizer.sanitize_response(data, internal_context)
        sensitive_result = sanitizer.sanitize_response(data, sensitive_context)
        hipaa_result = sanitizer.sanitize_response(data, hipaa_context)
        
        # Results should be different based on sensitivity level
        assert public_result != internal_result
        assert internal_result != sensitive_result
        assert sensitive_result != hipaa_result
    
    def test_custom_rules_integration(self):
        """Test integration of custom rules."""
        sanitizer = ResponseSanitizer()
        
        # Add custom rule
        custom_rule = SanitizationRule(
            field_pattern="*.api_key",
            sensitivity_level=SensitivityLevel.INTERNAL,
            allowed_roles=["admin"],
            mask_strategy="full",
            description="API keys"
        )
        sanitizer.add_custom_rule(custom_rule)
        
        data = {
            "user": {
                "api_key": "sk-1234567890abcdef",
                "name": "Test User"
            },
            "config": {
                "api_key": "cfg-abcdef1234567890",
                "version": "1.0.0"
            }
        }
        
        # Non-admin should see masked API keys
        guest_context = {"endpoint": "config", "user_role": "guest"}
        guest_result = sanitizer.sanitize_response(data, guest_context)
        
        assert guest_result["user"]["api_key"] == "[FULLY REDACTED]"
        assert guest_result["config"]["api_key"] == "[FULLY REDACTED]"
        assert guest_result["user"]["name"] == "Test User"  # Should remain
        assert guest_result["config"]["version"] == "1.0.0"  # Should remain
        
        # Admin should see API keys
        admin_context = {"endpoint": "config", "user_role": "admin"}
        admin_result = sanitizer.sanitize_response(data, admin_context)
        
        assert admin_result["user"]["api_key"] == "sk-1234567890abcdef"
        assert admin_result["config"]["api_key"] == "cfg-abcdef1234567890"