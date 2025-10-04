"""
Response Sanitizer Middleware for AI Therapist.

Provides automatic PII detection and redaction in API responses,
configurable sensitivity levels, and role-based data exposure controls.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class SensitivityLevel(Enum):
    """Sensitivity levels for response sanitization."""
    PUBLIC = "public"  # No PII filtering
    INTERNAL = "internal"  # Basic PII filtering
    SENSITIVE = "sensitive"  # Strict PII filtering
    HIPAA = "hipaa"  # HIPAA-compliant filtering


@dataclass
class SanitizationRule:
    """Rule for sanitizing specific fields or data patterns."""
    field_pattern: str
    sensitivity_level: SensitivityLevel
    allowed_roles: List[str]
    mask_strategy: str = "partial"
    description: Optional[str] = None


@dataclass
class ResponseSanitizerConfig:
    """Configuration for response sanitization."""
    enabled: bool = True
    default_sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    auto_detect_pii: bool = True
    log_sanitization: bool = True
    exclude_endpoints: List[str] = None  # Endpoints to exclude from sanitization
    custom_rules: List[SanitizationRule] = None


class ResponseSanitizer:
    """Middleware for sanitizing API responses to prevent PII exposure."""

    def __init__(self, pii_protection=None, config: Optional[ResponseSanitizerConfig] = None):
        """Initialize response sanitizer."""
        self.logger = logging.getLogger(__name__)
        self.config = config or ResponseSanitizerConfig()

        # Import PII protection
        if pii_protection:
            self.pii_protection = pii_protection
        else:
            from .pii_protection import PIIProtection
            self.pii_protection = PIIProtection()

        # Load configuration from environment
        self._load_env_config()

        # Default sanitization rules
        self.sanitization_rules = self._load_default_rules()

        # Custom rules
        if self.config.custom_rules:
            self.sanitization_rules.extend(self.config.custom_rules)

        # Statistics
        self.stats = {
            'responses_sanitized': 0,
            'pii_instances_masked': 0,
            'sanitization_errors': 0
        }

    def _load_env_config(self):
        """Load configuration from environment variables."""
        self.config.enabled = os.getenv("RESPONSE_SANITIZATION_ENABLED", "true").lower() == "true"
        self.config.auto_detect_pii = os.getenv("AUTO_PII_DETECTION_ENABLED", "true").lower() == "true"
        self.config.log_sanitization = os.getenv("SANITIZATION_LOGGING_ENABLED", "true").lower() == "true"

        # Default sensitivity level
        sensitivity_env = os.getenv("DEFAULT_SENSITIVITY_LEVEL", "internal").upper()
        try:
            self.config.default_sensitivity = SensitivityLevel[sensitivity_env]
        except KeyError:
            self.logger.warning(f"Invalid sensitivity level: {sensitivity_env}, using default")

        # Exclude endpoints
        exclude_env = os.getenv("SANITIZATION_EXCLUDE_ENDPOINTS", "")
        if exclude_env:
            self.config.exclude_endpoints = [e.strip() for e in exclude_env.split(",")]

    def _load_default_rules(self) -> List[SanitizationRule]:
        """Load default sanitization rules."""
        return [
            # User profile fields
            SanitizationRule(
                field_pattern="*.email",
                sensitivity_level=SensitivityLevel.INTERNAL,
                allowed_roles=["admin", "therapist"],
                description="Email addresses in user profiles"
            ),
            SanitizationRule(
                field_pattern="*.phone",
                sensitivity_level=SensitivityLevel.SENSITIVE,
                allowed_roles=["admin", "therapist"],
                description="Phone numbers"
            ),
            SanitizationRule(
                field_pattern="*.medical_info.*",
                sensitivity_level=SensitivityLevel.HIPAA,
                allowed_roles=["therapist", "admin"],
                description="Medical information (HIPAA protected)"
            ),
            SanitizationRule(
                field_pattern="*.address",
                sensitivity_level=SensitivityLevel.SENSITIVE,
                allowed_roles=["admin"],
                description="Physical addresses"
            ),
            SanitizationRule(
                field_pattern="*.financial_info.*",
                sensitivity_level=SensitivityLevel.HIPAA,
                allowed_roles=["admin"],
                description="Financial information"
            ),
            # Voice session data
            SanitizationRule(
                field_pattern="*.conversation_history.*.text",
                sensitivity_level=SensitivityLevel.SENSITIVE,
                allowed_roles=["therapist", "admin"],
                description="Voice conversation content"
            ),
            SanitizationRule(
                field_pattern="*.transcription",
                sensitivity_level=SensitivityLevel.SENSITIVE,
                allowed_roles=["therapist", "admin"],
                description="Audio transcriptions"
            ),
            # Audit and log data
            SanitizationRule(
                field_pattern="*.ip_address",
                sensitivity_level=SensitivityLevel.INTERNAL,
                allowed_roles=["admin"],
                description="IP addresses in logs"
            ),
            SanitizationRule(
                field_pattern="*.session_id",
                sensitivity_level=SensitivityLevel.INTERNAL,
                allowed_roles=["admin", "therapist"],
                description="Session identifiers"
            )
        ]

    def sanitize_response(self, response_data: Any, request_context: Dict[str, Any]) -> Any:
        """
        Sanitize API response data.

        Args:
            response_data: The response data to sanitize
            request_context: Context including user_role, endpoint, method, etc.

        Returns:
            Sanitized response data
        """
        if not self.config.enabled:
            return response_data

        try:
            # Check if endpoint should be excluded
            endpoint = request_context.get('endpoint', '')
            if self._should_exclude_endpoint(endpoint):
                return response_data

            # Get sensitivity level
            sensitivity = self._determine_sensitivity_level(request_context)

            # Get user role
            user_role = request_context.get('user_role')

            # Sanitize based on data type
            if isinstance(response_data, dict):
                sanitized = self._sanitize_dict(response_data, sensitivity, user_role, request_context)
            elif isinstance(response_data, list):
                sanitized = self._sanitize_list(response_data, sensitivity, user_role, request_context)
            elif isinstance(response_data, str):
                sanitized = self._sanitize_text(response_data, sensitivity, user_role, request_context)
            else:
                # Other data types (int, float, bool) - return as-is
                return response_data

            # Update statistics
            self.stats['responses_sanitized'] += 1

            # Log sanitization if enabled
            if self.config.log_sanitization and sanitized != response_data:
                self.logger.info(f"Response sanitized for endpoint {endpoint}, user_role: {user_role}")

            return sanitized

        except Exception as e:
            self.logger.error(f"Error sanitizing response: {e}")
            self.stats['sanitization_errors'] += 1
            # Return original data on error to avoid breaking responses
            return response_data

    def _sanitize_dict(self, data: Dict[str, Any], sensitivity: SensitivityLevel,
                      user_role: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary response data."""
        sanitized = {}
        sanitized_any = False

        for key, value in data.items():
            # Check field-specific rules
            field_rule = self._get_field_rule(f"*.{key}", sensitivity, user_role)
            original_value = value
            was_sanitized = False
            
            if field_rule:
                if self._should_mask_field(field_rule, user_role):
                    sanitized[key] = self._mask_field_value(value, field_rule)
                    self.stats['pii_instances_masked'] += 1
                    sanitized_any = True
                    was_sanitized = True
                    continue

            # Recursively sanitize nested data
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value, sensitivity, user_role, context)
                # Check if nested dict was sanitized
                if "_sanitized" in sanitized[key]:
                    sanitized_any = True
                    was_sanitized = True
            elif isinstance(value, list):
                sanitized[key] = self._sanitize_list(value, sensitivity, user_role, context)
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_text(value, sensitivity, user_role, context)
                # Check if text was sanitized
                if sanitized[key] != original_value:
                    sanitized_any = True
                    was_sanitized = True
            else:
                sanitized[key] = value

        # Add sanitization flag if any PII was masked
        if sanitized_any:
            sanitized["_sanitized"] = True

        return sanitized

    def _sanitize_list(self, data: List[Any], sensitivity: SensitivityLevel,
                      user_role: Optional[str], context: Dict[str, Any]) -> List[Any]:
        """Sanitize list response data."""
        sanitized = []

        for item in data:
            if isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item, sensitivity, user_role, context))
            elif isinstance(item, list):
                sanitized.append(self._sanitize_list(item, sensitivity, user_role, context))
            elif isinstance(item, str):
                sanitized.append(self._sanitize_text(item, sensitivity, user_role, context))
            else:
                sanitized.append(item)

        return sanitized

    def _sanitize_text(self, text: str, sensitivity: SensitivityLevel,
                      user_role: Optional[str], context: Dict[str, Any]) -> str:
        """Sanitize text content."""
        if not self.config.auto_detect_pii:
            return text

        # Use PII protection for text sanitization
        context_info = context.get('endpoint', '')
        return self.pii_protection.sanitize_text(text, context_info, user_role)

    def _should_exclude_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint should be excluded from sanitization."""
        if not self.config.exclude_endpoints:
            return False

        return any(excl in endpoint for excl in self.config.exclude_endpoints)

    def _determine_sensitivity_level(self, request_context: Dict[str, Any]) -> SensitivityLevel:
        """Determine sensitivity level for the request."""
        # Check explicit sensitivity in context
        if 'sensitivity_level' in request_context:
            try:
                return SensitivityLevel[request_context['sensitivity_level'].upper()]
            except KeyError:
                pass

        # Check user role for sensitivity
        user_role = request_context.get('user_role')
        if user_role in ['guest', 'public']:
            return SensitivityLevel.PUBLIC
        elif user_role == 'admin':
            return SensitivityLevel.INTERNAL
        elif user_role == 'therapist':
            return SensitivityLevel.SENSITIVE
        else:
            # Default sensitivity
            return self.config.default_sensitivity

    def _get_field_rule(self, field_path: str, sensitivity: SensitivityLevel,
                       user_role: Optional[str]) -> Optional[SanitizationRule]:
        """Get sanitization rule for a field path."""
        for rule in self.sanitization_rules:
            if self._matches_field_pattern(field_path, rule.field_pattern):
                # Check if sensitivity level applies
                if rule.sensitivity_level.value <= sensitivity.value:
                    return rule
        return None

    def _matches_field_pattern(self, field_path: str, pattern: str) -> bool:
        """Check if field path matches pattern (simple wildcard matching)."""
        # Convert pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        import re
        return bool(re.match(f"^{regex_pattern}$", field_path))

    def _should_mask_field(self, rule: SanitizationRule, user_role: Optional[str]) -> bool:
        """Determine if field should be masked based on rule and user role."""
        if not rule.allowed_roles:
            return False

        if not user_role:
            return True  # Mask if no user role

        return user_role.lower() not in [role.lower() for role in rule.allowed_roles]

    def _mask_field_value(self, value: Any, rule: SanitizationRule) -> Any:
        """Mask field value according to rule."""
        if rule.mask_strategy == "remove":
            return "[REDACTED]"
        elif rule.mask_strategy == "full":
            return "[FULLY REDACTED]"
        elif rule.mask_strategy == "hash":
            import hashlib
            return hashlib.sha256(str(value).encode()).hexdigest()[:16]
        else:  # partial (default)
            if isinstance(value, str):
                if len(value) <= 4:
                    return "*" * len(value)
                return f"{value[:2]}***{value[-2:]}" if len(value) > 4 else f"{value[0]}***{value[-1]}"
            else:
                return "[MASKED]"

    def add_custom_rule(self, rule: SanitizationRule):
        """Add custom sanitization rule."""
        self.sanitization_rules.append(rule)
        self.logger.info(f"Added custom sanitization rule: {rule.field_pattern}")

    def remove_custom_rule(self, field_pattern: str):
        """Remove custom sanitization rule."""
        self.sanitization_rules = [
            rule for rule in self.sanitization_rules
            if rule.field_pattern != field_pattern
        ]
        self.logger.info(f"Removed custom sanitization rule: {field_pattern}")

    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        return self.stats.copy()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on response sanitizer."""
        return {
            "status": "healthy" if self.config.enabled else "disabled",
            "config": {
                "enabled": self.config.enabled,
                "auto_detect_pii": self.config.auto_detect_pii,
                "default_sensitivity": self.config.default_sensitivity.value,
                "rules_count": len(self.sanitization_rules)
            },
            "statistics": self.get_sanitization_stats(),
            "pii_protection_status": self.pii_protection.health_check()
        }


# Flask middleware integration
class ResponseSanitizationMiddleware:
    """Flask middleware for automatic response sanitization."""

    def __init__(self, app=None, sanitizer=None):
        """Initialize middleware."""
        self.sanitizer = sanitizer or ResponseSanitizer()
        if app:
            self.init_app(app)

    def init_app(self, app):
        """Initialize with Flask app."""
        @app.after_request
        def sanitize_response(response):
            try:
                # Only sanitize JSON responses
                if response.content_type and 'application/json' in response.content_type:
                    data = response.get_json()

                    # Get request context
                    from flask import request, g
                    context = {
                        'endpoint': request.endpoint,
                        'method': request.method,
                        'user_role': getattr(g, 'user_role', None) if hasattr(g, 'user_role') else None,
                        'path': request.path
                    }

                    # Sanitize data
                    sanitized_data = self.sanitizer.sanitize_response(data, context)

                    # Update response
                    response.set_data(json.dumps(sanitized_data).encode('utf-8'))

            except Exception as e:
                # Log error but don't break response
                logging.error(f"Response sanitization middleware error: {e}")

            return response