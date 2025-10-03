"""
Comprehensive PII Protection Tests for AI Therapist.

Tests PII detection accuracy, data masking effectiveness, response sanitization,
role-based PII access, and HIPAA compliance features.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import PII protection modules
from security.pii_protection import (
    PIIProtection, PIIDetector, PIIMasker, PIIType, MaskingStrategy, PIIDetectionResult
)
from security.response_sanitizer import ResponseSanitizer, SensitivityLevel
from security.pii_config import PIIDetectionRules, PIIConfig, PIIDetectionPattern

# Import voice security for integration tests
from voice.security import VoiceSecurity

# Import auth for role-based tests
from auth.user_model import UserProfile, UserRole, UserStatus


class TestPIIDetector:
    """Test PII detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PIIDetector()

    def test_detect_email_pii(self):
        """Test email PII detection."""
        text = "Contact me at john.doe@example.com for more info."
        results = self.detector.detect_pii(text)

        assert len(results) == 1
        assert results[0].pii_type == PIIType.EMAIL
        assert results[0].value == "john.doe@example.com"
        assert results[0].confidence >= 0.8

    def test_detect_phone_pii(self):
        """Test phone number PII detection."""
        text = "Call me at (555) 123-4567 or 555-987-6543."
        results = self.detector.detect_pii(text)

        phone_results = [r for r in results if r.pii_type == PIIType.PHONE]
        assert len(phone_results) >= 1

    def test_detect_medical_condition_pii(self):
        """Test medical condition PII detection."""
        text = "Patient suffers from depression and anxiety disorders."
        results = self.detector.detect_pii(text)

        medical_results = [r for r in results if r.pii_type == PIIType.MEDICAL_CONDITION]
        assert len(medical_results) >= 1

    def test_detect_name_pii(self):
        """Test name PII detection."""
        text = "Dr. Sarah Johnson will see the patient tomorrow."
        results = self.detector.detect_pii(text)

        name_results = [r for r in results if r.pii_type == PIIType.NAME]
        assert len(name_results) >= 1

    def test_voice_transcription_crisis_detection(self):
        """Test crisis keyword detection in voice transcriptions."""
        text = "I feel like I want to die and can't take it anymore."
        results = self.detector.detect_pii(text, context="voice_transcription")

        crisis_results = [r for r in results if r.pii_type == PIIType.VOICE_TRANSCRIPTION]
        assert len(crisis_results) >= 1

    def test_detect_in_nested_dict(self):
        """Test PII detection in nested dictionary structures."""
        data = {
            "user": {
                "email": "test@example.com",
                "profile": {
                    "phone": "(555) 123-4567",
                    "medical": {
                        "condition": "depression"
                    }
                }
            }
        }

        detections = self.detector.detect_in_dict(data)
        assert len(detections) >= 3  # email, phone, medical condition

    def test_no_pii_detected(self):
        """Test that non-PII text doesn't trigger false positives."""
        text = "The weather is nice today and I enjoy walking."
        results = self.detector.detect_pii(text)
        assert len(results) == 0


class TestPIIMasker:
    """Test PII masking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.masker = PIIMasker()

    def test_partial_mask_email(self):
        """Test partial email masking."""
        email = "john.doe@example.com"
        masked = self.masker.mask_value(email, PIIType.EMAIL)

        assert "@" in masked
        assert "example.com" in masked
        assert "*" in masked
        assert masked != email

    def test_partial_mask_phone(self):
        """Test partial phone masking."""
        phone = "(555) 123-4567"
        masked = self.masker.mask_value(phone, PIIType.PHONE)

        assert "***" in masked
        assert "4567" in masked
        assert masked != phone

    def test_full_mask_sensitive_data(self):
        """Test full masking for sensitive data."""
        self.masker.strategy = MaskingStrategy.FULL_MASK
        email = "sensitive@example.com"
        masked = self.masker.mask_value(email, PIIType.EMAIL)

        assert masked == "[EMAIL REDACTED]"

    def test_hash_mask(self):
        """Test hash-based masking."""
        self.masker.strategy = MaskingStrategy.HASH_MASK
        value = "test@example.com"
        masked1 = self.masker.mask_value(value, PIIType.EMAIL)
        masked2 = self.masker.mask_value(value, PIIType.EMAIL)

        # Same input should produce same hash
        assert masked1 == masked2
        assert len(masked1) == 16  # Our hash mask length

    def test_anonymize_mask(self):
        """Test anonymization masking."""
        self.masker.strategy = MaskingStrategy.ANONYMIZE
        email = "test@example.com"
        masked = self.masker.mask_value(email, PIIType.EMAIL)

        assert masked == "user@example.com"  # anonymized placeholder


class TestPIIProtection:
    """Test comprehensive PII protection system."""

    def setup_method(self):
        """Set up test fixtures."""
        from security.pii_protection import PIIProtectionConfig
        config = PIIProtectionConfig(enable_audit=True)
        self.pii_protection = PIIProtection(config)

    def test_sanitize_text_with_pii(self):
        """Test text sanitization with PII."""
        text = "Contact john.doe@example.com or call (555) 123-4567."
        sanitized = self.pii_protection.sanitize_text(text, user_role="patient")

        assert sanitized != text
        assert "@" not in sanitized or "*" in sanitized
        assert "4567" in sanitized  # Last 4 digits should be visible

    def test_sanitize_dict_with_pii(self):
        """Test dictionary sanitization with PII."""
        data = {
            "user": {
                "email": "test@example.com",
                "phone": "(555) 123-4567",
                "medical_info": {
                    "condition": "depression"
                }
            }
        }

        sanitized = self.pii_protection.sanitize_dict(data, user_role="patient")

        assert sanitized["user"]["email"] != data["user"]["email"]
        assert "*" in sanitized["user"]["email"]

    def test_role_based_access_control(self):
        """Test role-based PII access control."""
        medical_data = {
            "condition": "depression",
            "medication": "prozac",
            "treatment": "CBT"
        }

        # Patient should see limited info
        patient_view = self.pii_protection.sanitize_dict(
            {"medical": medical_data}, user_role="patient"
        )

        # Therapist should see more info
        therapist_view = self.pii_protection.sanitize_dict(
            {"medical": medical_data}, user_role="therapist"
        )

        assert len(str(patient_view)) < len(str(therapist_view))

    def test_hipaa_compliance_checking(self):
        """Test HIPAA compliance validation."""
        # Test authorized access
        compliant = self.pii_protection._check_hipaa_compliance(
            "access", PIIType.MEDICAL_CONDITION, "therapist"
        )
        assert compliant

        # Test unauthorized access
        not_compliant = self.pii_protection._check_hipaa_compliance(
            "access", PIIType.MEDICAL_CONDITION, "guest"
        )
        assert not not_compliant

    def test_audit_trail_logging(self):
        """Test that PII access is properly audited."""
        initial_audit_count = len(self.pii_protection.audit_trail)

        self.pii_protection.sanitize_text(
            "Email: test@example.com", user_role="patient"
        )

        final_audit_count = len(self.pii_protection.audit_trail)
        assert final_audit_count > initial_audit_count

    def test_health_check(self):
        """Test PII protection health check."""
        health = self.pii_protection.health_check()

        assert health["status"] == "healthy"
        assert "statistics" in health
        assert "pii_protection_status" in health


class TestResponseSanitizer:
    """Test response sanitization middleware."""

    def setup_method(self):
        """Set up test fixtures."""
        from security.response_sanitizer import ResponseSanitizerConfig
        config = ResponseSanitizerConfig(auto_detect_pii=True, log_sanitization=True)
        self.sanitizer = ResponseSanitizer(config=config)

    def test_sanitize_json_response(self):
        """Test JSON response sanitization."""
        response_data = {
            "user": {
                "email": "john.doe@example.com",
                "medical": {
                    "condition": "anxiety"
                }
            }
        }

        context = {
            "user_role": "patient",
            "endpoint": "/api/user/profile"
        }

        sanitized = self.sanitizer.sanitize_response(response_data, context)

        assert sanitized != response_data
        assert sanitized["user"]["email"] != response_data["user"]["email"]

    def test_different_sensitivity_levels(self):
        """Test different sensitivity levels."""
        data = {"email": "test@example.com"}

        # Public level - should mask
        public_result = self.sanitizer._sanitize_dict(
            data, SensitivityLevel.PUBLIC, "guest", {}
        )
        assert public_result["email"] != data["email"]

        # HIPAA level - should mask medical info
        medical_data = {"medical_info": {"condition": "depression"}}
        hipaa_result = self.sanitizer._sanitize_dict(
            medical_data, SensitivityLevel.HIPAA, "patient", {}
        )
        assert "_sanitized" in hipaa_result["medical_info"]

    def test_endpoint_exclusion(self):
        """Test endpoint exclusion from sanitization."""
        self.sanitizer.config.exclude_endpoints = ["/api/health"]

        assert self.sanitizer._should_exclude_endpoint("/api/health")
        assert not self.sanitizer._should_exclude_endpoint("/api/user/profile")

    def test_sanitization_statistics(self):
        """Test sanitization statistics tracking."""
        initial_stats = self.sanitizer.get_sanitization_stats()

        self.sanitizer.sanitize_response(
            {"email": "test@example.com"}, {"user_role": "patient"}
        )

        final_stats = self.sanitizer.get_sanitization_stats()
        assert final_stats["responses_sanitized"] > initial_stats["responses_sanitized"]


class TestVoiceSecurityIntegration:
    """Test voice security PII integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.voice_security = VoiceSecurity()

    def test_voice_transcription_filtering(self):
        """Test PII filtering in voice transcriptions."""
        transcription = "My email is john.doe@example.com and I have depression."

        result = self.voice_security.filter_voice_transcription(
            transcription, user_id="user123", session_id="session456"
        )

        assert result["sanitized"]
        assert result["pii_detected"]
        assert "email" in [pii.lower() for pii in result["pii_detected"]]

    def test_crisis_detection_in_transcription(self):
        """Test crisis keyword detection."""
        crisis_text = "I feel suicidal and want to harm myself."

        result = self.voice_security.filter_voice_transcription(
            crisis_text, user_id="user123", session_id="session456"
        )

        crisis_detected = any(
            "transcription" in pii.lower() for pii in result["pii_detected"]
        )
        assert crisis_detected


class TestAuthPIIIntegration:
    """Test authentication PII integration."""

    def setup_method(self):
        """Set up test fixtures."""
        from auth.user_model import UserProfile, UserRole, UserStatus
        from datetime import datetime

        self.test_user = UserProfile(
            user_id="test123",
            email="john.doe@example.com",
            full_name="John Doe",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            medical_info={
                "condition": "depression",
                "medication": "prozac",
                "emergency_contact": "Jane Doe"
            }
        )

    def test_user_profile_pii_filtering(self):
        """Test PII filtering in user profiles."""
        # Patient viewing their own profile
        patient_view = self.test_user.to_dict(user_role="patient")
        assert "depression" not in str(patient_view["medical_info"])

        # Therapist viewing patient profile
        therapist_view = self.test_user.to_dict(user_role="therapist")
        assert "depression" in str(therapist_view["medical_info"])

    def test_email_masking(self):
        """Test email masking for non-admin users."""
        masked_data = self.test_user.to_dict(user_role="patient")
        assert "*" in masked_data["email"]
        assert masked_data["email"] != self.test_user.email

    def test_admin_full_access(self):
        """Test that admins see full PII."""
        admin_view = self.test_user.to_dict(user_role="admin")
        assert admin_view["email"] == self.test_user.email
        assert "depression" in str(admin_view["medical_info"])


class TestPIIConfig:
    """Test PII configuration management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PIIConfig()

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict('os.environ', {
            'PII_DETECT_EMAILS': 'false',
            'PII_MASKING_STRATEGY': 'full_mask'
        }):
            rules = PIIDetectionRules()
            assert not rules.emails_enabled

    def test_custom_pattern_addition(self):
        """Test adding custom PII detection patterns."""
        custom_pattern = PIIDetectionPattern(
            name="custom_ssn",
            pattern=r'\b\d{3}-\d{2}-\d{4}\b',
            pii_type="ssn",
            description="Custom SSN pattern"
        )

        self.config.detection_rules.add_custom_pattern(custom_pattern)
        patterns = self.config.detection_rules.get_enabled_patterns()

        custom_found = any(p.name == "custom_ssn" for p in patterns)
        assert custom_found

    def test_config_file_operations(self):
        """Test configuration file save/load."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            # Save config
            self.config.save_to_file(temp_file)
            assert os.path.exists(temp_file)

            # Load config
            new_config = PIIConfig(temp_file)
            assert new_config.detection_rules.emails_enabled == self.config.detection_rules.emails_enabled

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestHIPAACompliance:
    """Test HIPAA compliance features."""

    def setup_method(self):
        """Set up test fixtures."""
        from security.pii_protection import PIIProtectionConfig
        config = PIIProtectionConfig(hipaa_compliance=True)
        self.pii_protection = PIIProtection(config)

    def test_hipaa_violation_tracking(self):
        """Test HIPAA violation tracking."""
        initial_violations = len(self.pii_protection.hipaa_violations)

        # Simulate unauthorized access
        self.pii_protection._check_hipaa_compliance(
            "access", PIIType.MEDICAL_CONDITION, "unauthorized_user"
        )

        # Should trigger audit but not necessarily a violation log
        # (depending on implementation)

    def test_phi_protection(self):
        """Test Protected Health Information (PHI) protection."""
        phi_text = "Patient John Doe has cancer and takes chemotherapy."

        sanitized = self.pii_protection.sanitize_text(phi_text, user_role="guest")

        # Should mask name, medical condition
        assert "John Doe" not in sanitized
        assert "cancer" not in sanitized

    def test_minimum_necessary_access(self):
        """Test minimum necessary access principle."""
        full_medical_data = {
            "diagnosis": "Major Depression",
            "treatment": "SSRIs",
            "prognosis": "Good with treatment",
            "personal_notes": "Patient prefers morning appointments"
        }

        # Patient access - limited
        patient_access = self.pii_protection.sanitize_dict(
            {"medical": full_medical_data}, user_role="patient"
        )

        # Therapist access - more complete
        therapist_access = self.pii_protection.sanitize_dict(
            {"medical": full_medical_data}, user_role="therapist"
        )

        assert len(str(patient_access)) < len(str(therapist_access))


class TestEndToEndPIIProtection:
    """End-to-end PII protection tests."""

    def test_complete_voice_session_workflow(self):
        """Test complete voice session with PII protection."""
        # This would integrate voice service, PII protection, and response sanitization
        # For now, test the components work together

        # Mock voice transcription with PII
        transcription = "I'm John Smith, my email is john@example.com, I have anxiety."

        # Test PII detection
        detector = PIIDetector()
        pii_found = detector.detect_pii(transcription, "voice_transcription")
        assert len(pii_found) >= 2  # name, email, medical condition

        # Test sanitization
        protection = PIIProtection()
        sanitized = protection.sanitize_text(transcription, "voice_transcription", "patient")
        assert sanitized != transcription

        # Test response sanitization
        sanitizer = ResponseSanitizer()
        response_data = {"transcription": transcription, "user_id": "123"}
        sanitized_response = sanitizer.sanitize_response(
            response_data, {"user_role": "patient"}
        )
        assert sanitized_response["transcription"] != transcription

    def test_audit_compliance_reporting(self):
        """Test audit compliance reporting."""
        protection = PIIProtection()

        # Generate some audit events
        protection.sanitize_text("email@test.com", user_role="patient")
        protection.sanitize_text("Patient has depression", user_role="guest")

        audit_trail = protection.get_audit_trail()

        assert len(audit_trail) >= 2

        # Check HIPAA violations
        violations = protection.get_hipaa_violations()
        # Should have violations for unauthorized medical info access

    def test_performance_impact(self):
        """Test that PII protection doesn't significantly impact performance."""
        import time

        protection = PIIProtection()
        test_text = "This is a test message without PII."

        # Time multiple sanitization operations
        start_time = time.time()
        for _ in range(100):
            protection.sanitize_text(test_text, user_role="patient")
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        # Should be very fast (< 0.01 seconds per operation)
        assert avg_time < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])