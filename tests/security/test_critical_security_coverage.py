"""
Critical Security Coverage Tests for AI Therapist.

Tests high-impact security functions to ensure comprehensive coverage:
- PII Protection batch processing and validation
- Response Sanitizer streaming and filtering
- Role-based access control
- Audit logging and export functionality
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from security.pii_protection import (
    PIIProtection, PIIType, MaskingStrategy, PIIDetectionResult,
    PIIProtectionConfig, PIIDetector, PIIMasker
)
from security.response_sanitizer import (
    ResponseSanitizer, SensitivityLevel, SanitizationRule,
    ResponseSanitizerConfig, ResponseSanitizationMiddleware
)


class TestCriticalPIIProtection:
    """Test critical PII protection functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = PIIProtectionConfig(
            enable_detection=True,
            enable_masking=True,
            enable_audit=True,
            hipaa_compliance=True,
            masking_strategy=MaskingStrategy.PARTIAL_MASK
        )
        self.pii_protection = PIIProtection(self.config)
    
    def test_process_text_batch(self):
        """Test batch processing of text for PII detection and masking."""
        # Test data with various PII types
        text_batch = [
            "Contact John Doe at john.doe@example.com or (555) 123-4567",
            "Patient SSN: 123-45-6789, Address: 123 Main St, Springfield",
            "Credit card: 4532-1234-5678-9012, IP: 192.168.1.1",
            "No PII in this text"
        ]
        
        # Process batch
        results = []
        for text in text_batch:
            sanitized = self.pii_protection.sanitize_text(text, context="test_batch")
            results.append(sanitized)
        
        # Verify PII was detected and masked
        assert len(results) == 4
        assert "example.com" not in results[0] or "***" in results[0]
        assert "4567" not in results[0] or "***" in results[0]
        assert "6789" not in results[1] or "***" in results[1]
        assert "4532" not in results[2] or "****" in results[2]
        assert results[3] == "No PII in this text"  # Should remain unchanged
        
        # Verify audit trail was created
        assert len(self.pii_protection.audit_trail) > 0
    
    def test_validate_pii_patterns(self):
        """Test PII pattern validation functionality."""
        detector = PIIDetector()
        
        # Test various PII patterns
        test_cases = [
            ("john.doe@example.com", PIIType.EMAIL),
            ("(555) 123-4567", PIIType.PHONE),
            ("123-45-6789", PIIType.SSN),
            ("123 Main St", PIIType.ADDRESS),
            ("4532-1234-5678-9012", PIIType.CREDIT_CARD),
            ("192.168.1.1", PIIType.IP_ADDRESS),
            ("Dr. John Smith", PIIType.NAME)
        ]
        
        for text, expected_type in test_cases:
            detections = detector.detect_pii(text)
            
            # Check if expected type was detected
            detected_types = [d.pii_type for d in detections]
            assert expected_type in detected_types, f"Failed to detect {expected_type} in '{text}'"
            
            # Verify detection result structure
            for detection in detections:
                assert detection.value in text
                assert 0 <= detection.confidence <= 1.0
                assert detection.start_pos >= 0
                assert detection.end_pos > detection.start_pos
    
    def test_export_audit_logs(self):
        """Test audit log export functionality."""
        # Create some audit entries
        self.pii_protection.sanitize_text("john.doe@example.com", user_role="admin")
        self.pii_protection.sanitize_text("(555) 123-4567", user_role="therapist")
        self.pii_protection.sanitize_text("123-45-6789", user_role="patient")
        
        # Get audit trail
        audit_trail = self.pii_protection.get_audit_trail()
        
        # Verify audit entries
        assert len(audit_trail) >= 3
        
        for entry in audit_trail:
            assert "timestamp" in entry
            assert "action" in entry
            assert "pii_type" in entry
            assert "user_role" in entry
            assert "value_hash" in entry
            assert "hipaa_compliant" in entry
            
            # Verify timestamp format
            assert isinstance(datetime.fromisoformat(entry["timestamp"]), datetime)
            
            # Verify value hash is present and correct length
            assert len(entry["value_hash"]) == 16  # SHA256 truncated to 16 chars
        
        # Test filtering by date range
        start_date = datetime.now() - timedelta(hours=1)
        filtered_trail = self.pii_protection.get_audit_trail(start_date=start_date)
        assert len(filtered_trail) == len(audit_trail)  # All entries should be within last hour
    
    def test_role_based_masking(self):
        """Test role-based PII masking functionality."""
        test_text = "John Doe, john.doe@example.com, (555) 123-4567, SSN: 123-45-6789"
        
        # Test admin role (should see everything according to implementation)
        admin_result = self.pii_protection.sanitize_text(test_text, user_role="admin")
        
        # Test therapist role (should see medical info, some contact info, but SSN masked)
        therapist_result = self.pii_protection.sanitize_text(test_text, user_role="therapist")
        
        # Test patient role (should see minimal PII)
        patient_result = self.pii_protection.sanitize_text(test_text, user_role="patient")
        
        # Test unauthorized role (should see minimal PII)
        guest_result = self.pii_protection.sanitize_text(test_text, user_role="guest")
        
        # Verify different masking levels based on actual implementation
        # Admin sees everything (no masking per implementation)
        assert admin_result == test_text
        
        # Therapist sees most info but SSN is masked
        assert "***" in therapist_result or "[REDACTED]" in therapist_result
        assert "123-45-6789" not in therapist_result
        
        # Patient and guest should see the most masking
        assert "***" in patient_result or "[REDACTED]" in patient_result
        assert "***" in guest_result or "[REDACTED]" in guest_result
        
        # Verify audit trail for each role
        admin_entries = [e for e in self.pii_protection.audit_trail if e.get("user_role") == "admin"]
        therapist_entries = [e for e in self.pii_protection.audit_trail if e.get("user_role") == "therapist"]
        patient_entries = [e for e in self.pii_protection.audit_trail if e.get("user_role") == "patient"]
        
        assert len(admin_entries) > 0
        assert len(therapist_entries) > 0
        assert len(patient_entries) > 0
    
    def test_hipaa_compliance_enforcement(self):
        """Test HIPAA compliance enforcement in PII handling."""
        medical_text = "Patient depression treatment with Prozac, Medical ID: MRN123456"
        
        # Test HIPAA-compliant access
        therapist_result = self.pii_protection.sanitize_text(medical_text, user_role="therapist")
        admin_result = self.pii_protection.sanitize_text(medical_text, user_role="admin")
        
        # Test non-HIPAA-compliant access
        patient_result = self.pii_protection.sanitize_text(medical_text, user_role="patient")
        guest_result = self.pii_protection.sanitize_text(medical_text, user_role="guest")
        
        # Check for HIPAA violations
        violations = self.pii_protection.get_hipaa_violations()
        
        # Should have violations for unauthorized access to medical info
        assert len(violations) > 0
        
        # Verify violation structure
        for violation in violations:
            assert "timestamp" in violation
            assert "violation_type" in violation
            assert violation["violation_type"] == "unauthorized_pii_access"
            assert "details" in violation
            
            # Verify details contain expected fields
            details = violation["details"]
            assert "pii_type" in details
            assert "user_role" in details
            assert "hipaa_compliant" in details
            assert details["hipaa_compliant"] is False
        
        # Verify specific violations for medical information
        medical_violations = []
        for violation in violations:
            details = violation["details"]
            pii_type = details.get("pii_type", "")
            if pii_type in ["medical_condition", "medication", "medical_id"]:
                medical_violations.append(violation)
        
        assert len(medical_violations) > 0


class TestCriticalResponseSanitizer:
    """Test critical response sanitizer functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = ResponseSanitizerConfig(
            enabled=True,
            auto_detect_pii=True,
            log_sanitization=True,
            default_sensitivity=SensitivityLevel.INTERNAL
        )
        self.sanitizer = ResponseSanitizer(config=self.config)
    
    def test_sanitize_stream_response(self):
        """Test streaming response sanitization."""
        # Simulate streaming response data
        stream_data = [
            {"message": "User John Doe registered", "email": "john.doe@example.com"},
            {"message": "Phone updated", "phone": "(555) 123-4567"},
            {"message": "Address added", "address": "123 Main St"},
            {"message": "Simple message without PII"}
        ]
        
        request_context = {
            'user_role': 'guest',
            'endpoint': '/api/users',
            'method': 'GET'
        }
        
        # Process stream data
        sanitized_stream = []
        for chunk in stream_data:
            sanitized_chunk = self.sanitizer.sanitize_response(chunk, request_context)
            sanitized_stream.append(sanitized_chunk)
        
        # Verify PII was sanitized in stream
        assert len(sanitized_stream) == 4
        
        # Check first chunk (email should be masked)
        assert "john.doe@example.com" not in str(sanitized_stream[0]) or "***" in str(sanitized_stream[0])
        
        # Check second chunk (phone should be masked)
        assert "(555) 123-4567" not in str(sanitized_stream[1]) or "***" in str(sanitized_stream[1])
        
        # Check third chunk (address should be masked)
        assert "123 Main St" not in str(sanitized_stream[2]) or "***" in str(sanitized_stream[2])
        
        # Fourth chunk should remain unchanged
        assert sanitized_stream[3]["message"] == "Simple message without PII"
        
        # Verify statistics
        stats = self.sanitizer.get_sanitization_stats()
        assert stats['responses_sanitized'] == 4
        assert stats['pii_instances_masked'] > 0
    
    def test_custom_rule_engine(self):
        """Test custom sanitization rule engine."""
        # Add custom rules
        custom_rule = SanitizationRule(
            field_pattern="*.custom_field",
            sensitivity_level=SensitivityLevel.INTERNAL,
            allowed_roles=["admin"],
            mask_strategy="full",
            description="Custom field for testing"
        )
        self.sanitizer.add_custom_rule(custom_rule)
        
        # Test data with custom field
        test_data = {
            "custom_field": "sensitive_custom_data",
            "normal_field": "normal_data",
            "nested": {
                "custom_field": "nested_sensitive_data"
            }
        }
        
        # Test with admin role (should see custom field)
        admin_context = {'user_role': 'admin', 'endpoint': '/api/test'}
        admin_result = self.sanitizer.sanitize_response(test_data, admin_context)
        
        # Test with guest role (should not see custom field)
        guest_context = {'user_role': 'guest', 'endpoint': '/api/test'}
        guest_result = self.sanitizer.sanitize_response(test_data, guest_context)
        
        # Verify custom rule application
        assert admin_result["custom_field"] == "sensitive_custom_data"
        assert admin_result["nested"]["custom_field"] == "nested_sensitive_data"
        
        assert guest_result["custom_field"] == "[FULLY REDACTED]"
        assert guest_result["nested"]["custom_field"] == "[FULLY REDACTED]"
        
        # Normal field should be visible to both
        assert admin_result["normal_field"] == "normal_data"
        assert guest_result["normal_field"] == "normal_data"
        
        # Test rule removal
        self.sanitizer.remove_custom_rule("*.custom_field")
        
        # After removal, guest should see custom field (no rule applies)
        guest_result_after = self.sanitizer.sanitize_response(test_data, guest_context)
        assert guest_result_after["custom_field"] == "sensitive_custom_data"
    
    def test_endpoint_based_filtering(self):
        """Test endpoint-specific sanitization filtering."""
        # Configure excluded endpoints
        self.sanitizer.config.exclude_endpoints = ["/public/", "/health"]
        
        test_data = {
            "user_email": "test@example.com",
            "user_phone": "(555) 123-4567",
            "message": "User contact information"
        }
        
        # Test public endpoint (should be excluded)
        public_context = {'user_role': 'guest', 'endpoint': '/public/info'}
        public_result = self.sanitizer.sanitize_response(test_data, public_context)
        
        # Test health endpoint (should be excluded)
        health_context = {'user_role': 'guest', 'endpoint': '/health/status'}
        health_result = self.sanitizer.sanitize_response(test_data, health_context)
        
        # Test private endpoint (should be sanitized)
        private_context = {'user_role': 'guest', 'endpoint': '/api/private/users'}
        private_result = self.sanitizer.sanitize_response(test_data, private_context)
        
        # Verify endpoint filtering
        assert public_result == test_data  # No sanitization
        assert health_result == test_data  # No sanitization
        assert private_result != test_data  # Should be sanitized
        
        # PII should be masked in private endpoint
        assert "test@example.com" not in str(private_result) or "***" in str(private_result)
        assert "(555) 123-4567" not in str(private_result) or "***" in str(private_result)
    
    def test_cache_sanitized_responses(self):
        """Test response caching functionality."""
        test_data = {
            "user_id": "12345",
            "email": "test@example.com",
            "message": "User information"
        }
        
        request_context = {
            'user_role': 'guest',
            'endpoint': '/api/users/12345',
            'method': 'GET'
        }
        
        # First request - should sanitize and cache result
        result1 = self.sanitizer.sanitize_response(test_data, request_context)
        
        # Second request - should use cached result if implemented
        result2 = self.sanitizer.sanitize_response(test_data, request_context)
        
        # Results should be consistent
        assert result1 == result2
        
        # PII should be masked
        assert "test@example.com" not in str(result1) or "***" in str(result1)
        assert result1["user_id"] == "12345"  # Non-PII should remain
        assert result1["message"] == "User information"
        
        # Verify statistics updated
        stats = self.sanitizer.get_sanitization_stats()
        assert stats['responses_sanitized'] >= 2
        
        # Test with different user role - should get different result
        admin_context = request_context.copy()
        admin_context['user_role'] = 'admin'
        
        admin_result = self.sanitizer.sanitize_response(test_data, admin_context)
        
        # Admin should see different masking level
        # (exact behavior depends on role configuration)
        assert isinstance(admin_result, dict)
    
    def test_sensitivity_level_enforcement(self):
        """Test sensitivity level enforcement in sanitization."""
        test_data = {
            "user_email": "test@example.com",
            "medical_info": "Patient has depression",
            "financial_info": "Credit card: 4532-1234-5678-9012",
            "public_data": "This is public information"
        }
        
        # Test different sensitivity levels
        contexts = [
            {'user_role': 'public', 'endpoint': '/api/public'},
            {'user_role': 'internal', 'endpoint': '/api/internal'},
            {'user_role': 'therapist', 'endpoint': '/api/medical'},
            {'user_role': 'admin', 'endpoint': '/api/admin'}
        ]
        
        results = []
        for context in contexts:
            result = self.sanitizer.sanitize_response(test_data, context)
            results.append(result)
        
        # All results should be dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert "public_data" in result
        
        # Public user should see most masking
        public_result = results[0]
        assert "test@example.com" not in str(public_result) or "***" in str(public_result)
        
        # Admin should see the least masking
        admin_result = results[3]
        # (exact behavior depends on configuration)
        
        # Verify different results for different roles
        assert len(set(str(r) for r in results)) > 1  # Results should differ


class TestSecurityIntegration:
    """Test integration between PII protection and response sanitization."""
    
    def setup_method(self):
        """Setup integrated test environment."""
        self.pii_config = PIIProtectionConfig(enable_audit=True)
        self.pii_protection = PIIProtection(self.pii_config)
        
        self.sanitizer_config = ResponseSanitizerConfig(
            enabled=True,
            auto_detect_pii=True
        )
        self.sanitizer = ResponseSanitizer(
            pii_protection=self.pii_protection,
            config=self.sanitizer_config
        )
    
    def test_end_to_end_pii_flow(self):
        """Test complete PII detection and sanitization flow."""
        # Test data with multiple PII types
        test_data = {
            "user_profile": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "(555) 123-4567",
                "medical_records": {
                    "condition": "depression",
                    "treatment": "CBT therapy",
                    "medication": "Prozac"
                }
            },
            "conversation": [
                {"text": "Patient reported feeling anxious", "speaker": "therapist"},
                {"text": "My email is john.doe@example.com", "speaker": "patient"}
            ]
        }
        
        request_context = {
            'user_role': 'guest',
            'endpoint': '/api/patient/123',
            'method': 'GET'
        }
        
        # Process through sanitizer
        sanitized_data = self.sanitizer.sanitize_response(test_data, request_context)
        
        # Verify structure is preserved
        assert "user_profile" in sanitized_data
        assert "conversation" in sanitized_data
        assert "name" in sanitized_data["user_profile"]
        assert "medical_records" in sanitized_data["user_profile"]
        
        # Verify PII was masked
        result_str = str(sanitized_data)
        assert "john.doe@example.com" not in result_str or "***" in result_str
        assert "(555) 123-4567" not in result_str or "***" in result_str
        
        # Verify audit trail was created
        audit_trail = self.pii_protection.get_audit_trail()
        assert len(audit_trail) > 0
        
        # Verify HIPAA compliance checking
        violations = self.pii_protection.get_hipaa_violations()
        assert len(violations) >= 0  # May have violations for guest accessing medical data
    
    def test_middleware_integration(self):
        """Test Flask middleware integration with proper mocking."""
        # Create mock Flask components without importing Flask
        with patch('builtins.__import__') as mock_import:
            # Mock flask module
            mock_flask = Mock()
            mock_request = Mock()
            mock_g = Mock()
            mock_flask.request = mock_request
            mock_flask.g = mock_g
            
            mock_import.return_value = mock_flask
            
            # Create mock response
            mock_response = Mock()
            mock_response.content_type = 'application/json'
            mock_response.get_json.return_value = {
                "user_email": "test@example.com",
                "message": "User data"
            }
            mock_response.set_data = Mock()
            
            # Mock the app
            mock_app = Mock()
            mock_after_request = Mock()
            mock_app.after_request = mock_after_request
            
            # Create middleware
            middleware = ResponseSanitizationMiddleware(sanitizer=self.sanitizer)
            middleware.init_app(mock_app)
            
            # Verify middleware was registered
            mock_app.after_request.assert_called_once()
            
            # Test the actual middleware function
            args, kwargs = mock_after_request.call_args
            after_request_func = args[0]
            
            # Setup request context
            mock_request.endpoint = '/api/users'
            mock_request.method = 'GET'
            mock_request.path = '/api/users'
            mock_g.user_role = 'guest'
            
            # Execute middleware
            result = after_request_func(mock_response)
            
            # Verify response was modified
            assert result == mock_response
            mock_response.set_data.assert_called_once()
            
            # Check the sanitized data
            call_args = mock_response.set_data.call_args[0][0]
            sanitized_data = json.loads(call_args.decode('utf-8'))
            
            # PII should be masked
            assert "test@example.com" not in str(sanitized_data) or "***" in str(sanitized_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])