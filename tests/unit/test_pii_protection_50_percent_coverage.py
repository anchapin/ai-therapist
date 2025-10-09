"""
Final unit tests to complete 50% coverage target for security/pii_protection.py.
Targets remaining uncovered functions and edge cases.
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
    from security.pii_protection import PIIType, PIIDetectionResult, MaskingStrategy
    from security.pii_protection import PIIProtectionConfig, PIIAuditLog, PIIRiskLevel
    from security.pii_protection import PIISanitizer, PIIAnonymizer
    from security.pii_protection import PIIComplianceChecker
    from security.pii_protection import PIIAccessController
    from security.pii_protection import PIIDataClassifier
    from security.pii_protection import PIIRetentionPolicy
    from security.pii_protection import PIIReportGenerator
    from security.pii_protection import PIIEncryptionManager
except ImportError as e:
    pytest.skip(f"pii_protection module not available: {e}", allow_module_level=True)


class TestPIIProtection50PercentCoverage:
    """Final unit tests to reach 50% coverage for pii_protection.py."""
    
    @pytest.fixture
    def pii_detector(self):
        """Create a PII detector instance."""
        return PIIDetector()
    
    def test_mask_pii_with_strategies_complete(self, pii_detector):
        """Test PII masking with all available strategies."""
        test_cases = [
            ("123-45-6789", MaskingStrategy.FULL_MASK, "XXX-XX-XXXX"),
            ("john.doe@example.com", MaskingStrategy.PARTIAL_MASK, "j***@example.com"),
            ("555-123-4567", MaskingStrategy.REMOVE, ""),
            ("John Doe", MaskingStrategy.ANONYMIZE, "[REDACTED_NAME]"),
            ("sensitive_data", MaskingStrategy.HASH_MASK, "hash_value"),
        ]
        
        for original, strategy, expected in test_cases:
            result = pii_detector.mask_pii(original, strategy)
            assert result is not None
            if strategy != MaskingStrategy.REMOVE:
                assert result != original
            if strategy == MaskingStrategy.HASH_MASK:
                # Hash result should be consistent
                result2 = pii_detector.mask_pii(original, strategy)
                assert result == result2
    
    def test_anonymize_json_data_complete(self, pii_detector):
        """Test comprehensive JSON data anonymization."""
        json_data = {
            "patient_profile": {
                "name": "John Doe",
                "contact": {
                    "email": "john.doe@example.com",
                    "phone": "555-123-4567"
                },
                "medical": {
                    "ssn": "123-45-6789",
                    "conditions": ["depression", "anxiety"],
                    "medications": ["sertraline", "prozac"]
                },
                "insurance": {
                    "provider": "HealthCo",
                    "policy_number": "POL-12345"
                },
                "notes": "Patient reports feeling better with therapy",
                "metadata": {
                    "created_at": "2024-01-15",
                    "updated_at": "2024-01-20"
                }
            },
            "session_info": {
                "therapist": "Dr. Sarah Smith",
                "date": "2024-01-15",
                "duration": 60
            }
        }
        
        anonymized = pii_detector.anonymize_json_data(json_data)
        
        # PII should be masked
        patient = anonymized["patient_profile"]
        assert "John Doe" not in str(patient)
        assert "john.doe@example.com" not in str(patient)
        assert "555-123-4567" not in str(patient)
        assert "123-45-6789" not in str(patient)
        
        # Non-PII should be preserved
        session = anonymized["session_info"]
        assert session["date"] == "2024-01-15"
        assert session["duration"] == 60
        
        # Structure should be preserved
        assert "patient_profile" in anonymized
        assert "session_info" in anonymized
        assert isinstance(anonymized["patient_profile"], dict)
    
    def test_create_audit_log_entry_complete(self, pii_detector):
        """Test comprehensive audit log entry creation."""
        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": "user_123",
            "action": "ACCESS_PATIENT_DATA",
            "resource_id": "patient_456",
            "pii_types": ["SSN", "MEDICAL_RECORD"],
            "risk_level": "HIGH",
            "compliant": True,
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "session_id": "sess_789"
        }
        
        log_entry = pii_detector.create_audit_log_entry(audit_data)
        
        assert isinstance(log_entry, PIIAuditLog)
        assert log_entry.user_id == "user_123"
        assert log_entry.action == "ACCESS_PATIENT_DATA"
        assert log_entry.resource_id == "patient_456"
        assert log_entry.risk_level == "HIGH"
        assert log_entry.compliant is True
        assert log_entry.timestamp is not None
        assert "hash" in log_entry.__dict__ or "value_hash" in log_entry.__dict__
    
    def test_calculate_risk_level_complete(self, pii_detector):
        """Test comprehensive risk level calculation."""
        test_cases = [
            ([], PIIRiskLevel.NONE),
            ([PIIType.EMAIL], PIIRiskLevel.LOW),
            ([PIIType.PHONE, PIIType.ADDRESS], PIIRiskLevel.MEDIUM),
            ([PIIType.SSN, PIIType.CREDIT_CARD], PIIRiskLevel.HIGH),
            ([PIIType.SSN, PIIType.MEDICAL_RECORD, PIIType.CREDIT_CARD], PIIRiskLevel.CRITICAL),
            ([PIIType.EMAIL, PIIType.SSN], PIIRiskLevel.HIGH),  # Mixed levels
            ([PIIType.NAME, PIIType.EMAIL], PIIRiskLevel.MEDIUM),  # Lower risk combination
            ([PIIType.VOICE_TRANSCRIPTION], PIIRiskLevel.HIGH),  # Voice content is high risk
            ([PIIType.MEDICAL_CONDITION, PIIType.TREATMENT], PIIRiskLevel.MEDIUM),  # Medical risk
        ]
        
        for pii_types, expected_risk in test_cases:
            calculated_risk = pii_detector.calculate_risk_level(pii_types)
            assert calculated_risk == expected_risk
    
    def test_sanitize_response_complete(self, pii_detector):
        """Test comprehensive response sanitization."""
        test_responses = [
            "Contact John Doe at john.doe@example.com or call 555-123-4567",
            "Patient SSN: 123-45-6789, born 01/15/1985",
            "Credit card ending in 1111: 4111111111111111",
            "Dr. Sarah Smith will call tomorrow at 10 AM",
            "Regular business communication with no PII",
            "Emergency: patient reports suicidal thoughts",
            "Medical record: John Doe has depression, takes sertraline"
        ]
        
        for response in test_responses:
            sanitized = pii_detector.sanitize_response(response)
            
            # Should not contain original sensitive information
            assert "john.doe@example.com" not in sanitized
            assert "555-123-4567" not in sanitized
            assert "123-45-6789" not in sanitized
            assert "4111111111111111" not in sanitized
            
            # Should preserve basic communication
            assert len(sanitized) > 0
            assert "Contact" in sanitized or "Regular" in sanitized or "Emergency" in sanitized
    
    def test_batch_process_texts_complete(self, pii_detector):
        """Test comprehensive batch text processing."""
        texts = [
            "john.doe@example.com",
            "555-123-4567", 
            "123-45-6789",
            "Regular text",
            "Dr. Sarah Smith",
            "Patient has depression",
            "Call 1-800-SUICIDE",
            "Credit card: 4111111111111111"
        ]
        
        results = pii_detector.batch_process_texts(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) >= 0 for r in results)
        
        # Count total PII detected
        total_pii = sum(len(r) for r in results)
        assert total_pii >= 8  # Should detect at least 8 PII instances
        
        # Check specific text results
        email_results = results[0]
        assert len(email_results) >= 1
        assert email_results[0].pii_type == PIIType.EMAIL
        
        phone_results = results[1]
        assert len(phone_results) >= 1
        assert phone_results[0].pii_type == PIIType.PHONE
    
    def test_validate_compliance_complete(self, pii_detector):
        """Test comprehensive compliance validation."""
        compliance_cases = [
            # Case 1: No PII - should be compliant
            ({
                "data": "Regular business communication",
                "pii_count": 0,
                "max_allowed": 5,
                "user_role": "admin",
                "required_masking": "full"
            }, True),
            
            # Case 2: PII over limit - should be non-compliant
            ({
                "data": "john.doe@example.com and 555-123-4567",
                "pii_count": 2,
                "max_allowed": 1,
                "user_role": "patient",
                "required_masking": "full"
            }, False),
            
            # Case 3: High-risk PII with insufficient masking
            ({
                "data": "SSN: 123-45-6789",
                "pii_count": 1,
                "max_allowed": 2,
                "user_role": "patient",
                "required_masking": "full",
                "masking_applied": "partial"  # Insufficient masking
            }, False),
            
            # Case 4: Admin access - should be compliant with any PII
            ({
                "data": "SSN: 123-45-6789 and Credit: 4111111111111111",
                "pii_count": 2,
                "max_allowed": 0,
                "user_role": "admin",
                "required_masking": "none"
            }, True),
            
            # Case 5: HIPAA required but not met
            ({
                "data": "Medical record transmitted unencrypted",
                "pii_count": 1,
                "max_allowed": 1,
                "user_role": "therapist",
                "required_masking": "full",
                "encryption": "none",
                "hipaa_required": True
            }, False)
        ]
        
        for case_data, expected in compliance_cases:
            result = pii_detector.validate_compliance(case_data)
            assert result == expected
    
    def test_extract_pii_statistics_complete(self, pii_detector):
        """Test comprehensive PII statistics extraction."""
        text = """
        Multiple PII instances:
        john.doe@example.com (Email - Low Risk)
        555-123-4567 (Phone - Medium Risk)
        123-45-6789 (SSN - High Risk)
        depression (Medical Condition - High Risk)
        Dr. Sarah Smith (Name - Medium Risk)
        4111111111111111 (Credit Card - High Risk)
        Call 1-800-SUICIDE (Voice - High Risk)
        Additional regular text without PII
        """
        
        stats = pii_detector.extract_pii_statistics(text)
        
        assert isinstance(stats, dict)
        assert "total_pii_instances" in stats
        assert "pii_by_type" in stats
        assert "risk_levels" in stats
        assert "risk_score" in stats
        assert "compliance_status" in stats
        
        assert stats["total_pii_instances"] >= 7  # At least 7 PII instances
        
        # Check PII by type
        pii_by_type = stats["pii_by_type"]
        assert PIIType.EMAIL in pii_by_type
        assert PIIType.PHONE in pii_by_type
        assert PIIType.SSN in pii_by_type
        assert PIIType.MEDICAL_CONDITION in pii_by_type
        
        # Check risk levels
        risk_levels = stats["risk_levels"]
        assert "low" in risk_levels
        assert "medium" in risk_levels
        assert "high" in risk_levels
    
    def test_role_based_pii_masking(self, pii_detector):
        """Test role-based PII masking logic."""
        test_cases = [
            # Admin should see everything unmasked
            {
                "pii": "john.doe@example.com",
                "user_role": "admin",
                "pii_type": PIIType.EMAIL,
                "should_mask": False
            },
            
            # Therapist should see medical info but mask SSN
            {
                "pii": "depression",
                "user_role": "therapist", 
                "pii_type": PIIType.MEDICAL_CONDITION,
                "should_mask": False
            },
            {
                "pii": "123-45-6789",
                "user_role": "therapist",
                "pii_type": PIIType.SSN,
                "should_mask": True
            },
            
            # Patient should have everything masked
            {
                "pii": "john.doe@example.com",
                "user_role": "patient",
                "pii_type": PIIType.EMAIL,
                "should_mask": True
            },
            {
                "pii": "depression",
                "user_role": "patient",
                "pii_type": PIIType.MEDICAL_CONDITION,
                "should_mask": True
            }
        ]
        
        for case in test_cases:
            should_mask = pii_detector._should_mask_pii_for_role(
                case["pii_type"], case["user_role"]
            )
            assert should_mask == case["should_mask"]
    
    def test_hipaa_compliance_checking(self, pii_detector):
        """Test HIPAA compliance validation."""
        compliance_cases = [
            # Compliant cases
            {
                "action": "access",
                "pii_type": PIIType.MEDICAL_RECORD,
                "user_role": "doctor",
                "encryption": True,
                "audit": True
            },
            {
                "action": "modify", 
                "pii_type": PIIType.SSN,
                "user_role": "admin",
                "encryption": True,
                "audit": True
            },
            
            # Non-compliant cases
            {
                "action": "modify",
                "pii_type": PIIType.MEDICAL_RECORD,
                "user_role": "patient",
                "encryption": False,
                "audit": False
            },
            {
                "action": "delete",
                "pii_type": PIIType.SSN,
                "user_role": "therapist",
                "encryption": False,
                "audit": False
            }
        ]
        
        for case in compliance_cases:
            compliant = pii_detector._check_hipaa_compliance(
                case["action"], case["pii_type"], case["user_role"]
            )
            
            # Admin actions should be compliant regardless
            if case["user_role"] == "admin":
                assert compliant is True
            # Patient modifications should be non-compliant
            elif case["user_role"] == "patient" and case["action"] in ["modify", "delete"]:
                assert compliant is False
    
    def test_pii_data_classification(self, pii_detector):
        """Test PII data classification logic."""
        classification_cases = [
            # Low risk
            {
                "data": "john.doe@example.com",
                "context": "business_contact",
                "expected_classification": "low"
            },
            
            # Medium risk
            {
                "data": "555-123-4567",
                "context": "appointment_reminder",
                "expected_classification": "medium"
            },
            
            # High risk
            {
                "data": "123-45-6789",
                "context": "medical_record",
                "expected_classification": "high"
            },
            
            # Critical risk
            {
                "data": "I want to die",
                "context": "voice_crisis",
                "expected_classification": "critical"
            }
        ]
        
        for case in classification_cases:
            classification = pii_detector._classify_pii_data(
                case["data"], case["context"]
            )
            assert classification == case["expected_classification"]
    
    def test_pii_retention_policy(self, pii_detector):
        """Test PII data retention policy enforcement."""
        retention_cases = [
            {
                "pii_type": PIIType.EMAIL,
                "user_role": "patient",
                "created_days_ago": 365,
                "should_retain": True
            },
            {
                "pii_type": PIIType.SSN,
                "user_role": "patient",
                "created_days_ago": 2555,  # 7 years
                "should_retain": False  # Exceeds 7-year HIPAA limit
            },
            {
                "pii_type": PIIType.MEDICAL_RECORD,
                "user_role": "therapist",
                "created_days_ago": 3650,  # 10 years
                "should_retain": False  # Exceeds retention period
            },
            {
                "pii_type": PIIType.EMAIL,
                "user_role": "admin",
                "created_days_ago": 2555,
                "should_retain": True  # Admin records kept longer
            }
        ]
        
        for case in retention_cases:
            should_retain = pii_detector._check_retention_policy(
                case["pii_type"], case["user_role"], case["created_days_ago"]
            )
            assert should_retain == case["should_retain"]
    
    def test_generate_pii_report(self, pii_detector):
        """Test comprehensive PII reporting."""
        report_data = {
            "timeframe": "last_30_days",
            "user_role": "therapist",
            "include_details": True
        }
        
        report = pii_detector.generate_pii_report(report_data)
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "details" in report
        assert "recommendations" in report
        
        # Check summary
        summary = report["summary"]
        assert "total_pii_detected" in summary
        assert "risk_level_distribution" in summary
        assert "compliance_status" in summary
        
        # Check details
        details = report["details"]
        if report_data["include_details"]:
            assert "pii_instances" in details
            assert "audit_trail" in details
    
    def test_pii_encryption_decryption(self, pii_detector):
        """Test PII encryption and decryption."""
        sensitive_data = [
            "john.doe@example.com",
            "123-45-6789",
            "SSN: 987-65-4321",
            "Credit Card: 555555555555555"
        ]
        
        for data in sensitive_data:
            # Encrypt
            encrypted = pii_detector.encrypt_pii(data)
            assert encrypted is not None
            assert encrypted != data
            assert len(encrypted) >= len(data)
            
            # Decrypt
            decrypted = pii_detector.decrypt_pii(encrypted)
            assert decrypted == data
            
            # Encryption should be deterministic
            encrypted2 = pii_detector.encrypt_pii(data)
            assert encrypted == encrypted2
    
    def test_pii_data_export_import(self, pii_detector):
        """Test PII data export and import with proper handling."""
        export_data = {
            "patient_records": [
                {
                    "id": 1,
                    "name": "John Doe",
                    "email": "john@example.com",
                    "ssn": "123-45-6789",
                    "medical_history": "depression treatment"
                },
                {
                    "id": 2,
                    "name": "Jane Smith",
                    "email": "jane@example.com",
                    "ssn": "987-65-4321",
                    "medical_history": "anxiety treatment"
                }
            ],
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "exported_by": "admin",
                "purpose": "backup",
                "encryption": "AES-256"
            }
        }
        
        # Export
        exported = pii_detector.export_pii_data(export_data, format="json")
        assert exported is not None
        assert isinstance(exported, str)
        assert len(exported) > 0
        
        # Import should handle encrypted data
        imported = pii_detector.import_pii_data(exported, format="json")
        assert imported is not None
        assert "patient_records" in imported
        assert len(imported["patient_records"]) == 2
    
    def test_pii_data_breach_detection(self, pii_detector):
        """Test PII data breach detection."""
        breach_scenarios = [
            # Normal access - no breach
            {
                "access_count": 10,
                "time_window": "1_hour",
                "user_roles": ["therapist"],
                "data_types": [PIIType.MEDICAL_RECORD],
                "is_breach": False
            },
            
            # Excessive access - potential breach
            {
                "access_count": 1000,
                "time_window": "1_hour",
                "user_roles": ["patient"],
                "data_types": [PIIType.SSN],
                "is_breach": True
            },
            
            # Unauthorized role - breach
            {
                "access_count": 5,
                "time_window": "1_hour",
                "user_roles": ["unauthorized"],
                "data_types": [PIIType.SSN],
                "is_breach": True
            },
            
            # Off-hours access - potential breach
            {
                "access_count": 50,
                "time_window": "1_hour",
                "user_roles": ["therapist"],
                "data_types": [PIIType.MEDICAL_RECORD],
                "is_breach": True
            }
        ]
        
        for scenario in breach_scenarios:
            is_breach = pii_detector._detect_potential_breach(
                scenario["access_count"],
                scenario["time_window"],
                scenario["user_roles"],
                scenario["data_types"]
            )
            assert is_breach == scenario["is_breach"]
    
    def test_pii_data_anonymization_strategies(self, pii_detector):
        """Test advanced PII anonymization strategies."""
        test_data = {
            "patient": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "555-123-4567",
                "ssn": "123-45-6789",
                "medical_id": "PAT-12345"
            }
        }
        
        # Test different anonymization levels
        levels = ["minimal", "standard", "conservative", "aggressive"]
        
        for level in levels:
            anonymized = pii_detector.anonymize_data(test_data, level=level)
            
            assert "patient" in anonymized
            patient = anonymized["patient"]
            
            # Higher levels should have more masking
            if level == "minimal":
                # Should mask only the most sensitive
                assert "123-45-6789" not in str(patient)
            elif level == "aggressive":
                # Should mask everything
                assert "John Doe" not in str(patient)
                assert "john.doe@example.com" not in str(patient)
                assert "555-123-4567" not in str(patient)
                assert "PAT-12345" not in str(patient)
    
    def test_real_time_pii_monitoring(self, pii_detector):
        """Test real-time PII monitoring capabilities."""
        monitoring_events = [
            {
                "timestamp": datetime.now(),
                "event_type": "access",
                "user_id": "user_123",
                "resource": "patient_456",
                "pii_types": [PIIType.MEDICAL_RECORD],
                "action": "view"
            },
            {
                "timestamp": datetime.now(),
                "event_type": "modify",
                "user_id": "therapist_789",
                "resource": "patient_456",
                "pii_types": [PIIType.SSN, PIIType.EMAIL],
                "action": "update"
            }
        ]
        
        # Process monitoring events
        for event in monitoring_events:
            alert = pii_detector.process_pii_event(event)
            
            if event["pii_types"] and event["pii_types"][0] == PIIType.SSN:
                # SSN access should generate alert
                assert alert is not None
                assert alert["severity"] in ["medium", "high"]
            else:
                # Other events may or may not generate alerts
                assert isinstance(alert, (dict, type(None)))
    
    def test_cross_border_pii_handling(self, pii_detector):
        """Test cross-border PII data handling compliance."""
        transfer_cases = [
            # US to US - no restrictions
            {
                "from_country": "US",
                "to_country": "US",
                "pii_types": [PIIType.EMAIL, PIIType.PHONE],
                "compliant": True
            },
            
            # US to EU - GDPR applies
            {
                "from_country": "US", 
                "to_country": "DE",
                "pii_types": [PIIType.EMAIL, PIIType.SSN],
                "compliant": False
            },
            
            # EU to EU - GDPR compliant with proper consent
            {
                "from_country": "FR",
                "to_country": "DE",
                "pii_types": [PIIType.EMAIL],
                "consent": True,
                "compliant": True
            },
            
            # Any country to forbidden - always non-compliant
            {
                "from_country": "US",
                "to_country": "XX",
                "pii_types": [PIIType.SSN],
                "compliant": False
            }
        ]
        
        for case in transfer_cases:
            compliant = pii_validator._validate_cross_border_transfer(
                case["from_country"], case["to_country"], 
                case["pii_types"], case.get("consent", False)
            )
            assert compliant == case["compliant"]
    
    def test_pii_disaster_recovery(self, pii_detector):
        """Test PII disaster recovery procedures."""
        disaster_scenarios = [
            {
                "type": "data_corruption",
                "affected_data": ["patient_records", "audit_logs"],
                "backup_available": True,
                "recovery_possible": True
            },
            {
                "type": "data_deletion",
                "affected_data": ["audit_logs"],
                "backup_available": False,
                "recovery_possible": False
            },
            {
                "type": "encryption_breach",
                "affected_data": ["all_encrypted_data"],
                "backup_available": True,
                "recovery_possible": True
            }
        ]
        
        for scenario in disaster_scenarios:
            recovery_plan = pii_detector.create_disaster_recovery_plan(scenario)
            
            assert isinstance(recovery_plan, dict)
            assert "steps" in recovery_plan
            assert "timeline" in recovery_plan
            assert "data_loss" in recovery_plan
            
            if scenario["recovery_possible"]:
                assert len(recovery_plan["steps"]) > 0
            else:
                assert recovery_plan["data_loss"] > 0