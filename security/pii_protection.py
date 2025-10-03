"""
PII Protection Module for AI Therapist.

Provides comprehensive PII detection, classification, masking, and anonymization
functions with HIPAA compliance and real-time filtering capabilities.
"""

import os
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path


class PIIType(Enum):
    """Types of personally identifiable information."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    DOB = "date_of_birth"
    MEDICAL_ID = "medical_id"
    INSURANCE_ID = "insurance_id"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    LOCATION = "location"
    MEDICAL_CONDITION = "medical_condition"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    VOICE_TRANSCRIPTION = "voice_transcription"


class MaskingStrategy(Enum):
    """Strategies for masking PII data."""
    FULL_MASK = "full_mask"  # Replace entire value with mask
    PARTIAL_MASK = "partial_mask"  # Show partial data (e.g., first/last chars)
    HASH_MASK = "hash_mask"  # Hash the value
    REMOVE = "remove"  # Remove entirely
    ANONYMIZE = "anonymize"  # Replace with generic placeholder


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PIIProtectionConfig:
    """Configuration for PII protection."""
    enable_detection: bool = True
    enable_masking: bool = True
    enable_audit: bool = True
    hipaa_compliance: bool = True
    masking_strategy: MaskingStrategy = MaskingStrategy.PARTIAL_MASK
    sensitive_roles_only: bool = False  # Only apply to patients/therapists
    allowed_roles: List[str] = None  # Roles allowed to see full PII
    audit_log_path: Optional[str] = None


class PIIDetector:
    """Detects personally identifiable information in text and data."""

    def __init__(self):
        """Initialize PII detector with regex patterns."""
        self.logger = logging.getLogger(__name__)

        # Compile regex patterns for performance
        self.patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIType.PHONE: re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
            ),
            PIIType.DOB: re.compile(
                r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
            ),
            PIIType.MEDICAL_ID: re.compile(
                r'\b(?:MRN|Patient\s*ID|Medical\s*Record)[\s:]*([A-Za-z0-9\-]{5,})\b',
                re.IGNORECASE
            ),
            PIIType.INSURANCE_ID: re.compile(
                r'\b(?:Insurance\s*ID|Policy\s*Number|Member\s*ID)[\s:]*([A-Za-z0-9\-]{5,})\b',
                re.IGNORECASE
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ),
            PIIType.ADDRESS: re.compile(
                r'\b\d+\s+[A-Za-z0-9\s,.-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl|Court|Ct)\b',
                re.IGNORECASE
            ),
            PIIType.MEDICAL_CONDITION: re.compile(
                r'\b(?:depression|anxiety|schizophrenia|bipolar|PTSD|OCD|panic\s+attack|suicide|self-harm|cancer|diabetes|heart\s+disease)\b',
                re.IGNORECASE
            ),
            PIIType.MEDICATION: re.compile(
                r'\b(?:prozac|sertraline|zoloft|lexapro|citalopram|fluoxetine|paroxetine|paxil|escitalopram|venlafaxine|effexor)\b',
                re.IGNORECASE
            ),
            PIIType.TREATMENT: re.compile(
                r'\b(?:therapy|counseling|psychotherapy|cognitive\s+behavioral|CBT|dialectical\s+behavioral|DBT|exposure\s+therapy|medication\s+management)\b',
                re.IGNORECASE
            )
        }

        # Name detection patterns (more complex)
        self.name_patterns = [
            re.compile(r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # Title First Last
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # First Middle Last
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # First Last
        ]

    def detect_pii(self, text: str, context: Optional[str] = None) -> List[PIIDetectionResult]:
        """
        Detect PII in text content.

        Args:
            text: Text to analyze for PII
            context: Additional context (e.g., "medical_record", "voice_transcription")

        Returns:
            List of detected PII instances
        """
        if not text or not isinstance(text, str):
            return []

        results = []

        # Detect structured PII patterns
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                result = PIIDetectionResult(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,  # High confidence for regex matches
                    context=context
                )
                results.append(result)

        # Detect names (lower confidence)
        for pattern in self.name_patterns:
            for match in pattern.finditer(text):
                result = PIIDetectionResult(
                    pii_type=PIIType.NAME,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7,  # Lower confidence for names
                    context=context
                )
                results.append(result)

        # Context-specific detection
        if context == "voice_transcription":
            # Look for sensitive voice content
            voice_patterns = [
                re.compile(r'\b(?:i\s+want\s+to\s+die|i\s+feel\s+suicidal|kill\s+myself|harm\s+myself)\b', re.IGNORECASE),
                re.compile(r'\b(?:emergency|crisis|help\s+me)\b', re.IGNORECASE)
            ]
            for pattern in voice_patterns:
                for match in pattern.finditer(text):
                    result = PIIDetectionResult(
                        pii_type=PIIType.VOICE_TRANSCRIPTION,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.95,
                        context="crisis_voice_content"
                    )
                    results.append(result)

        return results

    def detect_in_dict(self, data: Dict[str, Any], path: str = "") -> List[Tuple[str, PIIDetectionResult]]:
        """
        Recursively detect PII in nested dictionary structures.

        Args:
            data: Dictionary to scan
            path: Current path in the structure

        Returns:
            List of (field_path, detection_result) tuples
        """
        results = []

        def _scan_value(value: Any, current_path: str):
            if isinstance(value, str):
                detections = self.detect_pii(value)
                for detection in detections:
                    results.append((current_path, detection))
            elif isinstance(value, dict):
                for key, val in value.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    _scan_value(val, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    new_path = f"{current_path}[{i}]"
                    _scan_value(item, new_path)

        _scan_value(data, path)
        return results


class PIIMasker:
    """Handles masking and anonymization of PII data."""

    def __init__(self, strategy: MaskingStrategy = MaskingStrategy.PARTIAL_MASK):
        """Initialize PII masker."""
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

    def mask_value(self, value: str, pii_type: PIIType) -> str:
        """
        Mask a PII value based on the configured strategy.

        Args:
            value: The PII value to mask
            pii_type: Type of PII for appropriate masking

        Returns:
            Masked value
        """
        if not value:
            return value

        if self.strategy == MaskingStrategy.REMOVE:
            return "[REDACTED]"
        elif self.strategy == MaskingStrategy.FULL_MASK:
            return self._full_mask(value, pii_type)
        elif self.strategy == MaskingStrategy.PARTIAL_MASK:
            return self._partial_mask(value, pii_type)
        elif self.strategy == MaskingStrategy.HASH_MASK:
            return self._hash_mask(value)
        elif self.strategy == MaskingStrategy.ANONYMIZE:
            return self._anonymize(value, pii_type)
        else:
            # Default to partial mask
            return self._partial_mask(value, pii_type)

    def _full_mask(self, value: str, pii_type: PIIType) -> str:
        """Replace entire value with mask."""
        if pii_type == PIIType.EMAIL:
            return "[EMAIL REDACTED]"
        elif pii_type == PIIType.PHONE:
            return "[PHONE REDACTED]"
        elif pii_type == PIIType.ADDRESS:
            return "[ADDRESS REDACTED]"
        elif pii_type == PIIType.NAME:
            return "[NAME REDACTED]"
        else:
            return "[REDACTED]"

    def _partial_mask(self, value: str, pii_type: PIIType) -> str:
        """Show partial data with sensitive parts masked."""
        if pii_type == PIIType.EMAIL:
            # Show first 2 chars and domain
            if "@" in value:
                local, domain = value.split("@", 1)
                if len(local) > 2:
                    return f"{local[:2]}***@{domain}"
                else:
                    return f"***@{domain}"
        elif pii_type == PIIType.PHONE:
            # Show last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
        elif pii_type == PIIType.ADDRESS:
            # Mask street number, keep street name
            parts = value.split()
            if parts:
                parts[0] = "***"
                return " ".join(parts)
        elif pii_type == PIIType.NAME:
            # Show first initial and last name
            parts = value.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}. {' '.join(parts[1:])}"
        elif pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"
        elif pii_type == PIIType.SSN:
            # Show last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"***-**-{digits[-4:]}"

        # Default: show first and last characters
        if len(value) <= 2:
            return "*" * len(value)
        return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"

    def _hash_mask(self, value: str) -> str:
        """Hash the value for anonymization."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _anonymize(self, value: str, pii_type: PIIType) -> str:
        """Replace with generic placeholder."""
        placeholders = {
            PIIType.EMAIL: "user@example.com",
            PIIType.PHONE: "(555) 123-4567",
            PIIType.ADDRESS: "123 Anonymous St",
            PIIType.NAME: "Anonymous User",
            PIIType.SSN: "XXX-XX-XXXX",
            PIIType.CREDIT_CARD: "XXXX-XXXX-XXXX-XXXX"
        }
        return placeholders.get(pii_type, "[ANONYMIZED]")


class PIIProtection:
    """Main PII protection system with detection, masking, and filtering."""

    def __init__(self, config: Optional[PIIProtectionConfig] = None):
        """Initialize PII protection system."""
        self.config = config or PIIProtectionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.detector = PIIDetector()
        self.masker = PIIMasker(self.config.masking_strategy)

        # Audit trail for PII access/modifications
        self.audit_trail: List[Dict[str, Any]] = []
        self.audit_enabled = self.config.enable_audit

        # HIPAA compliance tracking
        self.hipaa_violations: List[Dict[str, Any]] = []

        # Initialize from environment
        self._load_env_config()

    def _load_env_config(self):
        """Load configuration from environment variables."""
        self.config.enable_detection = os.getenv("PII_DETECTION_ENABLED", "true").lower() == "true"
        self.config.enable_masking = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
        self.config.hipaa_compliance = os.getenv("HIPAA_COMPLIANCE_ENABLED", "true").lower() == "true"
        self.config.audit_log_path = os.getenv("PII_AUDIT_LOG_PATH")

        # Masking strategy from env
        strategy_env = os.getenv("PII_MASKING_STRATEGY", "partial_mask").upper()
        try:
            self.config.masking_strategy = MaskingStrategy[strategy_env + "_MASK"]
            self.masker.strategy = self.config.masking_strategy
        except KeyError:
            self.logger.warning(f"Invalid masking strategy: {strategy_env}, using default")

    def sanitize_text(self, text: str, context: Optional[str] = None,
                     user_role: Optional[str] = None) -> str:
        """
        Sanitize text by detecting and masking PII.

        Args:
            text: Text to sanitize
            context: Context for detection
            user_role: User role for access control

        Returns:
            Sanitized text
        """
        if not self.config.enable_detection or not text:
            return text

        # Check role-based access
        if not self._has_pii_access(user_role):
            detections = self.detector.detect_pii(text, context)
            masked_text = text

            # Sort by position (reverse order to maintain indices)
            detections.sort(key=lambda x: x.start_pos, reverse=True)

            for detection in detections:
                if self._should_mask_for_role(detection.pii_type, user_role):
                    mask = self.masker.mask_value(detection.value, detection.pii_type)
                    masked_text = (
                        masked_text[:detection.start_pos] +
                        mask +
                        masked_text[detection.end_pos:]
                    )

                    # Audit the masking
                    if self.audit_enabled:
                        self._audit_pii_access(
                            "mask",
                            detection.pii_type,
                            detection.value,
                            user_role,
                            context
                        )

            return masked_text

        return text

    def sanitize_dict(self, data: Dict[str, Any], user_role: Optional[str] = None,
                     context: Optional[str] = None) -> Dict[str, Any]:
        """
        Sanitize dictionary by masking PII in nested structures.

        Args:
            data: Dictionary to sanitize
            user_role: User role for access control
            context: Context for detection

        Returns:
            Sanitized dictionary
        """
        if not self.config.enable_detection:
            return data

        result = data.copy()

        # Check role-based access
        if not self._has_pii_access(user_role):
            detections = self.detector.detect_in_dict(data)

            for field_path, detection in detections:
                if self._should_mask_for_role(detection.pii_type, user_role):
                    # Navigate to the field and mask it
                    self._mask_field_in_dict(result, field_path, detection)

                    # Audit the masking
                    if self.audit_enabled:
                        self._audit_pii_access(
                            "mask",
                            detection.pii_type,
                            detection.value,
                            user_role,
                            context,
                            field_path
                        )

        return result

    def _mask_field_in_dict(self, data: Dict[str, Any], field_path: str,
                           detection: PIIDetectionResult):
        """Mask a specific field in a nested dictionary."""
        parts = field_path.replace(']', '').replace('[', '.').split('.')
        current = data

        # Navigate to the field
        for part in parts[:-1]:
            if part not in current:
                return
            current = current[part]

        field_name = parts[-1]
        if field_name in current and isinstance(current[field_name], str):
            current[field_name] = self.masker.mask_value(
                current[field_name], detection.pii_type
            )

    def _has_pii_access(self, user_role: Optional[str]) -> bool:
        """Check if user role has access to full PII."""
        if not self.config.sensitive_roles_only:
            return True

        if not user_role or not self.config.allowed_roles:
            return False

        return user_role.lower() in [role.lower() for role in self.config.allowed_roles]

    def _should_mask_for_role(self, pii_type: PIIType, user_role: Optional[str]) -> bool:
        """Determine if PII should be masked for the given role."""
        # Always mask sensitive medical information for non-authorized roles
        sensitive_types = {
            PIIType.MEDICAL_CONDITION, PIIType.MEDICATION, PIIType.TREATMENT,
            PIIType.MEDICAL_ID, PIIType.VOICE_TRANSCRIPTION
        }

        if pii_type in sensitive_types and not self._has_pii_access(user_role):
            return True

        # Role-based masking
        if self.config.sensitive_roles_only and not self._has_pii_access(user_role):
            return True

        return False

    def _audit_pii_access(self, action: str, pii_type: PIIType, value: str,
                         user_role: Optional[str], context: Optional[str],
                         field_path: Optional[str] = None):
        """Audit PII access or modification."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "pii_type": pii_type.value,
            "user_role": user_role,
            "context": context,
            "field_path": field_path,
            "value_hash": hashlib.sha256(value.encode()).hexdigest()[:16],
            "hipaa_compliant": self._check_hipaa_compliance(action, pii_type, user_role)
        }

        self.audit_trail.append(audit_entry)

        # Log HIPAA violations
        if not audit_entry["hipaa_compliant"]:
            violation = {
                "timestamp": audit_entry["timestamp"],
                "violation_type": "unauthorized_pii_access",
                "details": audit_entry
            }
            self.hipaa_violations.append(violation)
            self.logger.warning(f"HIPAA violation detected: {violation}")

    def _check_hipaa_compliance(self, action: str, pii_type: PIIType,
                               user_role: Optional[str]) -> bool:
        """Check if action is HIPAA compliant."""
        if not self.config.hipaa_compliance:
            return True

        # Define HIPAA-compliant roles for different PII types
        hipaa_access_roles = {
            PIIType.MEDICAL_CONDITION: ["therapist", "admin"],
            PIIType.MEDICATION: ["therapist", "admin"],
            PIIType.TREATMENT: ["therapist", "admin"],
            PIIType.MEDICAL_ID: ["therapist", "admin"],
            PIIType.VOICE_TRANSCRIPTION: ["therapist", "admin"]
        }

        required_roles = hipaa_access_roles.get(pii_type)
        if not required_roles:
            return True  # Not HIPAA-protected

        return user_role and user_role.lower() in required_roles

    def get_audit_trail(self, user_id: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get PII access audit trail."""
        filtered_trail = self.audit_trail

        if user_id:
            filtered_trail = [entry for entry in filtered_trail if entry.get("user_id") == user_id]

        if start_date:
            filtered_trail = [entry for entry in filtered_trail
                            if datetime.fromisoformat(entry["timestamp"]) >= start_date]

        if end_date:
            filtered_trail = [entry for entry in filtered_trail
                            if datetime.fromisoformat(entry["timestamp"]) <= end_date]

        return filtered_trail

    def get_hipaa_violations(self) -> List[Dict[str, Any]]:
        """Get list of HIPAA violations."""
        return self.hipaa_violations.copy()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on PII protection system."""
        return {
            "status": "healthy",
            "components": {
                "detector": "operational",
                "masker": "operational",
                "auditing": "enabled" if self.audit_enabled else "disabled"
            },
            "statistics": {
                "audit_entries": len(self.audit_trail),
                "hipaa_violations": len(self.hipaa_violations),
                "detection_enabled": self.config.enable_detection,
                "masking_enabled": self.config.enable_masking
            }
        }