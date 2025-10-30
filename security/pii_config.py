"""
PII Configuration Module for AI Therapist.

Provides configurable PII detection rules, patterns, and settings
for HIPAA-compliant data handling with environment variable support.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Pattern
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PIIDetectionPattern:
    """PII detection pattern with regex and metadata."""
    name: str
    pattern: str
    pii_type: str
    confidence: float = 0.9
    description: Optional[str] = None
    enabled: bool = True
    compiled_pattern: Optional[Pattern] = None

    def __post_init__(self):
        """Compile regex pattern."""
        import re
        if self.pattern:
            try:
                self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                logging.warning(f"Invalid regex pattern for {self.name}: {e}")
                self.enabled = False


@dataclass
class PIIDetectionRules:
    """Comprehensive PII detection rules configuration."""

    # Personal identifiers
    names_enabled: bool = True
    emails_enabled: bool = True
    phones_enabled: bool = True
    addresses_enabled: bool = True
    ssn_enabled: bool = True
    dob_enabled: bool = True

    # Medical information
    medical_conditions_enabled: bool = True
    medications_enabled: bool = True
    treatments_enabled: bool = True
    medical_ids_enabled: bool = True

    # Financial information
    credit_cards_enabled: bool = True
    bank_accounts_enabled: bool = True
    insurance_ids_enabled: bool = True

    # Location data
    ip_addresses_enabled: bool = True
    location_data_enabled: bool = True

    # Voice-specific
    voice_transcriptions_enabled: bool = True
    crisis_keywords_enabled: bool = True

    # Custom patterns
    custom_patterns: List[PIIDetectionPattern] = field(default_factory=list)

    # Advanced settings
    context_aware_detection: bool = True
    fuzzy_matching: bool = False
    whitelist_words: List[str] = field(default_factory=list)
    blacklist_words: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default patterns and load from environment."""
        self._load_default_patterns()
        self._load_env_config()

    def _load_default_patterns(self):
        """Load default PII detection patterns."""
        self.custom_patterns.extend([
            # Enhanced name patterns
            PIIDetectionPattern(
                name="full_name_title",
                pattern=r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                pii_type="name",
                description="Names with titles"
            ),
            PIIDetectionPattern(
                name="full_name",
                pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s*[A-Z]?\.?[a-z]*\b',
                pii_type="name",
                confidence=0.8,
                description="Full names"
            ),

            # Medical conditions (expanded)
            PIIDetectionPattern(
                name="mental_health_conditions",
                pattern=r'\b(?:depression|anxiety|bipolar|schizophrenia|PTSD|OCD|panic\s+disorder|social\s+anxiety|generalized\s+anxiety|mood\s+disorder|personality\s+disorder)\b',
                pii_type="medical_condition",
                description="Mental health conditions"
            ),
            PIIDetectionPattern(
                name="physical_conditions",
                pattern=r'\b(?:cancer|diabetes|heart\s+disease|asthma|arthritis|fibromyalgia|chronic\s+pain|migraine|insomnia|eating\s+disorder)\b',
                pii_type="medical_condition",
                description="Physical health conditions"
            ),

            # Medications (expanded)
            PIIDetectionPattern(
                name="antidepressants",
                pattern=r'\b(?:prozac|fluoxetine|sertraline|zoloft|lexapro|escitalopram|citalopram|celexa|paxil|paroxetine|effexor|venlafaxine| Cymbalta|duloxetine|wellbutrin|bupropion)\b',
                pii_type="medication",
                description="Antidepressant medications"
            ),
            PIIDetectionPattern(
                name="anxiolytics",
                pattern=r'\b(?:xanax|alprazolam|ativan|lorazepam|klonopin|clonazepam|valium|diazepam|buspar|buspirone|hydroxyzine|atarax)\b',
                pii_type="medication",
                description="Anti-anxiety medications"
            ),

            # Treatments
            PIIDetectionPattern(
                name="therapy_types",
                pattern=r'\b(?:cognitive\s+behavioral|cognitive-behavioral|CBT|dialectical\s+behavioral|DBT|psychotherapy|counseling|group\s+therapy|individual\s+therapy|exposure\s+therapy|mindfulness|meditation)\b',
                pii_type="treatment",
                description="Therapy and treatment types"
            ),

            # Crisis keywords
            PIIDetectionPattern(
                name="suicide_keywords",
                pattern=r'\b(?:suicide|suicidal|kill\s+myself|end\s+it\s+all|want\s+to\s+die|no\s+reason\s+to\s+live|better\s+off\s+dead)\b',
                pii_type="voice_transcription",
                confidence=0.95,
                description="Suicide-related keywords"
            ),
            PIIDetectionPattern(
                name="harm_keywords",
                pattern=r'\b(?:harm\s+myself|self-harm|cutting|mutilation|hurt\s+myself|injure\s+myself)\b',
                pii_type="voice_transcription",
                confidence=0.95,
                description="Self-harm keywords"
            ),

            # Enhanced address patterns
            PIIDetectionPattern(
                name="us_address_full",
                pattern=r'\b\d+\s+[A-Za-z0-9\s,.-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl|Court|Ct|Circle|Cir|Parkway|Pkwy|Highway|Hwy)\s*,?\s*[A-Za-z\s]+,?\s*\d{5}\b',
                pii_type="address",
                description="Full US addresses"
            ),

            # Financial patterns
            PIIDetectionPattern(
                name="routing_account",
                pattern=r'\b\d{9}\s+\d{6,17}\b',  # Routing + account numbers
                pii_type="bank_account",
                description="Bank routing and account numbers"
            ),

            # Location patterns
            PIIDetectionPattern(
                name="coordinates",
                pattern=r'\b-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}\b',
                pii_type="location",
                description="GPS coordinates"
            ),
        ])

    def _load_env_config(self):
        """Load configuration from environment variables."""
        # Enable/disable detection categories
        self.names_enabled = os.getenv("PII_DETECT_NAMES", "true").lower() == "true"
        self.emails_enabled = os.getenv("PII_DETECT_EMAILS", "true").lower() == "true"
        self.phones_enabled = os.getenv("PII_DETECT_PHONES", "true").lower() == "true"
        self.addresses_enabled = os.getenv("PII_DETECT_ADDRESSES", "true").lower() == "true"
        self.ssn_enabled = os.getenv("PII_DETECT_SSN", "true").lower() == "true"
        self.dob_enabled = os.getenv("PII_DETECT_DOB", "true").lower() == "true"

        self.medical_conditions_enabled = os.getenv("PII_DETECT_MEDICAL_CONDITIONS", "true").lower() == "true"
        self.medications_enabled = os.getenv("PII_DETECT_MEDICATIONS", "true").lower() == "true"
        self.treatments_enabled = os.getenv("PII_DETECT_TREATMENTS", "true").lower() == "true"
        self.medical_ids_enabled = os.getenv("PII_DETECT_MEDICAL_IDS", "true").lower() == "true"

        self.credit_cards_enabled = os.getenv("PII_DETECT_CREDIT_CARDS", "true").lower() == "true"
        self.bank_accounts_enabled = os.getenv("PII_DETECT_BANK_ACCOUNTS", "true").lower() == "true"
        self.insurance_ids_enabled = os.getenv("PII_DETECT_INSURANCE_IDS", "true").lower() == "true"

        self.ip_addresses_enabled = os.getenv("PII_DETECT_IP_ADDRESSES", "true").lower() == "true"
        self.location_data_enabled = os.getenv("PII_DETECT_LOCATION", "true").lower() == "true"

        self.voice_transcriptions_enabled = os.getenv("PII_DETECT_VOICE_TRANSCRIPTIONS", "true").lower() == "true"
        self.crisis_keywords_enabled = os.getenv("PII_DETECT_CRISIS_KEYWORDS", "true").lower() == "true"

        # Advanced settings
        self.context_aware_detection = os.getenv("PII_CONTEXT_AWARE", "true").lower() == "true"
        self.fuzzy_matching = os.getenv("PII_FUZZY_MATCHING", "false").lower() == "true"

        # Load custom patterns from environment
        custom_patterns_env = os.getenv("PII_CUSTOM_PATTERNS")
        if custom_patterns_env:
            try:
                patterns_data = json.loads(custom_patterns_env)
                for pattern_data in patterns_data:
                    pattern = PIIDetectionPattern(**pattern_data)
                    self.custom_patterns.append(pattern)
            except (json.JSONDecodeError, TypeError) as e:
                logging.warning(f"Failed to load custom PII patterns from environment: {e}")

        # Load whitelist/blacklist
        whitelist_env = os.getenv("PII_WHITELIST_WORDS")
        if whitelist_env:
            self.whitelist_words = [word.strip() for word in whitelist_env.split(",")]

        blacklist_env = os.getenv("PII_BLACKLIST_WORDS")
        if blacklist_env:
            self.blacklist_words = [word.strip() for word in blacklist_env.split(",")]

    def get_enabled_patterns(self) -> List[PIIDetectionPattern]:
        """Get all enabled PII detection patterns."""
        enabled_patterns = []

        # Add patterns based on enabled categories
        if self.names_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "name" and p.enabled])
        if self.emails_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "email" and p.enabled])
        if self.phones_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "phone" and p.enabled])
        if self.addresses_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "address" and p.enabled])
        if self.medical_conditions_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "medical_condition" and p.enabled])
        if self.medications_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "medication" and p.enabled])
        if self.treatments_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "treatment" and p.enabled])
        if self.voice_transcriptions_enabled:
            enabled_patterns.extend([p for p in self.custom_patterns if p.pii_type == "voice_transcription" and p.enabled])
        
        # Add all other enabled patterns (including custom ones like SSN)
        for p in self.custom_patterns:
            if p.enabled and p not in enabled_patterns:
                # Check if this PII type is enabled
                pii_type_enabled = True
                if p.pii_type == "ssn" and not self.ssn_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "credit_card" and not self.credit_cards_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "bank_account" and not self.bank_accounts_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "insurance_id" and not self.insurance_ids_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "medical_id" and not self.medical_ids_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "dob" and not self.dob_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "ip_address" and not self.ip_addresses_enabled:
                    pii_type_enabled = False
                elif p.pii_type == "location" and not self.location_data_enabled:
                    pii_type_enabled = False
                
                if pii_type_enabled:
                    enabled_patterns.append(p)

        return enabled_patterns

    def add_custom_pattern(self, pattern: PIIDetectionPattern):
        """Add custom PII detection pattern."""
        self.custom_patterns.append(pattern)
        logging.info(f"Added custom PII pattern: {pattern.name}")

    def remove_custom_pattern(self, name: str):
        """Remove custom PII detection pattern."""
        self.custom_patterns = [p for p in self.custom_patterns if p.name != name]
        logging.info(f"Removed custom PII pattern: {name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "names_enabled": self.names_enabled,
            "emails_enabled": self.emails_enabled,
            "phones_enabled": self.phones_enabled,
            "addresses_enabled": self.addresses_enabled,
            "ssn_enabled": self.ssn_enabled,
            "dob_enabled": self.dob_enabled,
            "medical_conditions_enabled": self.medical_conditions_enabled,
            "medications_enabled": self.medications_enabled,
            "treatments_enabled": self.treatments_enabled,
            "medical_ids_enabled": self.medical_ids_enabled,
            "credit_cards_enabled": self.credit_cards_enabled,
            "bank_accounts_enabled": self.bank_accounts_enabled,
            "insurance_ids_enabled": self.insurance_ids_enabled,
            "ip_addresses_enabled": self.ip_addresses_enabled,
            "location_data_enabled": self.location_data_enabled,
            "voice_transcriptions_enabled": self.voice_transcriptions_enabled,
            "crisis_keywords_enabled": self.crisis_keywords_enabled,
            "context_aware_detection": self.context_aware_detection,
            "fuzzy_matching": self.fuzzy_matching,
            "custom_patterns_count": len(self.custom_patterns),
            "whitelist_words": self.whitelist_words,
            "blacklist_words": self.blacklist_words
        }


class PIIConfig:
    """Main PII configuration management."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize PII configuration."""
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file or os.getenv("PII_CONFIG_FILE")
        self.detection_rules = PIIDetectionRules()
        
        # Add missing attributes for test compatibility
        self.pii_detection_rules = self.detection_rules
        self.detection_enabled = os.getenv("PII_DETECTION_ENABLED", "true").lower() == "true"
        self.masking_enabled = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
        self.audit_enabled = os.getenv("PII_AUDIT_ENABLED", "true").lower() == "true"
        
        # Add masking settings
        self.masking_settings = {
            "mask_char": "*",
            "preserve_length": True,
            "preserve_format": True
        }
        
        # Add audit settings
        self.audit_settings = {
            "log_detections": True,
            "log_masking": True,
            "retention_days": 30
        }

        # Load from file if specified
        if self.config_file and Path(self.config_file).exists():
            self.load_from_file()

    def load_from_file(self, config_file_path: Optional[str] = None):
        """Load configuration from JSON file."""
        file_path = config_file_path or self.config_file
        if not file_path:
            raise ValueError("No configuration file path specified")
            
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)

            # Update detection rules
            for key, value in config_data.items():
                if hasattr(self.detection_rules, key):
                    setattr(self.detection_rules, key, value)

            # Load custom patterns
            if "custom_patterns" in config_data:
                self.detection_rules.custom_patterns = [
                    PIIDetectionPattern(**pattern_data)
                    for pattern_data in config_data["custom_patterns"]
                ]

            self.logger.info(f"Loaded PII configuration from {file_path}")
            
            # Return self for method chaining
            return self

        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load PII configuration: {e}")
            return None

    def load_pii_patterns(self):
        """Load PII patterns from detection rules."""
        return self.detection_rules.get_enabled_patterns()

    def save_to_file(self, file_path: Optional[str] = None):
        """Save configuration to JSON file."""
        file_path = file_path or self.config_file
        if not file_path:
            raise ValueError("No configuration file path specified")

        try:
            config_data = self.detection_rules.to_dict()
            config_data["custom_patterns"] = [
                {
                    "name": p.name,
                    "pattern": p.pattern,
                    "pii_type": p.pii_type,
                    "confidence": p.confidence,
                    "description": p.description,
                    "enabled": p.enabled
                }
                for p in self.detection_rules.custom_patterns
            ]

            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Saved PII configuration to {file_path}")

        except IOError as e:
            self.logger.error(f"Failed to save PII configuration: {e}")
            raise

    def get_detection_rules(self) -> PIIDetectionRules:
        """Get current detection rules."""
        return self.detection_rules

    def update_detection_rules(self, updates: Dict[str, Any]):
        """Update detection rules."""
        for key, value in updates.items():
            if hasattr(self.detection_rules, key):
                setattr(self.detection_rules, key, value)
                self.logger.info(f"Updated PII detection rule: {key} = {value}")
            else:
                self.logger.warning(f"Unknown PII detection rule: {key}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on PII configuration."""
        enabled_patterns = self.detection_rules.get_enabled_patterns()

        return {
            "status": "healthy",
            "config_file": self.config_file,
            "detection_rules": {
                "enabled_categories": sum([
                    self.detection_rules.names_enabled,
                    self.detection_rules.emails_enabled,
                    self.detection_rules.phones_enabled,
                    self.detection_rules.addresses_enabled,
                    self.detection_rules.medical_conditions_enabled,
                    self.detection_rules.medications_enabled,
                    self.detection_rules.voice_transcriptions_enabled
                ]),
                "custom_patterns": len(self.detection_rules.custom_patterns),
                "enabled_patterns": len(enabled_patterns),
                "whitelist_words": len(self.detection_rules.whitelist_words),
                "blacklist_words": len(self.detection_rules.blacklist_words)
            },
            "features": {
                "context_aware_detection": self.detection_rules.context_aware_detection,
                "fuzzy_matching": self.detection_rules.fuzzy_matching
            }
        }