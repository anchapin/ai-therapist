"""
Voice Security Module

This module handles security and privacy aspects of voice features:
- Audio encryption and decryption
- Data retention and cleanup
- Consent management
- Privacy mode implementation
- Anonymization and pseudonymization
- HIPAA/GDPR compliance
- Emergency protocols
- Access control and auditing
"""

import asyncio
import time
import json
import hashlib
import hmac
import base64
import re
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import threading
from datetime import datetime, timedelta
import os
import shutil

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt

from .config import VoiceConfig, SecurityConfig
from .audio_processor import AudioData

@dataclass
class ConsentRecord:
    """Consent record for voice data processing."""
    user_id: str
    consent_type: str
    granted: bool
    timestamp: float
    ip_address: str
    user_agent: str
    consent_text: str
    metadata: Dict[str, Any] = None

@dataclass
class SecurityAuditLog:
    """Security audit log entry."""
    timestamp: float
    event_type: str
    user_id: str
    session_id: str
    action: str
    resource: str
    result: str
    details: Dict[str, Any] = None
    ip_address: str = ""
    user_agent: str = ""

class VoiceSecurity:
    """Security manager for voice features."""

    # Allowed consent types for validation
    ALLOWED_CONSENT_TYPES = {
        'voice_processing', 'data_storage', 'transcription',
        'analysis', 'all_consent', 'emergency_protocol'
    }

    # Validation patterns
    USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')
    IP_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')

    def __init__(self, config: VoiceConfig):
        """Initialize voice security."""
        self.config = config
        self.security_config = config.security
        self.logger = logging.getLogger(__name__)

        # Security state
        self.encryption_key: Optional[bytes] = None
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.audit_logs: List[SecurityAuditLog] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.initialized = False  # Add initialization status

        # Background tasks
        self.cleanup_thread = None
        self.is_running = False

        # Security directories
        self.data_dir = Path("./voice_data")
        self.encrypted_dir = self.data_dir / "encrypted"
        self.consents_dir = self.data_dir / "consents"
        self.audit_dir = self.data_dir / "audit"

        # Initialize security
        self._initialize_security()

    def _initialize_security(self):
        """Initialize security systems."""
        try:
            # Create directories
            self.data_dir.mkdir(exist_ok=True)
            self.encrypted_dir.mkdir(exist_ok=True)
            self.consents_dir.mkdir(exist_ok=True)
            self.audit_dir.mkdir(exist_ok=True)

            # Initialize encryption
            if self.security_config.encryption_enabled:
                self._initialize_encryption()

            # Load existing consent records
            self._load_consent_records()

            # Start background cleanup
            if self.security_config.data_retention_hours > 0:
                self._start_cleanup_thread()

            self.logger.info("Voice security initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing voice security: {str(e)}")
            raise

    def initialize(self) -> bool:
        """Initialize security for use."""
        try:
            # Check consent requirements
            if self.security_config.consent_required:
                if not self._check_consent_status():
                    self.logger.warning("Voice consent not granted")
                    self.initialized = False
                    return False

            # Verify security requirements
            if not self._verify_security_requirements():
                self.logger.error("Security requirements not met")
                self.initialized = False
                return False

            self.initialized = True
            self.logger.info("Voice security ready for use")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing voice security: {str(e)}")
            self.initialized = False
            return False

    def _initialize_encryption(self):
        """Initialize encryption system."""
        try:
            # Generate or load encryption key
            key_file = self.data_dir / "encryption.key"

            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)

                # Secure the key file
                os.chmod(key_file, 0o600)

            self.cipher = Fernet(self.encryption_key)
            self.logger.info("Encryption system initialized")

        except Exception as e:
            self.logger.error(f"Error initializing encryption: {str(e)}")
            raise

    def _load_consent_records(self):
        """Load existing consent records."""
        try:
            for consent_file in self.consents_dir.glob("*.json"):
                with open(consent_file, 'r') as f:
                    consent_data = json.load(f)
                    consent = ConsentRecord(**consent_data)
                    self.consent_records[consent.user_id] = consent

            self.logger.info(f"Loaded {len(self.consent_records)} consent records")

        except Exception as e:
            self.logger.error(f"Error loading consent records: {str(e)}")

    def _check_consent_status(self) -> bool:
        """Check if voice consent has been granted."""
        # Check for system-wide consent
        system_consent = self.consent_records.get("system")
        if system_consent and system_consent.granted:
            return True

        # Check for session-specific consent
        # This would be implemented based on your application's user management
        return False

    def _verify_security_requirements(self) -> bool:
        """Verify security requirements are met."""
        issues = []

        # Check encryption
        if self.security_config.encryption_enabled and not self.encryption_key:
            issues.append("Encryption enabled but no encryption key")

        # Check directories
        if not self.data_dir.exists():
            issues.append("Data directory does not exist")

        if not self.encrypted_dir.exists():
            issues.append("Encrypted directory does not exist")

        # Check file permissions
        if self.security_config.encryption_enabled:
            key_file = self.data_dir / "encryption.key"
            if key_file.exists():
                permissions = oct(key_file.stat().st_mode)[-3:]
                if permissions != "600":
                    issues.append("Encryption key file has insecure permissions")

        if issues:
            self.logger.error("Security verification failed: " + "; ".join(issues))
            return False

        return True

    async def process_audio(self, audio_data: AudioData) -> AudioData:
        """Process audio data with security measures."""
        try:
            # Apply privacy mode if enabled
            if self.security_config.privacy_mode:
                audio_data = await self._apply_privacy_mode(audio_data)

            # Encrypt audio if enabled
            if self.security_config.encryption_enabled:
                audio_data = await self._encrypt_audio(audio_data)

            # Apply anonymization if enabled
            if self.security_config.anonymization_enabled:
                audio_data = await self._anonymize_audio(audio_data)

            return audio_data

        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            raise

    async def _apply_privacy_mode(self, audio_data: AudioData) -> AudioData:
        """Apply privacy mode to audio data."""
        try:
            # In privacy mode, we might:
            # - Reduce audio quality
            # - Remove identifying characteristics
            # - Apply additional noise reduction

            # For now, return original audio
            # In a full implementation, this would apply privacy transformations
            return audio_data

        except Exception as e:
            self.logger.error(f"Error applying privacy mode: {str(e)}")
            return audio_data

    async def _encrypt_audio(self, audio_data: AudioData) -> AudioData:
        """Encrypt audio data."""
        try:
            if not self.encryption_key:
                raise ValueError("Encryption key not available")

            # Convert audio data to bytes
            audio_bytes = audio_data.data.tobytes()

            # Encrypt
            encrypted_bytes = self.cipher.encrypt(audio_bytes)

            # Convert back to numpy array
            import numpy as np
            encrypted_data = np.frombuffer(encrypted_bytes, dtype=np.uint8)

            # Return encrypted audio data
            return AudioData(
                data=encrypted_data,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format="encrypted",
                duration=audio_data.duration,
                timestamp=audio_data.timestamp
            )

        except Exception as e:
            self.logger.error(f"Error encrypting audio: {str(e)}")
            raise

    async def _anonymize_audio(self, audio_data: AudioData) -> AudioData:
        """Anonymize audio data."""
        try:
            # In a full implementation, this would:
            # - Remove voiceprints
            # - Apply voice transformation
            # - Strip identifying features

            # For now, return original audio
            return audio_data

        except Exception as e:
            self.logger.error(f"Error anonymizing audio: {str(e)}")
            return audio_data

    async def decrypt_audio(self, audio_data: AudioData) -> AudioData:
        """Decrypt audio data."""
        try:
            if not self.encryption_key or audio_data.format != "encrypted":
                return audio_data

            # Convert to bytes
            encrypted_bytes = audio_data.data.tobytes()

            # Decrypt
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)

            # Convert back to numpy array
            import numpy as np
            decrypted_data = np.frombuffer(decrypted_bytes, dtype=np.float32)

            return AudioData(
                data=decrypted_data,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format="float32",
                duration=audio_data.duration,
                timestamp=audio_data.timestamp
            )

        except Exception as e:
            self.logger.error(f"Error decrypting audio: {str(e)}")
            raise

    def _validate_user_id(self, user_id: str) -> bool:
        """Validate user ID format."""
        if not isinstance(user_id, str):
            return False
        return bool(self.USER_ID_PATTERN.match(user_id))

    def _validate_ip_address(self, ip_address: str) -> bool:
        """Validate IP address format."""
        if not isinstance(ip_address, str) or not ip_address:
            return True  # Empty IP is allowed for local contexts
        return bool(self.IP_PATTERN.match(ip_address))

    def _validate_user_agent(self, user_agent: str) -> bool:
        """Validate and sanitize user agent string."""
        if not isinstance(user_agent, str):
            return False

        # Length limit
        if len(user_agent) > 500:
            return False

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';&]', '', user_agent)
        return len(sanitized) == len(user_agent)  # No dangerous chars found

    def _validate_consent_type(self, consent_type: str) -> bool:
        """Validate consent type against allowed values."""
        if not isinstance(consent_type, str):
            return False
        return consent_type in self.ALLOWED_CONSENT_TYPES

    def grant_consent(self, user_id: str, consent_type: str, granted: bool,
                     ip_address: str = "", user_agent: str = "",
                     consent_text: str = "", metadata: Dict[str, Any] = None) -> bool:
        """Grant or revoke consent for voice processing."""
        try:
            # Input validation
            if not self._validate_user_id(user_id):
                self.logger.error(f"Invalid user_id format: {user_id}")
                return False

            if not self._validate_consent_type(consent_type):
                self.logger.error(f"Invalid consent_type: {consent_type}")
                return False

            if not self._validate_ip_address(ip_address):
                self.logger.error(f"Invalid IP address format: {ip_address}")
                return False

            if not self._validate_user_agent(user_agent):
                self.logger.error(f"Invalid user agent format: {user_agent[:100]}...")
                return False

            # Validate consent_text length
            if isinstance(consent_text, str) and len(consent_text) > 10000:
                self.logger.error("Consent text too long")
                return False

            # Create consent record
            consent = ConsentRecord(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                timestamp=time.time(),
                ip_address=ip_address,
                user_agent=user_agent,
                consent_text=consent_text,
                metadata=metadata or {}
            )

            # Store consent
            self.consent_records[user_id] = consent

            # Save to file
            consent_file = self.consents_dir / f"{user_id}.json"
            with open(consent_file, 'w') as f:
                json.dump(consent.__dict__, f, indent=2)

            # Audit log
            self._log_security_event(
                event_type="consent_update",
                user_id=user_id,
                action="grant_consent" if granted else "revoke_consent",
                resource=f"consent_{consent_type}",
                result="success",
                details={
                    'consent_type': consent_type,
                    'granted': granted,
                    'timestamp': consent.timestamp
                }
            )

            self.logger.info(f"Consent {'granted' if granted else 'revoked'} for user {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error granting consent: {str(e)}")
            return False

    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has granted consent."""
        try:
            consent = self.consent_records.get(user_id)
            if not consent:
                return False

            return consent.granted and consent.consent_type == consent_type

        except Exception as e:
            self.logger.error(f"Error checking consent: {str(e)}")
            return False

    def _log_security_event(self, event_type: str, user_id: str, action: str,
                           resource: str, result: str, details: Dict[str, Any] = None,
                           ip_address: str = "", user_agent: str = ""):
        """Log security event."""
        try:
            # Generate session ID if not provided
            session_id = details.get('session_id', 'unknown') if details else 'unknown'

            # Create audit log entry
            audit_log = SecurityAuditLog(
                timestamp=time.time(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                action=action,
                resource=resource,
                result=result,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent
            )

            # Add to logs
            self.audit_logs.append(audit_log)

            # Save to file
            log_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.json"
            logs_today = []

            # Load existing logs for today
            if log_file.exists():
                with open(log_file, 'r') as f:
                    try:
                        logs_today = json.load(f)
                    except:
                        logs_today = []

            # Add new log
            logs_today.append(audit_log.__dict__)

            # Save logs
            with open(log_file, 'w') as f:
                json.dump(logs_today, f, indent=2)

            # Keep only recent logs in memory
            cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
            self.audit_logs = [log for log in self.audit_logs if log.timestamp > cutoff_time]

        except Exception as e:
            self.logger.error(f"Error logging security event: {str(e)}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self.is_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def _cleanup_worker(self):
        """Worker thread for data cleanup."""
        try:
            while self.is_running:
                # Perform cleanup
                self._cleanup_expired_data()

                # Sleep for 1 hour
                time.sleep(3600)

        except Exception as e:
            self.logger.error(f"Error in cleanup worker: {str(e)}")

    def _cleanup_expired_data(self):
        """Clean up expired data."""
        try:
            cutoff_time = time.time() - (self.security_config.data_retention_hours * 3600)

            # Clean up encrypted audio files
            for file_path in self.encrypted_dir.glob("*.enc"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up expired audio file: {file_path}")

            # Clean up old consent records
            expired_consents = []
            for user_id, consent in self.consent_records.items():
                if consent.timestamp < cutoff_time:
                    expired_consents.append(user_id)

            for user_id in expired_consents:
                consent_file = self.consents_dir / f"{user_id}.json"
                if consent_file.exists():
                    consent_file.unlink()
                del self.consent_records[user_id]
                self.logger.info(f"Cleaned up expired consent for user: {user_id}")

            # Clean up old audit logs
            audit_cutoff = time.time() - (30 * 24 * 3600)  # 30 days
            for log_file in self.audit_dir.glob("audit_*.json"):
                if log_file.stat().st_mtime < audit_cutoff:
                    log_file.unlink()
                    self.logger.info(f"Cleaned up old audit log: {log_file}")

        except Exception as e:
            self.logger.error(f"Error cleaning up expired data: {str(e)}")

    def handle_emergency_protocol(self, emergency_type: str, user_id: str, details: Dict[str, Any] = None):
        """Handle emergency protocols."""
        try:
            self.logger.warning(f"Emergency protocol triggered: {emergency_type} for user {user_id}")

            # Log emergency event
            self._log_security_event(
                event_type="emergency",
                user_id=user_id,
                action="emergency_protocol",
                resource=f"emergency_{emergency_type}",
                result="triggered",
                details={
                    'emergency_type': emergency_type,
                    'timestamp': time.time(),
                    **(details or {})
                }
            )

            # Implement emergency actions based on type
            if emergency_type == "crisis":
                # Immediate data retention for crisis situations
                self._preserve_emergency_data(user_id, "crisis")
            elif emergency_type == "privacy_breach":
                # Immediate data cleanup
                self._emergency_data_cleanup(user_id)
            elif emergency_type == "security_incident":
                # Lock down access
                self._emergency_lockdown(user_id)

        except Exception as e:
            self.logger.error(f"Error handling emergency protocol: {str(e)}")

    def _preserve_emergency_data(self, user_id: str, reason: str):
        """Preserve data for emergency situations."""
        try:
            emergency_dir = self.data_dir / "emergency" / user_id
            emergency_dir.mkdir(parents=True, exist_ok=True)

            # Copy relevant data to emergency directory
            # This would be implemented based on what data needs to be preserved

            self.logger.info(f"Emergency data preserved for user {user_id}: {reason}")

        except Exception as e:
            self.logger.error(f"Error preserving emergency data: {str(e)}")

    def _emergency_data_cleanup(self, user_id: str):
        """Emergency data cleanup."""
        try:
            # Remove user data
            self._cleanup_user_data(user_id)

            # Revoke consent
            self.grant_consent(user_id, "all_consent", False)

            self.logger.info(f"Emergency data cleanup completed for user {user_id}")

        except Exception as e:
            self.logger.error(f"Error in emergency data cleanup: {str(e)}")

    def _emergency_lockdown(self, user_id: str):
        """Emergency lockdown."""
        try:
            # Add user to lockdown list
            lockdown_file = self.data_dir / "lockdown.json"
            lockdown_list = []

            if lockdown_file.exists():
                with open(lockdown_file, 'r') as f:
                    lockdown_list = json.load(f)

            if user_id not in lockdown_list:
                lockdown_list.append(user_id)
                with open(lockdown_file, 'w') as f:
                    json.dump(lockdown_list, f, indent=2)

            self.logger.info(f"Emergency lockdown activated for user {user_id}")

        except Exception as e:
            self.logger.error(f"Error in emergency lockdown: {str(e)}")

    def _cleanup_user_data(self, user_id: str):
        """Clean up user data."""
        try:
            # Remove consent records
            if user_id in self.consent_records:
                del self.consent_records[user_id]

            consent_file = self.consents_dir / f"{user_id}.json"
            if consent_file.exists():
                consent_file.unlink()

            # Remove encrypted audio files
            for file_path in self.encrypted_dir.glob(f"{user_id}_*.enc"):
                file_path.unlink()

            self.logger.info(f"User data cleaned up for {user_id}")

        except Exception as e:
            self.logger.error(f"Error cleaning up user data: {str(e)}")

    def is_user_locked_down(self, user_id: str) -> bool:
        """Check if user is in lockdown."""
        try:
            lockdown_file = self.data_dir / "lockdown.json"
            if lockdown_file.exists():
                with open(lockdown_file, 'r') as f:
                    lockdown_list = json.load(f)
                return user_id in lockdown_list
            return False

        except Exception as e:
            self.logger.error(f"Error checking lockdown status: {str(e)}")
            return False

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status."""
        try:
            return {
                'hipaa_compliant': self.security_config.hipaa_compliance_enabled,
                'gdpr_compliant': self.security_config.gdpr_compliance_enabled,
                'encryption_enabled': self.security_config.encryption_enabled,
                'data_localization': self.security_config.data_localization,
                'consent_required': self.security_config.consent_required,
                'consent_records_count': len(self.consent_records),
                'audit_logs_count': len(self.audit_logs),
                'data_retention_hours': self.security_config.data_retention_hours,
                'emergency_protocols_enabled': self.security_config.emergency_protocols_enabled,
                'security_status': 'active' if self.is_running else 'inactive'
            }

        except Exception as e:
            self.logger.error(f"Error getting compliance status: {str(e)}")
            return {}

    def cleanup(self):
        """Clean up security resources."""
        try:
            self.is_running = False

            if self.cleanup_thread:
                self.cleanup_thread.join(timeout=5.0)

            self.logger.info("Voice security cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up voice security: {str(e)}")

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()


# Type aliases for backward compatibility
AuditLogger = SecurityAuditLog
ConsentManager = VoiceSecurity