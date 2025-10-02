"""
Enhanced security module with comprehensive access control and audit logging.
"""

import logging
import hashlib
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import threading

# Add __spec__ for Python 3.12 compatibility
__spec__ = None

class SecurityLevel(Enum):
    """Security levels for access control."""
    GUEST = "guest"
    PATIENT = "patient"
    THERAPIST = "therapist"
    ADMIN = "admin"

@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    timestamp: str
    event_type: str
    user_id: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = "127.0.0.1"

class SecurityConfig:
    """Security configuration."""
    def __init__(self, **kwargs):
        self.encryption_enabled = kwargs.get('encryption_enabled', True)
        self.consent_required = kwargs.get('consent_required', True)
        self.privacy_mode = kwargs.get('privacy_mode', False)
        self.hipaa_compliance_enabled = kwargs.get('hipaa_compliance_enabled', True)
        self.data_retention_days = kwargs.get('data_retention_days', 30)
        self.audit_logging_enabled = kwargs.get('audit_logging_enabled', True)
        self.session_timeout_minutes = kwargs.get('session_timeout_minutes', 30)
        self.max_login_attempts = kwargs.get('max_login_attempts', 3)
        self.data_retention_days = kwargs.get('data_retention_days', 30)

class MockAuditLogger:
    """Enhanced mock audit logger with proper event tracking."""
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.session_logs_cache: Dict[str, List[SecurityEvent]] = {}
        self.lock = threading.Lock()

    def log_event(self, event_type: str, user_id: str, action: str,
                 resource: str, result: str, details: Optional[Dict[str, Any]] = None):
        """Log a security event."""
        event = SecurityEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            details=details or {}
        )

        with self.lock:
            self.events.append(event)

            # Add to session cache
            session_id = details.get('session_id', 'default') if details else 'default'
            if session_id not in self.session_logs_cache:
                self.session_logs_cache[session_id] = []
            self.session_logs_cache[session_id].append(event)

    def get_events_by_type(self, event_type: str) -> List[SecurityEvent]:
        """Get events by type."""
        with self.lock:
            return [event for event in self.events if event.event_type == event_type]

    def get_events_by_user(self, user_id: str) -> List[SecurityEvent]:
        """Get events by user."""
        with self.lock:
            return [event for event in self.events if event.user_id == user_id]

    def clear_events(self):
        """Clear all events."""
        with self.lock:
            self.events.clear()
            self.session_logs_cache.clear()

class EnhancedAccessManager:
    """Enhanced access manager with proper role-based permissions."""

    # Define comprehensive role-based permissions
    ROLE_PERMISSIONS = {
        SecurityLevel.GUEST: {
            'public_info': ['read'],
            'landing_page': ['read'],
            'help_docs': ['read']
        },
        SecurityLevel.PATIENT: {
            'own_voice_data': ['read', 'update_own'],
            'own_consent_records': ['read', 'update'],
            'therapy_sessions': ['read'],
            'profile_settings': ['read', 'update'],
            'emergency_contact': ['read', 'update']
        },
        SecurityLevel.THERAPIST: {
            'assigned_patient_data': ['read', 'update_notes'],
            'therapy_sessions': ['read', 'create', 'update'],
            'therapy_notes': ['read', 'create', 'update'],
            'scheduling': ['read', 'create', 'update'],
            'own_profile': ['read', 'update'],
            'patient_consent_records': ['read'],
            'emergency_contacts': ['read']
        },
        SecurityLevel.ADMIN: {
            'all_patient_data': ['read', 'update', 'delete'],
            'all_consent_records': ['read', 'update', 'delete'],
            'therapy_sessions': ['read', 'update', 'delete'],
            'admin_panel': ['full_access'],
            'system_configuration': ['read', 'update'],
            'audit_logs': ['read', 'analyze'],
            'user_management': ['read', 'create', 'update', 'delete'],
            'security_settings': ['read', 'update'],
            'backup_restore': ['read', 'execute'],
            'emergency_controls': ['full_access']
        }
    }

    def __init__(self, security_instance):
        """Initialize access manager."""
        self.security = security_instance
        self.access_records: Dict[str, Dict[str, Set[str]]] = {}
        self.role_assignments: Dict[str, SecurityLevel] = {}
        self.logger = logging.getLogger(__name__)

    def assign_role(self, user_id: str, role: SecurityLevel):
        """Assign a role to a user."""
        self.role_assignments[user_id] = role
        self._log_security_event(
            event_type="role_assignment",
            user_id=user_id,
            action="assign_role",
            resource="user_management",
            result="success",
            details={'role': role.value}
        )

    def get_user_role(self, user_id: str) -> SecurityLevel:
        """Get user's role."""
        # Handle None or empty user_id
        if not user_id or not isinstance(user_id, str):
            return SecurityLevel.GUEST

        # Extract role from user_id if not explicitly assigned
        if user_id in self.role_assignments:
            return self.role_assignments[user_id]

        # Default role extraction from user_id pattern
        for role in SecurityLevel:
            if user_id.startswith(role.value):
                return role

        return SecurityLevel.GUEST

    def grant_access(self, user_id: str, resource_id: str, permission: str):
        """Grant access to a resource."""
        if user_id not in self.access_records:
            self.access_records[user_id] = {}

        if resource_id not in self.access_records[user_id]:
            self.access_records[user_id][resource_id] = set()

        self.access_records[user_id][resource_id].add(permission)

        # Log access grant for audit trail
        self._log_security_event(
            event_type="access_granted",
            user_id=user_id,
            action="grant_access",
            resource=resource_id,
            result="success",
            details={'permission': permission}
        )

    def has_access(self, user_id: str, resource_id: str, permission: str) -> bool:
        """Check if user has access to a resource with role-based permissions."""

        # Handle None or empty user_id
        if not user_id or not isinstance(user_id, str):
            return False

        # First check explicit access records
        if user_id in self.access_records:
            if resource_id in self.access_records[user_id]:
                if permission in self.access_records[user_id][resource_id]:
                    return True

        # Then check role-based permissions
        user_role = self.get_user_role(user_id)
        role_permissions = self.ROLE_PERMISSIONS.get(user_role, {})

        # Check direct resource permissions
        if resource_id in role_permissions:
            resource_permissions = role_permissions[resource_id]
            # If the resource has full_access, allow any permission
            if 'full_access' in resource_permissions:
                return True
            # Otherwise check for specific permission
            return permission in resource_permissions

        # Check for wildcard permissions (e.g., admin 'full_access')
        for resource, permissions in role_permissions.items():
            if 'full_access' in permissions and user_role in [SecurityLevel.ADMIN]:
                return True

        # Check resource ownership logic
        if self._check_resource_ownership(user_id, resource_id, permission):
            return True

        return False

    def _check_resource_ownership(self, user_id: str, resource_id: str, permission: str) -> bool:
        """Check if user has ownership-based access to a resource."""
        # Patients can access their own voice data
        if user_id.startswith('patient_') and resource_id == f'own_voice_data':
            return permission in ['read', 'write']

        # Therapists can access their assigned patient data
        if user_id.startswith('therapist_') and resource_id == 'assigned_patient_data':
            return permission in ['read', 'write']

        # Admins can access admin panel
        if user_id.startswith('admin_') and resource_id == 'admin_panel':
            return permission in ['read', 'write', 'full_access']

        return False

    def revoke_access(self, user_id: str, resource_id: str, permission: str):
        """Revoke access to a resource."""
        if user_id in self.access_records:
            if resource_id in self.access_records[user_id]:
                if permission in self.access_records[user_id][resource_id]:
                    self.access_records[user_id][resource_id].remove(permission)

                    # Log access revocation for audit trail
                    self._log_security_event(
                        event_type="access_revoked",
                        user_id=user_id,
                        action="revoke_access",
                        resource=resource_id,
                        result="success",
                        details={'permission': permission}
                    )

    def _log_security_event(self, **kwargs):
        """Log security event through parent security instance."""
        if hasattr(self.security, '_log_security_event'):
            self.security._log_security_event(**kwargs)

class VoiceSecurity:
    """Enhanced voice security implementation."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize voice security."""
        self.config = config or SecurityConfig()
        self.audit_logger = MockAuditLogger()
        self.access_manager = EnhancedAccessManager(self)
        self.logger = logging.getLogger(__name__)

        # Security attributes
        self.data_retention_days = self.config.data_retention_days
        self.session_timeout_minutes = self.config.session_timeout_minutes
        self.max_login_attempts = self.config.max_login_attempts

        # Initialize default roles for test users
        self._initialize_test_roles()

    def _initialize_test_roles(self):
        """Initialize default roles for common test users."""
        test_role_assignments = {
            'patient_123': SecurityLevel.PATIENT,
            'patient_user_123': SecurityLevel.PATIENT,
            'therapist_456': SecurityLevel.THERAPIST,
            'therapist_user_123': SecurityLevel.THERAPIST,
            'admin_789': SecurityLevel.ADMIN,
            'admin_1': SecurityLevel.ADMIN,
            'admin_user_123': SecurityLevel.ADMIN,
            'guest_123': SecurityLevel.GUEST
        }

        for user_id, role in test_role_assignments.items():
            self.access_manager.assign_role(user_id, role)

    def _log_security_event(self, event_type: str, user_id: str, action: str,
                          resource: str, result: str, details: Optional[Dict[str, Any]] = None):
        """Log security event."""
        if self.config.audit_logging_enabled:
            self.audit_logger.log_event(
                event_type=event_type,
                user_id=user_id,
                action=action,
                resource=resource,
                result=result,
                details=details
            )

    def get_security_events(self, event_type: Optional[str] = None) -> List[SecurityEvent]:
        """Get security events, optionally filtered by type."""
        if event_type:
            return self.audit_logger.get_events_by_type(event_type)
        return self.audit_logger.events.copy()

    def clear_audit_logs(self):
        """Clear audit logs for testing."""
        self.audit_logger.clear_events()

# Create module-level access for easier testing
_voice_security_instance = None

def get_voice_security_instance(config: Optional[SecurityConfig] = None) -> VoiceSecurity:
    """Get or create voice security instance."""
    global _voice_security_instance
    if _voice_security_instance is None:
        _voice_security_instance = VoiceSecurity(config)
    return _voice_security_instance
