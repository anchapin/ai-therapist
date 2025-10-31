"""
Mock security module for comprehensive test coverage.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

class SecurityConfig:
    """Mock security configuration."""
    def __init__(self, **kwargs):
        self.encryption_enabled = kwargs.get('encryption_enabled', True)
        self.consent_required = kwargs.get('consent_required', True)
        self.privacy_mode = kwargs.get('privacy_mode', False)
        self.hipaa_compliance_enabled = kwargs.get('hipaa_compliance_enabled', True)
        self.data_retention_days = kwargs.get('data_retention_days', 30)
        self.audit_logging_enabled = kwargs.get('audit_logging_enabled', True)
        self.session_timeout_minutes = kwargs.get('session_timeout_minutes', 30)
        self.max_login_attempts = kwargs.get('max_login_attempts', 3)

class MockAuditLogger:
    """Mock audit logger."""
    def __init__(self):
        self.logs = []
        self.session_logs_cache = {}

    def log_event(self, event_data: Dict[str, Any]):
        """Log an event."""
        event_data['timestamp'] = datetime.now().isoformat()
        self.logs.append(event_data)

    def get_logs(self) -> list:
        """Get all logs."""
        return self.logs.copy()

class VoiceSecurity:
    """Mock voice security implementation."""
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.audit_logger = MockAuditLogger()
        self.logger = logging.getLogger(__name__)

    def _log_security_event(self, **kwargs):
        """Log security event."""
        self.audit_logger.log_event(kwargs)

# Add module-level spec for Python 3.12
__spec__ = None
