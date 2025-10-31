"""Tests for voice security mock module"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the voice directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'voice'))

from security_mock import SecurityConfig, MockAuditLogger, VoiceSecurity


class TestSecurityConfig:
    """Test SecurityConfig class"""
    
    def test_security_config_default_initialization(self):
        """Test that SecurityConfig initializes with default values"""
        config = SecurityConfig()
        assert config.encryption_enabled is True
        assert config.consent_required is True
        assert config.privacy_mode is False
        assert config.hipaa_compliance_enabled is True
        assert config.data_retention_days == 30
        assert config.audit_logging_enabled is True
        assert config.session_timeout_minutes == 30
        assert config.max_login_attempts == 3
    
    def test_security_config_custom_initialization(self):
        """Test that SecurityConfig accepts custom values"""
        config = SecurityConfig(
            encryption_enabled=False,
            consent_required=False,
            privacy_mode=True,
            hipaa_compliance_enabled=False,
            data_retention_days=60,
            audit_logging_enabled=False,
            session_timeout_minutes=60,
            max_login_attempts=5
        )
        assert config.encryption_enabled is False
        assert config.consent_required is False
        assert config.privacy_mode is True
        assert config.hipaa_compliance_enabled is False
        assert config.data_retention_days == 60
        assert config.audit_logging_enabled is False
        assert config.session_timeout_minutes == 60
        assert config.max_login_attempts == 5


class TestMockAuditLogger:
    """Test MockAuditLogger class"""
    
    def test_mock_audit_logger_initialization(self):
        """Test that MockAuditLogger initializes correctly"""
        logger = MockAuditLogger()
        assert logger is not None
        assert hasattr(logger, 'logs')
        assert hasattr(logger, 'session_logs_cache')
        assert logger.logs == []
        assert logger.session_logs_cache == {}
    
    def test_log_event(self):
        """Test event logging"""
        logger = MockAuditLogger()
        event_data = {"event": "test_event", "user": "test_user"}
        
        logger.log_event(event_data)
        
        assert len(logger.logs) == 1
        assert "timestamp" in logger.logs[0]
        assert logger.logs[0]["event"] == "test_event"
        assert logger.logs[0]["user"] == "test_user"
    
    def test_get_logs(self):
        """Test getting logs"""
        logger = MockAuditLogger()
        event_data = {"event": "test_event", "user": "test_user"}
        
        logger.log_event(event_data)
        logs = logger.get_logs()
        
        assert isinstance(logs, list)
        assert len(logs) == 1
        # Verify it returns a copy, not the original list
        logs.append({"new": "event"})
        assert len(logger.logs) == 1


class TestVoiceSecurity:
    """Test VoiceSecurity class"""
    
    def test_voice_security_default_initialization(self):
        """Test that VoiceSecurity initializes with default config"""
        security = VoiceSecurity()
        assert security is not None
        assert isinstance(security.config, SecurityConfig)
        assert isinstance(security.audit_logger, MockAuditLogger)
        assert hasattr(security, 'logger')
    
    def test_voice_security_custom_config_initialization(self):
        """Test that VoiceSecurity accepts custom config"""
        config = SecurityConfig(encryption_enabled=False, privacy_mode=True)
        security = VoiceSecurity(config)
        
        assert security.config.encryption_enabled is False
        assert security.config.privacy_mode is True
    
    def test_log_security_event(self):
        """Test security event logging"""
        security = VoiceSecurity()
        
        security._log_security_event(
            event_type="test_event",
            user_id="test_user",
            details="test details"
        )
        
        logs = security.audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "test_event"
        assert logs[0]["user_id"] == "test_user"
        assert logs[0]["details"] == "test details"
        assert "timestamp" in logs[0]


class TestSecurityMockIntegration:
    """Test integration of security mock components"""
    
    def test_security_components_work_together(self):
        """Test that security mock components work together"""
        config = SecurityConfig(
            encryption_enabled=True,
            audit_logging_enabled=True
        )
        security = VoiceSecurity(config)
        
        # Log a security event
        security._log_security_event(
            event_type="access_attempt",
            user_id="test_user",
            resource="voice_service",
            success=True
        )
        
        # Verify the event was logged
        logs = security.audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "access_attempt"
        assert logs[0]["success"] is True
        
        # Verify configuration is applied
        assert security.config.encryption_enabled is True
        assert security.config.audit_logging_enabled is True