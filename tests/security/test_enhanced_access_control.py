"""
Fixed security tests with proper access control and audit logging.
"""

import pytest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from voice.enhanced_security import (
    VoiceSecurity,
    SecurityConfig,
    EnhancedAccessManager,
    SecurityLevel,
    get_voice_security_instance
)

class TestAccessControl:
    """Fixed access control tests."""

    @pytest.fixture
    def security_config(self):
        """Create security configuration for testing."""
        return SecurityConfig(
            encryption_enabled=True,
            consent_required=True,
            privacy_mode=False,
            hipaa_compliance_enabled=True,
            data_retention_days=30,
            audit_logging_enabled=True,
            session_timeout_minutes=30,
            max_login_attempts=3
        )

    @pytest.fixture
    def security(self, security_config):
        """Create VoiceSecurity instance for testing."""
        return VoiceSecurity(security_config)

    def test_basic_access_control(self, security):
        """Test basic access control functionality."""
        user_id = 'patient_123'
        resource = 'own_voice_data'
        permission = 'read'

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)

        # Test granted access
        has_access = security.access_manager.has_access(user_id, resource, permission)
        assert has_access == True

        # Test denied access
        denied_access = security.access_manager.has_access(user_id, 'restricted_resource', 'admin')
        assert denied_access == False

    def test_role_based_access_control(self, security):
        """Test comprehensive role-based access control with proper isolation."""
        # Clear any existing audit logs
        security.clear_audit_logs()

        # Define role permissions and test cross-role access denial
        role_tests = [
            # (user_id, resource, permission, should_have_access)
            ('patient_123', 'own_voice_data', 'read', True),
            ('patient_123', 'therapy_notes', 'read', False),  # Patient can't access therapy notes
            ('patient_123', 'admin_panel', 'full_access', False),  # Patient can't access admin
            ('therapist_456', 'assigned_patient_data', 'read', True),
            ('therapist_456', 'admin_panel', 'full_access', False),  # Therapist can't access admin
            ('therapist_456', 'system_configuration', 'read', False),  # Therapist can't access system config
            ('admin_1', 'admin_panel', 'full_access', True),
            ('admin_1', 'system_configuration', 'read', True),  # Admin CAN access system config
            ('admin_1', 'all_patient_data', 'read', True)
        ]

        for user_id, resource, permission, expected_access in role_tests:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == expected_access, f"{user_id} should {'have' if expected_access else 'not have'} {permission} access to {resource}"

    def test_access_control_audit_integration(self, security):
        """Test access control audit integration."""
        # Clear existing logs
        security.clear_audit_logs()

        user_id = 'audit_test_user'
        resource = 'test_resource'
        permission = 'read'

        # Grant access (this should create an audit log)
        security.access_manager.grant_access(user_id, resource, permission)

        # Check that grant was logged
        grant_logs = security.get_security_events('access_granted')
        assert len(grant_logs) >= 1, "Access grant should be logged"

        # Verify log details
        grant_log = grant_logs[0]
        assert grant_log.user_id == user_id
        assert grant_log.resource == resource
        assert grant_log.action == 'grant_access'
        assert grant_log.result == 'success'

        # Revoke access (this should also create an audit log)
        security.access_manager.revoke_access(user_id, resource, permission)

        # Check that revoke was logged
        revoke_logs = security.get_security_events('access_revoked')
        assert len(revoke_logs) >= 1, "Access revocation should be logged"

    def test_access_control_data_isolation(self, security):
        """Test access control data isolation between different user types."""
        # Clear existing logs
        security.clear_audit_logs()

        # Test that different users cannot access each other's data
        isolation_tests = [
            # (user_id, resource, permission, should_have_access, description)
            ('patient_123', 'own_voice_data', 'read', True, "Patient can read own data"),
            ('patient_456', 'own_voice_data', 'read', True, "Different patient can read own data"),
            ('patient_123', 'own_voice_data', 'read', True, "Patient can read own data (role-based)"),
            ('therapist_456', 'assigned_patient_data', 'read', True, "Therapist can read assigned data"),
            ('admin_1', 'system_configuration', 'read', True, "Admin can read system config"),
            ('patient_123', 'assigned_patient_data', 'read', False, "Patient cannot read therapist data"),
            ('guest_123', 'system_configuration', 'read', False, "Guest cannot read system config")
        ]

        for user_id, resource, permission, expected_access, description in isolation_tests:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == expected_access, f"{description}: {user_id} {'should' if expected_access else 'should not'} have access to {resource}"

    def test_access_control_malicious_attempts(self, security):
        """Test access control against malicious attempts."""
        malicious_attempts = [
            ('patient_123; DROP TABLE users; --', 'user_data', 'read'),
            ('patient_123', 'admin_panel', 'full_access'),
            ('../../../etc/passwd', 'user_data', 'read'),
            ('<script>alert("xss")</script>', 'user_data', 'read')
        ]

        for user_id, resource, permission in malicious_attempts:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == False, f"Malicious attempt should be denied: {user_id} on {resource}"

    def test_access_control_session_management(self, security):
        """Test access control with session management."""
        user_id = 'session_user_123'
        resource = 'session_resource'
        permission = 'read'

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)

        # Test access within session
        has_access = security.access_manager.has_access(user_id, resource, permission)
        assert has_access == True

        # Verify session timeout is configured
        assert security.session_timeout_minutes == 30

    def test_access_control_concurrent_sessions(self, security):
        """Test access control with concurrent sessions."""
        users = ['user1', 'user2', 'user3']
        resource = 'shared_resource'
        permission = 'read'

        # Grant access to multiple users
        for user_id in users:
            security.access_manager.grant_access(user_id, resource, permission)

        # All users should have access
        for user_id in users:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == True

    def test_access_control_granular_permissions(self, security):
        """Test granular permission control."""
        user_id = 'granular_user'
        resource = 'granular_resource'

        # Grant specific permissions
        permissions = ['read', 'write']
        for permission in permissions:
            security.access_manager.grant_access(user_id, resource, permission)

        # Test granted permissions
        for permission in permissions:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == True

        # Test non-granted permission
        has_access = security.access_manager.has_access(user_id, resource, 'delete')
        assert has_access == False

    def test_access_control_privilege_escalation_attempts(self, security):
        """Test access control against privilege escalation attempts."""
        # Patient trying to access admin resources
        patient_user = 'patient_123'
        admin_resources = [
            ('admin_panel', 'full_access'),
            ('system_configuration', 'update'),
            ('user_management', 'delete'),
            ('security_settings', 'update')
        ]

        for resource, permission in admin_resources:
            has_access = security.access_manager.has_access(patient_user, resource, permission)
            assert has_access == False, f"Patient should not have {permission} access to {resource}"

        # Therapist trying to access admin resources
        therapist_user = 'therapist_456'
        for resource, permission in admin_resources:
            has_access = security.access_manager.has_access(therapist_user, resource, permission)
            assert has_access == False, f"Therapist should not have {permission} access to {resource}"

    def test_access_control_dynamic_permission_changes(self, security):
        """Test dynamic permission changes."""
        user_id = 'dynamic_user'
        resource = 'dynamic_resource'
        permission = 'read'

        # Initially no access
        has_access = security.access_manager.has_access(user_id, resource, permission)
        assert has_access == False

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)
        has_access = security.access_manager.has_access(user_id, resource, permission)
        assert has_access == True

        # Revoke access
        security.access_manager.revoke_access(user_id, resource, permission)
        has_access = security.access_manager.has_access(user_id, resource, permission)
        assert has_access == False

    def test_access_control_resource_ownership(self, security):
        """Test resource ownership-based access control."""
        # Test that users can access their own resources
        own_resource_tests = [
            ('patient_123', 'own_voice_data'),
            ('therapist_456', 'assigned_patient_data'),
            ('admin_1', 'admin_panel')
        ]

        for user_id, resource in own_resource_tests:
            has_access = security.access_manager.has_access(user_id, resource, 'read')
            assert has_access == True, f"User should have read access to own resource: {resource}"

        # Test that users cannot access others' resources
        other_resource_tests = [
            ('patient_123', 'therapist_456_assigned_data'),
            ('therapist_456', 'patient_123_own_data'),
            ('guest_123', 'admin_1_panel')
        ]

        for user_id, resource in other_resource_tests:
            has_access = security.access_manager.has_access(user_id, resource, 'read')
            assert has_access == False, f"User should not have access to others' resource: {resource}"

    def test_access_control_emergency_access(self, security):
        """Test emergency access controls."""
        # Admin should have emergency access
        admin_user = 'admin_1'
        emergency_resources = [
            ('emergency_controls', 'full_access'),
            ('emergency_contacts', 'read'),
            ('crisis_intervention', 'execute')
        ]

        for resource, permission in emergency_resources:
            has_access = security.access_manager.has_access(admin_user, resource, permission)
            assert has_access == True, f"Admin should have emergency {permission} access to {resource}"

        # Regular users should not have emergency access
        regular_user = 'patient_123'
        for resource, permission in emergency_resources:
            has_access = security.access_manager.has_access(regular_user, resource, permission)
            assert has_access == False, f"Regular user should not have emergency {permission} access to {resource}"

    def test_access_control_authentication_bypass_attempts(self, security):
        """Test access control against authentication bypass attempts."""
        bypass_attempts = [
            ('', 'user_data', 'read'),  # Empty user ID
            (None, 'user_data', 'read'),  # None user ID
            ('../../admin', 'admin_panel', 'full_access'),  # Path traversal
            ('" OR "1"="1', 'user_data', 'read'),  # SQL injection
            ('<admin>true</admin>', 'admin_panel', 'full_access'),  # XML injection
        ]

        for user_id, resource, permission in bypass_attempts:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == False, f"Authentication bypass attempt should be denied: {user_id}"

    def test_access_control_performance_under_load(self, security):
        """Test access control performance under load."""
        import time

        user_id = 'performance_user'
        resource = 'performance_resource'
        permission = 'read'

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)

        # Test multiple access checks
        start_time = time.time()
        for i in range(1000):
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == True
        end_time = time.time()

        # Performance should be reasonable (less than 1 second for 1000 checks)
        assert (end_time - start_time) < 1.0, "Access control performance should be fast"

    def test_access_control_revocation_cascade(self, security):
        """Test access control revocation cascade effects."""
        user_id = 'cascade_user'
        resources = ['resource1', 'resource2', 'resource3']
        permissions = ['read', 'write']

        # Grant multiple permissions
        for resource in resources:
            for permission in permissions:
                security.access_manager.grant_access(user_id, resource, permission)

        # Verify all access granted
        for resource in resources:
            for permission in permissions:
                has_access = security.access_manager.has_access(user_id, resource, permission)
                assert has_access == True

        # Revoke all access to a specific resource
        security.access_manager.revoke_access(user_id, 'resource2', 'read')
        security.access_manager.revoke_access(user_id, 'resource2', 'write')

        # Verify revocation
        has_read_access = security.access_manager.has_access(user_id, 'resource2', 'read')
        has_write_access = security.access_manager.has_access(user_id, 'resource2', 'write')
        assert has_read_access == False
        assert has_write_access == False

        # Verify other resources still have access
        for resource in ['resource1', 'resource3']:
            for permission in permissions:
                has_access = security.access_manager.has_access(user_id, resource, permission)
                assert has_access == True

if __name__ == '__main__':
    pytest.main([__file__])
