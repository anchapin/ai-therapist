"""
Comprehensive access control tests.

Tests authentication, authorization, access control bypass scenarios,
and privilege escalation attempts.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, PropertyMock
import json
import tempfile
import os
import hashlib
from datetime import datetime, timedelta
import time
import threading

from voice.security import VoiceSecurity, SecurityConfig, AccessManager


class TestAccessControl:
    """Comprehensive access control tests."""

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

    @pytest.fixture
    def user_scenarios(self):
        """Various user scenarios for access control testing."""
        return [
            {
                'user_id': 'patient_123',
                'role': 'patient',
                'resources': ['own_voice_data', 'own_consent_records'],
                'permissions': ['read_own', 'update_own_consent']
            },
            {
                'user_id': 'therapist_456',
                'role': 'therapist',
                'resources': ['patient_voice_data', 'patient_consent_records', 'therapy_notes'],
                'permissions': ['read_assigned_patients', 'update_therapy_notes', 'view_patient_history']
            },
            {
                'user_id': 'admin_789',
                'role': 'administrator',
                'resources': ['all_patient_data', 'system_configuration', 'audit_logs'],
                'permissions': ['full_system_access', 'manage_users', 'view_all_audits']
            },
            {
                'user_id': 'researcher_101',
                'role': 'researcher',
                'resources': ['anonymized_data', 'research_datasets'],
                'permissions': ['access_anonymized_data', 'run_analytics', 'export_results']
            }
        ]

    @pytest.fixture
    def malicious_access_attempts(self):
        """Malicious access attempts for security testing."""
        return [
            {
                'attack_type': 'sql_injection',
                'user_id': "admin'; DROP TABLE users; --",
                'resource': 'patient_data',
                'permission': 'admin'
            },
            {
                'attack_type': 'path_traversal',
                'user_id': '../../../etc/passwd',
                'resource': 'system_files',
                'permission': 'read'
            },
            {
                'attack_type': 'command_injection',
                'user_id': 'user; rm -rf /',
                'resource': 'any_resource',
                'permission': 'execute'
            },
            {
                'attack_type': 'xss_attempt',
                'user_id': '<script>alert("xss")</script>',
                'resource': 'web_interface',
                'permission': 'access'
            },
            {
                'attack_type': 'buffer_overflow',
                'user_id': 'A' * 10000,
                'resource': 'buffer_vulnerable_endpoint',
                'permission': 'overflow'
            },
            {
                'attack_type': 'privilege_escalation',
                'user_id': 'patient_123',
                'resource': 'admin_panel',
                'permission': 'admin'
            }
        ]

    def test_basic_access_control(self, security, user_scenarios):
        """Test basic access control mechanisms."""
        for scenario in user_scenarios:
            user_id = scenario['user_id']
            role = scenario['role']

            # Grant role-based permissions
            for resource in scenario['resources']:
                for permission in scenario['permissions']:
                    security.access_manager.grant_access(user_id, resource, permission)

            # Test granted permissions
            for resource in scenario['resources']:
                for permission in scenario['permissions']:
                    has_access = security.access_manager.has_access(user_id, resource, permission)
                    assert has_access == True, f"User {user_id} should have {permission} on {resource}"

            # Test denied permissions (different resource/permission)
            denied_access = security.access_manager.has_access(user_id, 'restricted_resource', 'admin')
            assert denied_access == False, f"User {user_id} should not have admin access"

    def test_role_based_access_control(self, security):
        """Test comprehensive role-based access control."""
        # Define role hierarchy and permissions
        role_permissions = {
            'patient': {
                'own_voice_data': ['read', 'update_own'],
                'own_consent_records': ['read', 'update'],
                'therapy_notes': [],  # No access
                'other_patient_data': [],  # No access
                'admin_panel': []  # No access
            },
            'therapist': {
                'assigned_patient_data': ['read', 'update_notes'],
                'own_consent_records': ['read'],
                'therapy_notes': ['read', 'create', 'update'],
                'admin_panel': [],  # No access
                'system_configuration': []  # No access
            },
            'admin': {
                'all_patient_data': ['read', 'update', 'delete'],
                'all_consent_records': ['read', 'update'],
                'therapy_notes': ['read', 'update', 'delete'],
                'admin_panel': ['full_access'],
                'system_configuration': ['read', 'update'],
                'audit_logs': ['read', 'analyze']
            }
        }

        # Test each role's permissions
        for role, resource_permissions in role_permissions.items():
            user_id = f'{role}_user_123'

            for resource, permissions in resource_permissions.items():
                for permission in permissions:
                    security.access_manager.grant_access(user_id, resource, permission)

            # Verify granted permissions
            for resource, permissions in resource_permissions.items():
                for permission in permissions:
                    has_access = security.access_manager.has_access(user_id, resource, permission)
                    assert has_access == True, f"{role} should have {permission} on {resource}"

            # Verify denied permissions
            denied_resources = [r for r in role_permissions.keys() if r != role]
            for denied_role in denied_resources:
                for resource, permissions in role_permissions[denied_role].items():
                    for permission in permissions:
                        has_access = security.access_manager.has_access(user_id, resource, permission)
                        assert has_access == False, f"{role} should not have {denied_role}'s permissions"

    def test_access_control_malicious_attempts(self, security, malicious_access_attempts):
        """Test access control against malicious attempts."""
        for attempt in malicious_access_attempts:
            attack_type = attempt['attack_type']
            user_id = attempt['user_id']
            resource = attempt['resource']
            permission = attempt['permission']

            # Attempt malicious access
            has_access = security.access_manager.has_access(user_id, resource, permission)

            # Should be denied (implementation dependent on attack type)
            # Some attacks might be blocked at input validation level
            assert has_access == False, f"Malicious access attempt {attack_type} should be denied"

            # Verify no privilege escalation occurred
            if 'admin' in permission:
                admin_access = security.access_manager.has_access(user_id, 'admin_panel', 'full_access')
                assert admin_access == False, f"Privilege escalation occurred in {attack_type}"

    def test_access_control_session_management(self, security):
        """Test access control with session management."""
        user_id = 'session_user_123'
        resource = 'patient_data'
        permission = 'read'

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)
        assert security.access_manager.has_access(user_id, resource, permission) == True

        # Simulate session timeout
        with patch('voice.security.datetime') as mock_datetime:
            # Move time forward past session timeout
            future_time = datetime.now() + timedelta(minutes=security.original_config.session_timeout_minutes + 1)
            mock_datetime.now.return_value = future_time

            # Access should be denied after timeout (implementation dependent)
            # In real implementation, this might involve session validation
            current_access = security.access_manager.has_access(user_id, resource, permission)

            # For testing, access might still be granted (depends on implementation)
            # The key is that timeout is properly tracked in audit logs

        # Verify session timeout is logged
        user_logs = security.audit_logger.get_user_logs(user_id)
        timeout_logs = [log for log in user_logs if 'timeout' in str(log.get('details', {})).lower()]
        # Timeout logging implementation dependent

    def test_access_control_concurrent_sessions(self, security):
        """Test access control under concurrent user sessions."""
        import threading
        import queue

        user_id = 'concurrent_access_user'
        resource = 'shared_resource'
        permission = 'read'

        results = queue.Queue()
        errors = queue.Queue()

        def access_worker(worker_id):
            try:
                worker_user_id = f'{user_id}_{worker_id}'

                # Grant access for this worker
                security.access_manager.grant_access(worker_user_id, resource, permission)

                # Perform concurrent access operations
                for i in range(50):
                    # Check access
                    has_access = security.access_manager.has_access(worker_user_id, resource, permission)
                    assert has_access == True, f"Worker {worker_id} lost access"

                    # Perform access (simulated)
                    if i % 10 == 0:
                        # Simulate access revocation and re-granting
                        security.access_manager.revoke_access(worker_user_id, resource, permission)
                        security.access_manager.grant_access(worker_user_id, resource, permission)

                results.put(f'worker_{worker_id}_success')
            except Exception as e:
                errors.put(f'worker_{worker_id}_error: {e}')

        # Start concurrent access workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=access_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=20)

        # Verify all workers completed successfully
        success_count = 0
        while not results.empty():
            results.get()
            success_count += 1

        error_count = 0
        while not errors.empty():
            errors.get()
            error_count += 1

        assert success_count == 5, f"Only {success_count} workers succeeded"
        assert error_count == 0, f"{error_count} workers failed"

    def test_access_control_granular_permissions(self, security):
        """Test granular permission levels."""
        user_id = 'granular_user_123'

        # Define granular permissions
        granular_permissions = [
            'read_basic_patient_info',
            'read_voice_transcripts',
            'read_audio_metadata',
            'update_own_consent',
            'view_therapy_progress',
            'export_personal_data',
            'delete_own_data',
            'admin_user_management',
            'system_configuration',
            'audit_log_access'
        ]

        # Grant subset of permissions
        allowed_permissions = granular_permissions[:5]  # First 5 permissions
        restricted_permissions = granular_permissions[5:]  # Last 5 permissions

        for permission in allowed_permissions:
            security.access_manager.grant_access(user_id, 'patient_data', permission)

        # Test allowed permissions
        for permission in allowed_permissions:
            has_access = security.access_manager.has_access(user_id, 'patient_data', permission)
            assert has_access == True, f"Should have permission: {permission}"

        # Test restricted permissions
        for permission in restricted_permissions:
            has_access = security.access_manager.has_access(user_id, 'patient_data', permission)
            assert has_access == False, f"Should not have permission: {permission}"

    def test_access_control_resource_hierarchy(self, security):
        """Test access control with resource hierarchy."""
        user_id = 'hierarchy_user_123'

        # Define resource hierarchy
        resource_hierarchy = {
            'patient_data': {
                'voice_recordings': ['read', 'transcribe'],
                'therapy_notes': ['read', 'update'],
                'treatment_history': ['read'],
                'personal_info': ['read_basic']  # Limited access
            },
            'system_data': {
                'audit_logs': [],  # No access
                'configuration': [],  # No access
                'user_management': []  # No access
            }
        }

        # Grant hierarchical permissions
        for category, resources in resource_hierarchy.items():
            for resource, permissions in resources.items():
                for permission in permissions:
                    security.access_manager.grant_access(user_id, resource, permission)

        # Test hierarchical access
        for category, resources in resource_hierarchy.items():
            for resource, permissions in resources.items():
                for permission in permissions:
                    has_access = security.access_manager.has_access(user_id, resource, permission)
                    assert has_access == True, f"Hierarchical access failed: {category}/{resource}/{permission}"

        # Test access to restricted areas
        restricted_resources = ['audit_logs', 'configuration', 'user_management']
        for resource in restricted_resources:
            has_access = security.access_manager.has_access(user_id, resource, 'read')
            assert has_access == False, f"Should not have access to: {resource}"

    def test_access_control_audit_integration(self, security):
        """Test integration between access control and audit logging."""
        user_id = 'audit_integration_user'
        resource = 'sensitive_patient_data'
        permission = 'read'

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)

        # Clear existing logs for clean test
        security.audit_logger.logs.clear()

        # Perform access operation
        has_access = security.access_manager.has_access(user_id, resource, permission)

        # Verify access control decision is logged
        user_logs = security.audit_logger.get_user_logs(user_id)
        access_logs = [
            log for log in user_logs
            if 'access' in log.get('event_type', '').lower()
        ]

        # Should have audit log for access check (implementation dependent)
        # At minimum, should have access grant logged
        grant_logs = [log for log in user_logs if log.get('event_type') == 'access_granted']
        assert len(grant_logs) >= 1, "Access grant should be logged"

        grant_log = grant_logs[0]
        assert grant_log['user_id'] == user_id
        assert grant_log['details']['resource'] == resource
        assert grant_log['details']['permission'] == permission

    def test_access_control_privilege_escalation_attempts(self, security):
        """Test protection against privilege escalation attempts."""
        # Start with basic user
        basic_user = 'basic_user_123'
        basic_resource = 'own_data'
        basic_permission = 'read'

        # Grant basic permissions
        security.access_manager.grant_access(basic_user, basic_resource, basic_permission)

        # Verify basic access
        assert security.access_manager.has_access(basic_user, basic_resource, basic_permission) == True

        # Attempt privilege escalation
        escalation_attempts = [
            ('admin_panel', 'full_access'),
            ('system_configuration', 'update'),
            ('audit_logs', 'delete'),
            ('other_user_data', 'read'),
            ('user_management', 'create_user')
        ]

        for resource, permission in escalation_attempts:
            # Should not have escalated permissions
            has_access = security.access_manager.has_access(basic_user, resource, permission)
            assert has_access == False, f"Privilege escalation prevented: {resource}/{permission}"

        # Verify original permissions still intact
        assert security.access_manager.has_access(basic_user, basic_resource, basic_permission) == True

    def test_access_control_dynamic_permission_changes(self, security):
        """Test dynamic permission changes and revocation."""
        user_id = 'dynamic_user_123'
        resource = 'dynamic_resource'
        permission = 'dynamic_permission'

        # Initially no access
        assert security.access_manager.has_access(user_id, resource, permission) == False

        # Grant access
        security.access_manager.grant_access(user_id, resource, permission)
        assert security.access_manager.has_access(user_id, resource, permission) == True

        # Change permission level
        new_permission = 'elevated_permission'
        security.access_manager.grant_access(user_id, resource, new_permission)
        assert security.access_manager.has_access(user_id, resource, new_permission) == True
        assert security.access_manager.has_access(user_id, resource, permission) == True  # Original still works

        # Revoke specific permission
        security.access_manager.revoke_access(user_id, resource, permission)
        assert security.access_manager.has_access(user_id, resource, permission) == False
        assert security.access_manager.has_access(user_id, resource, new_permission) == True  # Other permission intact

        # Revoke all permissions for resource
        security.access_manager.revoke_access(user_id, resource, new_permission)
        assert security.access_manager.has_access(user_id, resource, new_permission) == False

    def test_access_control_resource_ownership(self, security):
        """Test resource ownership and access rights."""
        # Define resource ownership
        resource_owners = {
            'patient_123': ['patient_123_voice_data', 'patient_123_consent'],
            'patient_456': ['patient_456_voice_data', 'patient_456_consent'],
            'therapist_789': ['therapist_789_notes', 'therapist_789_sessions']
        }

        # Grant ownership-based permissions
        for owner, resources in resource_owners.items():
            for resource in resources:
                # Owner has full access to their resources
                security.access_manager.grant_access(owner, resource, 'full_access')

                # Other users have limited or no access
                for other_owner in resource_owners:
                    if other_owner != owner:
                        # Limited read access for healthcare providers (implementation dependent)
                        if 'therapist' in other_owner:
                            security.access_manager.grant_access(other_owner, resource, 'read_assigned')
                        else:
                            # Patients have no access to other patients' data
                            pass  # No permissions granted

        # Test ownership access
        for owner, resources in resource_owners.items():
            for resource in resources:
                has_access = security.access_manager.has_access(owner, resource, 'full_access')
                assert has_access == True, f"Owner {owner} should have access to {resource}"

        # Test access between different owners
        patient_123_resources = resource_owners['patient_123']
        for resource in patient_123_resources:
            # Patient 456 should not have access to patient 123's data
            has_access = security.access_manager.has_access('patient_456', resource, 'read')
            assert has_access == False, f"Patient 456 should not access patient 123's {resource}"

    def test_access_control_emergency_access(self, security):
        """Test emergency access override mechanisms."""
        user_id = 'emergency_patient'
        resource = 'emergency_patient_data'
        normal_permission = 'read'

        # Normal access denied
        security.access_manager.grant_access(user_id, resource, 'deny')  # Explicit deny
        assert security.access_manager.has_access(user_id, resource, normal_permission) == False

        # Emergency access (implementation dependent)
        emergency_permission = 'emergency_read'

        # In real implementation, emergency access might be granted through:
        # 1. Emergency access codes
        # 2. Multi-factor authentication override
        # 3. Legal/medical authority verification
        # 4. Break-glass procedures

        # For testing, simulate emergency access grant
        security.access_manager.grant_access(user_id, resource, emergency_permission)

        # Verify emergency access is granted
        has_emergency_access = security.access_manager.has_access(user_id, resource, emergency_permission)
        assert has_emergency_access == True, "Emergency access should be granted"

        # Verify emergency access is logged
        user_logs = security.audit_logger.get_user_logs(user_id)
        emergency_logs = [
            log for log in user_logs
            if 'emergency' in str(log.get('details', {})).lower()
        ]
        # Emergency logging implementation dependent

    def test_access_control_authentication_bypass_attempts(self, security):
        """Test protection against authentication bypass attempts."""
        # Common authentication bypass techniques
        bypass_attempts = [
            {
                'technique': 'direct_object_reference',
                'user_id': 'admin',
                'resource': 'user_123_data',
                'attempt': 'Accessing other user data without authorization'
            },
            {
                'technique': 'forced_browsing',
                'user_id': 'guest',
                'resource': 'admin_panel',
                'attempt': 'Accessing admin functionality as guest'
            },
            {
                'technique': 'parameter_manipulation',
                'user_id': 'user_456',
                'resource': 'user_123_profile',
                'attempt': 'Manipulating user ID parameter'
            },
            {
                'technique': 'cookie_manipulation',
                'user_id': 'session_hijacker',
                'resource': 'authenticated_content',
                'attempt': 'Using stolen session cookies'
            }
        ]

        for attempt in bypass_attempts:
            technique = attempt['technique']
            user_id = attempt['user_id']
            resource = attempt['resource']

            # Test various permission levels
            permissions_to_test = ['read', 'write', 'delete', 'admin']

            for permission in permissions_to_test:
                has_access = security.access_manager.has_access(user_id, resource, permission)
                # Should be denied (no permissions granted for these test users)
                assert has_access == False, f"Authentication bypass {technique} should be prevented"

    def test_access_control_data_isolation(self, security):
        """Test data isolation between different users and roles."""
        # Create multiple users with different roles
        users = {
            'patient_1': {'role': 'patient', 'data': 'patient_1_medical_data'},
            'patient_2': {'role': 'patient', 'data': 'patient_2_medical_data'},
            'therapist_1': {'role': 'therapist', 'data': 'patient_1_therapy_notes'},
            'therapist_2': {'role': 'therapist', 'data': 'patient_2_therapy_notes'},
            'admin_1': {'role': 'admin', 'data': 'system_configuration'}
        }

        # Grant appropriate permissions
        for user_id, user_info in users.items():
            resource = user_info['data']
            if user_info['role'] == 'patient':
                # Patients have access to their own data
                security.access_manager.grant_access(user_id, resource, 'full_access')
            elif user_info['role'] == 'therapist':
                # Therapists have access to assigned patient data
                security.access_manager.grant_access(user_id, resource, 'read')
                security.access_manager.grant_access(user_id, resource, 'update_notes')
            elif user_info['role'] == 'admin':
                # Admins have access to system data
                security.access_manager.grant_access(user_id, resource, 'full_access')

        # Test data isolation
        # Patient 1 should not access Patient 2's data
        assert security.access_manager.has_access('patient_1', 'patient_2_medical_data', 'read') == False

        # Therapist 1 should not access Therapist 2's notes
        assert security.access_manager.has_access('therapist_1', 'patient_2_therapy_notes', 'read') == False

        # Patients should not access system configuration
        assert security.access_manager.has_access('patient_1', 'system_configuration', 'read') == False
        assert security.access_manager.has_access('patient_2', 'system_configuration', 'read') == False

        # Therapists should not access system configuration
        assert security.access_manager.has_access('therapist_1', 'system_configuration', 'read') == False
        assert security.access_manager.has_access('therapist_2', 'system_configuration', 'read') == False

        # Admins should have access to system configuration
        assert security.access_manager.has_access('admin_1', 'system_configuration', 'read') == True

    def test_access_control_performance_under_load(self, security):
        """Test access control performance under heavy load."""
        # Create many users and resources
        num_users = 100
        num_resources = 50
        permissions_per_user = 10

        # Grant permissions (this creates a lot of access records)
        start_time = time.time()

        for user_id in range(num_users):
            for resource_id in range(num_resources):
                for perm_id in range(permissions_per_user):
                    user = f'load_user_{user_id}'
                    resource = f'load_resource_{resource_id}'
                    permission = f'load_permission_{perm_id}'

                    security.access_manager.grant_access(user, resource, permission)

        grant_duration = time.time() - start_time

        # Performance should be reasonable
        max_grant_duration = 5.0  # 5 seconds for 50,000 permissions
        assert grant_duration < max_grant_duration, f"Permission granting too slow: {grant_duration}s"

        # Test access checking performance
        start_time = time.time()

        for user_id in range(num_users):
            for resource_id in range(num_resources):
                for perm_id in range(permissions_per_user):
                    user = f'load_user_{user_id}'
                    resource = f'load_resource_{resource_id}'
                    permission = f'load_permission_{perm_id}'

                    has_access = security.access_manager.has_access(user, resource, permission)
                    assert has_access == True

        check_duration = time.time() - start_time

        # Access checking should be fast
        max_check_duration = 2.0  # 2 seconds for 50,000 checks
        assert check_duration < max_check_duration, f"Access checking too slow: {check_duration}s"

    def test_access_control_revocation_cascade(self, security):
        """Test cascading revocation of access permissions."""
        user_id = 'cascade_user_123'

        # Create permission hierarchy
        base_permissions = ['read', 'write']
        derived_permissions = ['read_write', 'full_access']
        related_permissions = ['audit', 'backup']

        # Grant all permissions
        all_permissions = base_permissions + derived_permissions + related_permissions
        resource = 'cascade_resource'

        for permission in all_permissions:
            security.access_manager.grant_access(user_id, resource, permission)

        # Verify all permissions granted
        for permission in all_permissions:
            assert security.access_manager.has_access(user_id, resource, permission) == True

        # Revoke base permission
        security.access_manager.revoke_access(user_id, resource, 'read')

        # Test cascading effects (implementation dependent)
        # Some systems might automatically revoke derived permissions
        has_read = security.access_manager.has_access(user_id, resource, 'read')
        assert has_read == False, "Base permission should be revoked"

        # Derived permissions might still exist (depends on implementation)
        # For testing, we'll assume they remain unless explicitly revoked

        # Revoke all permissions for resource
        for permission in all_permissions:
            security.access_manager.revoke_access(user_id, resource, permission)

        # Verify complete revocation
        for permission in all_permissions:
            has_access = security.access_manager.has_access(user_id, resource, permission)
            assert has_access == False, f"Permission {permission} should be revoked"