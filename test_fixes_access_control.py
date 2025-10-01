#!/usr/bin/env python3
"""
Fix for access control test failures.

The main issue is in test_role_based_access_control where the test logic
doesn't properly isolate user permissions by role.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice'))

def test_access_control_fixed():
    """Fixed version of the role-based access control test."""
    from voice.security import VoiceSecurity

    print("Testing fixed access control logic...")

    # Initialize security
    security = VoiceSecurity()

    # Define role permissions (same as original test)
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

    # Create separate users for each role
    users = {}
    for role in role_permissions.keys():
        users[role] = f'{role}_user_123'

    # Grant permissions for each role to their respective user
    for role, user_id in users.items():
        resource_permissions = role_permissions[role]

        for resource, permissions in resource_permissions.items():
            for permission in permissions:
                if permission:  # Only grant if permission is not empty
                    print(f"Granting {user_id} access to {resource} with permission {permission}")
                    security.access_manager.grant_access(user_id, resource, permission)

    # Test each role's permissions
    for role, user_id in users.items():
        resource_permissions = role_permissions[role]

        # Verify granted permissions
        for resource, permissions in resource_permissions.items():
            for permission in permissions:
                if permission:  # Only check if permission is not empty
                    has_access = security.access_manager.has_access(user_id, resource, permission)
                    assert has_access == True, f"{role} should have {permission} on {resource}"
                    print(f"✓ {user_id} correctly has {permission} on {resource}")

        # Verify denied permissions from other roles
        denied_resources = [r for r in role_permissions.keys() if r != role]
        for denied_role in denied_resources:
            for resource, permissions in role_permissions[denied_role].items():
                for permission in permissions:
                    if permission:  # Only check if permission is not empty
                        has_access = security.access_manager.has_access(user_id, resource, permission)
                        assert has_access == False, f"{role} should not have {denied_role}'s permission {permission} on {resource}"
                        print(f"✓ {user_id} correctly denied {permission} on {resource} (belongs to {denied_role})")

    print("✅ Access control test passed!")
    return True

if __name__ == "__main__":
    try:
        test_access_control_fixed()
        print("Access control fix verified successfully!")
    except Exception as e:
        print(f"❌ Access control test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)