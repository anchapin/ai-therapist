#!/usr/bin/env python3
"""
Test security module compatibility with pytest requirements.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_security_imports():
    """Test security module imports work correctly."""
    print("=== Testing Security Module Imports ===")

    try:
        print("1. Importing voice.config...")
        from voice.config import VoiceConfig
        print("   ✓ VoiceConfig imported successfully")

        print("2. Creating VoiceConfig instance...")
        config = VoiceConfig()
        print(f"   ✓ VoiceConfig created with encryption={config.security.encryption_enabled}")

        print("3. Importing voice.security components...")
        from voice.security import VoiceSecurity, SecurityConfig, AuditLogger, ConsentManager
        print("   ✓ All security components imported successfully")

        print("4. Creating VoiceSecurity instance...")
        security = VoiceSecurity(config)
        print(f"   ✓ VoiceSecurity created successfully")

        print("5. Testing security properties...")
        assert hasattr(security, 'config')
        assert hasattr(security, 'encryption_enabled')
        assert hasattr(security, 'consent_required')
        assert hasattr(security, 'privacy_mode')
        assert hasattr(security, 'audit_logging_enabled')
        assert hasattr(security, 'data_retention_days')
        print("   ✓ All required security properties present")

        print("6. Testing security functionality...")
        # Test encryption
        test_data = b"test_security_data"
        user_id = "test_user_123"

        encrypted = security.encrypt_data(test_data, user_id)
        assert encrypted != test_data
        print("   ✓ Data encryption works")

        decrypted = security.decrypt_data(encrypted, user_id)
        assert decrypted == test_data
        print("   ✓ Data decryption works")

        # Test audit logging
        log_entry = security.audit_logger.log_event(
            event_type="TEST_EVENT",
            user_id=user_id,
            details={"test": True}
        )
        assert log_entry is not None
        assert 'event_id' in log_entry
        assert 'timestamp' in log_entry
        print("   ✓ Audit logging works")

        # Test consent management
        consent_record = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type="TEST_CONSENT",
            granted=True
        )
        assert consent_record is not None
        assert consent_record['granted'] == True

        has_consent = security.consent_manager.has_consent(user_id, "TEST_CONSENT")
        assert has_consent == True
        print("   ✓ Consent management works")

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytest_fixture_compatibility():
    """Test compatibility with pytest fixtures from the test file."""
    print("\n=== Testing Pytest Fixture Compatibility ===")

    try:
        from voice.config import VoiceConfig
        from voice.security import VoiceSecurity

        # Simulate pytest fixture setup from test_security_compliance.py
        config = VoiceConfig()
        config.encryption_enabled = True
        config.consent_required = True
        config.privacy_mode = True
        config.audit_logging_enabled = True
        config.data_retention_days = 30

        print("1. Creating test configuration...")
        # Simulate the config fixture
        assert config.encryption_enabled == True
        assert config.consent_required == True
        assert config.privacy_mode == True
        assert config.audit_logging_enabled == True
        assert config.data_retention_days == 30
        print("   ✓ Test configuration matches pytest fixture requirements")

        print("2. Creating security instance...")
        # Simulate the security fixture
        security = VoiceSecurity(config)
        print("   ✓ Security instance created")

        print("3. Testing security initialization assertions...")
        # These are the exact assertions from test_security_initialization
        assert security.config == config
        assert security.encryption_enabled == config.encryption_enabled
        assert security.consent_required == config.consent_required
        assert security.privacy_mode == config.privacy_mode
        assert security.audit_logging_enabled == config.audit_logging_enabled
        print("   ✓ All security initialization assertions pass")

        print("4. Testing audit logging functionality assertions...")
        # Test audit logging functionality assertions
        event_type = "VOICE_INPUT"
        session_id = "test_session_123"
        user_id = "test_user_456"
        details = {"duration": 5.2, "provider": "openai"}

        log_entry = security.audit_logger.log_event(
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            details=details
        )

        assert log_entry['event_type'] == event_type
        assert log_entry['session_id'] == session_id
        assert log_entry['user_id'] == user_id
        assert log_entry['details'] == details
        assert 'timestamp' in log_entry
        assert 'event_id' in log_entry
        print("   ✓ All audit logging assertions pass")

        print("5. Testing audit log retrieval...")
        # Create test logs for retrieval test
        for i in range(5):
            security.audit_logger.log_event(
                event_type="VOICE_INPUT",
                session_id=session_id,
                user_id="test_user",
                details={"iteration": i}
            )

        # Retrieve logs
        logs = security.audit_logger.get_session_logs(session_id)
        assert len(logs) == 5
        print("   ✓ Audit log retrieval works")

        print("6. Testing consent management assertions...")
        # Test consent management assertions
        user_id = "test_user_789"
        consent_type = "VOICE_DATA_PROCESSING"

        # Test consent recording
        consent_record = security.consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version="1.0"
        )

        assert consent_record['user_id'] == user_id
        assert consent_record['consent_type'] == consent_type
        assert consent_record['granted'] == True
        assert consent_record['version'] == "1.0"
        print("   ✓ Consent recording assertions pass")

        # Test consent verification
        has_consent = security.consent_manager.has_consent(user_id, consent_type)
        assert has_consent == True
        print("   ✓ Consent verification works")

        # Test consent withdrawal
        security.consent_manager.withdraw_consent(user_id, consent_type)
        has_consent = security.consent_manager.has_consent(user_id, consent_type)
        assert has_consent == False
        print("   ✓ Consent withdrawal works")

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encryption_functionality():
    """Test encryption functionality with test data."""
    print("\n=== Testing Encryption Functionality ===")

    try:
        from voice.config import VoiceConfig
        from voice.security import VoiceSecurity

        config = VoiceConfig()
        security = VoiceSecurity(config)

        print("1. Testing basic data encryption...")
        test_data = b"sensitive_voice_data"
        user_id = "test_user_123"

        # Encrypt data
        encrypted_data = security.encrypt_data(test_data, user_id)
        assert encrypted_data != test_data
        assert isinstance(encrypted_data, bytes)
        print("   ✓ Basic data encryption works")

        # Decrypt data
        decrypted_data = security.decrypt_data(encrypted_data, user_id)
        assert decrypted_data == test_data
        print("   ✓ Basic data decryption works")

        print("2. Testing user isolation...")
        # Test with different user (should fail)
        try:
            security.decrypt_data(encrypted_data, "different_user")
            print("   ✗ User isolation failed - should not allow decryption by different user")
            return False
        except Exception:
            print("   ✓ User isolation works - different user cannot decrypt")

        print("3. Testing audio data encryption...")
        # Test encrypted audio data fixture simulation
        import cryptography.fernet
        key = cryptography.fernet.Fernet.generate_key()
        cipher = cryptography.fernet.Fernet(key)
        test_audio_data = b'test_audio_data'
        encrypted_audio_data = cipher.encrypt(test_audio_data)

        user_id = "test_user_456"
        encrypted_audio = security.encrypt_audio_data(encrypted_audio_data, user_id)
        assert encrypted_audio != encrypted_audio_data
        print("   ✓ Audio data encryption works")

        # Decrypt audio
        decrypted_audio = security.decrypt_audio_data(encrypted_audio, user_id)
        assert decrypted_audio == encrypted_audio_data
        print("   ✓ Audio data decryption works")

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests."""
    print("=== Security Module Compatibility Test ===")
    print("Testing compatibility with pytest requirements from test_security_compliance.py\n")

    tests = [
        test_security_imports,
        test_pytest_fixture_compatibility,
        test_encryption_functionality
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test {test.__name__} failed!")

    print(f"\n{'='*60}")
    print(f"Compatibility Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All compatibility tests passed!")
        print("The security module should work with pytest security tests.")
        return 0
    else:
        print("❌ Some compatibility tests failed!")
        print("The security module needs fixes to work with pytest.")
        return 1

if __name__ == "__main__":
    sys.exit(main())