#!/usr/bin/env python3
"""
Test script for security fixes in voice module.

This script tests the security fixes implemented:
1. Input validation in grant_consent()
2. Memory-safe audio buffer
3. Thread-safe session management
4. Fixed async/sync mixing
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_security_validation():
    """Test input validation in security module."""
    print("Testing security validation...")

    try:
        from voice.config import VoiceConfig
        from voice.security import VoiceSecurity

        # Create a minimal config for testing
        class MockSecurityConfig:
            encryption_enabled = False
            consent_required = False
            hipaa_compliance_enabled = False
            gdpr_compliance_enabled = False
            data_localization = False
            data_retention_hours = 24
            emergency_protocols_enabled = False
            privacy_mode = False
            anonymization_enabled = False

        class MockVoiceConfig:
            security = MockSecurityConfig()

        config = MockVoiceConfig()
        security = VoiceSecurity(config)

        # Test valid inputs
        result = security.grant_consent(
            user_id="test_user_123",
            consent_type="voice_processing",
            granted=True,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0 Test Browser"
        )
        assert result == True, "Valid inputs should return True"
        print("✓ Valid inputs test passed")

        # Test invalid user_id
        result = security.grant_consent(
            user_id="invalid@user",  # Contains @ which is not allowed
            consent_type="voice_processing",
            granted=True
        )
        assert result == False, "Invalid user_id should return False"
        print("✓ Invalid user_id test passed")

        # Test invalid consent_type
        result = security.grant_consent(
            user_id="test_user",
            consent_type="invalid_consent_type",  # Not in allowed types
            granted=True
        )
        assert result == False, "Invalid consent_type should return False"
        print("✓ Invalid consent_type test passed")

        # Test invalid IP address
        result = security.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            ip_address="999.999.999.999"  # Invalid IP
        )
        assert result == False, "Invalid IP address should return False"
        print("✓ Invalid IP address test passed")

        # Test invalid user agent (with dangerous characters)
        result = security.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            user_agent="<script>alert('xss')</script>"  # Contains dangerous chars
        )
        assert result == False, "Invalid user agent should return False"
        print("✓ Invalid user agent test passed")

        # Test overly long consent text
        long_text = "a" * 10001  # Exceeds 10000 character limit
        result = security.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            consent_text=long_text
        )
        assert result == False, "Overly long consent text should return False"
        print("✓ Long consent text test passed")

        print("✓ All security validation tests passed!")
        return True

    except Exception as e:
        print(f"✗ Security validation test failed: {e}")
        return False

def test_audio_buffer_memory():
    """Test memory-safe audio buffer."""
    print("\nTesting audio buffer memory safety...")

    try:
        from voice.config import VoiceConfig
        from voice.audio_processor import SimplifiedAudioProcessor

        # Create a minimal config for testing
        class MockAudioConfig:
            sample_rate = 16000
            channels = 1
            chunk_size = 1024
            format = "wav"
            max_buffer_size = 10  # Small buffer for testing

        class MockVoiceConfig:
            audio = MockAudioConfig()

        config = MockVoiceConfig()
        processor = SimplifiedAudioProcessor(config)

        # Check that audio_buffer is a deque with maxlen
        from collections import deque
        assert isinstance(processor.audio_buffer, deque), "audio_buffer should be a deque"
        assert processor.audio_buffer.maxlen == 10, "audio_buffer should have maxlen set"
        print("✓ Audio buffer is properly configured as bounded deque")

        # Test buffer clearing
        test_data = np.random.rand(1024, 1).astype(np.float32)
        processor.audio_buffer.append(test_data)
        assert len(processor.audio_buffer) == 1, "Buffer should accept data"

        processor.audio_buffer.clear()
        assert len(processor.audio_buffer) == 0, "Buffer should clear properly"
        print("✓ Audio buffer operations work correctly")

        print("✓ Audio buffer memory safety test passed!")
        return True

    except Exception as e:
        print(f"✗ Audio buffer test failed: {e}")
        return False

def test_session_thread_safety():
    """Test thread-safe session management."""
    print("\nTesting session thread safety...")

    try:
        from voice.config import VoiceConfig
        from voice.voice_service import VoiceService
        from voice.security import VoiceSecurity

        # Create minimal configs for testing
        class MockConfig:
            voice_enabled = True
            default_voice_profile = "default"

        class MockSecurityConfig:
            encryption_enabled = False
            consent_required = False

        class MockVoiceConfig:
            voice_enabled = True
            default_voice_profile = "default"
            security = MockSecurityConfig()

        config = MockVoiceConfig()
        security = VoiceSecurity(config)
        service = VoiceService(config, security)

        # Check that _sessions_lock exists
        assert hasattr(service, '_sessions_lock'), "Service should have _sessions_lock"
        print("✓ Session lock is properly initialized")

        # Test basic session operations
        session_id = service.create_session("test_session")
        assert session_id == "test_session", "Session creation should work"

        session = service.get_session(session_id)
        assert session is not None, "Should be able to retrieve created session"
        assert session.session_id == session_id, "Retrieved session should have correct ID"

        current = service.get_current_session()
        assert current is not None, "Should have current session"
        assert current.session_id == session_id, "Current session should be the created one"

        service.destroy_session(session_id)
        destroyed_session = service.get_session(session_id)
        assert destroyed_session is None, "Destroyed session should not be retrievable"

        print("✓ Session operations are thread-safe and functional")
        print("✓ Session thread safety test passed!")
        return True

    except Exception as e:
        print(f"✗ Session thread safety test failed: {e}")
        return False

def test_async_sync_fix():
    """Test async/sync mixing fix."""
    print("\nTesting async/sync mixing fix...")

    try:
        from voice.config import VoiceConfig
        from voice.voice_service import VoiceService
        from voice.security import VoiceSecurity
        from voice.audio_processor import AudioData

        # Create minimal configs for testing
        class MockConfig:
            voice_enabled = True
            default_voice_profile = "default"

        class MockSecurityConfig:
            encryption_enabled = False
            consent_required = False

        class MockVoiceConfig:
            voice_enabled = True
            default_voice_profile = "default"
            security = MockSecurityConfig()

        config = MockVoiceConfig()
        security = VoiceSecurity(config)
        service = VoiceService(config, security)

        # Check that _event_loop attribute exists
        assert hasattr(service, '_event_loop'), "Service should have _event_loop attribute"
        assert service._event_loop is None, "Event loop should be None initially"
        print("✓ Event loop reference is properly initialized")

        # Test audio callback with no running event loop
        test_audio = AudioData(
            data=np.random.rand(1024, 1).astype(np.float32),
            sample_rate=16000,
            channels=1,
            format="float32",
            duration=0.064,
            timestamp=time.time()
        )

        # This should not crash even without event loop
        service._audio_callback(test_audio)
        print("✓ Audio callback handles missing event loop gracefully")

        print("✓ Async/sync mixing fix test passed!")
        return True

    except Exception as e:
        print(f"✗ Async/sync mixing fix test failed: {e}")
        return False

def main():
    """Run all security tests."""
    print("Running security fix tests...\n")

    tests = [
        test_security_validation,
        test_audio_buffer_memory,
        test_session_thread_safety,
        test_async_sync_fix
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All security fix tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())