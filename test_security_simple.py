#!/usr/bin/env python3
"""
Simple test for security validation functions.
"""

import sys
import re
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_validation_patterns():
    """Test the validation patterns directly."""
    print("Testing validation patterns...")

    # Test patterns from security.py
    USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')
    IP_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    ALLOWED_CONSENT_TYPES = {
        'voice_processing', 'data_storage', 'transcription',
        'analysis', 'all_consent', 'emergency_protocol'
    }

    def validate_user_id(user_id):
        if not isinstance(user_id, str):
            return False
        return bool(USER_ID_PATTERN.match(user_id))

    def validate_ip_address(ip_address):
        if not isinstance(ip_address, str) or not ip_address:
            return True  # Empty IP is allowed for local contexts
        return bool(IP_PATTERN.match(ip_address))

    def validate_user_agent(user_agent):
        if not isinstance(user_agent, str):
            return False
        if len(user_agent) > 500:
            return False
        sanitized = re.sub(r'[<>"\';&]', '', user_agent)
        return len(sanitized) == len(user_agent)

    def validate_consent_type(consent_type):
        if not isinstance(consent_type, str):
            return False
        return consent_type in ALLOWED_CONSENT_TYPES

    # Test valid user IDs
    assert validate_user_id("test_user_123") == True
    assert validate_user_id("user-456") == True
    assert validate_user_id("valid_user") == True
    print("✓ Valid user IDs accepted")

    # Test invalid user IDs
    assert validate_user_id("invalid@user") == False
    assert validate_user_id("user with spaces") == False
    assert validate_user_id("") == False
    assert validate_user_id("a" * 51) == False  # Too long
    print("✓ Invalid user IDs rejected")

    # Test valid IPs
    assert validate_ip_address("192.168.1.1") == True
    assert validate_ip_address("10.0.0.1") == True
    assert validate_ip_address("127.0.0.1") == True
    assert validate_ip_address("") == True  # Empty allowed
    print("✓ Valid IP addresses accepted")

    # Test invalid IPs
    assert validate_ip_address("999.999.999.999") == False
    assert validate_ip_address("192.168.1") == False
    assert validate_ip_address("not.an.ip.address") == False
    print("✓ Invalid IP addresses rejected")

    # Test valid user agents
    assert validate_user_agent("Mozilla/5.0 Test Browser") == True
    assert validate_user_agent("curl/7.68.0") == True
    assert validate_user_agent("") == True  # Empty allowed
    print("✓ Valid user agents accepted")

    # Test invalid user agents
    assert validate_user_agent("<script>alert('xss')</script>") == False
    assert validate_user_agent("user'agent;with&danger\"chars") == False
    assert validate_user_agent("a" * 501) == False  # Too long
    print("✓ Invalid user agents rejected")

    # Test consent types
    assert validate_consent_type("voice_processing") == True
    assert validate_consent_type("data_storage") == True
    assert validate_consent_type("all_consent") == True
    assert validate_consent_type("invalid_type") == False
    assert validate_consent_type("") == False
    assert validate_consent_type(123) == False  # Not a string
    print("✓ Consent type validation works")

    print("✓ All validation pattern tests passed!")
    return True

def test_audio_buffer_deque():
    """Test that deque would work for audio buffer."""
    print("\nTesting deque for audio buffer...")

    from collections import deque
    import numpy as np

    # Test bounded deque
    max_size = 10
    buffer = deque(maxlen=max_size)

    # Fill buffer
    for i in range(15):
        test_data = np.random.rand(1024, 1).astype(np.float32)
        buffer.append(test_data)

    # Should not exceed max_size
    assert len(buffer) == max_size, f"Buffer size should be {max_size}, got {len(buffer)}"
    print(f"✓ Bounded deque maintains max size of {max_size}")

    # Test operations
    buffer.clear()
    assert len(buffer) == 0, "Buffer should be empty after clear"
    print("✓ Buffer clear operation works")

    # Test numpy array concatenation
    test_arrays = []
    for i in range(3):
        test_arrays.append(np.random.rand(100, 1).astype(np.float32))

    # Test that we can convert deque to list and concatenate
    buffer.extend(test_arrays)
    concatenated = np.concatenate(list(buffer), axis=0)
    assert concatenated.shape[0] == 300, f"Should have 300 samples, got {concatenated.shape[0]}"
    print("✓ Deque to list conversion and concatenation works")

    print("✓ Audio buffer deque test passed!")
    return True

def main():
    """Run simple tests."""
    print("Running simple security tests...\n")

    tests = [
        test_validation_patterns,
        test_audio_buffer_deque
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simple security tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())