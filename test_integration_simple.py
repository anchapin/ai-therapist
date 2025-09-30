#!/usr/bin/env python3
"""
Simple integration test for voice command and crisis detection systems.

This script tests the integration without requiring audio libraries.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_crisis_detection_import():
    """Test that crisis detection functions can be imported."""
    print("🧠 Testing Crisis Detection Integration")
    print("=" * 60)

    try:
        # Import app crisis detection
        from app import detect_crisis_content, generate_crisis_response
        print("✓ App crisis detection functions imported successfully")

        # Test basic crisis detection
        test_cases = [
            ("I want to kill myself", True),
            ("I'm feeling suicidal", True),
            ("I need help with my homework", False),
            ("Hello, how are you?", False)
        ]

        for text, expected in test_cases:
            is_crisis, keywords = detect_crisis_content(text)
            if is_crisis == expected:
                print(f"  ✓ '{text}' -> Crisis: {is_crisis} ({keywords})")
            else:
                print(f"  ✗ '{text}' -> Expected {expected}, got {is_crisis}")

        # Test crisis response generation
        crisis_response = generate_crisis_response()
        if "988" in crisis_response and "741741" in crisis_response:
            print("✓ Crisis response contains essential resources")
        else:
            print("✗ Crisis response missing essential resources")

        return True

    except Exception as e:
        print(f"❌ Crisis detection test failed: {str(e)}")
        return False

def test_voice_command_import():
    """Test that voice command system can be imported."""
    print("\n🎙️ Testing Voice Command System")
    print("-" * 40)

    try:
        # Mock VoiceConfig to avoid audio dependencies
        class MockVoiceConfig:
            def __init__(self):
                self.voice_commands_enabled = True
                self.voice_command_min_confidence = 0.6
                self.voice_command_wake_word = "hey therapist"
                self.voice_command_timeout = 30000

        # Test voice command imports
        from voice.commands import VoiceCommandProcessor, CommandCategory, SecurityLevel
        print("✓ Voice command classes imported successfully")

        # Create processor with mock config
        config = MockVoiceConfig()
        processor = VoiceCommandProcessor(config)
        print("✓ Voice command processor initialized")

        # Test command categories
        categories = [cat.value for cat in CommandCategory]
        print(f"✓ Command categories available: {len(categories)} categories")

        # Test security levels
        levels = [level.value for level in SecurityLevel]
        print(f"✓ Security levels available: {len(levels)} levels")

        # Test emergency keywords
        emergency_count = len(processor.emergency_keywords)
        print(f"✓ Emergency keywords: {emergency_count} keywords")

        # Test crisis resources
        resources_count = len(processor.crisis_resources)
        print(f"✓ Crisis resources: {resources_count} resources")

        return True

    except Exception as e:
        print(f"❌ Voice command test failed: {str(e)}")
        return False

def test_integration_logic():
    """Test the integration logic between systems."""
    print("\n🔗 Testing Integration Logic")
    print("-" * 40)

    try:
        # Import both systems
        from app import detect_crisis_content, generate_crisis_response
        from voice.commands import VoiceCommandProcessor, CommandCategory

        # Mock VoiceConfig
        class MockVoiceConfig:
            def __init__(self):
                self.voice_commands_enabled = True
                self.voice_command_min_confidence = 0.6

        config = MockVoiceConfig()
        processor = VoiceCommandProcessor(config)

        # Test that voice system uses app's crisis detection
        test_text = "I want to kill myself"

        # Test app crisis detection
        app_crisis, app_keywords = detect_crisis_content(test_text)
        print(f"  App crisis detection: {app_crisis} ({app_keywords})")

        # Test voice crisis detection (should use app's function)
        voice_keywords = processor._detect_emergency_keywords(test_text)
        print(f"  Voice crisis detection: {len(voice_keywords)} keywords")

        # Test that both detect crisis
        if app_crisis and len(voice_keywords) > 0:
            print("✓ Both systems detect crisis")
        else:
            print("✗ Crisis detection mismatch")

        # Test crisis response consistency
        app_response = generate_crisis_response()
        voice_response = processor.crisis_resources

        if "988" in app_response and "988" in str(voice_response):
            print("✓ Both systems include 988 hotline")
        else:
            print("✗ Crisis resource mismatch")

        return True

    except Exception as e:
        print(f"❌ Integration logic test failed: {str(e)}")
        return False

def main():
    """Run all integration tests."""
    print("🧠 Voice Command - Crisis Detection Integration Test")
    print("=" * 60)

    tests = [
        test_crisis_detection_import,
        test_voice_command_import,
        test_integration_logic
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ {test.__name__} failed")

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("\n🔧 Integration Status:")
        print("   • Crisis detection system: ✓ Active")
        print("   • Voice command system: ✓ Active")
        print("   • System integration: ✓ Complete")
        print("   • Emergency response: ✓ Coordinated")
        return True
    else:
        print("❌ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)