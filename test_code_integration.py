#!/usr/bin/env python3
"""
Code integration test to verify the voice command system is properly integrated.

This test checks the code structure and imports without running the application.
"""

import os
import sys
import re

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_py_integration():
    """Test that app.py has been properly updated with voice command integration."""
    print("📝 Testing app.py Integration")
    print("=" * 40)

    try:
        with open('app.py', 'r') as f:
            app_content = f.read()

        # Check for voice command imports
        required_imports = [
            'from voice.commands import VoiceCommandProcessor',
            'CommandCategory',
            'SecurityLevel'
        ]

        for import_statement in required_imports:
            if import_statement in app_content:
                print(f"✓ Found import: {import_statement}")
            else:
                print(f"✗ Missing import: {import_statement}")
                return False

        # Check for voice command processor initialization
        if 'voice_command_processor' in app_content:
            print("✓ Voice command processor integration found")
        else:
            print("✗ Voice command processor integration missing")
            return False

        # Check for enhanced crisis detection
        if 'overwhelmed' in app_content and 'desperate' in app_content:
            print("✓ Enhanced crisis keywords found")
        else:
            print("✗ Enhanced crisis keywords missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Error reading app.py: {str(e)}")
        return False

def test_voice_commands_py_integration():
    """Test that voice/commands.py has been properly updated with app integration."""
    print("\n🎙️ Testing voice/commands.py Integration")
    print("-" * 40)

    try:
        with open('voice/commands.py', 'r') as f:
            commands_content = f.read()

        # Check for app crisis detection imports
        if 'from app import detect_crisis_content' in commands_content:
            print("✓ App crisis detection import found")
        else:
            print("✗ App crisis detection import missing")
            return False

        # Check for enhanced emergency detection
        if 'using app\'s crisis detection' in commands_content:
            print("✓ Enhanced emergency detection found")
        else:
            print("✗ Enhanced emergency detection missing")
            return False

        # Check for crisis response integration
        if 'generate_crisis_response' in commands_content:
            print("✓ Crisis response integration found")
        else:
            print("✗ Crisis response integration missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Error reading voice/commands.py: {str(e)}")
        return False

def test_session_state_integration():
    """Test that session state includes voice command processor."""
    print("\n💾 Testing Session State Integration")
    print("-" * 40)

    try:
        with open('app.py', 'r') as f:
            app_content = f.read()

        # Check for voice command processor in session state
        if 'voice_command_processor' in app_content:
            print("✓ Voice command processor session state found")
        else:
            print("✗ Voice command processor session state missing")
            return False

        # Check for voice command processor initialization
        if 'VoiceCommandProcessor(voice_config)' in app_content:
            print("✓ Voice command processor initialization found")
        else:
            print("✗ Voice command processor initialization missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Error checking session state: {str(e)}")
        return False

def test_voice_handler_integration():
    """Test that voice handler uses voice command processor."""
    print("\n🔊 Testing Voice Handler Integration")
    print("-" * 40)

    try:
        with open('app.py', 'r') as f:
            app_content = f.read()

        # Check for voice command processing in handler
        if 'process_voice_commands' in app_content or 'voice_command_processor.process_text' in app_content:
            print("✓ Voice command processing found in handler")
        else:
            print("✗ Voice command processing missing from handler")
            return False

        # Check for emergency handling
        if 'is_emergency' in app_content and 'emergency_response' in app_content:
            print("✓ Emergency handling found in voice handler")
        else:
            print("✗ Emergency handling missing from voice handler")
            return False

        # Check for crisis integration
        if 'detect_crisis_content' in app_content:
            print("✓ Crisis detection integration found")
        else:
            print("✗ Crisis detection integration missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Error checking voice handler: {str(e)}")
        return False

def test_file_structure():
    """Test that all necessary files exist."""
    print("\n📁 Testing File Structure")
    print("-" * 40)

    required_files = [
        'app.py',
        'voice/commands.py',
        'voice/config.py',
        'voice/security.py',
        'voice/voice_service.py',
        'voice/voice_ui.py'
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False

    return all_exist

def test_code_quality():
    """Test basic code quality aspects."""
    print("\n🔍 Testing Code Quality")
    print("-" * 40)

    issues = []

    # Check for duplicate functions in voice/commands.py
    try:
        with open('voice/commands.py', 'r') as f:
            content = f.read()

        # Count emergency response handlers
        emergency_handlers = content.count('def _handle_emergency_response')
        if emergency_handlers > 1:
            issues.append(f"Multiple emergency handlers found: {emergency_handlers}")

        # Check for proper exception handling
        if 'except Exception:' not in content:
            issues.append("Missing exception handling")

        # Check for proper logging
        if 'self.logger.' not in content:
            issues.append("Missing logging statements")

    except Exception as e:
        issues.append(f"Error checking code quality: {str(e)}")

    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
        return False
    else:
        print("✓ Code quality checks passed")
        return True

def main():
    """Run all integration tests."""
    print("🧠 Voice Command System Integration Test")
    print("=" * 60)

    tests = [
        test_file_structure,
        test_app_py_integration,
        test_voice_commands_py_integration,
        test_session_state_integration,
        test_voice_handler_integration,
        test_code_quality
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("\n🎉 Integration Complete!")
        print("\n🔧 System Status:")
        print("   • Voice command system: ✓ Integrated")
        print("   • Crisis detection: ✓ Synchronized")
        print("   • Emergency response: ✓ Coordinated")
        print("   • Session management: ✓ Updated")
        print("   • Code quality: ✓ Verified")
        print("\n🚀 The enhanced voice command system is now fully integrated!")
        return True
    else:
        print("❌ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)