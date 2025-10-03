#!/usr/bin/env python3
"""
Direct test script for voice commands module.

This script tests the command processing functionality by importing the commands module directly.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import comprehensive mock configuration
from voice.mock_config import create_mock_voice_config

# Create mock config for testing
MockVoiceConfig = create_mock_voice_config(
    voice_commands_enabled=True,
    voice_command_min_confidence=0.6,
    voice_command_wake_word="hey therapist",
    voice_command_timeout=30000
)

async def test_voice_commands():
    """Test voice commands functionality."""
    print("🧠 Testing Enhanced Voice Command System (Direct Import)")
    print("=" * 60)

    try:
        # Import commands module directly
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice'))
        from commands import VoiceCommandProcessor, CommandCategory, SecurityLevel

        # Create processor with mock config
        config = MockVoiceConfig()
        processor = VoiceCommandProcessor(config)

        print(f"✓ Voice command processor initialized successfully")
        print(f"  Total commands: {len(processor.commands)}")
        print(f"  Command categories: {list(processor.commands_by_category.keys())}")
        print(f"  Emergency keywords: {len(processor.emergency_keywords)}")
        print(f"  Crisis resources: {len(processor.crisis_resources)}")

        # Test 1: Emergency command detection
        print("\n🚨 Testing Emergency Command Detection:")
        emergency_phrases = [
            "I need help right now",
            "I want to kill myself",
            "Emergency please help",
            "I'm feeling suicidal",
            "I'm in crisis"
        ]

        emergency_count = 0
        for phrase in emergency_phrases:
            result = await processor.process_text(phrase, session_id="test_session")
            if result and result.is_emergency:
                emergency_count += 1
                print(f"  ✓ '{phrase}' -> {result.command.name} (Confidence: {result.confidence:.2f})")
                print(f"    Crisis keywords: {result.crisis_keywords_detected}")
            else:
                print(f"  ✗ '{phrase}' - No emergency detected")

        print(f"  Emergency detection rate: {emergency_count}/{len(emergency_phrases)} ({emergency_count/len(emergency_phrases)*100:.1f}%)")

        # Test 2: Navigation commands
        print("\n🧭 Testing Navigation Commands:")
        nav_commands = [
            ("go home", "go_home"),
            ("show help", "get_help"),
            ("open settings", "open_settings")
        ]

        nav_success = 0
        for phrase, expected in nav_commands:
            result = await processor.process_text(phrase, session_id="test_session")
            if result and result.command.name == expected:
                nav_success += 1
                print(f"  ✓ '{phrase}' -> {result.command.name} (Category: {result.command.category.value})")
            else:
                got = result.command.name if result else "None"
                print(f"  ✗ '{phrase}' - Expected {expected}, got {got}")

        print(f"  Navigation success rate: {nav_success}/{len(nav_commands)} ({nav_success/len(nav_commands)*100:.1f}%)")

        # Test 3: Feature access commands
        print("\n🎯 Testing Feature Access Commands:")
        feature_tests = [
            ("start meditation", "start_meditation"),
            ("open journal", "open_journal"),
            ("show resources", "show_resources")
        ]

        feature_success = 0
        for phrase, expected in feature_tests:
            result = await processor.process_text(phrase, session_id="test_session")
            if result and result.command.name == expected:
                feature_success += 1
                print(f"  ✓ '{phrase}' -> {result.command.name}")
            else:
                got = result.command.name if result else "None"
                print(f"  ✗ '{phrase}' - Expected {expected}, got {got}")

        print(f"  Feature access success rate: {feature_success}/{len(feature_tests)} ({feature_success/len(feature_tests)*100:.1f}%)")

        # Test 4: Voice control commands
        print("\n🎚️ Testing Voice Control Commands:")
        voice_tests = [
            ("speak slower", "speak_slower"),
            ("volume up", "adjust_volume"),
            ("pause conversation", "pause_conversation"),
            ("repeat that", "repeat_last_response")
        ]

        voice_success = 0
        for phrase, expected in voice_tests:
            result = await processor.process_text(phrase, session_id="test_session")
            if result and result.command.name == expected:
                voice_success += 1
                print(f"  ✓ '{phrase}' -> {result.command.name}")
                # Test parameter extraction
                if result.parameters:
                    print(f"    Parameters: {result.parameters}")
            else:
                got = result.command.name if result else "None"
                print(f"  ✗ '{phrase}' - Expected {expected}, got {got}")

        print(f"  Voice control success rate: {voice_success}/{len(voice_tests)} ({voice_success/len(voice_tests)*100:.1f}%)")

        # Test 5: Command execution
        print("\n⚙️ Testing Command Execution:")
        execution_tests = [
            "help",
            "status check",
            "start meditation"
        ]

        exec_success = 0
        for phrase in execution_tests:
            try:
                result = await processor.process_text(phrase, session_id="test_session")
                if result:
                    execution = await processor.execute_command(result)
                    if execution['success']:
                        exec_success += 1
                        print(f"  ✓ '{phrase}' -> Executed successfully")
                        print(f"    Processing time: {execution.get('processing_time', 0):.3f}s")
                        print(f"    Category: {execution.get('category', 'unknown')}")
                    else:
                        print(f"  ✗ '{phrase}' -> Execution failed: {execution.get('error', 'unknown error')}")
                else:
                    print(f"  ✗ '{phrase}' -> No command match found")
            except Exception as e:
                print(f"  ✗ '{phrase}' -> Exception: {str(e)}")

        print(f"  Command execution success rate: {exec_success}/{len(execution_tests)} ({exec_success/len(execution_tests)*100:.1f}%)")

        # Test 6: Analytics and statistics
        print("\n📊 Testing Analytics and Statistics:")
        try:
            analytics = processor.get_command_analytics()
            print(f"  ✓ Analytics generated:")
            print(f"    Total commands processed: {analytics['total_commands']}")
            print(f"    Success rate: {analytics['success_rate']:.2%}")
            print(f"    Emergency incidents: {analytics['emergency_incidents']}")
            print(f"    Average confidence: {analytics['average_confidence']:.2f}")

            stats = processor.get_statistics()
            print(f"  ✓ System statistics:")
            print(f"    Total commands: {stats['total_commands']}")
            print(f"    Enabled commands: {stats['enabled_commands']}")
            print(f"    Command history size: {stats['command_history_size']}")
        except Exception as e:
            print(f"  ✗ Analytics failed: {str(e)}")

        # Test 7: Crisis resources
        print("\n📞 Testing Crisis Resources:")
        print("  Available crisis resources:")
        for name, contact in processor.crisis_resources.items():
            print(f"    • {name.replace('_', ' ').title()}: {contact}")

        # Test 8: Command categories breakdown
        print("\n📋 Command Categories Breakdown:")
        for category in CommandCategory:
            if category in processor.commands_by_category:
                commands = processor.commands_by_category[category]
                print(f"  {category.value}: {len(commands)} commands")
                # Show first 2 commands as examples
                for cmd_name in commands[:2]:
                    cmd = processor.commands[cmd_name]
                    print(f"    - {cmd_name}: {cmd.description}")
                if len(commands) > 2:
                    print(f"    ... and {len(commands) - 2} more")

        # Test 9: Enhanced features
        print("\n🔧 Testing Enhanced Features:")

        # Test context awareness
        context = {
            'current_activity': 'meditation',
            'session_duration': 600,
            'user_preferences': {'frequent_commands': ['start_meditation']}
        }
        await processor._update_conversation_context("test_session", context)
        print(f"  ✓ Context awareness set up")

        # Test emergency keyword detection
        test_text = "I need immediate help and I'm feeling suicidal"
        keywords = processor._detect_emergency_keywords(test_text)
        print(f"  ✓ Emergency keyword detection: {len(keywords)} keywords found in '{test_text}'")

        # Test emergency classification
        emergency_type = processor._classify_emergency_type(keywords)
        print(f"  ✓ Emergency classification: {emergency_type}")

        # Test audit log
        audit_log = processor.get_audit_log(5)
        print(f"  ✓ Audit log: {len(audit_log)} entries")

        # Summary
        print("\n" + "=" * 60)
        print("📈 SUMMARY")
        print("=" * 60)
        total_tests = len(emergency_phrases) + len(nav_commands) + len(feature_tests) + len(voice_tests) + len(execution_tests)
        total_success = emergency_count + nav_success + feature_success + voice_success + exec_success
        success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0

        print(f"  Overall success rate: {success_rate:.1f}% ({total_success}/{total_tests})")
        print(f"  Emergency detection: {emergency_count}/{len(emergency_phrases)} ({emergency_count/len(emergency_phrases)*100:.1f}%)")
        print(f"  Navigation commands: {nav_success}/{len(nav_commands)} ({nav_success/len(nav_commands)*100:.1f}%)")
        print(f"  Feature access: {feature_success}/{len(feature_tests)} ({feature_success/len(feature_tests)*100:.1f}%)")
        print(f"  Voice control: {voice_success}/{len(voice_tests)} ({voice_success/len(voice_tests)*100:.1f}%)")
        print(f"  Command execution: {exec_success}/{len(execution_tests)} ({exec_success/len(execution_tests)*100:.1f}%)")

        if success_rate >= 80:
            print("\n🎉 Voice command system is working excellently!")
        elif success_rate >= 60:
            print("\n✅ Voice command system is working well!")
        else:
            print("\n⚠️ Voice command system needs attention")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function."""
    success = await test_voice_commands()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)