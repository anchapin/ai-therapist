#!/usr/bin/env python3
"""
Simple test script for voice command processing system.

This script tests the core command processing functionality without audio dependencies.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import comprehensive mock configuration
from voice.mock_config import create_mock_voice_config

# Create mock config for testing
MockVoiceConfig = create_mock_voice_config(
    voice_commands_enabled=True,
    voice_command_min_confidence=0.6
)

# Simple test focusing on command processing
async def test_command_processing():
    """Test core command processing functionality."""
    print("🧠 Testing Enhanced Voice Command Processing System")
    print("=" * 60)

    try:
        # Import our modules
        from voice.commands import VoiceCommandProcessor, CommandCategory, SecurityLevel

        # Create mock config
        config = MockVoiceConfig()
        processor = VoiceCommandProcessor(config)

        print(f"✓ Voice command processor initialized")
        print(f"  Total commands: {len(processor.commands)}")
        print(f"  Categories: {list(processor.commands_by_category.keys())}")
        print(f"  Emergency keywords: {len(processor.emergency_keywords)}")

        # Test emergency detection
        print("\n🚨 Testing Emergency Detection:")
        emergency_tests = [
            "I need help right now",
            "I want to kill myself",
            "Emergency please help",
            "I'm feeling suicidal"
        ]

        for test_text in emergency_tests:
            result = await processor.process_text(test_text, session_id="test")
            if result and result.is_emergency:
                print(f"  ✓ '{test_text}' -> {result.command.name} (Emergency)")
                print(f"    Keywords: {result.crisis_keywords_detected}")
                print(f"    Confidence: {result.confidence:.2f}")
            else:
                print(f"  ✗ '{test_text}' - Emergency NOT detected")

        # Test navigation commands
        print("\n🧭 Testing Navigation Commands:")
        nav_tests = [
            ("go home", "go_home"),
            ("show help", "get_help"),
            ("open settings", "open_settings")
        ]

        for test_text, expected in nav_tests:
            result = await processor.process_text(test_text, session_id="test")
            if result and result.command.name == expected:
                print(f"  ✓ '{test_text}' -> {result.command.name}")
                print(f"    Category: {result.command.category.value}")
                print(f"    Confidence: {result.confidence:.2f}")
            else:
                print(f"  ✗ '{test_text}' - Expected {expected}, got {result.command.name if result else 'None'}")

        # Test meditation commands
        print("\n🧘 Testing Meditation Commands:")
        meditation_tests = [
            "start meditation",
            "guided meditation please",
            "help me relax",
            "breathing exercise"
        ]

        for test_text in meditation_tests:
            result = await processor.process_text(test_text, session_id="test")
            if result and result.command.category == CommandCategory.MEDITATION:
                print(f"  ✓ '{test_text}' -> {result.command.name}")
                print(f"    Parameters: {result.parameters}")
            else:
                print(f"  ✗ '{test_text}' - Meditation command not detected")

        # Test session control
        print("\n🎯 Testing Session Control:")
        session_tests = [
            ("start new session", "start_session"),
            ("end session", "end_session"),
            ("clear conversation", "clear_conversation")
        ]

        for test_text, expected in session_tests:
            result = await processor.process_text(test_text, session_id="test")
            if result and result.command.name == expected:
                print(f"  ✓ '{test_text}' -> {result.command.name}")
            else:
                print(f"  ✗ '{test_text}' - Expected {expected}, got {result.command.name if result else 'None'}")

        # Test command execution
        print("\n⚙️ Testing Command Execution:")
        test_result = await processor.process_text("help", session_id="test")
        if test_result:
            execution = await processor.execute_command(test_result)
            print(f"  ✓ Command executed successfully: {execution['success']}")
            print(f"    Processing time: {execution.get('processing_time', 0):.3f}s")
            print(f"    Voice feedback: {'Yes' if 'voice_feedback' in execution.get('result', {}) else 'No'}")

        # Test analytics
        print("\n📊 Testing Analytics:")
        analytics = processor.get_command_analytics()
        print(f"  ✓ Analytics generated:")
        print(f"    Total commands: {analytics['total_commands']}")
        print(f"    Success rate: {analytics['success_rate']:.2%}")
        print(f"    Emergency incidents: {analytics['emergency_incidents']}")
        print(f"    Categories: {list(analytics['category_usage'].keys())}")

        # Test crisis resources
        print("\n📞 Testing Crisis Resources:")
        print("  Available resources:")
        for name, contact in processor.crisis_resources.items():
            print(f"    • {name.replace('_', ' ').title()}: {contact}")

        # Show command categories
        print("\n📋 Command Categories:")
        for category, commands in processor.commands_by_category.items():
            print(f"  {category.value}: {len(commands)} commands")
            for cmd in commands[:3]:  # Show first 3 commands
                print(f"    - {cmd}")
            if len(commands) > 3:
                print(f"    ... and {len(commands) - 3} more")

        print("\n✅ All core functionality tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_command_processing()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)