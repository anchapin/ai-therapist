#!/usr/bin/env python3
"""
Test script for enhanced voice command processing system.

This script tests:
- Emergency command detection and response
- Command categories and natural language processing
- Confidence scoring and context awareness
- Logging and analytics functionality
- Integration with existing systems
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.config import VoiceConfig
from voice.commands import VoiceCommandProcessor, CommandCategory, SecurityLevel

class TestVoiceCommandSystem:
    """Test class for enhanced voice command system."""

    def __init__(self):
        """Initialize test environment."""
        self.config = VoiceConfig()
        self.processor = VoiceCommandProcessor(self.config)
        self.test_results = []

    async def test_emergency_command_detection(self):
        """Test emergency command detection and response."""
        print("\n=== Testing Emergency Command Detection ===")

        emergency_test_cases = [
            "I need help right now",
            "I'm having a crisis",
            "I want to kill myself",
            "Emergency help please",
            "I'm feeling suicidal",
            "Call crisis line",
            "I need immediate assistance"
        ]

        for test_case in emergency_test_cases:
            try:
                result = await self.processor.process_text(test_case, session_id="test_session")
                if result and result.is_emergency:
                    print(f"✓ Emergency detected: '{test_case}' -> {result.command.name}")
                    print(f"  Crisis keywords: {result.crisis_keywords_detected}")
                    print(f"  Confidence: {result.confidence:.2f}")

                    # Test command execution
                    execution_result = await self.processor.execute_command(result)
                    print(f"  Execution success: {execution_result['success']}")
                    if execution_result['success']:
                        print(f"  Voice feedback available: {'voice_feedback' in execution_result['result']}")
                else:
                    print(f"✗ Emergency NOT detected: '{test_case}'")
                    self.test_results.append(f"Emergency detection failed for: {test_case}")

            except Exception as e:
                print(f"✗ Error testing '{test_case}': {str(e)}")
                self.test_results.append(f"Error testing emergency case '{test_case}': {str(e)}")

    async def test_navigation_commands(self):
        """Test navigation command functionality."""
        print("\n=== Testing Navigation Commands ===")

        navigation_test_cases = [
            ("go home", "go_home"),
            ("take me to home page", "go_home"),
            ("show help", "get_help"),
            ("what commands can I use", "get_help"),
            ("open settings", "open_settings"),
            ("voice settings", "open_settings")
        ]

        for test_text, expected_command in navigation_test_cases:
            try:
                result = await self.processor.process_text(test_text, session_id="test_session")
                if result and result.command.name == expected_command:
                    print(f"✓ Navigation: '{test_text}' -> {result.command.name}")
                    print(f"  Category: {result.command.category.value}")
                    print(f"  Confidence: {result.confidence:.2f}")
                else:
                    print(f"✗ Navigation failed: '{test_text}' (expected {expected_command})")
                    if result:
                        print(f"  Got: {result.command.name} instead")
                    self.test_results.append(f"Navigation failed for: {test_text}")

            except Exception as e:
                print(f"✗ Error testing '{test_text}': {str(e)}")
                self.test_results.append(f"Error testing navigation '{test_text}': {str(e)}")

    async def test_session_control_commands(self):
        """Test session control command functionality."""
        print("\n=== Testing Session Control Commands ===")

        session_test_cases = [
            ("start new session", "start_session"),
            ("let's talk", "start_session"),
            ("end session", "end_session"),
            ("goodbye therapist", "end_session"),
            ("clear conversation", "clear_conversation"),
            ("start over", "clear_conversation")
        ]

        for test_text, expected_command in session_test_cases:
            try:
                result = await self.processor.process_text(test_text, session_id="test_session")
                if result and result.command.name == expected_command:
                    print(f"✓ Session control: '{test_text}' -> {result.command.name}")
                    print(f"  Category: {result.command.category.value}")
                    print(f"  Confidence: {result.confidence:.2f}")
                else:
                    print(f"✗ Session control failed: '{test_text}' (expected {expected_command})")
                    if result:
                        print(f"  Got: {result.command.name} instead")
                    self.test_results.append(f"Session control failed for: {test_text}")

            except Exception as e:
                print(f"✗ Error testing '{test_text}': {str(e)}")
                self.test_results.append(f"Error testing session control '{test_text}': {str(e)}")

    async def test_feature_access_commands(self):
        """Test feature access command functionality."""
        print("\n=== Testing Feature Access Commands ===")

        feature_test_cases = [
            ("start meditation", "start_meditation"),
            ("guided meditation", "start_meditation"),
            ("help me relax", "start_meditation"),
            ("open journal", "open_journal"),
            ("journal my thoughts", "open_journal"),
            ("show resources", "show_resources"),
            ("therapy resources", "show_resources")
        ]

        for test_text, expected_command in feature_test_cases:
            try:
                result = await self.processor.process_text(test_text, session_id="test_session")
                if result and result.command.name == expected_command:
                    print(f"✓ Feature access: '{test_text}' -> {result.command.name}")
                    print(f"  Category: {result.command.category.value}")
                    print(f"  Confidence: {result.confidence:.2f}")
                else:
                    print(f"✗ Feature access failed: '{test_text}' (expected {expected_command})")
                    if result:
                        print(f"  Got: {result.command.name} instead")
                    self.test_results.append(f"Feature access failed for: {test_text}")

            except Exception as e:
                print(f"✗ Error testing '{test_text}': {str(e)}")
                self.test_results.append(f"Error testing feature access '{test_text}': {str(e)}")

    async def test_voice_control_commands(self):
        """Test voice control command functionality."""
        print("\n=== Testing Voice Control Commands ===")

        voice_test_cases = [
            ("speak slower", "speak_slower"),
            ("can you talk slower please", "speak_slower"),
            ("speak faster", "speak_faster"),
            ("volume up", "adjust_volume"),
            ("make it louder", "adjust_volume"),
            ("pause for a moment", "pause_conversation"),
            ("repeat that", "repeat_last_response")
        ]

        for test_text, expected_command in voice_test_cases:
            try:
                result = await self.processor.process_text(test_text, session_id="test_session")
                if result and result.command.name == expected_command:
                    print(f"✓ Voice control: '{test_text}' -> {result.command.name}")
                    print(f"  Category: {result.command.category.value}")
                    print(f"  Confidence: {result.confidence:.2f}")
                else:
                    print(f"✗ Voice control failed: '{test_text}' (expected {expected_command})")
                    if result:
                        print(f"  Got: {result.command.name} instead")
                    self.test_results.append(f"Voice control failed for: {test_text}")

            except Exception as e:
                print(f"✗ Error testing '{test_text}': {str(e)}")
                self.test_results.append(f"Error testing voice control '{test_text}': {str(e)}")

    async def test_enhanced_parameters_extraction(self):
        """Test enhanced parameter extraction."""
        print("\n=== Testing Enhanced Parameter Extraction ===")

        parameter_test_cases = [
            "volume up a lot",
            "speak a little slower",
            "start a quick meditation",
            "change to a calm voice",
            "I need immediate help"
        ]

        for test_text in parameter_test_cases:
            try:
                result = await self.processor.process_text(test_text, session_id="test_session")
                if result:
                    print(f"✓ Parameters: '{test_text}' -> {result.parameters}")
                    if result.command.name == "adjust_volume":
                        print(f"  Volume direction: {result.parameters.get('direction', 'unknown')}")
                        print(f"  Volume magnitude: {result.parameters.get('magnitude', 'unknown')}")
                    elif result.command.name == "start_meditation":
                        print(f"  Meditation duration: {result.parameters.get('duration', 'unknown')}")
                        print(f"  Meditation type: {result.parameters.get('type', 'unknown')}")
                else:
                    print(f"✗ No command match found for: '{test_text}'")
                    self.test_results.append(f"No command match for: {test_text}")

            except Exception as e:
                print(f"✗ Error testing parameters for '{test_text}': {str(e)}")
                self.test_results.append(f"Error testing parameters '{test_text}': {str(e)}")

    async def test_command_analytics(self):
        """Test command analytics and statistics."""
        print("\n=== Testing Command Analytics ===")

        try:
            # Get analytics
            analytics = self.processor.get_command_analytics()
            print(f"✓ Analytics generated successfully")
            print(f"  Total commands processed: {analytics['total_commands']}")
            print(f"  Success rate: {analytics['success_rate']:.2%}")
            print(f"  Emergency incidents: {analytics['emergency_incidents']}")
            print(f"  Average confidence: {analytics['average_confidence']:.2f}")
            print(f"  Categories available: {list(analytics['category_usage'].keys())}")

            # Get command statistics
            stats = self.processor.get_statistics()
            print(f"✓ System statistics generated")
            print(f"  Total commands registered: {stats['total_commands']}")
            print(f"  Enabled commands: {stats['enabled_commands']}")
            print(f"  Emergency commands triggered: {stats.get('emergency_commands_triggered', 0)}")

        except Exception as e:
            print(f"✗ Error testing analytics: {str(e)}")
            self.test_results.append(f"Analytics test failed: {str(e)}")

    async def test_context_awareness(self):
        """Test context-aware command processing."""
        print("\n=== Testing Context Awareness ===")

        try:
            # Set up conversation context
            context = {
                'current_activity': 'meditation',
                'session_duration': 600,  # 10 minutes
                'user_preferences': {
                    'frequent_commands': ['start_meditation', 'pause_conversation']
                }
            }

            await self.processor._update_conversation_context("test_session", context)

            # Test context-aware command matching
            result = await self.processor.process_text("pause", session_id="test_session")
            if result:
                context_score = await self.processor._get_context_score("test_session", result.command)
                print(f"✓ Context awareness: 'pause' -> {result.command.name}")
                print(f"  Context score: {context_score:.2f}")
                print(f"  Current activity: {context['current_activity']}")
            else:
                print(f"✗ Context awareness failed for 'pause'")
                self.test_results.append("Context awareness test failed")

        except Exception as e:
            print(f"✗ Error testing context awareness: {str(e)}")
            self.test_results.append(f"Context awareness test failed: {str(e)}")

    async def test_audit_logging(self):
        """Test audit logging functionality."""
        print("\n=== Testing Audit Logging ===")

        try:
            # Execute a command to generate audit log
            result = await self.processor.process_text("help", session_id="test_session")
            if result:
                await self.processor.execute_command(result)

            # Get audit log
            audit_log = self.processor.get_audit_log(10)
            print(f"✓ Audit log generated with {len(audit_log)} entries")
            if audit_log:
                latest_entry = audit_log[-1]
                print(f"  Latest entry: {latest_entry.get('command_name', 'unknown')} at {latest_entry.get('timestamp', 'unknown')}")
                print(f"  Success: {latest_entry.get('success', False)}")

        except Exception as e:
            print(f"✗ Error testing audit logging: {str(e)}")
            self.test_results.append(f"Audit logging test failed: {str(e)}")

    async def run_all_tests(self):
        """Run all tests and generate report."""
        print("🧠 Starting Enhanced Voice Command System Tests")
        print("=" * 60)

        start_time = datetime.now()

        # Run all test methods
        test_methods = [
            self.test_emergency_command_detection,
            self.test_navigation_commands,
            self.test_session_control_commands,
            self.test_feature_access_commands,
            self.test_voice_control_commands,
            self.test_enhanced_parameters_extraction,
            self.test_command_analytics,
            self.test_context_awareness,
            self.test_audit_logging
        ]

        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"✗ Test method {test_method.__name__} failed: {str(e)}")
                self.test_results.append(f"Test method {test_method.__name__} failed: {str(e)}")

        # Generate test report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        if not self.test_results:
            print("✅ ALL TESTS PASSED! 🎉")
            print(f"   Test duration: {duration:.2f} seconds")
            print(f"   Commands tested: {len(self.processor.commands)}")
            print(f"   Emergency detection: ✓ Active")
            print(f"   Natural language processing: ✓ Active")
            print(f"   Context awareness: ✓ Active")
            print(f"   Audit logging: ✓ Active")
        else:
            print(f"⚠️  {len(self.test_results)} ISSUES FOUND:")
            for i, issue in enumerate(self.test_results, 1):
                print(f"   {i}. {issue}")
            print(f"\n   Test duration: {duration:.2f} seconds")
            print(f"   Commands tested: {len(self.processor.commands)}")

        print("\n🔧 SYSTEM CAPABILITIES:")
        print(f"   • Emergency keyword detection: {len(self.processor.emergency_keywords)} keywords")
        print(f"   • Command categories: {len(self.processor.commands_by_category)} categories")
        print(f"   • Crisis resources: {len(self.processor.crisis_resources)} resources")
        print(f"   • Security levels: {len([level for level in SecurityLevel])} levels")
        print(f"   • Wake word: '{self.processor.wake_word}'")

        return len(self.test_results) == 0

async def main():
    """Main test function."""
    try:
        tester = TestVoiceCommandSystem()
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)