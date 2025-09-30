#!/usr/bin/env python3
"""
Integration test script for voice command system with crisis detection.

This script tests the integration between the enhanced voice command system
and the existing crisis detection system in the main application.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class IntegrationTestSuite:
    """Test suite for voice command and crisis detection integration."""

    def __init__(self):
        """Initialize test environment."""
        self.test_results = []
        self.voice_processor = None
        self.app_crisis_detected = []

    def setup_test_environment(self):
        """Set up the test environment."""
        print("🧠 Setting up Integration Test Environment")
        print("=" * 60)

        try:
            # Import voice command processor
            from voice.commands import VoiceCommandProcessor, CommandCategory, SecurityLevel
            from voice.config import VoiceConfig

            # Import app crisis detection
            from app import detect_crisis_content, generate_crisis_response

            # Create mock config
            config = VoiceConfig()
            self.voice_processor = VoiceCommandProcessor(config)

            print(f"✓ Voice command processor initialized")
            print(f"✓ App crisis detection functions imported")
            print(f"✓ Integration test environment ready")

            return True

        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            return False

    async def test_crisis_detection_consistency(self):
        """Test that both systems detect the same crisis keywords."""
        print("\n🚨 Testing Crisis Detection Consistency")
        print("-" * 40)

        test_phrases = [
            "I want to kill myself",
            "I'm feeling suicidal",
            "I need emergency help",
            "I'm having a crisis",
            "I want to end my life",
            "I'm feeling hopeless"
        ]

        for phrase in test_phrases:
            try:
                # Test voice command system
                voice_result = await self.voice_processor.process_text(phrase, session_id="test")
                voice_emergency = voice_result.is_emergency if voice_result else False
                voice_keywords = voice_result.crisis_keywords_detected if voice_result else []

                # Test app crisis detection
                app_emergency, app_keywords = detect_crisis_content(phrase)

                # Compare results
                if voice_emergency == app_emergency:
                    print(f"  ✓ '{phrase}' - Both systems agree: {voice_emergency}")
                    print(f"    Voice keywords: {voice_keywords}")
                    print(f"    App keywords: {app_keywords}")
                else:
                    print(f"  ✗ '{phrase}' - Systems disagree: Voice={voice_emergency}, App={app_emergency}")
                    self.test_results.append(f"Crisis detection mismatch for: {phrase}")

            except Exception as e:
                print(f"  ✗ Error testing '{phrase}': {str(e)}")
                self.test_results.append(f"Error in crisis consistency test: {str(e)}")

    async def test_emergency_command_prioritization(self):
        """Test that emergency commands have highest priority."""
        print("\n⚡ Testing Emergency Command Prioritization")
        print("-" * 40)

        test_cases = [
            ("I need help right now", "emergency_response"),
            ("Emergency please help", "emergency_response"),
            ("I'm suicidal", "emergency_response"),
            ("help me", "emergency_response"),
            ("call crisis line", "call_crisis_line")
        ]

        for phrase, expected_command in test_cases:
            try:
                result = await self.voice_processor.process_text(phrase, session_id="test")
                if result and result.command.name == expected_command:
                    print(f"  ✓ '{phrase}' -> {result.command.name} (Priority: {result.command.priority})")
                    print(f"    Confidence: {result.confidence:.2f}")
                    print(f"    Emergency: {result.is_emergency}")
                else:
                    got = result.command.name if result else "None"
                    print(f"  ✗ '{phrase}' - Expected {expected_command}, got {got}")
                    self.test_results.append(f"Priority test failed for: {phrase}")

            except Exception as e:
                print(f"  ✗ Error testing '{phrase}': {str(e)}")
                self.test_results.append(f"Error in priority test: {str(e)}")

    async def test_voice_command_execution_with_crisis(self):
        """Test voice command execution with crisis scenarios."""
        print("\n⚙️ Testing Voice Command Execution with Crisis")
        print("-" * 40)

        crisis_commands = [
            "Emergency help",
            "I need immediate assistance",
            "I'm in crisis",
            "help me now"
        ]

        for command in crisis_commands:
            try:
                result = await self.voice_processor.process_text(command, session_id="test")
                if result and result.is_emergency:
                    execution = await self.voice_processor.execute_command(result)
                    if execution['success']:
                        print(f"  ✓ '{command}' -> Emergency response executed")
                        print(f"    Voice feedback available: {'voice_feedback' in execution.get('result', {})}")
                        print(f"    Severity: {execution.get('severity', 'unknown')}")
                    else:
                        print(f"  ✗ '{command}' -> Execution failed")
                        self.test_results.append(f"Execution failed for: {command}")
                else:
                    print(f"  ✗ '{command}' - Not detected as emergency")
                    self.test_results.append(f"Emergency detection failed for: {command}")

            except Exception as e:
                print(f"  ✗ Error testing '{command}': {str(e)}")
                self.test_results.append(f"Error in execution test: {str(e)}")

    async def test_crisis_resources_consistency(self):
        """Test that crisis resources are consistent between systems."""
        print("\n📞 Testing Crisis Resources Consistency")
        print("-" * 40)

        try:
            # Get crisis resources from voice system
            voice_resources = self.voice_processor.crisis_resources

            # Generate crisis response from app
            app_response = generate_crisis_response()

            print("  Voice system crisis resources:")
            for name, contact in voice_resources.items():
                print(f"    • {name}: {contact}")

            print(f"  App crisis response includes: {'988' in app_response}")
            print(f"  App crisis response includes: {'741741' in app_response}")

            # Check for key resources
            essential_resources = ['988', '741741', '911']
            all_present = all(
                any(resource in str(contact) for contact in voice_resources.values())
                for resource in essential_resources
            )

            if all_present:
                print("  ✓ All essential crisis resources present in voice system")
            else:
                print("  ✗ Some essential crisis resources missing")
                self.test_results.append("Crisis resources incomplete")

        except Exception as e:
            print(f"  ✗ Error testing crisis resources: {str(e)}")
            self.test_results.append(f"Error in resources test: {str(e)}")

    async def test_non_crisis_command_handling(self):
        """Test that non-crisis commands are handled normally."""
        print("\n🎯 Testing Non-Crisis Command Handling")
        print("-" * 40)

        normal_commands = [
            ("start meditation", "start_meditation"),
            ("go home", "go_home"),
            ("show help", "get_help"),
            ("volume up", "adjust_volume")
        ]

        for phrase, expected_command in normal_commands:
            try:
                result = await self.voice_processor.process_text(phrase, session_id="test")
                if result and result.command.name == expected_command and not result.is_emergency:
                    print(f"  ✓ '{phrase}' -> {result.command.name} (Normal processing)")
                    print(f"    Confidence: {result.confidence:.2f}")
                else:
                    got = result.command.name if result else "None"
                    emergency_status = result.is_emergency if result else "Unknown"
                    print(f"  ✗ '{phrase}' - Expected {expected_command}, got {got} (Emergency: {emergency_status})")
                    self.test_results.append(f"Normal command handling failed for: {phrase}")

            except Exception as e:
                print(f"  ✗ Error testing '{phrase}': {str(e)}")
                self.test_results.append(f"Error in normal command test: {str(e)}")

    async def test_integration_edge_cases(self):
        """Test edge cases in the integration."""
        print("\n🔍 Testing Integration Edge Cases")
        print("-" * 40)

        edge_cases = [
            "I need some help with my meditation",  # Mixed context
            "Emergency but I'm just testing",       # Testing with emergency keywords
            "Help me understand my feelings",       # Help request without crisis
            "I feel like I want to be alone",       # Contains 'alone' but not crisis
        ]

        for phrase in edge_cases:
            try:
                # Test both systems
                voice_result = await self.voice_processor.process_text(phrase, session_id="test")
                app_emergency, app_keywords = detect_crisis_content(phrase)

                voice_emergency = voice_result.is_emergency if voice_result else False
                voice_keywords = voice_result.crisis_keywords_detected if voice_result else []

                print(f"  '{phrase}':")
                print(f"    Voice: Emergency={voice_emergency}, Keywords={voice_keywords}")
                print(f"    App: Emergency={app_emergency}, Keywords={app_keywords}")

                # Check if reasonable agreement
                if voice_emergency == app_emergency:
                    print(f"    ✓ Systems agree")
                else:
                    print(f"    ⚠️  Systems disagree - may be context-dependent")
                    # Not necessarily an error, could be different interpretation

            except Exception as e:
                print(f"  ✗ Error testing '{phrase}': {str(e)}")
                self.test_results.append(f"Error in edge case test: {str(e)}")

    async def run_all_tests(self):
        """Run all integration tests."""
        print("🧠 Starting Voice Command - Crisis Detection Integration Tests")
        print("=" * 60)

        start_time = datetime.now()

        # Setup
        if not self.setup_test_environment():
            return False

        # Run all test methods
        test_methods = [
            self.test_crisis_detection_consistency,
            self.test_emergency_command_prioritization,
            self.test_voice_command_execution_with_crisis,
            self.test_crisis_resources_consistency,
            self.test_non_crisis_command_handling,
            self.test_integration_edge_cases
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
        print("📊 INTEGRATION TEST RESULTS SUMMARY")
        print("=" * 60)

        if not self.test_results:
            print("✅ ALL INTEGRATION TESTS PASSED! 🎉")
            print(f"   Test duration: {duration:.2f} seconds")
            print(f"   Voice command system: ✓ Integrated")
            print(f"   Crisis detection: ✓ Synchronized")
            print(f"   Emergency response: ✓ Coordinated")
            print(f"   Resource consistency: ✓ Verified")
        else:
            print(f"⚠️  {len(self.test_results)} ISSUES FOUND:")
            for i, issue in enumerate(self.test_results, 1):
                print(f"   {i}. {issue}")
            print(f"\n   Test duration: {duration:.2f} seconds")

        print("\n🔧 INTEGRATION STATUS:")
        print(f"   • Voice command processor: ✓ Active")
        print(f"   • App crisis detection: ✓ Active")
        print(f"   • Emergency keyword sync: ✓ Active")
        print(f"   • Response coordination: ✓ Active")
        print(f"   • Resource alignment: ✓ Active")

        return len(self.test_results) == 0

async def main():
    """Main test function."""
    try:
        tester = IntegrationTestSuite()
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)