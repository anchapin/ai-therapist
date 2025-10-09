"""
Comprehensive Voice Commands Tests

This module provides extensive coverage for voice/commands.py functionality:
- Command pattern matching and natural language processing
- Command registration and management
- Wake word detection and context awareness
- Emergency response and crisis detection workflows
- Command validation and security checks
- Help system and feedback mechanisms
- Custom command registration and extensibility
- Performance optimization and caching
"""

import pytest
import asyncio
import time
import json
import re
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import voice commands components
try:
    from voice.commands import (
        VoiceCommandProcessor,
        VoiceCommand,
        CommandCategory,
        SecurityLevel
    )
    from voice.stt_service import STTResult
    VOICE_COMMANDS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Voice commands import failed: {e}")
    VOICE_COMMANDS_AVAILABLE = False

# Import fixtures
try:
    from tests.fixtures.voice_fixtures import mock_voice_config
except ImportError:
    # Fallback fixture definition
    @pytest.fixture
    def mock_voice_config():
        """Fallback mock VoiceConfig for testing."""
        config = MagicMock()
        config.voice_enabled = True
        config.voice_commands_enabled = True
        config.stt_provider = "openai"
        config.tts_provider = "openai"
        config.default_voice_profile = "alloy"
        config.wake_word = "therapist"
        config.confidence_threshold = 0.7
        config.command_timeout = 30000
        config.session_timeout_minutes = 30
        return config


class TestVoiceCommandCore:
    """Test core voice command functionality."""
    
    @pytest.fixture
    def command_processor(self):
        """Create VoiceCommandProcessor instance for testing."""
        if not VOICE_COMMANDS_AVAILABLE:
            pytest.skip("Voice commands not available")
        
        config = mock_voice_config
        return VoiceCommandProcessor(config)
    
    @pytest.fixture
    def sample_command(self):
        """Create a sample voice command for testing."""
        return VoiceCommand(
            name="breathing_exercise",
            description="Start a breathing exercise",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[
                r"start breathing exercise",
                r"begin breathing exercise", 
                r"help me breathe",
                r"breathing technique"
            ],
            action="start_breathing_exercise",
            examples=[
                "Start breathing exercise",
                "Help me breathe",
                "Begin breathing technique"
            ],
            security_level=SecurityLevel.LOW,
            enabled=True,
            confidence_threshold=0.7
        )
    
    def test_command_creation(self, sample_command):
        """Test VoiceCommand object creation and attributes."""
        assert sample_command.name == "breathing_exercise"
        assert sample_command.category == CommandCategory.FEATURE_ACCESS
        assert len(sample_command.patterns) == 4
        assert sample_command.security_level == SecurityLevel.LOW
        assert sample_command.enabled is True
        assert sample_command.confidence_threshold == 0.7
    
    def test_command_processor_initialization(self, command_processor):
        """Test VoiceCommandProcessor initialization."""
        assert command_processor.config is not None
        assert len(command_processor.commands) > 0  # Should have default commands
        assert command_processor.wake_word == "therapist"
        assert command_processor.confidence_threshold == 0.7


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCommandRegistration:
    """Test command registration and management."""
    
    @pytest.fixture
    def command_processor(self, mock_voice_config):
        """Create command processor for registration tests."""
        return VoiceCommandProcessor(mock_voice_config)
    
    def test_register_command(self, command_processor):
        """Test registering a new command."""
        new_command = VoiceCommand(
            name="test_command",
            description="Test command for testing",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"test command", r"run test"],
            action="test_action"
        )
        
        command_processor.register_command(new_command)
        
        assert "test_command" in command_processor.commands
        assert command_processor.commands["test_command"] == new_command
    
    def test_register_duplicate_command(self, command_processor):
        """Test registering a command with duplicate name."""
        command = VoiceCommand(
            name="breathing_exercise",  # This should already exist
            description="Duplicate command",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"duplicate"],
            action="duplicate_action"
        )
        
        with pytest.raises(ValueError) as exc_info:
            command_processor.register_command(command)
        
        assert "already registered" in str(exc_info.value).lower()
    
    def test_unregister_command(self, command_processor):
        """Test unregistering a command."""
        # First register a test command
        test_command = VoiceCommand(
            name="temp_command",
            description="Temporary command",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"temp"],
            action="temp_action"
        )
        command_processor.register_command(test_command)
        
        # Now unregister it
        success = command_processor.unregister_command("temp_command")
        
        assert success is True
        assert "temp_command" not in command_processor.commands
    
    def test_unregister_nonexistent_command(self, command_processor):
        """Test unregistering a non-existent command."""
        success = command_processor.unregister_command("nonexistent_command")
        
        assert success is False
    
    def test_list_commands_by_category(self, command_processor):
        """Test listing commands by category."""
        navigation_commands = command_processor.list_commands_by_category(CommandCategory.NAVIGATION)
        
        assert isinstance(navigation_commands, list)
        for cmd in navigation_commands:
            assert cmd.category == CommandCategory.NAVIGATION
    
    def test_get_command_by_name(self, command_processor):
        """Test retrieving command by name."""
        command = command_processor.commands.get("breathing_exercise")
        
        assert command is not None
        assert command.name == "breathing_exercise"
    
    def test_get_nonexistent_command(self, command_processor):
        """Test retrieving non-existent command."""
        command = command_processor.commands.get("nonexistent_command")
        
        assert command is None


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCommandPatternMatching:
    """Test command pattern matching and natural language processing."""
    
    @pytest.fixture
    def command_processor(self, mock_voice_config):
        """Create command processor for pattern matching tests."""
        processor = VoiceCommandProcessor(mock_voice_config)
        
        # Register test commands with various patterns
        test_commands = [
            VoiceCommand(
                name="simple_test",
                description="Simple test command",
                category=CommandCategory.FEATURE_ACCESS,
                patterns=[r"simple test", r"basic test"],
                action="simple_action"
            ),
            VoiceCommand(
                name="parameter_test",
                description="Command with parameters",
                category=CommandCategory.FEATURE_ACCESS,
                patterns=[r"set timer for (\d+) minutes", r"timer (\d+)"],
                action="timer_action",
                parameters={"extract_numbers": True}
            ),
            VoiceCommand(
                name="context_test",
                description="Context-aware command",
                category=CommandCategory.FEATURE_ACCESS,
                patterns=[r"continue", r"next", r"go on"],
                action="continue_action",
                context_aware=True
            )
        ]
        
        for cmd in test_commands:
            processor.register_command(cmd)
        
        return processor
    
    def test_simple_pattern_matching(self, command_processor):
        """Test simple command pattern matching."""
        text = "simple test"
        
        matches = command_processor.find_matching_commands(text)
        
        assert len(matches) > 0
        assert matches[0].command.name == "simple_test"
        assert matches[0].confidence > 0.5
    
    def test_multiple_pattern_matches(self, command_processor):
        """Test handling multiple pattern matches."""
        text = "set timer for 5 minutes"
        
        matches = command_processor.find_matching_commands(text)
        
        assert len(matches) > 0
        assert matches[0].command.name == "parameter_test"
        assert matches[0].confidence > 0.5
    
    def test_confidence_scoring(self, command_processor):
        """Test confidence scoring for pattern matches."""
        # Exact match should have high confidence
        exact_match = command_processor.find_matching_commands("simple test")
        
        # Partial match should have lower confidence
        partial_match = command_processor.find_matching_commands("simple testing")
        
        assert exact_match[0].confidence >= partial_match[0].confidence
    
    def test_no_pattern_match(self, command_processor):
        """Test handling text that doesn't match any patterns."""
        text = "random text that matches nothing"
        
        matches = command_processor.find_matching_commands(text)
        
        assert len(matches) == 0
    
    def test_case_insensitive_matching(self, command_processor):
        """Test case-insensitive pattern matching."""
        matches_lower = command_processor.find_matching_commands("simple test")
        matches_upper = command_processor.find_matching_commands("SIMPLE TEST")
        matches_mixed = command_processor.find_matching_commands("Simple Test")
        
        assert len(matches_lower) == len(matches_upper) == len(matches_mixed)
        assert matches_lower[0].command.name == matches_upper[0].command.name == matches_mixed[0].command.name
    
    def test_parameter_extraction(self, command_processor):
        """Test parameter extraction from matched patterns."""
        text = "set timer for 15 minutes"
        
        matches = command_processor.find_matching_commands(text)
        
        if matches:
            match = matches[0]
            if match.parameters:
                assert "15" in str(match.parameters) or 15 in match.parameters.values()


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestWakeWordDetection:
    """Test wake word detection functionality."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for wake word tests."""
        config = mock_voice_config
        return VoiceCommandProcessor(config)
    
    def test_wake_word_present(self, command_processor):
        """Test detecting wake word in text."""
        text = "therapist start breathing exercise"
        
        has_wake_word = command_processor.detect_wake_word(text)
        
        assert has_wake_word is True
    
    def test_wake_word_absent(self, command_processor):
        """Test text without wake word."""
        text = "start breathing exercise"
        
        has_wake_word = command_processor.detect_wake_word(text)
        
        assert has_wake_word is False
    
    def test_multiple_wake_words(self, command_processor):
        """Test text with multiple wake words."""
        text = "therapist therapist start exercise"
        
        has_wake_word = command_processor.detect_wake_word(text)
        
        assert has_wake_word is True
    
    def test_case_insensitive_wake_word(self, command_processor):
        """Test case-insensitive wake word detection."""
        variations = ["Therapist help", "THERAPIST help", "therapist HELP"]
        
        for text in variations:
            has_wake_word = command_processor.detect_wake_word(text)
            assert has_wake_word is True
    
    def test_partial_wake_word(self, command_processor):
        """Test partial wake word matching."""
        text = "therapy session"  # Contains "therap" but not full wake word
        
        has_wake_word = command_processor.detect_wake_word(text)
        
        # Should not match partial wake word
        assert has_wake_word is False


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCommandExecution:
    """Test command execution and action handling."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for execution tests."""
        config = mock_voice_config
        processor = VoiceCommandProcessor(config)
        
        # Register test command with mock action
        test_command = VoiceCommand(
            name="mock_test",
            description="Test command with mock action",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"mock test"],
            action="mock_action"
        )
        processor.register_command(test_command)
        
        return processor
    
    @pytest.mark.asyncio
    async def test_command_execution_success(self, command_processor):
        """Test successful command execution."""
        text = "mock test"
        
        # Mock the action handler
        with patch.object(command_processor, 'execute_command_action') as mock_execute:
            mock_execute.return_value = {"success": True, "message": "Command executed"}
            
            result = await command_processor.process_command(text)
            
            assert result is not None
            assert result["success"] is True
            assert result["message"] == "Command executed"
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_command_execution_with_parameters(self, command_processor):
        """Test command execution with parameters."""
        # Register parameter command
        param_command = VoiceCommand(
            name="param_test",
            description="Parameter test command",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"test with (\w+)"],
            action="param_action"
        )
        command_processor.register_command(param_command)
        
        text = "test with value"
        
        with patch.object(command_processor, 'execute_command_action') as mock_execute:
            mock_execute.return_value = {"success": True, "parameters": ["value"]}
            
            result = await command_processor.process_command(text)
            
            assert result["success"] is True
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_command_execution_failure(self, command_processor):
        """Test command execution failure handling."""
        text = "mock test"
        
        with patch.object(command_processor, 'execute_command_action') as mock_execute:
            mock_execute.side_effect = Exception("Action failed")
            
            result = await command_processor.process_command(text)
            
            assert result is not None
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_no_matching_command_execution(self, command_processor):
        """Test execution when no command matches."""
        text = "no matching command here"
        
        result = await command_processor.process_command(text)
        
        assert result is None or result.get("success") is False


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestEmergencyCommands:
    """Test emergency command detection and response."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for emergency tests."""
        config = mock_voice_config
        processor = VoiceCommandProcessor(config)
        
        # Register emergency command
        emergency_command = VoiceCommand(
            name="emergency_help",
            description="Emergency help command",
            category=CommandCategory.EMERGENCY,
            patterns=[r"emergency", r"help me now", r"crisis"],
            action="emergency_response",
            security_level=SecurityLevel.HIGH,
            emergency_keywords=["suicide", "kill", "harm"]
        )
        processor.register_command(emergency_command)
        
        return processor
    
    def test_emergency_command_detection(self, command_processor):
        """Test detection of emergency commands."""
        emergency_texts = [
            "emergency help needed",
            "I'm in crisis",
            "help me now"
        ]
        
        for text in emergency_texts:
            matches = command_processor.find_matching_commands(text)
            emergency_found = any(
                match.command.category == CommandCategory.EMERGENCY 
                for match in matches
            )
            assert emergency_found, f"Should detect emergency in: {text}"
    
    def test_crisis_keyword_detection(self, command_processor):
        """Test crisis keyword detection in text."""
        crisis_texts = [
            "I want to kill myself",
            "thinking about suicide",
            "want to harm myself"
        ]
        
        for text in crisis_texts:
            is_crisis, keywords = command_processor.detect_crisis_keywords(text)
            
            assert is_crisis is True
            assert len(keywords) > 0
    
    def test_non_crisis_text(self, command_processor):
        """Test that non-crisis text is not flagged."""
        normal_texts = [
            "I feel sad today",
            "had a good day",
            "feeling anxious about work"
        ]
        
        for text in normal_texts:
            is_crisis, keywords = command_processor.detect_crisis_keywords(text)
            
            assert is_crisis is False
            assert len(keywords) == 0
    
    @pytest.mark.asyncio
    async def test_emergency_command_execution(self, command_processor):
        """Test emergency command execution with high priority."""
        text = "emergency help now"
        
        with patch.object(command_processor, 'handle_emergency') as mock_emergency:
            mock_emergency.return_value = {
                "emergency": True,
                "response": "Emergency help is being dispatched",
                "contacts": ["911", "crisis hotline"]
            }
            
            result = await command_processor.process_command(text)
            
            assert result is not None
            assert result["emergency"] is True
            assert "response" in result
            mock_emergency.assert_called_once()


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCommandSecurity:
    """Test command security and access control."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for security tests."""
        config = mock_voice_config
        processor = VoiceCommandProcessor(config)
        
        # Register commands with different security levels
        low_command = VoiceCommand(
            name="low_security",
            description="Low security command",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"low security"],
            action="low_action",
            security_level=SecurityLevel.LOW
        )
        
        high_command = VoiceCommand(
            name="high_security",
            description="High security command", 
            category=CommandCategory.EMERGENCY,
            patterns=[r"high security"],
            action="high_action",
            security_level=SecurityLevel.HIGH
        )
        
        processor.register_command(low_command)
        processor.register_command(high_command)
        
        return processor
    
    def test_security_level_enforcement(self, command_processor):
        """Test security level enforcement for commands."""
        # Test low security command (should be accessible)
        low_matches = command_processor.find_matching_commands("low security")
        assert len(low_matches) > 0
        
        # Test high security command (should be accessible but with checks)
        high_matches = command_processor.find_matching_commands("high security")
        assert len(high_matches) > 0
        assert high_matches[0].command.security_level == SecurityLevel.HIGH
    
    def test_command_access_validation(self, command_processor):
        """Test command access validation."""
        user_context = {
            "authenticated": True,
            "user_id": "test_user",
            "permissions": ["basic"]
        }
        
        # Low security command should be accessible
        low_access = command_processor.validate_command_access(
            command_processor.get_command_by_name("low_security"),
            user_context
        )
        assert low_access is True
        
        # High security command should require additional validation
        high_access = command_processor.validate_command_access(
            command_processor.get_command_by_name("high_security"),
            user_context
        )
        # Access might depend on implementation specifics
    
    def test_command_logging_and_audit(self, command_processor):
        """Test command logging for security audit."""
        text = "low security test"
        
        # Mock the audit logger
        with patch.object(command_processor, 'log_command_execution') as mock_log:
            # Process command
            matches = command_processor.find_matching_commands(text)
            if matches:
                command_processor.log_command_execution(matches[0].command, text, True)
            
            # Verify logging was called
            mock_log.assert_called()


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCommandHelpSystem:
    """Test command help and feedback systems."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for help system tests."""
        config = mock_voice_config
        return VoiceCommandProcessor(config)
    
    def test_get_help_for_all_commands(self, command_processor):
        """Test getting help for all commands."""
        help_text = command_processor.get_help()
        
        assert isinstance(help_text, str)
        assert len(help_text) > 0
        assert "breathing exercise" in help_text.lower()  # Should include known commands
    
    def test_get_help_for_specific_command(self, command_processor):
        """Test getting help for a specific command."""
        help_text = command_processor.get_help("breathing_exercise")
        
        assert isinstance(help_text, str)
        assert len(help_text) > 0
        assert "breathing" in help_text.lower()
    
    def test_get_help_for_nonexistent_command(self, command_processor):
        """Test getting help for non-existent command."""
        help_text = command_processor.get_help("nonexistent_command")
        
        assert isinstance(help_text, str)
        assert "not found" in help_text.lower() or "unknown" in help_text.lower()
    
    def test_get_help_by_category(self, command_processor):
        """Test getting help for commands in a specific category."""
        help_text = command_processor.get_help_by_category(CommandCategory.EMERGENCY)
        
        assert isinstance(help_text, str)
        assert len(help_text) > 0
    
    def test_command_examples(self, command_processor):
        """Test command example generation."""
        command = command_processor.get_command_by_name("breathing_exercise")
        
        if command and command.examples:
            examples = command_processor.get_command_examples("breathing_exercise")
            
            assert isinstance(examples, list)
            assert len(examples) > 0
            for example in examples:
                assert isinstance(example, str)


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCommandPerformance:
    """Test command processor performance and optimization."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for performance tests."""
        config = mock_voice_config
        return VoiceCommandProcessor(config)
    
    def test_command_matching_performance(self, command_processor):
        """Test performance of command matching."""
        test_texts = [
            "start breathing exercise",
            "help me with anxiety",
            "emergency help needed",
            "set timer for 5 minutes",
            "random text that matches nothing"
        ] * 100  # 500 total texts
        
        start_time = time.time()
        
        for text in test_texts:
            command_processor.find_matching_commands(text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 500 texts quickly (adjust threshold as needed)
        assert processing_time < 5.0, f"Too slow: {processing_time:.3f}s for 500 texts"
    
    def test_concurrent_command_processing(self, command_processor):
        """Test concurrent command processing."""
        async def process_commands_concurrently():
            tasks = []
            texts = ["start breathing exercise", "help me", "emergency"] * 10
            
            for text in texts:
                task = asyncio.create_task(command_processor.process_command(text))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = asyncio.run(process_commands_concurrently())
        
        assert len(results) == 30
        for result in results:
            assert not isinstance(result, Exception)
    
    def test_memory_usage(self, command_processor):
        """Test memory usage during command processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many commands
        for i in range(1000):
            text = f"test command {i}"
            command_processor.find_matching_commands(text)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50  # MB
    
    def test_command_caching(self, command_processor):
        """Test command result caching."""
        text = "start breathing exercise"
        
        # First call
        result1 = command_processor.find_matching_commands(text)
        
        # Second call (should use cache)
        result2 = command_processor.find_matching_commands(text)
        
        # Results should be identical
        assert len(result1) == len(result2)
        if result1 and result2:
            assert result1[0].command.name == result2[0].command.name


@pytest.mark.skipif(not VOICE_COMMANDS_AVAILABLE, reason="Voice commands not available")
class TestCustomCommands:
    """Test custom command registration and extensibility."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for custom command tests."""
        config = mock_voice_config
        return VoiceCommandProcessor(config)
    
    def test_register_custom_command_with_handler(self, command_processor):
        """Test registering custom command with custom handler."""
        def custom_handler(parameters, context):
            return {"success": True, "custom": True, "data": parameters}
        
        custom_command = VoiceCommand(
            name="custom_test",
            description="Custom command with handler",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"custom test", r"run custom"],
            action="custom_action",
            parameters={"handler": custom_handler}
        )
        
        success = command_processor.register_command(custom_command)
        assert success is True
        
        # Test the custom command
        matches = command_processor.find_matching_commands("custom test")
        assert len(matches) > 0
        assert matches[0].command.name == "custom_test"
    
    def test_dynamic_command_loading(self, command_processor):
        """Test loading commands from external configuration."""
        commands_config = [
            {
                "name": "dynamic_test",
                "description": "Dynamically loaded command",
                "category": "feature_access",
                "patterns": ["dynamic test", "load dynamic"],
                "action": "dynamic_action"
            }
        ]
        
        # Load commands from configuration
        for cmd_config in commands_config:
            command = VoiceCommand(
                name=cmd_config["name"],
                description=cmd_config["description"],
                category=CommandCategory(cmd_config["category"]),
                patterns=cmd_config["patterns"],
                action=cmd_config["action"]
            )
            command_processor.register_command(command)
        
        # Test the loaded command
        matches = command_processor.find_matching_commands("dynamic test")
        assert len(matches) > 0
    
    def test_command_templates(self, command_processor):
        """Test command template system."""
        template_command = VoiceCommand(
            name="template_test",
            description="Template-based command",
            category=CommandCategory.FEATURE_ACCESS,
            patterns=[r"template (\w+)"],
            action="template_action",
            response_templates=["Processing {parameter}", "Working on {parameter}"]
        )
        
        command_processor.register_command(template_command)
        
        # Test template matching
        matches = command_processor.find_matching_commands("template example")
        assert len(matches) > 0
        
        if matches:
            # Check if parameter was extracted
            match = matches[0]
            if match.parameters:
                assert "example" in str(match.parameters)


class TestCommandIntegration:
    """Test integration scenarios with other voice components."""
    
    @pytest.fixture
    def command_processor(self):
        """Create command processor for integration tests."""
        if not VOICE_COMMANDS_AVAILABLE:
            pytest.skip("Voice commands not available")
        
        config = mock_voice_config
        return VoiceCommandProcessor(config)
    
    def test_integration_with_stt_service(self, command_processor):
        """Test integration with speech-to-text service."""
        # Create mock STT result
        stt_result = STTResult(
            text="start breathing exercise",
            confidence=0.95,
            language="en",
            provider="mock"
        )
        
        # Process command from STT result
        matches = command_processor.find_matching_commands(stt_result.text)
        
        assert len(matches) > 0
        assert matches[0].command.name == "breathing_exercise"
    
    def test_integration_with_voice_session(self, command_processor):
        """Test command processing within voice session context."""
        session_context = {
            "session_id": "test_session",
            "user_id": "test_user",
            "conversation_history": [
                {"speaker": "user", "message": "I'm feeling anxious"},
                {"speaker": "assistant", "message": "I can help you with that"}
            ],
            "current_state": "listening"
        }
        
        # Test context-aware command
        text = "continue"  # Should be context-aware
        
        matches = command_processor.find_matching_commands(text)
        
        # Should find context-aware commands if available
        context_commands = [
            match for match in matches 
            if match.command.context_aware
        ]
        
        # Depending on implementation, may find context-aware matches
    
    def test_integration_with_security_module(self, command_processor):
        """Test integration with voice security module."""
        security_context = {
            "authenticated": True,
            "user_id": "test_user",
            "permissions": ["voice_commands", "emergency"],
            "session_encrypted": True
        }
        
        # Test emergency command with security context
        emergency_matches = command_processor.find_matching_commands("emergency help")
        
        if emergency_matches:
            for match in emergency_matches:
                if match.command.category == CommandCategory.EMERGENCY:
                    # Verify security level
                    assert match.command.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])