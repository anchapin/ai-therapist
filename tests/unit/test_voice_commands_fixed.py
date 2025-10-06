"""
Comprehensive unit tests for voice/commands.py module.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import re

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.commands import (
        VoiceCommandProcessor, VoiceCommand, CommandMatch, CommandCategory,
        SecurityLevel, AudioData, STTResult
    )
    from voice.config import VoiceConfig
except ImportError as e:
    pytest.skip(f"voice.commands module not available: {e}", allow_module_level=True)


class TestVoiceCommand:
    """Test VoiceCommand dataclass."""
    
    def test_voice_command_creation(self):
        """Test creating a voice command."""
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.NAVIGATION,
            patterns=[r"test pattern"],
            action="test_action"
        )
        
        assert command.name == "test_command"
        assert command.description == "Test command"
        assert command.category == CommandCategory.NAVIGATION
        assert command.patterns == [r"test pattern"]
        assert command.action == "test_action"
        assert command.security_level == SecurityLevel.LOW
        assert command.enabled == True
        assert command.cooldown == 0
        assert command.priority == 0
        assert command.require_wake_word == True
        assert command.context_aware == False
        assert command.confidence_threshold == 0.6
    
    def test_voice_command_with_parameters(self):
        """Test creating a voice command with parameters."""
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.SETTINGS,
            patterns=[r"test pattern"],
            action="test_action",
            parameters={"param1": "value1"},
            examples=["example1", "example2"],
            security_level=SecurityLevel.MEDIUM,
            cooldown=5,
            priority=10,
            require_wake_word=False,
            context_aware=True,
            confidence_threshold=0.8,
            emergency_keywords=["emergency"],
            response_templates=["response1"],
            fallback_commands=["fallback1"],
            metadata={"meta": "data"}
        )
        
        assert command.parameters == {"param1": "value1"}
        assert command.examples == ["example1", "example2"]
        assert command.security_level == SecurityLevel.MEDIUM
        assert command.cooldown == 5
        assert command.priority == 10
        assert command.require_wake_word == False
        assert command.context_aware == True
        assert command.confidence_threshold == 0.8
        assert command.emergency_keywords == ["emergency"]
        assert command.response_templates == ["response1"]
        assert command.fallback_commands == ["fallback1"]
        assert command.metadata == {"meta": "data"}


class TestCommandMatch:
    """Test CommandMatch dataclass."""
    
    def test_command_match_creation(self):
        """Test creating a command match."""
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.NAVIGATION,
            patterns=[r"test pattern"],
            action="test_action"
        )
        
        match = CommandMatch(
            command=command,
            confidence=0.9,
            parameters={"param": "value"},
            matched_text="test text",
            timestamp=time.time(),
            context_score=0.5,
            alternative_matches=[{"cmd": "alt1"}],
            processing_time=0.1,
            session_id="session123",
            user_id="user123",
            is_emergency=False,
            crisis_keywords_detected=[],
            match_method="pattern"
        )
        
        assert match.command == command
        assert match.confidence == 0.9
        assert match.parameters == {"param": "value"}
        assert match.matched_text == "test text"
        assert match.context_score == 0.5
        assert match.alternative_matches == [{"cmd": "alt1"}]
        assert match.processing_time == 0.1
        assert match.session_id == "session123"
        assert match.user_id == "user123"
        assert match.is_emergency == False
        assert match.crisis_keywords_detected == []
        assert match.match_method == "pattern"


class TestVoiceCommandProcessor:
    """Test VoiceCommandProcessor class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock voice config."""
        config = Mock(spec=VoiceConfig)
        config.voice_commands_enabled = True
        config.voice_command_wake_word = "hey therapist"
        config.voice_command_timeout = 30000
        config.voice_command_min_confidence = 0.6
        return config
    
    @pytest.fixture
    def processor(self, mock_config):
        """Create a voice command processor with mock config."""
        return VoiceCommandProcessor(mock_config)
    
    def test_processor_initialization(self, processor, mock_config):
        """Test processor initialization."""
        assert processor.config == mock_config
        assert processor.wake_word_enabled == True
        assert processor.wake_word == "hey therapist"
        assert processor.wake_word_detected == False
        assert len(processor.commands) > 0  # Should have default commands
        assert len(processor.command_handlers) > 0
        assert len(processor.emergency_handlers) > 0
        assert processor.metrics['total_commands_processed'] == 0
        assert processor.metrics['successful_commands'] == 0
        assert processor.metrics['emergency_commands_triggered'] == 0
    
    def test_register_command(self, processor):
        """Test registering a new command."""
        new_command = VoiceCommand(
            name="new_test_command",
            description="New test command",
            category=CommandCategory.HELP,
            patterns=[r"new test pattern"],
            action="new_test_action"
        )
        
        processor.register_command(new_command)
        
        assert "new_test_command" in processor.commands
        assert processor.commands["new_test_command"] == new_command
        assert "new_test_command" in processor.commands_by_category[CommandCategory.HELP]
    
    def test_unregister_command(self, processor):
        """Test unregistering a command."""
        # First register a command
        new_command = VoiceCommand(
            name="temp_command",
            description="Temporary command",
            category=CommandCategory.HELP,
            patterns=[r"temp pattern"],
            action="temp_action"
        )
        processor.register_command(new_command)
        
        # Verify it was registered
        assert "temp_command" in processor.commands
        assert "temp_command" in processor.commands_by_category[CommandCategory.HELP]
        
        # Then unregister it
        processor.unregister_command("temp_command")
        
        assert "temp_command" not in processor.commands
        # Note: commands_by_category is not automatically cleaned up in the implementation
    
    def test_register_command_handler(self, processor):
        """Test registering a custom command handler."""
        async def custom_handler(params):
            return {"custom": "response"}
        
        processor.register_command_handler("custom_action", custom_handler)
        
        assert "custom_action" in processor.command_handlers
        assert processor.command_handlers["custom_action"] == custom_handler
    
    def test_get_available_commands(self, processor):
        """Test getting available commands."""
        commands = processor.get_available_commands()
        
        assert isinstance(commands, list)
        assert len(commands) > 0
        
        for cmd in commands:
            assert 'name' in cmd
            assert 'description' in cmd
            assert 'patterns' in cmd
            assert 'action' in cmd
            assert 'examples' in cmd
            assert 'security_level' in cmd
    
    def test_check_wake_word_in_text(self, processor):
        """Test wake word detection in text."""
        # Test with wake word present
        assert processor._check_wake_word_in_text("hey therapist help me") == True
        assert processor._check_wake_word_in_text("Hey Therapist, can you help?") == True
        
        # Test without wake word
        assert processor._check_wake_word_in_text("help me") == False
        assert processor._check_wake_word_in_text("hello assistant") == False
    
    def test_detect_emergency_keywords(self, processor):
        """Test emergency keyword detection."""
        # Test with emergency keywords
        emergency_keywords = processor._detect_emergency_keywords("I want to kill myself")
        assert len(emergency_keywords) > 0
        assert any("kill myself" in keyword for keyword in emergency_keywords)
        
        # Test without emergency keywords
        emergency_keywords = processor._detect_emergency_keywords("I want to talk about my day")
        assert len(emergency_keywords) == 0
    
    def test_classify_emergency_type(self, processor):
        """Test emergency type classification."""
        # Test suicide prevention
        suicide_type = processor._classify_emergency_type(["suicide", "kill myself"])
        assert suicide_type == "suicide_prevention"
        
        # Test immediate danger
        danger_type = processor._classify_emergency_type(["danger", "violence"])
        assert danger_type == "immediate_danger"
        
        # Test general crisis
        crisis_type = processor._classify_emergency_type(["help", "crisis"])
        assert crisis_type == "crisis_intervention"
        
        # Test no keywords
        default_type = processor._classify_emergency_type([])
        assert default_type == "crisis_intervention"
    
    def test_calculate_confidence(self, processor):
        """Test confidence calculation."""
        # Create a mock match
        pattern = r"test pattern"
        text = "this is a test pattern"
        match = re.search(pattern, text, re.IGNORECASE)
        
        confidence = processor._calculate_confidence(match, text, pattern)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # Should have some confidence for a good match
    
    def test_extract_parameters(self, processor):
        """Test parameter extraction."""
        # Create a command with parameters
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.VOICE_CONTROL,
            patterns=[r"adjust volume (?P<direction>up|down)"],
            action="test_action",
            parameters={"default": "value"}
        )
        
        # Create a match
        text = "adjust volume up"
        match = re.search(command.patterns[0], text, re.IGNORECASE)
        
        parameters = processor._extract_parameters(match, command)
        
        assert "direction" in parameters
        assert parameters["direction"] == "up"
        assert "default" in parameters
        assert parameters["default"] == "value"
    
    def test_extract_volume_parameters(self, processor):
        """Test volume parameter extraction."""
        # Test volume up
        params = processor._extract_volume_parameters("turn volume up a lot")
        assert params.get("direction") == "up"
        assert params.get("magnitude") == "large"
        
        # Test volume down
        params = processor._extract_volume_parameters("make it softer a bit")
        assert params.get("direction") == "down"
        assert params.get("magnitude") == "small"
        
        # Test medium adjustment
        params = processor._extract_volume_parameters("adjust volume")
        assert params.get("direction") == "up"  # Default
        assert params.get("magnitude") == "medium"
    
    def test_extract_voice_parameters(self, processor):
        """Test voice parameter extraction."""
        # Test male voice
        params = processor._extract_voice_parameters("change to male voice")
        assert params.get("voice_type") == "male"
        
        # Test calm voice
        params = processor._extract_voice_parameters("use a calm voice")
        assert params.get("voice_type") == "calm"
        
        # Test no voice type
        params = processor._extract_voice_parameters("change voice")
        assert "voice_type" not in params
    
    def test_extract_emergency_parameters(self, processor):
        """Test emergency parameter extraction."""
        # Test high urgency
        params = processor._extract_emergency_parameters("immediate help needed")
        assert params.get("urgency") == "high"
        # emergency_keywords might be empty if no specific keywords are found
        
        # Test medium urgency
        params = processor._extract_emergency_parameters("need help")
        assert params.get("urgency") == "medium"
        
        # Test low urgency
        params = processor._extract_emergency_parameters("feeling sad")
        assert params.get("urgency") == "low"
    
    def test_extract_meditation_parameters(self, processor):
        """Test meditation parameter extraction."""
        # Test short breathing meditation
        params = processor._extract_meditation_parameters("quick breathing exercise")
        assert params.get("duration") == "short"
        assert params.get("type") == "breathing"
        
        # Test long mindfulness meditation
        params = processor._extract_meditation_parameters("long mindfulness session")
        assert params.get("duration") == "long"
        assert params.get("type") == "mindfulness"
        
        # Test general meditation
        params = processor._extract_meditation_parameters("start meditation")
        assert params.get("duration") == "medium"
        assert params.get("type") == "general"
    
    @pytest.mark.asyncio
    async def test_process_text_with_wake_word(self, processor):
        """Test text processing with wake word."""
        processor.wake_word_enabled = True
        processor.wake_word_detected = False
        
        result = await processor.process_text("hey therapist")
        
        assert result is not None
        assert result.command.name == "wake_word"
        assert result.confidence == 0.9
        assert processor.wake_word_detected == True
    
    @pytest.mark.asyncio
    async def test_process_text_with_emergency(self, processor):
        """Test text processing with emergency keywords."""
        processor.wake_word_enabled = True
        processor.wake_word_detected = False
        
        result = await processor.process_text("I want to kill myself")
        
        assert result is not None
        assert result.is_emergency == True
        assert len(result.crisis_keywords_detected) > 0
        assert result.command.category == CommandCategory.EMERGENCY
    
    @pytest.mark.asyncio
    async def test_process_text_with_command(self, processor):
        """Test text processing with regular command."""
        processor.wake_word_enabled = False
        
        result = await processor.process_text("what can you do")
        
        assert result is not None
        assert result.command.name == "get_help"
        assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_process_text_disabled(self, processor):
        """Test text processing when disabled."""
        processor.config.voice_commands_enabled = False
        
        result = await processor.process_text("help")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_match_command(self, processor):
        """Test command matching."""
        processor.wake_word_enabled = False
        
        result = await processor._match_command("start a new therapy session")
        
        assert result is not None
        assert result.command.name == "start_session"
        assert result.confidence > 0.0
        assert "start" in result.matched_text.lower()
    
    @pytest.mark.asyncio
    async def test_match_command_with_priority(self, processor):
        """Test command matching with priority."""
        # Create two commands with different priorities
        high_priority = VoiceCommand(
            name="high_priority_test",
            description="High priority command",
            category=CommandCategory.HELP,
            patterns=[r"test help me"],
            action="high_action",
            priority=100
        )
        
        low_priority = VoiceCommand(
            name="low_priority_test",
            description="Low priority command",
            category=CommandCategory.HELP,
            patterns=[r"test help me"],
            action="low_action",
            priority=1
        )
        
        processor.register_command(high_priority)
        processor.register_command(low_priority)
        
        result = await processor._match_command("test help me")
        
        assert result is not None
        assert result.command.name == "high_priority_test"  # Should match higher priority
    
    @pytest.mark.asyncio
    async def test_match_command_with_cooldown(self, processor):
        """Test command matching with cooldown."""
        # Create a command with cooldown
        cooldown_command = VoiceCommand(
            name="cooldown_test",
            description="Cooldown test command",
            category=CommandCategory.HELP,
            patterns=[r"test cooldown now"],
            action="cooldown_action",
            cooldown=1  # 1 second cooldown
        )
        
        processor.register_command(cooldown_command)
        
        # First call should match
        result1 = await processor._match_command("test cooldown now")
        assert result1 is not None
        assert result1.command.name == "cooldown_test"
        
        # Immediate second call should not match due to cooldown
        result2 = await processor._match_command("test cooldown now")
        assert result2 is None
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # Third call should match again
        result3 = await processor._match_command("test cooldown now")
        assert result3 is not None
        assert result3.command.name == "cooldown_test"
    
    @pytest.mark.asyncio
    async def test_execute_command_success(self, processor):
        """Test successful command execution."""
        # Create a command match
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.HELP,
            patterns=[r"test"],
            action="show_help"
        )
        
        match = CommandMatch(
            command=command,
            confidence=0.9,
            parameters={},
            matched_text="test",
            timestamp=time.time()
        )
        
        result = await processor.execute_command(match)
        
        assert result is not None
        assert result['success'] == True
        assert result['command'] == "test_command"
        assert result['action'] == "show_help"
        assert 'result' in result
        assert 'processing_time' in result
        assert processor.metrics['successful_commands'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_command_no_handler(self, processor):
        """Test command execution with no handler."""
        # Create a command with no handler
        command = VoiceCommand(
            name="no_handler_command",
            description="No handler command",
            category=CommandCategory.HELP,
            patterns=[r"no handler"],
            action="no_handler_action"
        )
        
        match = CommandMatch(
            command=command,
            confidence=0.9,
            parameters={},
            matched_text="no handler",
            timestamp=time.time()
        )
        
        result = await processor.execute_command(match)
        
        assert result is not None
        assert result['success'] == False
        assert result['error'] == 'No handler found'
        assert processor.metrics['successful_commands'] == 0
    
    @pytest.mark.asyncio
    async def test_execute_command_exception(self, processor):
        """Test command execution with exception."""
        # Create a handler that raises an exception
        async def failing_handler(params):
            raise ValueError("Test error")
        
        processor.register_command_handler("failing_action", failing_handler)
        
        command = VoiceCommand(
            name="failing_command",
            description="Failing command",
            category=CommandCategory.HELP,
            patterns=[r"failing"],
            action="failing_action"
        )
        
        match = CommandMatch(
            command=command,
            confidence=0.9,
            parameters={},
            matched_text="failing",
            timestamp=time.time()
        )
        
        result = await processor.execute_command(match)
        
        assert result is not None
        assert result['success'] == False
        assert 'error' in result
        assert processor.metrics['successful_commands'] == 0
    
    @pytest.mark.asyncio
    async def test_handle_start_session(self, processor):
        """Test start session handler."""
        result = await processor._handle_start_session({})
        
        assert result is not None
        assert result['action'] == 'start_session'
        assert 'voice_feedback' in result
        assert "I'm here to listen" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_end_session(self, processor):
        """Test end session handler."""
        result = await processor._handle_end_session({})
        
        assert result is not None
        assert result['action'] == 'end_session'
        assert 'voice_feedback' in result
        assert "Thank you for sharing" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_emergency_response(self, processor):
        """Test emergency response handler."""
        result = await processor._handle_emergency_response({})
        
        assert result is not None
        assert result['action'] == 'emergency_response'
        assert result['severity'] == 'critical'
        assert 'voice_feedback' in result
        assert 'resources' in result
        assert processor.metrics['emergency_commands_triggered'] == 1
    
    @pytest.mark.asyncio
    async def test_handle_suicide_prevention(self, processor):
        """Test suicide prevention handler."""
        result = await processor._handle_suicide_prevention({})
        
        assert result is not None
        assert 'voice_feedback' in result
        assert 'resources' in result
        assert 'immediate_actions' in result
        assert "988" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_crisis_intervention(self, processor):
        """Test crisis intervention handler."""
        result = await processor._handle_crisis_intervention({})
        
        assert result is not None
        assert 'voice_feedback' in result
        assert 'resources' in result
        assert 'immediate_actions' in result
        assert "741741" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_emergency_contact(self, processor):
        """Test emergency contact handler."""
        result = await processor._handle_emergency_contact({})
        
        assert result is not None
        assert 'voice_feedback' in result
        assert 'resources' in result
        assert 'immediate_actions' in result
        assert "911" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_immediate_danger(self, processor):
        """Test immediate danger handler."""
        result = await processor._handle_immediate_danger({})
        
        assert result is not None
        assert 'voice_feedback' in result
        assert 'resources' in result
        assert 'immediate_actions' in result
        assert "911" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_adjust_speech_speed(self, processor):
        """Test adjust speech speed handler."""
        result = await processor._handle_adjust_speech_speed({"speed": "slower"})
        
        assert result is not None
        assert result['action'] == 'adjust_speech_speed'
        assert result['parameters']['speed'] == "slower"
        assert "slower" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_change_voice_profile(self, processor):
        """Test change voice profile handler."""
        result = await processor._handle_change_voice_profile({"voice_type": "male"})
        
        assert result is not None
        assert result['action'] == 'change_voice_profile'
        assert result['parameters']['voice_type'] == "male"
        assert "male" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_show_help(self, processor):
        """Test show help handler."""
        result = await processor._handle_show_help({})
        
        assert result is not None
        assert result['action'] == 'show_help'
        assert 'voice_feedback' in result
        assert "commands" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_adjust_volume(self, processor):
        """Test adjust volume handler."""
        result = await processor._handle_adjust_volume({"direction": "up"})
        
        assert result is not None
        assert result['action'] == 'adjust_volume'
        assert result['parameters']['direction'] == "up"
        assert "up" in result['voice_feedback']
    
    @pytest.mark.asyncio
    async def test_handle_check_status(self, processor):
        """Test check status handler."""
        result = await processor._handle_check_status({})
        
        assert result is not None
        assert result['action'] == 'check_status'
        assert 'voice_feedback' in result
        assert "working normally" in result['voice_feedback']
    
    def test_get_command_history(self, processor):
        """Test getting command history."""
        # Add some mock history
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.HELP,
            patterns=[r"test"],
            action="test_action"
        )
        
        match = CommandMatch(
            command=command,
            confidence=0.9,
            parameters={},
            matched_text="test",
            timestamp=time.time()
        )
        
        processor.command_history.append(match)
        
        history = processor.get_command_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert 'command' in history[0]
        assert 'confidence' in history[0]
        assert 'matched_text' in history[0]
        assert 'timestamp' in history[0]
    
    def test_clear_command_history(self, processor):
        """Test clearing command history."""
        # Add some mock history
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.HELP,
            patterns=[r"test"],
            action="test_action"
        )
        
        match = CommandMatch(
            command=command,
            confidence=0.9,
            parameters={},
            matched_text="test",
            timestamp=time.time()
        )
        
        processor.command_history.append(match)
        assert len(processor.command_history) > 0
        
        processor.clear_command_history()
        assert len(processor.command_history) == 0
    
    def test_get_statistics(self, processor):
        """Test getting processor statistics."""
        stats = processor.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_commands' in stats
        assert 'enabled_commands' in stats
        assert 'command_history_size' in stats
        assert 'wake_word_enabled' in stats
        assert 'wake_word_detected' in stats
        assert 'last_wake_word_time' in stats
        assert 'min_confidence' in stats
    
    def test_get_audit_log(self, processor):
        """Test getting audit log."""
        # Add some mock audit entries
        processor.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': 'test_event',
            'command_name': 'test_command'
        })
        
        audit_log = processor.get_audit_log()
        
        assert isinstance(audit_log, list)
        assert len(audit_log) > 0
        assert 'timestamp' in audit_log[0]
        assert 'event_type' in audit_log[0]
    
    def test_get_command_analytics(self, processor):
        """Test getting command analytics."""
        # Add some mock statistics
        processor.metrics['total_commands_processed'] = 10
        processor.metrics['successful_commands'] = 8
        processor.metrics['average_confidence'] = 0.85
        
        analytics = processor.get_command_analytics()
        
        assert isinstance(analytics, dict)
        assert 'total_commands' in analytics
        assert 'success_rate' in analytics
        assert 'average_confidence' in analytics
        assert 'top_commands' in analytics
        assert 'category_usage' in analytics
        assert 'recent_activity' in analytics
    
    def test_cleanup(self, processor):
        """Test processor cleanup."""
        # Add some data to clean up
        processor.command_history.append({"test": "data"})
        processor.audit_log.append({"test": "data"})
        processor.conversation_context["test"] = {"data": "test"}
        
        # Verify data exists
        assert len(processor.command_history) > 0
        assert len(processor.audit_log) > 0
        assert len(processor.conversation_context) > 0
        
        # Cleanup
        processor.cleanup()
        
        # Verify data is cleared
        assert len(processor.command_history) == 0
        assert len(processor.audit_log) == 0
        assert len(processor.conversation_context) == 0
    
    @pytest.mark.asyncio
    async def test_update_conversation_context(self, processor):
        """Test updating conversation context."""
        session_id = "test_session"
        context = {
            'current_activity': 'meditation',
            'user_preferences': {'frequent_commands': ['start_meditation']},
            'session_duration': 300
        }
        
        await processor._update_conversation_context(session_id, context)
        
        assert session_id in processor.conversation_context
        assert processor.conversation_context[session_id] == context
    
    @pytest.mark.asyncio
    async def test_get_context_score(self, processor):
        """Test getting context score."""
        session_id = "test_session"
        command = VoiceCommand(
            name="start_meditation",
            description="Start meditation",
            category=CommandCategory.MEDITATION,
            patterns=[r"start meditation"],
            action="start_meditation"
        )
        
        # Set up context
        context = {
            'current_activity': 'meditation',
            'user_preferences': {'frequent_commands': ['start_meditation']},
            'session_duration': 400
        }
        await processor._update_conversation_context(session_id, context)
        
        score = await processor._get_context_score(session_id, command)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have some relevance score
    
    @pytest.mark.asyncio
    async def test_log_emergency_event(self, processor):
        """Test logging emergency events."""
        parameters = {
            'matched_text': 'I want to kill myself',
            'session_id': 'test_session',
            'user_id': 'test_user'
        }
        
        await processor._log_emergency_event("test_emergency", parameters)
        
        assert len(processor.audit_log) > 0
        assert processor.audit_log[-1]['event_type'] == "test_emergency"
        assert processor.audit_log[-1]['severity'] == "critical"
        assert processor.metrics['emergency_commands_triggered'] == 1
    
    def test_update_command_statistics(self, processor):
        """Test updating command statistics."""
        command_name = "test_command"
        
        # First execution
        processor._update_command_statistics(command_name, True, 0.9)
        stats = processor.command_stats[command_name]
        assert stats['executions'] == 1
        assert stats['success_rate'] == 1.0
        assert stats['average_confidence'] == 0.9
        
        # Second execution
        processor._update_command_statistics(command_name, False, 0.7)
        stats = processor.command_stats[command_name]
        assert stats['executions'] == 2
        assert stats['success_rate'] == 0.5  # (1.0 + 0.0) / 2
        assert stats['average_confidence'] == 0.8  # (0.9 + 0.7) / 2
    
    @pytest.mark.asyncio
    async def test_process_audio_disabled(self, processor):
        """Test audio processing when disabled."""
        processor.config.voice_commands_enabled = False
        
        audio_data = Mock(spec=AudioData)
        result = await processor.process_audio(audio_data)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_audio_with_wake_word(self, processor):
        """Test audio processing with wake word detection."""
        processor.wake_word_enabled = True
        processor.wake_word_detected = False
        
        # Mock wake word detection
        with patch.object(processor, '_detect_wake_word', return_value=True):
            audio_data = Mock(spec=AudioData)
            result = await processor.process_audio(audio_data)
            
            assert result is not None
            assert result.command.name == "wake_word"
            assert processor.wake_word_detected == True
    
    @pytest.mark.asyncio
    async def test_process_audio_timeout(self, processor):
        """Test audio processing with wake word timeout."""
        processor.wake_word_enabled = True
        processor.wake_word_detected = True
        processor.last_wake_word_time = time.time() - 40  # 40 seconds ago (past 30s timeout)
        
        audio_data = Mock(spec=AudioData)
        result = await processor.process_audio(audio_data)
        
        assert result is None
        assert processor.wake_word_detected == False
    
    @pytest.mark.asyncio
    async def test_calculate_enhanced_confidence(self, processor):
        """Test enhanced confidence calculation."""
        command = VoiceCommand(
            name="test_command",
            description="Test command",
            category=CommandCategory.HELP,
            patterns=[r"test pattern"],
            action="test_action",
            priority=10
        )
        
        pattern = r"test pattern"
        text = "this is a test pattern"
        match = re.search(pattern, text, re.IGNORECASE)
        
        confidence = await processor._calculate_enhanced_confidence(match, text, command, 0.5)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_extract_enhanced_parameters(self, processor):
        """Test enhanced parameter extraction."""
        command = VoiceCommand(
            name="adjust_volume",
            description="Adjust volume",
            category=CommandCategory.VOICE_CONTROL,
            patterns=[r"adjust volume"],
            action="adjust_volume"
        )
        
        text = "adjust volume up a lot"
        match = re.search(command.patterns[0], text, re.IGNORECASE)
        
        parameters = await processor._extract_enhanced_parameters(match, command, text)
        
        assert "direction" in parameters
        assert parameters["direction"] == "up"
        assert "magnitude" in parameters
        assert parameters["magnitude"] == "large"