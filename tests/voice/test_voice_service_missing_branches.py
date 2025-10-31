"""
Comprehensive tests for VoiceService missing branch coverage.
Tests for add_conversation_entry, update_voice_settings, health_check, and queue handlers.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os

# Mock problematic imports before loading voice modules
if 'torch' not in sys.modules:
    sys.modules['torch'] = MagicMock()
if 'whisper' not in sys.modules:
    sys.modules['whisper'] = MagicMock()
if 'langchain_ollama' not in sys.modules:
    sys.modules['langchain_ollama'] = MagicMock()
if 'langchain_core' not in sys.modules:
    sys.modules['langchain_core'] = MagicMock()
if 'langchain_core.language_models' not in sys.modules:
    sys.modules['langchain_core.language_models'] = MagicMock()
if 'langchain_core.prompt_values' not in sys.modules:
    sys.modules['langchain_core.prompt_values'] = MagicMock()
if 'app' not in sys.modules:
    sys.modules['app'] = MagicMock()

from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.config import VoiceConfig
from voice.audio_processor import AudioData


class TestVoiceServiceMissingBranches:
    """Test VoiceService missing branch coverage."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock voice configuration."""
        config = Mock(spec=VoiceConfig)
        config.stt_enabled = True
        config.tts_enabled = True
        config.commands_enabled = True
        config.security_enabled = True
        config.max_session_duration = 3600
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.session_timeout = 300
        config.default_voice_profile = "calm_therapist"
        return config
    
    @pytest.fixture
    def mock_security(self):
        """Create a mock security service."""
        security = Mock()
        security.encrypt_audio = AsyncMock(return_value=b'encrypted_audio')
        security.decrypt_audio = AsyncMock(return_value=b'decrypted_audio')
        security.sanitize_transcription = Mock(return_value="sanitized text")
        security.is_healthy = True
        return security
    
    @pytest.fixture
    def voice_service(self, mock_config, mock_security):
        """Create a VoiceService instance with mocked dependencies."""
        with patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'), \
             patch('voice.voice_service.VoiceCommandProcessor'):
            
            service = VoiceService(mock_config, mock_security)
            service.session_repo = Mock()
            service.voice_data_repo = Mock()
            service.audit_repo = Mock()
            service.consent_repo = Mock()
            service._db_initialized = True
            
            return service

    # Tests for add_conversation_entry
    def test_add_conversation_entry_old_convention(self, voice_service):
        """Test add_conversation_entry with old calling convention (session_id, speaker, text)."""
        session_id = voice_service.create_session(user_id="test_user")
        
        # Old convention: session_id, speaker, text
        result = voice_service.add_conversation_entry(session_id, "user", "Hello therapist")
        
        assert result is True
        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0]['speaker'] == 'user'
        assert history[0]['text'] == 'Hello therapist'
        assert 'timestamp' in history[0]

    def test_add_conversation_entry_new_convention(self, voice_service):
        """Test add_conversation_entry with new calling convention (session_id, dict)."""
        session_id = voice_service.create_session(user_id="test_user")
        
        # New convention: session_id, entry_dict
        entry = {
            'type': 'user_input',
            'text': 'I feel anxious',
            'confidence': 0.95,
            'timestamp': time.time()
        }
        result = voice_service.add_conversation_entry(session_id, entry)
        
        assert result is True
        history = voice_service.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0]['type'] == 'user_input'
        assert history[0]['speaker'] == 'user'
        assert history[0]['text'] == 'I feel anxious'

    def test_add_conversation_entry_invalid_args_too_few(self, voice_service):
        """Test add_conversation_entry with invalid arguments (too few)."""
        session_id = voice_service.create_session(user_id="test_user")
        
        # Only one argument - should fail
        result = voice_service.add_conversation_entry(session_id)
        
        assert result is False

    def test_add_conversation_entry_invalid_args_wrong_count(self, voice_service):
        """Test add_conversation_entry with invalid arguments (wrong count)."""
        session_id = voice_service.create_session(user_id="test_user")
        
        # Four arguments - invalid
        result = voice_service.add_conversation_entry(session_id, "user", "text", "extra")
        
        assert result is False

    def test_add_conversation_entry_invalid_session_id_type(self, voice_service):
        """Test add_conversation_entry with invalid session_id type."""
        # Pass a dict instead of string for session_id
        result = voice_service.add_conversation_entry({'invalid': 'session'}, "user", "text")
        
        assert result is False

    def test_add_conversation_entry_nonexistent_session(self, voice_service):
        """Test add_conversation_entry with non-existent session."""
        result = voice_service.add_conversation_entry("nonexistent_session", "user", "text")
        
        assert result is False

    def test_add_conversation_entry_metrics_increment(self, voice_service):
        """Test that add_conversation_entry increments metrics correctly."""
        session_id = voice_service.create_session(user_id="test_user")
        initial_interactions = voice_service.metrics['total_interactions']
        
        # Add user input - should increment
        voice_service.add_conversation_entry(session_id, {
            'type': 'user_input',
            'text': 'Hello'
        })
        assert voice_service.metrics['total_interactions'] == initial_interactions + 1
        
        # Add assistant output - should increment
        voice_service.add_conversation_entry(session_id, {
            'type': 'assistant_output',
            'text': 'Hi there'
        })
        assert voice_service.metrics['total_interactions'] == initial_interactions + 2

    def test_add_conversation_entry_ai_speaker_mapping(self, voice_service):
        """Test add_conversation_entry maps speaker='ai' correctly."""
        session_id = voice_service.create_session(user_id="test_user")
        
        voice_service.add_conversation_entry(session_id, "ai", "How are you feeling?")
        
        history = voice_service.get_conversation_history(session_id)
        assert history[0]['speaker'] == 'ai'
        assert history[0]['type'] == 'assistant_response'

    # Tests for update_voice_settings
    def test_update_voice_settings_parameter_order_1(self, voice_service):
        """Test update_voice_settings with (settings, session_id) order."""
        session_id = voice_service.create_session(user_id="test_user")
        
        settings = {'voice_speed': 1.5, 'volume': 0.8}
        result = voice_service.update_voice_settings(settings, session_id)
        
        assert result is True
        session = voice_service.get_session(session_id)
        assert session.metadata['voice_settings']['voice_speed'] == 1.5
        assert session.metadata['voice_settings']['volume'] == 0.8

    def test_update_voice_settings_parameter_order_2(self, voice_service):
        """Test update_voice_settings with (session_id, settings) order."""
        session_id = voice_service.create_session(user_id="test_user")
        
        settings = {'voice_pitch': 1.3}
        result = voice_service.update_voice_settings(session_id, settings)
        
        assert result is True
        session = voice_service.get_session(session_id)
        assert session.metadata['voice_settings']['voice_pitch'] == 1.3

    def test_update_voice_settings_invalid_session_id_type(self, voice_service):
        """Test update_voice_settings with invalid session_id type (dict instead of string)."""
        settings = {'voice_speed': 1.2}
        
        # Pass a dict for session_id - should fail with unhashable type error
        result = voice_service.update_voice_settings({'invalid': 'session'}, settings)
        
        assert result is False

    def test_update_voice_settings_nonexistent_session(self, voice_service):
        """Test update_voice_settings with non-existent session."""
        settings = {'voice_speed': 1.2}
        
        # Non-existent session should still return True (doesn't update anything)
        result = voice_service.update_voice_settings(settings, "nonexistent_session")
        
        assert result is True  # Method returns True even if session not found

    def test_update_voice_settings_no_session_id(self, voice_service):
        """Test update_voice_settings with only settings (no session_id)."""
        settings = {'voice_speed': 1.4}
        
        result = voice_service.update_voice_settings(settings)
        
        assert result is True  # Should succeed even without session

    # Tests for health_check
    def test_health_check_component_degraded_stt_mock(self, voice_service):
        """Test health_check when STT service has no transcribe_audio method (mock service)."""
        # Remove both health_check and transcribe_audio methods from STT service
        # to trigger the mock detection path
        if hasattr(voice_service.stt_service, 'health_check'):
            delattr(voice_service.stt_service, 'health_check')
        if hasattr(voice_service.stt_service, 'transcribe_audio'):
            delattr(voice_service.stt_service, 'transcribe_audio')
        
        health = voice_service.health_check()
        
        assert 'stt_service' in health
        assert health['stt_service']['status'] == 'mock'
        assert 'Using mock STT service' in health['stt_service']['issues']

    def test_health_check_component_degraded_tts_mock(self, voice_service):
        """Test health_check when TTS service has no synthesize_speech method (mock service)."""
        # Remove both health_check and synthesize_speech methods from TTS service
        # to trigger the mock detection path
        if hasattr(voice_service.tts_service, 'health_check'):
            delattr(voice_service.tts_service, 'health_check')
        if hasattr(voice_service.tts_service, 'synthesize_speech'):
            delattr(voice_service.tts_service, 'synthesize_speech')
        
        health = voice_service.health_check()
        
        assert 'tts_service' in health
        assert health['tts_service']['status'] == 'mock'
        assert 'Using mock TTS service' in health['tts_service']['issues']

    def test_health_check_component_unhealthy(self, voice_service):
        """Test health_check when a component reports unhealthy status."""
        # Mock STT service to return unhealthy status
        voice_service.stt_service.health_check = Mock(return_value={'status': 'unhealthy', 'issues': ['API key invalid']})
        
        health = voice_service.health_check()
        
        assert health['overall_status'] == 'degraded'
        assert health['stt_service']['status'] == 'unhealthy'

    def test_health_check_all_components_healthy(self, voice_service):
        """Test health_check when all components are healthy."""
        # Mock all components to return healthy
        voice_service.audio_processor.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
        voice_service.stt_service.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
        voice_service.tts_service.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
        voice_service.command_processor.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
        voice_service.security.health_check = Mock(return_value={'status': 'healthy', 'issues': []})
        
        health = voice_service.health_check()
        
        assert health['overall_status'] == 'healthy'
        assert health['audio_processor']['status'] == 'healthy'
        assert health['stt_service']['status'] == 'healthy'
        assert health['tts_service']['status'] == 'healthy'

    # Tests for queue handlers
    @pytest.mark.asyncio
    async def test_handle_start_session_queue(self, voice_service):
        """Test _handle_start_session queue handler."""
        # Initialize the queue
        voice_service._ensure_queue_initialized()
        
        # Inject item into queue
        data = {'session_id': 'test_session_123'}
        await voice_service.voice_queue.put(("start_session", data))
        
        # Process one loop tick
        await voice_service._process_voice_queue()
        
        # Verify session was created
        assert 'test_session_123' in voice_service.sessions

    @pytest.mark.asyncio
    async def test_handle_stop_session_queue(self, voice_service):
        """Test _handle_stop_session queue handler."""
        # Create a session first
        session_id = voice_service.create_session(user_id="test_user")
        voice_service._ensure_queue_initialized()
        
        # Inject item into queue
        data = {'session_id': session_id}
        await voice_service.voice_queue.put(("stop_session", data))
        
        # Process one loop tick
        await voice_service._process_voice_queue()
        
        # Verify session was destroyed
        assert session_id not in voice_service.sessions

    @pytest.mark.asyncio
    async def test_handle_start_listening_queue(self, voice_service):
        """Test _handle_start_listening queue handler."""
        # Create a session first
        session_id = voice_service.create_session(user_id="test_user")
        voice_service._ensure_queue_initialized()
        
        # Mock audio processor
        voice_service.audio_processor.start_recording = Mock(return_value=True)
        
        # Inject item into queue
        data = {'session_id': session_id}
        await voice_service.voice_queue.put(("start_listening", data))
        
        # Process one loop tick
        await voice_service._process_voice_queue()
        
        # Verify session is in listening state
        session = voice_service.get_session(session_id)
        assert session.state == VoiceSessionState.LISTENING

    @pytest.mark.asyncio
    async def test_handle_stop_listening_queue(self, voice_service):
        """Test _handle_stop_listening queue handler."""
        # Create a session and start listening
        session_id = voice_service.create_session(user_id="test_user")
        voice_service.start_listening(session_id)
        voice_service._ensure_queue_initialized()
        
        # Mock audio processor
        voice_service.audio_processor.stop_recording = Mock(return_value=AudioData(b'test', 16000, 1))
        
        # Inject item into queue
        data = {'session_id': session_id}
        await voice_service.voice_queue.put(("stop_listening", data))
        
        # Process one loop tick
        await voice_service._process_voice_queue()
        
        # Verify session state changed from listening
        session = voice_service.get_session(session_id)
        assert session.state != VoiceSessionState.LISTENING

    @pytest.mark.asyncio
    async def test_handle_speak_text_queue(self, voice_service):
        """Test _handle_speak_text queue handler."""
        # Create a session first
        session_id = voice_service.create_session(user_id="test_user")
        voice_service._ensure_queue_initialized()
        
        # Mock the speak_text method
        voice_service.speak_text = AsyncMock()
        
        # Inject item into queue
        data = {
            'text': 'Hello, how are you feeling today?',
            'session_id': session_id,
            'voice_profile': 'calm'
        }
        await voice_service.voice_queue.put(("speak_text", data))
        
        # Process one loop tick
        await voice_service._process_voice_queue()
        
        # Verify speak_text was called
        voice_service.speak_text.assert_called_once_with(
            'Hello, how are you feeling today?',
            session_id,
            'calm'
        )

    @pytest.mark.asyncio
    async def test_handle_unknown_command_queue(self, voice_service):
        """Test queue handler with unknown command."""
        voice_service._ensure_queue_initialized()
        
        # Inject unknown command into queue
        data = {'test': 'data'}
        await voice_service.voice_queue.put(("unknown_command", data))
        
        # Process one loop tick - should log warning but not crash
        await voice_service._process_voice_queue()
        
        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_queue_empty_handling(self, voice_service):
        """Test queue processing when queue is empty."""
        voice_service._ensure_queue_initialized()
        
        # Process empty queue - should complete without error
        await voice_service._process_voice_queue()
        
        # Test passes if no exception is raised
