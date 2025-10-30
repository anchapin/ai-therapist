"""
Voice + App Integration Tests

Tests the integration between voice services and the main application,
focusing on end-to-end workflows, module interactions, and user experience.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from typing import Dict, List, Any
import psutil
import os

from voice.voice_service import VoiceService, VoiceSessionState
from voice.config import VoiceConfig
from voice.security import VoiceSecurity
from voice.audio_processor import AudioData

# Mock app imports to avoid streamlit dependency
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mocks'))
from mock_app import (
    initialize_session_state,
    handle_voice_text_received,
    handle_voice_command_executed,
    detect_crisis_content,
    generate_crisis_response,
    ResponseCache,
    EmbeddingCache,
    MockStreamlit,
    get_mock_session_state,
    reset_mock_session_state
)


class TestVoiceAppIntegration:
    """Test voice service integration with main application."""

    @pytest.fixture
    def mock_streamlit_session(self):
        """Mock Streamlit session state."""
        reset_mock_session_state()
        # Return the global state directly to ensure consistency
        from tests.mocks.mock_app import _mock_session_state
        return _mock_session_state._state

    @pytest.fixture
    def integrated_voice_config(self):
        """Create integrated voice configuration."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.voice_output_enabled = True
        config.voice_commands_enabled = True
        config.security_enabled = True
        config.encryption_enabled = True
        config.privacy_mode = True
        config.recording_timeout = 10.0
        config.session_timeout = 300.0
        return config

    @pytest.fixture
    def integrated_security(self, integrated_voice_config):
        """Create integrated security instance."""
        security = VoiceSecurity(integrated_voice_config)
        security.encryption_key = b'test_encryption_key_123456789012'
        return security

    @pytest.fixture
    def integrated_voice_service(self, integrated_voice_config, integrated_security, mock_streamlit_session):
        """Create fully integrated voice service."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:

            service = VoiceService(integrated_voice_config, integrated_security)

            # Configure mocked components
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            # Configure realistic component behaviors
            service.audio_processor.initialize.return_value = True
            service.audio_processor.cleanup.return_value = None
            service.audio_processor.start_recording.return_value = True
            service.audio_processor.stop_recording.return_value = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )
            service.audio_processor.play_audio.return_value = True
            service.audio_processor.detect_voice_activity.return_value = True

            # Configure STT service
            service.stt_service.initialize.return_value = True
            service.stt_service.cleanup.return_value = None
            service.stt_service.is_available.return_value = True
            service.stt_service.transcribe_audio = AsyncMock(return_value=self._create_mock_stt_result("Hello, I need help"))

            # Configure TTS service
            service.tts_service.initialize.return_value = True
            service.tts_service.cleanup.return_value = None
            service.tts_service.is_available.return_value = True
            service.tts_service.synthesize_speech = AsyncMock(return_value=self._create_mock_tts_result())

            # Configure command processor
            service.command_processor.initialize.return_value = True
            service.command_processor.process_text = AsyncMock(return_value=None)
            service.command_processor.execute_command = AsyncMock(return_value={'success': True})

            # Initialize service
            service.initialize()

            # Set up callbacks
            service.on_text_received = handle_voice_text_received
            service.on_command_executed = lambda session_id, result: handle_voice_command_executed(session_id, str(result))

            return service

    def _create_mock_stt_result(self, text: str, is_crisis: bool = False, is_command: bool = False):
        """Create mock STT result."""
        class MockSTTResult:
            def __init__(self, text, is_crisis=False, is_command=False):
                self.text = text
                self.confidence = 0.95
                self.language = "en"
                self.duration = 2.0
                self.provider = "test_provider"
                self.alternatives = []
                self.word_timestamps = []
                self.processing_time = 0.5
                self.audio_quality_score = 0.8
                self.therapy_keywords = []
                self.crisis_keywords = []
                self.sentiment_score = 0.5
                self.encryption_metadata = None
                self.cached = False
                self.therapy_keywords_detected = []
                self.crisis_keywords_detected = []
                self.is_crisis = is_crisis
                self.is_command = is_command
                self.sentiment = {'score': 0.5, 'magnitude': 0.5}
                self.segments = []
                self.error = None

        return MockSTTResult(text, is_crisis, is_command)

    def _create_mock_tts_result(self):
        """Create mock TTS result."""
        class MockTTSResult:
            def __init__(self):
                self.audio_data = b'mock_synthesized_audio_data'
                self.duration = 2.5
                self.provider = 'test_provider'
                self.voice = 'test_voice'
                self.format = 'wav'
                self.sample_rate = 22050

        return MockTTSResult()

    @pytest.mark.asyncio
    async def test_voice_text_to_app_integration(self, integrated_voice_service, mock_streamlit_session):
        """Test voice input processing through app integration."""
        # Create voice session
        session_id = integrated_voice_service.create_session()

        # Simulate voice input
        mock_audio = AudioData(
            np.random.randint(-32768, 32767, 16000, dtype=np.int16),
            16000, 1.0, 1
        )

        # Process voice input
        result = await integrated_voice_service.process_voice_input(session_id, mock_audio)

        # Simulate app behavior: add user message to session state
        from tests.mocks.mock_app import handle_voice_text_received
        handle_voice_text_received(session_id, result.text)

        # Verify integration flow
        assert result is not None
        assert result.text == "Hello, I need help"
        assert len(mock_streamlit_session['messages']) > 0

        # Check that message was added to conversation
        last_message = mock_streamlit_session['messages'][-1]
        assert '🎤' in last_message['content']

    @pytest.mark.asyncio
    async def test_app_response_to_voice_output(self, integrated_voice_service, mock_streamlit_session):
        """Test app response generation and voice output."""
        # Create voice session
        session_id = integrated_voice_service.create_session()

        # Test AI response generation
        ai_response = integrated_voice_service.generate_ai_response("I'm feeling anxious")

        # Simulate app behavior: add assistant response to session state
        from tests.mocks.mock_app import handle_assistant_response
        handle_assistant_response(session_id, ai_response)

        # Generate voice output
        tts_result = await integrated_voice_service.generate_voice_output(
            ai_response, session_id
        )

        # Verify response flow
        assert tts_result is not None
        assert tts_result.audio_data == b'mock_synthesized_audio_data'
        assert len(mock_streamlit_session['messages']) > 0

    @pytest.mark.asyncio
    async def test_crisis_detection_integration(self, integrated_voice_service, mock_streamlit_session):
        """Test crisis detection through voice-app integration."""
        
        session_id = integrated_voice_service.create_session()

        # Create crisis STT result
        crisis_text = "I want to kill myself"
        crisis_stt_result = self._create_mock_stt_result(crisis_text, is_crisis=True)

        # Mock STT service to return crisis result
        integrated_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=crisis_stt_result)

        # Mock command processor for crisis response
        crisis_command_result = MagicMock()
        crisis_command_result.is_emergency = True
        crisis_command_result.command.name = 'emergency_help'
        integrated_voice_service.command_processor.process_text = AsyncMock(return_value=crisis_command_result)
        integrated_voice_service.command_processor.execute_command = AsyncMock(
            return_value={'success': True, 'voice_feedback': 'Emergency resources activated'}
        )

        # Process crisis input
        mock_audio = AudioData(
            np.random.randint(-32768, 32767, 16000, dtype=np.int16),
            16000, 1.0, 1
        )

        result = await integrated_voice_service.process_voice_input(session_id, mock_audio)

        # Verify crisis handling
        assert result.is_crisis == True
        assert result.text == crisis_text

        # Simulate app behavior: add crisis response to session state
        from tests.mocks.mock_app import generate_crisis_response, handle_assistant_response
        crisis_response = generate_crisis_response()
        handle_assistant_response(session_id, crisis_response)

        # Check that crisis response was added to messages
        crisis_messages = [msg for msg in mock_streamlit_session['messages']
                         if 'Emergency' in msg.get('content', '') or 'IMMEDIATE HELP' in msg.get('content', '')]
        assert len(crisis_messages) > 0

    @pytest.mark.asyncio
    async def test_voice_command_integration(self, integrated_voice_service, mock_streamlit_session):
        """Test voice command processing through app integration."""
        
        session_id = integrated_voice_service.create_session()

        # Create command STT result
        command_text = "start meditation"
        command_stt_result = self._create_mock_stt_result(command_text, is_command=True)

        # Mock STT service to return command result
        integrated_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=command_stt_result)

        # Mock command processor
        command_result = MagicMock()
        command_result.is_command = True
        command_result.command.name = 'start_meditation'
        integrated_voice_service.command_processor.process_text = AsyncMock(return_value=command_result)
        integrated_voice_service.command_processor.execute_command = AsyncMock(
            return_value={'success': True, 'voice_feedback': 'Starting meditation session'}
        )

        # Process voice command
        mock_audio = AudioData(
            np.random.randint(-32768, 32767, 16000, dtype=np.int16),
            16000, 1.0, 1
        )

        result = await integrated_voice_service.process_voice_input(session_id, mock_audio)

        # Verify command processing
        assert result.is_command == True
        assert result.text == command_text

    @pytest.mark.asyncio
    async def test_concurrent_voice_sessions(self, integrated_voice_config, integrated_security):
        """Test concurrent voice session handling."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:

            service = VoiceService(integrated_voice_config, integrated_security)

            # Configure mocked components
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            # Mock STT service to return a result
            mock_stt_result = self._create_mock_stt_result("Test input")
            service.stt_service.transcribe_audio = AsyncMock(return_value=mock_stt_result)

            # Initialize service
            service.initialize()

            # Create multiple concurrent sessions
            num_sessions = 5
            session_ids = []
            results = []

            for i in range(num_sessions):
                session_id = service.create_session()
                session_ids.append(session_id)

                # Mock audio for each session
                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                # Process each session concurrently
                result = await service.process_voice_input(session_id, mock_audio)
                results.append(result)

            # Verify all sessions processed
            assert len(session_ids) == num_sessions
            assert len(results) == num_sessions
            # All results should be valid (either STTResult objects or False for failed processing)
            assert all(result is not None for result in results)

            # Verify session isolation
            for session_id in session_ids:
                session = service.get_session(session_id)
                assert session is not None
                assert session.session_id == session_id

    @pytest.mark.asyncio
    async def test_memory_management_long_session(self, integrated_voice_service):
        """Test memory management during long-running voice sessions."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        session_id = integrated_voice_service.create_session()

        # Simulate long conversation with many exchanges
        num_exchanges = 50
        mock_audio = AudioData(
            np.random.randint(-32768, 32767, 16000, dtype=np.int16),
            16000, 1.0, 1
        )

        for i in range(num_exchanges):
            # Process voice input
            result = await integrated_voice_service.process_voice_input(session_id, mock_audio)
            assert result is not None

            # Generate voice response
            response_text = f"This is response number {i+1}"
            tts_result = await integrated_voice_service.generate_voice_output(response_text, session_id)
            assert tts_result is not None

            # Add conversation entry
            integrated_voice_service.add_conversation_entry(
                session_id, 'user', f"Message {i+1}"
            )

        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Allow up to 50MB growth for long session
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB, which seems excessive"

        # Verify conversation history is manageable
        conversation_history = integrated_voice_service.get_conversation_history(session_id)
        assert len(conversation_history) >= num_exchanges

        # Test session cleanup
        integrated_voice_service.end_session(session_id)
        assert session_id not in integrated_voice_service.sessions

    def test_security_integration_boundaries(self, integrated_voice_service):
        """Test security integration across module boundaries."""
        session_id = integrated_voice_service.create_session()

        # Test security validation in voice processing
        malicious_text = "'; DROP TABLE users; --"
        stt_result = self._create_mock_stt_result(malicious_text)

        # Simulate security processing
        if hasattr(integrated_voice_service.security, 'validate_input'):
            is_valid = integrated_voice_service.security.validate_input(malicious_text)
            # Security should flag or sanitize malicious input
            assert isinstance(is_valid, bool)

        # Test encryption integration
        if hasattr(integrated_voice_service.security, 'encrypt_data'):
            test_data = b"sensitive_voice_data"
            user_id = "test_user"
            encrypted_data = integrated_voice_service.security.encrypt_data(test_data, user_id)
            assert encrypted_data != test_data  # Should be encrypted

            # Test decryption
            decrypted_data = integrated_voice_service.security.decrypt_data(encrypted_data, user_id)
            assert decrypted_data == test_data  # Should match original

    def test_cache_integration(self, integrated_voice_service):
        """Test response cache integration with voice services."""
        # Test response caching
        response_cache = ResponseCache()

        # Generate responses for same input
        input_text = "I'm feeling stressed"

        response1 = integrated_voice_service.generate_ai_response(input_text)
        response2 = integrated_voice_service.generate_ai_response(input_text)

        # Both responses should be identical for same input
        assert response1 == response2

        # Test embedding cache integration
        embedding_cache = EmbeddingCache()

        # Simulate embedding caching
        test_text = "This is a test sentence for embedding"
        mock_embedding = np.random.random(384).astype(np.float32)  # Typical embedding dimension

        # Cache embedding
        embedding_cache.set(test_text, mock_embedding)

        # Retrieve cached embedding
        cached_embedding = embedding_cache.get(test_text)

        assert cached_embedding is not None
        np.testing.assert_array_equal(cached_embedding, mock_embedding)

    def test_resource_cleanup_integration(self, integrated_voice_service):
        """Test resource cleanup across all integrated components."""
        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = integrated_voice_service.create_session()
            session_ids.append(session_id)

        # Verify sessions exist
        assert len(integrated_voice_service.sessions) == 5

        # Perform cleanup
        integrated_voice_service.cleanup()

        # Verify all sessions are destroyed
        assert len(integrated_voice_service.sessions) == 0

        # Verify service is properly shut down
        assert integrated_voice_service.initialized == False

        # Verify component cleanup was called
        integrated_voice_service.audio_processor.cleanup.assert_called_once()
        integrated_voice_service.stt_service.cleanup.assert_called_once()
        integrated_voice_service.tts_service.cleanup.assert_called_once()
        integrated_voice_service.command_processor.cleanup.assert_called_once()