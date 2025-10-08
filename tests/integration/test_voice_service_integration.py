"""
Voice Service Integration Tests

Comprehensive integration tests for the voice service system,
testing real functionality with minimal mocking.
"""

import pytest
import asyncio
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import tempfile

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from voice.voice_service import VoiceService, VoiceSessionState
    from voice.config import VoiceConfig
    from voice.audio_processor import AudioData
    from voice.security import VoiceSecurity
    VOICE_MODULES_AVAILABLE = True
except ImportError as e:
    VOICE_MODULES_AVAILABLE = False
    print(f"Warning: Voice modules not available: {e}")


class TestVoiceServiceIntegration:
    """Real integration tests for voice service functionality."""

    @pytest.fixture
    def voice_config(self):
        """Create comprehensive voice configuration for integration testing."""
        if not VOICE_MODULES_AVAILABLE:
            pytest.skip("Voice modules not available")
            
        config = VoiceConfig()
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.voice_output_enabled = True
        config.voice_commands_enabled = True
        config.security_enabled = True
        config.session_timeout_minutes = 30
        config.audio_sample_rate = 16000
        config.audio_channels = 1
        config.audio_chunk_size = 1024
        config.stt_provider = "mock"  # Use mock for CI
        config.tts_provider = "mock"  # Use mock for CI
        config.encryption_enabled = True
        config.data_retention_days = 1
        return config

    @pytest.fixture
    def voice_service(self, voice_config):
        """Create real voice service instance with mocked external dependencies."""
        if not VOICE_MODULES_AVAILABLE:
            pytest.skip("Voice modules not available")
            
        # Mock external dependencies but keep internal integration
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:
            
            # Create security instance
            security = VoiceSecurity(voice_config)
            service = VoiceService(voice_config, security)
            
            # Configure mocked components with realistic behavior
            mock_audio_processor.return_value.initialize.return_value = True
            mock_audio_processor.return_value.cleanup.return_value = None
            mock_audio_processor.return_value.start_recording.return_value = True
            mock_audio_processor.return_value.stop_recording.return_value = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )
            
            # Configure STT service
            mock_stt_service.return_value.initialize.return_value = True
            mock_stt_service.return_value.is_available.return_value = True
            mock_stt_service.return_value.transcribe_audio = AsyncMock(
                return_value=MagicMock(
                    text="Hello therapist",
                    confidence=0.95,
                    is_crisis=False,
                    is_command=False
                )
            )
            
            # Configure TTS service
            mock_tts_service.return_value.initialize.return_value = True
            mock_tts_service.return_value.is_available.return_value = True
            mock_tts_service.return_value.synthesize_speech = AsyncMock(
                return_value=MagicMock(
                    audio_data=b"mock_tts_audio",
                    duration=2.5,
                    provider="mock"
                )
            )
            
            # Configure command processor
            mock_command_processor.return_value.initialize.return_value = True
            mock_command_processor.return_value.process_text = AsyncMock(return_value=None)
            mock_command_processor.return_value.execute_command = AsyncMock(
                return_value={"success": True}
            )
            
            # Initialize the service
            service.initialize()
            
            yield service
            
            # Cleanup
            service.cleanup()

    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        duration = 2.0  # 2 seconds
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Generate speech-like audio (sine wave with harmonics)
        t = np.linspace(0, duration, samples)
        audio_data = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # Fundamental
            0.1 * np.sin(2 * np.pi * 880 * t)      # Harmonic
        )
        
        return AudioData(
            data=audio_data.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            channels=1,
            format="float32"
        )

    @pytest.mark.asyncio
    async def test_voice_service_initialization(self, voice_service):
        """Test voice service initialization and configuration."""
        assert voice_service is not None
        assert voice_service.initialized == True
        assert voice_service.config is not None
        assert voice_service.security is not None
        
        # Test service availability
        assert voice_service.is_available() == True

    @pytest.mark.asyncio
    async def test_voice_session_lifecycle(self, voice_service):
        """Test complete voice session lifecycle."""
        user_id = "test_user_123"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        assert session_id is not None
        assert isinstance(session_id, str)
        
        # Verify session exists and is accessible
        session = voice_service.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.user_id == user_id
        assert session.state == VoiceSessionState.ACTIVE
        
        # Test session listing
        user_sessions = voice_service.get_user_sessions(user_id)
        assert len(user_sessions) == 1
        assert session_id in user_sessions
        
        # End session
        voice_service.end_session(session_id)
        assert session_id not in voice_service.sessions
        
        # Verify session cleanup
        cleanup_session = voice_service.get_session(session_id)
        assert cleanup_session is None

    @pytest.mark.asyncio
    async def test_voice_input_processing(self, voice_service, sample_audio_data):
        """Test voice input processing pipeline."""
        user_id = "test_user_input"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        
        # Process voice input
        result = await voice_service.process_voice_input(session_id, sample_audio_data)
        
        # Verify processing result
        assert result is not None
        assert hasattr(result, 'text')
        assert result.text == "Hello therapist"
        assert result.confidence == 0.95
        assert result.is_crisis == False
        assert result.is_command == False
        
        # Clean up
        voice_service.end_session(session_id)

    @pytest.mark.asyncio
    async def test_voice_output_generation(self, voice_service):
        """Test voice output generation."""
        user_id = "test_user_output"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        
        # Generate voice output
        response_text = "I understand how you're feeling. Let's talk more about that."
        tts_result = await voice_service.generate_voice_output(response_text, session_id)
        
        # Verify TTS result
        assert tts_result is not None
        assert hasattr(tts_result, 'audio_data')
        assert tts_result.audio_data == b"mock_tts_audio"
        assert hasattr(tts_result, 'duration')
        assert tts_result.duration == 2.5
        
        # Clean up
        voice_service.end_session(session_id)

    @pytest.mark.asyncio
    async def test_concurrent_voice_sessions(self, voice_service):
        """Test handling multiple concurrent voice sessions."""
        num_sessions = 5
        session_ids = []
        
        # Create multiple sessions
        for i in range(num_sessions):
            user_id = f"concurrent_user_{i}"
            session_id = voice_service.create_session(user_id)
            session_ids.append(session_id)
        
        # Verify all sessions exist
        for session_id in session_ids:
            session = voice_service.get_session(session_id)
            assert session is not None
            assert session.state == VoiceSessionState.ACTIVE
        
        # Test concurrent session operations
        concurrent_tasks = []
        for i, session_id in enumerate(session_ids):
            task = voice_service.generate_ai_response(f"Test message {i}")
            concurrent_tasks.append(task)
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*concurrent_tasks)
        
        # Verify all responses
        assert len(responses) == num_sessions
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Clean up all sessions
        for session_id in session_ids:
            voice_service.end_session(session_id)
        
        # Verify cleanup
        assert len(voice_service.sessions) == 0

    @pytest.mark.asyncio
    async def test_crisis_detection_integration(self, voice_service, sample_audio_data):
        """Test crisis detection integration."""
        user_id = "test_crisis_user"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        
        # Mock crisis detection in STT service
        crisis_stt_result = MagicMock(
            text="I want to kill myself",
            confidence=0.95,
            is_crisis=True,
            is_command=False,
            crisis_keywords=["kill", "suicide"],
            therapy_keywords=[]
        )
        
        voice_service.stt_service.transcribe_audio = AsyncMock(return_value=crisis_stt_result)
        
        # Process crisis input
        result = await voice_service.process_voice_input(session_id, sample_audio_data)
        
        # Verify crisis detection
        assert result is not None
        assert result.is_crisis == True
        assert result.text == "I want to kill myself"
        
        # Verify crisis handling in session
        session = voice_service.get_session(session_id)
        assert session.crisis_detected == True
        
        # Clean up
        voice_service.end_session(session_id)

    @pytest.mark.asyncio
    async def test_voice_command_integration(self, voice_service, sample_audio_data):
        """Test voice command processing integration."""
        user_id = "test_command_user"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        
        # Mock command detection in STT service
        command_stt_result = MagicMock(
            text="start meditation",
            confidence=0.95,
            is_crisis=False,
            is_command=True,
            command_keywords=["start", "meditation"]
        )
        
        voice_service.stt_service.transcribe_audio = AsyncMock(return_value=command_stt_result)
        
        # Mock command processing result
        command_result = MagicMock(
            is_command=True,
            command=MagicMock(name="start_meditation"),
            parameters={}
        )
        
        voice_service.command_processor.process_text = AsyncMock(return_value=command_result)
        voice_service.command_processor.execute_command = AsyncMock(
            return_value={"success": True, "voice_feedback": "Starting meditation session"}
        )
        
        # Process command input
        result = await voice_service.process_voice_input(session_id, sample_audio_data)
        
        # Verify command detection
        assert result is not None
        assert result.is_command == True
        assert result.text == "start meditation"
        
        # Clean up
        voice_service.end_session(session_id)

    @pytest.mark.asyncio
    async def test_conversation_history_management(self, voice_service, sample_audio_data):
        """Test conversation history management across sessions."""
        user_id = "test_history_user"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        
        # Add multiple conversation entries
        messages = [
            ("user", "Hello therapist"),
            ("assistant", "Hello! How can I help you today?"),
            ("user", "I'm feeling anxious"),
            ("assistant", "I understand anxiety can be difficult. Let's explore that.")
        ]
        
        for role, content in messages:
            voice_service.add_conversation_entry(session_id, role, content)
        
        # Verify conversation history
        history = voice_service.get_conversation_history(session_id)
        assert len(history) == len(messages)
        
        # Verify message content and order
        for i, (role, content) in enumerate(messages):
            entry = history[i]
            assert entry.role == role
            assert entry.content == content
            assert entry.timestamp is not None
        
        # Test history filtering
        user_messages = voice_service.get_conversation_history(session_id, role_filter="user")
        assistant_messages = voice_service.get_conversation_history(session_id, role_filter="assistant")
        
        assert len(user_messages) == 2
        assert len(assistant_messages) == 2
        
        # Clean up
        voice_service.end_session(session_id)

    def test_security_integration(self, voice_service):
        """Test security features integration."""
        user_id = "test_security_user"
        
        # Create session
        session_id = voice_service.create_session(user_id)
        
        # Test data encryption
        test_data = b"sensitive_voice_data"
        encrypted_data = voice_service.security.encrypt_data(test_data, user_id)
        
        assert encrypted_data != test_data
        assert encrypted_data is not None
        
        # Test data decryption
        decrypted_data = voice_service.security.decrypt_data(encrypted_data, user_id)
        assert decrypted_data == test_data
        
        # Test input validation
        malicious_input = "'; DROP TABLE users; --"
        sanitized_input = voice_service.security.sanitize_input(malicious_input)
        assert "DROP TABLE" not in sanitized_input
        
        # Clean up
        voice_service.end_session(session_id)

    def test_memory_management(self, voice_service):
        """Test memory management and resource cleanup."""
        initial_sessions = len(voice_service.sessions)
        
        # Create multiple sessions
        session_ids = []
        for i in range(10):
            user_id = f"memory_test_user_{i}"
            session_id = voice_service.create_session(user_id)
            session_ids.append(session_id)
            
            # Add conversation data
            for j in range(5):
                voice_service.add_conversation_entry(
                    session_id, "user", f"Test message {j}"
                )
        
        # Verify sessions created
        assert len(voice_service.sessions) == initial_sessions + 10
        
        # Test memory usage reporting
        memory_usage = voice_service.get_memory_usage()
        assert isinstance(memory_usage, dict)
        assert "session_count" in memory_usage
        assert "total_conversation_entries" in memory_usage
        
        # Clean up all sessions
        for session_id in session_ids:
            voice_service.end_session(session_id)
        
        # Verify cleanup
        assert len(voice_service.sessions) == initial_sessions

    def test_service_health_check(self, voice_service):
        """Test service health monitoring and diagnostics."""
        health_status = voice_service.get_health_status()
        
        assert isinstance(health_status, dict)
        assert "service_status" in health_status
        assert "session_count" in health_status
        assert "components_status" in health_status
        
        # Verify service is healthy
        assert health_status["service_status"] == "healthy"
        
        # Verify component status
        components = health_status["components_status"]
        assert "audio_processor" in components
        assert "stt_service" in components
        assert "tts_service" in components
        assert "command_processor" in components
        assert "security" in components

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, voice_service):
        """Test error handling and recovery mechanisms."""
        user_id = "test_error_user"
        session_id = voice_service.create_session(user_id)
        
        # Test handling invalid session ID
        invalid_result = await voice_service.process_voice_input(
            "invalid_session_id", None
        )
        assert invalid_result is None or invalid_result is False
        
        # Test handling invalid audio data
        invalid_audio = None
        invalid_result = await voice_service.process_voice_input(
            session_id, invalid_audio
        )
        # Should handle gracefully without crashing
        assert True  # Test passes if no exception is raised
        
        # Test service recovery after component failure
        # Simulate component failure
        original_is_available = voice_service.stt_service.is_available
        voice_service.stt_service.is_available.return_value = False
        
        # Service should still be functional with fallbacks
        health_status = voice_service.get_health_status()
        assert health_status["service_status"] in ["healthy", "degraded"]
        
        # Restore service
        voice_service.stt_service.is_available = original_is_available
        
        # Clean up
        voice_service.end_session(session_id)
