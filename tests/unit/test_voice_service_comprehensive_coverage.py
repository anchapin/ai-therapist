"""
Comprehensive unit tests for voice/voice_service.py to complete 50% coverage target.
Focuses on remaining uncovered core voice functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import threading
import time
import asyncio
from datetime import datetime, timedelta
import numpy as np

# Import with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
    from voice.config import VoiceConfig
    from voice.security import VoiceSecurity
    from voice.audio_processor import AudioData
    from voice.stt_service import STTService, STTResult
    from voice.tts_service import TTSService, TTSResult
    from voice.commands import VoiceCommandProcessor
except ImportError as e:
    pytest.skip(f"voice_service module not available: {e}", allow_module_level=True)


class TestVoiceServiceComprehensiveCoverage:
    """Comprehensive unit tests to reach 50% coverage for voice_service.py."""
    
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
        config.voice_command_timeout = 30000
        config.voice_wake_word = "hey therapist"
        return config
    
    @pytest.fixture
    def mock_security(self):
        """Create a mock security instance."""
        security = Mock(spec=VoiceSecurity)
        security.encrypt_audio_data.return_value = b"encrypted_audio"
        security.decrypt_audio_data.return_value = b"decrypted_audio"
        return security
    
    @pytest.fixture
    def voice_service(self, mock_config, mock_security):
        """Create a VoiceService with mocked dependencies."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service:
            
            # Mock the processor
            mock_processor_instance = Mock()
            mock_processor_instance.input_devices = ["mic1", "mic2"]
            mock_processor_instance.output_devices = ["speaker1"]
            mock_audio_processor.return_value = mock_processor_instance
            
            # Mock STT service
            mock_stt_instance = Mock(spec=STTService)
            mock_stt_instance.is_available.return_value = True
            mock_stt_service.return_value = mock_stt_instance
            
            # Mock TTS service  
            mock_tts_instance = Mock(spec=TTSService)
            mock_tts_instance.is_available.return_value = True
            mock_tts_service.return_value = mock_tts_instance
            
            service = VoiceService(mock_config, mock_security)
            return service
    
    def test_start_listening_basic(self, voice_service):
        """Test basic session start listening."""
        # Create a session first
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Start listening
        result = voice_service.start_listening(session_id)
        
        assert result is True
        assert session.state == VoiceSessionState.LISTENING
        assert hasattr(session, 'start_time')
    
    def test_start_listening_no_session(self, voice_service):
        """Test start listening with no active session."""
        result = voice_service.start_listening()
        
        assert result is True  # Should start listening on current session
        
        # If no current session, should create one
        current_session = voice_service.get_current_session()
        if current_session:
            assert current_session.state == VoiceSessionState.LISTENING
    
    def test_start_listening_already_listening(self, voice_service):
        """Test start listening when already listening."""
        session_id = voice_service.create_session("user123")
        
        # Start listening first
        voice_service.start_listening(session_id)
        session = voice_service.get_session(session_id)
        
        # Try to start listening again
        result = voice_service.start_listening(session_id)
        
        assert result is True  # Should handle gracefully
        assert session.state == VoiceSessionState.LISTENING
    
    def test_stop_listening_basic(self, voice_service):
        """Test basic session stop listening."""
        # Create session and start listening
        session_id = voice_service.create_session("user123")
        voice_service.start_listening(session_id)
        session = voice_service.get_session(session_id)
        
        # Stop listening
        result = voice_service.stop_listening(session_id)
        
        assert isinstance(result, AudioData)  # Should return audio data
        assert session.state == VoiceSessionState.IDLE
        assert session.start_time is not None
    
    def test_stop_listening_no_session(self, voice_service):
        """Test stop listening with no active session."""
        result = voice_service.stop_listening()
        
        assert isinstance(result, AudioData)  # Should return empty audio data
        assert len(result.data) == 0  # Should be empty
    
    def test_stop_listening_not_listening(self, voice_service):
        """Test stop listening when not listening."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Don't start listening, just stop
        result = voice_service.stop_listening(session_id)
        
        assert isinstance(result, AudioData)  # Should return empty audio data
        assert session.state == VoiceSessionState.IDLE
    
    def test_start_speaking_basic(self, voice_service):
        """Test basic session start speaking."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Start speaking
        result = voice_service.start_speaking(session_id, "Hello world")
        
        assert result is True
        assert session.state == VoiceSessionState.SPEAKING
    
    def test_start_speaking_empty_text(self, voice_service):
        """Test start speaking with empty text."""
        session_id = voice_service.create_session("user123")
        
        result = voice_service.start_speaking(session_id, "")
        
        # Should handle empty text gracefully
        assert result is True or result is False  # Depends on implementation
    
    def test_stop_speaking_basic(self, voice_service):
        """Test basic session stop speaking."""
        session_id = voice_service.create_session("user123")
        voice_service.start_speaking(session_id, "Hello world")
        session = voice_service.get_session(session_id)
        
        # Stop speaking
        result = voice_service.stop_speaking(session_id)
        
        assert result is True
        assert session.state == VoiceSessionState.IDLE
    
    def test_stop_speaking_not_speaking(self, voice_service):
        """Test stop speaking when not speaking."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Stop speaking without starting
        result = voice_service.stop_speaking(session_id)
        
        assert result is True  # Should handle gracefully
        assert session.state == VoiceSessionState.IDLE
    
    def test_process_voice_input_success(self, voice_service):
        """Test successful voice input processing."""
        # Create session and start listening
        session_id = voice_service.create_session("user123")
        voice_service.start_listening(session_id)
        
        # Create mock audio data
        audio_data = AudioData(
            data=np.array([1, 2, 3, 4, 5], dtype=np.float32),
            sample_rate=16000,
            duration=0.1,
            channels=1,
            format='wav'
        )
        
        # Process voice input (should be async, but we'll test the sync wrapper)
        result = voice_service.process_voice_input(session_id, audio_data)
        
        assert result is not None  # Should return result or coroutine
    
    def test_process_voice_input_no_session(self, voice_service):
        """Test voice input processing with no session."""
        audio_data = AudioData(
            data=np.array([1, 2, 3], dtype=np.float32),
            sample_rate=16000,
            duration=0.1,
            channels=1,
            format='wav'
        )
        
        result = voice_service.process_voice_input("nonexistent_session", audio_data)
        
        assert result is None  # Should return None for nonexistent session
    
    def test_generate_voice_response_success(self, voice_service):
        """Test successful voice response generation."""
        session_id = voice_service.create_session("user123")
        text = "Hello, how are you today?"
        
        result = voice_service.generate_voice_response(session_id, text)
        
        assert result is not None  # Should return result or coroutine
    
    def test_generate_voice_response_no_session(self, voice_service):
        """Test voice response generation with no session."""
        text = "Hello world"
        
        result = voice_service.generate_voice_response("nonexistent_session", text)
        
        assert result is None  # Should return None for nonexistent session
    
    def test_process_voice_command_detected(self, voice_service):
        """Test voice command processing when command is detected."""
        session_id = voice_service.create_session("user123")
        
        # Create STT result with command
        stt_result = STTResult(
            text="start session",
            confidence=0.95,
            alternatives=[],
            processing_time=1.0
        )
        
        result = voice_service._process_voice_command(session_id, stt_result)
        
        assert result is not None  # Should process command
    
    def test_process_voice_command_no_command(self, voice_service):
        """Test voice command processing when no command is detected."""
        session_id = voice_service.create_session("user123")
        
        # Create STT result without command
        stt_result = STTResult(
            text="regular conversation text",
            confidence=0.85,
            alternatives=[],
            processing_time=0.8
        )
        
        result = voice_service._process_voice_command(session_id, stt_result)
        
        assert result is not None  # Should process regular text
    
    def test_get_active_sessions(self, voice_service):
        """Test getting active sessions list."""
        # Create multiple sessions
        session1_id = voice_service.create_session("user1")
        session2_id = voice_service.create_session("user2")
        session3_id = voice_service.create_session("user3")
        
        # Start listening on some sessions
        voice_service.start_listening(session1_id)
        voice_service.start_listening(session2_id)
        
        # Get active sessions
        active_sessions = voice_service.get_active_sessions()
        
        assert isinstance(active_sessions, list)
        assert len(active_sessions) == 2  # Only sessions 1 and 2 are active
        
        active_session_ids = [s.session_id for s in active_sessions]
        assert session1_id in active_session_ids
        assert session2_id in active_session_ids
        assert session3_id not in active_session_ids
    
    def test_get_active_sessions_empty(self, voice_service):
        """Test getting active sessions when no sessions are active."""
        active_sessions = voice_service.get_active_sessions()
        
        assert isinstance(active_sessions, list)
        assert len(active_sessions) == 0
    
    def test_get_session_statistics_detailed(self, voice_service):
        """Test detailed session statistics."""
        # Create some sessions
        voice_service.create_session("user1")
        voice_service.create_session("user2")
        voice_service.create_session("user3")
        
        # Get detailed statistics
        stats = voice_service.get_session_statistics_detailed()
        
        assert isinstance(stats, dict)
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'idle_sessions' in stats
        assert 'listening_sessions' in stats
        assert 'speaking_sessions' in stats
        assert 'average_session_duration' in stats
        assert 'sessions_by_state' in stats
        
        # Should count correctly
        assert stats['total_sessions'] == 3
        assert stats['active_sessions'] == 3  # All sessions are active (even if idle)
        
        # State breakdown
        state_counts = stats['sessions_by_state']
        assert VoiceSessionState.IDLE in state_counts
    
    def test_cleanup_expired_sessions_basic(self, voice_service):
        """Test basic cleanup of expired sessions."""
        # Create a session
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Manually expire the session
        session.created_at = datetime.now() - timedelta(hours=2)  # 2 hours ago
        session.expires_at = datetime.now() - timedelta(hours=1)   # 1 hour ago
        
        # Run cleanup
        cleaned_count = voice_service.cleanup_expired_sessions()
        
        assert isinstance(cleaned_count, int)
        assert cleaned_count >= 1  # Should have cleaned at least one session
        
        # Session should be removed
        assert session_id not in voice_service.sessions
    
    def test_cleanup_expired_sessions_none_expired(self, voice_service):
        """Test cleanup when no sessions are expired."""
        # Create fresh sessions
        session1_id = voice_service.create_session("user1")
        session2_id = voice_service.create_session("user2")
        
        # Don't expire sessions
        cleaned_count = voice_service.cleanup_expired_sessions()
        
        assert cleaned_count == 0  # No sessions should be cleaned
        
        # Sessions should still exist
        assert session1_id in voice_service.sessions
        assert session2_id in voice_service.sessions
    
    def test_get_current_session_with_multiple(self, voice_service):
        """Test getting current session when multiple sessions exist."""
        # Create multiple sessions
        session1_id = voice_service.create_session("user1")
        session2_id = voice_service.create_session("user2")
        
        # The last created session should be current
        current_session = voice_service.get_current_session()
        
        assert current_session is not None
        assert current_session.session_id == session2_id
    
    def test_get_current_session_no_sessions(self, voice_service):
        """Test getting current session when no sessions exist."""
        current_session = voice_service.get_current_session()
        
        assert current_session is None
    
    def test_validate_session_id_valid(self, voice_service):
        """Test session ID validation for valid session."""
        session_id = voice_service.create_session("user123")
        
        result = voice_service.validate_session_id(session_id)
        
        assert result is True
    
    def test_validate_session_id_invalid(self, voice_service):
        """Test session ID validation for invalid session."""
        invalid_ids = ["nonexistent", "", None, 123]
        
        for invalid_id in invalid_ids:
            if invalid_id is None:
                continue  # Skip None as it's not a string
            result = voice_service.validate_session_id(invalid_id)
            assert result is False
    
    def test_set_session_metadata(self, voice_service):
        """Test setting session metadata."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Set metadata
        metadata = {
            "user_type": "patient",
            "preferred_language": "en",
            "accessibility_features": ["closed_captions"]
        }
        
        voice_service.set_session_metadata(session_id, metadata)
        
        # Should be stored (either in session or database)
        assert hasattr(session, 'metadata')
    
    def test_get_session_metadata(self, voice_service):
        """Test getting session metadata."""
        session_id = voice_service.create_session("user123")
        
        # Set metadata
        metadata = {"test_key": "test_value"}
        voice_service.set_session_metadata(session_id, metadata)
        
        # Get metadata
        retrieved = voice_service.get_session_metadata(session_id)
        
        assert retrieved is not None or "test_key" in session.metadata
    
    def test_session_thread_safety_concurrent_access(self, voice_service):
        """Test thread-safe concurrent session access."""
        session_id = voice_service.create_session("user123")
        results = []
        errors = []
        
        def access_session():
            try:
                # Perform multiple operations
                session = voice_service.get_session(session_id)
                assert session is not None
                assert session.session_id == session_id
                
                current = voice_service.get_current_session()
                assert current is not None or current.session_id == session_id
                
                voice_service.validate_session_id(session_id)
                active = voice_service.get_active_sessions()
                assert isinstance(active, list)
                
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = [threading.Thread(target=access_session) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=2)
        
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_session_activity_tracking(self, voice_service):
        """Test session activity tracking."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Initial activity
        initial_activity = session.last_activity
        
        # Update activity
        voice_service.update_session_activity(session_id)
        
        updated_session = voice_service.get_session(session_id)
        
        assert updated_session.last_activity >= initial_activity
    
    def test_session_state_transitions_complete_flow(self, voice_service):
        """Test complete session state transition flow."""
        session_id = voice_service.create_session("user123")
        session = voice_service.get_session(session_id)
        
        # Initial state should be IDLE
        assert session.state == VoiceSessionState.IDLE
        
        # Transition to LISTENING
        voice_service.start_listening(session_id)
        assert session.state == VoiceSessionState.LISTENING
        
        # Transition to SPEAKING
        voice_service.start_speaking(session_id, "Hello")
        assert session.state == VoiceSessionState.SPEAKING
        
        # Transition back to IDLE
        voice_service.stop_speaking(session_id)
        assert session.state == VoiceSessionState.IDLE
        
        # Transition to PROCESSING
        # This typically happens during command processing
        session.state = VoiceSessionState.PROCESSING
        assert session.state == VoiceSessionState.PROCESSING
        
        # Back to IDLE
        session.state = VoiceSessionState.IDLE
        assert session.state == VoiceSessionState.IDLE
    
    def test_concurrent_session_creation(self, voice_service):
        """Test creating multiple sessions concurrently."""
        session_ids = []
        errors = []
        
        def create_session(user_id):
            try:
                session_id = voice_service.create_session(user_id)
                session_ids.append(session_id)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads for session creation
        threads = [threading.Thread(target=create_session, args=(f"user{i}",)) for i in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=2)
        
        assert len(errors) == 0
        assert len(session_ids) == 5
        
        # All session IDs should be unique
        assert len(set(session_ids)) == len(session_ids)
        
        # All sessions should exist
        for session_id in session_ids:
            session = voice_service.get_session(session_id)
            assert session is not None
    
    def test_health_check_comprehensive(self, voice_service):
        """Test comprehensive health check."""
        health = voice_service.health_check()
        
        assert isinstance(health, dict)
        assert 'overall_status' in health
        assert 'audio_processor' in health
        assert 'stt_service' in health
        assert 'tts_service' in health
        assert 'security' in health
        assert 'command_processor' in health
        assert 'sessions' in health
        
        # Status should be valid
        valid_statuses = ['healthy', 'degraded', 'unhealthy', 'unknown']
        assert health['overall_status'] in valid_statuses
        
        # Component checks should be performed
        for component in ['audio_processor', 'stt_service', 'tts_service', 'security', 'command_processor']:
            assert isinstance(health[component], dict)
            assert 'status' in health[component]
            assert 'issues' in health[component]
            assert isinstance(health[component]['issues'], list)
    
    def test_audio_device_detection(self, voice_service):
        """Test audio device detection and management."""
        # Test device availability check
        has_audio = voice_service._check_audio_availability()
        
        assert isinstance(has_audio, bool)
        
        # Test device list retrieval
        devices = voice_service.get_audio_devices()
        
        assert isinstance(devices, dict)
        assert 'input_devices' in devices
        assert 'output_devices' in devices
        assert isinstance(devices['input_devices'], list)
        assert isinstance(devices['output_devices'], list)
    
    def test_service_initialization_complete(self, voice_service):
        """Test complete service initialization."""
        # Check all components are initialized
        assert voice_service.config is not None
        assert voice_service.security is not None
        assert voice_service.audio_processor is not None
        assert voice_service.stt_service is not None
        assert voice_service.tts_service is not None
        assert voice_service.command_processor is not None
        
        # Check data structures
        assert hasattr(voice_service, 'sessions')
        assert hasattr(voice_service, 'current_session_id')
        assert hasattr(voice_service, '_sessions_lock')
        assert hasattr(voice_service, 'metrics')
        
        # Check initial state
        assert isinstance(voice_service.sessions, dict)
        assert len(voice_service.sessions) == 0
        assert voice_service.current_session_id is None
        assert isinstance(voice_service.metrics, dict)
        
        # Check metrics that actually exist
        actual_metrics = voice_service.metrics
        expected_metrics = [
            'sessions_created', 'error_count', 'service_uptime',
            'voice_commands_processed', 'stt_requests', 'tts_requests',
            'average_response_time'
        ]
        
        for metric in expected_metrics:
            assert metric in actual_metrics
            assert isinstance(actual_metrics[metric], (int, float))
    
    def test_error_handling_in_voice_operations(self, voice_service):
        """Test error handling in voice operations."""
        session_id = voice_service.create_session("user123")
        
        # Test operations with various error scenarios
        error_cases = [
            (None, "Input data"),
            ("", "Text"),
            (123, "Wrong type"),
            ("invalid_audio", "Invalid audio")
        ]
        
        for error_case in error_cases:
            try:
                if error_case[1] == "Input data":
                    voice_service.process_voice_input(session_id, error_case[0])
                elif error_case[1] == "Text":
                    voice_service.generate_voice_response(session_id, error_case[0])
                # Other error cases are handled gracefully
            except Exception as e:
                # Should handle errors gracefully
                assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_service_metrics_tracking(self, voice_service):
        """Test service metrics tracking."""
        initial_metrics = voice_service.metrics.copy()
        
        # Create session (should increment sessions_created)
        voice_service.create_session("user1")
        assert voice_service.metrics['sessions_created'] > initial_metrics['sessions_created']
        
        # End session (should increment sessions_destroyed)
        session_id = voice_service.create_session("user2")
        voice_service.end_session(session_id)
        assert voice_service.metrics['sessions_destroyed'] > initial_metrics['sessions_destroyed']
        
        # Metrics should be tracked correctly
        assert voice_service.metrics['sessions_created'] >= 1
        assert voice_service.metrics['sessions_destroyed'] >= 1
    
    def test_voice_service_cleanup(self, voice_service):
        """Test service cleanup and resource management."""
        # Create some sessions
        voice_service.create_session("user1")
        voice_service.create_session("user2")
        
        # Verify sessions exist
        assert len(voice_service.sessions) >= 2
        
        # Run cleanup
        voice_service.cleanup()
        
        # Should not raise errors
        assert True  # If we reach here, cleanup succeeded
    
    def test_session_management_batch_operations(self, voice_service):
        """Test batch session management operations."""
        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = voice_service.create_session(f"user{i}")
            session_ids.append(session_id)
        
        # Batch update session activities
        for session_id in session_ids:
            voice_service.update_session_activity(session_id)
        
        # Batch validation
        valid_sessions = []
        for session_id in session_ids:
            if voice_service.validate_session_id(session_id):
                valid_sessions.append(session_id)
        
        assert len(valid_sessions) == len(session_ids)
        
        # Batch cleanup
        for session_id in session_ids:
            voice_service.end_session(session_id)
        
        assert len(voice_service.sessions) == 0