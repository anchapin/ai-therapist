"""
Comprehensive unit tests for voice/optimized_voice_service.py

Covers critical gaps in coverage analysis for optimized voice services:
- Session management with memory efficiency
- Voice processing with performance optimizations
- Real-time performance metrics and caching
- Thread pool management and concurrent operations
- Health checks and service statistics
- Mock services for performance testing
- Error handling and edge cases
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import time
import asyncio
import threading
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voice.optimized_voice_service import (
    OptimizedVoiceSession,
    OptimizedVoiceSessionState,
    VoiceProcessingMetrics,
    OptimizedVoiceService,
    OptimizedSTTService,
    OptimizedTTSService,
    OptimizedCommandProcessor
)


class TestOptimizedVoiceSession(unittest.TestCase):
    """Test OptimizedVoiceSession class."""

    def test_session_creation(self):
        """Test voice session creation."""
        session_id = "test_session"
        start_time = time.time()

        session = OptimizedVoiceSession(
            session_id=session_id,
            state=OptimizedVoiceSessionState.IDLE,
            start_time=start_time,
            last_activity=start_time,
            voice_profile="default"
        )

        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.state, OptimizedVoiceSessionState.IDLE)
        self.assertEqual(session.start_time, start_time)
        self.assertEqual(session.last_activity, start_time)
        self.assertEqual(session.voice_profile, "default")
        self.assertEqual(session.max_history_size, 50)
        self.assertEqual(session.max_audio_buffer_size, 10)

    def test_session_metadata_initialization(self):
        """Test session metadata initialization."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile"
        )

        # Verify metadata was initialized
        self.assertIn('created_at', session.metadata)
        self.assertIn('voice_settings', session.metadata)
        self.assertEqual(session.metadata['voice_settings']['voice_speed'], 1.2)
        self.assertEqual(session.metadata['voice_settings']['optimization_mode'], 'latency')

    def test_session_metrics_initialization(self):
        """Test session metrics initialization."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile"
        )

        # Verify metrics were initialized
        self.assertEqual(session.metrics['total_interactions'], 0)
        self.assertEqual(session.metrics['average_response_time'], 0.0)
        self.assertEqual(session.metrics['error_count'], 0)
        self.assertEqual(session.metrics['memory_usage_mb'], 0.0)

    def test_add_conversation_entry(self):
        """Test adding conversation entries."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile"
        )

        entry = {
            'type': 'user',
            'text': 'Hello, I need help',
            'timestamp': time.time()
        }

        session.add_conversation_entry(entry)

        # Verify entry was added
        self.assertEqual(len(session.conversation_history), 1)
        self.assertEqual(session.conversation_history[0], entry)

    def test_conversation_history_size_limit(self):
        """Test conversation history size management."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile",
            max_history_size=3  # Small limit for testing
        )

        # Add entries beyond limit
        for i in range(5):
            session.add_conversation_entry({
                'type': 'user',
                'text': f'Message {i}',
                'timestamp': time.time()
            })

        # Should maintain size limit
        self.assertEqual(len(session.conversation_history), 3)
        # Should keep most recent entries
        self.assertEqual(session.conversation_history[-1]['text'], 'Message 4')

    def test_add_audio_data(self):
        """Test adding audio data."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile"
        )

        # Mock OptimizedAudioData
        audio_data = Mock()
        audio_data.data = np.array([0.1, 0.2, 0.3])

        session.add_audio_data(audio_data)

        # Verify audio data was added
        self.assertEqual(len(session.audio_buffer), 1)
        self.assertEqual(session.audio_buffer[0], audio_data)

    def test_audio_buffer_size_limit(self):
        """Test audio buffer size management."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile",
            max_audio_buffer_size=2  # Small limit for testing
        )

        # Add audio data beyond limit
        for i in range(4):
            audio_data = Mock()
            audio_data.data = np.array([0.1, 0.2, 0.3])
            session.add_audio_data(audio_data)

        # Should maintain size limit
        self.assertEqual(len(session.audio_buffer), 2)

    def test_update_activity(self):
        """Test activity timestamp updates."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile"
        )

        old_activity = session.last_activity
        time.sleep(0.01)  # Small delay

        session.update_activity()

        # Activity timestamp should be updated
        self.assertGreater(session.last_activity, old_activity)

    def test_get_session_duration(self):
        """Test session duration calculation."""
        start_time = time.time()
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.IDLE,
            start_time=start_time,
            last_activity=start_time,
            voice_profile="test_profile"
        )

        # Wait a bit
        time.sleep(0.1)

        duration = session.get_session_duration()

        # Should be approximately 0.1 seconds
        self.assertGreater(duration, 0.05)
        self.assertLess(duration, 1.0)

    def test_session_cleanup(self):
        """Test session cleanup."""
        session = OptimizedVoiceSession(
            session_id="test",
            state=OptimizedVoiceSessionState.LISTENING,
            start_time=time.time(),
            last_activity=time.time(),
            voice_profile="test_profile"
        )

        # Add some data
        session.add_conversation_entry({'type': 'user', 'text': 'test'})
        audio_data = Mock()
        session.add_audio_data(audio_data)

        # Cleanup
        session.cleanup()

        # Verify cleanup
        self.assertEqual(len(session.conversation_history), 0)
        self.assertEqual(len(session.audio_buffer), 0)
        self.assertEqual(session.state, OptimizedVoiceSessionState.IDLE)


class TestVoiceProcessingMetrics(unittest.TestCase):
    """Test VoiceProcessingMetrics class."""

    def test_metrics_creation(self):
        """Test voice processing metrics creation."""
        start_time = time.time()
        end_time = start_time + 0.5

        metrics = VoiceProcessingMetrics(
            operation_type="speech_to_text",
            start_time=start_time,
            end_time=end_time,
            duration=0.5,
            session_id="test_session",
            success=True
        )

        self.assertEqual(metrics.operation_type, "speech_to_text")
        self.assertEqual(metrics.start_time, start_time)
        self.assertEqual(metrics.end_time, end_time)
        self.assertEqual(metrics.duration, 0.5)
        self.assertEqual(metrics.session_id, "test_session")
        self.assertTrue(metrics.success)
        self.assertIsNone(metrics.error_message)

    def test_metrics_with_error(self):
        """Test metrics with error information."""
        metrics = VoiceProcessingMetrics(
            operation_type="text_to_speech",
            start_time=time.time(),
            end_time=time.time() + 0.3,
            duration=0.3,
            session_id="test_session",
            success=False,
            error_message="Processing failed"
        )

        self.assertFalse(metrics.success)
        self.assertEqual(metrics.error_message, "Processing failed")

    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = VoiceProcessingMetrics(
            operation_type="speech_to_text",
            start_time=time.time(),
            end_time=time.time() + 0.5,
            duration=0.5,
            session_id="test_session",
            success=True,
            error_message="No errors"
        )

        result = metrics.to_dict()

        expected = {
            'operation_type': 'speech_to_text',
            'duration': 0.5,
            'session_id': 'test_session',
            'success': True,
            'error_message': 'No errors'
        }

        self.assertEqual(result, expected)


class TestOptimizedVoiceService(unittest.TestCase):
    """Test OptimizedVoiceService class."""

    def setUp(self):
        """Set up voice service tests."""
        # Mock config and security
        self.mock_config = Mock()
        self.mock_config.voice_enabled = True
        self.mock_config.max_concurrent_sessions = 10

        self.mock_security = Mock()

        self.voice_service = OptimizedVoiceService(self.mock_config, self.mock_security)

    def test_service_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.voice_service.audio_processor)
        self.assertEqual(self.voice_service.max_concurrent_sessions, 10)
        self.assertFalse(self.voice_service.is_running)
        self.assertEqual(len(self.voice_service.sessions), 0)

    def test_service_initialization_disabled(self):
        """Test service initialization when disabled."""
        self.mock_config.voice_enabled = False

        with patch.object(self.voice_service.logger, 'info') as mock_info:
            result = self.voice_service.initialize()

            # Should return False when disabled
            self.assertFalse(result)
            mock_info.assert_called_once()

    def test_create_session(self):
        """Test session creation."""
        session_id = self.voice_service.create_session("test_session", "therapist")

        # Verify session was created
        self.assertIn(session_id, self.voice_service.sessions)
        session = self.voice_service.sessions[session_id]
        self.assertEqual(session.session_id, "test_session")
        self.assertEqual(session.voice_profile, "therapist")

    def test_create_session_default_id(self):
        """Test session creation with default ID."""
        session_id = self.voice_service.create_session()

        # Should generate ID automatically
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.voice_service.sessions)

    def test_create_session_duplicate_id(self):
        """Test session creation with duplicate ID."""
        session_id = "duplicate_session"

        # Create first session
        result1 = self.voice_service.create_session(session_id)

        # Try to create duplicate (should return existing ID)
        result2 = self.voice_service.create_session(session_id)

        # Should return same ID
        self.assertEqual(result1, result2)
        self.assertEqual(len(self.voice_service.sessions), 1)

    def test_create_session_max_concurrent(self):
        """Test session creation at max concurrent limit."""
        self.voice_service.max_concurrent_sessions = 1

        # Create first session
        self.voice_service.create_session("session1")

        # Try to create second session (should fail)
        with self.assertRaises(RuntimeError):
            self.voice_service.create_session("session2")

    def test_get_session(self):
        """Test session retrieval."""
        session_id = self.voice_service.create_session("test_session")

        # Retrieve session
        session = self.voice_service.get_session(session_id)

        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, "test_session")

    def test_get_session_not_found(self):
        """Test session retrieval for non-existent session."""
        session = self.voice_service.get_session("nonexistent")

        self.assertIsNone(session)

    def test_end_session(self):
        """Test session termination."""
        session_id = self.voice_service.create_session("test_session")

        # Verify session exists
        self.assertIn(session_id, self.voice_service.sessions)

        # End session
        result = self.voice_service.end_session(session_id)

        # Verify session was removed
        self.assertTrue(result)
        self.assertNotIn(session_id, self.voice_service.sessions)

    def test_end_session_not_found(self):
        """Test session termination for non-existent session."""
        result = self.voice_service.end_session("nonexistent")

        # Should return False for non-existent session
        self.assertFalse(result)

    def test_cache_operations(self):
        """Test caching operations."""
        # Test STT cache
        cache_key = "test_audio_key"
        mock_result = Mock()
        mock_result.text = "Cached transcription"

        # Cache result
        self.voice_service._cache_stt_result(cache_key, mock_result)

        # Retrieve from cache
        result = self.voice_service._get_cached_stt_result(cache_key)

        self.assertEqual(result, mock_result)

    def test_cache_size_management(self):
        """Test cache size management."""
        # Fill cache beyond limit
        for i in range(self.voice_service._cache_max_size + 10):
            cache_key = f"key_{i}"
            mock_result = Mock()
            self.voice_service._cache_stt_result(cache_key, mock_result)

        # Should maintain size limit
        self.assertEqual(len(self.voice_service._stt_cache), self.voice_service._cache_max_size)

    def test_generate_cache_keys(self):
        """Test cache key generation."""
        # Mock audio data
        audio_data = Mock()
        audio_data.data = np.array([0.1, 0.2, 0.3])
        audio_data.sample_rate = 16000

        # Test audio cache key generation
        audio_key = self.voice_service._generate_cache_key(audio_data)

        self.assertIsInstance(audio_key, str)
        self.assertIn("audio", audio_key)

        # Test text cache key generation
        text_key = self.voice_service._generate_text_cache_key("Hello world", "therapist")

        self.assertIsInstance(text_key, str)
        self.assertIn("text", text_key)
        self.assertIn("therapist", text_key)

    def test_service_statistics(self):
        """Test service statistics generation."""
        # Create some sessions and interactions
        session1 = self.voice_service.create_session("session1")
        session2 = self.voice_service.create_session("session2")

        # Simulate some interactions
        self.voice_service.metrics['total_interactions'] = 10
        self.voice_service.metrics['sessions_created'] = 2

        stats = self.voice_service.get_service_statistics()

        # Verify statistics
        self.assertIn('uptime', stats)
        self.assertIn('sessions_count', stats)
        self.assertIn('total_interactions', stats)
        self.assertEqual(stats['sessions_count'], 2)
        self.assertEqual(stats['total_interactions'], 10)

    def test_health_check(self):
        """Test comprehensive health check."""
        health = self.voice_service.health_check()

        # Verify health check structure
        self.assertIn('overall_status', health)
        self.assertIn('components', health)
        self.assertIn('performance', health)

        # Verify components
        self.assertIn('audio_processor', health['components'])
        self.assertIn('sessions', health['components'])
        self.assertIn('caching', health['components'])

    def test_service_cleanup(self):
        """Test service cleanup."""
        # Create some sessions
        self.voice_service.create_session("session1")
        self.voice_service.create_session("session2")

        # Add some cache entries
        self.voice_service._cache_stt_result("key1", Mock())
        self.voice_service._cache_tts_result("key2", Mock())

        # Setup callbacks
        self.voice_service.on_text_received = Mock()
        self.voice_service.on_audio_played = Mock()

        # Perform cleanup
        self.voice_service.cleanup()

        # Verify cleanup
        self.assertFalse(self.voice_service.is_running)
        self.assertEqual(len(self.voice_service.sessions), 0)
        self.assertEqual(len(self.voice_service._stt_cache), 0)
        self.assertEqual(len(self.voice_service._tts_cache), 0)
        self.assertIsNone(self.voice_service.on_text_received)
        self.assertIsNone(self.voice_service.on_audio_played)


class TestVoiceProcessing(unittest.TestCase):
    """Test voice processing functionality."""

    def setUp(self):
        """Set up voice processing tests."""
        self.mock_config = Mock()
        self.mock_config.voice_enabled = True
        self.mock_config.max_concurrent_sessions = 10

        self.mock_security = Mock()

        self.voice_service = OptimizedVoiceService(self.mock_config, self.mock_security)

    def test_process_voice_input_success(self):
        """Test successful voice input processing."""
        # Mock STT service
        mock_stt_result = Mock()
        mock_stt_result.text = "Hello, I need help with anxiety"
        mock_stt_result.confidence = 0.95
        mock_stt_result.provider = "optimized"

        self.voice_service.stt_service = Mock()
        self.voice_service.stt_service.transcribe_audio = AsyncMock(return_value=mock_stt_result)

        # Mock audio data
        audio_data = Mock()
        audio_data.data = np.array([0.1, 0.2, 0.3])
        audio_data.sample_rate = 16000

        # Setup callback
        received_text = []
        def text_callback(session_id, text):
            received_text.append((session_id, text))

        self.voice_service.on_text_received = text_callback

        # Process voice input
        async def run_test():
            result = await self.voice_service.process_voice_input(audio_data)
            return result

        # Run async test
        result = asyncio.run(run_test())

        # Verify processing
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Hello, I need help with anxiety")
        self.assertEqual(len(received_text), 1)

    def test_process_voice_input_with_caching(self):
        """Test voice input processing with caching."""
        # Setup cached result
        audio_data = Mock()
        audio_data.data = np.array([0.1, 0.2, 0.3])
        audio_data.sample_rate = 16000

        cache_key = self.voice_service._generate_cache_key(audio_data)
        mock_result = Mock()
        mock_result.text = "Cached result"

        self.voice_service._cache_stt_result(cache_key, mock_result)

        # Process voice input (should use cache)
        async def run_test():
            result = await self.voice_service.process_voice_input(audio_data)
            return result

        result = asyncio.run(run_test())

        # Verify cached result was used
        self.assertEqual(result.text, "Cached result")

    def test_generate_voice_output_success(self):
        """Test successful voice output generation."""
        # Mock TTS service
        mock_tts_result = Mock()
        mock_tts_result.audio_data = b'fake_audio_data'
        mock_tts_result.duration = 2.5
        mock_tts_result.provider = "optimized"
        mock_tts_result.sample_rate = 22050

        self.voice_service.tts_service = Mock()
        self.voice_service.tts_service.synthesize_speech = AsyncMock(return_value=mock_tts_result)

        # Setup callback
        played_audio = []
        def audio_callback(audio_data):
            played_audio.append(audio_data)

        self.voice_service.on_audio_played = audio_callback

        # Generate voice output
        async def run_test():
            result = await self.voice_service.generate_voice_output("Hello, how are you?")
            return result

        result = asyncio.run(run_test())

        # Verify generation
        self.assertIsNotNone(result)
        self.assertEqual(result.audio_data, b'fake_audio_data')
        self.assertEqual(len(played_audio), 1)

    def test_generate_voice_output_with_caching(self):
        """Test voice output generation with caching."""
        # Setup cached result
        cache_key = self.voice_service._generate_text_cache_key("Hello world", "therapist")
        mock_result = Mock()
        mock_result.audio_data = b'cached_audio'

        self.voice_service._cache_tts_result(cache_key, mock_result)

        # Generate voice output (should use cache)
        async def run_test():
            result = await self.voice_service.generate_voice_output("Hello world", "session1")
            return result

        result = asyncio.run(run_test())

        # Verify cached result was used
        self.assertEqual(result.audio_data, b'cached_audio')

    def test_process_voice_input_error_handling(self):
        """Test error handling in voice input processing."""
        # Mock STT service to raise exception
        self.voice_service.stt_service = Mock()
        self.voice_service.stt_service.transcribe_audio = AsyncMock(side_effect=Exception("STT error"))

        audio_data = Mock()
        audio_data.data = np.array([0.1, 0.2, 0.3])

        # Process voice input (should handle error gracefully)
        async def run_test():
            result = await self.voice_service.process_voice_input(audio_data)
            return result

        result = asyncio.run(run_test())

        # Should return None on error
        self.assertIsNone(result)

    def test_generate_voice_output_error_handling(self):
        """Test error handling in voice output generation."""
        # Mock TTS service to raise exception
        self.voice_service.tts_service = Mock()
        self.voice_service.tts_service.synthesize_speech = AsyncMock(side_effect=Exception("TTS error"))

        # Generate voice output (should handle error gracefully)
        async def run_test():
            result = await self.voice_service.generate_voice_output("Hello world")
            return result

        result = asyncio.run(run_test())

        # Should return None on error
        self.assertIsNone(result)


class TestMockServices(unittest.TestCase):
    """Test optimized mock services."""

    def test_optimized_stt_service(self):
        """Test optimized STT service."""
        stt_service = OptimizedSTTService()

        # Test transcription
        async def run_test():
            audio_data = Mock()
            audio_data.data = np.array([0.1, 0.2, 0.3])
            result = await stt_service.transcribe_audio(audio_data)
            return result

        result = asyncio.run(run_test())

        # Verify mock result
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "This is an optimized transcription result")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.provider, "optimized_stt")
        self.assertFalse(result.is_crisis)
        self.assertFalse(result.is_command)

    def test_optimized_tts_service(self):
        """Test optimized TTS service."""
        tts_service = OptimizedTTSService()

        # Test synthesis
        async def run_test():
            result = await tts_service.synthesize_speech("Hello world", "therapist")
            return result

        result = asyncio.run(run_test())

        # Verify mock result
        self.assertIsNotNone(result)
        self.assertEqual(result.audio_data, b"optimized_audio_Hello world")
        self.assertEqual(result.provider, "optimized_tts")
        self.assertEqual(result.voice, "therapist")
        self.assertEqual(result.sample_rate, 22050)

    def test_optimized_command_processor(self):
        """Test optimized command processor."""
        command_processor = OptimizedCommandProcessor()

        # Test command processing
        async def run_test():
            result = await command_processor.process_text("some text")
            return result

        result = asyncio.run(run_test())

        # Should return None (no commands detected)
        self.assertIsNone(result)

        # Test command execution
        async def run_execute():
            execution_result = await command_processor.execute_command({'command': 'test'})
            return execution_result

        execution_result = asyncio.run(run_execute())

        # Should return success
        self.assertTrue(execution_result['success'])


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations."""

    def setUp(self):
        """Set up performance tests."""
        self.mock_config = Mock()
        self.mock_config.voice_enabled = True
        self.mock_config.max_concurrent_sessions = 10

        self.mock_security = Mock()

        self.voice_service = OptimizedVoiceService(self.mock_config, self.mock_security)

    def test_thread_pool_configuration(self):
        """Test thread pool configuration."""
        # Verify thread pool was created with correct configuration
        self.assertIsNotNone(self.voice_service.executor)
        self.assertEqual(self.voice_service.executor._max_workers, min(32, (os.cpu_count() or 1) + 4))
        self.assertEqual(self.voice_service.executor._thread_name_prefix, "voice_opt")

    def test_processing_queue_configuration(self):
        """Test processing queue configuration."""
        # Verify processing queue was created with correct size
        self.assertIsNotNone(self.voice_service.processing_queue)
        self.assertEqual(self.voice_service.processing_queue.maxsize, 100)

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        # Create sessions with cache hits
        session1 = self.voice_service.create_session("session1")
        session2 = self.voice_service.create_session("session2")

        # Simulate cache hits
        session1.metrics['cache_hits'] = 5
        session1.metrics['total_interactions'] = 10
        session2.metrics['cache_hits'] = 3
        session2.metrics['total_interactions'] = 6

        # Update cache hit rate
        self.voice_service._update_cache_hit_rate()

        # Calculate expected rate: (5 + 3) / (10 + 6) = 0.5
        expected_rate = 0.5
        self.assertEqual(self.voice_service.metrics['cache_hit_rate'], expected_rate)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)