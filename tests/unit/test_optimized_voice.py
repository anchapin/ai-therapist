"""
Comprehensive unit tests for voice/optimized_voice_service.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any

# Import modules using absolute imports for better compatibility
import os
import sys

# Add the project root to Python path for reliable imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import with improved error handling for __spec__ compatibility
try:
    from voice.optimized_voice_service import (
        OptimizedVoiceService,
        VoiceSession,
        VoiceCommand,
        VoiceServiceState,
        OptimizedAudioData
    )
except AttributeError as e:
    if "__spec__" in str(e):
        # Handle __spec__ compatibility issue during test collection
        import importlib.util

        # Load modules manually to avoid __spec__ issues
        def safe_import_module(module_name, from_path):
            # Extract the actual module filename from the module name
            module_filename = module_name.split(".")[-1]
            spec = importlib.util.spec_from_file_location(
                module_name,
                from_path / f"{module_filename}.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            return None

        # Get the voice module path
        voice_path = project_root / "voice"
        voice_module = safe_import_module("voice.optimized_voice_service", voice_path)

        if voice_module:
            OptimizedVoiceService = voice_module.OptimizedVoiceService
            VoiceSession = voice_module.VoiceSession
            VoiceCommand = voice_module.VoiceCommand
            VoiceServiceState = voice_module.VoiceServiceState
            OptimizedAudioData = voice_module.OptimizedAudioData
        else:
            raise ImportError("Could not import voice.optimized_voice_service")
    else:
        raise

class TestOptimizedVoiceService(unittest.TestCase):
    """Test OptimizedVoiceService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'stt_provider': 'openai',
            'tts_provider': 'openai',
            'audio_sample_rate': 16000,
            'max_session_duration': 3600,
            'max_sessions': 10,
            'enable_caching': True
        }
        self.service = OptimizedVoiceService(self.config)

    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.stt_provider, 'openai')
        self.assertEqual(self.service.tts_provider, 'openai')
        self.assertEqual(self.service.audio_sample_rate, 16000)
        self.assertTrue(self.service.enable_caching)
        self.assertEqual(self.service.state, VoiceServiceState.IDLE)

    async def test_async_initialization(self):
        """Test async initialization."""
        result = await self.service.initialize()

        self.assertTrue(result)
        self.assertEqual(self.service.state, VoiceServiceState.READY)
        self.assertTrue(self.service.is_initialized)

    async def test_session_management(self):
        """Test session lifecycle."""
        await self.service.initialize()

        # Start session
        session_id = await self.service.start_session("test_user")

        self.assertIsNotNone(session_id)
        self.assertTrue(self.service.is_session_active(session_id))

        # Get session info
        session_info = self.service.get_session_info(session_id)
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info['user_id'], 'test_user')

        # End session
        summary = await self.service.end_session(session_id)

        self.assertIn('session_id', summary)
        self.assertIn('duration', summary)
        self.assertFalse(self.service.is_session_active(session_id))

    async def test_voice_input_processing(self):
        """Test voice input processing."""
        await self.service.initialize()
        session_id = await self.service.start_session("test_user")

        # Process voice input
        audio_data = b"mock_audio_data"
        transcription = await self.service.process_voice_input(audio_data, session_id)

        self.assertIsInstance(transcription, str)
        self.assertTrue(len(transcription) > 0)

        # Check session buffer
        session_info = self.service.get_session_info(session_id)
        self.assertEqual(session_info['audio_count'], 1)
        self.assertEqual(session_info['transcript_count'], 1)

    async def test_voice_output_generation(self):
        """Test voice output generation."""
        await self.service.initialize()
        session_id = await self.service.start_session("test_user")

        # Generate voice output
        text = "Hello, this is a test."
        audio_data = await self.service.generate_voice_output(text, session_id)

        self.assertIsInstance(audio_data, bytes)
        self.assertTrue(len(audio_data) > 0)

    async def test_command_processing(self):
        """Test command processing."""
        await self.service.initialize()
        session_id = await self.service.start_session("test_user")

        # Process command
        response = await self.service.process_command("hello", session_id)

        self.assertIn('status', response)
        self.assertIn('response', response)
        self.assertEqual(response['status'], 'success')

    def test_service_statistics(self):
        """Test service statistics."""
        stats = self.service.get_service_stats()

        self.assertIn('state', stats)
        self.assertIn('is_initialized', stats)
        self.assertIn('active_sessions', stats)
        self.assertIn('total_sessions', stats)

    def test_active_sessions_list(self):
        """Test getting active sessions list."""
        active_sessions = self.service.get_active_sessions()
        self.assertIsInstance(active_sessions, list)

    def test_session_not_found_errors(self):
        """Test error handling for non-existent sessions."""
        with self.assertRaises(Exception):
            asyncio.run(self.service.end_session("non_existent_session"))

        result = self.service.get_session_info("non_existent_session")
        self.assertIsNone(result)

class TestVoiceSession(unittest.TestCase):
    """Test VoiceSession class."""

    def test_session_creation(self):
        """Test session creation."""
        session = VoiceSession(
            session_id="test_session",
            user_id="test_user",
            start_time=time.time(),
            state=VoiceServiceState.READY
        )

        self.assertEqual(session.session_id, "test_session")
        self.assertEqual(session.user_id, "test_user")
        self.assertEqual(session.state, VoiceServiceState.READY)
        self.assertEqual(len(session.audio_buffer), 0)
        self.assertEqual(len(session.transcript_buffer), 0)

class TestVoiceCommand(unittest.TestCase):
    """Test VoiceCommand class."""

    def test_command_creation(self):
        """Test command creation."""
        command = VoiceCommand(
            command="test command",
            confidence=0.95,
            timestamp=time.time(),
            session_id="test_session"
        )

        self.assertEqual(command.command, "test command")
        self.assertEqual(command.confidence, 0.95)
        self.assertEqual(command.session_id, "test_session")

class TestAsyncMethods(unittest.TestCase):
    """Test async methods using pytest-asyncio style."""

    @pytest.mark.asyncio
    async def test_initialization_async(self):
        """Test async initialization."""
        service = OptimizedVoiceService()
        result = await service.initialize()
        self.assertTrue(result)

    @pytest.mark.asyncio
    async def test_session_lifecycle_async(self):
        """Test full session lifecycle."""
        service = OptimizedVoiceService()
        await service.initialize()

        session_id = await service.start_session("test_user")
        self.assertTrue(service.is_session_active(session_id))

        summary = await service.end_session(session_id)
        self.assertFalse(service.is_session_active(session_id))
        self.assertEqual(summary['session_id'], session_id)

    @pytest.mark.asyncio
    async def test_voice_processing_async(self):
        """Test voice processing pipeline."""
        service = OptimizedVoiceService()
        await service.initialize()
        session_id = await service.start_session("test_user")

        # Process input
        transcription = await service.process_voice_input(b"test_audio", session_id)
        self.assertIsInstance(transcription, str)

        # Generate output
        audio_output = await self.service.generate_voice_output("test text", session_id)
        self.assertIsInstance(audio_output, bytes)

if __name__ == '__main__':
    unittest.main()
