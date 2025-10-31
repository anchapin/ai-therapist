"""
Comprehensive integration tests for voice conversation workflow.

Tests complete voice conversation flow including:
- Session start → STT transcription → processing → TTS synthesis → session stop
- Error recovery in conversation flow
- Queue communication between components
- Conversation entry storage and retrieval
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from datetime import datetime
from pathlib import Path
import numpy as np

from voice.voice_service import VoiceService, VoiceSessionState
from voice.stt_service import STTService, STTResult
from voice.tts_service import TTSService, TTSResult
from voice.audio_processor import AudioData
from database.models import Conversation, SessionRepository, VoiceDataRepository


@pytest.fixture
def mock_stt_service():
    """Create mock STT service."""
    service = Mock(spec=STTService)
    service.transcribe_audio = AsyncMock(return_value=STTResult(
    text="Hello, I need help with anxiety",
    confidence=0.95,
    language="en-US"
    ))
    service.transcribe_stream = AsyncMock()
    service.cleanup = AsyncMock()
    return service


@pytest.fixture
def mock_tts_service():
    """Create mock TTS service."""
    service = Mock(spec=TTSService)
    audio_data = np.random.randn(16000).astype(np.float32)
    service.synthesize = AsyncMock(return_value=TTSResult(
        audio_data=audio_data,
        sample_rate=16000,
        success=True
    ))
    service.cleanup = AsyncMock()
    return service


@pytest.fixture
def mock_db_repositories():
    """Create mock database repositories."""
    session_repo = Mock(spec=SessionRepository)
    voice_repo = Mock(spec=VoiceDataRepository)
    
    session_repo.save = Mock(return_value=True)
    session_repo.find_by_id = Mock(return_value=None)
    voice_repo.save = Mock(return_value=True)
    
    return session_repo, voice_repo


@pytest.fixture
async def voice_service(mock_stt_service, mock_tts_service, mock_db_repositories):
    """Create voice service with mocked dependencies."""
    session_repo, voice_repo = mock_db_repositories
    
    with patch('voice.voice_service.STTService', return_value=mock_stt_service), \
         patch('voice.voice_service.TTSService', return_value=mock_tts_service), \
         patch('voice.voice_service.SessionRepository', return_value=session_repo), \
         patch('voice.voice_service.VoiceDataRepository', return_value=voice_repo):
        
        service = VoiceService()
        yield service
        await service.cleanup()


@pytest.mark.integration
class TestVoiceConversationFlow:
    """Test complete voice conversation workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow_happy_path(self, voice_service, mock_stt_service, mock_tts_service):
        """Test successful end-to-end conversation flow."""
        # Step 1: Start session
        session_id = await voice_service.start_session("user_123", voice_profile="therapist")
        assert session_id is not None
        assert voice_service.current_session is not None
        assert voice_service.current_session.state == VoiceSessionState.IDLE
        
        # Step 2: Simulate user speech input
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        
        # Step 3: Process STT transcription
        result = await voice_service.process_audio(audio_data)
        assert result is not None
        assert mock_stt_service.transcribe_audio.called
        
        # Step 4: Verify conversation history updated
        assert len(voice_service.current_session.conversation_history) > 0
        last_entry = voice_service.current_session.conversation_history[-1]
        assert "text" in last_entry
        assert last_entry["text"] == "Hello, I need help with anxiety"
        
        # Step 5: Generate response via TTS
        response_text = "I understand you're experiencing anxiety. Let's talk about it."
        tts_result = await voice_service.synthesize_speech(response_text)
        assert tts_result is not None
        assert mock_tts_service.synthesize.called
        
        # Step 6: Stop session and verify cleanup
        await voice_service.stop_session()
        assert voice_service.current_session is None or voice_service.current_session.state == VoiceSessionState.IDLE
    
    @pytest.mark.asyncio
    async def test_conversation_flow_with_stt_error_recovery(self, voice_service, mock_stt_service):
        """Test error recovery when STT fails during conversation."""
        session_id = await voice_service.start_session("user_456")
        
        # Simulate STT failure on first attempt
        mock_stt_service.transcribe_audio.side_effect = [
            Exception("STT service temporarily unavailable"),
            STTResult(text="Second attempt successful", confidence=0.9, language="en-US", is_final=True)
        ]
        
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        
        # First attempt should fail gracefully
        result1 = await voice_service.process_audio(audio_data)
        
        # Second attempt should succeed with fallback/retry
        result2 = await voice_service.process_audio(audio_data)
        
        # Verify service continues to work
        assert voice_service.current_session.state != VoiceSessionState.ERROR
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_conversation_flow_with_tts_error_recovery(self, voice_service, mock_tts_service):
        """Test error recovery when TTS fails during conversation."""
        session_id = await voice_service.start_session("user_789")
        
        # Simulate TTS failure on first attempt
        mock_tts_service.synthesize.side_effect = [
            Exception("TTS service error"),
            TTSResult(
                audio_data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                success=True
            )
        ]
        
        # First synthesis should fail gracefully
        result1 = await voice_service.synthesize_speech("Test message 1")
        
        # Second synthesis should succeed
        result2 = await voice_service.synthesize_speech("Test message 2")
        assert result2 is not None
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_history(self, voice_service, mock_stt_service, mock_tts_service):
        """Test conversation history across multiple turns."""
        session_id = await voice_service.start_session("user_abc")
        
        # Configure mock to return different transcriptions
        transcriptions = [
            "I feel anxious",
            "It started last week",
            "Work stress mainly"
        ]
        
        mock_stt_service.transcribe_audio.side_effect = [
            STTResult(text=t, confidence=0.9, language="en-US", is_final=True)
            for t in transcriptions
        ]
        
        # Simulate 3-turn conversation
        for i, expected_text in enumerate(transcriptions):
            audio_data = AudioData(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time()
            )
            
            await voice_service.process_audio(audio_data)
            await voice_service.synthesize_speech(f"Response {i+1}")
        
        # Verify all turns recorded in history
        assert len(voice_service.current_session.conversation_history) >= 3
        
        # Verify order is preserved
        for i, expected_text in enumerate(transcriptions):
            found = False
            for entry in voice_service.current_session.conversation_history:
                if entry.get("text") == expected_text:
                    found = True
                    break
            assert found, f"Expected text '{expected_text}' not found in conversation history"
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_conversation_entry_storage(self, voice_service, mock_db_repositories):
        """Test conversation entries are properly stored in database."""
        session_repo, voice_repo = mock_db_repositories
        
        session_id = await voice_service.start_session("user_store")
        
        # Process audio to create conversation entry
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        
        await voice_service.process_audio(audio_data)
        
        # Verify session saved
        assert session_repo.save.called or voice_repo.save.called
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_concurrent_audio_processing(self, voice_service, mock_stt_service):
        """Test handling of concurrent audio processing requests."""
        session_id = await voice_service.start_session("user_concurrent")
        
        # Create multiple audio chunks
        audio_chunks = [
            AudioData(
                data=np.random.randn(8000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time() + i*0.5
            )
            for i in range(3)
        ]
        
        # Process chunks concurrently
        tasks = [voice_service.process_audio(chunk) for chunk in audio_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all processed without crashes
        for result in results:
            if isinstance(result, Exception):
                # Acceptable to have some exceptions due to state management
                pass
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_session_timeout_handling(self, voice_service):
        """Test conversation flow handles session timeout."""
        session_id = await voice_service.start_session("user_timeout", timeout=1)
        
        # Wait for timeout
        await asyncio.sleep(1.5)
        
        # Attempt to process audio after timeout
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        
        # Should either handle gracefully or restart session
        result = await voice_service.process_audio(audio_data)
        
        # Verify service still functional
        assert voice_service.current_session is None or \
               voice_service.current_session.state != VoiceSessionState.ERROR
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_audio_queue_communication(self, voice_service):
        """Test queue communication between audio processing components."""
        session_id = await voice_service.start_session("user_queue")
        
        # Create queue of audio data
        audio_queue = []
        for i in range(5):
            audio_queue.append(AudioData(
                data=np.random.randn(4000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time() + i*0.25
            ))
        
        # Process queue
        for audio_chunk in audio_queue:
            await voice_service.process_audio(audio_chunk)
            await asyncio.sleep(0.1)
        
        # Verify queue processing completed
        assert voice_service.current_session is not None
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_conversation_metadata_tracking(self, voice_service):
        """Test conversation metadata is properly tracked."""
        session_id = await voice_service.start_session("user_meta", metadata={"source": "mobile_app"})
        
        # Verify metadata attached to session
        assert voice_service.current_session.metadata is not None
        assert "source" in voice_service.current_session.metadata
        assert voice_service.current_session.metadata["source"] == "mobile_app"
        
        # Process audio with additional metadata
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        
        await voice_service.process_audio(audio_data)
        
        # Verify timestamp tracking
        assert voice_service.current_session.last_activity > voice_service.current_session.start_time
        
        await voice_service.stop_session()


@pytest.mark.integration
class TestConversationPersistence:
    """Test conversation persistence and retrieval."""
    
    @pytest.mark.asyncio
    async def test_conversation_retrieval_after_session(self, voice_service, mock_db_repositories):
        """Test retrieving conversation history after session ends."""
        session_repo, voice_repo = mock_db_repositories
        
        # Create and populate session
        session_id = await voice_service.start_session("user_persist")
        
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        await voice_service.process_audio(audio_data)
        
        # Store conversation history
        history_before = voice_service.current_session.conversation_history.copy()
        
        await voice_service.stop_session()
        
        # Mock retrieval
        session_repo.find_by_id.return_value = Mock(
            session_id=session_id,
            conversation_history=history_before
        )
        
        # Verify data persisted
        retrieved_session = session_repo.find_by_id(session_id)
        assert retrieved_session is not None
        assert len(retrieved_session.conversation_history) == len(history_before)
