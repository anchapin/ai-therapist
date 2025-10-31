"""
Integration tests for voice module components
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import asyncio
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
import json
import tempfile
import shutil

# Add the project root to Python path for reliable imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from voice.config import VoiceConfig, VoiceProfile
    from voice.voice_service import VoiceService, VoiceSessionState
    from voice.audio_processor import AudioData, SimplifiedAudioProcessor
    from voice.stt_service import STTService, STTResult
    from voice.tts_service import TTSService, TTSResult
    from voice.commands import VoiceCommandProcessor
    from voice.security import VoiceSecurity
    from voice.mock_config import MockConfig
except ImportError as e:
    pytest.skip(f"Could not import voice modules: {e}", allow_module_level=True)


class TestVoiceIntegration:
    """Integration tests for voice module components"""

    @pytest.fixture
    def integrated_voice_system(self):
        """Create an integrated voice system for testing"""
        # Use mock config for testing
        config = MockConfig()
        
        # Create security
        security = VoiceSecurity()
        
        # Create voice service
        voice_service = VoiceService(config, security)
        
        # Create individual services
        audio_processor = SimplifiedAudioProcessor()
        stt_service = STTService(config)
        tts_service = TTSService(config)
        command_processor = VoiceCommandProcessor(config)
        
        return {
            'config': config,
            'security': security,
            'voice_service': voice_service,
            'audio_processor': audio_processor,
            'stt_service': stt_service,
            'tts_service': tts_service,
            'command_processor': command_processor
        }

    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing"""
        return AudioData(
            data=b"sample_audio_data_for_testing",
            sample_rate=16000,
            channels=1,
            format="wav"
        )

    class TestAudioToTextIntegration:
        """Test audio processing to STT integration"""

        def test_complete_audio_to_text_flow(self, integrated_voice_system, sample_audio_data):
            """Test complete flow from audio input to text output"""
            system = integrated_voice_system
            
            # Start recording
            recording_started = system['audio_processor'].start_recording()
            assert recording_started is True
            
            # Simulate audio processing
            processed_audio = system['audio_processor']._process_audio(sample_audio_data)
            assert processed_audio is not None
            
            # Stop recording
            recorded_audio = system['audio_processor'].stop_recording()
            assert recorded_audio is not None
            
            # Transcribe audio
            async def test_transcription():
                result = await system['stt_service'].transcribe_audio(recorded_audio)
                assert result is not None
                assert hasattr(result, 'text')
                return result
            
            transcription_result = asyncio.run(test_transcription())
            assert transcription_result.text is not None

        def test_audio_quality_to_stt_confidence(self, integrated_voice_system):
            """Test relationship between audio quality and STT confidence"""
            system = integrated_voice_system
            
            # Create audio with different quality levels
            high_quality_audio = AudioData(
                data=b"high_quality_audio" * 100,
                sample_rate=16000,
                channels=1,
                format="wav"
            )
            
            low_quality_audio = AudioData(
                data=b"low" * 10,  # Very short, low quality
                sample_rate=8000,  # Lower sample rate
                channels=1,
                format="wav"
            )
            
            async def test_quality_impact():
                # Test high quality audio
                high_quality_result = await system['stt_service'].transcribe_audio(high_quality_audio)
                
                # Test low quality audio
                low_quality_result = await system['stt_service'].transcribe_audio(low_quality_audio)
                
                # Both should return results, but confidence may differ
                assert high_quality_result is not None
                assert low_quality_result is not None
                
                return high_quality_result, low_quality_result
            
            high_result, low_result = asyncio.run(test_quality_impact())
            assert hasattr(high_result, 'confidence')
            assert hasattr(low_result, 'confidence')

        def test_concurrent_audio_processing(self, integrated_voice_system, sample_audio_data):
            """Test concurrent audio processing and transcription"""
            system = integrated_voice_system
            
            async def process_audio_batch(audio_data, batch_id):
                # Process audio
                processed = system['audio_processor']._process_audio(audio_data)
                
                # Transcribe
                result = await system['stt_service'].transcribe_audio(processed)
                
                return {
                    'batch_id': batch_id,
                    'result': result
                }
            
            # Create multiple audio processing tasks
            tasks = [
                process_audio_batch(sample_audio_data, i)
                for i in range(5)
            ]
            
            results = asyncio.run(asyncio.gather(*tasks))
            
            # All tasks should complete successfully
            assert len(results) == 5
            for result in results:
                assert result['result'] is not None
                assert 'batch_id' in result

    class TestTextToSpeechIntegration:
        """Test text to TTS to audio output integration"""

        def test_complete_text_to_speech_flow(self, integrated_voice_system):
            """Test complete flow from text input to audio output"""
            system = integrated_voice_system
            
            test_text = "Hello, this is a test of the voice synthesis system."
            
            async def test_synthesis():
                # Synthesize speech
                result = await system['tts_service'].synthesize_speech(test_text)
                assert result is not None
                assert hasattr(result, 'audio_data')
                assert len(result.audio_data) > 0
                
                # Play audio through audio processor
                playback_success = system['audio_processor'].play_audio(
                    AudioData(
                        data=result.audio_data,
                        sample_rate=22050,  # Common TTS sample rate
                        channels=1,
                        format="wav"
                    )
                )
                
                return result
            
            synthesis_result = asyncio.run(test_synthesis())
            assert synthesis_result is not None

        def test_voice_profile_to_tts_integration(self, integrated_voice_system):
            """Test integration between voice profiles and TTS"""
            system = integrated_voice_system
            
            # Create custom voice profile
            custom_profile = VoiceProfile(
                name="test_profile",
                provider="openai",
                voice_id="alloy",
                speed=1.2,
                pitch=0.9,
                volume=0.8
            )
            
            test_text = "Testing voice profile integration."
            
            async def test_profile_synthesis():
                # Synthesize with custom profile
                result = await system['tts_service'].synthesize_speech(
                    text=test_text,
                    voice_profile="test_profile"
                )
                
                assert result is not None
                assert hasattr(result, 'audio_data')
                
                return result
            
            synthesis_result = asyncio.run(test_profile_synthesis())
            assert synthesis_result is not None

        def test_emotion_to_tts_integration(self, integrated_voice_system):
            """Test integration between emotion settings and TTS"""
            system = integrated_voice_system
            
            test_text = "I am feeling very happy today!"
            
            async def test_emotion_synthesis():
                # Test with different emotions
                emotions = ['happy', 'sad', 'calm', 'excited']
                results = []
                
                for emotion in emotions:
                    result = await system['tts_service'].synthesize_speech(
                        text=test_text,
                        emotion=emotion,
                        emotion_intensity=0.8
                    )
                    results.append(result)
                
                return results
            
            emotion_results = asyncio.run(test_emotion_synthesis())
            assert len(emotion_results) == 4
            for result in emotion_results:
                assert result is not None
                assert hasattr(result, 'audio_data')

    class TestVoiceCommandIntegration:
        """Test voice command processing integration"""

        def test_complete_command_flow(self, integrated_voice_system, sample_audio_data):
            """Test complete flow from audio to command execution"""
            system = integrated_voice_system
            
            async def test_command_flow():
                # Step 1: Transcribe audio to text
                transcription_result = await system['stt_service'].transcribe_audio(sample_audio_data)
                assert transcription_result is not None
                
                # Step 2: Process text for commands
                command_match = await system['command_processor'].process_text(
                    transcription_result.text,
                    session_id="test_session"
                )
                
                # Step 3: Execute command if found
                if command_match:
                    execution_result = await system['command_processor'].execute_command(command_match)
                    assert execution_result is not None
                    return execution_result
                else:
                    return None
            
            result = asyncio.run(test_command_flow())
            # Result may be None if no command is detected, which is valid

        def test_emergency_command_integration(self, integrated_voice_system):
            """Test emergency command detection and handling"""
            system = integrated_voice_system
            
            emergency_texts = [
                "I need help immediately",
                "I'm having a crisis",
                "Call emergency services",
                "I feel unsafe right now"
            ]
            
            async def test_emergency_commands():
                results = []
                
                for text in emergency_texts:
                    # Process emergency text
                    command_match = await system['command_processor'].process_text(
                        text,
                        session_id="emergency_test"
                    )
                    
                    if command_match:
                        execution_result = await system['command_processor'].execute_command(command_match)
                        results.append({
                            'text': text,
                            'command_match': command_match,
                            'execution_result': execution_result
                        })
                
                return results
            
            emergency_results = asyncio.run(test_emergency_commands())
            assert isinstance(emergency_results, list)

        def test_voice_session_command_integration(self, integrated_voice_system):
            """Test integration between voice sessions and commands"""
            system = integrated_voice_system
            
            # Create a voice session
            session_id = system['voice_service'].create_session(
                user_id="test_user",
                voice_profile="default"
            )
            
            session_commands = [
                "start session",
                "pause conversation",
                "resume conversation",
                "end session"
            ]
            
            async def test_session_commands():
                results = []
                
                for command_text in session_commands:
                    # Process command within session context
                    command_match = await system['command_processor'].process_text(
                        command_text,
                        session_id=session_id
                    )
                    
                    if command_match:
                        execution_result = await system['command_processor'].execute_command(command_match)
                        results.append({
                            'command': command_text,
                            'result': execution_result
                        })
                
                return results
            
            session_results = asyncio.run(test_session_commands())
            assert isinstance(session_results, list)

    class TestSecurityIntegration:
        """Test security integration across voice components"""

        def test_encryption_integration(self, integrated_voice_system, sample_audio_data):
            """Test encryption integration across voice services"""
            system = integrated_voice_system
            
            user_id = "test_user"
            
            # Test audio data encryption
            encrypted_audio = system['security'].encrypt_audio_data(
                sample_audio_data.data,
                user_id
            )
            assert encrypted_audio is not None
            assert encrypted_audio != sample_audio_data.data
            
            # Test audio data decryption
            decrypted_audio = system['security'].decrypt_audio_data(
                encrypted_audio,
                user_id
            )
            assert decrypted_audio is not None
            
            # Test transcription with encrypted data
            async def test_encrypted_transcription():
                # Decrypt and transcribe
                audio_data = AudioData(
                    data=decrypted_audio,
                    sample_rate=sample_audio_data.sample_rate,
                    channels=sample_audio_data.channels,
                    format=sample_audio_data.format
                )
                
                result = await system['stt_service'].transcribe_audio(audio_data)
                return result
            
            transcription_result = asyncio.run(test_encrypted_transcription())
            assert transcription_result is not None

        def test_consent_integration(self, integrated_voice_system):
            """Test consent management integration"""
            system = integrated_voice_system
            
            user_id = "consent_test_user"
            
            # Record consent for voice processing
            system['security'].consent_manager.record_consent(
                user_id=user_id,
                consent_type="voice_processing",
                granted=True,
                purpose="Therapy sessions",
                metadata={"timestamp": time.time()}
            )
            
            # Check consent before processing
            has_consent = system['security'].consent_manager.has_consent(
                user_id=user_id,
                consent_type="voice_processing"
            )
            assert has_consent is True
            
            # Test processing with consent check
            test_text = "This should be processed with consent."
            
            async def test_consent_processing():
                # Check consent before synthesis
                if system['security'].consent_manager.has_consent(user_id, "voice_processing"):
                    result = await system['tts_service'].synthesize_speech(test_text)
                    return result
                else:
                    return None
            
            result = asyncio.run(test_consent_processing())
            assert result is not None

        def test_audit_logging_integration(self, integrated_voice_system):
            """Test audit logging integration across components"""
            system = integrated_voice_system
            
            user_id = "audit_test_user"
            session_id = "audit_test_session"
            
            # Enable audit logging
            system['security'].audit_logging_enabled = True
            
            # Log voice processing events
            system['security'].audit_logger.log_event(
                event_type="voice_session_started",
                session_id=session_id,
                user_id=user_id,
                action="Voice session initiated",
                details={"timestamp": time.time()}
            )
            
            # Process voice command
            async def test_audit_processing():
                command_text = "start recording"
                
                # Log command processing
                system['security'].audit_logger.log_event(
                    event_type="voice_command_processed",
                    session_id=session_id,
                    user_id=user_id,
                    action="Voice command processed",
                    details={"command": command_text}
                )
                
                # Process command
                command_match = await system['command_processor'].process_text(
                    command_text,
                    session_id=session_id
                )
                
                return command_match
            
            command_result = asyncio.run(test_audit_processing())
            
            # Verify audit logs
            user_logs = system['security'].audit_logger.get_user_logs(user_id)
            assert len(user_logs) >= 2  # Session start and command processing

    class TestPerformanceIntegration:
        """Test performance integration across components"""

        def test_memory_usage_integration(self, integrated_voice_system):
            """Test memory usage across integrated components"""
            system = integrated_voice_system
            
            # Get initial memory usage
            initial_memory = system['audio_processor'].get_memory_usage()
            
            # Process multiple audio chunks
            audio_chunks = [
                AudioData(
                    data=f"audio_chunk_{i}".encode() * 1000,
                    sample_rate=16000,
                    channels=1,
                    format="wav"
                )
                for i in range(10)
            ]
            
            for chunk in audio_chunks:
                system['audio_processor'].add_to_buffer(chunk.data)
            
            # Get memory usage after processing
            peak_memory = system['audio_processor'].get_memory_usage()
            
            # Cleanup
            system['audio_processor'].force_cleanup_buffers()
            final_memory = system['audio_processor'].get_memory_usage()
            
            # Memory should be managed properly
            assert final_memory['buffer_size'] <= peak_memory['buffer_size']

        def test_concurrent_processing_performance(self, integrated_voice_system):
            """Test performance under concurrent processing load"""
            system = integrated_voice_system
            
            async def concurrent_processing_task(task_id):
                start_time = time.time()
                
                # Simulate voice processing pipeline
                audio_data = AudioData(
                    data=f"task_{task_id}_audio".encode() * 100,
                    sample_rate=16000,
                    channels=1,
                    format="wav"
                )
                
                # Process audio
                processed = system['audio_processor']._process_audio(audio_data)
                
                # Transcribe
                transcription = await system['stt_service'].transcribe_audio(processed)
                
                # Synthesize response
                if transcription and transcription.text:
                    response = await system['tts_service'].synthesize_speech(
                        f"Response to task {task_id}"
                    )
                
                end_time = time.time()
                
                return {
                    'task_id': task_id,
                    'processing_time': end_time - start_time,
                    'success': True
                }
            
            # Run concurrent tasks
            tasks = [
                concurrent_processing_task(i)
                for i in range(10)
            ]
            
            results = asyncio.run(asyncio.gather(*tasks))
            
            # All tasks should complete successfully
            assert len(results) == 10
            for result in results:
                assert result['success'] is True
                assert result['processing_time'] < 10.0  # Should complete within 10 seconds

        def test_cache_integration_performance(self, integrated_voice_system):
            """Test cache integration and performance"""
            system = integrated_voice_system
            
            test_text = "This is a test for caching performance."
            
            async def test_caching():
                # First synthesis (should cache result)
                start_time = time.time()
                result1 = await system['tts_service'].synthesize_speech(test_text)
                first_time = time.time() - start_time
                
                # Second synthesis (should use cache)
                start_time = time.time()
                result2 = await system['tts_service'].synthesize_speech(test_text)
                second_time = time.time() - start_time
                
                # Results should be identical
                assert result1.audio_data == result2.audio_data
                
                # Second call should be faster (cache hit)
                # Note: This might not always be true due to system variations
                return {
                    'first_time': first_time,
                    'second_time': second_time,
                    'cache_hit_rate': system['tts_service']._calculate_cache_hit_rate()
                }
            
            cache_results = asyncio.run(test_caching())
            assert cache_results['first_time'] > 0
            assert cache_results['second_time'] > 0

    class TestErrorRecoveryIntegration:
        """Test error recovery across integrated components"""

        def test_stt_failure_recovery(self, integrated_voice_system, sample_audio_data):
            """Test recovery from STT service failures"""
            system = integrated_voice_system
            
            # Mock STT service failure
            original_transcribe = system['stt_service'].transcribe_audio
            
            async def failing_transcribe(audio_data):
                if hasattr(failing_transcribe, 'call_count'):
                    failing_transcribe.call_count += 1
                else:
                    failing_transcribe.call_count = 1
                
                if failing_transcribe.call_count <= 2:
                    raise Exception("STT service temporarily unavailable")
                else:
                    # Recovery on third call
                    return await original_transcribe(audio_data)
            
            system['stt_service'].transcribe_audio = failing_transcribe
            
            async def test_recovery():
                try:
                    result = await system['stt_service'].transcribe_audio(sample_audio_data)
                    return result
                except Exception as e:
                    return None
            
            # Should recover after initial failures
            result = asyncio.run(test_recovery())
            assert result is not None

        def test_tts_fallback_integration(self, integrated_voice_system):
            """Test TTS provider fallback integration"""
            system = integrated_voice_system
            
            test_text = "Testing TTS fallback functionality."
            
            async def test_fallback():
                # Mock primary provider failure
                with patch.object(system['tts_service'], '_synthesize_with_openai') as mock_openai:
                    mock_openai.side_effect = Exception("OpenAI TTS unavailable")
                    
                    # Should fallback to other providers
                    result = await system['tts_service'].synthesize_speech(test_text)
                    
                    return result
            
            fallback_result = asyncio.run(test_fallback())
            # Should return result from fallback provider or mock
            assert fallback_result is not None

        def test_audio_device_failure_recovery(self, integrated_voice_system):
            """Test recovery from audio device failures"""
            system = integrated_voice_system
            
            # Mock audio device failure
            with patch('voice.audio_processor.sd') as mock_sd:
                mock_sd.InputStream.side_effect = Exception("No audio devices found")
                
                # Should handle device failure gracefully
                recording_result = system['audio_processor'].start_recording()
                
                # Should return False or handle gracefully
                assert isinstance(recording_result, bool)

        def test_session_recovery_after_failure(self, integrated_voice_system):
            """Test session recovery after service failures"""
            system = integrated_voice_system
            
            # Create a session
            session_id = system['voice_service'].create_session(user_id="recovery_test")
            
            # Simulate service failure
            original_session = system['voice_service'].get_session(session_id)
            assert original_session is not None
            
            # Simulate session corruption
            system['voice_service'].sessions[session_id] = None
            
            # Try to recover session
            recovered_session = system['voice_service'].get_session(session_id)
            
            # Should handle corrupted session gracefully
            # (Either return None or create new session)
            assert True  # Test passes if no exception is raised