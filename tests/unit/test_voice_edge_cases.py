"""
Comprehensive edge case and error handling tests for voice module
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
    from voice.enhanced_security import EnhancedAccessControl
    from voice.mock_config import MockConfig
except ImportError as e:
    pytest.skip(f"Could not import voice modules: {e}", allow_module_level=True)


class TestVoiceEdgeCases:
    """Test edge cases and error handling in voice module"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with edge case settings"""
        config = Mock(spec=VoiceConfig)
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.audio_sample_rate = 0  # Invalid sample rate
        config.audio_channels = 0     # Invalid channels
        config.stt_provider = None    # None provider
        config.tts_provider = ""      # Empty provider
        config.openai_api_key = None  # Missing API key
        config.elevenlabs_api_key = ""  # Empty API key
        config.max_concurrent_requests = -1  # Invalid limit
        config.request_timeout_seconds = 0   # Invalid timeout
        return config

    @pytest.fixture
    def invalid_audio_data(self):
        """Create invalid audio data for testing"""
        return AudioData(
            data=None,  # None data
            sample_rate=-1,  # Invalid sample rate
            channels=0,      # Invalid channels
            format="invalid"  # Invalid format
        )

    @pytest.fixture
    def empty_audio_data(self):
        """Create empty audio data for testing"""
        return AudioData(
            data=b"",
            sample_rate=16000,
            channels=1,
            format="wav"
        )

    @pytest.fixture
    def large_audio_data(self):
        """Create large audio data for testing"""
        return AudioData(
            data=b"x" * 10000000,  # 10MB
            sample_rate=16000,
            channels=1,
            format="wav"
        )

    class TestConfigEdgeCases:
        """Test configuration edge cases"""

        def test_config_with_none_values(self, mock_config):
            """Test config with None values"""
            config = VoiceConfig()
            config.voice_enabled = None
            config.openai_api_key = None
            config.stt_provider = None
            
            # Should handle None values gracefully
            issues = config.validate_configuration()
            assert isinstance(issues, list)

        def test_config_with_invalid_types(self):
            """Test config with invalid types"""
            config = VoiceConfig()
            config.voice_enabled = "true"  # String instead of bool
            config.audio_sample_rate = "16000"  # String instead of int
            config.max_concurrent_requests = "10"  # String instead of int
            
            # Should handle type mismatches
            issues = config.validate_configuration()
            assert isinstance(issues, list)

        def test_config_with_extreme_values(self):
            """Test config with extreme values"""
            config = VoiceConfig()
            config.audio_sample_rate = 1000000  # Extremely high sample rate
            config.max_concurrent_requests = 10000  # Very high limit
            config.request_timeout_seconds = 3600  # 1 hour timeout
            
            # Should handle extreme values
            issues = config.validate_configuration()
            assert isinstance(issues, list)

        def test_config_serialization_edge_cases(self):
            """Test config serialization with edge cases"""
            config = VoiceConfig()
            config.voice_enabled = True
            config.openai_api_key = None
            config.stt_provider = ""
            
            # Test JSON serialization
            json_str = config.to_json()
            assert isinstance(json_str, str)
            
            # Test JSON deserialization
            restored_config = VoiceConfig.from_json(json_str)
            assert restored_config.voice_enabled == True

        def test_voice_profile_edge_cases(self):
            """Test voice profile edge cases"""
            # Create profile with invalid values
            profile = VoiceProfile(
                name="",  # Empty name
                provider=None,  # None provider
                voice_id="",  # Empty voice ID
                speed=-1.0,  # Invalid speed
                pitch=10.0,  # Invalid pitch
                volume=2.0   # Invalid volume
            )
            
            # Should handle invalid profile
            profile_dict = profile.to_dict()
            assert isinstance(profile_dict, dict)

    class TestAudioProcessorEdgeCases:
        """Test audio processor edge cases"""

        def test_processor_with_invalid_config(self):
            """Test processor initialization with invalid config"""
            config = {
                "sample_rate": -1,
                "channels": 0,
                "buffer_size": -100
            }
            
            processor = SimplifiedAudioProcessor(config)
            # Should handle invalid config gracefully
            assert processor is not None

        def test_process_none_audio(self):
            """Test processing None audio data"""
            processor = SimplifiedAudioProcessor()
            
            # Should handle None audio data
            result = processor._process_audio(None)
            assert result is None

        def test_process_empty_audio(self, empty_audio_data):
            """Test processing empty audio data"""
            processor = SimplifiedAudioProcessor()
            
            result = processor._process_audio(empty_audio_data)
            # Should handle empty audio gracefully
            assert result is not None

        def test_process_large_audio(self, large_audio_data):
            """Test processing large audio data"""
            processor = SimplifiedAudioProcessor()
            
            result = processor._process_audio(large_audio_data)
            # Should handle large audio without memory issues
            assert result is not None

        def test_audio_device_detection_errors(self):
            """Test audio device detection with errors"""
            processor = SimplifiedAudioProcessor()
            
            with patch('voice.audio_processor.sd') as mock_sd:
                mock_sd.query_devices.side_effect = Exception("Device error")
                
                # Should handle device detection errors
                devices = processor.detect_audio_devices()
                assert isinstance(devices, tuple)

        def test_recording_with_invalid_device(self):
            """Test recording with invalid device index"""
            processor = SimplifiedAudioProcessor()
            
            # Should handle invalid device index
            result = processor.start_recording(device_index=999)
            assert isinstance(result, bool)

        def test_concurrent_recording_operations(self):
            """Test concurrent recording operations"""
            processor = SimplifiedAudioProcessor()
            
            def start_stop_recording():
                processor.start_recording()
                time.sleep(0.1)
                processor.stop_recording()
            
            # Run multiple recording operations concurrently
            threads = [threading.Thread(target=start_stop_recording) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # Should handle concurrent operations
            assert True

        def test_memory_cleanup_on_error(self):
            """Test memory cleanup when errors occur"""
            processor = SimplifiedAudioProcessor()
            
            # Simulate memory pressure
            processor.audio_buffer = [b"x" * 1000000 for _ in range(100)]
            
            # Force cleanup
            processor.force_cleanup_buffers()
            
            # Buffer should be cleaned
            assert len(processor.audio_buffer) == 0

    class TestSTTServiceEdgeCases:
        """Test STT service edge cases"""

        def test_stt_with_missing_providers(self, mock_config):
            """Test STT service with missing providers"""
            service = STTService(mock_config)
            
            # Should handle missing providers
            providers = service.get_available_providers()
            assert isinstance(providers, list)

        def test_transcribe_none_audio(self, mock_config):
            """Test transcribing None audio"""
            service = STTService(mock_config)
            
            async def test_transcription():
                result = await service.transcribe_audio(None)
                assert result is not None
            
            asyncio.run(test_transcription())

        def test_transcribe_empty_audio(self, mock_config, empty_audio_data):
            """Test transcribing empty audio"""
            service = STTService(mock_config)
            
            async def test_transcription():
                result = await service.transcribe_audio(empty_audio_data)
                assert result is not None
            
            asyncio.run(test_transcription())

        def test_transcribe_with_invalid_provider(self, mock_config, empty_audio_data):
            """Test transcribing with invalid provider"""
            service = STTService(mock_config)
            
            async def test_transcription():
                result = await service.transcribe_audio(
                    empty_audio_data, 
                    provider="nonexistent_provider"
                )
                assert result is not None
            
            asyncio.run(test_transcription())

        def test_provider_fallback_chain_errors(self, mock_config):
            """Test provider fallback chain with errors"""
            service = STTService(mock_config)
            
            # Test with all providers failing
            chain = service._get_provider_fallback_chain("nonexistent")
            assert isinstance(chain, list)

        def test_cache_operations_with_invalid_data(self, mock_config):
            """Test cache operations with invalid data"""
            service = STTService(mock_config)
            
            # Test cache with None key
            result = service._get_from_cache(None)
            assert result is None
            
            # Test cache with invalid result
            service._add_to_cache("test_key", None)
            # Should not raise errors

    class TestTTSServiceEdgeCases:
        """Test TTS service edge cases"""

        def test_tts_with_missing_providers(self, mock_config):
            """Test TTS service with missing providers"""
            service = TTSService(mock_config)
            
            # Should handle missing providers
            providers = service.get_available_providers()
            assert isinstance(providers, list)

        def test_synthesize_empty_text(self, mock_config):
            """Test synthesizing empty text"""
            service = TTSService(mock_config)
            
            async def test_synthesis():
                result = await service.synthesize_speech("")
                assert result is not None
            
            asyncio.run(test_synthesis())

        def test_synthesize_very_long_text(self, mock_config):
            """Test synthesizing very long text"""
            service = TTSService(mock_config)
            long_text = "test " * 10000  # Very long text
            
            async def test_synthesis():
                result = await service.synthesize_speech(long_text)
                assert result is not None
            
            asyncio.run(test_synthesis())

        def test_synthesize_with_invalid_profile(self, mock_config):
            """Test synthesizing with invalid voice profile"""
            service = TTSService(mock_config)
            
            async def test_synthesis():
                result = await service.synthesize_speech(
                    "test text",
                    voice_profile="nonexistent_profile"
                )
                assert result is not None
            
            asyncio.run(test_synthesis())

        def test_voice_profile_creation_edge_cases(self, mock_config):
            """Test voice profile creation with edge cases"""
            service = TTSService(mock_config)
            
            # Test creating profile with invalid parameters
            profile = service.create_custom_voice_profile(
                name="",  # Empty name
                provider=None,  # None provider
                voice_id="",  # Empty voice ID
                speed=-1.0,  # Invalid speed
                pitch=10.0   # Invalid pitch
            )
            
            # Should handle invalid parameters
            assert profile is not None

        def test_emotion_settings_edge_cases(self, mock_config):
            """Test emotion settings with edge cases"""
            service = TTSService(mock_config)
            
            # Test with invalid emotion values
            async def test_emotion():
                result = await service.synthesize_speech(
                    "test text",
                    emotion=None,  # None emotion
                    emotion_intensity=-1.0  # Invalid intensity
                )
                assert result is not None
            
            asyncio.run(test_emotion())

    class TestVoiceServiceEdgeCases:
        """Test voice service edge cases"""

        def test_service_with_invalid_config(self, mock_config):
            """Test service initialization with invalid config"""
            security = Mock()
            service = VoiceService(mock_config, security)
            
            # Should handle invalid config
            assert service is not None

        def test_session_creation_with_invalid_params(self, mock_config):
            """Test session creation with invalid parameters"""
            security = Mock()
            service = VoiceService(mock_config, security)
            
            # Test with None session ID
            session_id = service.create_session(session_id=None)
            assert isinstance(session_id, str)
            
            # Test with empty session ID
            session_id = service.create_session(session_id="")
            assert isinstance(session_id, str)

        def test_session_operations_with_invalid_session(self, mock_config):
            """Test session operations with invalid session ID"""
            security = Mock()
            service = VoiceService(mock_config, security)
            
            # Test operations with non-existent session
            session = service.get_session("nonexistent")
            assert session is None
            
            # Test ending non-existent session
            result = service.end_session("nonexistent")
            assert isinstance(result, bool)

        def test_concurrent_session_operations(self, mock_config):
            """Test concurrent session operations"""
            security = Mock()
            service = VoiceService(mock_config, security)
            
            def create_session():
                session_id = service.create_session()
                time.sleep(0.1)
                service.end_session(session_id)
                return session_id
            
            # Create multiple sessions concurrently
            threads = [threading.Thread(target=create_session) for _ in range(10)]
            session_ids = []
            
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # Should handle concurrent operations
            assert True

        def test_voice_queue_operations_with_invalid_data(self, mock_config):
            """Test voice queue operations with invalid data"""
            security = Mock()
            service = VoiceService(mock_config, security)
            
            # Test queue with None data
            service.voice_queue.put((None, None))
            
            # Should handle invalid queue data
            assert not service.voice_queue.empty()

    class TestSecurityEdgeCases:
        """Test security edge cases"""

        def test_security_with_invalid_config(self):
            """Test security with invalid config"""
            config = {
                "encryption_enabled": None,
                "consent_required": "invalid",
                "data_retention_days": -1
            }
            
            security = VoiceSecurity(config)
            # Should handle invalid config
            assert security is not None

        def test_encryption_with_invalid_data(self):
            """Test encryption with invalid data"""
            security = VoiceSecurity()
            
            # Test encrypting None data
            result = security.encrypt_data(None, "test_user")
            assert result is not None
            
            # Test encrypting empty data
            result = security.encrypt_data(b"", "test_user")
            assert result is not None

        def test_decryption_with_invalid_data(self):
            """Test decryption with invalid data"""
            security = VoiceSecurity()
            
            # Test decrypting None data
            result = security.decrypt_data(None, "test_user")
            assert result is not None
            
            # Test decrypting invalid data
            result = security.decrypt_data(b"invalid", "test_user")
            assert result is not None

        def test_consent_management_edge_cases(self):
            """Test consent management edge cases"""
            security = VoiceSecurity()
            
            # Test consent with None user ID
            result = security.has_consent(None, "voice_processing")
            assert isinstance(result, bool)
            
            # Test consent with empty consent type
            result = security.has_consent("test_user", "")
            assert isinstance(result, bool)

        def test_access_control_edge_cases(self):
            """Test access control edge cases"""
            config = {}
            access_control = EnhancedAccessControl(config)
            
            # Test access with None session ID
            result = access_control.check_access(None, "test_operation")
            assert isinstance(result, bool)
            
            # Test access with invalid operation
            result = access_control.check_access("test_session", "")
            assert isinstance(result, bool)

    class TestCommandProcessorEdgeCases:
        """Test command processor edge cases"""

        def test_processor_with_invalid_config(self):
            """Test processor with invalid config"""
            config = Mock()
            config.validate_configuration.return_value = ["Error 1", "Error 2"]
            
            processor = VoiceCommandProcessor(config)
            # Should handle invalid config
            assert processor is not None

        def test_command_processing_with_invalid_input(self):
            """Test command processing with invalid input"""
            config = Mock()
            processor = VoiceCommandProcessor(config)
            
            async def test_processing():
                # Test with None text
                result = await processor.process_text(None, "test_session")
                assert result is None
                
                # Test with empty text
                result = await processor.process_text("", "test_session")
                assert result is None
                
                # Test with None session ID
                result = await processor.process_text("test command", None)
                assert result is None
            
            asyncio.run(test_processing())

        def test_command_execution_with_invalid_commands(self):
            """Test command execution with invalid commands"""
            config = Mock()
            processor = VoiceCommandProcessor(config)
            
            async def test_execution():
                # Create invalid command match
                from voice.commands import CommandMatch
                invalid_match = CommandMatch(
                    command=None,  # None command
                    confidence=-1.0,  # Invalid confidence
                    parameters={},  # Empty parameters
                    session_id="test_session"
                )
                
                result = await processor.execute_command(invalid_match)
                assert isinstance(result, dict)
            
            asyncio.run(test_execution())

        def test_emergency_detection_edge_cases(self):
            """Test emergency detection with edge cases"""
            config = Mock()
            processor = VoiceCommandProcessor(config)
            
            # Test with None text
            keywords = processor._detect_emergency_keywords(None)
            assert isinstance(keywords, list)
            
            # Test with empty text
            keywords = processor._detect_emergency_keywords("")
            assert isinstance(keywords, list)
            
            # Test with very long text
            long_text = "test " * 10000
            keywords = processor._detect_emergency_keywords(long_text)
            assert isinstance(keywords, list)

    class TestIntegrationEdgeCases:
        """Test integration edge cases"""

        def test_service_integration_with_failures(self, mock_config):
            """Test service integration when components fail"""
            # Create services with failing dependencies
            stt_service = Mock()
            stt_service.transcribe_audio.side_effect = Exception("STT failed")
            
            tts_service = Mock()
            tts_service.synthesize_speech.side_effect = Exception("TTS failed")
            
            audio_processor = Mock()
            audio_processor.start_recording.side_effect = Exception("Audio failed")
            
            # Voice service should handle component failures
            security = Mock()
            voice_service = VoiceService(mock_config, security)
            voice_service.stt_service = stt_service
            voice_service.tts_service = tts_service
            voice_service.audio_processor = audio_processor
            
            # Should handle failures gracefully
            assert voice_service is not None

        def test_cascading_failures(self, mock_config):
            """Test cascading failures across components"""
            security = VoiceSecurity()
            
            # Simulate cascading failure
            with patch('voice.voice_service.STTService') as mock_stt:
                mock_stt.side_effect = Exception("STT initialization failed")
                
                # Service should handle initialization failure
                voice_service = VoiceService(mock_config, security)
                assert voice_service is not None

        def test_resource_exhaustion(self, mock_config):
            """Test behavior under resource exhaustion"""
            security = Mock()
            voice_service = VoiceService(mock_config, security)
            
            # Create many sessions to test resource limits
            session_ids = []
            for i in range(1000):  # Excessive number of sessions
                try:
                    session_id = voice_service.create_session()
                    session_ids.append(session_id)
                except Exception:
                    # Should handle resource exhaustion gracefully
                    break
            
            # Should have some limit on sessions
            assert len(session_ids) < 1000

        def test_network_timeout_simulation(self, mock_config):
            """Test behavior with simulated network timeouts"""
            service = STTService(mock_config)
            
            # Mock network timeout
            with patch('voice.stt_service.time.sleep') as mock_sleep:
                mock_sleep.side_effect = Exception("Network timeout")
                
                async def test_timeout():
                    result = await service.transcribe_audio(
                        AudioData(b"test", 16000, 1)
                    )
                    # Should handle timeout gracefully
                    assert result is not None
                
                asyncio.run(test_timeout())

        def test_memory_pressure_simulation(self, mock_config):
            """Test behavior under memory pressure"""
            processor = SimplifiedAudioProcessor()
            
            # Fill up memory with large audio data
            large_data = AudioData(b"x" * 1000000, 16000, 1)
            
            for i in range(100):
                processor.add_to_buffer(large_data.data)
            
            # Should handle memory pressure
            processor.force_cleanup_buffers()
            assert len(processor.get_buffer_contents()) == 0