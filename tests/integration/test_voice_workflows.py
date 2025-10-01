"""
End-to-End Voice Workflow Integration Tests

Tests complete voice therapy scenarios including:
- Multi-turn therapy conversations
- Crisis intervention workflows
- Voice command sequences
- Session management workflows
- Emergency response scenarios
- Performance under realistic therapy conditions
"""

import pytest
import asyncio
import time
import threading
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
import psutil
import os
from typing import Dict, List, Any

from voice.voice_service import VoiceService, VoiceSessionState
from voice.config import VoiceConfig
from voice.security import VoiceSecurity
from voice.commands import VoiceCommandProcessor, CommandCategory, SecurityLevel
from voice.audio_processor import AudioData
from voice.stt_service import STTResult
from voice.tts_service import TTSResult, EmotionType


class TestVoiceWorkflowsIntegration:
    """Test end-to-end voice therapy workflows."""

    @pytest.fixture
    def therapy_session_config(self):
        """Create configuration for therapy session testing."""
        config = VoiceConfig()
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.voice_output_enabled = True
        config.voice_commands_enabled = True
        config.security_enabled = True
        config.encryption_enabled = True
        config.session_timeout = 1800  # 30 minutes for therapy sessions
        config.voice_command_wake_word = "hey therapist"
        config.voice_command_timeout = 30000
        config.voice_command_min_confidence = 0.7
        return config

    @pytest.fixture
    def therapy_security(self, therapy_session_config):
        """Create security instance for therapy sessions."""
        security = VoiceSecurity(therapy_session_config)
        security.encryption_key = b'therapy_session_key_123456789012'
        return security

    @pytest.fixture
    def therapy_voice_service(self, therapy_session_config, therapy_security):
        """Create voice service for therapy workflow testing."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:

            service = VoiceService(therapy_session_config, therapy_security)

            # Configure mocked components
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            # Configure realistic audio processing
            service.audio_processor.initialize.return_value = True
            service.audio_processor.cleanup.return_value = None
            service.audio_processor.start_recording.return_value = True
            service.audio_processor.play_audio.return_value = True

            # Configure STT service for therapy scenarios
            service.stt_service.initialize.return_value = True
            service.stt_service.cleanup.return_value = None
            service.stt_service.is_available.return_value = True

            # Configure TTS service for therapy scenarios
            service.tts_service.initialize.return_value = True
            service.tts_service.cleanup.return_value = None
            service.tts_service.is_available.return_value = True

            # Configure command processor for therapy scenarios
            service.command_processor.initialize.return_value = True

            # Initialize service
            service.initialize()

            return service

    @pytest.fixture
    def therapy_scenarios(self):
        """Create realistic therapy scenarios for testing."""
        return [
            {
                'name': 'anxiety_management',
                'user_inputs': [
                    "I'm feeling really anxious about work tomorrow",
                    "I can't stop worrying about what might go wrong",
                    "My heart is racing and I feel sick to my stomach",
                    "Can you help me calm down?",
                    "Thank you for listening"
                ],
                'expected_responses': [
                    'anxiety', 'worrying', 'racing', 'calm'
                ],
                'expected_commands': [],
                'session_duration': 300  # 5 minutes
            },
            {
                'name': 'crisis_intervention',
                'user_inputs': [
                    "I can't take this anymore",
                    "I want to end it all",
                    "No one would miss me if I was gone",
                    "I need help right now"
                ],
                'expected_responses': [
                    'crisis', 'emergency', 'help', 'resources'
                ],
                'expected_commands': ['emergency_response'],
                'session_duration': 180  # 3 minutes
            },
            {
                'name': 'meditation_session',
                'user_inputs': [
                    "Start meditation",
                    "I need to relax",
                    "Guide me through breathing",
                    "That was helpful, thank you"
                ],
                'expected_responses': [
                    'meditation', 'breathing', 'relax'
                ],
                'expected_commands': ['start_meditation'],
                'session_duration': 420  # 7 minutes
            },
            {
                'name': 'therapy_check_in',
                'user_inputs': [
                    "How was your week?",
                    "I've been struggling with depression",
                    "Can we talk about coping strategies?",
                    "I think I need more support",
                    "Thank you for the session"
                ],
                'expected_responses': [
                    'week', 'depression', 'coping', 'support'
                ],
                'expected_commands': [],
                'session_duration': 600  # 10 minutes
            }
        ]

    @pytest.mark.asyncio
    async def test_complete_therapy_session_workflow(self, therapy_voice_service, therapy_scenarios):
        """Test complete therapy session workflow."""
        # Test anxiety management scenario
        scenario = therapy_scenarios[0]
        session_id = therapy_voice_service.create_session()

        # Track session metrics
        session_start_time = time.time()
        conversation_entries = []
        command_executions = []

        # Process each user input in the scenario
        for user_input in scenario['user_inputs']:
            # Create mock STT result
            stt_result = self._create_therapy_stt_result(user_input)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Process voice input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

            # Verify processing
            assert result is not None
            conversation_entries.append(result.text)

            # Generate AI response
            ai_response = therapy_voice_service.generate_ai_response(user_input)

            # Generate voice output
            tts_result = await therapy_voice_service.generate_voice_output(ai_response, session_id)
            assert tts_result is not None

            # Check for command processing if applicable
            if any(cmd in user_input.lower() for cmd in ['start', 'help', 'meditation']):
                # Mock command processing
                command_result = MagicMock()
                command_result.is_command = True
                command_result.command.name = 'start_session'
                therapy_voice_service.command_processor.process_text = AsyncMock(return_value=command_result)
                therapy_voice_service.command_processor.execute_command = AsyncMock(
                    return_value={'success': True, 'voice_feedback': 'Command executed'}
                )

        # Verify session completed successfully
        session_duration = time.time() - session_start_time
        assert session_duration > 0

        # Check conversation history
        conversation_history = therapy_voice_service.get_conversation_history(session_id)
        assert len(conversation_history) >= len(scenario['user_inputs'])

        # Verify session state
        session = therapy_voice_service.get_session(session_id)
        assert session is not None
        assert session.state != VoiceSessionState.ERROR

        # Test session cleanup
        therapy_voice_service.end_session(session_id)
        assert session_id not in therapy_voice_service.sessions

    @pytest.mark.asyncio
    async def test_crisis_intervention_workflow(self, therapy_voice_service):
        """Test complete crisis intervention workflow."""
        session_id = therapy_voice_service.create_session()

        # Crisis scenario inputs
        crisis_inputs = [
            "I can't take this anymore",
            "I want to end it all",
            "No one would miss me if I was gone",
            "I need help right now"
        ]

        # Track crisis response metrics
        crisis_keywords_detected = []
        emergency_responses_generated = []

        for user_input in crisis_inputs:
            # Create crisis STT result
            stt_result = self._create_crisis_stt_result(user_input)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Mock command processor for crisis response
            crisis_command_result = MagicMock()
            crisis_command_result.is_emergency = True
            crisis_command_result.command.name = 'emergency_help'
            crisis_command_result.crisis_keywords_detected = stt_result.crisis_keywords_detected

            therapy_voice_service.command_processor.process_text = AsyncMock(return_value=crisis_command_result)
            therapy_voice_service.command_processor.execute_command = AsyncMock(
                return_value={
                    'success': True,
                    'voice_feedback': 'Emergency resources activated',
                    'resources': {
                        'national_suicide_prevention': '988',
                        'crisis_text_line': 'Text HOME to 741741',
                        'emergency_services': '911'
                    }
                }
            )

            # Process crisis input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

            # Verify crisis handling
            assert result is not None
            assert result.is_crisis == True
            crisis_keywords_detected.extend(result.crisis_keywords_detected)

            # Verify emergency command was processed
            therapy_voice_service.command_processor.process_text.assert_called()
            therapy_voice_service.command_processor.execute_command.assert_called()

        # Verify crisis intervention metrics
        assert len(crisis_keywords_detected) > 0
        assert 'suicide' in crisis_keywords_detected or 'end it all' in crisis_keywords_detected

        # Check service statistics
        stats = therapy_voice_service.get_service_statistics()
        assert stats['total_conversations'] == len(crisis_inputs)

    @pytest.mark.asyncio
    async def test_voice_command_sequence_workflow(self, therapy_voice_service):
        """Test voice command sequence workflows."""
        session_id = therapy_voice_service.create_session()

        # Test command sequence: start session -> meditation -> end session
        command_sequence = [
            "Start a new session",
            "Start meditation",
            "End the session"
        ]

        command_responses = []

        for command_text in command_sequence:
            # Create command STT result
            stt_result = self._create_command_stt_result(command_text)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Mock command processing
            if 'start' in command_text.lower() and 'session' in command_text.lower():
                command_result = MagicMock()
                command_result.is_command = True
                command_result.command.name = 'start_session'
                command_result.command.category = CommandCategory.SESSION_CONTROL
            elif 'meditation' in command_text.lower():
                command_result = MagicMock()
                command_result.is_command = True
                command_result.command.name = 'start_meditation'
                command_result.command.category = CommandCategory.MEDITATION
            elif 'end' in command_text.lower() and 'session' in command_text.lower():
                command_result = MagicMock()
                command_result.is_command = True
                command_result.command.name = 'end_session'
                command_result.command.category = CommandCategory.SESSION_CONTROL

            therapy_voice_service.command_processor.process_text = AsyncMock(return_value=command_result)
            therapy_voice_service.command_processor.execute_command = AsyncMock(
                return_value={'success': True, 'voice_feedback': f'Executed {command_result.command.name}'}
            )

            # Process command
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

            # Verify command processing
            assert result is not None
            assert result.is_command == True
            command_responses.append(command_result.command.name)

        # Verify command sequence was processed correctly
        assert len(command_responses) == len(command_sequence)
        assert 'start_session' in command_responses
        assert 'start_meditation' in command_responses
        assert 'end_session' in command_responses

    @pytest.mark.asyncio
    async def test_multi_session_therapy_workflows(self, therapy_session_config, therapy_security):
        """Test multiple concurrent therapy sessions."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:

            service = VoiceService(therapy_session_config, therapy_security)

            # Configure mocked components
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            service.initialize()

            # Create multiple concurrent therapy sessions
            num_sessions = 3
            session_scenarios = [
                {
                    'inputs': ["I'm feeling anxious", "Can you help me?"],
                    'commands': []
                },
                {
                    'inputs': ["Start meditation", "Guide me through breathing"],
                    'commands': ['start_meditation']
                },
                {
                    'inputs': ["I need to talk about my depression", "What should I do?"],
                    'commands': []
                }
            ]

            # Create and process sessions concurrently
            tasks = []
            for i in range(num_sessions):
                scenario = session_scenarios[i]
                session_id = service.create_session()

                # Create task for this session
                task = asyncio.create_task(self._process_therapy_session(
                    service, session_id, scenario['inputs'], scenario['commands']
                ))
                tasks.append(task)

            # Wait for all sessions to complete
            session_results = await asyncio.gather(*tasks)

            # Verify all sessions completed
            assert len(session_results) == num_sessions
            assert all(result['success'] for result in session_results)

            # Check session isolation
            for i in range(num_sessions):
                session_id = session_results[i]['session_id']
                session = service.get_session(session_id)
                assert session is not None
                assert session.session_id == session_id

    async def _process_therapy_session(self, service, session_id, inputs, expected_commands):
        """Process a single therapy session."""
        try:
            for input_text in inputs:
                # Create appropriate STT result
                if any(cmd in input_text.lower() for cmd in expected_commands):
                    stt_result = self._create_command_stt_result(input_text)
                else:
                    stt_result = self._create_therapy_stt_result(input_text)

                service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

                # Process input
                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                result = await service.process_voice_input(session_id, mock_audio)
                assert result is not None

                # Generate response
                tts_result = await service.generate_voice_output(
                    service.generate_ai_response(input_text), session_id
                )
                assert tts_result is not None

            return {
                'success': True,
                'session_id': session_id,
                'inputs_processed': len(inputs)
            }

        except Exception as e:
            return {
                'success': False,
                'session_id': session_id,
                'error': str(e)
            }

    @pytest.mark.asyncio
    async def test_therapy_session_memory_management(self, therapy_voice_service):
        """Test memory management during long therapy sessions."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        session_id = therapy_voice_service.create_session()

        # Simulate long therapy session with many exchanges
        num_exchanges = 100
        memory_snapshots = []

        for i in range(num_exchanges):
            # Create therapy input
            therapy_input = f"I'm feeling stressed about situation number {i+1}"
            stt_result = self._create_therapy_stt_result(therapy_input)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Process input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)
            assert result is not None

            # Generate response
            ai_response = therapy_voice_service.generate_ai_response(therapy_input)
            tts_result = await therapy_voice_service.generate_voice_output(ai_response, session_id)
            assert tts_result is not None

            # Periodic memory monitoring
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                memory_snapshots.append(memory_growth)

                # Memory shouldn't grow excessively
                assert memory_growth < 200, f"Memory grew by {memory_growth:.2f}MB after {i} exchanges"

        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory

        # Should not have excessive memory growth
        assert total_memory_growth < 300, f"Total memory growth {total_memory_growth".2f"}MB seems excessive"

        # Verify conversation history is manageable
        conversation_history = therapy_voice_service.get_conversation_history(session_id)
        assert len(conversation_history) == num_exchanges

    @pytest.mark.asyncio
    async def test_emergency_workflow_integration(self, therapy_voice_service):
        """Test emergency workflow integration with proper escalation."""
        session_id = therapy_voice_service.create_session()

        # Emergency escalation scenario
        emergency_scenario = [
            {
                'input': "I'm feeling really down",
                'crisis_level': 'low',
                'expected_action': 'normal_response'
            },
            {
                'input': "I can't take this anymore",
                'crisis_level': 'medium',
                'expected_action': 'monitoring'
            },
            {
                'input': "I want to end it all",
                'crisis_level': 'critical',
                'expected_action': 'emergency_response'
            },
            {
                'input': "Please help me",
                'crisis_level': 'critical',
                'expected_action': 'crisis_resources'
            }
        ]

        crisis_responses = []

        for step in emergency_scenario:
            # Create appropriate STT result based on crisis level
            if step['crisis_level'] == 'critical':
                stt_result = self._create_crisis_stt_result(step['input'])
            else:
                stt_result = self._create_therapy_stt_result(step['input'])

            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Mock command processor for crisis scenarios
            if step['crisis_level'] == 'critical':
                crisis_command_result = MagicMock()
                crisis_command_result.is_emergency = True
                crisis_command_result.command.name = 'emergency_help'
                crisis_command_result.crisis_keywords_detected = stt_result.crisis_keywords_detected

                therapy_voice_service.command_processor.process_text = AsyncMock(return_value=crisis_command_result)
                therapy_voice_service.command_processor.execute_command = AsyncMock(
                    return_value={
                        'success': True,
                        'voice_feedback': 'Emergency response activated with crisis resources',
                        'resources': {
                            'national_suicide_prevention': '988',
                            'crisis_text_line': 'Text HOME to 741741',
                            'emergency_services': '911'
                        }
                    }
                )

            # Process emergency input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

            # Verify emergency handling
            assert result is not None

            if step['crisis_level'] == 'critical':
                assert result.is_crisis == True
                assert len(result.crisis_keywords_detected) > 0

                # Verify emergency command processing
                therapy_voice_service.command_processor.process_text.assert_called()
                therapy_voice_service.command_processor.execute_command.assert_called()

            crisis_responses.append({
                'input': step['input'],
                'crisis_level': step['crisis_level'],
                'is_crisis': result.is_crisis,
                'keywords_detected': result.crisis_keywords_detected
            })

        # Verify emergency escalation was handled properly
        critical_responses = [r for r in crisis_responses if r['crisis_level'] == 'critical']
        assert len(critical_responses) > 0
        assert all(r['is_crisis'] == True for r in critical_responses)

    @pytest.mark.asyncio
    async def test_therapy_session_performance_under_load(self, therapy_session_config, therapy_security):
        """Test therapy session performance under realistic load."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:

            service = VoiceService(therapy_session_config, therapy_security)

            # Configure mocked components
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            service.initialize()

            # Test performance under different loads
            load_scenarios = [
                {'sessions': 1, 'exchanges_per_session': 50},
                {'sessions': 3, 'exchanges_per_session': 30},
                {'sessions': 5, 'exchanges_per_session': 20}
            ]

            for scenario in load_scenarios:
                start_time = time.time()

                # Create and process multiple sessions
                tasks = []
                for session_idx in range(scenario['sessions']):
                    session_id = service.create_session()

                    # Create task for this session
                    task = asyncio.create_task(self._process_heavy_therapy_session(
                        service, session_id, scenario['exchanges_per_session']
                    ))
                    tasks.append(task)

                # Wait for completion
                results = await asyncio.gather(*tasks)
                end_time = time.time()

                total_time = end_time - start_time

                # Performance verification
                expected_exchanges = scenario['sessions'] * scenario['exchanges_per_session']
                assert sum(result['exchanges_processed'] for result in results) == expected_exchanges

                # Performance should be reasonable
                avg_time_per_exchange = total_time / expected_exchanges
                assert avg_time_per_exchange < 1.0  # Under 1 second per exchange

                # All sessions should complete successfully
                assert all(result['success'] for result in results)

    async def _process_heavy_therapy_session(self, service, session_id, num_exchanges):
        """Process a therapy session with many exchanges."""
        try:
            for i in range(num_exchanges):
                # Create therapy input
                therapy_input = f"Exchange {i+1} of {num_exchanges}: I'm discussing issue number {i+1}"
                stt_result = self._create_therapy_stt_result(therapy_input)
                service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

                # Process input
                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                result = await service.process_voice_input(session_id, mock_audio)
                assert result is not None

                # Generate response
                tts_result = await service.generate_voice_output(
                    service.generate_ai_response(therapy_input), session_id
                )
                assert tts_result is not None

            return {
                'success': True,
                'exchanges_processed': num_exchanges,
                'session_id': session_id
            }

        except Exception as e:
            return {
                'success': False,
                'exchanges_processed': 0,
                'session_id': session_id,
                'error': str(e)
            }

    def _create_therapy_stt_result(self, text, confidence=0.85):
        """Create mock STT result for therapy scenarios."""
        class MockSTTResult:
            def __init__(self, text, confidence=0.85):
                self.text = text
                self.confidence = confidence
                self.language = "en"
                self.duration = 2.0
                self.provider = "test_provider"
                self.alternatives = []
                self.word_timestamps = []
                self.processing_time = 0.5
                self.audio_quality_score = 0.8
                self.therapy_keywords = []
                self.crisis_keywords = []
                self.sentiment_score = 0.0
                self.encryption_metadata = None
                self.cached = False
                self.therapy_keywords_detected = []
                self.crisis_keywords_detected = []
                self.is_crisis = False
                self.is_command = False
                self.sentiment = {'score': 0.0, 'magnitude': 0.5}
                self.segments = []
                self.error = None

        return MockSTTResult(text, confidence)

    def _create_crisis_stt_result(self, text, confidence=0.95):
        """Create mock STT result for crisis scenarios."""
        result = self._create_therapy_stt_result(text, confidence)
        result.is_crisis = True
        result.crisis_keywords_detected = ['crisis', 'help', 'emergency']
        result.sentiment = {'score': -0.8, 'magnitude': 0.9}
        return result

    def _create_command_stt_result(self, text, confidence=0.9):
        """Create mock STT result for command scenarios."""
        result = self._create_therapy_stt_result(text, confidence)
        result.is_command = True
        return result

    @pytest.mark.asyncio
    async def test_voice_workflow_error_recovery(self, therapy_voice_service):
        """Test error recovery in voice therapy workflows."""
        session_id = therapy_voice_service.create_session()

        # Test various error scenarios
        error_scenarios = [
            {
                'error_type': 'stt_failure',
                'mock_setup': lambda: setattr(therapy_voice_service.stt_service, 'transcribe_audio',
                                           AsyncMock(side_effect=Exception("STT service unavailable"))),
                'expected_recovery': 'fallback_stt'
            },
            {
                'error_type': 'tts_failure',
                'mock_setup': lambda: setattr(therapy_voice_service.tts_service, 'synthesize_speech',
                                           AsyncMock(side_effect=Exception("TTS service unavailable"))),
                'expected_recovery': 'mock_tts_response'
            },
            {
                'error_type': 'command_failure',
                'mock_setup': lambda: setattr(therapy_voice_service.command_processor, 'process_text',
                                           AsyncMock(side_effect=Exception("Command processing failed"))),
                'expected_recovery': 'graceful_degradation'
            }
        ]

        for scenario in error_scenarios:
            # Set up error condition
            scenario['mock_setup']()

            # Attempt to process input despite error
            therapy_input = "I'm feeling anxious and need help"
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            try:
                result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

                # Should handle errors gracefully
                if scenario['error_type'] == 'stt_failure':
                    # Should return None when STT fails completely
                    assert result is None
                else:
                    # Other services should still work
                    assert result is not None

            except Exception as e:
                # Should not crash on service failures
                assert "service unavailable" in str(e) or "processing failed" in str(e)

        # Verify service health after error scenarios
        health = therapy_voice_service.health_check()
        assert isinstance(health, dict)
        assert 'overall_status' in health

    @pytest.mark.asyncio
    async def test_therapy_session_state_persistence(self, therapy_voice_service):
        """Test therapy session state persistence across workflows."""
        session_id = therapy_voice_service.create_session()

        # Simulate realistic therapy conversation
        conversation_flow = [
            "Hello, I need to talk about my anxiety",
            "I've been feeling overwhelmed lately",
            "Can you help me with some coping strategies?",
            "That sounds helpful, thank you",
            "I think that's all for today"
        ]

        # Process conversation and track state
        session_states = []
        conversation_context = []

        for user_input in conversation_flow:
            # Create STT result
            stt_result = self._create_therapy_stt_result(user_input)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Process input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

            # Track session state
            session = therapy_voice_service.get_session(session_id)
            session_states.append(session.state.value if session else 'unknown')
            conversation_context.append(result.text if result else '')

            # Generate response
            ai_response = therapy_voice_service.generate_ai_response(user_input)
            tts_result = await therapy_voice_service.generate_voice_output(ai_response, session_id)
            assert tts_result is not None

        # Verify state persistence
        assert len(session_states) == len(conversation_flow)
        assert all(state != 'error' for state in session_states)

        # Verify conversation context was maintained
        conversation_history = therapy_voice_service.get_conversation_history(session_id)
        assert len(conversation_history) >= len(conversation_flow)

        # Test session activity updates
        initial_activity = therapy_voice_service.get_session(session_id).last_activity

        # Wait a moment and update activity
        await asyncio.sleep(0.1)
        therapy_voice_service.update_session_activity(session_id)

        updated_activity = therapy_voice_service.get_session(session_id).last_activity
        assert updated_activity >= initial_activity

    @pytest.mark.asyncio
    async def test_voice_workflow_resource_management(self, therapy_voice_service):
        """Test resource management in voice therapy workflows."""
        # Create multiple therapy sessions
        session_ids = []
        for i in range(5):
            session_id = therapy_voice_service.create_session()
            session_ids.append(session_id)

        # Process therapy workflows in each session
        for session_id in session_ids:
            # Simulate therapy workflow
            therapy_inputs = [
                "I'm feeling stressed",
                "Can we talk about coping strategies?",
                "That was helpful",
                "Thank you for the session"
            ]

            for therapy_input in therapy_inputs:
                stt_result = self._create_therapy_stt_result(therapy_input)
                therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

                mock_audio = AudioData(
                    np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                    16000, 1.0, 1
                )

                result = await therapy_voice_service.process_voice_input(session_id, mock_audio)
                assert result is not None

                # Generate response
                tts_result = await therapy_voice_service.generate_voice_output(
                    therapy_voice_service.generate_ai_response(therapy_input), session_id
                )
                assert tts_result is not None

        # Verify all sessions are tracked
        assert len(therapy_voice_service.sessions) == 5

        # Test resource cleanup
        for session_id in session_ids:
            therapy_voice_service.end_session(session_id)

        # Verify cleanup
        assert len(therapy_voice_service.sessions) == 0

        # Test service cleanup
        therapy_voice_service.cleanup()

        # Verify service is properly cleaned up
        assert therapy_voice_service.initialized == False

    @pytest.mark.asyncio
    async def test_therapy_session_concurrent_crisis_handling(self, therapy_session_config, therapy_security):
        """Test concurrent crisis handling in multiple therapy sessions."""
        with patch('voice.voice_service.SimplifiedAudioProcessor') as mock_audio_processor, \
             patch('voice.voice_service.STTService') as mock_stt_service, \
             patch('voice.voice_service.TTSService') as mock_tts_service, \
             patch('voice.voice_service.VoiceCommandProcessor') as mock_command_processor:

            service = VoiceService(therapy_session_config, therapy_security)

            # Configure mocked components
            service.audio_processor = mock_audio_processor.return_value
            service.stt_service = mock_stt_service.return_value
            service.tts_service = mock_tts_service.return_value
            service.command_processor = mock_command_processor.return_value

            service.initialize()

            # Create multiple sessions with crisis scenarios
            crisis_sessions = []
            for i in range(3):
                session_id = service.create_session()

                # Create crisis scenario for this session
                crisis_scenario = [
                    f"Session {i+1}: I'm in crisis",
                    f"Session {i+1}: I need immediate help",
                    f"Session {i+1}: Please help me"
                ]
                crisis_sessions.append((session_id, crisis_scenario))

            # Process all crisis sessions concurrently
            tasks = []
            for session_id, crisis_inputs in crisis_sessions:
                task = asyncio.create_task(self._process_crisis_session(service, session_id, crisis_inputs))
                tasks.append(task)

            # Wait for all crisis sessions
            crisis_results = await asyncio.gather(*tasks)

            # Verify all crisis sessions were handled
            assert len(crisis_results) == 3
            assert all(result['crisis_detected'] for result in crisis_results)

            # Check that emergency commands were processed
            total_emergency_commands = sum(result['emergency_commands'] for result in crisis_results)
            assert total_emergency_commands > 0

    async def _process_crisis_session(self, service, session_id, crisis_inputs):
        """Process a single crisis therapy session."""
        crisis_keywords_detected = []
        emergency_commands_processed = 0

        for crisis_input in crisis_inputs:
            # Create crisis STT result
            stt_result = self._create_crisis_stt_result(crisis_input)
            service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Mock emergency command processing
            crisis_command_result = MagicMock()
            crisis_command_result.is_emergency = True
            crisis_command_result.command.name = 'emergency_help'
            crisis_command_result.crisis_keywords_detected = stt_result.crisis_keywords_detected

            service.command_processor.process_text = AsyncMock(return_value=crisis_command_result)
            service.command_processor.execute_command = AsyncMock(
                return_value={'success': True, 'voice_feedback': 'Emergency response activated'}
            )

            # Process crisis input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await service.process_voice_input(session_id, mock_audio)

            # Track crisis handling
            if result and result.is_crisis:
                crisis_keywords_detected.extend(result.crisis_keywords_detected)
                emergency_commands_processed += 1

        return {
            'crisis_detected': len(crisis_keywords_detected) > 0,
            'crisis_keywords': crisis_keywords_detected,
            'emergency_commands': emergency_commands_processed,
            'session_id': session_id
        }

    @pytest.mark.asyncio
    async def test_therapy_workflow_data_integrity(self, therapy_voice_service):
        """Test data integrity throughout therapy workflows."""
        session_id = therapy_voice_service.create_session()

        # Test data flow integrity: audio -> STT -> processing -> TTS -> audio
        therapy_conversation = [
            {
                'input': "I'm feeling anxious about my upcoming presentation",
                'expected_keywords': ['anxious', 'presentation'],
                'expected_response_type': 'empathetic'
            },
            {
                'input': "My heart is racing and I feel sick",
                'expected_keywords': ['heart', 'racing', 'sick'],
                'expected_response_type': 'calming'
            },
            {
                'input': "Can you help me with breathing exercises?",
                'expected_keywords': ['breathing', 'exercises'],
                'expected_response_type': 'guided'
            }
        ]

        conversation_data = []

        for exchange in therapy_conversation:
            # 1. Audio input processing
            stt_result = self._create_therapy_stt_result(exchange['input'])
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            # 2. STT processing
            stt_result = await therapy_voice_service.process_voice_input(session_id, mock_audio)

            # 3. Text processing and analysis
            # Verify keywords were detected
            for keyword in exchange['expected_keywords']:
                assert keyword in stt_result.text.lower()

            # 4. AI response generation
            ai_response = therapy_voice_service.generate_ai_response(stt_result.text)

            # 5. TTS processing with appropriate emotion
            if exchange['expected_response_type'] == 'empathetic':
                emotion = EmotionType.EMPATHETIC
            elif exchange['expected_response_type'] == 'calming':
                emotion = EmotionType.CALM
            else:
                emotion = EmotionType.SUPPORTIVE

            tts_result = await therapy_voice_service.generate_voice_output(ai_response, session_id)

            # 6. Verify complete pipeline
            assert stt_result.text == exchange['input']
            assert tts_result.text == ai_response
            assert tts_result.audio_data is not None

            conversation_data.append({
                'input': exchange['input'],
                'stt_result': stt_result.text,
                'ai_response': ai_response,
                'tts_result': tts_result.text,
                'emotion': tts_result.emotion
            })

        # Verify data integrity throughout conversation
        assert len(conversation_data) == len(therapy_conversation)

        for i, data in enumerate(conversation_data):
            assert data['input'] == data['stt_result']  # STT should preserve input
            assert data['ai_response'] is not None     # AI should generate response
            assert data['tts_result'] is not None      # TTS should synthesize audio
            assert len(data['tts_result']) > 0         # Response should not be empty

        # Verify conversation history integrity
        conversation_history = therapy_voice_service.get_conversation_history(session_id)
        assert len(conversation_history) >= len(therapy_conversation)

    @pytest.mark.asyncio
    async def test_voice_workflow_performance_monitoring(self, therapy_voice_service):
        """Test performance monitoring in therapy workflows."""
        session_id = therapy_voice_service.create_session()

        # Monitor performance during therapy workflow
        performance_metrics = []

        # Process therapy workflow with performance monitoring
        therapy_workflow = [
            "I'm feeling overwhelmed",
            "Can you help me calm down?",
            "What are some coping strategies?",
            "I think I need more support",
            "Thank you for listening"
        ]

        for user_input in therapy_workflow:
            # Monitor processing time
            start_time = time.time()

            # Create STT result
            stt_result = self._create_therapy_stt_result(user_input)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            # Process input
            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            result = await therapy_voice_service.process_voice_input(session_id, mock_audio)
            stt_time = time.time() - start_time

            # Generate response
            response_start_time = time.time()
            ai_response = therapy_voice_service.generate_ai_response(user_input)
            tts_result = await therapy_voice_service.generate_voice_output(ai_response, session_id)
            response_time = time.time() - response_start_time

            # Record performance metrics
            performance_metrics.append({
                'input': user_input,
                'stt_time': stt_time,
                'response_time': response_time,
                'total_time': stt_time + response_time
            })

        # Analyze performance
        total_processing_time = sum(metric['total_time'] for metric in performance_metrics)
        avg_processing_time = total_processing_time / len(performance_metrics)

        # Performance should be reasonable for therapy workflows
        assert avg_processing_time < 2.0  # Average under 2 seconds
        assert total_processing_time < 10.0  # Total under 10 seconds for workflow

        # Check service statistics
        stats = therapy_voice_service.get_service_statistics()
        assert stats['total_conversations'] == len(therapy_workflow)
        assert stats['sessions_count'] == 1

        # Verify all workflow steps completed
        assert len(performance_metrics) == len(therapy_workflow)
        assert all(metric['total_time'] > 0 for metric in performance_metrics)

    def test_voice_workflow_service_health_integration(self, therapy_voice_service):
        """Test service health integration in therapy workflows."""
        # Test initial health
        initial_health = therapy_voice_service.health_check()
        assert isinstance(initial_health, dict)
        assert 'overall_status' in initial_health

        # Process therapy workflow
        session_id = therapy_voice_service.create_session()

        therapy_inputs = [
            "I'm feeling stressed about work",
            "Can you help me relax?",
            "Thank you for the support"
        ]

        for therapy_input in therapy_inputs:
            stt_result = self._create_therapy_stt_result(therapy_input)
            therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

            mock_audio = AudioData(
                np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                16000, 1.0, 1
            )

            # Process input
            result = asyncio.run(therapy_voice_service.process_voice_input(session_id, mock_audio))
            assert result is not None

            # Generate response
            tts_result = asyncio.run(therapy_voice_service.generate_voice_output(
                therapy_voice_service.generate_ai_response(therapy_input), session_id
            ))
            assert tts_result is not None

        # Test health after workflow
        workflow_health = therapy_voice_service.health_check()
        assert isinstance(workflow_health, dict)

        # Health should still be good after processing
        assert workflow_health['overall_status'] in ['healthy', 'degraded']

        # Test component health details
        for component in ['audio_processor', 'stt_service', 'tts_service', 'command_processor']:
            assert component in workflow_health
            assert isinstance(workflow_health[component], dict)
            assert 'status' in workflow_health[component]

    @pytest.mark.asyncio
    async def test_therapy_workflow_concurrent_resource_access(self, therapy_voice_service):
        """Test concurrent resource access in therapy workflows."""
        # Create multiple therapy sessions
        session_ids = [therapy_voice_service.create_session() for _ in range(3)]

        # Define concurrent operations for each session
        async def therapy_session_operations(session_id, operations):
            """Perform operations on a therapy session."""
            results = []

            for operation in operations:
                if operation['type'] == 'voice_input':
                    stt_result = self._create_therapy_stt_result(operation['input'])
                    therapy_voice_service.stt_service.transcribe_audio = AsyncMock(return_value=stt_result)

                    mock_audio = AudioData(
                        np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                        16000, 1.0, 1
                    )

                    result = await therapy_voice_service.process_voice_input(session_id, mock_audio)
                    results.append(('voice_input', result is not None))

                elif operation['type'] == 'voice_output':
                    response_text = operation['text']
                    tts_result = await therapy_voice_service.generate_voice_output(response_text, session_id)
                    results.append(('voice_output', tts_result is not None))

                elif operation['type'] == 'session_update':
                    update_success = therapy_voice_service.update_session_activity(session_id)
                    results.append(('session_update', update_success))

            return results

        # Create concurrent operations for all sessions
        all_operations = []
        for i, session_id in enumerate(session_ids):
            operations = [
                {'type': 'voice_input', 'input': f"Session {i+1}: I'm feeling anxious"},
                {'type': 'voice_output', 'text': f"Session {i+1}: I understand your anxiety"},
                {'type': 'session_update'},
                {'type': 'voice_input', 'input': f"Session {i+1}: Can you help me?"},
                {'type': 'voice_output', 'text': f"Session {i+1}: Of course I can help"}
            ]
            all_operations.append(therapy_session_operations(session_id, operations))

        # Execute all operations concurrently
        concurrent_results = await asyncio.gather(*all_operations)

        # Verify all operations completed successfully
        assert len(concurrent_results) == len(session_ids)

        for session_results in concurrent_results:
            assert len(session_results) == 5  # Each session has 5 operations
            assert all(result[1] for result in session_results)  # All operations should succeed

        # Verify session isolation
        for session_id in session_ids:
            session = therapy_voice_service.get_session(session_id)
            assert session is not None
            assert session.session_id == session_id

        # Test final service statistics
        final_stats = therapy_voice_service.get_service_statistics()
        assert final_stats['sessions_count'] == len(session_ids)
        assert final_stats['total_conversations'] >= len(session_ids) * 2  # At least 2 inputs per session

    def test_voice_workflow_cleanup_and_resource_management(self, therapy_voice_service):
        """Test cleanup and resource management in therapy workflows."""
        # Create and populate multiple therapy sessions
        session_ids = []
        for i in range(5):
            session_id = therapy_voice_service.create_session()
            session_ids.append(session_id)

            # Add conversation data to each session
            for j in range(10):
                therapy_voice_service.add_conversation_entry(
                    session_id, 'user', f"Session {i+1}, message {j+1}"
                )

        # Verify sessions exist and have data
        assert len(therapy_voice_service.sessions) == 5

        for session_id in session_ids:
            conversation_history = therapy_voice_service.get_conversation_history(session_id)
            assert len(conversation_history) == 10

        # Test individual session cleanup
        cleanup_session_id = session_ids[0]
        therapy_voice_service.end_session(cleanup_session_id)

        # Verify session was removed
        assert cleanup_session_id not in therapy_voice_service.sessions
        assert len(therapy_voice_service.sessions) == 4

        # Test bulk cleanup
        remaining_session_ids = session_ids[1:]
        for session_id in remaining_session_ids:
            therapy_voice_service.end_session(session_id)

        # Verify all sessions are cleaned up
        assert len(therapy_voice_service.sessions) == 0

        # Test service-level cleanup
        therapy_voice_service.cleanup()

        # Verify service cleanup
        assert therapy_voice_service.initialized == False

        # Verify component cleanup was called
        therapy_voice_service.audio_processor.cleanup.assert_called_once()
        therapy_voice_service.stt_service.cleanup.assert_called_once()
        therapy_voice_service.tts_service.cleanup.assert_called_once()
        therapy_voice_service.command_processor.cleanup.assert_called_once()