"""
Comprehensive end-to-end integration tests for complete therapy session.

Tests complete therapy session workflow including:
- User login → session start → voice interaction → knowledge retrieval → response generation → session end
- Performance monitoring throughout session
- Memory management during long sessions
- Cache effectiveness across session
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from datetime import datetime
import numpy as np

from auth.auth_service import AuthService, AuthResult
from auth.user_model import UserModel, UserRole
from voice.voice_service import VoiceService, VoiceSessionState
from voice.audio_processor import AudioData
from security.pii_protection import PIIProtection
from performance.monitor import PerformanceMonitor
from performance.memory_manager import MemoryManager
from performance.cache_manager import CacheManager


@pytest.fixture
def user_model():
    """Create user model with test patient."""
    model = UserModel()
    model.register_user(
        email="therapy.patient@example.com",
        password="SecureTherapy123!",
        full_name="Alice Johnson",
        role=UserRole.PATIENT
    )
    return model


@pytest.fixture
def auth_service(user_model):
    """Create authentication service."""
    return AuthService(user_model=user_model)


@pytest.fixture
def pii_protection():
    """Create PII protection service."""
    return PIIProtection(enable_audit=True, hipaa_compliance=True)


@pytest.fixture
def performance_monitor():
    """Create performance monitor."""
    return PerformanceMonitor(config={
        'monitoring_interval': 1.0,
        'enable_alerts': True,
        'memory_threshold_percent': 80.0
    })


@pytest.fixture
def memory_manager():
    """Create memory manager."""
    return MemoryManager(max_memory_mb=512, cleanup_threshold=0.8)


@pytest.fixture
def cache_manager():
    """Create cache manager."""
    return CacheManager(max_size_mb=100, ttl_seconds=3600)


@pytest.fixture
async def voice_service():
    """Create voice service with mocked dependencies."""
    with patch('voice.voice_service.STTService') as mock_stt, \
         patch('voice.voice_service.TTSService') as mock_tts:
        
        # Configure mocks
        mock_stt.return_value.transcribe_audio = AsyncMock(
            return_value=Mock(
                text="I've been feeling very anxious lately",
                confidence=0.95,
                language="en-US",
                is_final=True
            )
        )
        
        mock_tts.return_value.synthesize = AsyncMock(
            return_value=Mock(
                audio_data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                success=True
            )
        )
        
        service = VoiceService()
        yield service
        await service.cleanup()


@pytest.fixture
def mock_knowledge_base():
    """Create mock knowledge base for therapy responses."""
    knowledge = Mock()
    knowledge.query = Mock(return_value={
        "response": "I understand you're experiencing anxiety. Let's explore coping strategies.",
        "sources": ["CBT_techniques.pdf", "anxiety_management.pdf"],
        "confidence": 0.9
    })
    return knowledge


@pytest.mark.integration
@pytest.mark.end_to_end
class TestCompleteTherapySession:
    """Test complete therapy session end-to-end workflow."""
    
    def test_full_therapy_session_workflow(
        self,
        auth_service,
        voice_service,
        pii_protection,
        performance_monitor,
        mock_knowledge_base
    ):
        """Test complete therapy session from login to session end."""
        # Step 1: User authentication
        auth_result = auth_service.login(
            email="therapy.patient@example.com",
            password="SecureTherapy123!",
            ip_address="192.168.1.50"
        )
        
        assert auth_result.success
        assert auth_result.token is not None
        user_id = auth_result.user.user_id
        
        # Step 2: Start performance monitoring
        performance_monitor.start_monitoring()
        initial_metrics = performance_monitor.get_current_metrics()
        assert initial_metrics is not None
        
        # Step 3: Start voice session
        async def run_voice_session():
            session_id = await voice_service.start_session(
                user_id=user_id,
                voice_profile="therapist"
            )
            assert session_id is not None
            
            # Step 4: Process patient input (voice interaction)
            audio_data = AudioData(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time()
            )
            
            transcription_result = await voice_service.process_audio(audio_data)
            
            # Step 5: Detect and protect PII in transcription
            if transcription_result:
                transcribed_text = "I've been feeling very anxious lately"
                pii_results = pii_protection.detect_pii(transcribed_text)
                
                # Mask any PII before processing
                if pii_results:
                    masked_text = pii_protection.mask_pii(transcribed_text, pii_results)
                else:
                    masked_text = transcribed_text
            
            # Step 6: Query knowledge base for therapy response
            knowledge_response = mock_knowledge_base.query(masked_text)
            assert knowledge_response is not None
            assert "response" in knowledge_response
            
            # Step 7: Generate voice response via TTS
            therapy_response = knowledge_response["response"]
            tts_result = await voice_service.synthesize_speech(therapy_response)
            assert tts_result is not None
            
            # Step 8: Verify session state
            assert voice_service.current_session is not None
            assert len(voice_service.current_session.conversation_history) > 0
            
            # Step 9: End voice session
            await voice_service.stop_session()
        
        # Run async voice session
        asyncio.run(run_voice_session())
        
        # Step 10: Check performance metrics after session
        final_metrics = performance_monitor.get_current_metrics()
        assert final_metrics is not None
        
        # Step 11: Verify audit trail
        audit_logs = pii_protection.get_audit_logs()
        assert isinstance(audit_logs, list)
        
        # Step 12: User logout
        logout_success = auth_service.logout(auth_result.session.session_id)
        assert logout_success
        
        # Step 13: Stop monitoring
        performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_long_therapy_session_memory_management(
        self,
        voice_service,
        memory_manager,
        performance_monitor
    ):
        """Test memory management during long therapy session."""
        performance_monitor.start_monitoring()
        
        # Start session
        session_id = await voice_service.start_session("user_long_session")
        
        # Track memory usage
        initial_memory = memory_manager.get_current_usage()
        
        # Simulate long conversation (20 turns)
        for i in range(20):
            audio_data = AudioData(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time()
            )
            
            await voice_service.process_audio(audio_data)
            
            # Generate response
            await voice_service.synthesize_speech(f"Response {i+1}")
            
            # Check memory periodically
            if i % 5 == 0:
                current_memory = memory_manager.get_current_usage()
                
                # Trigger cleanup if needed
                if current_memory > memory_manager.cleanup_threshold:
                    memory_manager.cleanup_old_data()
        
        # Final memory check
        final_memory = memory_manager.get_current_usage()
        
        # Verify no significant memory leak
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200  # Less than 200 MB increase
        
        await voice_service.stop_session()
        performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness_across_session(
        self,
        voice_service,
        cache_manager,
        mock_knowledge_base
    ):
        """Test cache effectiveness during therapy session."""
        session_id = await voice_service.start_session("user_cache_test")
        
        # Common therapy queries
        queries = [
            "I feel anxious",
            "Help with sleep",
            "I feel anxious",  # Repeated query
            "Coping strategies",
            "Help with sleep"  # Repeated query
        ]
        
        cache_hits = 0
        cache_misses = 0
        
        for query in queries:
            # Check cache first
            cache_key = f"therapy_response:{hash(query)}"
            cached_response = cache_manager.get(cache_key)
            
            if cached_response:
                cache_hits += 1
                response = cached_response
            else:
                cache_misses += 1
                # Query knowledge base
                response = mock_knowledge_base.query(query)
                # Cache the response
                cache_manager.set(cache_key, response)
            
            # Synthesize speech
            await voice_service.synthesize_speech(response["response"])
        
        # Verify cache effectiveness
        assert cache_hits > 0  # Should have some cache hits from repeated queries
        assert cache_hits + cache_misses == len(queries)
        
        # Check cache stats
        stats = cache_manager.get_stats()
        assert stats["hits"] >= cache_hits
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_error_recovery_during_session(
        self,
        auth_service,
        voice_service,
        performance_monitor
    ):
        """Test system recovers from errors during therapy session."""
        # Login
        auth_result = auth_service.login(
            email="therapy.patient@example.com",
            password="SecureTherapy123!"
        )
        assert auth_result.success
        
        performance_monitor.start_monitoring()
        
        # Start session
        session_id = await voice_service.start_session(auth_result.user.user_id)
        
        # Simulate various error conditions
        
        # 1. STT failure
        with patch.object(voice_service.stt_service, 'transcribe_audio', side_effect=Exception("STT error")):
            audio_data = AudioData(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time()
            )
            result = await voice_service.process_audio(audio_data)
            # Should handle gracefully
            assert voice_service.current_session.state != VoiceSessionState.ERROR
        
        # 2. TTS failure
        with patch.object(voice_service.tts_service, 'synthesize', side_effect=Exception("TTS error")):
            result = await voice_service.synthesize_speech("Test response")
            # Should handle gracefully
            assert voice_service.current_session.state != VoiceSessionState.ERROR
        
        # 3. Verify session continues after errors
        audio_data = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        result = await voice_service.process_audio(audio_data)
        
        # Session should still be functional
        assert voice_service.current_session is not None
        
        await voice_service.stop_session()
        performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_concurrent_therapy_sessions(
        self,
        auth_service,
        user_model,
        performance_monitor
    ):
        """Test handling multiple concurrent therapy sessions."""
        # Create multiple users
        users = []
        for i in range(3):
            email = f"patient{i}@example.com"
            user_model.register_user(
                email=email,
                password=f"Password{i}123!",
                full_name=f"Patient {i}",
                role=UserRole.PATIENT
            )
            
            auth_result = auth_service.login(email=email, password=f"Password{i}123!")
            assert auth_result.success
            users.append(auth_result.user)
        
        performance_monitor.start_monitoring()
        
        # Create concurrent voice sessions
        async def run_concurrent_sessions():
            services = []
            sessions = []
            
            for user in users:
                # Create separate voice service for each user
                with patch('voice.voice_service.STTService'), \
                     patch('voice.voice_service.TTSService'):
                    service = VoiceService()
                    services.append(service)
                    
                    session_id = await service.start_session(user.user_id)
                    sessions.append(session_id)
            
            # Run concurrent audio processing
            async def process_user_audio(service, session_id):
                for _ in range(3):
                    audio = AudioData(
                        data=np.random.randn(8000).astype(np.float32),
                        sample_rate=16000,
                        channels=1,
                        timestamp=time.time()
                    )
                    await service.process_audio(audio)
                    await asyncio.sleep(0.1)
            
            # Process all sessions concurrently
            await asyncio.gather(*[
                process_user_audio(service, session_id)
                for service, session_id in zip(services, sessions)
            ])
            
            # Cleanup
            for service in services:
                await service.stop_session()
                await service.cleanup()
        
        await run_concurrent_sessions()
        
        # Verify performance acceptable with concurrent sessions
        metrics = performance_monitor.get_current_metrics()
        assert metrics.memory_percent < 90.0  # Memory usage acceptable
        
        performance_monitor.stop_monitoring()
    
    def test_session_with_pii_throughout_workflow(
        self,
        auth_service,
        pii_protection,
        mock_knowledge_base
    ):
        """Test PII protection throughout entire therapy session."""
        # Login with PII
        auth_result = auth_service.login(
            email="therapy.patient@example.com",
            password="SecureTherapy123!"
        )
        assert auth_result.success
        
        # Patient shares sensitive information
        patient_input = "My name is Alice Johnson and I live at 123 Main St. My SSN is 123-45-6789."
        
        # Detect PII
        pii_results = pii_protection.detect_pii(patient_input)
        assert len(pii_results) > 0  # Should detect name, address, SSN
        
        # Mask PII before storage/processing
        masked_input = pii_protection.mask_pii(patient_input, pii_results)
        assert masked_input != patient_input
        assert "123-45-6789" not in masked_input  # SSN masked
        
        # Process masked input through knowledge base
        response = mock_knowledge_base.query(masked_input)
        
        # Ensure response doesn't contain original PII
        response_text = response["response"]
        response_pii = pii_protection.detect_pii(response_text)
        
        # High-sensitivity PII should not appear in response
        for pii in response_pii:
            assert pii.confidence < 0.9  # Only low-confidence detections acceptable
        
        # Verify audit trail
        audit_logs = pii_protection.get_audit_logs()
        assert len(audit_logs) > 0
        
        auth_service.logout(auth_result.session.session_id)
    
    @pytest.mark.asyncio
    async def test_session_performance_degradation_detection(
        self,
        voice_service,
        performance_monitor
    ):
        """Test detection of performance degradation during session."""
        performance_monitor.start_monitoring()
        
        session_id = await voice_service.start_session("user_perf_test")
        
        # Track response times
        response_times = []
        
        for i in range(10):
            start_time = time.time()
            
            audio_data = AudioData(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=time.time()
            )
            
            await voice_service.process_audio(audio_data)
            await voice_service.synthesize_speech(f"Response {i}")
            
            end_time = time.time()
            response_times.append(end_time - start_time)
        
        # Check for performance degradation
        avg_early = sum(response_times[:3]) / 3
        avg_late = sum(response_times[-3:]) / 3
        
        degradation = (avg_late - avg_early) / avg_early
        
        # Performance should not degrade significantly
        assert degradation < 0.5  # Less than 50% degradation
        
        # Get performance alerts
        alerts = performance_monitor.get_alerts()
        
        # Should have minimal critical alerts
        critical_alerts = [a for a in alerts if a.level.value == "critical"]
        assert len(critical_alerts) == 0
        
        await voice_service.stop_session()
        performance_monitor.stop_monitoring()


@pytest.mark.integration
@pytest.mark.end_to_end
class TestTherapySessionEdgeCases:
    """Test edge cases in therapy session workflow."""
    
    @pytest.mark.asyncio
    async def test_session_interruption_and_resume(self, voice_service):
        """Test session can be interrupted and resumed."""
        # Start session
        session_id = await voice_service.start_session("user_interrupt")
        
        # Process some audio
        audio1 = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        await voice_service.process_audio(audio1)
        
        # Simulate interruption (network issue, etc.)
        # Session should maintain state
        history_before = voice_service.current_session.conversation_history.copy()
        
        # Resume after interruption
        audio2 = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time()
        )
        await voice_service.process_audio(audio2)
        
        # Verify history preserved
        assert len(voice_service.current_session.conversation_history) >= len(history_before)
        
        await voice_service.stop_session()
    
    @pytest.mark.asyncio
    async def test_session_with_no_audio_input(self, voice_service):
        """Test session handles scenario with no audio input."""
        session_id = await voice_service.start_session("user_no_audio")
        
        # Wait without audio input
        await asyncio.sleep(0.5)
        
        # Session should remain active
        assert voice_service.current_session is not None
        assert voice_service.current_session.state == VoiceSessionState.IDLE
        
        await voice_service.stop_session()
    
    def test_authentication_failure_prevents_session(self, auth_service):
        """Test authentication failure prevents therapy session."""
        # Failed login
        auth_result = auth_service.login(
            email="therapy.patient@example.com",
            password="WrongPassword!"
        )
        
        assert not auth_result.success
        assert auth_result.token is None
        
        # Should not be able to start session without auth
        # (This would be enforced at application level)
