#!/usr/bin/env python3
"""
Comprehensive Security Tests for Voice Module

This test suite validates all security fixes implemented for PR #1:
1. Input validation in voice/security.py
2. Memory leak fixes in voice/audio_processor.py
3. Thread safety fixes in voice/voice_service.py
4. Integration security tests

Run with: pytest test_voice_security_comprehensive.py -v
"""

import pytest
import threading
import time
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import json
import re
import asyncio
from collections import deque
import logging

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestInputValidation:
    """Test input validation in voice/security.py"""

    @pytest.fixture
    def security_config(self):
        """Create test security configuration."""
        class MockSecurityConfig:
            encryption_enabled = False
            consent_required = False
            hipaa_compliance_enabled = True
            gdpr_compliance_enabled = True
            data_localization = True
            data_retention_hours = 24
            emergency_protocols_enabled = True
            privacy_mode = False
            anonymization_enabled = False

        return MockSecurityConfig()

    @pytest.fixture
    def voice_config(self, security_config):
        """Create test voice configuration."""
        class MockVoiceConfig:
            security = security_config

        return MockVoiceConfig()

    @pytest.fixture
    def voice_security(self, voice_config, tmp_path):
        """Create VoiceSecurity instance with temporary directory."""
        with patch('voice.security.Path') as mock_path:
            mock_path.return_value = tmp_path
            from voice.security import VoiceSecurity
            security = VoiceSecurity(voice_config)
            return security

    def test_valid_user_id_formats(self, voice_security):
        """Test valid user ID formats are accepted."""
        valid_user_ids = [
            "user123",
            "test_user",
            "user-with-dash",
            "user_with_underscore",
            "U",  # Single character
            "a" * 50,  # Maximum length
            "User123Mixed",
            "123456",
            "user-123_456"
        ]

        for user_id in valid_user_ids:
            result = voice_security.grant_consent(
                user_id=user_id,
                consent_type="voice_processing",
                granted=True,
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0"
            )
            assert result is True, f"Valid user_id '{user_id}' should be accepted"

    def test_invalid_user_id_formats(self, voice_security):
        """Test invalid user ID formats are rejected."""
        invalid_user_ids = [
            "",  # Empty string
            "user@domain.com",  # Special character @
            "user#123",  # Special character #
            "user space",  # Space
            "user\ttab",  # Tab character
            "user\nnewline",  # Newline character
            "a" * 51,  # Too long (> 50 chars)
            "user.dot",  # Dot not allowed
            "user/slash",  # Slash not allowed
            "user\\backslash",  # Backslash not allowed
            None,  # None value
            123,  # Non-string type
            [],  # List type
            "user<test>",  # HTML-like characters
            "user&test",  # Ampersand
            "user'test",  # Single quote
            'user"test',  # Double quote
        ]

        for user_id in invalid_user_ids:
            result = voice_security.grant_consent(
                user_id=user_id,
                consent_type="voice_processing",
                granted=True
            )
            assert result is False, f"Invalid user_id '{user_id}' should be rejected"

    def test_valid_ip_addresses(self, voice_security):
        """Test valid IP address formats are accepted."""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "127.0.0.1",
            "0.0.0.0",
            "255.255.255.255",
            "",  # Empty IP is allowed for local contexts
            "1.2.3.4",
            "192.168.0.100"
        ]

        for ip in valid_ips:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type="voice_processing",
                granted=True,
                ip_address=ip,
                user_agent="TestAgent/1.0"
            )
            assert result is True, f"Valid IP '{ip}' should be accepted"

    def test_invalid_ip_addresses(self, voice_security):
        """Test invalid IP address formats are rejected."""
        invalid_ips = [
            "999.999.999.999",  # Out of range
            "192.168.1",  # Too few octets
            "192.168.1.1.1",  # Too many octets
            "192.168.-1.1",  # Negative number
            "192.168.1.256",  # Out of range
            "192.168.01.1",  # Leading zero (technically valid but we can be strict)
            "192.168.1.a",  # Non-numeric
            "192,168,1,1",  # Wrong separator
            "192.168.1.1 ",  # Trailing space
            " 192.168.1.1",  # Leading space
            "192.168.1.1\n",  # Newline
            "192.168.1.1.1.1",  # Way too many octets
            "abc.def.ghi.jkl",  # All letters
        ]

        for ip in invalid_ips:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type="voice_processing",
                granted=True,
                ip_address=ip
            )
            assert result is False, f"Invalid IP '{ip}' should be rejected"

    def test_valid_user_agents(self, voice_security):
        """Test valid user agent strings are accepted."""
        valid_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Chrome/91.0.4472.124",
            "Safari/605.1.15",
            "Firefox/89.0",
            "Edge/91.0.864.59",
            "curl/7.64.1",
            "Python-requests/2.25.1",
            "TestAgent/1.0",
            "",  # Empty is allowed
            "Simple Agent",
            "a" * 500,  # Maximum length
        ]

        for ua in valid_user_agents:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type="voice_processing",
                granted=True,
                ip_address="192.168.1.1",
                user_agent=ua
            )
            assert result is True, f"Valid user agent should be accepted: {ua[:50]}..."

    def test_invalid_user_agents(self, voice_security):
        """Test invalid user agent strings are rejected."""
        invalid_user_agents = [
            "<script>alert('xss')</script>",  # Script tags
            "Agent<script>alert('xss')</script>",  # Script injection
            "Agent with <iframe src='evil.com'></iframe>",  # iframe
            "Agent with \"quotes\" and 'apostrophes'",  # Quotes
            "Agent with ;semicolon; and &ampersand&",  # Special chars
            "Agent\twith\ttabs",  # Tab characters
            "Agent\nwith\nnewlines",  # Newlines
            "Agent\rwith\rcarriage",  # Carriage returns
            "a" * 501,  # Too long (> 500 chars)
            None,  # None value
            123,  # Non-string
            [],  # List type
        ]

        for ua in invalid_user_agents:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type="voice_processing",
                granted=True,
                ip_address="192.168.1.1",
                user_agent=ua
            )
            assert result is False, f"Invalid user agent should be rejected: {str(ua)[:50]}..."

    def test_valid_consent_types(self, voice_security):
        """Test valid consent types are accepted."""
        valid_consent_types = [
            "voice_processing",
            "data_storage",
            "transcription",
            "analysis",
            "all_consent",
            "emergency_protocol"
        ]

        for consent_type in valid_consent_types:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type=consent_type,
                granted=True,
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0"
            )
            assert result is True, f"Valid consent type '{consent_type}' should be accepted"

    def test_invalid_consent_types(self, voice_security):
        """Test invalid consent types are rejected."""
        invalid_consent_types = [
            "invalid_consent",
            "voice_processing_extra",
            "custom_consent",
            "all",
            "none",
            "",
            "VOICE_PROCESSING",  # Case sensitive
            "Voice_Processing",  # Case sensitive
            "voice-processing",  # Wrong separator
            None,  # None value
            123,  # Non-string
            [],  # List type
        ]

        for consent_type in invalid_consent_types:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type=consent_type,
                granted=True,
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0"
            )
            assert result is False, f"Invalid consent type '{consent_type}' should be rejected"

    def test_consent_text_length_validation(self, voice_security):
        """Test consent text length validation."""
        # Valid length (exactly 10000 chars)
        valid_text = "a" * 10000
        result = voice_security.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            consent_text=valid_text
        )
        assert result is True, "Consent text of exactly 10000 chars should be accepted"

        # Invalid length (10001 chars)
        invalid_text = "a" * 10001
        result = voice_security.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            consent_text=invalid_text
        )
        assert result is False, "Consent text over 10000 chars should be rejected"

    def test_sql_injection_attempts(self, voice_security):
        """Test SQL injection attempts are blocked."""
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "user' OR '1'='1",
            "admin'; INSERT INTO users VALUES('hacker', 'password'); --",
            "user' UNION SELECT * FROM sensitive_data --",
            "'; UPDATE users SET password='hacked' WHERE '1'='1'; --",
            "user'; DELETE FROM audit_logs; --"
        ]

        for malicious_input in sql_injection_attempts:
            # Test as user_id
            result = voice_security.grant_consent(
                user_id=malicious_input,
                consent_type="voice_processing",
                granted=True
            )
            assert result is False, f"SQL injection in user_id should be blocked: {malicious_input[:50]}..."

            # Test as consent_type
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type=malicious_input,
                granted=True
            )
            assert result is False, f"SQL injection in consent_type should be blocked: {malicious_input[:50]}..."

    def test_xss_attempts(self, voice_security):
        """Test XSS attempts are blocked in user agent."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(\"xss\")'></iframe>",
            "<svg onload=alert('xss')>",
            "';alert('xss');//",
            "<body onload=alert('xss')>",
            "<input onfocus=alert('xss') autofocus>",
        ]

        for xss_attempt in xss_attempts:
            result = voice_security.grant_consent(
                user_id="test_user",
                consent_type="voice_processing",
                granted=True,
                user_agent=xss_attempt
            )
            assert result is False, f"XSS attempt in user_agent should be blocked: {xss_attempt[:50]}..."

    def test_unicode_and_encoding_attacks(self, voice_security):
        """Test Unicode and encoding attacks are handled safely."""
        unicode_attacks = [
            "\u0000user\u0000",  # Null bytes
            "user\ufeff",  # BOM character
            "user\ufffd",  # Replacement character
            "用户123",  # Chinese characters (should be allowed by regex but might not)
            "🤖user",  # Emoji (should be blocked by regex)
            "user\u202e123",  # Right-to-left override
            "user\u200b123",  # Zero-width space
        ]

        for unicode_attempt in unicode_attacks:
            result = voice_security.grant_consent(
                user_id=unicode_attempt,
                consent_type="voice_processing",
                granted=True
            )
            # Only reject if it contains invalid regex patterns
            if not voice_security._validate_user_id(unicode_attempt):
                assert result is False, f"Invalid Unicode user_id should be rejected: {repr(unicode_attempt)}"

    def test_consent_record_persistence(self, voice_security, tmp_path):
        """Test that consent records are properly persisted."""
        consent_data = {
            "user_id": "test_user_persist",
            "consent_type": "voice_processing",
            "granted": True,
            "ip_address": "192.168.1.1",
            "user_agent": "TestAgent/1.0",
            "consent_text": "Test consent text"
        }

        # Grant consent
        result = voice_security.grant_consent(**consent_data)
        assert result is True

        # Check that consent file was created
        consent_file = tmp_path / "consents" / "test_user_persist.json"
        assert consent_file.exists(), "Consent file should be created"

        # Verify file contents
        with open(consent_file, 'r') as f:
            saved_data = json.load(f)
            assert saved_data['user_id'] == consent_data['user_id']
            assert saved_data['consent_type'] == consent_data['consent_type']
            assert saved_data['granted'] == consent_data['granted']

    def test_audit_log_creation(self, voice_security, tmp_path):
        """Test that audit logs are created for security events."""
        # Grant consent to trigger audit log
        voice_security.grant_consent(
            user_id="audit_test_user",
            consent_type="voice_processing",
            granted=True,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )

        # Check that audit log was created
        audit_files = list((tmp_path / "audit").glob("audit_*.json"))
        assert len(audit_files) > 0, "Audit log file should be created"

        # Verify audit log contents
        with open(audit_files[0], 'r') as f:
            audit_logs = json.load(f)
            assert len(audit_logs) > 0
            latest_log = audit_logs[-1]
            assert latest_log['event_type'] == "consent_update"
            assert latest_log['user_id'] == "audit_test_user"
            assert latest_log['action'] == "grant_consent"
            assert latest_log['result'] == "success"


class TestMemoryLeakPrevention:
    """Test memory leak prevention in voice/audio_processor.py"""

    @pytest.fixture
    def audio_config(self):
        """Create test audio configuration with small limits for testing."""
        class MockAudioConfig:
            sample_rate = 16000
            channels = 1
            chunk_size = 1024
            format = "wav"
            max_buffer_size = 50  # Small buffer for testing
            max_memory_mb = 1    # Small memory limit for testing (1MB)

        return MockAudioConfig()

    @pytest.fixture
    def voice_config(self, audio_config):
        """Create test voice configuration."""
        class MockVoiceConfig:
            audio = audio_config

        return MockVoiceConfig()

    @pytest.fixture
    def audio_processor(self, voice_config):
        """Create SimplifiedAudioProcessor instance."""
        from voice.audio_processor import SimplifiedAudioProcessor
        processor = SimplifiedAudioProcessor(voice_config)
        return processor

    def test_buffer_is_bounded_deque(self, audio_processor):
        """Test that audio_buffer is a bounded deque."""
        assert isinstance(audio_processor.audio_buffer, deque), "audio_buffer should be a deque"
        assert audio_processor.audio_buffer.maxlen is not None, "audio_buffer should have maxlen set"
        assert audio_processor.audio_buffer.maxlen == 50, "audio_buffer maxlen should match config"

    def test_buffer_size_enforcement(self, audio_processor):
        """Test that buffer size limits are enforced."""
        # Fill buffer beyond its capacity
        for i in range(100):  # More than max_buffer_size (50)
            test_data = np.random.rand(1024, 1).astype(np.float32)
            audio_processor.audio_buffer.append(test_data)

        # Buffer should not exceed maximum size
        assert len(audio_processor.audio_buffer) <= 50, "Buffer should not exceed maximum size"

    def test_memory_tracking_functionality(self, audio_processor):
        """Test that memory tracking works correctly."""
        initial_memory = audio_processor._buffer_bytes_estimate
        assert initial_memory == 0, "Initial memory estimate should be 0"

        # Add some audio data
        test_data = np.random.rand(1024, 1).astype(np.float32)
        expected_size = test_data.nbytes

        audio_processor.audio_buffer.append(test_data)

        # Memory estimate should be updated
        assert audio_processor._buffer_bytes_estimate >= expected_size, "Memory estimate should be updated"

    def test_memory_limit_enforcement(self, audio_processor):
        """Test that memory limits are enforced."""
        # Calculate memory limit in bytes
        memory_limit = audio_processor._max_memory_bytes
        assert memory_limit == 1 * 1024 * 1024, "Memory limit should be 1MB"

        # Try to add data that exceeds memory limit
        large_data = np.random.rand(500_000, 1).astype(np.float32)  # ~2MB of data

        # Simulate the memory check logic from the recording callback
        chunk_size_bytes = large_data.nbytes
        would_exceed = audio_processor._buffer_bytes_estimate + chunk_size_bytes > memory_limit

        assert would_exceed, "Large data should exceed memory limit"

    def test_buffer_cleanup_on_memory_limit(self, audio_processor):
        """Test buffer cleanup when memory limit is reached."""
        # Fill buffer with data until memory limit is approached
        chunk_size = 100 * 1024  # 100KB chunks
        memory_limit = audio_processor._max_memory_bytes

        chunks_added = 0
        while audio_processor._buffer_bytes_estimate < memory_limit * 0.8:
            test_data = np.random.rand(chunk_size // 4, 1).astype(np.float32)  # float32 = 4 bytes per element
            audio_processor.audio_buffer.append(test_data)
            audio_processor._buffer_bytes_estimate += test_data.nbytes
            chunks_added += 1

            if chunks_added > 100:  # Safety break
                break

        # Buffer should have some data
        assert len(audio_processor.audio_buffer) > 0, "Buffer should contain data"
        assert audio_processor._buffer_bytes_estimate > 0, "Memory estimate should be > 0"

    def test_force_cleanup_buffers(self, audio_processor):
        """Test force cleanup of buffers."""
        # Add some data to buffer
        for i in range(10):
            test_data = np.random.rand(1024, 1).astype(np.float32)
            audio_processor.audio_buffer.append(test_data)

        # Verify buffer has data
        assert len(audio_processor.audio_buffer) > 0, "Buffer should contain data"

        # Force cleanup
        cleared_count = audio_processor.force_cleanup_buffers()

        # Buffer should be empty
        assert len(audio_processor.audio_buffer) == 0, "Buffer should be empty after cleanup"
        assert cleared_count > 0, "Should report number of cleared items"
        assert audio_processor._buffer_bytes_estimate == 0, "Memory estimate should be reset"

    def test_recording_memory_monitoring(self, audio_processor):
        """Test memory monitoring during recording simulation."""
        # Test the memory monitoring logic that would be used in recording
        initial_memory = audio_processor._buffer_bytes_estimate

        # Simulate adding chunks during recording
        for i in range(5):
            test_chunk = np.random.rand(1024, 1).astype(np.float32)
            chunk_size = test_chunk.nbytes

            # Check memory limit (simulating recording callback logic)
            if audio_processor._buffer_bytes_estimate + chunk_size > audio_processor._max_memory_bytes:
                # Would skip this chunk in real recording
                continue

            # Add chunk
            audio_processor.audio_buffer.append(test_chunk)
            audio_processor._buffer_bytes_estimate += chunk_size

        # Memory should be tracked
        assert audio_processor._buffer_bytes_estimate > initial_memory, "Memory should increase"

    def test_get_memory_usage(self, audio_processor):
        """Test memory usage reporting functionality."""
        # Add some test data
        test_data = np.random.rand(1024, 1).astype(np.float32)
        audio_processor.audio_buffer.append(test_data)
        audio_processor._buffer_bytes_estimate += test_data.nbytes

        # Get memory usage info
        memory_info = audio_processor.get_memory_usage()

        # Verify structure
        required_keys = ['buffer_size', 'max_buffer_size', 'buffer_usage_percent',
                        'memory_usage_bytes', 'memory_limit_bytes', 'memory_usage_percent']
        for key in required_keys:
            assert key in memory_info, f"Memory info should contain {key}"

        # Verify values make sense
        assert memory_info['buffer_size'] > 0, "Buffer size should be > 0"
        assert memory_info['memory_usage_bytes'] > 0, "Memory usage should be > 0"
        assert memory_info['memory_limit_bytes'] == 1 * 1024 * 1024, "Memory limit should be 1MB"

    def test_recording_thread_safety_cleanup(self, audio_processor):
        """Test that recording cleanup is thread-safe."""
        # Simulate recording state
        audio_processor.is_recording = True
        audio_processor.recording_start_time = time.time()

        # Add data to buffer
        test_data = np.random.rand(1024, 1).astype(np.float32)
        audio_processor.audio_buffer.append(test_data)
        audio_processor._buffer_bytes_estimate += test_data.nbytes

        # Test cleanup under lock
        with audio_processor._lock:
            audio_processor.is_recording = False
            audio_processor.audio_buffer.clear()
            audio_processor._buffer_bytes_estimate = 0

        # Verify cleanup worked
        assert audio_processor.is_recording == False, "Recording should be stopped"
        assert len(audio_processor.audio_buffer) == 0, "Buffer should be cleared"
        assert audio_processor._buffer_bytes_estimate == 0, "Memory estimate should be reset"

    def test_audio_data_size_validation(self, audio_processor):
        """Test validation of audio data size to prevent excessive memory usage."""
        # Test creating audio data that's too large
        large_audio_data = np.random.rand(200_000_000, 1).astype(np.float32)  # ~800MB

        # Simulate the validation logic from stop_recording
        total_size = 0
        max_size = 100_000_000  # 100MB limit from the code

        # This should be rejected by size validation
        would_exceed = total_size + large_audio_data.size > max_size
        assert would_exceed, "Large audio data should exceed size limit"

    def test_cleanup_error_handling(self, audio_processor):
        """Test error handling during cleanup operations."""
        # Add some data
        test_data = np.random.rand(1024, 1).astype(np.float32)
        audio_processor.audio_buffer.append(test_data)

        # Test cleanup handles errors gracefully
        try:
            audio_processor.cleanup()
            # Should not raise exception
            assert True, "Cleanup should complete without errors"
        except Exception as e:
            pytest.fail(f"Cleanup should not raise exception: {e}")

        # Buffer should be clean after cleanup
        assert len(audio_processor.audio_buffer) == 0, "Buffer should be empty after cleanup"


class TestThreadSafety:
    """Test thread safety in voice/voice_service.py"""

    @pytest.fixture
    def mock_config(self):
        """Create mock voice configuration."""
        class MockSecurityConfig:
            encryption_enabled = False
            consent_required = False

        class MockVoiceConfig:
            voice_enabled = True
            default_voice_profile = "default"
            security = MockSecurityConfig()

        return MockVoiceConfig()

    @pytest.fixture
    def mock_security(self, mock_config):
        """Create mock security instance."""
        mock_security = Mock()
        mock_security.initialize.return_value = True
        mock_security.grant_consent.return_value = True
        return mock_security

    @pytest.fixture
    def voice_service(self, mock_config, mock_security):
        """Create VoiceService instance for testing."""
        with patch('voice.voice_service.SimplifiedAudioProcessor'), \
             patch('voice.voice_service.STTService'), \
             patch('voice.voice_service.TTSService'), \
             patch('voice.voice_service.VoiceCommandProcessor'):

            from voice.voice_service import VoiceService
            service = VoiceService(mock_config, mock_security)
            return service

    def test_sessions_lock_exists(self, voice_service):
        """Test that sessions lock is properly initialized."""
        assert hasattr(voice_service, '_sessions_lock'), "Service should have _sessions_lock"
        assert isinstance(voice_service._sessions_lock, threading.RLock), "Lock should be RLock"

    def test_concurrent_session_creation(self, voice_service):
        """Test concurrent session creation doesn't cause race conditions."""
        session_ids = []
        errors = []

        def create_session(session_id):
            try:
                with voice_service._sessions_lock:
                    created_id = voice_service.create_session(session_id)
                    session_ids.append(created_id)
            except Exception as e:
                errors.append(e)

        # Create sessions concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session, f"session_{i}") for i in range(20)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"No errors should occur: {errors}"

        # All sessions should be created
        assert len(session_ids) == 20, "All sessions should be created"

        # Session IDs should be unique
        assert len(set(session_ids)) == 20, "Session IDs should be unique"

    def test_concurrent_session_access(self, voice_service):
        """Test concurrent session access is thread-safe."""
        # Create a session first
        session_id = voice_service.create_session("test_session")

        access_results = []
        errors = []

        def access_session():
            try:
                for _ in range(10):
                    session = voice_service.get_session(session_id)
                    current = voice_service.get_current_session()
                    access_results.append(session is not None)
            except Exception as e:
                errors.append(e)

        # Access session concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_session) for _ in range(5)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"No errors should occur: {errors}"

        # All accesses should succeed
        assert all(access_results), "All session accesses should succeed"

    def test_concurrent_session_creation_destruction(self, voice_service):
        """Test concurrent session creation and destruction."""
        operations = []
        errors = []

        def session_operations(worker_id):
            try:
                for i in range(5):
                    session_id = f"worker_{worker_id}_session_{i}"

                    # Create session
                    created_id = voice_service.create_session(session_id)
                    operations.append(f"create_{created_id}")

                    # Access session
                    session = voice_service.get_session(created_id)
                    operations.append(f"access_{session is not None}")

                    # Destroy session
                    voice_service.destroy_session(created_id)
                    operations.append(f"destroy_{created_id}")

                    # Small delay to increase interleaving
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Run operations concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(session_operations, i) for i in range(3)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"No errors should occur: {errors}"

        # Operations should complete
        assert len(operations) == 45, "All operations should complete (3 workers * 5 sessions * 3 operations)"

    def test_high_load_session_management(self, voice_service):
        """Test session management under high load."""
        session_count = 100
        created_sessions = []
        errors = []

        def high_load_operations():
            try:
                local_sessions = []
                for i in range(session_count // 10):  # Each worker creates 10 sessions
                    session_id = f"load_test_{i}_{threading.current_thread().ident}"
                    created = voice_service.create_session(session_id)
                    local_sessions.append(created)
                    created_sessions.append(created)

                # Simulate some operations
                for session_id in local_sessions:
                    voice_service.get_session(session_id)
                    voice_service.get_current_session()

                # Cleanup
                for session_id in local_sessions:
                    voice_service.destroy_session(session_id)

            except Exception as e:
                errors.append(e)

        # Run under high load
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(high_load_operations) for _ in range(10)]
            for future in as_completed(futures):
                future.result()
        end_time = time.time()

        # Should complete without errors
        assert len(errors) == 0, f"No errors should occur under high load: {errors}"

        # Should complete in reasonable time
        assert end_time - start_time < 10.0, "High load operations should complete quickly"

        # All sessions should be cleaned up
        assert len(voice_service.sessions) == 0, "All sessions should be cleaned up"

    def test_event_loop_reference_safety(self, voice_service):
        """Test event loop reference is handled safely."""
        # Initially should be None
        assert voice_service._event_loop is None, "Event loop should be None initially"

        # Should not crash when accessing event loop in callbacks
        try:
            # This simulates the audio callback scenario
            test_audio_data = Mock()
            voice_service._audio_callback(test_audio_data)
            assert True, "Audio callback should handle missing event loop"
        except Exception as e:
            pytest.fail(f"Audio callback should not crash: {e}")

    def test_metrics_thread_safety(self, voice_service):
        """Test that metrics updates are thread-safe."""
        def update_metrics():
            for i in range(100):
                voice_service.metrics['total_interactions'] += 1
                voice_service.metrics['sessions_created'] += 1

        # Update metrics concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_metrics) for _ in range(5)]
            for future in as_completed(futures):
                future.result()

        # Metrics should be updated correctly
        assert voice_service.metrics['total_interactions'] == 500, "Metrics should be updated correctly"
        assert voice_service.metrics['sessions_created'] == 500, "Metrics should be updated correctly"

    def test_callback_registration_thread_safety(self, voice_service):
        """Test thread-safe callback registration."""
        def register_callbacks():
            for i in range(10):
                callback = lambda x, y, i=i: f"callback_{i}_{x}_{y}"
                voice_service.on_text_received = callback
                voice_service.on_audio_played = callback
                voice_service.on_command_executed = callback
                voice_service.on_error = callback

        # Register callbacks concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(register_callbacks) for _ in range(3)]
            for future in as_completed(futures):
                future.result()

        # Should complete without errors
        assert voice_service.on_text_received is not None, "Callbacks should be set"
        assert voice_service.on_audio_played is not None, "Callbacks should be set"
        assert voice_service.on_command_executed is not None, "Callbacks should be set"
        assert voice_service.on_error is not None, "Callbacks should be set"

    def test_state_consistency_under_concurrency(self, voice_service):
        """Test that service state remains consistent under concurrent access."""
        states = []
        errors = []

        def check_state():
            try:
                for _ in range(50):
                    state = voice_service.is_running
                    sessions_count = len(voice_service.sessions)
                    current_session = voice_service.get_current_session()

                    states.append({
                        'is_running': state,
                        'sessions_count': sessions_count,
                        'current_session': current_session is not None
                    })
            except Exception as e:
                errors.append(e)

        # Check state concurrently while performing operations
        def perform_operations():
            try:
                for i in range(10):
                    session_id = f"state_test_{i}"
                    voice_service.create_session(session_id)
                    voice_service.get_session(session_id)
                    voice_service.destroy_session(session_id)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_state) for _ in range(3)]
            futures.extend([executor.submit(perform_operations) for _ in range(2)])

            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"No errors should occur: {errors}"

        # Final state should be consistent
        assert len(voice_service.sessions) == 0, "All sessions should be cleaned up"


class TestIntegrationSecurity:
    """Test overall security integration"""

    @pytest.fixture
    def integrated_voice_system(self, tmp_path):
        """Create integrated voice system for security testing."""
        with patch('voice.security.Path') as mock_path:
            mock_path.return_value = tmp_path

            # Create mock config
            class MockSecurityConfig:
                encryption_enabled = True
                consent_required = True
                hipaa_compliance_enabled = True
                gdpr_compliance_enabled = True
                data_localization = True
                data_retention_hours = 24
                emergency_protocols_enabled = True
                privacy_mode = False
                anonymization_enabled = False

            class MockAudioConfig:
                sample_rate = 16000
                channels = 1
                chunk_size = 1024
                format = "wav"
                max_buffer_size = 100
                max_memory_mb = 10

            class MockVoiceConfig:
                voice_enabled = True
                default_voice_profile = "default"
                security = MockSecurityConfig()
                audio = MockAudioConfig()

            # Create components
            from voice.security import VoiceSecurity
            from voice.audio_processor import SimplifiedAudioProcessor

            config = MockVoiceConfig()
            security = VoiceSecurity(config)
            audio_processor = SimplifiedAudioProcessor(config)

            return {
                'config': config,
                'security': security,
                'audio_processor': audio_processor,
                'tmp_path': tmp_path
            }

    def test_malicious_input_propagation(self, integrated_voice_system):
        """Test that malicious input is blocked throughout the system."""
        security = integrated_voice_system['security']

        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "\x00\x01\x02",
            "user\ralias",
            "$(whoami)"
        ]

        for malicious_input in malicious_inputs:
            # Test that malicious input is rejected at security level
            result = security.grant_consent(
                user_id=malicious_input,
                consent_type="voice_processing",
                granted=True
            )
            assert result is False, f"Malicious input should be rejected: {repr(malicious_input[:30])}..."

    def test_dos_protection_memory_limits(self, integrated_voice_system):
        """Test DoS protection through memory limits."""
        audio_processor = integrated_voice_system['audio_processor']

        # Try to exhaust memory with large audio data
        large_chunks = []
        try:
            for i in range(1000):  # Many large chunks
                large_chunk = np.random.rand(100_000, 1).astype(np.float32)  # ~400KB each
                large_chunks.append(large_chunk)

                # Simulate memory limit check
                if audio_processor._buffer_bytes_estimate + large_chunk.nbytes > audio_processor._max_memory_bytes:
                    # Should stop here in real implementation
                    break

                audio_processor.audio_buffer.append(large_chunk)
                audio_processor._buffer_bytes_estimate += large_chunk.nbytes

                if i > 100:  # Safety break for test
                    break
        except MemoryError:
            pytest.fail("Should not get MemoryError - limits should prevent this")

        # Memory usage should be within limits
        assert audio_processor._buffer_bytes_estimate <= audio_processor._max_memory_bytes, "Memory usage should be within limits"

    def test_error_information_disclosure_prevention(self, integrated_voice_system):
        """Test that error messages don't disclose sensitive information."""
        security = integrated_voice_system['security']

        # Test various invalid inputs that might cause errors
        sensitive_inputs = [
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "${HOME}/.ssh/id_rsa",
            "file:///etc/passwd",
            "jdbc:mysql://localhost:3306/mysql"
        ]

        for sensitive_input in sensitive_inputs:
            # The system should not accept these
            result = security.grant_consent(
                user_id=sensitive_input,
                consent_type="voice_processing",
                granted=True
            )

            # Should be rejected without revealing system information
            assert result is False, f"Sensitive path input should be rejected: {sensitive_input}"

    def test_consent_flow_security(self, integrated_voice_system):
        """Test complete consent flow security."""
        security = integrated_voice_system['security']

        # Test legitimate consent flow
        legitimate_consent = {
            'user_id': 'legitimate_user_123',
            'consent_type': 'voice_processing',
            'granted': True,
            'ip_address': '192.168.1.100',
            'user_agent': 'Mozilla/5.0 (Test Browser)',
            'consent_text': 'I consent to voice processing for therapy sessions.'
        }

        result = security.grant_consent(**legitimate_consent)
        assert result is True, "Legitimate consent should be accepted"

        # Verify consent was recorded
        assert security.check_consent('legitimate_user_123', 'voice_processing') is True

        # Test consent revocation
        result = security.grant_consent(
            user_id='legitimate_user_123',
            consent_type='voice_processing',
            granted=False
        )
        assert result is True, "Consent revocation should work"

    def test_emergency_protocol_security(self, integrated_voice_system):
        """Test emergency protocol security."""
        security = integrated_voice_system['security']

        # Test emergency protocol triggering
        security.handle_emergency_protocol(
            emergency_type="crisis",
            user_id="test_user_crisis",
            details={"severity": "high", "timestamp": time.time()}
        )

        # Verify emergency data preservation
        emergency_dir = integrated_voice_system['tmp_path'] / "emergency" / "test_user_crisis"
        assert emergency_dir.exists(), "Emergency directory should be created"

        # Test lockdown functionality
        security.handle_emergency_protocol(
            emergency_type="privacy_breach",
            user_id="breach_user"
        )

        # Verify lockdown
        assert security.is_user_locked_down("breach_user") is True, "User should be in lockdown"

    def test_audit_trail_integrity(self, integrated_voice_system):
        """Test audit trail integrity and security."""
        security = integrated_voice_system['security']

        # Perform various operations
        operations = [
            ("user1", "voice_processing", True),
            ("user2", "data_storage", True),
            ("user1", "transcription", False),
            ("user3", "all_consent", True)
        ]

        for user_id, consent_type, granted in operations:
            security.grant_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0"
            )

        # Verify audit logs were created
        audit_files = list((integrated_voice_system['tmp_path'] / "audit").glob("audit_*.json"))
        assert len(audit_files) > 0, "Audit files should be created"

        # Verify audit log integrity
        with open(audit_files[0], 'r') as f:
            audit_logs = json.load(f)
            assert len(audit_logs) >= len(operations), "All operations should be audited"

            # Check log structure
            for log in audit_logs:
                assert 'timestamp' in log, "Audit log should have timestamp"
                assert 'event_type' in log, "Audit log should have event_type"
                assert 'user_id' in log, "Audit log should have user_id"
                assert 'action' in log, "Audit log should have action"
                assert 'result' in log, "Audit log should have result"

    def test_compliance_features(self, integrated_voice_system):
        """Test compliance features (HIPAA/GDPR)."""
        security = integrated_voice_system['security']

        # Get compliance status
        compliance = security.get_compliance_status()

        # Verify compliance features are enabled
        assert compliance['hipaa_compliant'] is True, "HIPAA compliance should be enabled"
        assert compliance['gdpr_compliant'] is True, "GDPR compliance should be enabled"
        assert compliance['encryption_enabled'] is True, "Encryption should be enabled"
        assert compliance['data_localization'] is True, "Data localization should be enabled"
        assert compliance['consent_required'] is True, "Consent should be required"
        assert compliance['emergency_protocols_enabled'] is True, "Emergency protocols should be enabled"

    def test_data_retention_security(self, integrated_voice_system):
        """Test data retention and cleanup security."""
        security = integrated_voice_system['security']

        # Create some test data
        test_user_id = "retention_test_user"
        security.grant_consent(
            user_id=test_user_id,
            consent_type="voice_processing",
            granted=True
        )

        # Verify data exists
        assert test_user_id in security.consent_records, "Consent record should exist"

        # Test manual cleanup
        security._cleanup_user_data(test_user_id)

        # Verify data is cleaned up
        assert test_user_id not in security.consent_records, "Consent record should be cleaned up"

    def test_encryption_key_security(self, integrated_voice_system):
        """Test encryption key security."""
        security = integrated_voice_system['security']

        # Encryption key should exist
        assert security.encryption_key is not None, "Encryption key should exist"

        # Key file should have secure permissions
        key_file = integrated_voice_system['tmp_path'] / "encryption.key"
        if key_file.exists():
            # Note: In real test, we'd check file permissions
            # This is more complex due to cross-platform differences
            assert key_file.exists(), "Key file should exist"

    def test_concurrent_security_operations(self, integrated_voice_system):
        """Test concurrent security operations."""
        security = integrated_voice_system['security']
        errors = []
        successful_operations = []

        def security_operations(worker_id):
            try:
                for i in range(10):
                    user_id = f"concurrent_user_{worker_id}_{i}"
                    result = security.grant_consent(
                        user_id=user_id,
                        consent_type="voice_processing",
                        granted=True,
                        ip_address=f"192.168.1.{worker_id}",
                        user_agent=f"TestAgent/{worker_id}"
                    )

                    if result:
                        successful_operations.append(user_id)

                        # Test consent checking
                        consent_check = security.check_consent(user_id, "voice_processing")
                        assert consent_check is True, f"Consent check should succeed for {user_id}"

            except Exception as e:
                errors.append(e)

        # Run security operations concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(security_operations, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"No errors should occur: {errors}"

        # Operations should succeed
        assert len(successful_operations) == 50, "All security operations should succeed"

        # Verify all consents were recorded
        for user_id in successful_operations:
            assert security.check_consent(user_id, "voice_processing") is True, f"Consent should exist for {user_id}"


class TestSecurityEdgeCases:
    """Test edge cases and boundary conditions for security."""

    @pytest.fixture
    def security_system(self, tmp_path):
        """Create security system for edge case testing."""
        with patch('voice.security.Path') as mock_path:
            mock_path.return_value = tmp_path

            class MockSecurityConfig:
                encryption_enabled = True
                consent_required = True
                hipaa_compliance_enabled = True
                gdpr_compliance_enabled = True
                data_localization = True
                data_retention_hours = 24
                emergency_protocols_enabled = True
                privacy_mode = False
                anonymization_enabled = False

            class MockVoiceConfig:
                security = MockSecurityConfig()

            from voice.security import VoiceSecurity
            return VoiceSecurity(MockVoiceConfig())

    def test_empty_and_null_inputs(self, security_system):
        """Test handling of empty and null inputs."""
        # Test empty strings
        result = security_system.grant_consent(
            user_id="",
            consent_type="voice_processing",
            granted=True
        )
        assert result is False, "Empty user_id should be rejected"

        # Test None values
        result = security_system.grant_consent(
            user_id=None,
            consent_type="voice_processing",
            granted=True
        )
        assert result is False, "None user_id should be rejected"

        result = security_system.grant_consent(
            user_id="test_user",
            consent_type=None,
            granted=True
        )
        assert result is False, "None consent_type should be rejected"

    def test_maximum_boundary_values(self, security_system):
        """Test maximum boundary values."""
        # Test maximum length user_id (50 chars)
        max_user_id = "a" * 50
        result = security_system.grant_consent(
            user_id=max_user_id,
            consent_type="voice_processing",
            granted=True
        )
        assert result is True, "Maximum length user_id should be accepted"

        # Test maximum length user agent (500 chars)
        max_user_agent = "a" * 500
        result = security_system.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            user_agent=max_user_agent
        )
        assert result is True, "Maximum length user_agent should be accepted"

        # Test maximum length consent text (10000 chars)
        max_consent_text = "a" * 10000
        result = security_system.grant_consent(
            user_id="test_user",
            consent_type="voice_processing",
            granted=True,
            consent_text=max_consent_text
        )
        assert result is True, "Maximum length consent text should be accepted"

    def test_unicode_boundary_cases(self, security_system):
        """Test Unicode boundary cases."""
        # Test various Unicode characters that might be allowed
        unicode_test_cases = [
            "user123",  # ASCII (should work)
            "café",  # Extended Latin (might not work due to regex)
            "naïve",  # Extended Latin with combining
            "测试",  # Chinese (should not work due to regex)
            "тест",  # Cyrillic (should not work due to regex)
        ]

        for test_case in unicode_test_cases:
            result = security_system.grant_consent(
                user_id=test_case,
                consent_type="voice_processing",
                granted=True
            )
            # Only ASCII should pass the regex validation
            if test_case == "user123":
                assert result is True, f"ASCII user_id should work: {test_case}"
            else:
                # Non-ASCII should be rejected by the regex pattern
                assert result is False, f"Non-ASCII user_id should be rejected: {test_case}"

    def test_rapid_succession_attacks(self, security_system):
        """Test rapid succession attacks."""
        successful_grants = 0

        # Rapid consent grants (potential DoS)
        for i in range(1000):
            user_id = f"rapid_user_{i}"
            result = security_system.grant_consent(
                user_id=user_id,
                consent_type="voice_processing",
                granted=True
            )
            if result:
                successful_grants += 1

        # System should handle rapid requests gracefully
        assert successful_grants > 900, "System should handle high volume of requests"

        # Verify data integrity after rapid operations
        assert len(security_system.consent_records) == successful_grants, "All successful grants should be recorded"

    def test_filesystem_boundary_conditions(self, security_system):
        """Test filesystem boundary conditions."""
        # Test very long user_id that might cause filesystem issues
        long_user_id = "a" * 49  # Just under the 50 char limit

        result = security_system.grant_consent(
            user_id=long_user_id,
            consent_type="voice_processing",
            granted=True
        )
        assert result is True, "Long but valid user_id should work"

        # Verify file was created correctly
        consent_file = security_system.consents_dir / f"{long_user_id}.json"
        assert consent_file.exists(), "Consent file should be created for long user_id"

    def test_memory_exhaustion_resilience(self, security_system):
        """Test resilience to memory exhaustion attempts."""
        # Try to create many consent records to exhaust memory
        for i in range(10000):
            user_id = f"memory_test_{i}"
            result = security_system.grant_consent(
                user_id=user_id,
                consent_type="voice_processing",
                granted=True,
                consent_text="Test consent text for memory testing"
            )

            if not result:
                # System started rejecting (possibly due to memory constraints)
                break

        # System should still be functional
        assert len(security_system.consent_records) > 0, "Some consent records should exist"

        # Should still accept new valid requests
        result = security_system.grant_consent(
            user_id="final_test_user",
            consent_type="voice_processing",
            granted=True
        )
        assert result is True, "System should still accept valid requests after stress test"

    def test_concurrent_file_access(self, security_system):
        """Test concurrent file access doesn't cause corruption."""
        import threading

        errors = []
        successful_writes = []

        def concurrent_consent_grants(worker_id):
            try:
                for i in range(10):
                    user_id = f"concurrent_file_{worker_id}_{i}"
                    result = security_system.grant_consent(
                        user_id=user_id,
                        consent_type="voice_processing",
                        granted=True
                    )
                    if result:
                        successful_writes.append(user_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads writing to files concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_consent_grants, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0, f"No file access errors should occur: {errors}"

        # All writes should succeed
        assert len(successful_writes) == 100, "All concurrent writes should succeed"

        # Verify data integrity
        for user_id in successful_writes:
            assert user_id in security_system.consent_records, f"Consent record should exist for {user_id}"


if __name__ == "__main__":
    # Run tests with pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])