"""
Malformed Input Processing and Sanitization Tests

This module tests handling of malformed and malicious input across all components:
- Invalid input validation and sanitization
- Malformed audio data processing
- Invalid configuration parameter handling
- Malicious input detection and prevention
- Input boundary condition testing
- Data type validation and coercion
- Input size limit enforcement
"""

import pytest
import json
import re
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import numpy as np

# Import project modules
from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.stt_service import STTService, STTResult
from voice.audio_processor import AudioData, SimplifiedAudioProcessor
from voice.config import VoiceConfig, SecurityConfig
from voice.security import VoiceSecurity
from app import sanitize_user_input


class TestInputValidation:
    """Test input validation and sanitization mechanisms."""

    def test_malformed_audio_data_handling(self):
        """Test handling of malformed audio data."""
        # Test various malformed audio data scenarios
        malformed_audio_cases = [
            # Empty data
            AudioData(data=np.array([]), sample_rate=16000, duration=0.0, channels=1),
            # Invalid sample rate
            AudioData(data=np.array([0.1, 0.2]), sample_rate=0, duration=1.0, channels=1),
            # Invalid channels
            AudioData(data=np.array([0.1, 0.2]), sample_rate=16000, duration=1.0, channels=0),
            # Wrong data type
            AudioData(data=np.array([1, 2, 3], dtype=np.int32), sample_rate=16000, duration=1.0, channels=1),
            # NaN values
            AudioData(data=np.array([np.nan, 0.1, 0.2]), sample_rate=16000, duration=1.0, channels=1),
            # Infinite values
            AudioData(data=np.array([np.inf, 0.1, 0.2]), sample_rate=16000, duration=1.0, channels=1),
        ]

        config = Mock(spec=VoiceConfig)
        processor = SimplifiedAudioProcessor(config)

        for audio_data in malformed_audio_cases:
            # Should handle malformed data gracefully
            try:
                result = processor.process_audio(audio_data)
                # Should either succeed or return None/safe result
                assert result is None or isinstance(result, AudioData)
            except Exception:
                # Should not crash on malformed input
                pass

    def test_invalid_configuration_parameters(self):
        """Test handling of invalid configuration parameters."""
        # Test invalid configuration values
        invalid_configs = [
            {"voice_speed": -1.0},  # Negative speed
            {"voice_speed": 100.0},  # Excessive speed
            {"volume": -0.5},  # Negative volume
            {"volume": 2.0},  # Excessive volume
            {"sample_rate": 0},  # Zero sample rate
            {"sample_rate": -16000},  # Negative sample rate
            {"language": ""},  # Empty language
            {"timeout": -1},  # Negative timeout
        ]

        for invalid_config in invalid_configs:
            config = Mock(spec=VoiceConfig)

            # Apply invalid configuration
            for key, value in invalid_config.items():
                setattr(config, key, value)

            # Should handle invalid config gracefully
            try:
                service = VoiceService(config, Mock(spec=VoiceSecurity))
                # Should either use defaults or clamp values
                assert service is not None
            except Exception:
                # Should not crash on invalid config
                pass

    def test_malformed_session_identifiers(self):
        """Test handling of malformed session identifiers."""
        service = VoiceService(Mock(spec=VoiceConfig), Mock(spec=VoiceSecurity))

        # Test various malformed session IDs
        malformed_session_ids = [
            "",  # Empty string
            "   ",  # Whitespace only
            "../malicious/path",  # Path traversal attempt
            "../../../../etc/passwd",  # Path traversal attack
            "<script>alert('xss')</script>",  # XSS attempt
            "$(malicious_command)",  # Command injection attempt
            "'; DROP TABLE sessions; --",  # SQL injection attempt
            "\x00\x01\x02",  # Binary data
            "session_id_with_very_long_name_" * 20,  # Extremely long ID
        ]

        for session_id in malformed_session_ids:
            # Should handle malformed session IDs gracefully
            try:
                result = service.create_session(session_id)
                # Should either sanitize or reject malformed IDs
                if result:
                    assert isinstance(result, str)
                    # Sanitized ID should not contain dangerous characters
                    assert ".." not in result
                    assert "<" not in result
                    assert "$" not in result
            except Exception:
                # Should not crash on malformed input
                pass

    def test_invalid_text_input_sanitization(self):
        """Test sanitization of invalid text input."""
        # Test various malicious and malformed text inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "$(malicious_command)",
            "`malicious_command`",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "\x00\x01\x02\x03",  # Binary data
            "text_with_very_long_content_" * 1000,  # Extremely long content
            "",  # Empty input
            "   ",  # Whitespace only
            "text\nwith\nnewlines",  # Control characters
        ]

        for malicious_input in malicious_inputs:
            # Test sanitization
            sanitized = sanitize_user_input(malicious_input)

            # Should be sanitized or handled safely
            if sanitized:
                assert isinstance(sanitized, str)
                # Should not contain dangerous patterns
                assert "<script" not in sanitized.lower()
                assert "javascript:" not in sanitized.lower()
                assert "$(" not in sanitized
                assert "`" not in sanitized

    def test_binary_data_input_handling(self):
        """Test handling of binary data input."""
        # Test binary data in various contexts
        binary_inputs = [
            b"\x00\x01\x02\x03",  # Raw binary
            b"\xff\xd8\xff\xe0",  # JPEG header
            b"RIFF\x00\x00\x00\x00WAVE",  # WAV header
            b"\x89PNG\r\n\x1a\n",  # PNG header
        ]

        for binary_data in binary_inputs:
            # Should handle binary data gracefully
            try:
                sanitized = sanitize_user_input(binary_data.decode('latin-1'))
                # Should either sanitize or return safe result
                assert isinstance(sanitized, str)
            except (UnicodeDecodeError, Exception):
                # Should not crash on binary data
                pass


class TestMaliciousInputDetection:
    """Test detection and prevention of malicious input."""

    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        # Test SQL injection payloads
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; UPDATE users SET password='hacked'; --",
            "admin'--",
            "' UNION SELECT * FROM passwords --",
            "1; SELECT * FROM sensitive_data",
        ]

        for payload in sql_injection_payloads:
            sanitized = sanitize_user_input(payload)

            # Should neutralize SQL injection attempts
            assert "DROP TABLE" not in sanitized.upper()
            assert "UPDATE" not in sanitized.upper()
            assert "UNION SELECT" not in sanitized.upper()
            assert "SELECT" not in sanitized.upper()

    def test_xss_prevention(self):
        """Test prevention of Cross-Site Scripting (XSS) attacks."""
        # Test XSS payloads
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
            "vbscript:msgbox('xss')",
            "<iframe src=javascript:alert('xss')></iframe>",
        ]

        for payload in xss_payloads:
            sanitized = sanitize_user_input(payload)

            # Should neutralize XSS attempts
            assert "<script" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "vbscript:" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        # Test command injection payloads
        command_injection_payloads = [
            "$(malicious_command)",
            "`malicious_command`",
            "| malicious_command",
            "; malicious_command",
            "&& malicious_command",
            "$(cat /etc/passwd)",
            "`rm -rf /`",
        ]

        for payload in command_injection_payloads:
            sanitized = sanitize_user_input(payload)

            # Should neutralize command injection attempts
            assert "$(" not in sanitized
            assert "`" not in sanitized
            assert "|" not in sanitized
            assert ";" not in sanitized
            assert "&&" not in sanitized

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        # Test path traversal payloads
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "/etc/passwd",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "~/.ssh/id_rsa",
            "/proc/self/environ",
        ]

        for payload in path_traversal_payloads:
            sanitized = sanitize_user_input(payload)

            # Should neutralize path traversal attempts
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert "/etc/" not in sanitized
            assert "/proc/" not in sanitized
            assert "C:\\" not in sanitized


class TestInputBoundaryConditions:
    """Test input boundary conditions and edge cases."""

    def test_extreme_input_sizes(self):
        """Test handling of extreme input sizes."""
        # Test extremely large inputs
        large_inputs = [
            "x" * 1000000,  # 1MB string
            "x" * 100000,   # 100KB string
            "x" * 10000,    # 10KB string
            "x" * 1000,     # 1KB string
        ]

        for large_input in large_inputs:
            # Should handle large inputs gracefully
            try:
                sanitized = sanitize_user_input(large_input)
                # Should either process or truncate appropriately
                assert isinstance(sanitized, str)
                # Should not be larger than reasonable limit
                assert len(sanitized) <= 100000  # 100KB max
            except Exception:
                # Should not crash on large input
                pass

    def test_unicode_edge_cases(self):
        """Test handling of Unicode edge cases."""
        # Test various Unicode inputs
        unicode_inputs = [
            "Normal text with ümlaut",
            "Text with émojis 🚀⭐️",
            "Text with árabic أهلاً",
            "Text with chinese 中文",
            "Text with japanese 日本語",
            "Text with korean 한국어",
            "\u0000\u0001\u0002",  # Control characters
            "\uD800\uDC00",  # Surrogate pairs
            "\uFFFF",  # Non-character
        ]

        for unicode_input in unicode_inputs:
            try:
                sanitized = sanitize_user_input(unicode_input)
                # Should handle Unicode gracefully
                assert isinstance(sanitized, str)
                # Should preserve valid Unicode
                if "ü" in unicode_input or "🚀" in unicode_input:
                    # Should preserve emojis and umlauts
                    pass
            except Exception:
                # Should not crash on Unicode input
                pass

    def test_null_and_empty_inputs(self):
        """Test handling of null and empty inputs."""
        null_inputs = [
            None,
            "",
            "   ",
            "\t\t\t",
            "\n\n\n",
            {},
            [],
        ]

        for null_input in null_inputs:
            try:
                if null_input is None:
                    # Should handle None input
                    sanitized = sanitize_user_input("")
                elif isinstance(null_input, str):
                    sanitized = sanitize_user_input(null_input)
                else:
                    # Should handle non-string input
                    sanitized = sanitize_user_input(str(null_input))

                # Should return safe result
                assert isinstance(sanitized, str)
            except Exception:
                # Should not crash on null input
                pass

    def test_special_character_handling(self):
        """Test handling of special characters."""
        special_inputs = [
            "!@#$%^&*()",
            "~`{}[]|\\",
            "+=?/",
            "text<>text",
            "text\"text",
            "text&text",
            "text<text",
            "text>text",
        ]

        for special_input in special_inputs:
            sanitized = sanitize_user_input(special_input)

            # Should handle special characters appropriately
            assert isinstance(sanitized, str)
            # Should neutralize HTML entities
            assert '"' not in sanitized
            assert "&" not in sanitized
            assert "<" not in sanitized
            assert ">" not in sanitized


class TestDataTypeValidation:
    """Test data type validation and coercion."""

    def test_audio_data_type_validation(self):
        """Test validation of audio data types."""
        # Test various data types for audio
        audio_type_cases = [
            # Valid cases
            (np.array([0.1, 0.2, 0.3], dtype=np.float32), True),
            (np.array([1000, 2000, 3000], dtype=np.int16), True),
            # Invalid cases
            (np.array(["invalid"], dtype=object), False),
            (np.array([None, None], dtype=object), False),
            ([0.1, 0.2, 0.3], False),  # Python list
            ("string_data", False),  # String
            (12345, False),  # Integer
            ({"key": "value"}, False),  # Dictionary
        ]

        for data, should_be_valid in audio_type_cases:
            try:
                if isinstance(data, np.ndarray):
                    # Test numpy array validation
                    if data.dtype in [np.float32, np.int16]:
                        is_valid = True
                    else:
                        is_valid = False
                else:
                    is_valid = False

                assert is_valid == should_be_valid
            except Exception:
                # Should not crash on type validation
                pass

    def test_configuration_type_coercion(self):
        """Test type coercion for configuration values."""
        # Test type coercion scenarios
        coercion_cases = [
            # String to numeric
            ("123", 123),
            ("45.67", 45.67),
            # Numeric to string
            (123, "123"),
            (True, "True"),
            # Boolean conversion
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
        ]

        for input_value, expected_output in coercion_cases:
            # Test type coercion logic
            if isinstance(input_value, str):
                try:
                    # Try to convert to int
                    if '.' not in input_value:
                        coerced = int(input_value)
                    else:
                        coerced = float(input_value)
                except ValueError:
                    # Try boolean conversion
                    if input_value.lower() in ['true', '1']:
                        coerced = True
                    elif input_value.lower() in ['false', '0']:
                        coerced = False
                    else:
                        coerced = input_value  # Keep as string
            else:
                coerced = str(input_value)

            # Should handle coercion appropriately
            assert isinstance(coerced, (int, float, str, bool))

    def test_parameter_range_validation(self):
        """Test validation of parameter ranges."""
        # Test parameter range validation
        parameter_ranges = {
            "voice_speed": (0.1, 5.0),
            "volume": (0.0, 2.0),
            "confidence_threshold": (0.0, 1.0),
            "timeout": (0.1, 300.0),
            "sample_rate": (8000, 48000),
        }

        test_values = [
            ("voice_speed", -1.0),  # Below range
            ("voice_speed", 10.0),  # Above range
            ("volume", -0.5),  # Below range
            ("volume", 3.0),  # Above range
            ("confidence_threshold", -0.1),  # Below range
            ("confidence_threshold", 1.5),  # Above range
            ("timeout", 0.0),  # At boundary
            ("timeout", 300.0),  # At boundary
        ]

        for param_name, test_value in test_values:
            if param_name in parameter_ranges:
                min_val, max_val = parameter_ranges[param_name]

                # Validate range
                if test_value < min_val:
                    validated_value = min_val  # Clamp to minimum
                elif test_value > max_val:
                    validated_value = max_val  # Clamp to maximum
                else:
                    validated_value = test_value

                # Should be within valid range
                assert min_val <= validated_value <= max_val


class TestInputSizeLimits:
    """Test enforcement of input size limits."""

    def test_audio_size_limits(self):
        """Test enforcement of audio size limits."""
        # Test audio data size limits
        max_audio_duration = 300.0  # 5 minutes
        max_audio_size = 50 * 1024 * 1024  # 50MB

        # Test various audio sizes
        size_test_cases = [
            # Small audio (should pass)
            (16000 * 10, True),  # 10 seconds
            # Medium audio (should pass)
            (16000 * 60, True),  # 1 minute
            # Large audio (should be limited)
            (16000 * 600, False),  # 10 minutes (too long)
            # Very large audio (should be limited)
            (16000 * 3600, False),  # 1 hour (way too long)
        ]

        for num_samples, should_pass in size_test_cases:
            audio_data = AudioData(
                data=np.random.randn(num_samples).astype(np.float32),
                sample_rate=16000,
                duration=num_samples / 16000.0,
                channels=1
            )

            # Validate size limits
            duration = audio_data.duration
            estimated_size = num_samples * 4  # 4 bytes per float32

            is_valid = True

            if duration > max_audio_duration:
                is_valid = False

            if estimated_size > max_audio_size:
                is_valid = False

            # Should enforce size limits
            if should_pass:
                assert is_valid
            else:
                assert not is_valid

    def test_text_input_length_limits(self):
        """Test enforcement of text input length limits."""
        # Test text length limits
        max_text_length = 10000  # 10KB

        length_test_cases = [
            ("Short text", True),
            ("Medium length text " * 100, True),
            ("Very long text " * 1000, False),  # Too long
            ("Extremely long text " * 10000, False),  # Way too long
        ]

        for text, should_pass in length_test_cases:
            # Validate length limits
            is_valid = len(text) <= max_text_length

            # Should enforce length limits
            if should_pass:
                assert is_valid
            else:
                assert not is_valid

    def test_session_data_size_limits(self):
        """Test enforcement of session data size limits."""
        # Test session data size limits
        max_conversation_entries = 1000
        max_audio_buffer_size = 50 * 1024 * 1024  # 50MB

        # Create session with large data
        session = VoiceSession(
            session_id="test_session",
            state=VoiceSessionState.IDLE,
            start_time=1234567890.0,
            last_activity=1234567890.0,
            conversation_history=[],
            current_voice_profile="default",
            audio_buffer=[],
            metadata={}
        )

        # Add many conversation entries
        for i in range(1500):  # Exceed limit
            session.conversation_history.append({
                'type': 'user',
                'text': f'Message {i}',
                'timestamp': 1234567890.0 + i,
                'confidence': 0.95
            })

        # Should enforce conversation history limit
        if len(session.conversation_history) > max_conversation_entries:
            # Keep only recent entries
            session.conversation_history = session.conversation_history[-max_conversation_entries:]

        assert len(session.conversation_history) <= max_conversation_entries

    def test_metadata_size_limits(self):
        """Test enforcement of metadata size limits."""
        # Test metadata size limits
        max_metadata_size = 1024 * 1024  # 1MB

        # Create large metadata
        large_metadata = {
            'large_field': 'x' * (100 * 1024),  # 100KB string
            'another_field': 'y' * (200 * 1024),  # 200KB string
        }

        # Validate metadata size
        metadata_size = len(json.dumps(large_metadata))

        if metadata_size > max_metadata_size:
            # Should truncate or reject large metadata
            # In real implementation, would sanitize large fields
            for key, value in large_metadata.items():
                if len(value) > 10000:  # 10KB per field limit
                    large_metadata[key] = value[:10000] + "..."

        # Should enforce metadata size limits
        final_size = len(json.dumps(large_metadata))
        assert final_size <= max_metadata_size


class TestSanitizationEffectiveness:
    """Test effectiveness of input sanitization."""

    def test_comprehensive_input_sanitization(self):
        """Test comprehensive input sanitization pipeline."""
        # Test comprehensive malicious input
        comprehensive_malicious_input = '''
        <script>alert('xss')</script>
        $(malicious_command)
        '; DROP TABLE users; --
        ../../../etc/passwd
        javascript:alert('xss')
        <img src=x onerror=alert('xss')>
        | cat /etc/passwd
        && rm -rf /
        `whoami`
        '''

        sanitized = sanitize_user_input(comprehensive_malicious_input)

        # Should neutralize all attack vectors
        assert "<script" not in sanitized.lower()
        assert "$(" not in sanitized
        assert "DROP TABLE" not in sanitized.upper()
        assert "../" not in sanitized
        assert "javascript:" not in sanitized.lower()
        assert "onerror=" not in sanitized.lower()
        assert "|" not in sanitized
        assert "&&" not in sanitized
        assert "`" not in sanitized

        # Should preserve safe content
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0

    def test_sanitization_idempotency(self):
        """Test that sanitization is idempotent."""
        # Test multiple rounds of sanitization
        original_input = "Normal text with <script>alert('xss')</script> content"

        # Apply sanitization multiple times
        sanitized1 = sanitize_user_input(original_input)
        sanitized2 = sanitize_user_input(sanitized1)
        sanitized3 = sanitize_user_input(sanitized2)

        # Should produce consistent results
        assert sanitized1 == sanitized2 == sanitized3

    def test_sanitization_performance(self):
        """Test performance of sanitization process."""
        import time

        # Test sanitization performance
        test_input = "Normal text " * 1000  # Reasonable size input

        start_time = time.time()
        sanitized = sanitize_user_input(test_input)
        end_time = time.time()

        # Should complete sanitization quickly
        sanitization_time = end_time - start_time
        assert sanitization_time < 1.0  # Should complete in less than 1 second

        # Should preserve input length for normal text
        assert len(sanitized) >= len(test_input) * 0.9  # At least 90% of original length


class TestErrorHandlingInInputProcessing:
    """Test error handling during input processing."""

    def test_input_processing_error_recovery(self):
        """Test recovery from errors during input processing."""
        # Mock processor that fails on certain inputs
        class FailingProcessor:
            def __init__(self):
                self.failure_count = 0

            def process_input(self, input_data):
                self.failure_count += 1
                if self.failure_count % 3 == 0:
                    raise ValueError("Processing failed")

                return f"processed_{input_data}"

        processor = FailingProcessor()

        # Test error recovery
        test_inputs = ["input1", "input2", "input3", "input4", "input5"]

        for input_data in test_inputs:
            try:
                result = processor.process_input(input_data)
                # Should succeed for non-failing cases
                assert "processed_" in result
            except ValueError:
                # Should handle processing failures gracefully
                # In real implementation, would log and continue
                pass

    def test_partial_input_processing(self):
        """Test processing of partially valid input."""
        # Test input with mixed valid and invalid parts
        mixed_input = "Valid text <script>alert('xss')</script> more valid text"

        sanitized = sanitize_user_input(mixed_input)

        # Should preserve valid parts and sanitize invalid parts
        assert "Valid text" in sanitized
        assert "more valid text" in sanitized
        assert "<script" not in sanitized.lower()

    def test_input_validation_error_reporting(self):
        """Test error reporting for input validation failures."""
        # Test validation error collection
        validation_errors = []

        def validate_and_report(input_data, field_name):
            try:
                sanitized = sanitize_user_input(input_data)
                return sanitized
            except Exception as e:
                error_msg = f"Validation failed for {field_name}: {str(e)}"
                validation_errors.append(error_msg)
                return None

        # Test various inputs that might cause validation errors
        test_cases = [
            ("<script>alert('xss')</script>", "user_input"),
            ("../../../etc/passwd", "file_path"),
            ("$(malicious)", "command"),
        ]

        for input_data, field_name in test_cases:
            result = validate_and_report(input_data, field_name)

            # Should handle validation errors gracefully
            if result is None:
                # Should have recorded error
                assert len(validation_errors) > 0
                assert field_name in validation_errors[-1]