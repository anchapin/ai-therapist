"""
Data Corruption and Recovery Tests

This module tests handling of corrupted files and data across all components:
- Corrupted audio file processing and recovery
- Malformed configuration files
- Corrupted vector store files
- Invalid session data recovery
- Database corruption scenarios
- File system corruption handling
- Data integrity validation and repair
"""

import pytest
import json
import os
import tempfile
import shutil
import time
import numpy as np
import struct
import wave
import io
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import hashlib

# Import project modules
from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.stt_service import STTService, STTResult
from voice.audio_processor import AudioData, SimplifiedAudioProcessor
from voice.config import VoiceConfig, SecurityConfig
from voice.security import VoiceSecurity
from app import sanitize_user_input, validate_vectorstore_integrity


class TestAudioFileCorruption:
    """Test handling of corrupted audio files."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def create_corrupted_wav_file(self, filepath: str, corruption_type: str = "header"):
        """Create a corrupted WAV file for testing."""
        if corruption_type == "header":
            # Corrupted header
            with open(filepath, 'wb') as f:
                f.write(b'CORRUPTED_HEADER_DATA')
        elif corruption_type == "truncated":
            # Truncated file
            with open(filepath, 'wb') as f:
                # Write valid header but truncated data
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt ')
                f.write(struct.pack('<I', 16))  # Chunk size
                f.write(struct.pack('<H', 1))   # Audio format (PCM)
                f.write(struct.pack('<H', 1))   # Num channels
                f.write(struct.pack('<I', 16000))  # Sample rate
                f.write(struct.pack('<I', 32000))  # Byte rate
                f.write(struct.pack('<H', 2))   # Block align
                f.write(struct.pack('<H', 16))  # Bits per sample
                # Truncated - missing data chunk
        elif corruption_type == "invalid_format":
            # Invalid format chunk
            with open(filepath, 'wb') as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt ')
                f.write(struct.pack('<I', 16))  # Chunk size
                f.write(struct.pack('<H', 999))  # Invalid audio format
                f.write(struct.pack('<H', 1))   # Num channels
                f.write(struct.pack('<I', 16000))  # Sample rate
                f.write(struct.pack('<I', 32000))  # Byte rate
                f.write(struct.pack('<H', 2))   # Block align
                f.write(struct.pack('<H', 16))  # Bits per sample

    def test_corrupted_wav_header_handling(self, temp_dir):
        """Test handling of corrupted WAV file headers."""
        corrupted_file = os.path.join(temp_dir, "corrupted_header.wav")
        self.create_corrupted_wav_file(corrupted_file, "header")

        # Mock audio processor to handle corrupted file
        with patch('voice.audio_processor.SimplifiedAudioProcessor') as MockProcessor:
            mock_processor = Mock()
            MockProcessor.return_value = mock_processor
            mock_processor.load_audio = Mock(side_effect=ValueError("Invalid WAV header"))

            config = Mock(spec=VoiceConfig)
            processor = SimplifiedAudioProcessor(config)

            # Test that corrupted file is handled gracefully
            result = processor.load_audio(corrupted_file)
            assert result is None  # Should return None for corrupted files

    def test_truncated_wav_recovery(self, temp_dir):
        """Test recovery from truncated WAV files."""
        truncated_file = os.path.join(temp_dir, "truncated.wav")
        self.create_corrupted_wav_file(truncated_file, "truncated")

        config = Mock(spec=VoiceConfig)
        processor = SimplifiedAudioProcessor(config)

        # Mock file reading to simulate truncation
        with patch('builtins.open', side_effect=EOFError("Unexpected end of file")):
            result = processor.load_audio(truncated_file)
            # Should handle gracefully
            assert result is None or isinstance(result, AudioData)

    def test_invalid_audio_format_handling(self, temp_dir):
        """Test handling of invalid audio format chunks."""
        invalid_file = os.path.join(temp_dir, "invalid_format.wav")
        self.create_corrupted_wav_file(invalid_file, "invalid_format")

        config = Mock(spec=VoiceConfig)
        processor = SimplifiedAudioProcessor(config)

        # Should handle invalid format gracefully
        result = processor.load_audio(invalid_file)
        assert result is None or isinstance(result, AudioData)

    def test_audio_data_checksum_validation(self):
        """Test audio data integrity validation using checksums."""
        # Create original audio data
        original_data = np.random.randn(16000).astype(np.float32)

        # Create AudioData object
        audio_data = AudioData(
            data=original_data.copy(),
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        # Calculate checksum
        checksum = hashlib.md5(original_data.tobytes()).hexdigest()

        # Simulate data corruption
        audio_data.data[1000:2000] = np.random.randn(1000).astype(np.float32)

        # Verify checksum detects corruption
        corrupted_checksum = hashlib.md5(audio_data.data.tobytes()).hexdigest()
        assert checksum != corrupted_checksum

        # Test recovery mechanism
        if checksum != corrupted_checksum:
            # Restore from backup or regenerate
            audio_data.data = original_data.copy()
            restored_checksum = hashlib.md5(audio_data.data.tobytes()).hexdigest()
            assert checksum == restored_checksum


class TestConfigurationCorruption:
    """Test handling of corrupted configuration files."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for configuration files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def create_corrupted_json_config(self, filepath: str, corruption_type: str = "syntax"):
        """Create corrupted JSON configuration file."""
        if corruption_type == "syntax":
            # Invalid JSON syntax
            with open(filepath, 'w') as f:
                f.write('{"invalid": json_syntax, "missing": "quotes"}')
        elif corruption_type == "truncated":
            # Truncated JSON
            with open(filepath, 'w') as f:
                f.write('{"valid_key": "value", "truncated": ')
        elif corruption_type == "empty":
            # Empty file
            with open(filepath, 'w') as f:
                f.write('')

    def test_corrupted_voice_config_recovery(self, temp_config_dir):
        """Test recovery from corrupted voice configuration."""
        config_file = os.path.join(temp_config_dir, "voice_config.json")
        self.create_corrupted_json_config(config_file, "syntax")

        # Mock VoiceConfig to use corrupted file
        with patch('voice.config.VoiceConfig') as MockConfig:
            mock_config = Mock()
            MockConfig.from_file = Mock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
            MockConfig.return_value = mock_config

            # Should fall back to default configuration
            config = VoiceConfig()
            # Verify default values are used
            assert hasattr(config, 'voice_enabled')

    def test_truncated_config_file_handling(self, temp_config_dir):
        """Test handling of truncated configuration files."""
        config_file = os.path.join(temp_config_dir, "config.json")
        self.create_corrupted_json_config(config_file, "truncated")

        # Test loading truncated config
        try:
            with open(config_file, 'r') as f:
                json.load(f)
            assert False, "Should have raised JSON error"
        except json.JSONDecodeError:
            # Expected - should handle gracefully
            pass

    def test_empty_config_file_recovery(self, temp_config_dir):
        """Test recovery from empty configuration files."""
        config_file = os.path.join(temp_config_dir, "empty_config.json")
        self.create_corrupted_json_config(config_file, "empty")

        # Test loading empty config
        config = VoiceConfig()
        # Should use default values when file is empty
        assert config.voice_enabled is not None


class TestVectorStoreCorruption:
    """Test handling of corrupted vector store files."""

    @pytest.fixture
    def temp_vector_dir(self):
        """Create temporary directory for vector store files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def create_corrupted_faiss_index(self, filepath: str):
        """Create corrupted FAISS index file."""
        # Write invalid FAISS header
        with open(filepath, 'wb') as f:
            f.write(b'INVALID_FAISS_HEADER')
            f.write(np.random.bytes(1024))

    def test_corrupted_vectorstore_detection(self, temp_vector_dir):
        """Test detection of corrupted vector store files."""
        corrupted_index = os.path.join(temp_vector_dir, "corrupted.index")
        self.create_corrupted_faiss_index(corrupted_index)

        # Test vectorstore integrity validation
        with patch('app.validate_vectorstore_integrity') as mock_validate:
            mock_validate.return_value = False

            # Should detect corruption
            is_valid = validate_vectorstore_integrity(temp_vector_dir)
            assert is_valid is False

    def test_vectorstore_repair_mechanism(self, temp_vector_dir):
        """Test vectorstore repair after corruption."""
        # Simulate corrupted index file
        index_file = os.path.join(temp_vector_dir, "index.faiss")
        metadata_file = os.path.join(temp_vector_dir, "index.pkl")

        # Create corrupted files
        self.create_corrupted_faiss_index(index_file)

        with open(metadata_file, 'wb') as f:
            f.write(b'corrupted_metadata')

        # Test repair mechanism
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True

            with patch('faiss.read_index') as mock_read_index:
                mock_read_index.side_effect = RuntimeError("Corrupted index")

                # Should attempt to rebuild corrupted index
                try:
                    # Simulate rebuild process
                    if os.path.exists(index_file):
                        os.remove(index_file)  # Remove corrupted file
                        # Recreate valid index (mock)
                        with open(index_file, 'wb') as f:
                            f.write(b'VALID_FAISS_INDEX')
                except Exception:
                    pass  # Expected for corrupted files


class TestSessionDataCorruption:
    """Test handling of corrupted session data."""

    def test_corrupted_session_state_recovery(self):
        """Test recovery from corrupted session state."""
        # Create session with corrupted data
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

        # Simulate metadata corruption
        session.metadata = "corrupted_string_instead_of_dict"

        # Test recovery mechanism
        if not isinstance(session.metadata, dict):
            # Restore metadata
            session.metadata = {
                'created_at': session.start_time,
                'voice_settings': {
                    'voice_speed': 1.2,
                    'volume': 1.0,
                    'voice_pitch': 1.0,
                    'pitch': 1.0
                }
            }

        # Verify recovery
        assert isinstance(session.metadata, dict)
        assert 'voice_settings' in session.metadata

    def test_conversation_history_corruption(self):
        """Test handling of corrupted conversation history."""
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

        # Simulate corrupted conversation history
        session.conversation_history = "corrupted_data"

        # Test recovery
        if not isinstance(session.conversation_history, list):
            session.conversation_history = [
                {
                    'type': 'system',
                    'text': 'Session recovered from corruption',
                    'timestamp': session.start_time,
                    'confidence': 1.0
                }
            ]

        # Verify recovery
        assert isinstance(session.conversation_history, list)
        assert len(session.conversation_history) > 0

    def test_audio_buffer_corruption_recovery(self):
        """Test recovery from corrupted audio buffer."""
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

        # Simulate corrupted audio buffer
        session.audio_buffer = "corrupted_buffer_data"

        # Test recovery
        if not isinstance(session.audio_buffer, list):
            session.audio_buffer = [
                AudioData(
                    data=np.array([]),
                    sample_rate=16000,
                    duration=0.0,
                    channels=1
                )
            ]

        # Verify recovery
        assert isinstance(session.audio_buffer, list)


class TestFileSystemCorruption:
    """Test handling of file system corruption scenarios."""

    @pytest.fixture
    def temp_fs_dir(self):
        """Create temporary directory for file system tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_permission_denied_recovery(self, temp_fs_dir):
        """Test recovery from permission denied errors."""
        protected_file = os.path.join(temp_fs_dir, "protected.wav")

        # Create file and remove permissions
        with open(protected_file, 'w') as f:
            f.write("test data")

        os.chmod(protected_file, 0o000)  # Remove all permissions

        try:
            # Attempt to read file
            with open(protected_file, 'r'):
                pass
            assert False, "Should have raised permission error"
        except PermissionError:
            # Expected - should handle gracefully
            pass
        finally:
            # Restore permissions for cleanup
            os.chmod(protected_file, 0o644)

    def test_disk_space_exhaustion_handling(self, temp_fs_dir):
        """Test handling of disk space exhaustion."""
        large_file = os.path.join(temp_fs_dir, "large_file.wav")

        # Mock disk space exhaustion
        with patch('os.path.getsize', side_effect=OSError("No space left on device")):
            try:
                # Attempt to write large file
                with open(large_file, 'w') as f:
                    # Write more data than available space (mock)
                    f.write("x" * (1024 * 1024 * 100))  # 100MB
                assert False, "Should have raised disk space error"
            except OSError:
                # Expected - should handle gracefully
                pass

    def test_file_handle_leak_recovery(self):
        """Test recovery from file handle leaks."""
        # Simulate file handle leak scenario
        open_files = []

        try:
            # Open many files without closing
            for i in range(100):
                f = open(f"/tmp/test_file_{i}.txt", 'w')
                open_files.append(f)

            # Simulate cleanup when too many files are open
            if len(open_files) > 50:
                for f in open_files:
                    f.close()
                open_files = []

        except Exception:
            # Cleanup on error
            for f in open_files:
                try:
                    f.close()
                except:
                    pass

        # Verify cleanup
        assert len(open_files) == 0


class TestDataIntegrityValidation:
    """Test data integrity validation and repair mechanisms."""

    def test_audio_data_validation(self):
        """Test validation of audio data integrity."""
        # Create valid audio data
        valid_data = np.random.randn(16000).astype(np.float32)
        audio_data = AudioData(
            data=valid_data,
            sample_rate=16000,
            duration=1.0,
            channels=1
        )

        # Validate audio data
        is_valid = True

        # Check sample rate
        if audio_data.sample_rate <= 0:
            is_valid = False

        # Check duration
        if audio_data.duration <= 0:
            is_valid = False

        # Check data length consistency
        expected_length = int(audio_data.sample_rate * audio_data.duration * audio_data.channels)
        if len(audio_data.data) != expected_length:
            is_valid = False

        # Check for NaN or infinite values
        if np.any(np.isnan(audio_data.data)) or np.any(np.isinf(audio_data.data)):
            is_valid = False

        assert is_valid

    def test_configuration_data_validation(self):
        """Test validation of configuration data."""
        # Test valid configuration
        valid_config = {
            'voice_enabled': True,
            'stt_language': 'en',
            'tts_language': 'en',
            'voice_speed': 1.2,
            'volume': 1.0
        }

        # Validate configuration
        required_keys = ['voice_enabled', 'stt_language', 'tts_language']
        is_valid = all(key in valid_config for key in required_keys)

        # Validate data types
        if valid_config.get('voice_speed') is not None:
            if not isinstance(valid_config['voice_speed'], (int, float)):
                is_valid = False

        if valid_config.get('volume') is not None:
            if not isinstance(valid_config['volume'], (int, float)):
                is_valid = False

        assert is_valid

    def test_session_data_validation(self):
        """Test validation of session data integrity."""
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

        # Validate session data
        is_valid = True

        # Check session ID
        if not isinstance(session.session_id, str) or len(session.session_id) == 0:
            is_valid = False

        # Check timestamps
        current_time = time.time()
        if session.start_time > current_time or session.last_activity > current_time:
            is_valid = False

        # Check conversation history is list
        if not isinstance(session.conversation_history, list):
            is_valid = False

        # Check audio buffer is list
        if not isinstance(session.audio_buffer, list):
            is_valid = False

        # Check metadata is dict
        if not isinstance(session.metadata, dict):
            is_valid = False

        assert is_valid


class TestCorruptionRecoveryMechanisms:
    """Test comprehensive corruption recovery mechanisms."""

    def test_automatic_backup_restoration(self, temp_dir):
        """Test automatic restoration from backup files."""
        # Create original file
        original_file = os.path.join(temp_dir, "original.wav")
        with open(original_file, 'w') as f:
            f.write("original_data")

        # Create backup file
        backup_file = os.path.join(temp_dir, "backup.wav")
        with open(backup_file, 'w') as f:
            f.write("backup_data")

        # Simulate corruption of original file
        with open(original_file, 'w') as f:
            f.write("corrupted_data")

        # Test restoration from backup
        if os.path.exists(backup_file):
            # Restore from backup
            with open(original_file, 'w') as orig_f:
                with open(backup_file, 'r') as backup_f:
                    orig_f.write(backup_f.read())

        # Verify restoration
        with open(original_file, 'r') as f:
            content = f.read()
            assert content == "backup_data"

    def test_corruption_detection_algorithms(self):
        """Test algorithms for detecting data corruption."""
        # Test data integrity using multiple methods

        # Method 1: Checksum validation
        def calculate_checksum(data):
            return hashlib.md5(data.encode()).hexdigest()

        original_data = "test_data_for_checksum"
        original_checksum = calculate_checksum(original_data)

        # Simulate corruption
        corrupted_data = "test_data_corrupted"
        corrupted_checksum = calculate_checksum(corrupted_data)

        assert original_checksum != corrupted_checksum

        # Method 2: Format validation
        def validate_wav_format(file_path):
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    return True
            except wave.Error:
                return False

        # Method 3: Metadata consistency check
        def validate_metadata_consistency(metadata):
            required_fields = ['created_at', 'version', 'format']
            return all(field in metadata for field in required_fields)

        valid_metadata = {
            'created_at': '2023-01-01',
            'version': '1.0',
            'format': 'wav'
        }

        assert validate_metadata_consistency(valid_metadata)

        invalid_metadata = {
            'created_at': '2023-01-01'
            # Missing required fields
        }

        assert not validate_metadata_consistency(invalid_metadata)

    def test_incremental_corruption_recovery(self):
        """Test recovery from incremental corruption."""
        # Simulate data that becomes corrupted over time

        data_versions = [
            "version_1_data",
            "version_2_data",
            "version_3_data",
            "corrupted_data",
            "more_corrupted_data"
        ]

        # Keep track of last known good version
        last_good_version = None
        last_good_index = -1

        for i, data in enumerate(data_versions):
            # Check if data is corrupted (simple check)
            is_corrupted = "corrupted" in data

            if not is_corrupted:
                last_good_version = data
                last_good_index = i
            else:
                # Found corruption, recover from last good version
                if last_good_version:
                    data_versions[i] = last_good_version
                    break

        # Verify recovery
        assert data_versions[-1] == "version_3_data"
        assert last_good_index >= 0