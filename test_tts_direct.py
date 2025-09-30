#!/usr/bin/env python3
"""
Direct TTS Service Test

This test directly imports and tests the TTS service components.
"""

import sys
import os
from pathlib import Path

# Add the voice module to the path
sys.path.append(str(Path(__file__).parent))

def test_tts_service_direct():
    """Test TTS service by importing components directly."""
    try:
        # Import config directly
        from voice.config import VoiceConfig, VoiceProfile
        print("✅ VoiceConfig imported successfully")

        # Test configuration
        config = VoiceConfig()
        print(f"✅ VoiceConfig initialized")
        print(f"   - Voice enabled: {config.voice_enabled}")
        print(f"   - Default profile: {config.default_voice_profile}")
        print(f"   - Voice profiles: {len(config.voice_profiles)}")

        # Test OpenAI TTS configuration
        openai_configured = config.is_openai_tts_configured()
        print(f"   - OpenAI TTS configured: {openai_configured}")

        # Test ElevenLabs configuration
        elevenlabs_configured = config.is_elevenlabs_configured()
        print(f"   - ElevenLabs configured: {elevenlabs_configured}")

        # Test Piper configuration
        piper_configured = config.is_piper_configured()
        print(f"   - Piper TTS configured: {piper_configured}")

        # Test getting voice profile
        profile = config.get_voice_profile()
        print(f"✅ Default profile: {profile.name}")
        print(f"   - Description: {profile.description}")
        print(f"   - Voice ID: {profile.voice_id}")
        print(f"   - Pitch: {profile.pitch}")
        print(f"   - Speed: {profile.speed}")

        # Test voice profile settings
        settings = config.get_voice_profile_settings(profile.name)
        print(f"✅ Profile settings retrieved")
        print(f"   - Language: {settings['language']}")
        print(f"   - Gender: {settings['gender']}")
        print(f"   - Emotion: {settings['emotion']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_enums():
    """Test TTS enums and dataclasses."""
    try:
        # Create a mock AudioData class for testing
        import numpy as np
        from dataclasses import dataclass

        @dataclass
        class MockAudioData:
            data: np.ndarray
            sample_rate: int
            duration: float

        # Import TTS enums
        sys.path.insert(0, str(Path(__file__).parent / "voice"))

        # Read and execute the TTS service file to get the enums
        tts_file = Path(__file__).parent / "voice" / "tts_service.py"
        with open(tts_file, 'r') as f:
            tts_code = f.read()

        # Create a namespace and execute the code
        namespace = {}
        exec(tts_code, namespace)

        # Get the EmotionType enum
        EmotionType = namespace['EmotionType']
        print("✅ EmotionType enum imported")

        emotions = [e.value for e in EmotionType]
        print(f"   Available emotions: {emotions}")

        # Test dataclasses
        TTSResult = namespace['TTSResult']
        VoiceEmotionSettings = namespace['VoiceEmotionSettings']
        SSMLSettings = namespace['SSMLSettings']

        print("✅ TTS dataclasses imported")

        # Create test instances
        emotion_settings = VoiceEmotionSettings(
            emotion=EmotionType.CALM,
            pitch=0.9,
            speed=0.85
        )
        print(f"✅ VoiceEmotionSettings created: {emotion_settings.emotion.value}")

        ssml_settings = SSMLSettings()
        print(f"✅ SSMLSettings created: enabled={ssml_settings.enabled}")

        # Create mock audio data
        audio_data = MockAudioData(
            data=np.array([0.1, 0.2, 0.3]),
            sample_rate=22050,
            duration=0.1
        )

        tts_result = TTSResult(
            audio_data=audio_data,
            text="Test text",
            voice_profile="calm_therapist",
            provider="openai",
            duration=0.1
        )
        print(f"✅ TTSResult created: {tts_result.provider}")

        return True

    except Exception as e:
        print(f"❌ Error testing enums: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_methods():
    """Test configuration validation and methods."""
    try:
        from voice.config import VoiceConfig

        config = VoiceConfig()

        # Test configuration validation
        issues = config.validate_configuration()
        print(f"✅ Configuration validation:")
        if issues:
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("   No issues found")

        # Test provider preference
        preferred_tts = config.get_preferred_tts_service()
        print(f"✅ Preferred TTS service: {preferred_tts}")

        # Test configuration dictionary
        config_dict = config.to_dict()
        print(f"✅ Configuration dictionary created")
        print(f"   - Voice profiles: {config_dict['voice_profiles_count']}")
        print(f"   - Cache enabled: {config_dict['performance_cache']}")

        return True

    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_setup():
    """Test environment setup for TTS."""
    try:
        # Check environment variables
        env_vars = [
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "VOICE_ENABLED",
            "VOICE_OUTPUT_ENABLED",
            "VOICE_CACHE_ENABLED"
        ]

        print("✅ Environment variables:")
        for var in env_vars:
            value = os.getenv(var)
            if value:
                print(f"   - {var}: {'*' * 8 if 'API_KEY' in var else value}")
            else:
                print(f"   - {var}: Not set")

        # Check directories
        directories = [
            "./voice_profiles",
            "./knowledge",
            "./vectorstore"
        ]

        print("✅ Directories:")
        for directory in directories:
            path = Path(directory)
            if path.exists():
                print(f"   - {directory}: Exists")
            else:
                print(f"   - {directory}: Not found")

        return True

    except Exception as e:
        print(f"❌ Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct TTS service tests."""
    print("🎤 Direct TTS Service Test")
    print("=" * 50)

    tests = [
        ("TTS Configuration", test_tts_service_direct),
        ("TTS Enums & Classes", test_tts_enums),
        ("Configuration Methods", test_configuration_methods),
        ("Environment Setup", test_environment_setup)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All TTS service components are working!")
        print("\n📋 Setup Instructions:")
        print("1. Configure API keys:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("   export ELEVENLABS_API_KEY='your-elevenlabs-key'")
        print("   export ELEVENLABS_VOICE_ID='your-voice-id'")
        print("\n2. Install dependencies (in virtual environment):")
        print("   pip install -r requirements.txt")
        print("\n3. Test the full TTS service:")
        print("   python test_tts_service.py")
        print("\n4. Try the therapeutic example:")
        print("   python voice/examples/therapeutic_tts_example.py")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()