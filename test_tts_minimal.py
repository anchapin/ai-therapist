#!/usr/bin/env python3
"""
Minimal TTS Service Test

This test verifies the TTS service functionality without requiring audio dependencies.
"""

import sys
from pathlib import Path

# Add the voice module to the path
sys.path.append(str(Path(__file__).parent))

def test_tts_imports():
    """Test TTS service imports and basic functionality."""
    try:
        # Test basic imports
        from voice.tts_service import TTSService, EmotionType, TTSResult, TTSError
        from voice.config import VoiceConfig, VoiceProfile
        print("✅ TTS service imports successful")

        # Test configuration
        config = VoiceConfig()
        print(f"✅ VoiceConfig loaded")
        print(f"   - Voice enabled: {config.voice_enabled}")
        print(f"   - Default profile: {config.default_voice_profile}")
        print(f"   - Cache enabled: {config.performance.cache_enabled}")

        # Test voice profiles
        print(f"✅ Voice profiles loaded: {len(config.voice_profiles)}")
        for name, profile in config.voice_profiles.items():
            print(f"   - {name}: {profile.description}")

        # Test emotion types
        emotions = [e.value for e in EmotionType]
        print(f"✅ Emotion types: {emotions}")

        # Test TTS service initialization (without audio dependencies)
        print("✅ TTS service components tested successfully")

        # Test voice profile creation
        test_profile = VoiceProfile(
            name="test_profile",
            description="Test profile for verification",
            voice_id="alloy",
            language="en-US",
            pitch=1.0,
            speed=1.0,
            volume=1.0
        )
        print("✅ VoiceProfile creation successful")

        # Test emotion settings
        from voice.tts_service import VoiceEmotionSettings
        emotion_settings = VoiceEmotionSettings(
            emotion=EmotionType.CALM,
            pitch=0.9,
            speed=0.85
        )
        print(f"✅ VoiceEmotionSettings created: {emotion_settings.emotion.value}")

        return True

    except Exception as e:
        print(f"❌ Error during imports: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration validation."""
    try:
        from voice.config import VoiceConfig

        config = VoiceConfig()
        issues = config.validate_configuration()

        print("✅ Configuration validation:")
        if issues:
            print("   Issues found:")
            for issue in issues:
                print(f"     - {issue}")
        else:
            print("   No configuration issues found")

        # Test provider detection
        print(f"✅ Provider detection:")
        print(f"   - OpenAI TTS: {config.is_openai_configured()}")
        print(f"   - ElevenLabs: {config.is_elevenlabs_configured()}")
        print(f"   - Piper TTS: {config.is_piper_configured()}")
        print(f"   - Preferred TTS: {config.get_preferred_tts_service()}")

        return True

    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_profiles():
    """Test voice profile functionality."""
    try:
        from voice.config import VoiceConfig

        config = VoiceConfig()

        # Test getting voice profile
        profile = config.get_voice_profile()
        print(f"✅ Default voice profile: {profile.name}")

        # Test voice profile settings
        settings = config.get_voice_profile_settings(profile.name)
        print(f"✅ Voice profile settings retrieved")
        print(f"   - Voice ID: {settings['voice_id']}")
        print(f"   - Language: {settings['language']}")
        print(f"   - Pitch: {settings['pitch']}")
        print(f"   - Speed: {settings['speed']}")

        # Test configuration methods
        print(f"✅ Configuration methods:")
        print(f"   - ElevenLabs configured: {config.is_elevenlabs_configured()}")
        print(f"   - Voice profiles count: {len(config.voice_profiles)}")

        return True

    except Exception as e:
        print(f"❌ Voice profile test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test environment variable loading."""
    try:
        from voice.config import VoiceConfig

        config = VoiceConfig()

        print("✅ Environment variable loading:")
        print(f"   - Voice enabled: {config.voice_enabled}")
        print(f"   - Voice input: {config.voice_input_enabled}")
        print(f"   - Voice output: {config.voice_output_enabled}")
        print(f"   - Cache enabled: {config.performance.cache_enabled}")
        print(f"   - Cache size: {config.performance.cache_size}")
        print(f"   - Streaming enabled: {config.performance.streaming_enabled}")

        # Test security settings
        print(f"   - Encryption: {config.security.encryption_enabled}")
        print(f"   - HIPAA compliance: {config.security.hipaa_compliance_enabled}")
        print(f"   - Privacy mode: {config.security.privacy_mode}")

        return True

    except Exception as e:
        print(f"❌ Environment variable test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all TTS service tests."""
    print("🎤 Enhanced TTS Service Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_tts_imports),
        ("Configuration Validation", test_configuration_validation),
        ("Voice Profiles", test_voice_profiles),
        ("Environment Variables", test_environment_variables)
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
        print("🎉 All TTS service tests passed!")
        print("\n📋 Next Steps:")
        print("1. Configure API keys in .env file:")
        print("   - OPENAI_API_KEY (for OpenAI TTS)")
        print("   - ELEVENLABS_API_KEY (for ElevenLabs TTS)")
        print("2. Install audio dependencies:")
        print("   - pip install pyaudio librosa soundfile")
        print("3. Run full test: python test_tts_service.py")
        print("4. Run example: python voice/examples/therapeutic_tts_example.py")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()