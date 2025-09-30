#!/usr/bin/env python3
"""
Test script for the Enhanced Text-to-Speech Service

This script demonstrates the capabilities of the TTS service including:
- Multiple provider support (OpenAI, ElevenLabs, Piper)
- Therapeutic voice profiles
- Emotion and prosody control
- SSML support
- Voice caching
- Streaming synthesis
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the voice module to the path
sys.path.append(str(Path(__file__).parent))

from voice.tts_service import TTSService, EmotionType
from voice.config import VoiceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_tts_service():
    """Test the TTS service with various configurations."""

    print("🎤 Testing Enhanced TTS Service")
    print("=" * 50)

    # Initialize configuration
    config = VoiceConfig()

    # Initialize TTS service
    tts_service = TTSService(config)

    # Test availability
    print(f"✅ TTS Service Available: {tts_service.is_available()}")
    print(f"📋 Available Providers: {tts_service.get_available_providers()}")
    print(f"🎯 Preferred Provider: {tts_service.get_preferred_provider()}")

    if not tts_service.is_available():
        print("❌ No TTS providers available. Please configure API keys.")
        return

    # Test voice profiles
    print("\n🎭 Testing Voice Profiles:")
    print("-" * 30)

    available_profiles = list(tts_service.voice_profiles.keys())
    print(f"Available profiles: {available_profiles}")

    # Test each voice profile
    test_text = "Hello, I'm your AI therapist. How can I help you today?"

    for profile_name in available_profiles[:2]:  # Test first 2 profiles
        try:
            print(f"\n🔊 Testing profile: {profile_name}")

            # Get profile details
            profile_settings = tts_service.get_voice_profile_settings(profile_name)
            print(f"   Description: {profile_settings['description']}")
            print(f"   Voice ID: {profile_settings['voice_id']}")
            print(f"   Emotion: {profile_settings['emotion']}")

            # Test synthesis
            result = await tts_service.synthesize_speech(
                text=test_text,
                voice_profile=profile_name
            )

            print(f"   ✅ Synthesized successfully")
            print(f"   📏 Duration: {result.duration:.2f}s")
            print(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
            print(f"   🎵 Provider: {result.provider}")

            # Save audio file
            output_file = f"test_output_{profile_name}.wav"
            tts_service.save_audio(result.audio_data, output_file)
            print(f"   💾 Saved to: {output_file}")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")

    # Test emotion control
    print("\n😊 Testing Emotion Control:")
    print("-" * 30)

    emotion_test_text = "I understand you're feeling overwhelmed right now."

    for emotion in [EmotionType.CALM, EmotionType.EMPATHETIC, EmotionType.ENCOURAGING]:
        try:
            print(f"\n🎭 Testing emotion: {emotion.value}")

            result = await tts_service.synthesize_speech(
                text=emotion_test_text,
                emotion=emotion
            )

            print(f"   ✅ Synthesized with {emotion.value} emotion")
            print(f"   📏 Duration: {result.duration:.2f}s")

            # Save emotion-specific audio
            output_file = f"test_emotion_{emotion.value}.wav"
            tts_service.save_audio(result.audio_data, output_file)
            print(f"   💾 Saved to: {output_file}")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")

    # Test SSML support
    print("\n🎨 Testing SSML Support:")
    print("-" * 30)

    ssml_text = "It's <emphasis level='strong'>important</emphasis> to take care of your mental health."

    try:
        print("🔊 Testing with SSML enabled")
        result = await tts_service.synthesize_speech(
            text=ssml_text,
            ssml_enabled=True
        )

        print(f"   ✅ SSML synthesis successful")
        print(f"   📏 Duration: {result.duration:.2f}s")

        # Save SSML audio
        tts_service.save_audio(result.audio_data, "test_ssml.wav")
        print(f"   💾 Saved to: test_ssml.wav")

    except Exception as e:
        print(f"   ❌ SSML test failed: {str(e)}")

    # Test different providers
    print("\n🔄 Testing Provider Fallback:")
    print("-" * 30)

    providers = tts_service.get_available_providers()

    for provider in providers:
        try:
            print(f"🔊 Testing provider: {provider}")

            result = await tts_service.synthesize_speech(
                text="This is a test of the provider fallback system.",
                provider=provider
            )

            print(f"   ✅ {provider} synthesis successful")
            print(f"   📏 Duration: {result.duration:.2f}s")

        except Exception as e:
            print(f"   ❌ {provider} failed: {str(e)}")

    # Test streaming
    print("\n🌊 Testing Streaming Synthesis:")
    print("-" * 30)

    try:
        streaming_text = "This is a longer text to test the streaming functionality of the TTS service."
        print("🔊 Testing audio streaming...")

        chunk_count = 0
        total_duration = 0.0

        async for chunk in tts_service.synthesize_stream(streaming_text):
            chunk_count += 1
            total_duration += chunk.duration
            print(f"   📦 Received chunk {chunk_count}: {chunk.duration:.3f}s")

        print(f"   ✅ Streaming complete: {chunk_count} chunks, {total_duration:.2f}s total")

    except Exception as e:
        print(f"   ❌ Streaming test failed: {str(e)}")

    # Test voice caching
    print("\n💾 Testing Voice Caching:")
    print("-" * 30)

    cache_test_text = "This is a test of the voice caching system."

    # First synthesis (should miss cache)
    start_time = time.time()
    result1 = await tts_service.synthesize_speech(cache_test_text)
    first_time = time.time() - start_time

    # Second synthesis (should hit cache)
    start_time = time.time()
    result2 = await tts_service.synthesize_speech(cache_test_text)
    second_time = time.time() - start_time

    print(f"   📊 First synthesis: {first_time:.3f}s")
    print(f"   📊 Second synthesis: {second_time:.3f}s")
    print(f"   📈 Speed improvement: {((first_time - second_time) / first_time * 100):.1f}%")
    print(f"   📦 Cache size: {len(tts_service.audio_cache)}")

    # Test custom voice profile creation
    print("\n🎨 Testing Custom Voice Profile:")
    print("-" * 30)

    try:
        # Create a custom profile
        custom_profile = tts_service.create_custom_voice_profile(
            name="test_custom",
            base_profile="calm_therapist",
            modifications={
                "description": "Custom test profile",
                "pitch": 1.2,
                "speed": 0.8,
                "volume": 0.9
            }
        )

        print(f"   ✅ Created custom profile: {custom_profile.name}")

        # Test the custom profile
        result = await tts_service.synthesize_speech(
            text="This is a test of the custom voice profile.",
            voice_profile="test_custom"
        )

        print(f"   ✅ Custom profile synthesis successful")
        print(f"   📏 Duration: {result.duration:.2f}s")

        # Save custom profile audio
        tts_service.save_audio(result.audio_data, "test_custom_profile.wav")
        print(f"   💾 Saved to: test_custom_profile.wav")

    except Exception as e:
        print(f"   ❌ Custom profile test failed: {str(e)}")

    # Display statistics
    print("\n📊 TTS Service Statistics:")
    print("-" * 30)

    stats = tts_service.get_statistics()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

    # Test preload functionality
    print("\n⚡ Testing Response Preloading:")
    print("-" * 30)

    common_responses = [
        "I understand how you're feeling.",
        "That sounds really challenging.",
        "Let's explore this together.",
        "You're doing great by sharing this."
    ]

    tts_service.preload_common_responses(common_responses)
    print(f"   ✅ Preloaded {len(common_responses)} common responses")

    # Clean up
    print("\n🧹 Cleaning up...")
    tts_service.cleanup()
    print("   ✅ Cleanup complete")

    print("\n🎉 TTS Service Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    import time

    # Run the test
    asyncio.run(test_tts_service())