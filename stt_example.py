#!/usr/bin/env python3
"""
STT Service Usage Example

This script demonstrates how to use the comprehensive STT service
with multiple providers and advanced features.

Usage:
    python stt_example.py

Requirements:
    - Configure your environment variables in .env file
    - At least one STT provider should be configured
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from voice.config import VoiceConfig
from voice.audio_processor import AudioProcessor
from voice.stt_service import STTService, STTResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demonstrate_stt_features():
    """Demonstrate STT service features."""
    logger.info("=== STT Service Feature Demonstration ===")

    try:
        # Initialize configuration and services
        config = VoiceConfig()
        stt_service = STTService(config)
        audio_processor = AudioProcessor(config)

        # Check available providers
        available_providers = stt_service.get_available_providers()
        logger.info(f"Available STT providers: {available_providers}")

        if not available_providers:
            logger.error("No STT providers available. Please configure at least one provider.")
            return

        # 1. Test with silence (quick connectivity test)
        logger.info("\n1. Testing service connectivity...")
        try:
            from voice.audio_processor import AudioData
            import numpy as np
            import time

            # Create 1 second of silence
            silence_audio = AudioData(
                data=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                channels=1,
                format="float32",
                duration=1.0,
                timestamp=time.time()
            )

            result = await stt_service.transcribe_audio(silence_audio)
            logger.info(f"Silence test successful - Provider: {result.provider}, Confidence: {result.confidence:.3f}")

        except Exception as e:
            logger.warning(f"Silence test failed: {str(e)}")

        # 2. Demonstrate provider fallback
        logger.info("\n2. Demonstrating provider fallback...")
        preferred_provider = config.get_preferred_stt_service()
        logger.info(f"Preferred provider: {preferred_provider}")

        # 3. Show configuration
        logger.info("\n3. Current configuration:")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")

        # 4. Show therapy and crisis keywords
        logger.info("\n4. Therapy keywords monitored:")
        logger.info(f"  Therapy keywords: {stt_service.therapy_keywords[:5]}... (showing first 5)")
        logger.info(f"  Crisis keywords: {stt_service.crisis_keywords}")

        # 5. Performance statistics
        logger.info("\n5. Service statistics:")
        stats = stt_service.get_statistics()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # 6. Test with audio file (if available)
        logger.info("\n6. Testing with audio file (if available)...")
        test_audio_file = Path("./test_audio.wav")
        if test_audio_file.exists():
            try:
                result = await stt_service.transcribe_file(str(test_audio_file))
                logger.info(f"File transcription result:")
                logger.info(f"  Text: '{result.text}'")
                logger.info(f"  Provider: {result.provider}")
                logger.info(f"  Confidence: {result.confidence:.3f}")
                logger.info(f"  Language: {result.language}")
                logger.info(f"  Processing time: {result.processing_time:.3f}s")
                logger.info(f"  Audio quality: {result.audio_quality_score:.3f}")
                logger.info(f"  Therapy keywords: {result.therapy_keywords}")
                logger.info(f"  Crisis keywords: {result.crisis_keywords}")
                logger.info(f"  Sentiment score: {result.sentiment_score:.3f}")
                logger.info(f"  Cached: {result.cached}")
            except Exception as e:
                logger.error(f"File transcription failed: {str(e)}")
        else:
            logger.info("  No test audio file found at ./test_audio.wav")
            logger.info("  To test with a real audio file, place a WAV file at that location")

        # 7. Cache demonstration
        logger.info("\n7. Cache information:")
        logger.info(f"  Cache size: {len(stt_service.cache)} items")
        logger.info(f"  Max cache size: {stt_service.max_cache_size} items")

        # 8. Configuration validation
        logger.info("\n8. Configuration validation:")
        issues = config.validate_configuration()
        if issues:
            logger.warning("  Configuration issues found:")
            for issue in issues:
                logger.warning(f"    - {issue}")
        else:
            logger.info("  Configuration is valid")

        logger.info("\n=== STT Service Demonstration Complete ===")

    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise

def show_usage_instructions():
    """Show usage instructions."""
    logger.info("""
=== STT Service Usage Instructions ===

To use the STT service in your application:

1. Basic Usage:
   ```python
   from voice.stt_service import STTService
   from voice.config import VoiceConfig

   config = VoiceConfig()
   stt_service = STTService(config)

   # Transcribe audio data
   result = await stt_service.transcribe_audio(audio_data)
   print(f"Transcribed text: {result.text}")
   ```

2. Using specific provider:
   ```python
   result = await stt_service.transcribe_audio(audio_data, provider="openai")
   ```

3. Transcribing audio files:
   ```python
   result = await stt_service.transcribe_file("path/to/audio.wav")
   ```

4. Streaming transcription:
   ```python
   async for result in stt_service.transcribe_stream(audio_stream):
       print(f"Transcribed: {result.text}")
   ```

5. Accessing enhanced features:
   ```python
   print(f"Therapy keywords detected: {result.therapy_keywords}")
   print(f"Crisis keywords detected: {result.crisis_keywords}")
   print(f"Audio quality score: {result.audio_quality_score}")
   print(f"Sentiment score: {result.sentiment_score}")
   ```

Configuration:
- OpenAI Whisper API: Set OPENAI_API_KEY in .env
- Google Speech: Set GOOGLE_CLOUD_CREDENTIALS_PATH and GOOGLE_CLOUD_PROJECT_ID
- Local Whisper: Ensure whisper package is installed

Environment Variables:
- OPENAI_WHISPER_MODEL=whisper-1
- OPENAI_WHISPER_LANGUAGE=en
- GOOGLE_SPEECH_LANGUAGE_CODE=en-US
- WHISPER_MODEL=base
- WHISPER_LANGUAGE=en
""")

async def main():
    """Main function."""
    try:
        # Show instructions first
        show_usage_instructions()

        # Run demonstration
        await demonstrate_stt_features()

        logger.info("\nExample completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("\nExample interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)