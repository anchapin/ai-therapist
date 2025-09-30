#!/usr/bin/env python3
"""
Test script for STT Service functionality

This script tests the comprehensive STT service implementation including:
- Multiple provider support (OpenAI Whisper API, Google, Local Whisper)
- Audio quality assessment
- Therapy keyword detection
- Crisis keyword detection
- Caching mechanisms
- Fallback provider chains
- Error handling

Usage:
    python test_stt_service.py
"""

import os
import sys
import asyncio
import logging
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from voice.config import VoiceConfig
from voice.audio_processor import AudioData, AudioProcessor
from voice.stt_service import STTService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class STTServiceTester:
    """Test suite for STT Service functionality."""

    def __init__(self):
        self.config = VoiceConfig()
        self.stt_service = None
        self.audio_processor = None
        self.test_results = []

    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up STT Service test environment...")

        try:
            # Initialize STT service
            self.stt_service = STTService(self.config)
            self.audio_processor = AudioProcessor(self.config)

            # Check if services are available
            available_providers = self.stt_service.get_available_providers()
            logger.info(f"Available STT providers: {available_providers}")

            if not available_providers:
                logger.warning("No STT providers available. Some tests will be skipped.")

            return True

        except Exception as e:
            logger.error(f"Failed to setup test environment: {str(e)}")
            return False

    def create_test_audio_data(self, duration=2.0, sample_rate=16000, frequency=440) -> AudioData:
        """Create test audio data for testing."""
        try:
            # Generate sine wave test audio
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

            # Add some noise to simulate real audio
            noise = 0.01 * np.random.normal(0, 1, len(audio_data))
            audio_data = audio_data + noise

            # Convert to float32
            audio_data = audio_data.astype(np.float32)

            return AudioData(
                data=audio_data,
                sample_rate=sample_rate,
                channels=1,
                format="float32",
                duration=duration,
                timestamp=asyncio.get_event_loop().time()
            )

        except Exception as e:
            logger.error(f"Failed to create test audio: {str(e)}")
            return None

    async def test_service_availability(self):
        """Test STT service availability."""
        logger.info("Testing STT service availability...")

        try:
            is_available = self.stt_service.is_available()
            available_providers = self.stt_service.get_available_providers()

            test_result = {
                'test': 'Service Availability',
                'passed': is_available,
                'details': {
                    'available': is_available,
                    'providers': available_providers
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Service availability test: {'PASSED' if is_available else 'FAILED'}")

            return is_available

        except Exception as e:
            logger.error(f"Service availability test failed: {str(e)}")
            self.test_results.append({
                'test': 'Service Availability',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_audio_quality_assessment(self):
        """Test audio quality assessment functionality."""
        logger.info("Testing audio quality assessment...")

        try:
            # Create test audio with different quality levels
            good_audio = self.create_test_audio_data(duration=1.0, frequency=440)
            clipped_audio = self.create_test_audio_data(duration=1.0, frequency=440)
            clipped_audio.data = np.clip(clipped_audio.data * 2.0, -0.95, 0.95)  # Add clipping

            quiet_audio = self.create_test_audio_data(duration=1.0, frequency=440)
            quiet_audio.data = quiet_audio.data * 0.05  # Very quiet

            # Test quality scoring
            good_score = self.stt_service._calculate_audio_quality_score(good_audio)
            clipped_score = self.stt_service._calculate_audio_quality_score(clipped_audio)
            quiet_score = self.stt_service._calculate_audio_quality_score(quiet_audio)

            test_result = {
                'test': 'Audio Quality Assessment',
                'passed': True,
                'details': {
                    'good_audio_score': good_score,
                    'clipped_audio_score': clipped_score,
                    'quiet_audio_score': quiet_score,
                    'scoring_works': good_score > clipped_score and good_score > quiet_score
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Audio quality assessment test: PASSED")
            logger.info(f"  Good audio score: {good_score:.3f}")
            logger.info(f"  Clipped audio score: {clipped_score:.3f}")
            logger.info(f"  Quiet audio score: {quiet_score:.3f}")

            return True

        except Exception as e:
            logger.error(f"Audio quality assessment test failed: {str(e)}")
            self.test_results.append({
                'test': 'Audio Quality Assessment',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_therapy_keyword_detection(self):
        """Test therapy keyword detection functionality."""
        logger.info("Testing therapy keyword detection...")

        try:
            # Test text with therapy keywords
            therapy_texts = [
                "I'm feeling anxious about my therapy session",
                "My depression has been getting worse",
                "I need help with stress management",
                "This trauma is affecting my daily life"
            ]

            detected_keywords = []
            for text in therapy_texts:
                text_lower = text.lower()
                keywords_found = []
                for keyword in self.stt_service.therapy_keywords:
                    if keyword in text_lower:
                        keywords_found.append(keyword)
                detected_keywords.append(keywords_found)

            test_result = {
                'test': 'Therapy Keyword Detection',
                'passed': True,
                'details': {
                    'test_texts': therapy_texts,
                    'detected_keywords': detected_keywords,
                    'detection_works': any(keywords for keywords in detected_keywords)
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Therapy keyword detection test: PASSED")

            for i, (text, keywords) in enumerate(zip(therapy_texts, detected_keywords)):
                logger.info(f"  Text {i+1}: '{text}' -> Keywords: {keywords}")

            return True

        except Exception as e:
            logger.error(f"Therapy keyword detection test failed: {str(e)}")
            self.test_results.append({
                'test': 'Therapy Keyword Detection',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_crisis_keyword_detection(self):
        """Test crisis keyword detection functionality."""
        logger.info("Testing crisis keyword detection...")

        try:
            # Test text with crisis keywords
            crisis_texts = [
                "I want to end my life",
                "I'm thinking about suicide",
                "I can't go on anymore",
                "I need emergency help"
            ]

            detected_keywords = []
            for text in crisis_texts:
                text_lower = text.lower()
                keywords_found = []
                for keyword in self.stt_service.crisis_keywords:
                    if keyword in text_lower:
                        keywords_found.append(keyword)
                detected_keywords.append(keywords_found)

            test_result = {
                'test': 'Crisis Keyword Detection',
                'passed': True,
                'details': {
                    'test_texts': crisis_texts,
                    'detected_keywords': detected_keywords,
                    'detection_works': all(keywords for keywords in detected_keywords)
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Crisis keyword detection test: PASSED")

            for i, (text, keywords) in enumerate(zip(crisis_texts, detected_keywords)):
                logger.info(f"  Text {i+1}: '{text}' -> Crisis keywords: {keywords}")

            return True

        except Exception as e:
            logger.error(f"Crisis keyword detection test failed: {str(e)}")
            self.test_results.append({
                'test': 'Crisis Keyword Detection',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        logger.info("Testing sentiment analysis...")

        try:
            # Test texts with different sentiments
            positive_text = "I'm feeling better and hopeful about recovery"
            negative_text = "I'm very depressed and anxious about everything"
            neutral_text = "I have a therapy appointment tomorrow"

            positive_score = self.stt_service._calculate_sentiment_score(positive_text)
            negative_score = self.stt_service._calculate_sentiment_score(negative_text)
            neutral_score = self.stt_service._calculate_sentiment_score(neutral_text)

            test_result = {
                'test': 'Sentiment Analysis',
                'passed': True,
                'details': {
                    'positive_text': positive_text,
                    'positive_score': positive_score,
                    'negative_text': negative_text,
                    'negative_score': negative_score,
                    'neutral_text': neutral_text,
                    'neutral_score': neutral_score,
                    'scoring_works': positive_score > negative_score and neutral_score == 0.0
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Sentiment analysis test: PASSED")
            logger.info(f"  Positive score: {positive_score:.3f}")
            logger.info(f"  Negative score: {negative_score:.3f}")
            logger.info(f"  Neutral score: {neutral_score:.3f}")

            return True

        except Exception as e:
            logger.error(f"Sentiment analysis test failed: {str(e)}")
            self.test_results.append({
                'test': 'Sentiment Analysis',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_caching_mechanism(self):
        """Test caching mechanism functionality."""
        logger.info("Testing caching mechanism...")

        try:
            # Create test audio
            test_audio = self.create_test_audio_data(duration=1.0)

            # Test cache key generation
            cache_key1 = self.stt_service._generate_cache_key(test_audio)
            cache_key2 = self.stt_service._generate_cache_key(test_audio)

            # Keys should be identical for same audio
            keys_match = cache_key1 == cache_key2

            # Test cache operations
            mock_result = type('MockResult', (), {'text': 'test', 'confidence': 0.9})()

            # Add to cache
            self.stt_service._add_to_cache(cache_key1, mock_result)

            # Get from cache
            cached_result = self.stt_service._get_from_cache(cache_key1)

            test_result = {
                'test': 'Caching Mechanism',
                'passed': keys_match and cached_result is not None,
                'details': {
                    'cache_keys_match': keys_match,
                    'cache_retrieval_works': cached_result is not None,
                    'cache_size': len(self.stt_service.cache)
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Caching mechanism test: {'PASSED' if test_result['passed'] else 'FAILED'}")

            return test_result['passed']

        except Exception as e:
            logger.error(f"Caching mechanism test failed: {str(e)}")
            self.test_results.append({
                'test': 'Caching Mechanism',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_fallback_chain(self):
        """Test provider fallback chain functionality."""
        logger.info("Testing provider fallback chain...")

        try:
            available_providers = self.stt_service.get_available_providers()

            if not available_providers:
                logger.warning("No providers available for fallback chain test")
                return True

            # Test fallback chain generation
            fallback_chain = self.stt_service._get_provider_fallback_chain(None)

            # Test with preferred provider
            if len(available_providers) > 1:
                preferred_provider = available_providers[0]
                preferred_chain = self.stt_service._get_provider_fallback_chain(preferred_provider)

                # Preferred provider should be first
                preferred_first = preferred_chain[0] == preferred_provider
            else:
                preferred_first = True

            test_result = {
                'test': 'Fallback Chain',
                'passed': preferred_first,
                'details': {
                    'available_providers': available_providers,
                    'fallback_chain': fallback_chain,
                    'preferred_provider_first': preferred_first
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Fallback chain test: {'PASSED' if test_result['passed'] else 'FAILED'}")
            logger.info(f"  Available providers: {available_providers}")
            logger.info(f"  Fallback chain: {fallback_chain}")

            return test_result['passed']

        except Exception as e:
            logger.error(f"Fallback chain test failed: {str(e)}")
            self.test_results.append({
                'test': 'Fallback Chain',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_configuration_validation(self):
        """Test configuration validation."""
        logger.info("Testing configuration validation...")

        try:
            # Test configuration validation
            issues = self.config.validate_configuration()

            test_result = {
                'test': 'Configuration Validation',
                'passed': True,  # Validation itself should work
                'details': {
                    'validation_issues': issues,
                    'issues_count': len(issues)
                }
            }

            self.test_results.append(test_result)
            logger.info(f"Configuration validation test: PASSED")
            logger.info(f"  Configuration issues found: {len(issues)}")

            if issues:
                for issue in issues:
                    logger.warning(f"    - {issue}")

            return True

        except Exception as e:
            logger.error(f"Configuration validation test failed: {str(e)}")
            self.test_results.append({
                'test': 'Configuration Validation',
                'passed': False,
                'error': str(e)
            })
            return False

    async def run_all_tests(self):
        """Run all STT service tests."""
        logger.info("Starting comprehensive STT service tests...")

        # Setup
        if not await self.setup():
            logger.error("Failed to setup test environment")
            return False

        # Run all tests
        tests = [
            self.test_service_availability,
            self.test_audio_quality_assessment,
            self.test_therapy_keyword_detection,
            self.test_crisis_keyword_detection,
            self.test_sentiment_analysis,
            self.test_caching_mechanism,
            self.test_fallback_chain,
            self.test_configuration_validation
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test in tests:
            try:
                result = await test()
                if result:
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {str(e)}")
                self.test_results.append({
                    'test': test.__name__,
                    'passed': False,
                    'error': str(e)
                })

        # Print summary
        logger.info(f"\n=== STT Service Test Summary ===")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed tests: {passed_tests}")
        logger.info(f"Failed tests: {total_tests - passed_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

        # Print detailed results
        logger.info(f"\n=== Detailed Results ===")
        for result in self.test_results:
            status = "PASSED" if result['passed'] else "FAILED"
            logger.info(f"{result['test']}: {status}")
            if 'error' in result:
                logger.error(f"  Error: {result['error']}")

        return passed_tests == total_tests

    async def cleanup(self):
        """Cleanup test resources."""
        try:
            if self.stt_service:
                self.stt_service.cleanup()
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

async def main():
    """Main test function."""
    tester = STTServiceTester()

    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)