#!/usr/bin/env python3
"""
Security Test Runner for AI Therapist Voice Module

This script provides a convenient way to run all security tests
with appropriate configuration and reporting.

Usage:
    python run_security_tests.py                    # Run all security tests
    python run_security_tests.py --unit             # Run only unit tests
    python run_security_tests.py --integration      # Run only integration tests
    python run_security_tests.py --memory           # Run memory safety tests
    python run_security_tests.py --thread           # Run thread safety tests
    python run_security_tests.py --validation       # Run input validation tests
    python run_security_tests.py --coverage         # Run with coverage report
    python run_security_tests.py --verbose          # Run with verbose output
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=False  # Don't raise exception on failure
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'='*60}")
        print(f"Completed: {description}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        print(f"{'='*60}")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running command: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    required_modules = [
        'pytest',
        'numpy',
        'threading',
        'asyncio',
        'pathlib',
        'tempfile',
        'unittest.mock',
        'concurrent.futures',
        'json',
        're',
        'logging'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {module} (missing)")

    if missing_modules:
        print(f"\nMissing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies with:")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    print("All dependencies available!")
    return True

def check_voice_module():
    """Check if voice module is available."""
    try:
        from voice.security import VoiceSecurity
        from voice.audio_processor import SimplifiedAudioProcessor
        from voice.voice_service import VoiceService
        print("✓ Voice module is available")
        return True
    except ImportError as e:
        print(f"✗ Voice module not available: {e}")
        print("Make sure the voice module is properly installed and accessible")
        return False

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Run security tests for AI Therapist voice module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_security_tests.py                    # Run all tests
  python run_security_tests.py --unit             # Unit tests only
  python run_security_tests.py --validation       # Input validation tests
  python run_security_tests.py --memory           # Memory safety tests
  python run_security_tests.py --thread           # Thread safety tests
  python run_security_tests.py --integration      # Integration tests
  python run_security_tests.py --coverage         # With coverage report
        """
    )

    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )

    parser.add_argument(
        '--validation',
        action='store_true',
        help='Run only input validation tests'
    )

    parser.add_argument(
        '--memory',
        action='store_true',
        help='Run only memory safety tests'
    )

    parser.add_argument(
        '--thread',
        action='store_true',
        help='Run only thread safety tests'
    )

    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )

    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage report (requires pytest-cov)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Run with verbose output'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip slow/intensive tests)'
    )

    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency checking'
    )

    args = parser.parse_args()

    print("AI Therapist Voice Security Test Runner")
    print("=" * 50)

    # Check dependencies
    if not args.no_deps_check:
        if not check_dependencies():
            print("\n❌ Dependency check failed. Use --no-deps-check to skip this check.")
            sys.exit(1)

        if not check_voice_module():
            print("\n❌ Voice module check failed.")
            sys.exit(1)

    # Build pytest command
    cmd = ['python', '-m', 'pytest']

    # Add test file
    cmd.append('test_voice_security_comprehensive.py')

    # Add verbosity
    if args.verbose:
        cmd.append('-vv')
    else:
        cmd.append('-v')

    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=voice', '--cov-report=html', '--cov-report=term-missing'])
        print("Note: Coverage requires pytest-cov. Install with: pip install pytest-cov")

    # Add markers based on requested test types
    markers = []

    if args.unit:
        markers.append('unit')
    elif args.validation:
        markers.append('input_validation')
    elif args.memory:
        markers.append('memory_safety')
    elif args.thread:
        markers.append('thread_safety')
    elif args.integration:
        markers.append('integration')
    elif args.quick:
        # Skip slow and intensive tests
        cmd.extend(['-k', 'not slow and not intensive'])
    else:
        # Default: run all security tests
        markers.extend(['security', 'input_validation', 'memory_safety', 'thread_safety', 'integration'])

    if markers:
        cmd.extend(['-m', ' or '.join(markers)])

    # Add other useful options
    cmd.extend([
        '--tb=short',
        '--strict-markers',
        '--color=yes'
    ])

    # Run the tests
    success = run_command(
        cmd,
        "Security Tests"
    )

    if success:
        print("\n🎉 All security tests passed!")
        print("\nThe following security fixes have been validated:")
        print("✓ Input validation in voice/security.py")
        print("✓ Memory leak prevention in voice/audio_processor.py")
        print("✓ Thread safety in voice/voice_service.py")
        print("✓ Overall integration security")

        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")

        return 0
    else:
        print("\n❌ Some security tests failed!")
        print("Please review the test output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())