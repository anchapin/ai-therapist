#!/usr/bin/env python3
"""
Enhanced Test Runner
Runs the new comprehensive tests with proper error handling and reporting
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test_command(command, description):
    """Run a test command with error handling."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("STDOUT:", result.stdout[:500])
        else:
            print(f"❌ FAILED: {description}")
            print("STDERR:", result.stderr[:500])
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False

def main():
    """Run enhanced test suite."""
    print("🧪 AI Therapist - Enhanced Test Runner")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    test_results = {}
    
    # Test basic functionality first
    basic_tests = [
        ("python3 -m pytest tests/unit/test_memory_manager.py::TestMemoryStats -v", "Memory Manager Basic Tests"),
        ("python3 -m pytest tests/unit/test_app_core.py::TestSecurityFunctions -v", "Security Functions"),
        ("python3 -m pytest tests/unit/test_cache_manager.py -v", "Cache Manager"),
    ]
    
    for command, description in basic_tests:
        test_results[description] = run_test_command(command, description)
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST RUN SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for description, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {description}")
    
    print(f"\nResults: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    elif passed >= total // 2:
        print("⚠️  Some tests failed - check imports and modules")
        return 1
    else:
        print("🚨 Many tests failed - significant issues detected")
        return 2

if __name__ == "__main__":
    sys.exit(main())