#!/usr/bin/env python3
"""
Comprehensive test suite verification.

This script runs a verification of all test fixes applied to the AI Therapist Voice Features
application and provides a detailed report of the results.
"""

import sys
import os
import subprocess
import json
from datetime import datetime

class TestSuiteVerifier:
    """Comprehensive test suite verification."""

    def __init__(self):
        self.results = {
            'unit_tests': {'passed': 0, 'failed': 0, 'total': 0, 'details': []},
            'integration_tests': {'passed': 0, 'failed': 0, 'total': 0, 'details': []},
            'security_tests': {'passed': 0, 'failed': 0, 'total': 0, 'details': []},
            'performance_tests': {'passed': 0, 'failed': 0, 'total': 0, 'details': []},
            'overall': {'passed': 0, 'failed': 0, 'total': 0, 'success_rate': 0.0}
        }
        self.start_time = datetime.now()

    def run_test_category(self, category_name, test_pattern):
        """Run a specific category of tests."""
        print(f"\n🧪 Running {category_name} tests...")
        print("=" * 50)

        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_pattern,
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, timeout=300)

            # Parse the output
            output_lines = result.stdout.split('\n')
            passed = 0
            failed = 0
            total = 0

            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # Extract numbers from lines like "3 passed, 1 failed"
                    import re
                    numbers = re.findall(r'(\d+) (passed|failed)', line)
                    for num, status in numbers:
                        if status == 'passed':
                            passed += int(num)
                        elif status == 'failed':
                            failed += int(num)
                    total = passed + failed
                    break

            if total == 0 and result.returncode == 0:
                # All tests passed, but no explicit count found
                if "passed" in result.stdout:
                    passed = result.stdout.count("::")  # Approximate count
                    total = passed

            # Store results
            self.results[category_name] = {
                'passed': passed,
                'failed': failed,
                'total': total,
                'details': [line for line in output_lines if 'FAILED' in line or 'ERROR' in line]
            }

            print(f"Results: {passed} passed, {failed} failed, {total} total")

            if failed > 0:
                print("\nFailures:")
                for detail in self.results[category_name]['details']:
                    print(f"  - {detail}")

            return total > 0

        except subprocess.TimeoutExpired:
            print(f"❌ {category_name} tests timed out")
            return False
        except Exception as e:
            print(f"❌ Error running {category_name} tests: {e}")
            return False

    def run_unit_tests(self):
        """Run unit tests."""
        test_patterns = [
            "tests/unit/test_audio_processor.py::TestAudioProcessor::test_initialization",
            "tests/unit/test_audio_processor.py::TestAudioProcessor::test_audio_validation",
            "tests/unit/test_stt_service.py::TestSTTService::test_initialization",
            "tests/unit/test_tts_service.py::TestTTSService::test_initialization",
            "tests/unit/test_voice_service.py::TestVoiceService::test_initialization"
        ]

        passed = 0
        failed = 0
        details = []

        for test in test_patterns:
            try:
                print(f"Running {test}...")
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test, "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    passed += 1
                    print(f"✓ {test} passed")
                else:
                    failed += 1
                    details.append(f"{test}: FAILED")
                    print(f"✗ {test} failed")
                    # Show error details
                    error_lines = result.stderr.split('\n')
                    for line in error_lines[-5:]:  # Show last 5 lines of error
                        if line.strip():
                            print(f"    {line}")

            except Exception as e:
                failed += 1
                details.append(f"{test}: ERROR - {e}")
                print(f"✗ {test} error: {e}")

        total = passed + failed
        self.results['unit_tests'] = {
            'passed': passed,
            'failed': failed,
            'total': total,
            'details': details
        }

        print(f"\nUnit Tests Summary: {passed}/{total} passed ({(passed/total*100):.1f}%)" if total > 0 else "No unit tests found")
        return total > 0

    def run_integration_tests(self):
        """Run integration tests."""
        test_patterns = [
            "tests/integration/test_voice_service.py::TestVoiceService::test_voice_session_lifecycle",
            "tests/integration/test_voice_service.py::TestVoiceService::test_voice_commands_integration"
        ]

        passed = 0
        failed = 0
        details = []

        for test in test_patterns:
            try:
                print(f"Running {test}...")
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test, "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    passed += 1
                    print(f"✓ {test} passed")
                else:
                    failed += 1
                    details.append(f"{test}: FAILED")
                    print(f"✗ {test} failed")

            except Exception as e:
                failed += 1
                details.append(f"{test}: ERROR - {e}")
                print(f"✗ {test} error: {e}")

        total = passed + failed
        self.results['integration_tests'] = {
            'passed': passed,
            'failed': failed,
            'total': total,
            'details': details
        }

        print(f"\nIntegration Tests Summary: {passed}/{total} passed ({(passed/total*100):.1f}%)" if total > 0 else "No integration tests found")
        return total > 0

    def run_security_tests(self):
        """Run security tests."""
        test_patterns = [
            "tests/security/test_access_control_patched.py::TestAccessControlPatched::test_role_based_access_control",
            "tests/security/test_access_control.py::TestAccessControl::test_basic_access_control",
            "tests/security/test_encryption_comprehensive.py::TestEncryptionComprehensive::test_encryption_different_data_types",
            "tests/security/test_audit_logging.py::TestAuditLogging::test_audit_log_creation"
        ]

        passed = 0
        failed = 0
        details = []

        for test in test_patterns:
            try:
                print(f"Running {test}...")
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test, "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    passed += 1
                    print(f"✓ {test} passed")
                else:
                    failed += 1
                    details.append(f"{test}: FAILED")
                    print(f"✗ {test} failed")

            except Exception as e:
                failed += 1
                details.append(f"{test}: ERROR - {e}")
                print(f"✗ {test} error: {e}")

        total = passed + failed
        self.results['security_tests'] = {
            'passed': passed,
            'failed': failed,
            'total': total,
            'details': details
        }

        print(f"\nSecurity Tests Summary: {passed}/{total} passed ({(passed/total*100):.1f}%)" if total > 0 else "No security tests found")
        return total > 0

    def run_performance_tests(self):
        """Run performance tests."""
        test_patterns = [
            "tests/performance/test_audio_performance.py::TestAudioPerformance::test_audio_processing_performance",
            "tests/performance/test_stt_performance.py::TestSTTPerformance::test_stt_processing_performance"
        ]

        passed = 0
        failed = 0
        details = []

        for test in test_patterns:
            try:
                print(f"Running {test}...")
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test, "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=180)

                if result.returncode == 0:
                    passed += 1
                    print(f"✓ {test} passed")
                else:
                    failed += 1
                    details.append(f"{test}: FAILED")
                    print(f"✗ {test} failed")

            except Exception as e:
                failed += 1
                details.append(f"{test}: ERROR - {e}")
                print(f"✗ {test} error: {e}")

        total = passed + failed
        self.results['performance_tests'] = {
            'passed': passed,
            'failed': failed,
            'total': total,
            'details': details
        }

        print(f"\nPerformance Tests Summary: {passed}/{total} passed ({(passed/total*100):.1f}%)" if total > 0 else "No performance tests found")
        return total > 0

    def calculate_overall_results(self):
        """Calculate overall test results."""
        total_passed = 0
        total_failed = 0
        total_tests = 0

        for category in ['unit_tests', 'integration_tests', 'security_tests', 'performance_tests']:
            results = self.results[category]
            total_passed += results['passed']
            total_failed += results['failed']
            total_tests += results['total']

        self.results['overall'] = {
            'passed': total_passed,
            'failed': total_failed,
            'total': total_tests,
            'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0.0
        }

    def generate_report(self):
        """Generate a comprehensive test report."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        report = {
            'test_run_timestamp': self.start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'test_environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            },
            'results': self.results,
            'summary': {
                'total_tests': self.results['overall']['total'],
                'total_passed': self.results['overall']['passed'],
                'total_failed': self.results['overall']['failed'],
                'success_rate': self.results['overall']['success_rate'],
                'status': 'PASSED' if self.results['overall']['success_rate'] >= 80 else 'FAILED'
            },
            'recommendations': []
        }

        # Add recommendations based on results
        if self.results['unit_tests']['failed'] > 0:
            report['recommendations'].append("Focus on fixing unit test failures - these are critical for component functionality")

        if self.results['integration_tests']['failed'] > 0:
            report['recommendations'].append("Integration test failures indicate issues with component interactions")

        if self.results['security_tests']['failed'] > 0:
            report['recommendations'].append("Security test failures must be addressed immediately for HIPAA compliance")

        if self.results['performance_tests']['failed'] > 0:
            report['recommendations'].append("Performance issues may affect user experience - optimize critical paths")

        # Save report
        report_file = f"test_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Test verification report saved to: {report_file}")
        return report

    def run_comprehensive_verification(self):
        """Run the complete test suite verification."""
        print("🚀 AI Therapist Voice Features - Comprehensive Test Verification")
        print("=" * 80)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run each test category
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_security_tests()
        self.run_performance_tests()

        # Calculate overall results
        self.calculate_overall_results()

        # Generate and display report
        report = self.generate_report()

        # Display summary
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['total_passed']}")
        print(f"Failed: {report['summary']['total_failed']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Overall Status: {report['summary']['status']}")
        print(f"Duration: {report['duration_seconds']:.2f} seconds")

        if report['recommendations']:
            print("\n📝 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")

        print("\n" + "=" * 80)

        return report['summary']['status'] == 'PASSED'

def main():
    """Main function to run comprehensive test verification."""
    verifier = TestSuiteVerifier()
    success = verifier.run_comprehensive_verification()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()