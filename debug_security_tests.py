#!/usr/bin/env python3
"""
Quick security test runner to identify failing tests.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run security tests and capture output."""
    print("=== Running Security Tests ===")

    try:
        # Run security tests
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/security/test_security_compliance.py",
            "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=project_root)

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

        if result.returncode != 0:
            print("\n=== Test Failures Identified ===")
            # Try to run individual test methods to isolate failures
            run_individual_tests()
        else:
            print("\n=== All Security Tests Passed ===")

    except Exception as e:
        print(f"Error running tests: {e}")

def run_individual_tests():
    """Run individual test methods to isolate failures."""
    import pytest

    test_methods = [
        "test_security_initialization",
        "test_audit_logging_functionality",
        "test_audit_log_retrieval",
        "test_consent_management",
        "test_data_encryption",
        "test_audio_data_encryption",
        "test_privacy_mode_functionality",
        "test_data_retention_policy",
        "test_security_audit_trail",
        "test_access_control",
        "test_vulnerability_scanning",
        "test_incident_response",
        "test_compliance_reporting",
        "test_backup_and_recovery",
        "test_penetration_testing_preparation",
        "test_security_metrics",
        "test_cleanup"
    ]

    for test_method in test_methods:
        print(f"\n--- Running {test_method} ---")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"tests/security/test_security_compliance.py::TestSecurityCompliance::{test_method}",
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=project_root, timeout=30)

            if result.returncode != 0:
                print(f"FAILED: {test_method}")
                print(result.stderr)
            else:
                print(f"PASSED: {test_method}")

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {test_method}")
        except Exception as e:
            print(f"ERROR: {test_method} - {e}")

if __name__ == "__main__":
    main()