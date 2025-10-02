#!/usr/bin/env python3
"""
Run the access control test with the fix applied.
"""

import sys
import os
import subprocess

def main():
    """Run the access control test with the fix."""
    print("🧪 Running Access Control Test with Fix")
    print("=" * 50)

    # Apply the fix first
    try:
        import test_fix_access_control
        test_fix_access_control.apply_access_control_patch()
        print("✓ Access control patch applied")
    except Exception as e:
        print(f"✗ Failed to apply patch: {e}")
        return False

    # Run the specific test
    test_file = "tests/security/test_access_control.py::TestAccessControl::test_role_based_access_control"

    try:
        print(f"Running {test_file}...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=60)

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✅ Test passed successfully!")
            return True
        else:
            print("❌ Test failed")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)