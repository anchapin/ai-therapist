#!/usr/bin/env python3
"""
CI test runner that delegates to the real test runner.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and delegate to the real test runner
from tests.test_runner import main

if __name__ == "__main__":
    sys.exit(main())