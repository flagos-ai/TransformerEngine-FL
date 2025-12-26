#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Vendor plugin tests runner.

This script runs all vendor plugin integration tests:
- In-tree plugin tests
- Out-of-tree plugin tests

Usage:
    python run_vendor_tests.py              # Run all vendor tests
    python run_vendor_tests.py --intree     # Run only in-tree tests
    python run_vendor_tests.py --outtree    # Run only out-of-tree tests
    python run_vendor_tests.py --verbose    # Run with verbose output
"""

import argparse
import sys
import unittest
from pathlib import Path


def run_intree_tests(verbosity=2):
    """Run in-tree vendor plugin tests"""
    print("=" * 70)
    print("Running In-tree Vendor Plugin Tests")
    print("=" * 70)

    try:
        from test_vendor_intree import run_tests
        return run_tests()
    except ImportError as e:
        print(f"Error importing in-tree tests: {e}")
        return 1


def run_outtree_tests(verbosity=2):
    """Run out-of-tree vendor plugin tests"""
    print("=" * 70)
    print("Running Out-of-tree Vendor Plugin Tests")
    print("=" * 70)

    try:
        from test_vendor_outtree import run_tests
        return run_tests()
    except ImportError as e:
        print(f"Error importing out-of-tree tests: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run vendor plugin integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                 # Run all vendor tests
  %(prog)s --intree        # Run only in-tree tests
  %(prog)s --outtree       # Run only out-of-tree tests
  %(prog)s -v              # Run with verbose output
        """
    )

    parser.add_argument(
        "--intree",
        action="store_true",
        help="Run only in-tree plugin tests"
    )

    parser.add_argument(
        "--outtree",
        action="store_true",
        help="Run only out-of-tree plugin tests"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    # Add current directory to path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    results = []

    # Determine which tests to run
    run_all = not (args.intree or args.outtree)

    if run_all or args.intree:
        print("\n" + "=" * 70)
        print("IN-TREE VENDOR PLUGIN TESTS")
        print("=" * 70 + "\n")
        result = run_intree_tests(verbosity)
        results.append(("In-tree", result))

    if run_all or args.outtree:
        print("\n" + "=" * 70)
        print("OUT-OF-TREE VENDOR PLUGIN TESTS")
        print("=" * 70 + "\n")
        result = run_outtree_tests(verbosity)
        results.append(("Out-of-tree", result))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result == 0 else "❌ FAILED"
        print(f"{name:20s}: {status}")
        if result != 0:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 All vendor plugin tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
