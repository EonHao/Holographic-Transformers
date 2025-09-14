"""
Test suite for Holographic Transformers.

This module provides a comprehensive test suite for all components
of the Holographic Transformer implementation.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# Add current directory to path for test imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from test_shapes import run_shape_tests
from test_mask import run_mask_tests
from test_invariance import run_invariance_tests
from test_interference import run_interference_tests
from test_gradcheck import run_gradient_tests


def run_all_tests():
    """
    Run all test suites for Holographic Transformers.
    
    This function runs all available tests:
    1. Shape and dtype tests
    2. Padding mask tests
    3. Phase invariance tests
    4. Interference behavior tests
    5. Gradient computation tests
    """
    print("="*80)
    print("HOLOGRAPHIC TRANSFORMER COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    test_suites = [
        ("Shape and Data Type Tests", run_shape_tests),
        ("Padding Mask Tests", run_mask_tests),
        ("Phase Invariance Tests", run_invariance_tests),
        ("Interference Behavior Tests", run_interference_tests),
        ("Gradient Computation Tests", run_gradient_tests),
    ]
    
    passed_suites = 0
    total_suites = len(test_suites)
    
    for suite_name, test_function in test_suites:
        print(f"Running {suite_name}...")
        print("-" * 60)
        
        try:
            test_function()
            passed_suites += 1
            print(f"✓ {suite_name} PASSED")
        except Exception as e:
            print(f"✗ {suite_name} FAILED: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
        
        print()
    
    print("="*80)
    print(f"TEST SUITE SUMMARY: {passed_suites}/{total_suites} test suites passed")
    
    if passed_suites == total_suites:
        print("🎉 ALL TESTS PASSED! The Holographic Transformer implementation is working correctly.")
    else:
        print(f"⚠️  {total_suites - passed_suites} test suite(s) failed. Please check the implementation.")
    
    print("="*80)
    
    return passed_suites == total_suites


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
