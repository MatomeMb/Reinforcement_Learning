#!/usr/bin/env python3
"""
Quick test script to verify all scenarios work correctly.
Run this before submission to ensure everything functions.
"""

import subprocess
import sys
import os
import time

def run_test(command, description):
    """Run a test command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({elapsed:.1f}s)")
            print("Output highlights:")
            lines = result.stdout.split('\n')
            for line in lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"❌ FAILED ({elapsed:.1f}s)")
            print("Error output:")
            print(result.stderr[-500:])  # Show last 500 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (120s limit exceeded)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True

def main():
    """Run comprehensive functionality tests."""
    print("🚀 QUICK FUNCTIONALITY TEST FOR RL ASSIGNMENT")
    print("This will test all scenarios to ensure they work correctly.")
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    tests = [
        # Basic functionality tests (fast)
        ("python Scenario1.py -episodes 100", "Scenario 1 Basic"),
        ("python Scenario2.py -episodes 100", "Scenario 2 Basic"),
        ("python Scenario3.py -episodes 150", "Scenario 3 Basic"),
        
        # Stochastic tests
        ("python Scenario1.py -stochastic -episodes 50", "Scenario 1 Stochastic"),
        ("python Scenario2.py -stochastic -episodes 50", "Scenario 2 Stochastic"),
        ("python Scenario3.py -stochastic -episodes 50", "Scenario 3 Stochastic"),
        
        # Progress bar test
        ("python Scenario1.py -episodes 200 -show_progress", "Scenario 1 with Progress"),
    ]
    
    passed = 0
    total = len(tests)
    
    for command, description in tests:
        if run_test(command, description):
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your implementation is ready for submission.")
        
        # Check for required files
        required_files = [
            'Scenario1.py', 'Scenario2.py', 'Scenario3.py',
            'FourRooms.py', 'requirements.txt', 'README.md'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"⚠️  Missing files: {missing_files}")
        else:
            print("✅ All required files present")
            
        # Check for results
        result_files = [f for f in os.listdir('results') if f.endswith('.png')]
        print(f"📊 Generated {len(result_files)} visualization files")
        
        print("\n🚀 READY FOR SUBMISSION!")
        
    else:
        print(f"❌ {total - passed} tests failed. Please fix issues before submission.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())