#!/usr/bin/env python3
"""
Quick validation script for FALCON v3 implementation.
Tests imports and basic functionality without running full training.
"""

import sys
import traceback

def test_falcon_import():
    """Test FALCON v3 can be imported."""
    try:
        from optim.falcon import FALCON, falcon_filter_grad
        print("✓ FALCON v3 import successful")
        return True
    except Exception as e:
        print(f"✗ FALCON v3 import failed: {e}")
        traceback.print_exc()
        return False

def test_falcon_instantiation():
    """Test FALCON v3 can be instantiated."""
    try:
        import torch
        from optim.falcon import FALCON
        
        # Create dummy parameters
        params = [torch.randn(64, 32, 3, 3, requires_grad=True)]
        
        # Instantiate with v3 defaults
        opt = FALCON(
            params, 
            lr=3e-4, 
            weight_decay=5e-4,
            rank1_backend="poweriter",
            poweriter_steps=1,
            mask_interval=5,
            apply_stages=[3, 4],
            skip_mix_start=0.0,
            skip_mix_end=0.7,
            retain_energy_start=0.90,
            retain_energy_end=0.60,
            rank_k=1,
            late_rank_k_epoch=40,
        )
        
        print("✓ FALCON v3 instantiation successful")
        print(f"  - rank1_backend: {opt.rank1_backend}")
        print(f"  - mask_interval: {opt.mask_interval}")
        print(f"  - apply_stages: {opt.apply_stages}")
        return True
    except Exception as e:
        print(f"✗ FALCON v3 instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_train_imports():
    """Test train.py can be imported."""
    try:
        # Just check syntax by compiling
        with open('train.py', 'r') as f:
            code = f.read()
        compile(code, 'train.py', 'exec')
        print("✓ train.py syntax valid")
        return True
    except Exception as e:
        print(f"✗ train.py syntax error: {e}")
        traceback.print_exc()
        return False

def test_plot_script():
    """Test plot_results.py can be imported."""
    try:
        with open('scripts/plot_results.py', 'r') as f:
            code = f.read()
        compile(code, 'scripts/plot_results.py', 'exec')
        print("✓ plot_results.py syntax valid")
        return True
    except Exception as e:
        print(f"✗ plot_results.py syntax error: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print(" FALCON v3 Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("FALCON import", test_falcon_import),
        ("FALCON instantiation", test_falcon_instantiation),
        ("train.py syntax", test_train_imports),
        ("plot_results.py syntax", test_plot_script),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f" Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(" ✓ All tests passed! FALCON v3 is ready.")
    else:
        print(f" ✗ {total - passed} test(s) failed.")
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
