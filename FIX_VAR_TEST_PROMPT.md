# Fix VaR Test Issue in test_advanced_features.py

## Problem Description

The test_advanced_features.py file is currently at 80% pass rate due to a failing VaR (Value at Risk) test. The specific issue is in the `test_var_calculation` function where there's an assertion error.

## Error Details

```python
# Current failing test (line 369 in test_advanced_features.py):
assert var_95 > 0  # This assertion fails
```

The test expects VaR to be positive, but VaR (Value at Risk) is typically expressed as a negative number in financial risk management, representing the maximum expected loss.

## Required Fix

Please fix the VaR test assertion in test_advanced_features.py to properly handle the expected negative VaR values. The fix should:

1. Locate the test_var_calculation function in test_advanced_features.py
2. Change the assertion from `assert var_95 > 0` to `assert var_95 < 0`
3. Ensure the test properly validates that VaR is calculating losses (negative returns)
4. Verify that the fix maintains the financial accuracy of the VaR calculation

## Expected Outcome

After applying this fix:
- test_advanced_features.py should pass 100% of tests
- The VaR calculation should correctly represent potential losses as negative values
- All other tests should remain unaffected

## File Location

- File to modify: `test_advanced_features.py`
- Function to fix: `test_var_calculation`
- Line number: Around line 369

## Verification

After making the change, run:
```bash
python test_advanced_features.py
```

The output should show all tests passing with no assertion errors.
