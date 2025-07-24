# Prompt to Fix Financial Wavelet Prediction Project

Copy and paste this prompt into a new chat window:

---

**Fix the remaining issues in the Financial Wavelet Prediction project**

I have a financial prediction system that's 75% functional but needs some fixes to be fully operational. The project is located at: `c:/Users/aaron/AppData/Local/Programs/AppData/Local/Programs/Microsoft VS Code/financial_wavelet_prediction`

Please fix the following issues identified during testing:

## 1. XGBoost Model Issue (Priority: High)
**File**: `src/models/xgboost_predictor.py`
**Error**: `XGBModel.fit() got an unexpected keyword argument 'eval_metric'`
**Fix**: Remove or update the eval_metric parameter in the fit() method to match the current XGBoost API

## 2. River Drift Detection API Issue (Priority: High)
**File**: `src/advanced/adaptive_learner.py`
**Error**: `AttributeError: module 'river.drift' has no attribute 'DDM'`
**Fix**: Update the drift detection to use the current river API. The DDM class may have been moved or renamed. Check river documentation for the correct import.

## 3. ModelCompressor Abstract Class Issue (Priority: Medium)
**File**: `src/optimization/model_compressor.py`
**Error**: Cannot instantiate abstract class ModelCompressor
**Fix**: Either:
- Create a concrete implementation of ModelCompressor, or
- Remove the abstract methods and provide default implementations

## 4. Minor Issues to Address (Priority: Low)
- Update matplotlib style from 'seaborn' to 'seaborn-v0_8' in visualization files
- Fix the BacktestConfig import in `test_api.py` (should import from `src.evaluation.backtest_engine`)
- Address FutureWarnings about numpy dtype compatibility

## 5. Update Requirements File
After fixing the issues, update `requirements.txt` to include:
- hmmlearn
- river
- cvxpy

## Expected Outcome
After these fixes, all tests should pass:
- `python test_dtw_engine.py` 
- `python test_feature_engineering.py`
- `python test_models.py`
- `python test_evaluation.py`
- `python test_api.py`
- `python test_advanced_features.py`

The test results summary is available in `test_results_summary.md` for reference.

Please make these fixes and verify that all components are working correctly.

---

This prompt provides all the context needed to fix the remaining issues in the project.
