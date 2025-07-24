# YFinance MVP Sprint Plan

## Overview
Minimal code changes to transition from Polygon/synthetic data to YFinance with SQLite storage. Windows PowerShell compatible. No verbose additions - only essential changes.

---

## Sprint 1: Remove Synthetic Data Dependencies

### Goal
Strip out all synthetic data generation. Application should fail gracefully when no real data is available.

### Actions

1. **Remove mock data generation from `data_utils_polygon.py`**
   - Delete the `_generate_mock_data()` method entirely
   - Remove all fallback logic that generates synthetic data
   - When API fails, return `None` or empty DataFrame

2. **Remove synthetic fallback from key files**
   ```
   src/dashboard/data_utils_polygon.py     - Remove _generate_mock_data()
   src/dashboard/tools/pattern_explorer.py  - Remove synthetic statistics
   src/dashboard/visualizations/analytics.py - Remove create_sample_data()
   src/dashboard/components/sidebar.py      - Remove synthetic data fallback
   ```

3. **Clean model files**
   - Remove all "Generate synthetic data" code blocks from:
     - `src/models/ensemble_model.py`
     - `src/models/model_evaluator.py`
     - `src/models/model_trainer.py`
     - `src/models/sequence_predictor.py`
     - `src/models/transformer_predictor.py`
     - `src/models/xgboost_predictor.py`

4. **Leave tests/demos unchanged** (not critical for MVP)

### Result
Application returns empty/None when no real data available. No synthetic data pollution.

---

## Sprint 2: YFinance Demo with Smart Backoff

### Goal
Create minimal YFinance fetcher with exponential backoff and SQLite storage.

### Actions

1. **Create `yfinance_mvp.py`** with:
   ```python
   # Core features:
   - Exponential backoff: 1s → 2s → 4s → 8s → 16s
   - Rate limit detection (HTTP 429)
   - SQLite cache check before API call
   - Simple error handling
   ```

2. **Minimal database schema**
   ```sql
   CREATE TABLE IF NOT EXISTS price_data (
       ticker TEXT,
       date TEXT,
       open REAL,
       high REAL,
       low REAL,
       close REAL,
       volume INTEGER,
       timestamp INTEGER,
       PRIMARY KEY (ticker, date)
   )
   ```

3. **Test with single ticker**
   - Fetch AAPL for 1 month
   - Verify data stored in SQLite
   - Test backoff on rate limit

### Result
Proven YFinance fetcher that handles rate limits and persists data.

---

## Sprint 3: Replace Data Layer

### Goal
Swap Polygon for YFinance throughout the application. Minimal changes only.

### Actions

1. **Create `src/dashboard/data_utils_yfinance.py`**
   - Copy structure from `data_utils_polygon.py`
   - Replace Polygon API calls with YFinance
   - Use backoff logic from Sprint 2
   - No synthetic fallbacks - return None on failure

2. **Update import in `src/dashboard/data_utils.py`**
   ```python
   # Change from:
   from .data_utils_polygon import *
   # To:
   from .data_utils_yfinance import *
   ```

3. **Key replacement points**
   - Polygon API endpoints → YFinance download
   - Mock data fallbacks → Return None
   - Polygon cache → YFinance SQLite cache
   - API key handling → Remove (YFinance needs no key)

4. **Update components to handle None gracefully**
   - Pattern matcher/classifier
   - Visualizations
   - Dashboard components
   - Add simple checks: `if data is None or data.empty: return`

### Result
Dashboard runs exclusively on real YFinance data. No synthetic fallbacks.

---

## Implementation Notes

### Windows PowerShell Compatibility
- No `&&` command chaining
- Use `;` for sequential commands
- Example: `cd src; python yfinance_mvp.py`

### Minimal Code Philosophy
- Don't add features not explicitly needed
- Remove code rather than comment it out
- Simple error handling - just return None/empty
- No verbose logging or extra abstractions

### Testing Each Sprint
- **Sprint 1**: Run app, verify it shows "No data" messages
- **Sprint 2**: Run demo script, check SQLite has data
- **Sprint 3**: Run dashboard, verify real YFinance data displays

---

## Success Criteria

1. **No synthetic data** anywhere in production code
2. **YFinance integration** with proper rate limiting
3. **SQLite persistence** for all fetched data
4. **Dashboard works** with real market data only
5. **Graceful failures** when data unavailable
