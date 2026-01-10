# Bug Fix Summary - Model Persistence & UI Issues

## Issues Found and Fixed

### 1. ✅ Browser Cache Issue (CRITICAL)
**Problem**: Browser caching old JavaScript files (304 responses) causing old code to run
**Impact**: Model names showing as "undefined", registered models not showing correctly

**Solution**: Add cache-busting or proper cache headers

### 2. ✅ Scan Page Model Dropdown - Undefined Names
**Problem**: Models showing as "undefined (ollama)" in scan page dropdown
**Location**: `aegis/static/js/scan.js` lines 25-26, 35-36

**Root Cause**:
- JavaScript using `model.id` instead of `model.model_id`
- JavaScript using `model.display_name` instead of `model.name`
- API now returns different field names after database migration

**Fixed**: Updated scan.js to use:
```javascript
option.value = model.model_id || model.id;
option.textContent = model.name || model.display_name;
```

### 3. ✅ Ollama Models Showing Register Button After Registration
**Problem**: Register button shown even after model is registered
**Location**: `aegis/static/js/models.js` loadOllamaModels()

**Fixed**: Enhanced loadOllamaModels() to:
- Fetch both Ollama models AND registered models in parallel
- Check if model is already registered
- Show green "Registered" badge instead of Register button

### 4. ✅ Missing model_name in API Response
**Problem**: `/api/models` endpoint not returning `model_name` field
**Location**: `aegis/routes.py` line 131

**Fixed**: Added `model_name` to API response:
```python
"model_name": model["model_name"],
```

## Files Modified

### aegis/static/js/scan.js
**Lines 25-26, 35-36**: Fixed model field references
```javascript
// OLD
option.value = model.id;
option.textContent = model.display_name;

// NEW
option.value = model.model_id || model.id;
option.textContent = model.name || model.display_name;
```

### aegis/static/js/models.js
**Lines 8-60**: Enhanced loadOllamaModels() with registration checking
- Loads registered models in parallel
- Checks if Ollama model is already registered
- Shows appropriate badge/button

### aegis/routes.py
**Line 132**: Added model_name to response
```python
"model_name": model["model_name"],
```

### aegis/database/model_repository.py (NEW)
- Complete CRUD repository for models table
- 308 lines of model persistence logic

### aegis/database/provider_repository.py (NEW)
- Complete CRUD repository for providers table
- 272 lines of provider persistence logic

## How to Verify Fixes

### 1. Clear Browser Cache
**Option A - Hard Refresh**:
- Chrome/Edge: `Ctrl + Shift + R` or `Ctrl + F5`
- Firefox: `Ctrl + Shift + R`
- Safari: `Cmd + Shift + R`

**Option B - Clear Cache**:
- Open Developer Tools (F12)
- Right-click refresh button → "Empty Cache and Hard Reload"

### 2. Check Scan Page
1. Go to http://localhost:5000
2. Look at Models dropdown
3. Should see: "Qwen 2.5 Coder 7B (ollama)" not "undefined (ollama)"

### 3. Check Models Page
1. Go to http://localhost:5000/models
2. Click Ollama tab
3. Registered models should show green "Registered" badge
4. Unregistered models should show blue "Register" button

### 4. Test Model Persistence
1. Register an Ollama model
2. Restart Flask app: `python -m aegis`
3. Model should still be registered and loaded

## Remaining Known Issues

### Browser Cache Problem
**Issue**: Static files (CSS, JS) are being cached with 304 responses
**Impact**: Users don't get updated JavaScript after code changes

**Temporary Fix**:
1. Hard refresh (Ctrl+Shift+R)
2. Clear browser cache
3. Open in incognito/private mode

**Permanent Fix Options**:

#### Option 1: Add Version Query String
Update templates to include version in static file URLs:
```html
<script src="{{ url_for('static', filename='js/models.js', v=config.VERSION) }}"></script>
```

#### Option 2: Add Cache-Control Headers
```python
@main_bp.after_request
def add_header(response):
    if 'static' in request.path:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response
```

#### Option 3: Use Flask-Assets for Cache Busting
Install and configure Flask-Assets to append file hashes to filenames.

## Testing Checklist

- [ ] Scan page model dropdown shows proper names
- [ ] Ollama tab shows registered/unregistered status correctly
- [ ] Can register new Ollama model successfully
- [ ] Can toggle model on/off
- [ ] Can delete registered model
- [ ] Models persist after app restart
- [ ] Models load into registry on startup
- [ ] Scan works with registered models
- [ ] HuggingFace models page works
- [ ] Classic/Cloud models tabs work

## Additional Checks Needed

### 1. Check Model Loading on Startup
```bash
# Look for this in console output
python -m aegis
# Should see: "Initializing database at..."
# Should NOT see: "Warning: Failed to load Ollama model..."
```

### 2. Check Database Integrity
```bash
sqlite3 data/aegis.db "SELECT COUNT(*) FROM models;"
sqlite3 data/aegis.db "SELECT model_id, display_name, enabled FROM models;"
```

### 3. Check API Endpoints
```bash
# Test all model endpoints
curl http://localhost:5000/api/models
curl http://localhost:5000/api/models/ollama
curl http://localhost:5000/api/huggingface/models
```

### 4. Check for JavaScript Errors
1. Open browser console (F12)
2. Navigate to models page
3. Look for any red error messages
4. Check Network tab for failed requests

## Performance Notes

### Model Loading Time
- First run: Downloads models (can take 30s - 5min depending on size)
- Subsequent runs: Loads from cache (instant - few seconds)

### Database Queries
- All model lists are cached in memory (ModelRegistry)
- Database queries only on: registration, deletion, toggle, startup

## Security Notes

### API Keys in Database
- Stored in `config_json` field in models table
- Should be encrypted in production
- Currently stored as plain text (development only)

### Model Deletion
- Requires confirmation dialog
- Cannot delete built-in HuggingFace models
- Cascade deletes findings when scan is deleted

## Next Steps

1. **Immediate**: Clear browser cache and test
2. **Short-term**: Implement cache-busting for static files
3. **Long-term**: Add model versioning and update mechanism
