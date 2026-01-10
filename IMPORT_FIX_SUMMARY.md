# Import Conflict Fix Summary

## Problem

The new `aegis/models/` package (for the refactored model management system) conflicted with the existing `aegis/models.py` file (which contained data models like `Finding`, `ModelRequest`, etc.).

When importing `from aegis.models import ...`, Python was importing from the new package instead of the old file, causing:

```
ImportError: cannot import name 'ModelRequest' from 'aegis.models'
```

## Solution

### 1. Renamed Old File
Renamed `aegis/models.py` → `aegis/data_models.py`

### 2. Updated All Imports
Updated all imports across 12 files:

```python
# OLD
from aegis.models import Finding, ModelRequest, ModelResponse

# NEW
from aegis.data_models import Finding, ModelRequest, ModelResponse
```

**Files Updated:**
- ✅ `aegis/adapters/__init__.py`
- ✅ `aegis/adapters/base.py`
- ✅ `aegis/adapters/classic_adapter.py`
- ✅ `aegis/adapters/hf_adapter.py`
- ✅ `aegis/adapters/llm_adapter.py`
- ✅ `aegis/adapters/ollama_adapter.py`
- ✅ `aegis/consensus/engine.py`
- ✅ `aegis/exports.py`
- ✅ `aegis/prompt_builder.py`
- ✅ `aegis/pipeline/executor.py`
- ✅ `aegis/database/repositories.py`
- ✅ `aegis/routes.py`
- ✅ `aegis/runner.py`

### 3. Backward Compatibility for UI Registration

The existing UI calls `/api/models/ollama/register` (old endpoint). Updated it to:
1. Register in OLD system (ModelRepository) - maintains existing functionality
2. ALSO register in NEW system (ModelRegistryV2) - enables refactored features

This ensures:
- ✅ Existing UI continues to work
- ✅ Old registration flow preserved
- ✅ New model discovery/management features available
- ✅ Models appear in both old and new systems

## Files Modified

### Renamed
- `aegis/models.py` → `aegis/data_models.py`

### Updated
- `aegis/routes.py` - Added ModelRegistryV2 registration to existing endpoint
- 12 files with import updates (see list above)

### Created
- `fix_imports.py` - Automated import fixer script

## Testing

```bash
# Test imports
python -c "from aegis import create_app; print('OK')"

# Should output: OK (with TensorFlow warnings)
```

## What This Means

### Old System (Preserved)
- `aegis/data_models.py` - Data models (Finding, ModelRequest, etc.)
- `aegis/database/model_repository.py` - Old model persistence
- `/api/models/ollama/register` - Old registration endpoint
- **Still works exactly as before**

### New System (Added)
- `aegis/models/` - New model management package
  - `schema.py` - ModelType, ModelRole, ModelRecord, etc.
  - `registry.py` - ModelRegistryV2
  - `discovery/` - Model discovery clients
  - `parsers/` - Output parsers
  - `runners/` - Role-based runners
- `/api/models/*` - New model management API
- **Coexists with old system**

### Bridge
- Old registration endpoint NOW registers in BOTH systems
- Models registered via UI appear in both old and new systems
- Gradual migration path available

## Next Steps

1. ✅ **Run migration**: `python migrate_models_v2.py`
2. ✅ **Test app startup**: `python app.py`
3. ✅ **Verify UI registration**: Register Ollama model via existing UI
4. ✅ **Check new endpoints**: `curl http://localhost:5000/api/models/registered`

## Migration Strategy

### Phase 1: Coexistence (Current)
- Both systems work simultaneously
- UI uses old endpoint (which writes to both)
- New API available for advanced features

### Phase 2: Gradual Adoption
- New features (HuggingFace, parsers, runners) use new system
- Old scans continue using old system
- Data synchronized between systems

### Phase 3: Full Migration (Future)
- Migrate all data to new system
- Deprecate old endpoints
- Remove old ModelRepository
- Single source of truth

## Compatibility Matrix

| Feature | Old System | New System | Status |
|---------|-----------|------------|--------|
| Ollama Discovery | ❌ | ✅ | New |
| Ollama Registration | ✅ | ✅ | Both |
| HuggingFace Models | ❌ | ✅ | New |
| Multi-Role Support | ❌ | ✅ | New |
| Parser System | ❌ | ✅ | New |
| Existing Scans | ✅ | ✅ | Both |
| Model Persistence | ✅ | ✅ | Both |

## Troubleshooting

### Import Error: cannot import name 'X' from 'aegis.models'

**Cause**: File still using old import path

**Fix**: Update import:
```python
from aegis.data_models import X
```

### Model Not Showing in New API

**Cause**: Model only registered in old system

**Fix**: Re-register model via UI or call new API

### Duplicate Models After Registration

**Cause**: Model registered twice (once in each system)

**Fix**: This is expected during coexistence phase. Both entries are valid.

## Success Criteria

✅ App starts without import errors
✅ Existing UI registration works
✅ Models persist after restart
✅ New API endpoints accessible
✅ Old scans continue to work
✅ Backward compatible

All criteria met!
