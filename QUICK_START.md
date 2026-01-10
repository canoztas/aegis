# Quick Start - Model Management Refactor

## 1. Fix Applied ✅

The import conflict has been resolved:
- ✅ Renamed `models.py` → `data_models.py`
- ✅ Updated all imports (12 files)
- ✅ Backward compatibility maintained
- ✅ App starts successfully

## 2. Apply Database Migration

```bash
python migrate_models_v2.py
```

Expected output:
```
============================================================
Aegis Model Registry V2 Migration
============================================================

Database: data/aegis.db

[OK] Migration 003_model_registry_v2 applied successfully
[OK] All required columns present: {'roles_json', 'parser_id', 'model_type', 'status'}
[OK] Created huggingface provider

[SUCCESS] Migration completed successfully!
```

## 3. Start the Application

```bash
python app.py
```

The app will start on http://localhost:5000 (or http://localhost:7766 based on config)

## 4. Use Existing UI (No Changes Required)

Your existing UI works as before! When you register an Ollama model:
1. It registers in the OLD system (existing functionality)
2. It ALSO registers in the NEW system (new features enabled)

**No UI changes needed** - backward compatible!

## 5. Try New API Endpoints (Optional)

### Discover Ollama Models
```bash
curl http://localhost:5000/api/models/discovered/ollama
```

### List Registered Models (New)
```bash
curl http://localhost:5000/api/models/registered
```

### Register HuggingFace Model (New Feature)
```bash
curl -X POST http://localhost:5000/api/models/hf/register_preset \
  -H "Content-Type: application/json" \
  -d '{"preset_id": "codebert_insecure"}'
```

## 6. Verify Everything Works

### Test 1: Check App Imports
```bash
python -c "from aegis import create_app; print('✅ Imports OK')"
```

### Test 2: Register Ollama Model via UI
1. Go to http://localhost:5000/models
2. Click "Add Ollama" tab
3. Click "Register" on any model
4. Should see success message

### Test 3: Verify Persistence
```bash
# Check old system
sqlite3 data/aegis.db "SELECT model_id, display_name FROM models;"

# Should show your registered models
```

## What's Different?

### For Users
- **Nothing!** Existing UI works exactly the same
- **Bonus**: New models page has cleaner "My Models" view
- **Bonus**: HuggingFace models now available

### For Developers
- New model management API available
- Discovery system for Ollama models
- Parser architecture for custom outputs
- Role-based runner system
- Config-first model additions

## Architecture Overview

```
OLD SYSTEM (Still Works)
├── aegis/data_models.py          # Finding, ModelRequest, etc.
├── aegis/database/model_repository.py
└── /api/models/ollama/register   # Your UI calls this

NEW SYSTEM (Added)
├── aegis/models/                 # New package
│   ├── schema.py
│   ├── registry.py (ModelRegistryV2)
│   ├── discovery/
│   ├── parsers/
│   └── runners/
└── /api/models/*                 # New endpoints

BRIDGE
└── /api/models/ollama/register writes to BOTH systems
```

## Next Steps (Choose Your Path)

### Path A: Keep Using Existing UI
- No changes needed
- Everything works as before
- New features available when you're ready

### Path B: Try New Features
1. Register HuggingFace models
2. Use discovery API
3. Configure parsers
4. Build custom pipelines

### Path C: Full Migration (Later)
1. Migrate to new model management
2. Update UI to use new endpoints
3. Deprecate old system
4. Single source of truth

## Troubleshooting

### App Won't Start
```bash
# Check imports
python -c "from aegis import create_app; print('OK')"

# If error, check IMPORT_FIX_SUMMARY.md
```

### Migration Fails
```bash
# Check if database exists
ls data/aegis.db

# If not, create it first
python -m aegis.database
```

### UI Registration Not Working
- Hard refresh browser: Ctrl+Shift+R
- Check console for errors
- Verify endpoint: `curl -X POST http://localhost:5000/api/models/ollama/register ...`

## Documentation

- **MODEL_REFACTOR_GUIDE.md** - Complete guide (500+ lines)
- **REFACTOR_SUMMARY.md** - Delivery summary
- **IMPORT_FIX_SUMMARY.md** - Import conflict fix details
- **QUICK_START.md** - This file

## Success Checklist

- [ ] Migration applied successfully
- [ ] App starts without errors
- [ ] Can register Ollama model via UI
- [ ] Model persists after restart
- [ ] New API endpoints accessible
- [ ] No regressions in existing scans

All should be ✅!

## Support

For detailed information:
1. See **MODEL_REFACTOR_GUIDE.md**
2. Check API with `curl` commands above
3. Review logs for errors
4. Test with provided test files

**You're all set!** The refactor is complete and backward compatible. 🚀
