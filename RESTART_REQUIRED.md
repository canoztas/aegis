# 🔄 Restart Required - Role Mapping Fix

## What Changed

Fixed backward compatibility for legacy `role='scan'` values in the database.

### Changes Made:
1. ✅ Added legacy role mapping in `ModelRegistryV2._row_to_record()`
2. ✅ Added helper functions in `routes_models.py` for role parsing
3. ✅ Updated API endpoints to use legacy-aware role parsing

### Legacy Role Mapping:
- `"scan"` → `ModelRole.DEEP_SCAN`
- `"triage"` → `ModelRole.TRIAGE`
- `"deep_scan"` → `ModelRole.DEEP_SCAN`
- `"judge"` → `ModelRole.JUDGE`
- `"explain"` → `ModelRole.EXPLAIN`

## Action Required

**Please restart the app** to pick up the changes:

```bash
# Stop the current server (CTRL+C)
# Then restart:
python app.py
```

## Verification

After restart, test the new API:

```bash
curl http://127.0.0.1:5000/api/models/registered
```

Should return your registered models with `role='scan'` automatically mapped to `deep_scan`.

## Why This Happened

Old models in the database have `role='scan'` but the new `ModelRole` enum only supports:
- `triage`
- `deep_scan`
- `judge`
- `explain`
- `custom`

The fix adds automatic mapping for backward compatibility.

## Files Modified

1. `aegis/models/registry.py` - Added `ROLE_MAPPING` in `_row_to_record()`
2. `aegis/api/routes_models.py` - Added `parse_role()` and `parse_roles()` helpers

Both files now handle legacy role values gracefully.
