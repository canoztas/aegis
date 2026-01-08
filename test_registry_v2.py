#!/usr/bin/env python3
"""Test ModelRegistryV2."""
from aegis.registry_v2 import ModelRegistryV2

reg = ModelRegistryV2()

# List all models
models = reg.list_all_models()
print(f"Total models: {len(models)}\n")

# Models by role
print("Models by role:")
for role in ['scan', 'triage', 'deep_scan', 'judge']:
    role_models = reg.list_models_by_role(role)
    print(f"\n  {role.upper()}: {len(role_models)} models")
    for m in role_models[:5]:
        print(f"    - {m['display_name']} (weight: {m['weight']})")

# Test adapter creation
print("\n\nTesting adapter creation:")
test_model_ids = [
    "ollama:qwen2.5-coder:7b",
    "ollama:qwen2.5-coder:7b:triage",
]

for model_id in test_model_ids:
    adapter = reg.get_adapter(model_id)
    if adapter:
        print(f"  [OK] Created adapter for: {adapter.display_name}")
    else:
        print(f"  [ERR] Failed to create adapter for: {model_id}")
