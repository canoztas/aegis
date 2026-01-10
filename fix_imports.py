#!/usr/bin/env python3
"""Fix imports after renaming models.py to data_models.py"""

import os
from pathlib import Path

project_root = Path(__file__).parent

files_to_update = [
    "aegis/adapters/base.py",
    "aegis/adapters/classic_adapter.py",
    "aegis/adapters/hf_adapter.py",
    "aegis/adapters/llm_adapter.py",
    "aegis/adapters/ollama_adapter.py",
    "aegis/consensus/engine.py",
    "aegis/exports.py",
    "aegis/prompt_builder.py",
    "aegis/pipeline/executor.py",
    "aegis/database/repositories.py",
    "aegis/routes.py",
    "aegis/runner.py",
]

for file_path in files_to_update:
    full_path = project_root / file_path
    if not full_path.exists():
        print(f"[SKIP] {file_path} - not found")
        continue

    content = full_path.read_text(encoding='utf-8')
    updated_content = content.replace(
        "from aegis.models import",
        "from aegis.data_models import"
    )

    if content != updated_content:
        full_path.write_text(updated_content, encoding='utf-8')
        print(f"[OK] Updated {file_path}")
    else:
        print(f"[SKIP] {file_path} - no changes needed")

print("\n[DONE] Import fixes applied!")
