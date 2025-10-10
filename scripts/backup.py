#!/usr/bin/env python
from pathlib import Path
import shutil
src = Path("data/app.db")
dst = Path("data/app.backup.db")
if src.exists():
    shutil.copy2(src, dst)
    print("Backup created:", dst)
else:
    print("No DB found at", src)
