"""Unpack artifacts.zip into artifacts/, normalising Windows-style backslash paths (works on Windows, Linux and macOS)."""
import os, sys, zipfile
zip_path = sys.argv[1] if len(sys.argv) > 1 else 'artifacts/artifacts.zip'
out = sys.argv[2] if len(sys.argv) > 2 else 'artifacts'
n = 0
with zipfile.ZipFile(zip_path) as z:
    for m in z.infolist():
        name = m.filename.replace('\\', '/').lstrip('/')
        if not name or '..' in name.split('/'): continue
        dest = os.path.join(out, *name.split('/'))
        if name.endswith('/'): os.makedirs(dest, exist_ok=True); continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f: f.write(z.read(m))
        n += 1
print(f'extracted {n} files into {out}/')