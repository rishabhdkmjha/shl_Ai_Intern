
import json
import re
import sys
from pathlib import Path

def fix_and_validate(input_path, output_path=None):
    p = Path(input_path)
    if not p.exists():
        print(f"ERROR: File not found: {p}")
        sys.exit(1)
    
    print(f"Reading {p} ({p.stat().st_size} bytes)...")
    
    # Read as bytes first to handle encoding issues
    raw_bytes = p.read_bytes()
    
    # Try UTF-8, fall back to latin-1
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
        print("  Warning: fell back to latin-1 decoding")
    
    print("Cleaning control characters...")
    # Replace \r\n -> space, \r -> space
    text = text.replace("\r\n", " ").replace("\r", " ")
    # Remove remaining control chars except \t and \n
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    
    print("Validating JSON...")
    try:
        data = json.loads(text)
        print(f"SUCCESS: {len(data)} items parsed cleanly.")
    except json.JSONDecodeError as e:
        print(f"FAILED: {e}")
        # Show context around error
        start = max(0, e.pos - 50)
        end = min(len(text), e.pos + 50)
        print(f"  Context: ...{repr(text[start:end])}...")
        sys.exit(1)
    
    out = Path(output_path or input_path)
    out.write_text(text, encoding="utf-8")
    print(f"Saved clean JSON to: {out}")
    print("Now run: python -m scripts.build_index")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/catalog_raw.json"
    fix_and_validate(path)
