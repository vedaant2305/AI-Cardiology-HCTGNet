with open('ptbxl_pipeline.py', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Find the broken unindented block and fix it
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Fix the unindented comment and if block around line 277-291
    if line.strip() == '# Download metadata files only if missing':
        # Add proper indentation to this and following lines until we hit
        # the properly indented code again
        fixed_lines.append('    # Download metadata files only if missing\n')
        i += 1
        continue
    elif line.strip() == '# Individual records are loaded on-demand with pn_dir fallback':
        # Skip this comment line
        i += 1
        continue
    elif line.startswith('if not os.path.exists(metadata_path):'):
        fixed_lines.append('    if not os.path.exists(metadata_path):\n')
        i += 1
        continue
    elif line.startswith('    print("[DOWNLOAD]'):
        fixed_lines.append('        ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('    # Only download'):
        fixed_lines.append('        ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('    import urllib'):
        fixed_lines.append('        ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('    base_url'):
        fixed_lines.append('        ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('    for fname in'):
        fixed_lines.append('        ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('        fpath'):
        fixed_lines.append('            ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('        if not os.path.exists(fpath)'):
        fixed_lines.append('            ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('            print(f"  Downloading'):
        fixed_lines.append('                ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('            urllib.request.urlretrieve'):
        fixed_lines.append('                ' + line.lstrip())
        i += 1
        continue
    elif line.startswith('    print("  Metadata ready!")'):
        fixed_lines.append('        print("  Metadata ready!")\n')
        i += 1
        continue
    elif line.startswith('else:') and i < 295:
        fixed_lines.append('    else:\n')
        i += 1
        continue
    elif line.startswith('    print("[DATA] PTB-XL metadata found'):
        fixed_lines.append('        print("[DATA] PTB-XL metadata found in local cache.")\n')
        i += 1
        continue
    else:
        fixed_lines.append(line)
        i += 1

with open('ptbxl_pipeline.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print('Done! Verifying fix...')

# Verify
import ast
with open('ptbxl_pipeline.py', encoding='utf-8') as f:
    source = f.read()
try:
    ast.parse(source)
    print('No syntax errors found - file is fixed!')
except SyntaxError as e:
    print(f'Still has error at line {e.lineno}: {e.msg}')
    print(f'Text: {e.text}')