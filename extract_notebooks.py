import json
import os
import glob

notebooks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_code")
os.makedirs(output_dir, exist_ok=True)

# Clear existing .py files in output directory to avoid stale files
for old_py in glob.glob(os.path.join(output_dir, "*.py")):
    try:
        os.remove(old_py)
    except Exception as e:
        print(f"Error removing {old_py}: {e}")

notebook_files = glob.glob(os.path.join(notebooks_dir, "*.ipynb"))

for nb_file in notebook_files:
    basename = os.path.basename(nb_file)
    name, _ = os.path.splitext(basename)
    print(f"Extracting {basename}...")
    
    with open(nb_file, "r", encoding="utf-8") as f:
        try:
            nb_data = json.load(f)
        except Exception as e:
            print(f"Error loading {nb_file}: {e}")
            continue
            
    code_lines = []
    cell_idx = 1
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            code_lines.append(f"# === Cell {cell_idx} ===\n")
            source = cell.get("source", [])
            if isinstance(source, list):
                cell_code = "".join(source)
            else:
                cell_code = source
            
            if not cell_code.endswith("\n"):
                cell_code += "\n"
            code_lines.append(cell_code)
            code_lines.append("\n\n")
            cell_idx += 1
            
    out_file = os.path.join(output_dir, f"{name}.py")
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(code_lines)
    print(f"Saved to {out_file}")
