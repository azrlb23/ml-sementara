import os
import subprocess
import sys
import time

# List of scripts to run in order
pipeline_scripts = [
    "01_eda_and_preprocessing.py",
    "02.0_unsupervised_learning_method.py",
    "02.1_kmeans_standard.py",
    "02.2_kmeans_de.py",
    # 02.3_kmeans_pso.py and 02.4_kmeans_eoa.py are skipped per user instructions
    "03_comparison_analysis.py",
    "04.1.1_classification_decision_tree_standard.py",
    "04.1.2_classification_svm_standard.py",
    "04.1.3_classification_adaboost_standard.py",
    "04.1.4_classification_ann_standard.py",
    "04.2.1_classification_decision_tree_qlde.py",
    "04.2.2_classification_svm_qlde.py",
    "04.2.3_classification_adaboost_qlde.py",
    "04.2.4_classification_ann_qlde.py",
    "04.3.1_classification_decision_tree_de.py",
    "04.3.2_classification_svm_de.py",
    "04.3.3_classification_adaboost_de.py",
    "04.3.4_classification_ann_de.py",
    "04.4_classification_comparison.py"
]

base_dir = os.path.dirname(os.path.abspath(__file__))
extracted_dir = os.path.join(base_dir, "extracted_code")

# Copy current environment and set matplotlib to non-interactive Agg backend to avoid blocking on plt.show()
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

print("====================================================")
print("Starting overall data and model synchronization...")
print(f"Skipping PSO and EOA clustering models per instruction.")
print("====================================================")

start_time = time.time()

for idx, script in enumerate(pipeline_scripts, 1):
    script_path = os.path.join(extracted_dir, script)
    print(f"\n[{idx}/{len(pipeline_scripts)}] Running {script}...")
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script {script_path} not found.")
        sys.exit(1)
        
    script_start = time.time()
    
    # Run the script with Cwd set to extracted_code directory
    # so that relative paths like ../data/... resolve to workspace data/ directory
    process = subprocess.Popen(
        [sys.executable, script],
        cwd=extracted_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output in real time
    for line in process.stdout:
        print(f"  {line}", end="")
        
    process.wait()
    script_elapsed = time.time() - script_start
    
    if process.returncode != 0:
        print(f"\n❌ Error: {script} failed with return code {process.returncode} after {script_elapsed:.1f}s.")
        sys.exit(process.returncode)
    else:
        print(f"✓ Completed {script} in {script_elapsed:.1f}s.")

total_elapsed = time.time() - start_time
print("\n====================================================")
print(f"✓ All tasks completed successfully in {total_elapsed:.1f} seconds!")
print("====================================================")
