import os
import glob

DATASET_PATH = r"C:\Nizam\DBS-Foundation\Capstone - Ai Eng - Spendly_ Workspace\data"

# Cari semua data.yaml di dalam setiap kategori
yaml_files = glob.glob(os.path.join(DATASET_PATH, "**", "data.yaml"), recursive=True)
for yf in yaml_files:
    print(f"\n📄 {yf}")
    with open(yf) as f:
        print(f.read())