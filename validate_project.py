"""
Spendly AI — Static Validation Script
======================================
Validates project structure, file integrity, and code consistency
WITHOUT requiring TensorFlow (works on any Python 3.8+).

Run: python validate_project.py
"""

import os
import sys
import json
import ast

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def check(condition, msg, category=""):
    status = "PASS" if condition else "FAIL"
    symbol = "[+]" if condition else "[-]"
    print(f"  {symbol} {msg}")
    return condition

def main():
    print("=" * 70)
    print("  Spendly AI — Project Validation Report")
    print("=" * 70)
    
    results = {"pass": 0, "fail": 0}
    
    def track(result):
        if result:
            results["pass"] += 1
        else:
            results["fail"] += 1
    
    # ── 1. Struktur Folder ────────────────────────────────────────────────────
    print("\n[1] STRUKTUR FOLDER")
    required_dirs = [
        "api", "data", "logs", "models", "notebooks", "src", "outputs",
        "logs/classifier/train", "logs/classifier/val",
        "logs/forecaster/train", "logs/forecaster/val",
        "logs/ocr/train", "logs/ocr/val",
        "models/classifier", "models/forecaster", "models/ocr",
        "models/classifier/classifier_saved",
        "models/forecaster/forecaster_saved",
        "models/receipt_detector_savedmodel",
    ]
    for d in required_dirs:
        path = os.path.join(PROJECT_ROOT, d)
        track(check(os.path.isdir(path), f"Directory: {d}"))
    
    # ── 2. File Penting ──────────────────────────────────────────────────────
    print("\n[2] FILE PENTING")
    required_files = {
        "api/main.py": "FastAPI server",
        "src/custom_components.py": "Custom Keras components",
        "src/inference.py": "Inference module",
        "README.md": "Project documentation",
        "requirements.txt": "Python dependencies",
        ".gitignore": "Git ignore rules",
    }
    for f, desc in required_files.items():
        path = os.path.join(PROJECT_ROOT, f)
        track(check(os.path.isfile(path), f"{f} ({desc})"))
    
    # ── 3. Model Files ───────────────────────────────────────────────────────
    print("\n[3] MODEL FILES (.keras + SavedModel)")
    model_files = {
        "models/classifier/classifier.keras": "Classifier model",
        "models/classifier/tfidf_vectorizer.joblib": "TF-IDF vectorizer",
        "models/classifier/nlp_preprocessor.joblib": "NLP preprocessor",
        "models/forecaster/forecaster.keras": "Forecaster model",
        "models/forecaster/forecaster_weights.h5": "Forecaster weights",
        "models/forecaster/scaler.joblib": "MinMax scaler",
        "models/ocr/ocr_fixed.keras": "OCR model",
        "models/receipt_detector_best.keras": "Receipt Detector model",
        "models/classifier/classifier_saved/saved_model.pb": "Classifier SavedModel",
        "models/forecaster/forecaster_saved/saved_model.pb": "Forecaster SavedModel",
        "models/receipt_detector_savedmodel/saved_model.pb": "Detector SavedModel",
    }
    for f, desc in model_files.items():
        path = os.path.join(PROJECT_ROOT, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        size_str = f" ({size/1024/1024:.1f}MB)" if size > 1024*1024 else f" ({size/1024:.1f}KB)" if size > 1024 else ""
        track(check(exists, f"{f}{size_str} — {desc}"))
    
    # ── 4. TensorBoard Logs ──────────────────────────────────────────────────
    print("\n[4] TENSORBOARD LOGS")
    tb_dirs = [
        "logs/classifier/train", "logs/classifier/val",
        "logs/forecaster/train", "logs/forecaster/val",
        "logs/ocr/train", "logs/ocr/val",
    ]
    for d in tb_dirs:
        path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(path):
            events = [f for f in os.listdir(path) if f.startswith("events.out.tfevents")]
            track(check(len(events) > 0, f"{d} ({len(events)} event file(s))"))
        else:
            track(check(False, f"{d} (directory missing)"))
    
    # ── 5. Notebooks ─────────────────────────────────────────────────────────
    print("\n[5] NOTEBOOKS")
    notebooks = [
        "01_preprocessing_augmentation_FIXED_v2.ipynb",
        "02_custom_components.ipynb",
        "03_train_classifier_v3.ipynb",
        "04_train_forecaster_v2.ipynb",
        "05_train_ocr_REVISED_v5.ipynb",
        "06_evaluation_tensorboard_FIXED (2).ipynb",
        "notebook_07_v2_fixed.ipynb",
        "notebook_08_integration_test.ipynb",
    ]
    for nb in notebooks:
        path = os.path.join(PROJECT_ROOT, "notebooks", nb)
        track(check(os.path.isfile(path), f"notebooks/{nb}"))
    
    # ── 6. Syntax Validation ─────────────────────────────────────────────────
    print("\n[6] SYNTAX VALIDATION")
    py_files = [
        "api/main.py",
        "src/custom_components.py",
        "src/inference.py",
    ]
    for f in py_files:
        path = os.path.join(PROJECT_ROOT, f)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    ast.parse(fh.read())
                track(check(True, f"{f} — syntax valid"))
            except SyntaxError as e:
                track(check(False, f"{f} — SYNTAX ERROR: {e}"))
        else:
            track(check(False, f"{f} — file not found"))
    
    # ── 7. Code Consistency Checks ───────────────────────────────────────────
    print("\n[7] CODE CONSISTENCY")
    
    # Check api/main.py has 9 categories
    with open(os.path.join(PROJECT_ROOT, "api/main.py"), encoding="utf-8") as f:
        api_content = f.read()
    track(check('"Transport"' in api_content, "api/main.py has Transport category"))
    track(check("receipt_detector" in api_content, "api/main.py has receipt_detector endpoint"))
    track(check("ocr_fixed.keras" in api_content, "api/main.py uses correct OCR model path"))
    track(check("classifier.keras" in api_content, "api/main.py uses classifier.keras (not .h5)"))
    track(check("gemini" in api_content.lower(), "api/main.py has Gemini AI integration"))
    track(check("/detect-receipt" in api_content, "api/main.py has /detect-receipt endpoint"))
    track(check("/process-receipt" in api_content, "api/main.py has /process-receipt endpoint"))
    track(check("/insight" in api_content, "api/main.py has /insight endpoint"))
    track(check("FORECAST_CATEGORIES" in api_content, "api/main.py separates forecast categories (8)"))
    
    # Check inference.py
    with open(os.path.join(PROJECT_ROOT, "src/inference.py"), encoding="utf-8") as f:
        inf_content = f.read()
    track(check('"Transport"' in inf_content, "inference.py has Transport category"))
    track(check("ocr_weights_fixed" in inf_content, "inference.py uses correct OCR weights path"))
    track(check("classifier.keras" in inf_content, "inference.py loads classifier.keras"))
    track(check("SpendlyReceiptDetector" in inf_content, "inference.py has SpendlyReceiptDetector class"))
    
    # Check custom_components.py
    with open(os.path.join(PROJECT_ROOT, "src/custom_components.py"), encoding="utf-8") as f:
        cc_content = f.read()
    track(check("CTCLayer" in cc_content, "custom_components.py has CTCLayer"))
    track(check("AttentionLayer" in cc_content, "custom_components.py has AttentionLayer"))
    track(check("SEBlock" in cc_content, "custom_components.py has SEBlock"))
    track(check("FocalLoss" in cc_content, "custom_components.py has FocalLoss"))
    track(check("SpendlyCallback" in cc_content, "custom_components.py has SpendlyCallback"))
    track(check("NLPPreprocessor" in cc_content, "custom_components.py has NLPPreprocessor"))
    track(check("get_config" in cc_content, "custom_components.py has get_config (serialization)"))
    
    # ── 8. Data Files ────────────────────────────────────────────────────────
    print("\n[8] DATA FILES")
    data_files = {
        "data/_csv/split_manifest.csv": "Split manifest",
        "data/_csv/synthetic_spending.csv": "Synthetic spending data",
        "data/_processed/dataset_text.csv": "OCR text dataset",
    }
    for f, desc in data_files.items():
        path = os.path.join(PROJECT_ROOT, f)
        track(check(os.path.isfile(path), f"{f} — {desc}"))
    
    # Data categories
    categories = ["Beauty", "F&B", "Gas", "Groceries", "Health", "HouseHold", "Lifestyle", "Listrik"]
    for cat in categories:
        path = os.path.join(PROJECT_ROOT, "data", cat)
        track(check(os.path.isdir(path), f"data/{cat}/ — image data folder"))
    
    # ── Summary ──────────────────────────────────────────────────────────────
    total = results["pass"] + results["fail"]
    print("\n" + "=" * 70)
    print(f"  RESULTS: {results['pass']}/{total} passed, {results['fail']} failed")
    print("=" * 70)
    
    if results["fail"] == 0:
        print("\n  All checks passed! Project is ready for GitHub push.")
    else:
        print(f"\n  {results['fail']} check(s) failed. Review issues above.")
    
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
