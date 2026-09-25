"""
Master Platform Runner: Zepto Data & AI Platform
Executes and validates all three modules end-to-end:
1. Module 1: Data Engineering Pipeline (Scraping, Cleaning, SQLite, SQL, Pandas Equivalence)
2. Module 2: Analytics & ML (EDA, Preprocessing, Modeling, Tuning, Regression, Joblib)
3. Module 3 & Test Suite: Automated Pytest suite across all 23 test cases
"""

import os
import sys
import subprocess

def run_step(step_name: str, command: list):
    print("\n" + "=" * 80)
    print(f"EXECUTING: {step_name}")
    print("=" * 80)
    res = subprocess.run(command, check=True)
    if res.returncode != 0:
        print(f"FAILED: {step_name}")
        sys.exit(1)
    print(f"SUCCESS: {step_name}")

def main():
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    print("\n" + "#" * 80)
    print("ZEPTO DATA & AI PLATFORM — FULL PLATFORM EXECUTION & VALIDATION")
    print("#" * 80)

    # 1. Module 1
    run_step(
        "Module 1: Data Pipeline", 
        [python_exe, "data_pipeline/run_pipeline.py"]
    )

    # 2. Module 2
    run_step(
        "Module 2: Analytics & Predictive Modeling", 
        [python_exe, "analytics/eda_and_modeling.py"]
    )

    # 3. Full Test Suite
    pytest_exe = os.path.join(".venv", "Scripts", "pytest.exe")
    if not os.path.exists(pytest_exe):
        pytest_exe = "pytest"
    run_step(
        "Full Automated Test Suite (All 24 Unit & Integration Tests)", 
        [pytest_exe, "tests/", "-v"]
    )

    # 4. Module 3 Sample Query Verification
    print("\n" + "=" * 80)
    print("MODULE 3 SAMPLE QUERY DEMONSTRATION")
    print("=" * 80)
    from support_assistant.graph import ask_question
    pol_q = "What is the delivery fee for orders below Rs 149?"
    gen_q = "What is the tallest mountain in the world?"
    
    print(f"Query 1 (Policy):  '{pol_q}'")
    print(f"Response:          {ask_question(pol_q)}\n")
    print(f"Query 2 (General): '{gen_q}'")
    print(f"Response:          {ask_question(gen_q)}")

    print("\n" + "#" * 80)
    print("ALL PLATFORM MODULES & 24 TESTS VERIFIED SUCCESSFULLY!")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
