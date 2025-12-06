"""
Syntax and import check for TEMPORAL implementation.
Verifies all modules can be parsed without running them.
"""

import ast
import os
import sys


def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    """Check all Python files in the project"""
    print("\n" + "="*70)
    print("TEMPORAL SYNTAX VALIDATION")
    print("="*70 + "\n")

    files_to_check = [
        'config.py',
        'time_embeddings.py',
        'model.py',
        'train.py',
        'evaluate.py',
        'visualize.py',
        'test_implementation.py',
    ]

    results = []

    for filename in files_to_check:
        filepath = os.path.join(os.path.dirname(__file__), filename)

        if not os.path.exists(filepath):
            results.append((filename, False, "File not found"))
            continue

        valid, error = check_syntax(filepath)
        results.append((filename, valid, error))

    # Print results
    print("File Syntax Check Results:\n")

    passed = 0
    failed = 0

    for filename, valid, error in results:
        if valid:
            print(f"✓ {filename:30s} - VALID")
            passed += 1
        else:
            print(f"✗ {filename:30s} - INVALID")
            print(f"  Error: {error}")
            failed += 1

    print("\n" + "="*70)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("="*70 + "\n")

    if failed == 0:
        print("✅ All files have valid Python syntax!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: python test_implementation.py")
        print("3. Train models: python train.py --model both")
        print("4. Evaluate: python evaluate.py --model both")
        print("5. Visualize: python visualize.py")
        return 0
    else:
        print("❌ Some files have syntax errors. Please fix them before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
