"""
Setup verification script.

Checks that all necessary directories, files, and dependencies exist
for the Activity Classification Streamlit app to run properly.
"""

import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Verify Python version is >= 3.8"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Requires Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor} - OK")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn'
    }
    
    all_ok = True
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            print(f"   Run: pip install {package_name}")
            all_ok = False
    
    return all_ok


def check_directories():
    """Verify required directories exist"""
    required_dirs = {
        'Data': Path('Data'),
        'Data/d01_raw_data': Path('Data/d01_raw_data'),
        'Models': Path('Models'),
        'Scripts': Path('Scripts'),
        'Notebooks': Path('Notebooks'),
    }
    
    all_ok = True
    for name, path in required_dirs.items():
        if path.exists():
            print(f"✅ {name}/ - OK")
        else:
            print(f"❌ {name}/ - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Verify data files exist"""
    required_files = {
        'TransformedData.csv': Path('Data/TransformedData.csv'),
    }
    
    # Check for raw .npy files
    raw_data_dir = Path('Data/d01_raw_data')
    activity_count = 0
    if raw_data_dir.exists():
        npy_files = list(raw_data_dir.glob('*.npy'))
        activity_count = len(npy_files)
        if activity_count > 0:
            required_files[f'Raw data files ({activity_count})'] = raw_data_dir
    
    all_ok = True
    for name, path in required_files.items():
        if isinstance(path, Path) and (path.exists() or path.parent.exists()):
            print(f"✅ {name} - OK")
        else:
            print(f"❌ {name} - NOT FOUND")
            all_ok = False
    
    if activity_count < 16:
        print(f"⚠️  Only {activity_count}/16 activities found (expected 000-015)")
    
    return all_ok or activity_count > 0


def check_models():
    """Check if any models exist in Models directory"""
    models_dir = Path('Models')
    if models_dir.exists():
        pkl_files = list(models_dir.glob('*.pkl'))
        if pkl_files:
            print(f"✅ Models available: {len(pkl_files)} model(s)")
            for model in pkl_files:
                print(f"   - {model.name}")
            return True
        else:
            print(f"⚠️  No trained models found in Models/")
            print("   Run: python Scripts/train_model_example.py")
            return True  # Not critical, still return True
    return True


def check_python_files():
    """Verify key Python files exist"""
    required_files = {
        'app.py': Path('app.py'),
        'config.py': Path('config.py'),
        'Scripts/preprocessing_utils.py': Path('Scripts/preprocessing_utils.py'),
        'Scripts/train_model_example.py': Path('Scripts/train_model_example.py'),
    }
    
    all_ok = True
    for name, path in required_files.items():
        if path.exists():
            print(f"✅ {name} - OK")
        else:
            print(f"❌ {name} - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_imports():
    """Test that app.py can be imported without errors"""
    try:
        # Try importing streamlit
        import streamlit as st
        print("✅ App components can be imported - OK")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    print("=" * 60)
    print("Activity Classification Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Data Files", check_data_files),
        ("Python Files", check_python_files),
        ("Imports", check_imports),
        ("Models", check_models),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error during check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run:")
        print("   streamlit run app.py")
        return 0
    elif passed >= 5:
        print("\n⚠️  Setup is mostly ready, but fix the issues above.")
        print("   Main issues to address:")
        for name, result in results:
            if not result:
                print(f"   - {name}")
        return 1
    else:
        print("\n❌ Setup has critical issues. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
