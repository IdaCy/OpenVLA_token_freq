import sys
import importlib
import pkg_resources

# Define required packages and their expected versions
required_packages = {
    "timm": "0.9.10",
    "numpy": "2.2.1",
    "transformers": ">=4.0.0",
    "huggingface_hub": ">=0.15.1",
    "torch": "2.2.1",
    "accelerate": ">=0.27.2"
}


# Check the package existence and version
def check_packages():
    for package, expected_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = pkg_resources.get_distribution(package).version
            location = module.__file__
            print(f"✅ {package}: Found version {version} at {location}")
            
            # Version compatibility check
            if pkg_resources.parse_version(version) not in pkg_resources.Requirement.parse(f"{package}{expected_version}"):
                print(f"⚠️ Version mismatch for {package}: Expected {expected_version}, but found {version}")

        except ImportError:
            print(f"❌ {package}: Not installed")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: Installed but version info not found")


check_packages()

# Check Python executable and paths
print("\nPython executable:", sys.executable)
print("PYTHONPATH:", sys.path)
