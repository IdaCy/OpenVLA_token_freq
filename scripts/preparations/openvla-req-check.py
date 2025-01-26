import importlib.metadata

required_packages = {
    "timm": "0.9.10",
    "tokenizers": "0.19.1",
    "torch": "2.2.0",
    "torchvision": "0.16.0",
    "transformers": "4.40.1"
}

for package, required_version in required_packages.items():
    try:
        installed_version = importlib.metadata.version(package)
        if installed_version >= required_version:
            print(f"{package} - OK (installed: {installed_version}, required: {required_version})")
        else:
            print(f"{package} - WARNING: Installed {installed_version}, but required {required_version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package} - NOT INSTALLED")

print("Package check completed.")
