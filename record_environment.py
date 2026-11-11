from pathlib import Path
from importlib.metadata import version
import platform

root = Path(__file__).resolve().parent
packages = ["numpy", "nibabel", "pydicom", "matplotlib", "torch"]

requirements = [
    f"{package}=={version(package)}"
    for package in packages
]

requirements_file = root / "requirements-tested.txt"
requirements_file.write_text(
    "\n".join(requirements) + "\n",
    encoding="utf-8",
)

environment = [
    f"Python: {platform.python_version()}",
    f"System: {platform.system()}",
    f"Architecture: {platform.machine()}",
    "",
    *requirements,
]

environment_file = root / "docs" / "tested_environment.txt"
environment_file.parent.mkdir(exist_ok=True)
environment_file.write_text(
    "\n".join(environment) + "\n",
    encoding="utf-8",
)

print("\n".join(environment))
print("\nCreated requirements-tested.txt")
print("Created docs/tested_environment.txt")