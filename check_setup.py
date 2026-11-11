from pathlib import Path
import sys
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from configuration import CONFIG, path, inventory

for key in ("source_root", "prepared_dir", "training_dir", "test_dir"):
    location = path(key)
    if not location.is_dir():
        raise FileNotFoundError(f"{key}: {location}")
    print(f"OK: {key}")

checkpoint = path("training_dir") / "meilleur_modele.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
print("OK: saved model")

with (inventory() / "patients_repartition.csv").open(
    encoding="utf-8-sig", newline=""
) as stream:
    rows = list(csv.DictReader(stream, delimiter=";"))

groups = {
    name: {row["patient"] for row in rows if row["ensemble"] == name}
    for name in ("entrainement", "validation", "test")
}

expected = {"entrainement": 16, "validation": 4, "test": 4}

for name, patients in groups.items():
    if len(patients) != expected[name]:
        raise ValueError(f"Unexpected patient count: {name}")
    print(f"OK: {name}, {len(patients)} patients")

if sum(map(len, groups.values())) != len(set.union(*groups.values())):
    raise ValueError("Patient overlap between groups")

if groups["test"] != set(CONFIG["test_patients"]):
    raise ValueError("Test patients differ from configuration")

print("Setup checks passed. No training or evaluation performed.")