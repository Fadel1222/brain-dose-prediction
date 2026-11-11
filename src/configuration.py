from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
)

local = ROOT / "configs" / "local.json"
if local.is_file():
    CONFIG.update(json.loads(local.read_text(encoding="utf-8")))


def path(key):
    value = Path(CONFIG[key]).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def inventory():
    if CONFIG.get("inventory_dir"):
        return path("inventory_dir")
    matches = sorted(path("prepared_dir").glob("inventaire_2d_*"))
    if len(matches) != 1:
        raise ValueError(
            "Set inventory_dir in configs/local.json: expected one inventory."
        )
    return matches[0]


def patients(group):
    with (inventory() / "patients_repartition.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        return [
            row["patient"]
            for row in csv.DictReader(stream, delimiter=";")
            if row["ensemble"] == group
        ]


def prescription(patient):
    meta = json.loads(
        (path("prepared_dir") / patient / "preparation.json").read_text(
            encoding="utf-8"
        )
    )
    dose = float(meta["prescription_totale_Gy"])
    fractions = int(meta["nombre_fractions"])
    if dose <= 0 or fractions <= 0:
        raise ValueError(f"Invalid prescription: {patient}")
    return dose, fractions
