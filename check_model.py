from pathlib import Path
import importlib.util
import sys

import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from configuration import CONFIG, path
from essai_modele import PetitUNet

torch.set_num_threads(4)
torch.manual_seed(42)

checkpoint = torch.load(
    path("training_dir") / "meilleur_modele.pt",
    map_location="cpu",
    weights_only=True,
)

expected_epoch = CONFIG["expected_epoch"]
if expected_epoch is not None and checkpoint["epoque"] != expected_epoch:
    raise ValueError("Checkpoint epoch differs from configuration")

cleaned = PetitUNet()
cleaned.load_state_dict(checkpoint["modele"], strict=True)
cleaned.eval()

original_file = path("source_root") / "src" / "essai_modele.py"
spec = importlib.util.spec_from_file_location("original_model", original_file)
original_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_module)

original = original_module.PetitUNet()
original.load_state_dict(checkpoint["modele"], strict=True)
original.eval()

inputs = torch.rand(1, 6, 240, 240)

with torch.inference_mode():
    prediction = cleaned(inputs)
    reference = original(inputs)

if prediction.shape != (1, 1, 240, 240):
    raise ValueError(f"Unexpected output shape: {prediction.shape}")

if not torch.isfinite(prediction).all().item():
    raise ValueError("Prediction contains invalid values")

if (prediction < 0).any().item():
    raise ValueError("Prediction contains negative doses")

torch.testing.assert_close(prediction, reference, rtol=0, atol=0)

parameters = sum(p.numel() for p in cleaned.parameters())

print(f"OK: checkpoint loaded, epoch {checkpoint['epoque']}")
print(f"Parameters: {parameters:,}")
print(f"OK: output shape {tuple(prediction.shape)}")
print("OK: finite, non-negative predictions")
print("OK: cleaned and original outputs are identical")
print("No training performed. No patient data modified.")