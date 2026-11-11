from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
import csv
import json
import time

import nibabel as nib
import numpy as np
import torch
from torch import nn


def verifier(condition, message):
    if not condition:
        raise ValueError(message)


class Bloc(nn.Module):
    def __init__(self, entree, sortie):
        super().__init__()
        self.operations = nn.Sequential(
            nn.Conv2d(entree, sortie, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(sortie, sortie, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.operations(x)


class PetitUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = Bloc(6, 8)
        self.enc2 = Bloc(8, 16)
        self.fond = Bloc(16, 32)

        self.reduction = nn.MaxPool2d(2)

        self.montee2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = Bloc(32, 16)

        self.montee1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = Bloc(16, 8)

        self.sortie = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.reduction(e1))
        fond = self.fond(self.reduction(e2))

        d2 = self.montee2(fond)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self.montee1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        return self.sortie(d1)


def charger_exemples(lot, patients):
    entrees, cibles, masques, descriptions = [], [], [], []

    for patient in patients:
        dossier = lot / patient
        meta = json.loads(
            (dossier / "preparation.json").read_text(encoding="utf-8")
        )

        irm = nib.load(
            dossier / "t1c_minmax.nii.gz"
        ).get_fdata(dtype=np.float32)

        annotations = nib.load(
            dossier / "annotations.nii.gz"
        ).get_fdata(dtype=np.float32)

        dose = nib.load(
            dossier / "dose_relative.nii.gz"
        ).get_fdata(dtype=np.float32)

        masque_brut = nib.load(
            dossier / "masque_dose_valide.nii.gz"
        ).get_fdata(dtype=np.float32)

        verifier(
            irm.shape == annotations.shape == dose.shape == masque_brut.shape,
            f"{patient} : dimensions differentes",
        )
        verifier(irm.shape == (240, 240, 155), "Dimensions inattendues")
        verifier(np.isfinite(irm).all(), "IRM non finie")
        verifier(np.isin(annotations, [0, 1, 2, 3]).all(),
                 "Etiquettes invalides")
        verifier(np.isin(masque_brut, [0, 1]).all(),
                 "Masque non binaire")

        masque = masque_brut.astype(bool)

        # Une coupe annotee et une sans annotation, si disponible.
        tailles_annotations = np.count_nonzero(
            annotations, axis=(0, 1)
        )
        tailles_valides = np.count_nonzero(
            masque, axis=(0, 1)
        )

        annotees = np.flatnonzero(
            (tailles_annotations > 0) & (tailles_valides > 0)
        )
        verifier(annotees.size > 0, f"{patient} : aucune coupe annotee")

        z_annote = int(
            annotees[np.argmax(tailles_annotations[annotees])]
        )
        indices = [z_annote]

        sans_annotation = np.flatnonzero(
            (tailles_annotations == 0) & (tailles_valides > 0)
        )
        if sans_annotation.size:
            indices.append(
                int(sans_annotation[len(sans_annotation) // 2])
            )

        prescription = float(meta["prescription_totale_Gy"])
        fractions = int(meta["nombre_fractions"])

        verifier(prescription > 0 and fractions > 0,
                 "Prescription ou fractionnement invalide")

        for z in indices:
            ann = annotations[:, :, z]
            forme = irm[:, :, z].shape

            x = np.stack([
                irm[:, :, z],
                (ann == 1).astype(np.float32),
                (ann == 2).astype(np.float32),
                (ann == 3).astype(np.float32),
                np.full(forme, prescription / 35, dtype=np.float32),
                np.full(forme, fractions / 7, dtype=np.float32),
            ])

            y = dose[:, :, z][None, :, :]
            m = masque[:, :, z][None, :, :]

            verifier(np.isfinite(x).all(), "Entree non finie")
            verifier(np.isfinite(y[m]).all(), "Cible valide non finie")
            verifier((y[m] >= 0).all(), "Dose valide negative")
            verifier(m.any(), "Masque de validite vide")

            entrees.append(x)
            cibles.append(y)
            masques.append(m)
            descriptions.append(
                f"{patient}, axe 2, indice {z}, "
                f"{prescription} Gy / {fractions} fraction(s), "
                f"{int(m.sum())} pixels valides"
            )

    return (
        torch.from_numpy(np.stack(entrees)),
        torch.from_numpy(np.stack(cibles)),
        torch.from_numpy(np.stack(masques)),
        descriptions,
    )


def main():
    racine = Path(__file__).resolve().parent.parent

    lot = path("prepared_dir")
    verifier(lot.is_dir(), f"Lot introuvable : {lot}")

    inventaires = [inventory()]
    inventaire = inventaires[0]

    with (inventaires[0] / "patients_repartition.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        repartition = list(csv.DictReader(f, delimiter=";"))

    # Choisir un patient d'entrainement de chaque fractionnement.
    patients = []
    for fractions in (1, 7):
        candidats = sorted(
            ligne["patient"]
            for ligne in repartition
            if ligne["ensemble"] == "entrainement"
            and int(ligne["nombre_fractions"]) == fractions
        )
        verifier(bool(candidats), "Groupe d'entrainement manquant")
        patients.append(candidats[0])

    torch.manual_seed(20261103)
    torch.set_num_threads(4)

    debut_chargement = time.perf_counter()
    x, y, masque, descriptions = charger_exemples(lot, patients)
    duree_chargement = time.perf_counter() - debut_chargement

    modele = PetitUNet()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=0.001)
    pertes = []

    modele.train()
    debut_calcul = time.perf_counter()

    # Dix mises a jour sur le meme petit lot : essai technique.
    for iteration in range(10):
        optimiseur.zero_grad(set_to_none=True)
        prediction = modele(x)

        verifier(prediction.shape == y.shape,
                 "Dimensions de prediction incorrectes")

        # Selectionner AVANT de soustraire pour ignorer les NaN.
        erreur = prediction[masque] - y[masque]
        perte = (erreur ** 2).mean()

        verifier(torch.isfinite(perte).item(), "Perte non finie")
        perte.backward()

        for parametre in modele.parameters():
            verifier(
                parametre.grad is not None
                and torch.isfinite(parametre.grad).all().item(),
                "Gradient absent ou non fini",
            )

        optimiseur.step()
        pertes.append(float(perte.item()))
        print(
            f"Iteration {iteration + 1}/10 : perte = {pertes[-1]:.6f}",
            flush=True,
        )

    duree_calcul = time.perf_counter() - debut_calcul

    modele.eval()
    with torch.no_grad():
        prediction = modele(x)

    verifier(torch.isfinite(prediction).all().item(),
             "Prediction non finie")
    verifier((prediction >= 0).all().item(), "Prediction negative")

    bilan = [
        "ESSAI DU MODELE SUR DES COUPES D'ENTRAINEMENT",
        f"Lot : {lot}",
        *descriptions,
        f"Entrees : {tuple(x.shape)}",
        f"Cibles : {tuple(y.shape)}",
        f"Parametres du modele : "
        f"{sum(p.numel() for p in modele.parameters())}",
        f"Perte initiale : {pertes[0]:.6f}",
        f"Perte avant la derniere mise a jour : {pertes[-1]:.6f}",
        f"Chargement : {duree_chargement:.2f} seconde(s)",
        f"Calcul des 10 iterations : {duree_calcul:.2f} seconde(s)",
        f"Prediction apres essai : "
        f"{prediction.min().item():.6f} a {prediction.max().item():.6f}",
        "Pertes et gradients finis : oui",
        "Validation et test : non utilises",
        "Poids de cet essai : non sauvegardes",
        "Ce test ne mesure pas la generalisation du modele.",
    ]

    sortie = racine / "resultats" / "essai_modele"
    sortie.mkdir(parents=True, exist_ok=True)
    chemin = sortie / "bilan_essai_modele.txt"
    chemin.write_text("\n".join(bilan), encoding="utf-8")

    print("\n" + "\n".join(bilan))
    print(f"\nBilan enregistre : {chemin}")


if __name__ == "__main__":
    main()