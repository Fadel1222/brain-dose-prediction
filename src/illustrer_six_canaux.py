from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


RACINE = Path(__file__).resolve().parents[1]
LOT = path("prepared_dir")
SORTIE = RACINE / "resultats" / "figures_rapport"

# Patients du groupe d'entraînement uniquement.
PATIENTS = split_patients("entrainement")


def charger(chemin):
    return nib.load(str(chemin)).get_fdata(dtype=np.float32)


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)

    # Choisir la coupe qui représente le mieux les trois catégories.
    meilleure_note = (-1, -1)
    choix = None

    for patient in PATIENTS:
        annotations = charger(LOT / patient / "annotations.nii.gz")

        for z in range(annotations.shape[2]):
            coupe = annotations[:, :, z]
            comptes = [
                int(np.count_nonzero(coupe == etiquette))
                for etiquette in (1, 2, 3)
            ]

            # Priorité à la présence des trois catégories,
            # puis à la taille de la catégorie la moins représentée.
            note = (
                sum(nombre > 0 for nombre in comptes),
                min(comptes),
            )

            if note > meilleure_note:
                meilleure_note = note
                choix = (patient, z)

    if choix is None or meilleure_note[0] < 3:
        raise ValueError(
            "Aucune coupe contenant les trois catégories n'a été trouvée."
        )

    patient, z = choix
    dossier = LOT / patient

    irm = charger(dossier / "t1c_minmax.nii.gz")[:, :, z]
    annotations = charger(dossier / "annotations.nii.gz")[:, :, z]

    prescription, fractions = read_prescription(patient)

    canaux = [
        irm,
        (annotations == 1).astype(float),
        (annotations == 2).astype(float),
        (annotations == 3).astype(float),
        np.full(irm.shape, prescription / 35.0),
        np.full(irm.shape, fractions / 7.0),
    ]

    titres = [
        "Canal 1 — IRM normalisée",
        "Canal 2 — Nécrose",
        "Canal 3 — Tumeur rehaussée",
        "Canal 4 — Œdème",
        f"Canal 5 — Prescription / 35 Gy\nValeur = {prescription / 35:.4f}",
        f"Canal 6 — Fractions / 7\nValeur = {fractions / 7:.4f}",
    ]

    figure, axes = plt.subplots(
        2, 3, figsize=(12, 8), layout="constrained"
    )

    for indice, (axe, canal, titre) in enumerate(
        zip(axes.flat, canaux, titres)
    ):
        affichage = axe.imshow(
            canal.T,
            origin="lower",
            cmap="gray" if indice < 4 else "viridis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axe.set_title(titre, fontsize=11)
        axe.set_xticks([])
        axe.set_yticks([])
        figure.colorbar(
            affichage, ax=axe, shrink=0.75, ticks=[0, 0.5, 1]
        )

    figure.suptitle(
        f"Construction des six canaux d'entrée — {patient}, "
        f"axe 2, indice {z}\n"
        "Même coupe et même orientation — côtés du patient non indiqués",
        fontsize=13,
    )

    chemin = SORTIE / "six_canaux_entree.png"
    figure.savefig(chemin, dpi=200)
    plt.close(figure)

    print(f"Patient sélectionné : {patient}")
    print(f"Coupe : {z}")
    print(f"Prescription : {prescription:g} Gy / {fractions} fraction(s)")
    print(f"Image enregistrée : {chemin}")


if __name__ == "__main__":
    main()