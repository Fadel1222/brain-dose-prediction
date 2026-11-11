from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize


RACINE = Path(__file__).resolve().parents[1]

LOT = path("prepared_dir")

TEST = path("test_dir")

SORTIE = ROOT / "resultats" / "reliefs"
PATIENTS = {p: read_prescription(p)[0] for p in CONFIG["test_patients"]}


def charger(chemin):
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier absent : {chemin}")

    image = nib.load(str(chemin))
    tableau = image.get_fdata(dtype=np.float32)
    return image, tableau


def verifier_grille(reference, autre, nom):
    if reference.shape != autre.shape:
        raise ValueError(f"Dimensions différentes pour {nom}")

    if not np.allclose(reference.affine, autre.affine, atol=1e-4):
        raise ValueError(f"Grilles différentes pour {nom}")


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)

    for patient, prescription in PATIENTS.items():
        dossier = LOT / patient

        image_ref, reference = charger(
            dossier / "dose_relative.nii.gz"
        )
        image_pred, prediction = charger(
            TEST / f"{patient}_dose_relative_predite.nii.gz"
        )
        image_masque, masque = charger(
            dossier / "masque_dose_valide.nii.gz"
        )

        verifier_grille(image_ref, image_pred, "la prédiction")
        verifier_grille(image_ref, image_masque, "le masque")

        if reference.ndim != 3:
            raise ValueError(f"{patient} : volume 3D attendu")

        valide = masque > 0

        if not valide.any():
            raise ValueError(f"{patient} : masque valide vide")

        if not np.isfinite(reference[valide]).all():
            raise ValueError(f"{patient} : référence non finie")

        if not np.isfinite(prediction[valide]).all():
            raise ValueError(f"{patient} : prédiction non finie")

        # Conversion des doses relatives en Gy.
        reference = reference * prescription
        prediction = prediction * prescription

        # Même coupe pour les deux cartes :
        # celle contenant le maximum de référence valide.
        recherche = np.where(valide, reference, -np.inf)
        position = np.unravel_index(
            np.argmax(recherche), recherche.shape
        )
        z = int(position[2])

        masque_coupe = valide[:, :, z]

        # Recadrage commun autour de la région valide.
        positions = np.argwhere(masque_coupe)
        debut = np.maximum(positions.min(axis=0) - 5, 0)
        fin = np.minimum(
            positions.max(axis=0) + 6,
            np.array(masque_coupe.shape),
        )

        i0, j0 = map(int, debut)
        i1, j1 = map(int, fin)

        masque_zoom = masque_coupe[i0:i1, j0:j1]

        ref = np.where(
            masque_zoom,
            reference[i0:i1, j0:j1, z],
            np.nan,
        )
        pred = np.where(
            masque_zoom,
            prediction[i0:i1, j0:j1, z],
            np.nan,
        )

        # Coordonnées de grille ; aucune latéralité patient déduite.
        x, y = np.meshgrid(
            np.arange(i0, i1),
            np.arange(j0, j1),
            indexing="ij",
        )

        maximum = max(
            float(np.nanmax(ref)),
            float(np.nanmax(pred)),
        )
        if maximum <= 0:
            raise ValueError(f"{patient} : maximum de dose nul")

        normalisation = Normalize(vmin=0, vmax=maximum)
        figure = plt.figure(figsize=(13, 6), layout="constrained")

        for numero, (titre, dose) in enumerate(
            [("Dose de référence", ref), ("Dose prédite", pred)],
            start=1,
        ):
            axe = figure.add_subplot(1, 2, numero, projection="3d")

            axe.plot_surface(
                x,
                y,
                dose,
                cmap="inferno",
                norm=normalisation,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
            )

            axe.set_title(titre)
            axe.set_xlabel("Indice axe 0")
            axe.set_ylabel("Indice axe 1")
            axe.set_zlabel("Dose (Gy)")
            axe.set_xlim(i0, i1 - 1)
            axe.set_ylim(j0, j1 - 1)
            axe.set_zlim(0, maximum * 1.05)
            axe.view_init(elev=35, azim=-60)
            axe.set_box_aspect((1, 1, 0.65))

        barre = plt.cm.ScalarMappable(
            norm=normalisation, cmap="inferno"
        )
        figure.colorbar(
            barre,
            ax=figure.axes,
            shrink=0.65,
            pad=0.06,
            label="Dose (Gy)",
        )

        reserve = (
            " | Réserve de qualité documentée"
            if patient == "P08"
            else ""
        )

        figure.suptitle(
            f"{patient} — test final : dose en relief{reserve}\n"
            f"Axe 2, indice {z} — mêmes échelles pour les deux cartes\n"
            "Hauteur = dose ; côtés du patient non indiqués",
            fontsize=12,
        )

        chemin = SORTIE / f"{patient}_relief_test.png"
        figure.savefig(chemin, dpi=180)
        plt.close(figure)

        print(
            f"{patient} | coupe {z} | "
            f"maximum référence : {np.nanmax(ref):.2f} Gy | "
            f"maximum prédit sur cette coupe : "
            f"{np.nanmax(pred):.2f} Gy",
            flush=True,
        )
        print(f"Figure : {chemin}", flush=True)

    print("\nTerminé : quatre comparaisons en relief créées.")


if __name__ == "__main__":
    main()