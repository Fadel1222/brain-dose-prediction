from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
from datetime import datetime
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from essai_modele import PetitUNet
from entrainer_modele import (
    CoupesDataset,
    lire_csv,
    evaluer,
    verifier,
)


def afficher_contours(ax, annotations):
    for etiquette, couleur in (
        (1, "orange"),
        (2, "cyan"),
        (3, "lime"),
    ):
        masque = annotations == etiquette

        if masque.any() and not masque.all():
            ax.contour(
                masque.T.astype(float),
                levels=[0.5],
                colors=[couleur],
                linewidths=0.7,
                origin="lower",
            )


def creer_figure(patient, volume, prediction, sortie):
    irm = volume["irm"]
    annotations = volume["ann"]
    masque = volume["masque"]
    prescription = volume["prescription"]

    reference_gy = volume["dose"] * prescription
    prediction_gy = prediction * prescription

    erreur = prediction_gy - reference_gy

    # Choix des coupes sans selection manuelle.
    tailles = np.count_nonzero(
        (annotations > 0) & masque, axis=(0, 1)
    )
    z_annotations = int(np.argmax(tailles))

    indice_reference = np.unravel_index(
        np.argmax(np.where(masque, reference_gy, -np.inf)),
        reference_gy.shape,
    )
    indice_prediction = np.unravel_index(
        np.argmax(np.where(masque, prediction_gy, -np.inf)),
        prediction_gy.shape,
    )

    coupes = [
        ("Maximum de pixels annotés valides", z_annotations),
        ("Maximum de dose de référence", int(indice_reference[2])),
        ("Maximum de dose prédite valide", int(indice_prediction[2])),
    ]

    # Meme echelle pour reference et prediction, sur les trois coupes.
    maximum_dose = max(
        float(reference_gy[masque].max()),
        float(prediction_gy[masque].max()),
        1e-6,
    )
    maximum_erreur = max(
        float(np.abs(erreur[masque]).max()),
        1e-6,
    )

    fig, axes = plt.subplots(
        3, 4, figsize=(16, 12), layout="constrained"
    )

    titres = [
        "IRM et annotations",
        "Dose de référence",
        "Dose prédite",
        "Erreur : prédiction − référence",
    ]

    for ligne, (raison, z) in enumerate(coupes):
        m = masque[:, :, z]

        cartes = [
            np.ma.array(reference_gy[:, :, z], mask=~m).T,
            np.ma.array(prediction_gy[:, :, z], mask=~m).T,
            np.ma.array(erreur[:, :, z], mask=~m).T,
        ]

        axes[ligne, 0].imshow(
            irm[:, :, z].T,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )

        for colonne in (1, 2):
            image_dose = axes[ligne, colonne].imshow(
                cartes[colonne - 1],
                cmap="inferno",
                vmin=0,
                vmax=maximum_dose,
                origin="lower",
                interpolation="nearest",
            )

        image_erreur = axes[ligne, 3].imshow(
            cartes[2],
            cmap="RdBu_r",
            vmin=-maximum_erreur,
            vmax=maximum_erreur,
            origin="lower",
            interpolation="nearest",
        )

        for colonne in range(4):
            ax = axes[ligne, colonne]
            ax.set_facecolor("black")
            ax.set_title(
                f"{titres[colonne]}\nAxe 2, indice {z}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])

            afficher_contours(
                ax, annotations[:, :, z]
            )

        axes[ligne, 0].set_ylabel(raison, fontsize=9)

    fig.colorbar(
        image_dose,
        ax=axes[:, 1:3].ravel().tolist(),
        label="Dose (Gy)",
        shrink=0.8,
    )
    fig.colorbar(
        image_erreur,
        ax=axes[:, 3].ravel().tolist(),
        label="Écart signé (Gy) : rouge = surestimation",
        shrink=0.8,
    )

    fig.suptitle(
        f"{patient} — validation du meilleur modèle\n"
        f"{prescription:g} Gy / {volume['fractions']} fraction(s) | "
        "Annotations : 1 orange, 2 cyan, 3 vert\n"
        "Indices de grille ; côtés du patient non indiqués. "
        "Dose affichée uniquement dans le masque valide.",
        fontsize=12,
    )

    chemin = sortie / f"{patient}_comparaison_validation.png"
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

    return {
        "patient": patient,
        "coupes_affichees": [
            {"raison": raison, "indice_axe2": z}
            for raison, z in coupes
        ],
        "maximum_reference_Gy": float(reference_gy[masque].max()),
        "maximum_prediction_Gy": float(prediction_gy[masque].max()),
        "indices_max_reference": [int(v) for v in indice_reference],
        "indices_max_prediction": [int(v) for v in indice_prediction],
    }


def main():
    racine = Path(__file__).resolve().parent.parent

    entrainement = path("training_dir")
    chemin_modele = entrainement / "meilleur_modele.pt"

    verifier(chemin_modele.is_file(), "Meilleur modele introuvable")

    torch.set_num_threads(4)

    checkpoint = torch.load(
        chemin_modele,
        map_location="cpu",
        weights_only=True,
    )

    configuration = checkpoint["configuration"]
    lot = path("prepared_dir")
    inventaire = inventory()

    repartition = lire_csv(
        entrainement / "patients_repartition.csv"
    )
    patients = [
        p for p in repartition
        if p["ensemble"] == "validation"
    ]
    verifier(len(patients) == 4, "Quatre patients attendus")

    coupes = lire_csv(inventaire / "inventaire_coupes.csv")
    dataset = CoupesDataset(lot, patients, coupes)

    modele = PetitUNet()
    modele.load_state_dict(checkpoint["modele"], strict=True)
    modele.eval()

    # Recalculer le score pour verifier le modele charge.
    chargeur = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    mesures = evaluer(modele, chargeur, dataset)

    score_attendu = float(
        checkpoint["validation"]["resume"]["score"]
    )
    score_recalcule = mesures["resume"]["score"]

    verifier(
        np.isclose(
            score_recalcule, score_attendu,
            rtol=1e-4, atol=1e-7,
        ),
        "Score different du checkpoint : verifier les fichiers.",
    )

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sortie = ROOT / "resultats" / f"visualisation_validation_{horodatage}"
    sortie.mkdir(parents=True)

    details = []

    for patient, volume in dataset.volumes.items():
        print(f"Prediction : {patient}", flush=True)

        forme = volume["irm"].shape
        prediction = np.full(forme, np.nan, dtype=np.float32)

        # Predire uniquement les coupes utilisees en validation.
        indices = [
            i for i, (p, _) in enumerate(dataset.coupes)
            if p == patient
        ]

        with torch.no_grad():
            for debut in range(0, len(indices), 4):
                lot_indices = indices[debut:debut + 4]
                x = torch.stack([
                    dataset[i][0] for i in lot_indices
                ])
                cartes = modele(x)[:, 0].numpy()

                verifier(
                    np.isfinite(cartes).all(),
                    "Prediction non finie",
                )

                for i, carte in zip(lot_indices, cartes):
                    z = dataset.coupes[i][1]
                    prediction[:, :, z] = carte

        verifier(
            np.isfinite(prediction[volume["masque"]]).all(),
            "Prediction manquante dans le masque valide",
        )

        details.append(
            creer_figure(patient, volume, prediction, sortie)
        )

        # Sauvegarde numerique pour les calculs ulterieurs.
        # Les valeurs hors masque valide sont laissees a NaN.
        prediction_sauvee = np.where(
            volume["masque"], prediction, np.nan
        ).astype(np.float32)

        reference = nib.load(
            lot / patient / "dose_relative.nii.gz"
        )
        entete = reference.header.copy()
        entete.set_data_dtype(np.float32)
        entete.set_slope_inter(1, 0)

        image = nib.Nifti1Image(
            prediction_sauvee,
            reference.affine,
            header=entete,
        )
        image.set_qform(
            reference.get_qform(),
            int(reference.header["qform_code"]),
        )
        image.set_sform(
            reference.get_sform(),
            int(reference.header["sform_code"]),
        )

        nib.save(
            image,
            sortie / f"{patient}_dose_relative_predite.nii.gz",
        )

    bilan = [
        "CONTROLE VISUEL DES PREDICTIONS DE VALIDATION",
        f"Epoque du modele : {checkpoint['epoque']}",
        f"Score attendu : {score_attendu:.9f}",
        f"Score recalcule : {score_recalcule:.9f}",
        "Correspondance du score : verifiee",
        "Erreur affichee : prediction - reference, en Gy",
        "Rouge : surestimation ; bleu : sous-estimation",
        "Test : non utilise",
        "",
    ]

    for d in details:
        bilan.append(
            f"{d['patient']} : maximum reference "
            f"{d['maximum_reference_Gy']:.3f} Gy ; "
            f"maximum prediction {d['maximum_prediction_Gy']:.3f} Gy"
        )
        for coupe in d["coupes_affichees"]:
            bilan.append(
                f"  {coupe['raison']} : "
                f"axe 2, indice {coupe['indice_axe2']}"
            )

    (sortie / "bilan_visualisation.txt").write_text(
        "\n".join(bilan), encoding="utf-8"
    )
    (sortie / "details_visualisation.json").write_text(
        json.dumps(
            details, ensure_ascii=False,
            indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "\n".join(bilan))
    print(f"\nFigures et predictions : {sortie}")


if __name__ == "__main__":
    main()