from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
from datetime import datetime
import hashlib
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

SEUIL_DICE = 0.95


def sauver_json(chemin, contenu):
    chemin.write_text(
        json.dumps(
            contenu,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def dice_dose(prediction, reference, valide):
    zone_predite = (prediction >= SEUIL_DICE) & valide
    zone_reference = (reference >= SEUIL_DICE) & valide

    n_pred = int(zone_predite.sum())
    n_ref = int(zone_reference.sum())
    intersection = int((zone_predite & zone_reference).sum())

    if n_pred + n_ref == 0:
        valeur = None
    else:
        valeur = 2 * intersection / (n_pred + n_ref)

    return {
        "dice": valeur,
        "voxels_zone_predite": n_pred,
        "voxels_zone_reference": n_ref,
        "voxels_intersection": intersection,
    }


def contours(ax, annotations):
    for etiquette, couleur in (
        (1, "orange"), (2, "cyan"), (3, "lime")
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


def figure_test(patient, volume, prediction, sortie):
    valide = volume["masque"]
    prescription = volume["prescription"]

    reference_gy = volume["dose"] * prescription
    prediction_gy = prediction * prescription
    erreur = prediction_gy - reference_gy

    tailles = np.count_nonzero(
        (volume["ann"] > 0) & valide,
        axis=(0, 1),
    )

    maximum_reference = np.unravel_index(
        np.argmax(np.where(valide, reference_gy, -np.inf)),
        reference_gy.shape,
    )
    maximum_prediction = np.unravel_index(
        np.argmax(np.where(valide, prediction_gy, -np.inf)),
        prediction_gy.shape,
    )

    choix = [
        ("Plus de pixels annotés valides", int(np.argmax(tailles))),
        ("Maximum de référence", int(maximum_reference[2])),
        ("Maximum prédit", int(maximum_prediction[2])),
    ]

    maximum_dose = max(
        float(reference_gy[valide].max()),
        float(prediction_gy[valide].max()),
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

    for ligne, (raison, z) in enumerate(choix):
        m = valide[:, :, z]

        axes[ligne, 0].imshow(
            volume["irm"][:, :, z].T,
            cmap="gray", vmin=0, vmax=1,
            origin="lower", interpolation="nearest",
        )

        for colonne, carte in (
            (1, reference_gy),
            (2, prediction_gy),
        ):
            image_dose = axes[ligne, colonne].imshow(
                np.ma.array(carte[:, :, z], mask=~m).T,
                cmap="inferno",
                vmin=0, vmax=maximum_dose,
                origin="lower", interpolation="nearest",
            )

        image_erreur = axes[ligne, 3].imshow(
            np.ma.array(erreur[:, :, z], mask=~m).T,
            cmap="RdBu_r",
            vmin=-5, vmax=5,
            origin="lower", interpolation="nearest",
        )

        for colonne in range(4):
            ax = axes[ligne, colonne]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{titres[colonne]}\nAxe 2, indice {z}",
                fontsize=10,
            )
            contours(ax, volume["ann"][:, :, z])

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
        label="Écart (Gy) ; couleurs saturées au-delà de ±5 Gy",
        extend="both",
        shrink=0.8,
    )

    reserve = " | Réserve de qualité documentée" if patient == "P08" else ""
    fig.suptitle(
        f"{patient} — test final{reserve}\n"
        f"{prescription:g} Gy / {volume['fractions']} fraction(s)\n"
        "Annotations : 1 orange, 2 cyan, 3 vert. "
        "Indices de grille ; côtés du patient non indiqués.",
        fontsize=12,
    )

    fig.savefig(
        sortie / f"{patient}_comparaison_test.png",
        dpi=150,
    )
    plt.close(fig)

    return {
        "coupes_affichees": [
            {"raison": raison, "indice_axe2": z}
            for raison, z in choix
        ],
        "maximum_reference_Gy": float(reference_gy[valide].max()),
        "maximum_prediction_Gy": float(prediction_gy[valide].max()),
        "indices_max_reference": [int(v) for v in maximum_reference],
        "indices_max_prediction": [int(v) for v in maximum_prediction],
    }


def sauvegarder_prediction(lot, patient, prediction, valide, sortie):
    reference = nib.load(
        lot / patient / "dose_relative.nii.gz"
    )

    valeurs = np.where(
        valide, prediction, np.nan
    ).astype(np.float32)

    entete = reference.header.copy()
    entete.set_data_dtype(np.float32)
    entete.set_slope_inter(1, 0)

    image = nib.Nifti1Image(
        valeurs, reference.affine, header=entete
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


def main():
    racine = Path(__file__).resolve().parent.parent
    entrainement = path("training_dir")
    chemin_modele = entrainement / "meilleur_modele.pt"

    torch.set_num_threads(4)

    checkpoint = torch.load(
        chemin_modele,
        map_location="cpu",
        weights_only=True,
    )
    verifier(
        CONFIG["expected_epoch"] is None or checkpoint["epoque"] == CONFIG["expected_epoch"],
        "Epoque differente de la configuration.",
    )

    configuration = checkpoint["configuration"]
    lot = path("prepared_dir")
    inventaire = inventory()

    repartition = lire_csv(
        entrainement / "patients_repartition.csv"
    )
    patients = [
        p for p in repartition if p["ensemble"] == "test"
    ]

    verifier(
        len(patients) == len(CONFIG["test_patients"])
        and {p["patient"] for p in patients}
        == set(CONFIG["test_patients"]),
        "Patients de test inattendus.",
    )

    # Recuperer la reference sans la recalculer sur le test.
    ancienne_reference = json.loads(
        (entrainement / "reference_validation.json").read_text(
            encoding="utf-8"
        )
    )
    constantes = {
        int(cle): float(valeur)
        for cle, valeur in ancienne_reference[
            "constantes_par_nombre_fractions"
        ].items()
    }

    coupes = lire_csv(inventaire / "inventaire_coupes.csv")
    dataset = CoupesDataset(lot, patients, coupes)
    verifier(CONFIG["test_slices"] is None or len(dataset) == CONFIG["test_slices"], "Nombre de coupes different de la configuration.")

    chargeur = DataLoader(
        dataset, batch_size=4,
        shuffle=False, num_workers=0,
    )

    modele = PetitUNet()
    modele.load_state_dict(checkpoint["modele"], strict=True)
    modele.eval()

    # Bloquer explicitement les gradients.
    for parametre in modele.parameters():
        parametre.requires_grad_(False)

    mesures_modele = evaluer(modele, chargeur, dataset)
    mesures_reference = evaluer(
        None, chargeur, dataset, constantes
    )

    # Ajouter RMSE par patient puis moyenne entre patients.
    for mesures in (mesures_modele, mesures_reference):
        for d in mesures["patients"]:
            prescription = dataset.volumes[
                d["patient"]
            ]["prescription"]

            d["rmse_relative"] = float(np.sqrt(d["mse_relative"]))
            d["rmse_annotations_relative"] = float(
                np.sqrt(d["mse_annotations"])
            )
            d["rmse_Gy"] = d["rmse_relative"] * prescription
            d["rmse_annotations_Gy"] = (
                d["rmse_annotations_relative"] * prescription
            )

        for cle in (
            "rmse_relative", "rmse_annotations_relative",
            "rmse_Gy", "rmse_annotations_Gy",
        ):
            mesures["resume"][cle] = float(np.mean([
                d[cle] for d in mesures["patients"]
            ]))

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sortie = ROOT / "resultats" / f"test_final_{horodatage}"
    sortie.mkdir(parents=True)

    protocole = {
        "modele": str(chemin_modele),
        "modele_SHA256": hashlib.sha256(
            chemin_modele.read_bytes()
        ).hexdigest(),
        "epoque": int(checkpoint["epoque"]),
        "patients": sorted(dataset.volumes),
        "seuil_dice_dose_relative": SEUIL_DICE,
        "domaine": "Masque de dose valide de chaque patient",
        "dice": (
            "Calcul par patient sur les voxels du volume forme "
            "par les coupes 2D ; moyenne entre patients"
        ),
        "deux_zones_vides": (
            "Dice non defini, exclu de la moyenne et signale"
        ),
        "une_zone_vide": "Dice nul",
        "resume_erreurs": (
            "Moyenne des indicateurs par patient, "
            "avec le meme poids pour chaque patient"
        ),
        "constantes_reference": constantes,
        "reserve_P08": (
            "Correspondance entre toutes les annotations et "
            "les cibles traitees seulement partiellement clarifiee"
        ),
        "poids_modifies": False,
    }
    sauver_json(sortie / "protocole_test.json", protocole)

    details = []

    for patient, volume in dataset.volumes.items():
        print(f"Cartes de test : {patient}", flush=True)

        prediction = np.full(
            volume["irm"].shape, np.nan, dtype=np.float32
        )
        indices = [
            i for i, (p, _) in enumerate(dataset.coupes)
            if p == patient
        ]

        with torch.no_grad():
            for debut in range(0, len(indices), 4):
                groupe = indices[debut:debut + 4]
                x = torch.stack([
                    dataset[i][0] for i in groupe
                ])
                cartes = modele(x)[:, 0].numpy()

                verifier(
                    np.isfinite(cartes).all()
                    and (cartes >= 0).all(),
                    "Predictions invalides.",
                )

                for i, carte in zip(groupe, cartes):
                    z = dataset.coupes[i][1]
                    prediction[:, :, z] = carte

        valide = volume["masque"]
        verifier(
            np.isfinite(prediction[valide]).all(),
            "Predictions manquantes dans le domaine valide.",
        )

        prediction_constante = np.full(
            prediction.shape,
            constantes[volume["fractions"]],
            dtype=np.float32,
        )

        detail = {
            "patient": patient,
            "reserve_qualite": patient == "P08",
            "modele": dice_dose(
                prediction, volume["dose"], valide
            ),
            "reference_constante": dice_dose(
                prediction_constante, volume["dose"], valide
            ),
            **figure_test(
                patient, volume, prediction, sortie
            ),
        }
        details.append(detail)

        sauvegarder_prediction(
            lot, patient, prediction, valide, sortie
        )

    for nom, mesures in (
        ("modele", mesures_modele),
        ("reference_constante", mesures_reference),
    ):
        valeurs = [
            d[nom]["dice"] for d in details
            if d[nom]["dice"] is not None
        ]

        mesures["resume"]["dice_095"] = (
            float(np.mean(valeurs)) if valeurs else None
        )
        mesures["resume"]["patients_dice_defini"] = len(valeurs)

    sauver_json(
        sortie / "resultats_test.json",
        {
            "modele": mesures_modele,
            "reference_constante": mesures_reference,
            "dice_et_figures": details,
        },
    )

    bilan = [
        "EVALUATION FINALE SUR LES PATIENTS DE TEST",
        f"Modele fixe : epoque {checkpoint['epoque']}",
        "Patients : " + ", ".join(sorted(dataset.volumes)),
        f"Coupes utilisables : {len(dataset)}",
        "Moyennes donnant le meme poids a chaque patient",
        "Dice : zones de dose relative >= 0.95",
        "",
    ]

    for titre, mesures in (
        ("MODELE", mesures_modele),
        ("REFERENCE CONSTANTE", mesures_reference),
    ):
        r = mesures["resume"]
        dice = r["dice_095"]
        texte_dice = (
            f"{dice:.6f}" if dice is not None else "non defini"
        )
        bilan += [
            titre,
            f"Score combine : {r['score']:.6f}",
            f"MAE region : {r['mae_Gy']:.3f} Gy",
            f"MAE annotations : {r['mae_annotations_Gy']:.3f} Gy",
            f"RMSE region : {r['rmse_Gy']:.3f} Gy",
            f"RMSE annotations : {r['rmse_annotations_Gy']:.3f} Gy",
            f"Dice moyen : {texte_dice}",
            f"Patients avec Dice defini : "
            f"{r['patients_dice_defini']}/4",
            "",
        ]

    for d in mesures_modele["patients"]:
        detail = next(
            x for x in details if x["patient"] == d["patient"]
        )
        bilan.append(
            f"{d['patient']} : "
            f"MAE region={d['mae_Gy']:.3f} Gy ; "
            f"MAE annotations={d['mae_annotations_Gy']:.3f} Gy ; "
            f"Dice={detail['modele']['dice']}"
        )

    bilan += [
        "",
        "P08 : reserve de qualite documentee, patient conserve.",
        "Dice de zone de dose ; aucune segmentation tumorale evaluee.",
        "Les annotations ne sont pas assimilees aux PTV.",
        "Les maxima ne correspondent pas necessairement au meme voxel.",
        "Poids et reglages du modele : non modifies.",
        f"Sortie : {sortie}",
    ]

    (sortie / "bilan_test.txt").write_text(
        "\n".join(bilan), encoding="utf-8"
    )
    print("\n" + "\n".join(bilan))


if __name__ == "__main__":
    main()