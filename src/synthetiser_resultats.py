from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
import csv
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RACINE = Path(__file__).resolve().parents[1]

ENTRAINEMENT = path("training_dir")

TEST = path("test_dir")
SORTIE = ROOT / "resultats" / "synthese_rapport"


def lire_json(chemin):
    with chemin.open(encoding="utf-8-sig") as fichier:
        return json.load(fichier)


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)

    with (ENTRAINEMENT / "historique.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fichier:
        historique = list(csv.DictReader(fichier, delimiter=";"))

    if not historique:
        raise ValueError("Historique vide.")

    epoques = np.array([
        int(ligne["epoque"]) for ligne in historique
    ])

    scores = np.array([
        float(ligne["validation_score"]) for ligne in historique
    ])

    if not np.isfinite(scores).all():
        raise ValueError("Score de validation non fini.")

    meilleur = int(np.argmin(scores))
    epoque_retenue = int(epoques[meilleur])

    # 1. Courbe de validation.
    figure, axe = plt.subplots(figsize=(9, 5))

    axe.plot(
        epoques, scores,
        marker="o", color="tab:blue",
        label="Score de validation"
    )

    axe.scatter(
        [epoque_retenue], [scores[meilleur]],
        color="tab:red", s=75, zorder=3,
        label=f"Modèle retenu : époque {epoque_retenue}"
    )

    axe.axvline(
        epoque_retenue,
        color="tab:red", linestyle="--", alpha=0.5
    )

    axe.set(
        xlabel="Époque",
        ylabel="Score combiné sur les doses relatives",
        title="Sélection du modèle sur les 4 patients de validation"
    )
    axe.set_xticks(epoques)
    axe.grid(alpha=0.25)
    axe.legend()

    figure.tight_layout()
    figure.savefig(
        SORTIE / "courbe_validation.png", dpi=200
    )
    plt.close(figure)

    # 2. Lecture des mesures du test final.
    test = lire_json(TEST / "resultats_test.json")

    patients = test["modele"]["patients"]
    constantes = {
        ligne["patient"]: ligne
        for ligne in test["reference_constante"]["patients"]
    }
    dices = {
        ligne["patient"]: ligne["modele"]["dice"]
        for ligne in test["dice_et_figures"]
    }

    noms = [ligne["patient"] for ligne in patients]

    if set(noms) != set(CONFIG["test_patients"]):
        raise ValueError("Liste inattendue de patients du test.")

    x = np.arange(len(noms))
    largeur = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    comparaisons = [
        ("mae_Gy", "MAE dans la région valide"),
        ("mae_annotations_Gy", "MAE dans les annotations valides"),
    ]

    for axe, (cle, titre) in zip(axes, comparaisons):
        valeurs_modele = [ligne[cle] for ligne in patients]
        valeurs_constantes = [
            constantes[nom][cle] for nom in noms
        ]

        barres_reference = axe.bar(
            x - largeur / 2,
            valeurs_constantes,
            largeur,
            label="Référence constante",
            color="silver",
        )
        barres_modele = axe.bar(
            x + largeur / 2,
            valeurs_modele,
            largeur,
            label="Modèle",
            color="steelblue",
        )

        axe.bar_label(barres_reference, fmt="%.2f", padding=3)
        axe.bar_label(barres_modele, fmt="%.2f", padding=3)

        axe.set_xticks(x, [nom + "*" if nom == "P08" else nom for nom in noms])
        axe.set_ylabel("Erreur absolue moyenne (Gy)")
        axe.set_title(titre)
        axe.set_ylim(
            0,
            max(valeurs_modele + valeurs_constantes) * 1.20
        )
        axe.grid(axis="y", alpha=0.2)
        axe.set_axisbelow(True)
        axe.legend(fontsize=9)

    figure.suptitle(
        "Test final — comparaison par patient\n"
        "* P08 : réserve de qualité documentée, conservé dans le bilan"
    )

    figure.tight_layout()
    figure.savefig(
        SORTIE / "erreurs_test_par_patient.png", dpi=200
    )
    plt.close(figure)

    # 3. Tableau des métriques du modèle et de la référence.
    colonnes = [
        "patient",
        "MAE_region_modele_Gy",
        "MAE_region_constante_Gy",
        "MAE_annotations_modele_Gy",
        "MAE_annotations_constante_Gy",
        "RMSE_annotations_modele_Gy",
        "Dice_095_modele",
    ]

    def construire_ligne(nom, modele, constante, dice):
        return {
            "patient": nom,
            "MAE_region_modele_Gy": modele["mae_Gy"],
            "MAE_region_constante_Gy": constante["mae_Gy"],
            "MAE_annotations_modele_Gy":
                modele["mae_annotations_Gy"],
            "MAE_annotations_constante_Gy":
                constante["mae_annotations_Gy"],
            "RMSE_annotations_modele_Gy":
                modele["rmse_annotations_Gy"],
            "Dice_095_modele": dice,
        }

    lignes = [
        construire_ligne(
            ligne["patient"],
            ligne,
            constantes[ligne["patient"]],
            dices[ligne["patient"]],
        )
        for ligne in patients
    ]

    lignes.append(construire_ligne(
        "MOYENNE_PAR_PATIENT",
        test["modele"]["resume"],
        test["reference_constante"]["resume"],
        test["modele"]["resume"]["dice_095"],
    ))

    with (SORTIE / "tableau_test.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fichier:
        redacteur = csv.DictWriter(
            fichier, fieldnames=colonnes, delimiter=";"
        )
        redacteur.writeheader()
        redacteur.writerows(lignes)

    print(f"Époque retenue : {epoque_retenue}")
    print(f"Score de validation : {scores[meilleur]:.6f}")
    print(f"\nRésultats enregistrés dans : {SORTIE}")

    for nom in [
        "courbe_validation.png",
        "erreurs_test_par_patient.png",
        "tableau_test.csv",
    ]:
        print(f"  - {nom}")


if __name__ == "__main__":
    main()