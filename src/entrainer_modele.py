from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
from datetime import datetime
import csv
import json
import random
import shutil
import time

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from essai_modele import PetitUNet


GRAINE = CONFIG["seed"]
EPOQUES = CONFIG["epochs"]
PATIENCE = CONFIG["patience"]
TAILLE_LOT = CONFIG["batch_size"]
TAUX_APPRENTISSAGE = CONFIG["learning_rate"]


def verifier(condition, message):
    if not condition:
        raise ValueError(message)


def lire_csv(chemin):
    with chemin.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


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


def sauver_modele(chemin, contenu):
    temporaire = chemin.with_suffix(".tmp")
    torch.save(contenu, temporaire)
    temporaire.replace(chemin)


class CoupesDataset(Dataset):
    def __init__(self, lot, patients, inventaire):
        self.volumes = {}
        self.coupes = []

        for ligne_patient in patients:
            patient = ligne_patient["patient"]
            dossier = lot / patient

            meta = json.loads(
                (dossier / "preparation.json").read_text(
                    encoding="utf-8"
                )
            )

            verifier(
                float(ligne_patient["prescription_totale_Gy"])
                == float(meta["prescription_totale_Gy"])
                and int(ligne_patient["nombre_fractions"])
                == int(meta["nombre_fractions"]),
                f"{patient} : informations de traitement differentes",
            )

            fichiers = {
                "irm": "t1c_minmax.nii.gz",
                "ann": "annotations.nii.gz",
                "dose": "dose_relative.nii.gz",
                "masque": "masque_dose_valide.nii.gz",
            }

            images = {
                nom: nib.load(dossier / fichier)
                for nom, fichier in fichiers.items()
            }

            reference = images["irm"]
            for nom, image in images.items():
                verifier(
                    image.shape == reference.shape
                    and np.allclose(
                        image.affine,
                        reference.affine,
                        rtol=0,
                        atol=1e-4,
                    ),
                    f"{patient} : grille differente pour {nom}",
                )

            verifier(
                reference.shape == (240, 240, 155),
                f"{patient} : dimensions inattendues",
            )

            irm = images["irm"].get_fdata(dtype=np.float32)
            ann = np.asarray(images["ann"].dataobj)
            masque_brut = np.asarray(images["masque"].dataobj)
            dose = images["dose"].get_fdata(dtype=np.float32)

            verifier(
                np.isfinite(irm).all()
                and irm.min() >= 0
                and irm.max() <= 1,
                f"{patient} : IRM invalide",
            )
            verifier(
                np.isin(ann, [0, 1, 2, 3]).all(),
                f"{patient} : annotations invalides",
            )
            verifier(
                np.isin(masque_brut, [0, 1]).all(),
                f"{patient} : masque invalide",
            )

            ann = ann.astype(np.uint8)
            masque = masque_brut.astype(bool)

            verifier(
                np.isfinite(dose[masque]).all()
                and (dose[masque] >= 0).all(),
                f"{patient} : dose valide incorrecte",
            )

            # En memoire seulement : valeur de remplissage hors masque.
            # Ces zeros ne participent jamais a l'erreur.
            dose[~masque] = 0.0

            self.volumes[patient] = {
                "irm": irm,
                "ann": ann,
                "dose": dose,
                "masque": masque,
                "prescription": float(meta["prescription_totale_Gy"]),
                "fractions": int(meta["nombre_fractions"]),
            }

            lignes = [
                ligne for ligne in inventaire
                if ligne["patient"] == patient
                and int(ligne["utilisable"]) == 1
            ]

            indices = [int(ligne["indice_axe2"]) for ligne in lignes]
            verifier(
                len(indices) == len(set(indices)),
                f"{patient} : coupes dupliquees",
            )

            for ligne, z in zip(lignes, indices):
                verifier(0 <= z < irm.shape[2], "Indice invalide")
                verifier(
                    int(masque[:, :, z].sum())
                    == int(ligne["pixels_dose_valides"]),
                    f"{patient}, coupe {z} : masque different du bilan",
                )
                self.coupes.append((patient, z))

            verifier(
                len(lignes) == int(ligne_patient["coupes_utilisables"]),
                f"{patient} : nombre de coupes incorrect",
            )
            verifier(
                np.any((ann > 0) & masque),
                f"{patient} : aucune annotation avec dose valide",
            )
            print(f"Charge : {patient}", flush=True)

    def __len__(self):
        return len(self.coupes)

    def __getitem__(self, indice):
        patient, z = self.coupes[indice]
        volume = self.volumes[patient]

        irm = volume["irm"][:, :, z]
        ann = volume["ann"][:, :, z]
        masque = volume["masque"][:, :, z]
        dose = volume["dose"][:, :, z]

        x = np.stack([
            irm,
            (ann == 1).astype(np.float32),
            (ann == 2).astype(np.float32),
            (ann == 3).astype(np.float32),
            np.full_like(irm, volume["prescription"] / 35),
            np.full_like(irm, volume["fractions"] / 7),
        ]).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(dose[None].copy()),
            torch.from_numpy(masque[None].copy()),
            torch.from_numpy((ann > 0)[None].copy()),
            patient,
        )


def erreur_apprentissage(prediction, cible, masque, annotations):
    erreur_globale = (
        (prediction[masque] - cible[masque]) ** 2
    ).mean()

    masque_ann = masque & annotations

    if masque_ann.any().item():
        erreur_ann = (
            (prediction[masque_ann] - cible[masque_ann]) ** 2
        ).mean()
        return 0.5 * erreur_globale + 0.5 * erreur_ann

    return erreur_globale


def constantes_reference(dataset):
    statistiques = {}

    for volume in dataset.volumes.values():
        groupe = volume["fractions"]
        valeurs = volume["dose"][volume["masque"]]

        total, nombre = statistiques.get(groupe, (0.0, 0))
        statistiques[groupe] = (
            total + float(valeurs.sum(dtype=np.float64)),
            nombre + int(valeurs.size),
        )

    return {
        groupe: total / nombre
        for groupe, (total, nombre) in statistiques.items()
    }


def evaluer(modele, chargeur, dataset, constantes=None):
    statistiques = {
        patient: {
            "somme_carres": 0.0,
            "somme_absolue": 0.0,
            "nombre": 0,
            "somme_carres_ann": 0.0,
            "somme_absolue_ann": 0.0,
            "nombre_ann": 0,
        }
        for patient in dataset.volumes
    }

    if modele is not None:
        modele.eval()

    with torch.no_grad():
        for x, y, masque, annotations, patients in chargeur:
            if modele is None:
                niveaux = [
                    constantes[dataset.volumes[p]["fractions"]]
                    for p in patients
                ]
                prediction = torch.tensor(
                    niveaux, dtype=y.dtype
                ).view(-1, 1, 1, 1).expand_as(y)
            else:
                prediction = modele(x)

            verifier(
                torch.isfinite(prediction).all().item(),
                "Prediction de validation non finie",
            )

            for i, patient in enumerate(patients):
                s = statistiques[patient]
                m = masque[i]
                ma = m & annotations[i]

                erreur = prediction[i][m] - y[i][m]
                erreur_ann = prediction[i][ma] - y[i][ma]

                s["somme_carres"] += float(
                    erreur.square().double().sum().item()
                )
                s["somme_absolue"] += float(
                    erreur.abs().double().sum().item()
                )
                s["nombre"] += erreur.numel()

                s["somme_carres_ann"] += float(
                    erreur_ann.square().double().sum().item()
                )
                s["somme_absolue_ann"] += float(
                    erreur_ann.abs().double().sum().item()
                )
                s["nombre_ann"] += erreur_ann.numel()

    details = []

    for patient, s in statistiques.items():
        verifier(
            s["nombre"] > 0 and s["nombre_ann"] > 0,
            f"{patient} : evaluation vide",
        )
        prescription = dataset.volumes[patient]["prescription"]

        mse = s["somme_carres"] / s["nombre"]
        mse_ann = s["somme_carres_ann"] / s["nombre_ann"]
        mae = s["somme_absolue"] / s["nombre"]
        mae_ann = s["somme_absolue_ann"] / s["nombre_ann"]

        details.append({
            "patient": patient,
            "fractions": dataset.volumes[patient]["fractions"],
            "mse_relative": mse,
            "mse_annotations": mse_ann,
            "mae_relative": mae,
            "mae_annotations": mae_ann,
            "mae_Gy": mae * prescription,
            "mae_annotations_Gy": mae_ann * prescription,
            "score": 0.5 * mse + 0.5 * mse_ann,
        })

    # Chaque patient pese autant dans les indicateurs de validation.
    resume = {
        cle: float(np.mean([d[cle] for d in details]))
        for cle in (
            "score", "mse_relative", "mse_annotations",
            "mae_relative", "mae_annotations",
            "mae_Gy", "mae_annotations_Gy",
        )
    }

    return {"resume": resume, "patients": details}


def main():
    racine = Path(__file__).resolve().parent.parent
    lot = path("prepared_dir")

    inventaires = [inventory()]
    inventaire = inventaires[0]

    repartition = lire_csv(inventaire / "patients_repartition.csv")
    coupes = lire_csv(inventaire / "inventaire_coupes.csv")

    ids = [ligne["patient"] for ligne in repartition]
    verifier(len(ids) == len(set(ids)), "Patients dupliques")

    patients_train = [
        p for p in repartition if p["ensemble"] == "entrainement"
    ]
    patients_val = [
        p for p in repartition if p["ensemble"] == "validation"
    ]

    verifier(
        len(patients_train) == 16 and len(patients_val) == 4,
        "Repartition inattendue",
    )

    random.seed(GRAINE)
    np.random.seed(GRAINE)
    torch.manual_seed(GRAINE)
    torch.set_num_threads(4)

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sortie = racine / "resultats" / f"entrainement_{horodatage}"
    sortie.mkdir(parents=True)

    shutil.copy2(__file__, sortie / "entrainer_modele.py")
    shutil.copy2(
        Path(__file__).with_name("essai_modele.py"),
        sortie / "essai_modele.py",
    )
    shutil.copy2(
        inventaire / "patients_repartition.csv",
        sortie / "patients_repartition.csv",
    )

    shutil.copy2(ROOT / "configs" / "experiment.json", sortie / "experiment.json")
    shutil.copy2(Path(__file__).with_name("configuration.py"), sortie / "configuration.py")
    sauver_json(sortie / "resolved_settings.json", CONFIG)
    configuration = {
        "lot": str(lot),
        "inventaire": str(inventaire),
        "graine": GRAINE,
        "epoques_max": EPOQUES,
        "patience": PATIENCE,
        "taille_lot": TAILLE_LOT,
        "taux_apprentissage": TAUX_APPRENTISSAGE,
        "appareil": "cpu",
        "canaux": [
            "IRM", "annotation_1", "annotation_2",
            "annotation_3", "prescription_sur_35", "fractions_sur_7",
        ],
        "perte": (
            "0.5 MSE valide + 0.5 MSE annotations valides ; "
            "MSE valide seule pour les lots sans annotation"
        ),
        "selection_modele": (
            "Moyenne par patient du score de validation "
            "0.5 MSE valide + 0.5 MSE annotations"
        ),
        "test_utilise": False,
    }
    sauver_json(sortie / "configuration.json", configuration)

    print(f"Resultats : {sortie}", flush=True)
    print("Chargement en memoire des patients...", flush=True)

    train = CoupesDataset(lot, patients_train, coupes)
    validation = CoupesDataset(lot, patients_val, coupes)

    generateur = torch.Generator().manual_seed(GRAINE)

    chargeur_train = DataLoader(
        train,
        batch_size=TAILLE_LOT,
        shuffle=True,
        num_workers=0,
        generator=generateur,
    )
    chargeur_val = DataLoader(
        validation,
        batch_size=TAILLE_LOT,
        shuffle=False,
        num_workers=0,
    )

    # Reference simple : une dose constante par fractionnement,
    # estimee uniquement sur les patients d'entrainement.
    constantes = constantes_reference(train)
    reference = evaluer(
        None, chargeur_val, validation, constantes
    )
    sauver_json(sortie / "reference_validation.json", {
        "constantes_par_nombre_fractions": constantes,
        **reference,
    })

    print(
        "Score de la reference constante en validation : "
        f"{reference['resume']['score']:.6f}",
        flush=True,
    )

    # Nouveau modele : les poids du petit essai ne sont pas reutilises.
    modele = PetitUNet()
    optimiseur = torch.optim.Adam(
        modele.parameters(), lr=TAUX_APPRENTISSAGE
    )

    meilleur_score = float("inf")
    meilleure_epoque = 0
    sans_progres = 0
    historique = []
    debut_total = time.perf_counter()
    etat = "termine"

    try:
        for epoque in range(1, EPOQUES + 1):
            debut = time.perf_counter()
            modele.train()

            somme_pertes = 0.0
            nombre_images = 0

            for numero, (x, y, masque, ann, _) in enumerate(
                chargeur_train, start=1
            ):
                optimiseur.zero_grad(set_to_none=True)
                prediction = modele(x)

                perte = erreur_apprentissage(
                    prediction, y, masque, ann
                )
                verifier(
                    torch.isfinite(perte).item(),
                    "Perte d'entrainement non finie",
                )

                perte.backward()
                norme = torch.nn.utils.clip_grad_norm_(
                    modele.parameters(), max_norm=5.0
                )
                verifier(
                    torch.isfinite(norme).item(),
                    "Gradients non finis",
                )
                optimiseur.step()

                somme_pertes += perte.item() * x.shape[0]
                nombre_images += x.shape[0]

                if numero % 100 == 0:
                    print(
                        f"Epoque {epoque} : "
                        f"lot {numero}/{len(chargeur_train)}",
                        flush=True,
                    )

            mesures = evaluer(
                modele, chargeur_val, validation
            )
            score = mesures["resume"]["score"]
            duree = time.perf_counter() - debut

            ligne = {
                "epoque": epoque,
                "perte_train_pendant_mises_a_jour":
                    somme_pertes / nombre_images,
                **{
                    f"validation_{cle}": valeur
                    for cle, valeur in mesures["resume"].items()
                },
                "duree_secondes": duree,
            }
            historique.append(ligne)

            with (sortie / "historique.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                redacteur = csv.DictWriter(
                    f, fieldnames=list(ligne), delimiter=";"
                )
                redacteur.writeheader()
                redacteur.writerows(historique)

            checkpoint = {
                "epoque": epoque,
                "modele": modele.state_dict(),
                "optimiseur": optimiseur.state_dict(),
                "configuration": configuration,
                "validation": mesures,
            }

            sauver_modele(sortie / "dernier_modele.pt", checkpoint)

            if score < meilleur_score:
                meilleur_score = score
                meilleure_epoque = epoque
                sans_progres = 0

                sauver_modele(
                    sortie / "meilleur_modele.pt", checkpoint
                )
                sauver_json(
                    sortie / "meilleure_validation.json", mesures
                )
            else:
                sans_progres += 1

            print(
                f"Epoque {epoque}/{EPOQUES} | "
                f"train={ligne['perte_train_pendant_mises_a_jour']:.6f} | "
                f"val={score:.6f} | "
                f"MAE annotations={mesures['resume']['mae_annotations_Gy']:.3f} Gy | "
                f"{duree:.1f} s",
                flush=True,
            )

            if sans_progres >= PATIENCE:
                etat = "arret anticipe"
                print("Arret : validation sans amelioration.", flush=True)
                break

    except KeyboardInterrupt:
        etat = "interrompu par utilisateur"
        print(
            "\nInterruption : les sauvegardes des epoques "
            "completes restent disponibles.",
            flush=True,
        )

    duree_totale = time.perf_counter() - debut_total

    bilan = [
        "ENTRAINEMENT DU MODELE DE DOSE 2D",
        f"Etat : {etat}",
        f"Patients entrainement : {len(patients_train)}",
        f"Patients validation : {len(patients_val)}",
        f"Coupes entrainement : {len(train)}",
        f"Coupes validation : {len(validation)}",
        f"Epoques completes : {len(historique)}",
        f"Meilleure epoque : {meilleure_epoque}",
        f"Duree de la boucle : {duree_totale / 60:.2f} minutes",
        f"Score reference constante : "
        f"{reference['resume']['score']:.6f}",
    ]

    if meilleure_epoque:
        bilan.append(
            f"Meilleur score modele : {meilleur_score:.6f}"
        )
        bilan.append(
            "Modele meilleur que la reference sur ce score : "
            + ("oui" if meilleur_score < reference["resume"]["score"]
               else "non")
        )

    bilan += [
        "Les scores sont calcules en dose relative.",
        "Les annotations ne sont pas assimilees aux PTV.",
        "Ensemble de test : non utilise.",
        f"Dossier : {sortie}",
    ]

    (sortie / "bilan_entrainement.txt").write_text(
        "\n".join(bilan), encoding="utf-8"
    )
    print("\n" + "\n".join(bilan))


if __name__ == "__main__":
    main()