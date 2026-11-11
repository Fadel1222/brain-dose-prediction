"""Preparation numerique uniquement ; aucune validation clinique automatique.
Placer ce script dans src/ puis lancer : python src/preparer_candidats.py
Dependances : numpy, nibabel, pydicom (deja utilisees dans le projet).
"""
from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
from datetime import datetime
from collections import Counter
import csv
import json
import shutil
import sys

import nibabel as nib
import numpy as np
import pydicom


def verifier(condition, message):
    if not condition:
        raise ValueError(message)


def unique(elements, description):
    elements = list(elements)
    verifier(len(elements) == 1, f"{description} : {len(elements)} correspondances")
    return elements[0]


def normaliser(irm, dose, region, prescription):
    verifier(np.isfinite(irm).all(), "IRM non finie")
    minimum, maximum = float(irm.min()), float(irm.max())
    verifier(maximum > minimum, "IRM constante")
    verifier(np.isfinite(prescription) and prescription > 0, "Prescription invalide")
    irm_norm = ((irm.astype(np.float64) - minimum) / (maximum - minimum)).astype(np.float32)
    valide = region & np.isfinite(dose) & (dose >= 0)
    verifier(valide.any(), "Aucune dose valide dans la region")
    relative = np.full(dose.shape, np.nan, dtype=np.float32)
    relative[valide] = dose[valide] / prescription
    verifier(np.isfinite(relative[valide]).all(), "Debordement de dose relative")
    return irm_norm, relative, valide, minimum, maximum


def lire_plan(dossier):
    objets = {"RTPLAN": [], "RTDOSE": []}
    for chemin in sorted(dossier.rglob("*")):
        if not chemin.is_file() or chemin.suffix.lower() != ".dcm" or "__MACOSX" in chemin.parts:
            continue
        ds = pydicom.dcmread(chemin, stop_before_pixels=True)
        if getattr(ds, "Modality", "") in objets:
            objets[ds.Modality].append((chemin, ds))
    chemin_rd, rd = unique(objets["RTDOSE"], "RTDOSE")
    for nom, attendu in (("DoseUnits", "GY"), ("DoseType", "PHYSICAL"), ("DoseSummationType", "PLAN")):
        verifier(getattr(rd, nom, None) == attendu, f"RTDOSE : {nom} inattendu")
    lien = unique(getattr(rd, "ReferencedRTPlanSequence", []), "Reference au RTPLAN")
    chemin_rp, rp = unique(
        ((p, ds) for p, ds in objets["RTPLAN"] if str(ds.SOPInstanceUID) == str(lien.ReferencedSOPInstanceUID)),
        "RTPLAN associe",
    )
    cibles = [r for r in getattr(rp, "DoseReferenceSequence", []) if getattr(r, "DoseReferenceType", "") == "TARGET"]
    verifier(bool(cibles), "Aucune reference TARGET")
    verifier(all(hasattr(r, "TargetPrescriptionDose") for r in cibles), "Prescription TARGET absente")
    prescriptions = [float(r.TargetPrescriptionDose) for r in cibles]
    verifier(all(np.isfinite(v) and v > 0 for v in prescriptions), "Prescription invalide")
    verifier(np.allclose(prescriptions, prescriptions[0], rtol=0, atol=1e-6), "Prescriptions multiples differentes")
    groupe = unique(getattr(rp, "FractionGroupSequence", []), "Groupe de fractions")
    fractions = int(groupe.NumberOfFractionsPlanned)
    verifier(fractions > 0, "Nombre de fractions invalide")
    return {
        "prescription_totale_Gy": prescriptions[0], "nombre_fractions": fractions,
        "RTPLAN": str(chemin_rp.relative_to(dossier)),
        "RTDOSE": str(chemin_rd.relative_to(dossier)),
        "RTPLAN_UID": str(rp.SOPInstanceUID), "RTDOSE_UID": str(rd.SOPInstanceUID),
        "descriptions_cibles": [str(getattr(r, "DoseReferenceDescription", "")) for r in cibles],
    }


def sauvegarder(tableau, reference, chemin, dtype):
    entete = reference.header.copy()
    entete.set_data_dtype(dtype)
    entete.set_slope_inter(1, 0)
    image = nib.Nifti1Image(tableau.astype(dtype), reference.affine, header=entete)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, chemin)


def preparer(racine, patient, destination):
    dossier = racine / "data" / patient
    fichiers = [p for p in dossier.rglob("*") if p.is_file() and "__MACOSX" not in p.parts]
    chemins = {
        "IRM": unique((p for p in fichiers if p.name.lower() == "t1c.nii.gz" and p.parent.name.lower() == "baseline"), "IRM baseline"),
        "Annotations": unique((p for p in fichiers if p.name.lower() == f"{patient}_tumor_mask_baseline.nii.gz".lower()), "Annotations baseline"),
        "Cerveau": unique((p for p in fichiers if p.name.lower() == f"{patient}_brain_mask.nii.gz".lower()), "Masque cerebral"),
        "Dose": unique((p for p in fichiers if p.name.lower() == f"{patient}_RTP.nii.gz".lower()), "Dose NIfTI"),
    }
    images = {nom: nib.load(p) for nom, p in chemins.items()}
    reference = images["IRM"]
    verifier(len(reference.shape) == 3, "IRM 3D attendue")
    verifier(np.isfinite(reference.affine).all() and abs(np.linalg.det(reference.affine[:3, :3])) > 1e-12, "Affine invalide")
    for nom, image in images.items():
        verifier(image.shape == reference.shape and np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4), f"Grille differente : {nom}")
    donnees = {nom: image.get_fdata(dtype=np.float32) for nom, image in images.items()}
    for nom in ("Annotations", "Cerveau"):
        verifier(np.isfinite(donnees[nom]).all(), f"Masque non fini : {nom}")
    etiquettes = np.unique(donnees["Annotations"])
    verifier(np.isin(etiquettes, [0, 1, 2, 3]).all(), f"Etiquettes inattendues : {etiquettes}")
    verifier((donnees["Cerveau"] >= 0).all(), "Masque cerebral negatif")
    ann = donnees["Annotations"].astype(np.uint8)
    region = (donnees["Cerveau"] > 0) | (ann > 0)
    verifier(region.any(), "Region vide")
    plan = lire_plan(dossier)
    irm, dose, valide, minimum, maximum = normaliser(donnees["IRM"], donnees["Dose"], region, plan["prescription_totale_Gy"])
    bilan = {
        "patient": patient, "dimensions": list(reference.shape), **plan,
        "IRM_min_volume": minimum, "IRM_max_volume": maximum,
        "voxels_region": int(region.sum()), "voxels_dose_valides": int(valide.sum()),
        "dose_non_finie_dans_region": int(np.count_nonzero(region & ~np.isfinite(donnees["Dose"]))),
        "dose_negative_finie_dans_region": int(np.count_nonzero(region & np.isfinite(donnees["Dose"]) & (donnees["Dose"] < 0))),
        "annotations_sans_dose_valide": int(np.count_nonzero((ann > 0) & ~valide)),
        "dose_relative_max_valide": float(dose[valide].max()),
        "coupes_avec_dose_valide": int(np.any(valide, axis=(0, 1)).sum()),
        "sources": {n: str(p.relative_to(racine)) for n, p in chemins.items()},
        "convention_dose": "NaN hors masque de validite ; selectionner les pixels valides AVANT le calcul de la perte. Ne pas multiplier une erreur contenant NaN par zero.",
        "limite": "Preparation numerique ; ne certifie ni l'alignement anatomique ni la couverture clinique. P08 reste sous reserve de validation partielle.",
    }
    destination.mkdir()
    sauvegarder(region, reference, destination / "region_analyse.nii.gz", np.uint8)
    sauvegarder(valide, reference, destination / "masque_dose_valide.nii.gz", np.uint8)
    sauvegarder(ann, reference, destination / "annotations.nii.gz", np.uint8)
    sauvegarder(irm, reference, destination / "t1c_minmax.nii.gz", np.float32)
    sauvegarder(dose, reference, destination / "dose_relative.nii.gz", np.float32)
    (destination / "preparation.json").write_text(json.dumps(bilan, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return bilan


def main():
    racine = path("source_root")
    selection = racine / "docs" / "selection_patients.csv"
    with selection.open(encoding="utf-8-sig", newline="") as f:
        lecteur = csv.DictReader(f, delimiter=";")
        verifier({"patient", "statut", "motif"}.issubset(lecteur.fieldnames or []), "Colonnes de selection manquantes")
        lignes = list(lecteur)
    ids = [l["patient"].strip() for l in lignes]
    verifier(len(ids) == len(set(ids)), "Patients dupliques dans la selection")
    patients = [l["patient"].strip() for l in lignes if l["statut"].strip() == "candidat_a_controler"]
    verifier(bool(patients), "Aucun candidat")
    verifier(all(len(p) == 3 and p[0] == "P" and p[1:].isdigit() for p in patients), "Identifiant invalide")
    # Un dossier distinct a chaque execution : aucune preparation precedente ecrasee.
    sortie = ROOT / "data" / ("prepares_lot_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    sortie.mkdir(parents=True)
    shutil.copy2(selection, sortie / "selection_utilisee.csv")
    resultats = []
    for patient in patients:
        print(f"Preparation {patient}...", flush=True)
        temporaire = sortie / (patient + "_en_cours")
        try:
            bilan = preparer(racine, patient, temporaire)
            temporaire.rename(sortie / patient)
            resultats.append({"patient": patient, "resultat": "prepare", "bilan": bilan})
            print(f"  Prepare ; doses invalides dans R : {bilan['voxels_region'] - bilan['voxels_dose_valides']}", flush=True)
        except Exception as erreur:
            if temporaire.exists():
                shutil.rmtree(temporaire)
            resultats.append({"patient": patient, "resultat": "erreur", "message": str(erreur)})
            print(f"  ERREUR : {erreur}", flush=True)
    (sortie / "bilan_preparation.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    compteur = Counter(r["resultat"] for r in resultats)
    texte = [f"Candidats : {len(patients)}", f"Prepares : {compteur['prepare']}", f"Erreurs : {compteur['erreur']}"]
    for r in resultats:
        if r["resultat"] == "erreur":
            texte.append(f"{r['patient']} : ERREUR : {r['message']}")
        else:
            b = r["bilan"]
            texte.append(f"{r['patient']} : {b['prescription_totale_Gy']} Gy / {b['nombre_fractions']} fraction(s) ; non finies R={b['dose_non_finie_dans_region']} ; negatives R={b['dose_negative_finie_dans_region']} ; annotations sans dose valide={b['annotations_sans_dose_valide']}")
    texte.append("Preparation numerique uniquement ; selection non modifiee.")
    (sortie / "bilan_preparation.txt").write_text("\n".join(texte), encoding="utf-8")
    print("\n" + "\n".join(texte))
    print(f"\nSortie : {sortie}")
    return 1 if compteur['erreur'] else 0


if __name__ == "__main__":
    sys.exit(main())
