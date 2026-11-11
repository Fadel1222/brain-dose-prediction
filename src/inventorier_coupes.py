"""Inventorier les coupes sans les dupliquer et separer les patients.
Usage : python src/inventorier_coupes.py [--lot data/prepares_lot_...]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from configuration import CONFIG, ROOT, path, inventory, patients as split_patients, prescription as read_prescription
import random
from datetime import datetime

import nibabel as nib
import numpy as np

GRAINE = 20261103


def verifier(condition, message):
    if not condition:
        raise ValueError(message)


def repartir(bilans):
    groupes = {1: [], 7: []}
    for b in bilans:
        verifier(b['nombre_fractions'] in groupes, 'Fractionnement inattendu')
        groupes[b['nombre_fractions']].append(b['patient'])
    verifier(len(groupes[1]) == 17 and len(groupes[7]) == 7,
             'Ce partage attend 17 patients en une fraction et 7 en sept fractions.')
    verifier(len({b['patient'] for b in bilans}) == 24, 'Identifiants dupliques ou nombre incorrect')
    rng = random.Random(GRAINE)
    affectations = {}
    for fractions, nombres in ((1, (11, 3, 3)), (7, (5, 1, 1))):
        patients = sorted(groupes[fractions])
        rng.shuffle(patients)
        debut = 0
        for ensemble, nombre in zip(('entrainement', 'validation', 'test'), nombres):
            for patient in patients[debut:debut + nombre]:
                affectations[patient] = ensemble
            debut += nombre
    return affectations


def compter_coupes(region, valide, annotations):
    verifier(region.shape == valide.shape == annotations.shape and region.ndim == 3, 'Dimensions incorrectes')
    verifier(not np.any(valide & ~region), 'Dose valide hors region')
    lignes = []
    for z in range(region.shape[2]):
        nr = int(region[:, :, z].sum())
        nv = int(valide[:, :, z].sum())
        na = int(np.count_nonzero(annotations[:, :, z]))
        lignes.append({
            'indice_axe2': z, 'pixels_region': nr, 'pixels_dose_valides': nv,
            'pixels_annotation_1': int(np.count_nonzero(annotations[:, :, z] == 1)),
            'pixels_annotation_2': int(np.count_nonzero(annotations[:, :, z] == 2)),
            'pixels_annotation_3': int(np.count_nonzero(annotations[:, :, z] == 3)),
            'pixels_annotations': na,
            'utilisable': int(nv > 0),
        })
    return lignes


def ecrire_csv(chemin, lignes):
    with chemin.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]), delimiter=';')
        w.writeheader()
        w.writerows(lignes)


def main():
    racine = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lot', type=Path)
    args = parser.parse_args()
    if args.lot:
        lot = args.lot if args.lot.is_absolute() else racine / args.lot
    else:
        lot = path("prepared_dir")
    lot = lot.resolve()
    resultats = json.loads((lot / 'bilan_preparation.json').read_text(encoding='utf-8'))
    verifier(len(resultats) == 24 and all(r['resultat'] == 'prepare' for r in resultats), 'Lot incomplet : 24 preparations reussies attendues')
    bilans = [r['bilan'] for r in resultats]
    with (lot / 'selection_utilisee.csv').open(encoding='utf-8-sig', newline='') as f:
        selection = list(csv.DictReader(f, delimiter=';'))
    attendus = {r['patient'].strip() for r in selection if r['statut'].strip() == 'candidat_a_controler'}
    verifier(attendus == {b['patient'] for b in bilans}, 'Selection et lot differents')
    affectations = repartir(bilans)
    coupes, patients = [], []
    empreintes_irm = {}
    for b in sorted(bilans, key=lambda x: x['patient']):
        patient = b['patient']
        verifier(len(patient) == 3 and patient[0] == 'P' and patient[1:].isdigit(), 'Identifiant invalide')
        dossier = lot / patient
        meta = json.loads((dossier / 'preparation.json').read_text(encoding='utf-8'))
        verifier(meta == b, f'{patient} : bilan et parametres differents')
        images = {n: nib.load(dossier / f) for n, f in {
            'region': 'region_analyse.nii.gz', 'valide': 'masque_dose_valide.nii.gz',
            'annotations': 'annotations.nii.gz', 'IRM': 't1c_minmax.nii.gz',
            'dose': 'dose_relative.nii.gz',
        }.items()}
        ref = images['IRM']
        for nom, img in images.items():
            verifier(img.shape == ref.shape and np.allclose(img.affine, ref.affine, rtol=0, atol=1e-4), f'{patient} : grille differente ({nom})')
        region_brute = np.asarray(images['region'].dataobj)
        valide_brut = np.asarray(images['valide'].dataobj)
        ann = np.asarray(images['annotations'].dataobj)
        verifier(np.isin(region_brute, [0, 1]).all() and np.isin(valide_brut, [0, 1]).all(), f'{patient} : masque non binaire')
        verifier(np.isin(ann, [0, 1, 2, 3]).all(), f'{patient} : annotations invalides')
        region, valide = region_brute.astype(bool), valide_brut.astype(bool)
        irm = images['IRM'].get_fdata(dtype=np.float32)
        dose = images['dose'].get_fdata(dtype=np.float32)
        verifier(np.isfinite(irm).all() and irm.min() >= 0 and irm.max() <= 1, f'{patient} : IRM normalisee invalide')
        verifier(np.isfinite(dose[valide]).all() and (dose[valide] >= 0).all(), f'{patient} : dose valide incorrecte')
        verifier(np.isnan(dose[~valide]).all(), f'{patient} : convention des doses invalides differente')
        verifier(int(region.sum()) == b['voxels_region'] and int(valide.sum()) == b['voxels_dose_valides'], f'{patient} : nombres de voxels differents')
        empreinte = hashlib.sha256(np.ascontiguousarray(irm, dtype='<f4').tobytes()).hexdigest()
        verifier(empreinte not in empreintes_irm, f'IRM exactement dupliquee : {patient} et {empreintes_irm.get(empreinte)}. Partage interrompu.')
        empreintes_irm[empreinte] = patient
        lignes = compter_coupes(region, valide, ann)
        for ligne in lignes:
            coupes.append({'patient': patient, 'ensemble': affectations[patient],
                           'nombre_fractions': b['nombre_fractions'],
                           'prescription_totale_Gy': b['prescription_totale_Gy'], **ligne})
        patients.append({'patient': patient, 'ensemble': affectations[patient],
                         'nombre_fractions': b['nombre_fractions'],
                         'prescription_totale_Gy': b['prescription_totale_Gy'],
                         'coupes_totales': len(lignes),
                         'coupes_utilisables': sum(l['utilisable'] for l in lignes),
                         'coupes_utilisables_annotees': sum(l['utilisable'] and l['pixels_annotations'] > 0 for l in lignes),
                         'IRM_SHA256': empreinte})
        print(f'{patient} : inventorie', flush=True)
    # Ecrire seulement apres la reussite de tous les controles.
    sortie = lot / ('inventaire_2d_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    sortie.mkdir(parents=True)
    ecrire_csv(sortie / 'patients_repartition.csv', patients)
    ecrire_csv(sortie / 'inventaire_coupes.csv', coupes)
    protocole = {'lot_source': str(lot), 'graine': GRAINE, 'axe': 2,
                 'indices': 'Indices de grille a partir de zero ; aucune lateralite deduite',
                 'critere_coupe_utilisable': 'Au moins un pixel de dose valide',
                 'partage': affectations, 'remarque': 'Toutes les coupes utilisables, avec ou sans annotation. Aucun echantillonnage ou entrainement effectue. Test a garder ferme pendant le developpement. P08 conserve sa reserve documentee.'}
    (sortie / 'protocole.json').write_text(json.dumps(protocole, ensure_ascii=False, indent=2), encoding='utf-8')
    texte = ['Inventaire des coupes et partage par patient', f'Lot : {lot}', f'Graine : {GRAINE}', 'Axe de coupe : 2 ; indices a partir de zero', '']
    for ensemble in ('entrainement', 'validation', 'test'):
        ps = [p for p in patients if p['ensemble'] == ensemble]
        texte.append(f"{ensemble} : {len(ps)} patients ; {sum(p['nombre_fractions']==1 for p in ps)} en 1 fraction ; {sum(p['nombre_fractions']==7 for p in ps)} en 7 fractions ; {sum(p['coupes_utilisables'] for p in ps)} coupes utilisables ; {sum(p['coupes_utilisables_annotees'] for p in ps)} annotees")
        texte.append('Patients : ' + ', '.join(p['patient'] for p in ps))
    texte += ['', f'Coupes inventoriees : {len(coupes)}', f"Coupes utilisables : {sum(c['utilisable'] for c in coupes)}", 'Aucun patient partage entre ensembles. Aucun doublon exact d\'IRM detecte.', 'Ce controle ne detecte pas toutes les formes de duplication ou les identites patient anonymisees.', 'Les donnees sources et la selection sont conservees. Aucun modele entraine.']
    (sortie / 'bilan_inventaire.txt').write_text('\n'.join(texte), encoding='utf-8')
    print('\n' + '\n'.join(texte))
    print(f'\nSortie : {sortie}')


if __name__ == '__main__':
    main()
