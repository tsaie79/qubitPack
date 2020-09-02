from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.local_env import CrystalNN

import numpy as np

from pycdt.core.defectsmaker import ChargedDefectsStructures


def find_cation_anion(structure):
    cation, anion = None, None
    for idx, el in enumerate(list(dict.fromkeys(structure.species))):
        if el.is_metal:
            cation = list(dict.fromkeys(structure.species))[idx].name
        else:
            anion = list(dict.fromkeys(structure.species))[idx].name
    return cation, anion


def modify_vacuum(orig_st, vacuum):
    if vacuum < orig_st.lattice.c:
        print("Please Set Vacuum > original lattice!!")
    ase_offset = AseAtomsAdaptor.get_atoms(orig_st)
    ase_offset.center(vacuum=0.0, axis=2)
    try:
        offset = AseAtomsAdaptor.get_structure(ase_offset).lattice.c
    except Exception as err:
        print(err)
        offset = 0
    ase_atom_obj = AseAtomsAdaptor.get_atoms(orig_st)
    ase_atom_obj.center(vacuum=(vacuum-offset)/2, axis=2)
    return AseAtomsAdaptor.get_structure(ase_atom_obj)


def get_rand_vec(distance): #was 0.001
    # deals with zero vectors.
    vector = np.random.randn(3)
    vnorm = np.linalg.norm(vector)
    return vector / vnorm * distance if vnorm != 0 else get_rand_vec(distance)


def move_site(structure, tgt_sites_idx, displacement_vector, frac_coords=False):
    modified_structure = structure.translate_sites(tgt_sites_idx, displacement_vector, frac_coords=frac_coords)
    return modified_structure


def selective_dyn_site(structure, tgt_sites_idx):
    where = []
    for site_idx in range(len(structure.sites)):
        if site_idx in tgt_sites_idx:
            where.append([True, True, True])
        else:
            where.append([False, False, False])
    poscar = Poscar(structure)
    poscar.selective_dynamics = where
    modified_structure = poscar.structure
    return modified_structure


def modify_lattice(src_structure, tgt_structure):
    tgt_lattice = tgt_structure.lattice
    src_structure.modify_lattice(tgt_lattice)
    return src_structure


def defect_from_primitive_cell(orig_st, defect_type, natom, substitution=None, distort=0.002, vacuum_thickness=None):
    if vacuum_thickness:
        orig_st = modify_vacuum(orig_st, vacuum_thickness)
    else:
        orig_st = orig_st

    defects = ChargedDefectsStructures(orig_st, cellmax=natom, antisites_flag=True).defects

    # find NN in defect structure
    def find_nn(defect=defects, defect_type=defect_type):
        defect_st = defect[defect_type[0]][defect_type[1]]["supercell"]["structure"].get_sorted_structure()
        defect_site_in_bulk = defect[defect_type[0]][defect_type[1]]["bulk_supercell_site"]
        defect_entry = defect[defect_type[0]][defect_type[1]].pop("supercell")

        if defect_type[0] == "bulk":
            bulk_structure = defects[defect_type[0]]["supercell"]["structure"]
            return bulk_structure, None, defect_entry, None

        elif defect_type[0] == "substitutions":
            defect_site_in_bulk_index = defect_st.index(defect_site_in_bulk)
            print(defect_site_in_bulk, defect_site_in_bulk.to_unit_cell())
            NN = [defect_st.index(defect_st[nn['site_index']])
                  for nn in CrystalNN().get_nn_info(defect_st, defect_site_in_bulk_index)]
            bond_length = [defect_st.get_distance(defect_site_in_bulk_index, NN_index) for NN_index in NN]
            NN = dict(zip(NN, bond_length))
            print("=="*50, "\nBefore distortion: {}".format(NN))
            return defect_st, NN, defect_entry, defect_site_in_bulk_index

        elif defect_type[0] == "vacancies":
            bulk_st = defect["bulk"]["supercell"]["structure"]
            print(defect_site_in_bulk, defect_site_in_bulk.to_unit_cell())
            try:
                defect_site_in_bulk_index = bulk_st.index(defect_site_in_bulk)
            except ValueError:
                defect_site_in_bulk_index = bulk_st.index(defect_site_in_bulk.to_unit_cell())
            NN = [defect_st.index(bulk_st[nn['site_index']])
                  for nn in CrystalNN().get_nn_info(bulk_st, defect_site_in_bulk_index)]
            bond_length = [bulk_st.get_distance(defect_site_in_bulk_index, NN_index['site_index'])
                           for NN_index in CrystalNN().get_nn_info(bulk_st, defect_site_in_bulk_index)]
            NN = dict(zip(NN, bond_length))
            print("=="*50, "\nBefore distortion: {}".format(NN))
            return defect_st, NN, defect_entry, defect_site_in_bulk_index

    defect_st, NN, defect_entry, defect_site_in_bulk_index = find_nn()

    # Move ions around the vacancy randomly
    # add defect into NN for DOS
    bulk_st = defects["bulk"]["supercell"]["structure"]
    for site in NN.keys():
        perturb = get_rand_vec(distort)
        defect_st.translate_sites(site, perturb)
        bulk_st.translate_sites(site, perturb)
    if defect_type[0] == "substitutions":
        NN[defect_site_in_bulk_index] = 0
        bond_length = [defect_st.get_distance(defect_site_in_bulk_index, NN_index) for NN_index in NN]
        NN = dict(zip(NN.keys(), bond_length))
    elif defect_type[0] == "vacancies":
        bond_length = [bulk_st.get_distance(defect_site_in_bulk_index, NN_index['site_index'])
                       for NN_index in CrystalNN().get_nn_info(bulk_st, defect_site_in_bulk_index)]
        NN = dict(zip(NN.keys(), bond_length))

    print("After distortion: {}\n{}".format(NN, "=="*50))

    # To make a substitution to a NN among given element (default None)
    if substitution:
        defect_st.replace(list(NN.keys())[0], substitution)

    NN = list(NN.keys())
    print("Nearest neighbors = %s" % NN)
    # static_set.write_input(".")
    return defect_st, NN, defect_entry, defect_site_in_bulk_index

