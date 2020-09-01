from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np


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