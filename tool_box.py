from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar, Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.defects.core import Vacancy, Substitution
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pycdt.core.defectsmaker import ChargedDefectsStructures

from phonopy.phonon.irreps import character_table

from atomate.vasp.database import VaspCalcDb

import numpy as np
import pandas as pd
import os, shutil, datetime
from collections import OrderedDict, defaultdict

from monty.serialization import loadfn, dumpfn

def get_db(db_name, collection_name, user="Jeng_ro", password="qimin", port=12345):
    return VaspCalcDb(host="localhost", port=port, database=db_name,
                      collection=collection_name, user=user, password=password, authsource=db_name)

def find_cation_anion(structure):
    cation, anion = None, None
    for idx, el in enumerate(list(dict.fromkeys(structure.species))):
        if el.is_metal:
            cation = list(dict.fromkeys(structure.species))[idx].name
        else:
            anion = list(dict.fromkeys(structure.species))[idx].name
    return cation, anion


def set_vacuum(orig_st, vacuum):
    from mpinterfaces.utils import ensure_vacuum
    st = ensure_vacuum(orig_st, vacuum)
    return st


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

def get_site_idx_along_z(structure, z_min, z_max):
    tgt_sites = {}
    for idx, site in enumerate(structure.sites):
        c = site.c
        if z_min <= c <= z_max:
            tgt_sites[idx] = c
    return tgt_sites

def defect_from_primitive_cell(orig_st, defect_type, natom, substitution=None, distort=0.002, vacuum_thickness=None):
    """
    defect_type: ["vacancies", 0] / ["substitutions", 0]
    """
    if vacuum_thickness:
        orig_st = modify_vacuum(orig_st, vacuum_thickness)
    else:
        orig_st = orig_st

    defects = ChargedDefectsStructures(orig_st, cellmax=natom, antisites_flag=True).defects
    bulk_st = defects["bulk"]["supercell"]["structure"]

    if defect_type[0] == "bulk":
        bulk_structure = defects[defect_type[0]]["supercell"]["structure"]
        defect_entry = defects[defect_type[0]]
        defect_entry["supercell"].pop("structure")
        return bulk_structure, None, defect_entry, None

    # find NN in defect structure
    def find_nn(defect=defects, defect_type=defect_type):
        defect_st = defect[defect_type[0]][defect_type[1]]["supercell"]["structure"].get_sorted_structure()
        defect_site_in_bulk = defect[defect_type[0]][defect_type[1]]["bulk_supercell_site"]
        defect_entry = defect[defect_type[0]][defect_type[1]]
        defect_entry["supercell"].pop("structure")
        defect_entry["supercell"]["bulk"] = defects["bulk"]["supercell"]["structure"]

        if defect_type[0] == "substitutions":
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
            print("vac coord. = {}-{}".format(defect_site_in_bulk, defect_site_in_bulk.to_unit_cell()))
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
    # add defect into nn_dist for DOS
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

    # To make a substitution to a nn_dist among given element (default None)
    if substitution:
        defect_st.replace(list(NN.keys())[0], substitution)
        print("substitution coord = {}".format(defect_st[list(NN.keys())[0]]))
        defect_entry["complex"] = {"site": defect_st[list(NN.keys())[0]], "site_specie":  substitution}
        defect_entry["defect_type"] = "complex"
        defect_sites_in_bulk = [defect_st.sites[nn] for nn in NN.keys()]
        defect_st.sort()
        NN = [defect_st.index(nn) for nn in defect_sites_in_bulk]

    if not substitution:
        NN = list(NN.keys())

    print("Nearest neighbors = %s" % NN)
    # static_set.write_input(".")
    return defect_st, NN, defect_entry, defect_site_in_bulk_index


class GenDefect:
    def __init__(self, orig_st, defect_type, natom, vacuum_thickness=None, distort=None, sub_on_side=None, standardize_st=True):
        for site_property in orig_st.site_properties:
            orig_st.remove_site_property(site_property)
        if vacuum_thickness:
            self.orig_st = set_vacuum(orig_st, vacuum_thickness)
        else:
            self.orig_st = orig_st
        self.defect_type = defect_type
        self.natom = natom
        self.vacuum_thickness = vacuum_thickness
        self.distort = None
        self.site_info = None
        self.defects = ChargedDefectsStructures(self.orig_st, cellmax=natom, antisites_flag=True).defects
        self.bulk_st = self.defects["bulk"]["supercell"]["structure"]
        self.NN_for_sudo_bulk = []

        if (defect_type[0] == "substitutions") or (defect_type[0] == "vacancies"):
            self.defect_entry = self.defects[self.defect_type[0]][self.defect_type[1]]
            self.defect_st = self.defect_entry["supercell"]["structure"].get_sorted_structure()
            self.defect_site_in_bulk = self.defect_entry["bulk_supercell_site"]
            self.defect_site_in_bulk_index = None
            self.NN = None

            if defect_type[0] == "substitutions":
                try:
                    self.defect_site_in_bulk_index = self.defect_st.index(self.defect_site_in_bulk)
                except ValueError:
                    self.defect_site_in_bulk_index = self.defect_st.index(self.defect_site_in_bulk.to_unit_cell())
                self.NN = [self.defect_st.index(self.defect_st[nn['site_index']])
                           for nn in CrystalNN().get_nn_info(self.defect_st, self.defect_site_in_bulk_index)]
                self.pmg_obj = Substitution(self.orig_st, self.defect_entry["unique_site"])

            elif defect_type[0] == "vacancies":
                try:
                    self.defect_site_in_bulk_index = self.bulk_st.index(self.defect_site_in_bulk)
                except ValueError:
                    self.defect_site_in_bulk_index = self.bulk_st.index(self.defect_site_in_bulk.to_unit_cell())
                self.NN = [self.defect_st.index(self.bulk_st[nn['site_index']])
                           for nn in CrystalNN().get_nn_info(self.bulk_st, self.defect_site_in_bulk_index)]
                self.pmg_obj = Vacancy(self.orig_st, self.defect_entry["unique_site"])

            self.nn_dist = dict(before=None, after=None)
            self.nn_dist["before"] = dict(zip([str(idx) for idx in self.NN], range(len(self.NN))))
            print("defect site coord. = orig: {} unit: {}".format(
                self.defect_site_in_bulk, self.defect_site_in_bulk.to_unit_cell()))

            self.defect_entry["supercell"].pop("structure")
            self.defect_entry["supercell"]["bulk"] = self.bulk_st


        elif defect_type[0] == "bulk":
            self.defect_entry = self.defects[self.defect_type[0]]
            self.defect_entry["supercell"].pop("structure")
            self.defect_st = None

        else:
            print("!!!Please insert substitutions, vacancies, or bulk!!!")

        if standardize_st and self.defect_st:
            self.defect_st, self.site_info = phonopy_structure(self.defect_st)

        if defect_type[0] == "substitutions":
            self.substitutions(distort, sub_on_side)

        elif defect_type[0] == "vacancies":
            self.vacancies(distort, sub_on_side)

    def substitutions(self, distort, substitution):
        bond_length = [self.defect_st.get_distance(self.defect_site_in_bulk_index, NN_index)
                       for NN_index in self.NN]
        bond_length = np.array(bond_length).round(3)

        self.nn_dist["before"] = dict(zip([str(idx) for idx in self.NN], bond_length))

        if substitution:
            self.make_complex(substitution)

        self.NN.append(self.defect_site_in_bulk_index)
        self.nn_dist["before"][str(self.defect_site_in_bulk_index)] = 0
        print("==" * 50, "\nBefore distortion: {}".format(self.nn_dist["before"]))

        if distort:
            self.move_sites(distort)
            bond_length = [self.defect_st.get_distance(self.defect_site_in_bulk_index, NN_index)
                           for NN_index in self.NN]
            bond_length = np.array(bond_length).round(3)
            self.nn_dist["after"] = dict(zip([str(idx) for idx in self.NN], bond_length))
            print("After distortion: {}\n{}".format(self.nn_dist["after"], "==" * 50))

    def vacancies(self, distort, substitution):
        bond_length = [self.bulk_st.get_distance(self.defect_site_in_bulk_index, NN_index['site_index'])
                       for NN_index in CrystalNN().get_nn_info(self.bulk_st, self.defect_site_in_bulk_index)]
        bond_length = np.array(bond_length).round(3)

        self.nn_dist["before"] = dict(zip([str(idx) for idx in self.NN], bond_length))

        if substitution:
            self.make_complex(substitution)

        print("==" * 50, "\nBefore distortion: {}".format(self.nn_dist["before"]))

        if distort:
            sudo_bulk = self.move_sites(distort)
            bond_length = [sudo_bulk.get_distance(self.defect_site_in_bulk_index, NN_index['site_index'])
                           for NN_index in CrystalNN().get_nn_info(sudo_bulk, self.defect_site_in_bulk_index)]
            bond_length = np.array(bond_length).round(3)
            self.nn_dist["after"] = dict(zip([str(idx) for idx in self.NN], bond_length))
            print("After distortion: {}\n{}".format(self.nn_dist["after"], "==" * 50))

    def move_sites(self, distort):
        self.distort = distort
        sudo_bulk = self.bulk_st.copy()
        if self.NN_for_sudo_bulk:
            NN_tot = zip(self.NN, self.NN_for_sudo_bulk)
        else:
            NN_tot = zip(self.NN, self.NN)
        for site, sudo_bulk_site in NN_tot:
            perturb = get_rand_vec(distort)
            self.defect_st.translate_sites([site], perturb, frac_coords=False)
            sudo_bulk.translate_sites([sudo_bulk_site], perturb, frac_coords=False)
        return sudo_bulk

    def move_origin_to_defect(self):
        # center_site_idx = self.NN[nn_idx]
        # center_site_coords = self.defect_st[center_site_idx].coords
        center_site_coords = self.defect_site_in_bulk.coords
        self.defect_st.translate_sites(range(self.defect_st.num_sites), -1*center_site_coords, frac_coords=False)

    def make_complex(self, substitution):
        self.defect_entry["complex"] = {"site": [], "site_specie": []}
        for sub, idx in zip(substitution, range(len(substitution))):
            self.defect_st.replace(self.NN[idx], sub)
            print("substitution coord = {}".format(self.NN[idx]))

            self.NN_for_sudo_bulk.append(self.NN[idx])

            self.defect_entry["complex"]["site"].append(self.defect_st[self.NN[idx]])
            self.defect_entry["complex"]["site_specie"].append(sub)

        self.defect_entry["defect_type"] = "complex"

        defect_sites_in_bulk = [self.defect_st[nn] for nn in self.NN]

        self.defect_st.sort()
        if self.defect_type[0] == "substitutions":
            self.defect_site_in_bulk_index = self.defect_st.index(self.defect_site_in_bulk)

        self.NN = [self.defect_st.index(nn) for nn in defect_sites_in_bulk]
        self.nn_dist["before"] = dict(zip([str(idx) for idx in self.NN], self.nn_dist["before"].values()))

def get_unique_sites_from_wy(structure, symprec=1e-4):
    space_sym_analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    space_group = space_sym_analyzer.get_symmetry_dataset()["international"]
    space_group_number = space_sym_analyzer.get_symmetry_dataset()["number"]
    point_gp = space_sym_analyzer.get_symmetry_dataset()["pointgroup"]

    equivalent_atoms = list(space_sym_analyzer.get_symmetry_dataset()["equivalent_atoms"])
    equivalent_atoms_index = list(OrderedDict((x, equivalent_atoms.index(x)) for x in equivalent_atoms).values())

    site_syms = space_sym_analyzer.get_symmetry_dataset()["site_symmetry_symbols"]
    site_syms = [site_syms[i] for i in equivalent_atoms_index]
    wyckoffs = space_sym_analyzer.get_symmetry_dataset()["wyckoffs"]
    wyckoffs = [wyckoffs[i] for i in equivalent_atoms_index]
    species = structure.species
    species = [species[i].name for i in equivalent_atoms_index]
    print(species, site_syms, wyckoffs, point_gp)
    return dict(zip(["species", "site_syms", "wys", "pg", "unique_site_idx", "spg", "spg_number"],
                    [species, site_syms, wyckoffs, point_gp, equivalent_atoms_index, space_group, space_group_number]))

def get_good_ir_sites(structure, symprec=1e-4):
    species, site_syms, wyckoffs, pg, site_idxs, spg, spg_number = get_unique_sites_from_wy(structure, symprec).values()

    good_ir_species, good_ir_syms, good_ir_wy, good_ir_site_idx = [], [], [], []
    for specie, site_sym, wyckoffs, site_idx in zip(species, site_syms, wyckoffs, site_idxs):
        site_sym = [x for x in site_sym.split(".") if x][0]
        if site_sym == "-4m2":
            site_sym = "-42m"
        if site_sym == "2mm" or site_sym == "m2m":
            site_sym = "mm2"
        if site_sym == "1":
            continue

        irreps = character_table[site_sym][0]["character_table"]
        for irrep, char_vec in irreps.items():
            if char_vec[0] >= 2:
                good_ir_species.append(specie)
                good_ir_syms.append(site_sym)
                good_ir_wy.append(wyckoffs)
                good_ir_site_idx.append(site_idx)
                break
    print(good_ir_species, good_ir_syms, good_ir_wy, pg, good_ir_site_idx)
    return dict(zip(["species", "site_syms", "wys", "pg", "site_idx", "spg", "spg_number"],
                    [good_ir_species, good_ir_syms, good_ir_wy, pg, good_ir_site_idx, spg, spg_number]))

def get_interpolate_sts(i_st, f_st, disp_range=np.linspace(0, 2, 11), output_dir=None):
    '''
    atomic unit is adopted
    '''
    from pymatgen.core.units import ang_to_bohr
    # A. Alkauskas, Q. Yan, and C. G. Van de Walle, Physical Review B 90, 27 (2014)
    struct_i, sorted_symbols = i_st, i_st.symbol_set
    struct_f, sorted_symbols = f_st, f_st.symbol_set
    delta_R = struct_f.frac_coords - struct_i.frac_coords
    delta_R = (delta_R + 0.5) % 1 - 0.5

    lattice = struct_i.lattice.matrix #[None,:,:]
    delta_R = np.dot(delta_R, lattice)


    # Poscar(struct_i).write_file('disp_dir/POSCAR_i'.format(output_dir))
    # Poscar(struct_f).write_file('disp_dir/POSCAR_f'.format(output_dir))


    masses = np.array([spc.atomic_mass for spc in struct_i.species])
    delta_Q2 = masses[:,None] * delta_R ** 2
    delta_R2 = delta_R**2

    print('Delta_Q: {:3}'.format(np.sqrt(delta_Q2.sum())*ang_to_bohr))
    print('Delta_R: {:3}'.format(np.sqrt(delta_R2.sum())*ang_to_bohr))
    print('M: {:3}'.format(delta_Q2.sum()/delta_R2.sum()))
    info = {"Delta_Q":np.sqrt(delta_Q2.sum())*ang_to_bohr, "Delta_R": np.sqrt(delta_R2.sum())*ang_to_bohr,
            "M":delta_Q2.sum()/delta_R2.sum(), "unit":"atomic unit"}

    resulting_sts = []
    for frac in disp_range:
        disp = frac * delta_R
        struct = Structure(struct_i.lattice, struct_i.species,
                           struct_i.cart_coords + disp,
                           coords_are_cartesian=True)
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dumpfn(info, os.path.join(output_dir, "info.json"))
            struct.to("vasp", '{0}/POSCAR_{1:03d}.vasp'.format(output_dir, int(np.rint(frac*10))))
        resulting_sts.append((frac, struct))
    resulting_sts = dict(resulting_sts)
    return resulting_sts, info


def remove_entry_in_db(task_id, db_object, delete_fs_only=False, pmg_file=True, remove_dir=None):
    """
    remove entry and all Gridfs files in db

    """
    db = db_object
    entry = db.collection.find_one({"task_id":task_id})

    if pmg_file:
        remove_dict = {}
        for i in list(entry["calcs_reversed"][0].keys()):
            if "fs" in i:
                chunks = i.rsplit("_", 1)[0] + ".chunks"
                remove_dict[chunks] = entry["calcs_reversed"][0][i]
                files = i.rsplit("_", 1)[0] + ".files"
                remove_dict[files] = entry["calcs_reversed"][0][i]

        if delete_fs_only:
            for k, v in remove_dict.items():
                d = db.db[k]
                try:
                    d.collection.delete_one({"_id": v})
                except Exception as err:
                    print(err)
                    continue

        else:
            for k, v in remove_dict.items():
                d = db.db[k]
                try:
                    d.collection.delete_one({"_id": v})
                except Exception as err:
                    print(err)
                    continue

            if remove_dir:
                dir_path = os.path.join(remove_dir, db.db_name, db.collection.name, entry["dir_name"].split("/")[-1])
                shutil.rmtree(dir_path)
                print("removed {}".format(dir_path))

            db.collection.delete_one({"task_id": task_id})
            print("removed {}/{}/{}".format(db.db_name, db.collection.name, task_id))

    else:
        db.collection.delete_one({"task_id":task_id})

def get_lowest_unocc_band_idx(task_id, db_obj, nbands, prevent_JT=True, second_excite=False):

    eig = db_obj.get_eigenvals(task_id)
    spins = list(eig.keys())

    lowest_unocc_band_idx = []
    for spin in spins:
        band_idx = 0
        while eig[spin][0][band_idx][1] == 1:
            band_idx += 1
        lowest_unocc_band_idx.append(band_idx+1)
    lowest_unocc_band_idx = dict(zip(spins, lowest_unocc_band_idx))

    occu_configs = {}
    maj_spin = max(lowest_unocc_band_idx, key=lambda key: lowest_unocc_band_idx[key])
    low_band_idx = lowest_unocc_band_idx[maj_spin]
    if second_excite:
        occu_configs[maj_spin] = "{}*1 1*0 1*1 1*1 {}*0".format(low_band_idx-3, nbands-low_band_idx)
    elif prevent_JT:
        occu_configs[maj_spin] = "{}*1 1*0.5 1*0.5 1*1 {}*0".format(low_band_idx-3, nbands-low_band_idx)
    else:
        occu_configs[maj_spin] = "{}*1 1*0 1*1 {}*0".format(low_band_idx-2, nbands-low_band_idx)
    print("maj_spin: {}, occ:{}".format(maj_spin, occu_configs[maj_spin]))

    if len(spins) == 1:
        return maj_spin, occu_configs
    if len(spins) == 2:
        minor_spin = min(lowest_unocc_band_idx, key=lambda key: lowest_unocc_band_idx[key])
        if minor_spin == maj_spin:
            maj_spin, minor_spin = "1", "-1"
        low_band_idx = lowest_unocc_band_idx[minor_spin]
        occu_configs[minor_spin] = "{}*1 {}*0".format(low_band_idx-1, nbands-low_band_idx+1)
        print("minor_spin: {}, occ:{}".format(minor_spin, occu_configs[minor_spin]))
        return maj_spin, occu_configs

def phonopy_structure(orig_st):
    from subprocess import call, check_output, Popen
    import shutil

    path = os.path.expanduser(os.path.join("~", "standardize_st"))
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    orig_st.to("poscar", "POSCAR")
    call("phonopy --symmetry --tolerance 0.01 -c POSCAR".split(" "))
    std_st = Structure.from_file("PPOSCAR")
    std_st.to("poscar", "POSCAR")
    pos2aBR_out = check_output(["pos2aBR"], universal_newlines=True).split("\n")
    std_st = Structure.from_file("POSCAR_std")
    shutil.rmtree(path)
    return std_st, pos2aBR_out

def get_encut(st):
    from pymatgen.io.vasp.sets import MPRelaxSet
    encut = 1.3*max([potcar.enmax for potcar in MPRelaxSet(st).potcar])
    return encut

def cd_from_db(db, task_id):
    import os
    path = db.collection.find_one({"task_id": task_id})["calcs_reversed"][0]["dir_name"]
    os.chdir(path)

def find_scaling_for_2d_defect(pc, min_lc=15):
    x = 1
    while True:
        st = pc.copy()
        st.make_supercell([x,1,1])
        if st.lattice.a >= min_lc:
            break
        else:
            x += 1

    y = 1
    while True:
        st = pc.copy()
        st.make_supercell([1, y, 1])
        if st.lattice.b >= min_lc:
            break
        else:
            y += 1

    scaling = [x, y, 1]
    st = pc.copy()
    st.make_supercell(scaling)
    print("scaling matrix: {}".format(scaling))
    return scaling, st


def get_band_edges_characters(bs):
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()
        data = {}
        # data["cbm"] = {"up": {}, "down": {}}
        # data["vbm"] = {"up": {}, "down": {}}
        for name, band_edge in zip(["vbm", "cbm"], [vbm, cbm]):
            band_index = band_edge["band_index"]
            if band_index.get(Spin.up) != band_index.get(Spin.down):
                data.update({"{}_is_up_dn_band_idx_equal".format(name): False})
            else:
                data.update({"{}_is_up_dn_band_idx_equal".format(name): True})

            projections = band_edge["projections"]
            for spin, band_idx in band_index.items():
                if band_idx:
                    tot_proj_on_element = projections[spin].sum(axis=0)
                    max_tot_proj_element_idx = tot_proj_on_element.argmax()
                    max_tot_proj = tot_proj_on_element[max_tot_proj_element_idx]

                    max_proj_orbital_id = projections[spin][:, max_tot_proj_element_idx].argmax()
                    max_proj_orbital = Orbital(max_proj_orbital_id).name

                    data.update(
                        {
                            "{}_{}_max_el_idx".format(name, spin.name): max_tot_proj_element_idx,
                            "{}_{}_max_el".format(name, spin.name): bs.structure[
                                max_tot_proj_element_idx].species_string,
                            "{}_{}_max_proj".format(name, spin.name): max_tot_proj,
                            "{}_{}_proj_on_el".format(name, spin.name): tuple(tot_proj_on_element),
                            "{}_{}_max_proj_orbital".format(name, spin.name): max_proj_orbital,
                            "{}_{}_orbital_proj_on_el".format(name, spin.name): tuple(projections[spin][
                                                                                 max_proj_orbital_id, :])
                        })


                    # data[name][spin.name].update(
                    #     {
                    #         "max_element_idx": max_tot_proj_element_idx,
                    #         "max_element": bs.structure[max_tot_proj_element_idx].species_string,
                    #         "max_proj": max_tot_proj,
                    #         "proj_on_element": tot_proj_on_element,
                    #     }
                    # )
        print(data)
        spins = [(spin_vbm, spin_cbm) for spin_vbm in [Spin.up.name, Spin.down.name] for spin_cbm in [Spin.up.name, Spin.down.name]]
        for spin in spins:
            if data.get("vbm_{}_max_element_idx".format(spin[0]), "Yes") == \
                    data.get("cbm_{}max_element_idx".format(spin[1]), "No"):
                data["is_vbm_cbm_from_same_element"] = True
                break
            else:
                data["is_vbm_cbm_from_same_element"] = False

        return data

def plot_lopot(db, task_id):
    from matplotlib import pyplot as plt
    locpot = db.collection.find_one({"task_id":task_id})["calcs_reversed"][0]["output"]["locpot"]["2"]
    plt.plot(locpot)
    plt.show()


class IOTools:
    def __init__(self, cwd, pandas_df= None, excel_file=None, json_file=None):
        self.df = pandas_df
        self.excel_file = excel_file
        self.json_file = json_file
        self.cwd = cwd

    def read_excel(self, string_tuple_to_tuple=True):
        df = pd.read_excel(os.path.join(self.cwd, self.excel_file+".xlsx"))
        if string_tuple_to_tuple:
            import ast
            for k in df.keys():
                try:
                    df[k] = df[k].apply(ast.literal_eval)
                except Exception:
                    continue
        return df

    def read_json(self):
        return pd.DataFrame(oadfn(os.path.join(self.cwd, self.json_file+".json")))

    def to_excel(self, file_name, index=False):
        self.df.to_excel(
            os.path.join(self.cwd, "{}_{}.xlsx".format(file_name, str(datetime.datetime.now()))), index=index)

    def to_json(self, file_name, index=False):
        self.df.to_json(
            os.path.join(self.cwd, "{}_{}.json".format(file_name, str(datetime.datetime.now()))),
            orient="records", indent=4, index=index)

    def get_diff_btw_dfs(self, df1, df2):
        return  pd.concat([df1,df2]).drop_duplicates(keep=False)