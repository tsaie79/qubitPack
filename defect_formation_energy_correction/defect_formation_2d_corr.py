#%%
from pymatgen.analysis.defects.core import DefectEntry
from pymatgen.analysis.defects.core import Vacancy, Substitution
from pymatgen.analysis.defects.corrections import KumagaiCorrection
from pymatgen.io.vasp.inputs import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen import MPRester

from pycdt.core.chemical_potentials import get_mp_chempots_from_dpd

from atomate.vasp.database import VaspCalcDb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from collections import defaultdict
from glob import glob
import os, math
import pandas as pd

from sklearn.linear_model import LinearRegression


# DB_CONFIG_PATH = "/Users/jeng-yuantsai/PycharmProjects/git/my_pycharm_projects/database/db_config/" \
#                  "formation_energy_related"
# DB_CONFIG_LOCAL = defaultdict(str)
# DB_CONFIG_OWLS = defaultdict(str)

# for db_config in glob(DB_CONFIG_PATH+"/*"):
#     if "local" in db_config:
#         DB_CONFIG_LOCAL[os.path.basename(db_config)] = db_config
#     else:
#         DB_CONFIG_OWLS[os.path.basename(db_config)] = db_config
# db_config_index = input(
#     "List of db_config files as following:\n{}"
#     "\nPlease select db_config file:".format(
#         "\n".join(["[" + str(index) + "] " + os.path.basename(i) for index, i in enumerate(
#             glob(DB_CONFIG_PATH + "/*"))]
#                   )
#     )
# )
# db_config_index = 4
# db = dict(enumerate(glob(DB_CONFIG_PATH + "/*")))[int(db_config_index)]


class FormationEnergy2DCorr:

    def __init__(self, bulk_collection, bulk_task_id, vac_collection, vac_task_id, vbm_bg_collection, vbm_bg_task_id):
        # defect_type:bulk
        self.bulk_collection = bulk_collection
        self.vac_collection = vac_collection
        self.vbm_bg_collection = vbm_bg_collection
        self.bulk = self.bulk_collection.find_one({"task_id": bulk_task_id})
        self.bk_vbm_bg = self.vbm_bg_collection.find_one({"task_id": vbm_bg_task_id})
        self.bulk_st = Structure.from_dict(self.bulk["input"]["structure"])
        self.bulk_energy = self.bulk["calcs_reversed"][0]["output"]["energy"]
        self.bulk_vbm = self.bk_vbm_bg["calcs_reversed"][0]["output"]["vbm"]
        self.bulk_bandgap = self.bk_vbm_bg["calcs_reversed"][0]["output"]["bandgap"]
        # self.bulk_bandgap = 2.2713

        self.vac = self.vac_collection.find_one({"task_id": vac_task_id})
        self.vac_energy = self.vac["calcs_reversed"][0]["output"]["energy"]

        self.vac_nelect = MPRelaxSet(Structure.from_dict(self.vac["input"]["structure"])).nelect
        self.vac_charge = self.vac["charge_state"] or self.vac_nelect - self.vac["input"]["parameters"]["NELECT"]
        # self.vac_site = self.bulk_st[vac_site]

    @property
    def vacancy(self):
        return Vacancy(structure=self.bulk_st, defect_site=self.bulk_st[0], charge=self.vac_charge)

    def defect_entry(self, entry_id):
        parameters = {"vbm": self.bulk_vbm, "band_gap": self.bulk_bandgap}
        defect_entry = DefectEntry(defect=self.vacancy, uncorrected_energy=self.vac_energy-self.bulk_energy,
                                   parameters=parameters, entry_id=entry_id)
        return defect_entry

    # def formation_energy(self, fermi_level, chemical_potentials=None):
    #     return self.defect_entry.formation_energy(fermi_level=fermi_level, chemical_potentials=chemical_potentials)

    @classmethod
    def transition_levels(cls, chg_min, chg_max, bulk_nsites, defect_type, band_gap, transition_level):
        """
        In order to find correct entry for defect and bulk, they must have the following key-value:
        Defect: {"defect_type": defect_type} and {"task_type":"scf"}
        Bulk: {"defect_type": "bulk"} and {"task_type":"scf"}
        :param chg_min: min charge state
        :param chg_max: max charge state (min+1...etc)
        :param defect_nsites (list): geo spec
        :param bulk_nsites (list): geo spec
        :param defect_type: {vacancy: "W", substitution: "Se"} or {antisite:"Se"}
        :return: defect_tls (dict): Transition levels for each geo spec
        """
        lz = None
        area = None
        bulk_task_id = None
        defect_tls = []
        # Find right collection in DB to use
        owls_formation_energy = VaspCalcDb.from_db_file(
            db_file='/Users/jeng-yuantsai/PycharmProjects/git/my_pycharm_projects/database/'
                    'db_config/formation_energy_related/db_owls_formation_energy.json').collection
        db_w1se2_hse_local = VaspCalcDb.from_db_file(db_file=db).collection

        # 1. Find task_id in DB and create DefectEntry
        defect_entries = []
        data1 = []
        data2 = []
        for bulk_nsite in bulk_nsites:
            formation_energy_with_diff_charg = {}
            charges = list(range(chg_min, chg_max+1, 1))
            charges.remove(0)
            for charge in [0] + charges:
                print("=="*50, "Starts")

                # 2. Search for bulk entry
                for bulk_collection in [db_w1se2_hse_local, owls_formation_energy]:
                # bulk_collection = db_w1se2_hse_local
                #     bulks = bulk_collection.find(dict(defect_type="bulk", nsites=bulk_nsite))
                    # if bulks.count() == 0:
                    #     print("(Bulk) Not found on db_w1se2_hse_local.json, searching in db_owls_formation_energy.json!")
                    # bulk_collection = owls_formation_energy
                    bulks = bulk_collection.find(
                            dict(defect_type="bulk", nsites=bulk_nsite)
                    )
                    for bulk in bulks:
                        lz = bulk["input"]["structure"]["lattice"]["c"]
                        area = bulk["nsites"]
                        bulk_task_id = bulk["task_id"]

                        # 3. Search for defect entry
                        defect_entry_filter = {
                            "task_type": "scf",
                            "charge_state": charge,
                            "defect_type": defect_type,
                            "nsites": area-1 if "vacancy" in defect_type else area,
                            "input.structure.lattice.c": lz
                        }
                        vac_collection = db_w1se2_hse_local
                        defect_entry = vac_collection.find_one(defect_entry_filter)

                        if not defect_entry:
                            # print("(Defect) Not found on db_w1se2_hse_local.json, "
                            #       "searching in db_owls_formation_energy.json!")
                            vac_collection = owls_formation_energy
                            defect_entry = vac_collection.find_one(defect_entry_filter)

                        if defect_entry is None:
                            # print("!!"*25, "Not Found Compatible Defect!")
                            # print("=="*50, "Starts")

                            continue
                        # 3. Create DefectEntry
                        try:
                            print("=="*50)
                            # print("bulk_task_id: {}, defect_task_id: {}".format(bulk_task_id, defect_entry["task_id"]))
                            # print("BULK => lz:{}, nsites:{}, charge:{}, "
                            #       "bulk_task_id:{}".format(lz, area, charge, bulk_task_id))
                            _instance = cls(bulk_collection, bulk_task_id, vac_collection, defect_entry["task_id"])
                            d_entry = _instance.defect_entry(entry_id={
                                "area":
                                    area-1 if "vacancy" in defect_type else area,
                                "lz": defect_entry["input"]["structure"]["lattice"]["c"]})
                            defect_entries.append(d_entry)
                            # print(
                            #     "DEFECT => nsites:{}, lz:{:.3f}\ntask_id: {}, energy: {:.3f}, Ef: {:.3f}, vbm: {} ".format(
                            #         area-1 if "vacancy" in defect_type else area,
                            #         defect_entry["input"]["structure"]["lattice"]["c"],
                            #         defect_entry["task_id"],
                            #         d_entry.uncorrected_energy,
                            #         d_entry.formation_energy(fermi_level=0),
                            #         d_entry.parameters["vbm"]
                            #     )
                            # )
                            bulk_data = {
                                "type": "bulk",
                                "task_id": bulk_task_id,
                                "charge": charge,
                                "nsites": area,
                                "lz": lz,
                                "vbm": bulk["calcs_reversed"][0]["output"]["vbm"],
                                "bandgap": bulk["calcs_reversed"][0]["output"]["bandgap"],
                                "Etot": bulk["calcs_reversed"][0]["output"]["energy"]
                            }
                            defect_data = {
                                "type": "defect",
                                "task_id": defect_entry["task_id"],
                                "charge": d_entry.charge,
                                "nsites": d_entry.entry_id.get("area"),
                                "vbm": d_entry.parameters.get("vbm"),
                                "bandgap": d_entry.parameters.get("band_gap"),
                                "lz": d_entry.entry_id.get("lz"),
                                "Etot": _instance.vac_energy,
                                "Ef": d_entry.formation_energy(),
                            }
                            print(pd.DataFrame([bulk_data, defect_data]))
                            data1.append(defect_data)
                            data2.append(bulk_data)
                        except Exception as error:
                            print("error: {}".format(error))
                            continue
        print(pd.DataFrame(data1))
        print(pd.DataFrame(data2))
        defect_pd = pd.DataFrame(data1)

        # 4. Create DefectPhaseDiagram and find Chemical Potential from MPRester
        def find_tls_chempot(defect_entry):

            dpd = DefectPhaseDiagram(defect_entry,
                                     -2.7102,
                                     #min([i.parameters.get("vbm") for i in defect_entry]),
                                     max([i.parameters.get("band_gap") for i in defect_entry])
                                     )
            print("++" * 50, "Finally,")
            print('\ntransition levels:{}'.format(dpd.transition_level_map))

            red_comp = defect_entry[0].bulk_structure.composition.reduced_composition
            elt_set = list(defect_entry[0].bulk_structure.symbol_set)
            with MPRester() as mp:
                pd_ents = mp.get_entries_in_chemsys(elt_set, False)
            pd = PhaseDiagram(pd_ents)
            cp_dict = pd.get_all_chempots(red_comp)

            print('\nResulting chemical potentials:')
            for k, v in cp_dict.items():
                print("\t{}: {}".format(k, v))

            return dpd, cp_dict

        # 5. Create a dict containing all tls with different geo spec (Passing to calculate IE0)
        for nsite in defect_pd.nsites.unique():
            for lz in defect_pd.lz.unique():
                print((nsite, lz))
                nsite_condition = defect_pd["nsites"] == nsite
                lz_condition = defect_pd["lz"] == lz
                d_entry_with_same_geo = defect_pd[nsite_condition & lz_condition]["d_entry"]
                print(defect_pd[nsite_condition & lz_condition]["Ef"])

                dpd, cp = find_tls_chempot(d_entry_with_same_geo)
                print(dpd.entries)
                tls_map = dpd.transition_level_map
                print(tls_map)

                p = dpd.plot(ylim=None, ax_fontsize=1.3, fermi_level=None)

                stable_entries = dpd.stable_entries
                # for stable_entry, tls in tls_map.items():
                #     for en, chg in tls.items():
                #         lz = stable_entries[stable_entry][0].bulk_structure.lattice.c
                #         area = stable_entries[stable_entry][0].bulk_structure.num_sites
                #         # area = stable_entries[stable_entry][0].bulk_structure.lattice.a*\
                #         #        stable_entries[stable_entry][0].bulk_structure.lattice.b*\
                #         #        math.sin(stable_entries[stable_entry][0].bulk_structure.lattice.gamma)
                #         defect_tls.append((area, lz, tuple(sorted(chg)), en))
                # print(defect_tls)










        # 6. Get IE0
        def ionized_energy(tls, band_gap, charge_transition):
            """
            Get IE0 using linear regression
            :param defect_tls (dict): {(Area, lz): {(0,1):tl, ...}}
            :return:
            """
            data = []
            for tl in tls:
                data.append({
                    "charge": tl[2],
                    "area": tl[0],
                    "lz": tl[1],
                    "transition energy": tl[3]
                })

            pd_data = pd.DataFrame(data)
            print(pd.DataFrame(data))
            if not charge_transition:
                charge_transition = pd_data.area.unique()

            def linear_regression(X, y):
                regr = LinearRegression()
                regr.fit(X, y)
                print("==" * 50, "linear regression")
                # print("alpha:{}, beta:{}, IE0:{}, score:{}".format(regr.coef_[0], regr.coef_[1], regr.intercept_,
                #                                                    regr.score(X, y)))
                return regr.intercept_, regr.coef_[0], regr

            print(pd_data.area.unique())
            results = []
            for chg in charge_transition:
                IE_A_0 = []
                for i in pd_data.area.unique():
                    sheet = pd_data[pd_data["area"] == i]
                    A_X = np.array([[i] for i in sheet.lz.unique()])
                    A_y = None
                    if chg[0] * chg[1] == 0 and (chg[0] < 0 or chg[1] < 0):
                        A_y = sheet[sheet["charge"] == chg]["transition energy"]
                    elif chg[0] * chg[1] == 0 and (chg[0] > 0 or chg[1] > 0):
                        A_y = band_gap - sheet[sheet["charge"] == chg]["transition energy"]
                    elif (chg[0]*chg[1] < 0):
                        A_y = sheet[sheet["charge"] == chg]["transition energy"]
                    elif (chg[0]*chg[1] > 0):
                        if chg == (-2, -1):
                            A_y = sheet[sheet["charge"] == chg]["transition energy"]
                        else:
                            A_y = sheet[sheet["charge"] == chg]["transition energy"]
                    A_result = linear_regression(A_X, A_y)
                    print(pd.DataFrame([{"IE":v, "lz":k} for k, v in zip(A_X, A_y)]))
                    print("IE(A,0):{}, beta:{}".format(A_result[0], A_result[1]*i))
                    IE_A_0.append({"area": i/3, "IE(A,0)": A_result[0], "beta": A_result[1]*i})
                tot_X = np.array([[1/math.sqrt(i["area"])] for i in IE_A_0])
                tot_y = np.array([[i["IE(A,0)"]] for i in IE_A_0])
                print(pd.DataFrame([{"IE(A,0)":v, "1/sqrt(A)":k} for k,v in zip(tot_X, tot_y)]))
                tot = linear_regression(tot_X, tot_y)
                print(tot[0], tot[1])
                results.append({"charge": chg, "IE0": tot[0], "alpha": tot[1]})
            print(pd.DataFrame(results))

    @staticmethod
    def real_IE0(chg_min, chg_max, chemsys):
        IE_sheet = []
        owls_formation_energy = VaspCalcDb.from_db_file(
            db_file='/Users/jeng-yuantsai/PycharmProjects/git/my_pycharm_projects/database/'
                    'db_config/formation_energy_related/db_owls_formation_energy.json').collection
        db_w1se2_hse_local = VaspCalcDb.from_db_file(db_file=db).collection

        for charge in range(chg_min, chg_max + 1, 1):
            ionization_energy = defaultdict()
            if charge != 0:
                e = list(db_w1se2_hse_local.find(
                    {
                        "defect_formation_energy_linear_fitting": {"$exists": 1},
                        "charge_state": charge,
                        "chemsys": chemsys
                    }
                ))
                e2 = list(owls_formation_energy.find(
                    {
                        "defect_formation_energy_linear_fitting": {"$exists": 1},
                        "charge_state": charge,
                        "chemsys": chemsys
                    }
                ))
                temp_data = []
                e = e + e2
                for i in e:
                    temp = defaultdict()
                    temp["charge_state"] = i["charge_state"]
                    temp["task_id"] = i["task_id"]
                    temp["area"] = i["defect_formation_energy_linear_fitting"][3]
                    temp["lz"] = i["defect_formation_energy_linear_fitting"][4]
                    temp["IE"] = i["defect_formation_energy_linear_fitting"][2]
                    temp["bandgap"] = i["defect_formation_energy_linear_fitting"][5]
                    temp_data.append(temp)
                print(pd.DataFrame(temp_data))

                # Do linear regression y=a0+a1*x1+a2*x2+a3*x3 which a0=IE0, a1=alpha, a2=beta, and a3=-1;
                # x1=1/sqrt(S), x2=Lz/S, and x3=IE
                X = np.array(
                    [
                        [
                            1 / math.sqrt(i["defect_formation_energy_linear_fitting"][3]),
                            (i["defect_formation_energy_linear_fitting"][4] /
                             i["defect_formation_energy_linear_fitting"][3])
                        ]
                        for i in e
                    ]
                )
                y = np.array([i["defect_formation_energy_linear_fitting"][2] for i in e])

                def linear_regression(X, y):
                    regr = LinearRegression()
                    regr.fit(X, y)
                    print("=="*50, "linear regression")
                    print("alpha:{}, beta:{}, IE0:{}".format(regr.coef_[0], regr.coef_[1], regr.intercept_))
                    return regr.coef_[0], regr.coef_[1], regr.intercept_, regr.score(X, y)

                IE0 = linear_regression(X, y)

                # Solving equation:
                def solve_equation():
                    IE0 = np.linalg.solve(
                        np.array(
                            [
                                [
                                    1/math.sqrt(i["defect_formation_energy_linear_fitting"][3]),
                                    (i["defect_formation_energy_linear_fitting"][4] /
                                     i["defect_formation_energy_linear_fitting"][3]),
                                    1
                                ]
                                for i in e
                            ]
                        ),
                        np.array(
                            [i["defect_formation_energy_linear_fitting"][2] for i in e]
                        )
                    )
                    return IE0

                ionization_energy["charge"] = charge
                ionization_energy["alpha"] = IE0[0]
                ionization_energy["beta"] = IE0[1]
                ionization_energy["IE0"] = IE0[2]
                print("SCORE: {}". format(IE0[3]))
                IE_sheet.append(ionization_energy)

        print(pd.DataFrame(IE_sheet))
        return IE_sheet


class AnistropicCorrection:
    def __init__(self, bulk_db_file, bulk_entry_filter, defect_db_file, defect_entry_filter,
                 dielectric_db_file, dielectric_filter, charge, defect_site_in_bulk):

        self.bulk_db_file = bulk_db_file
        self.bulk_entry_filter = bulk_entry_filter
        self.bulk_entry = VaspCalcDb.from_db_file(self.bulk_db_file).collection.find_one(self.bulk_entry_filter)
        print(self.bulk_entry["task_id"])

        self.defect_db_file = defect_db_file
        self.defect_entry_filter = defect_entry_filter
        self.defect_entry = VaspCalcDb.from_db_file(self.defect_db_file).collection.find_one(self.defect_entry_filter)
        print(self.defect_entry["task_id"])

        # self.bulk_structure = Structure.from_file(os.path.join(bulk_path, "POSCAR.gz"))
        self.bulk_structure = Structure.from_dict(self.bulk_entry["input"]["structure"])
        # self.defect_structure = Structure.from_file(os.path.join(defect_path, "POSCAR.gz"))
        self.defect_structure = Structure.from_dict(self.defect_entry["input"]["structure"])

        # self.bulk_outcar = Outcar(os.path.join(bulk_path, "OUTCAR.gz"))
        self.bulk_outcar = self.bulk_entry["calcs_reversed"][0]["output"]["outcar"]
        # self.defect_outcar = Outcar(os.path.join(defect_path, "OUTCAR.gz"))
        self.defect_outcar = self.defect_entry["calcs_reversed"][0]["output"]["outcar"]
        self.defect_site_in_bulk = defect_site_in_bulk
        self.charge = charge
        # self.uncorrected_energy = Vasprun(os.path.join(defect_path, "vasprun.xml.gz")).final_energy - \
        #                           Vasprun(os.path.join(bulk_path, "vasprun.xml.gz")).final_energy
        self.uncorrected_energy = self.defect_entry["calcs_reversed"][0]["output"]["energy"] - \
                                  self.bulk_entry["calcs_reversed"][0]["output"]["energy"]
        # self.vbm = Vasprun(os.path.join(bulk_path, "vasprun.xml.gz")).get_band_structure().get_vbm()["energy"]
        self.vbm = self.bulk_entry["calcs_reversed"][0]["output"]["vbm"]
        self.bandgap = self.bulk_entry["calcs_reversed"][0]["output"]["bandgap"]
        self.dielectric_entry = VaspCalcDb.from_db_file(dielectric_db_file).collection.find_one(dielectric_filter)
        self.dielectric_tensor = self.dielectric_entry["calcs_reversed"][0]["output"]["outcar"]["dielectric_tensor"]

    def _site_matching_indices(self):
        matching = []
        for bulk_site, bulk_specie in enumerate(self.bulk_structure.species):
            sites_pair = []
            if bulk_specie == self.defect_structure.species[bulk_site]:
                sites_pair.append(bulk_site)
                sites_pair.append(bulk_site)
                matching.append(sites_pair)
        print("matching list: {}".format(matching))
        return matching

    def _kumagai_correction(self):
        kumagai_params = {
            "bulk_atomic_site_averages": self.bulk_outcar["electrostatic_potential"],
            "defect_atomic_site_averages": self.defect_outcar["electrostatic_potential"],
            "site_matching_indices": self._site_matching_indices(),
            "initial_defect_structure": self.defect_structure,
            "defect_frac_sc_coords": self.defect_structure.sites[self.defect_site_in_bulk].frac_coords  # defect_frac_sc_coords (array): Defect Position in fractional coordinates of the supercell
        }
        kc = KumagaiCorrection(self.dielectric_tensor)
        return kumagai_params, kc

    def defect_en(self):
        kumagai_parameters = self._kumagai_correction()[0]
        kumagai_parameters.update({"vbm": self.vbm, "bandgap": self.bandgap})
        substitution = Substitution(self.bulk_structure, self.bulk_structure[self.defect_site_in_bulk], self.charge)
        dn = DefectEntry(substitution, self.uncorrected_energy,
                    parameters=self._kumagai_correction()[0]
                    )
        kc_corr = self._kumagai_correction()[1].get_correction(dn)
        dn = DefectEntry(
            substitution, self.uncorrected_energy,
            parameters=kumagai_parameters,
            corrections=kc_corr
        )
        # a = self._kumagai_correction()[1]
        # a.get_correction(dn)
        # p = a.plot()
        # p.show()
        print(self.dielectric_tensor)
        print(dn.corrections)
        print(dn.uncorrected_energy+self.charge*self.vbm)
        print(dn.formation_energy())
        print(self.bandgap)


class FormationEnergy2D:

    def __init__(self, bulk_db_files, bulk_entry_filter, defect_db_files, defect_entry_filter,
                 bk_vbm_bg_db_files, bk_vbm_bg_filter):
        bulk_entries = []
        for bulk_db_file in bulk_db_files:
            bulk_db_col = VaspCalcDb.from_db_file(db_file=bulk_db_file).collection
            for e in bulk_db_col.find(bulk_entry_filter):
                bulk_entries.append(e)

        defect_entries = []
        for defect_db_file in defect_db_files:
            defect_db_col = VaspCalcDb.from_db_file(db_file=defect_db_file).collection
            for e in defect_db_col.find(defect_entry_filter):
                defect_entries.append(e)

        bk_vbm_bg_entries = []
        for bk_db_file in bk_vbm_bg_db_files:
            bk_vbm_bg_db_col = VaspCalcDb.from_db_file(db_file=bk_db_file).collection
            for e in bk_vbm_bg_db_col.find(bk_vbm_bg_filter):
                bk_vbm_bg_entries.append(e)

        pd1 = pd.DataFrame({"bulk": [i["task_id"] for i in bulk_entries]})
        pd2 = pd.DataFrame({"defect_id": [i["task_id"] for i in defect_entries]})
        pd3 = pd.DataFrame({"bk_vbm_bg": [i["task_id"] for i in bk_vbm_bg_entries]})
        print(pd.concat([pd1, pd2, pd3], axis=1))

        # pd4 = pd.DataFrame({
        #     "task_id": [i["task_id"] for i in defect_entries],
        #     "charge_state": [i["charge_state"] for i in defect_entries],
        #     "nsites":[i["nsites"] for i in defect_entries],
        #     "Lz":[i["calcs_reversed"][0]["input"]["structure"]["lattice"]["c"] for i in defect_entries],
        #     "magmom":[i["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"] for i in defect_entries]
        # }).sort_values(["nsites", "Lz", "charge_state"])
        # print("\n{}".format(pd4))

        print("bulk_entries: {}, defect_entries: {}, bk_vbm_bg_entries: {}".format(len(bulk_entries), len(defect_entries),
                                                                                   len(bk_vbm_bg_entries)))
        self.defect_entry_objects = defaultdict(list)
        for bulk, bk_vbm_bg in zip(bulk_entries, bk_vbm_bg_entries):
            nsites = bulk["nsites"]
            lz = bulk["input"]["structure"]["lattice"]["c"]
            for bk_vbm_bg in bk_vbm_bg_entries:
                if bk_vbm_bg["nsites"] == nsites and bk_vbm_bg["input"]["structure"]["lattice"]["c"] == lz:
                    vbm = bk_vbm_bg["calcs_reversed"][0]["output"]["vbm"]
                    bg = bk_vbm_bg["calcs_reversed"][0]["output"]["bandgap"]
                    for defect in defect_entries:
                        area = 0.5*defect["input"]["structure"]["lattice"]["a"]* \
                               defect["input"]["structure"]["lattice"]["b"]* \
                               math.sin(math.radians(defect["input"]["structure"]["lattice"]["gamma"]))
                        if defect["nsites"] == nsites-1 and defect["input"]["structure"]["lattice"]["c"] == lz:
                            d_obj = Vacancy(structure=Structure.from_dict(bulk["input"]["structure"]),
                                            defect_site=Structure.from_dict(bulk["input"]["structure"])[0],
                                            charge=defect["charge_state"])
                            defect_entry_obj = DefectEntry(
                                defect=d_obj,
                                uncorrected_energy=defect["calcs_reversed"][0]["output"]["energy"] -
                                                   bulk["calcs_reversed"][0]["output"]["energy"],
                                parameters={
                                    "vbm": vbm,
                                    "bandgap": bg,
                                    "task_id": defect["task_id"],
                                    "nsites": defect["nsites"],
                                    "Lz": defect["calcs_reversed"][0]["input"]["structure"]["lattice"]["c"],
                                    "magmom": defect["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]
                                }
                            )
                            self.defect_entry_objects[(area, lz)].append(defect_entry_obj)

                        elif defect["nsites"] == nsites and defect["input"]["structure"]["lattice"]["c"] == lz:
                            d_obj = Substitution(structure=Structure.from_dict(bulk["input"]["structure"]),
                                                 defect_site=Structure.from_dict(bulk["input"]["structure"])[0],
                                                 charge=defect["charge_state"])
                            defect_entry_obj = DefectEntry(
                                defect=d_obj,
                                uncorrected_energy=defect["calcs_reversed"][0]["output"]["energy"] -
                                                   bulk["calcs_reversed"][0]["output"]["energy"],
                                parameters={
                                    "vbm": vbm,
                                    "bandgap": bg,
                                    "task_id": defect["task_id"],
                                    "nsites": defect["nsites"],
                                    "Lz": defect["calcs_reversed"][0]["input"]["structure"]["lattice"]["c"],
                                    "magmom": defect["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]
                                }
                            )
                            self.defect_entry_objects[(area, lz)].append(defect_entry_obj)

    def transition_levels(self):
        # 4. Create DefectPhaseDiagram
        tls_info = defaultdict(dict)
        # show energy
        for geo, defect_entry_obj in self.defect_entry_objects.items():
            pd4 = pd.DataFrame({
                "task_id": [i.parameters["task_id"] for i in defect_entry_obj],
                "charge_state": [i.charge for i in defect_entry_obj],
                "nsites": [i.parameters["nsites"] for i in defect_entry_obj],
                "Lz": [i.parameters["Lz"] for i in defect_entry_obj],
                "magmom": [i.parameters["magmom"] for i in defect_entry_obj],
                "Ef": [i.formation_energy() for i in defect_entry_obj],
                "vbm": [i.parameters["vbm"] for i in defect_entry_obj],
                "bandgap": [i.parameters["bandgap"] for i in defect_entry_obj]
            }).sort_values(["nsites", "Lz", "charge_state"])
            print("\n{}".format(pd4))

            dpd = DefectPhaseDiagram(
                defect_entry_obj,
                vbm=defect_entry_obj[0].as_dict()["parameters"]["vbm"],
                band_gap=defect_entry_obj[0].as_dict()["parameters"]["bandgap"]
            )
            print('transition levels:{}'.format(dpd.transition_level_map))

            p = dpd.plot(ylim=None, ax_fontsize=1.3, fermi_level=None, title=geo)
            p.show()
            tls_info[tuple(list(geo)+[dpd.band_gap])] = \
                dict(zip([tuple(sorted(i)) for i in list(dpd.transition_level_map.values())[0].values()],
                         list(dpd.transition_level_map.values())[0].keys()))

        self.tls = tls_info
        return tls_info


    # 6. Get IE0
    def ionized_energy(self, calc_name, chemsys):

        data = []
        for geo, tl in self.tls.items():
            for chg, tl_energy in tl.items():
                data.append({
                    "charge": chg,
                    "area": geo[0],
                    "1/sqrt(area)": 1/math.sqrt(geo[0]),
                    "lz": geo[1],
                    "bandgap": geo[2],
                    "transition energy": tl_energy
                })

        pd_data = pd.DataFrame(data)
        pd_data.sort_values(["charge", "area", "lz"]).to_clipboard()
        print("\n{}".format(pd.DataFrame(data).sort_values(["charge", "area", "lz"])))

        def linear_regression(X, y):
            regr = LinearRegression()
            regr.fit(X, y)
            print("==" * 50, "linear regression")
            # print("alpha:{}, beta:{}, IE0:{}, score:{}".format(regr.coef_[0], regr.coef_[1], regr.intercept_,
            #                                                    regr.score(X, y)))
            return regr.intercept_, regr.coef_[0], "{:.2%}".format(regr.score(X,y)), regr.predict,

        fig, axes = plt.subplots(4,1,figsize=(5,8))
        fig.subplots_adjust(hspace=1, wspace=1)

        areas = np.sort(pd_data.area.unique())
        charge = pd_data.charge.unique()
        results = []
        for idx, chg in enumerate(charge):
            idx *= 2
            IE_A_0 = []
            for i in areas:
                sheet = pd_data[pd_data["area"] == i]
                A_X = np.array([[i] for i in sheet.lz.unique()])
                A_y = None
                if chg[0] * chg[1] == 0 and (chg[0] < 0 or chg[1] < 0):
                    A_y = sheet[sheet["charge"] == chg]["transition energy"]
                elif chg[0] * chg[1] == 0 and (chg[0] > 0 or chg[1] > 0):
                    print("right!!")
                    A_y = sheet[sheet["charge"] == chg]["bandgap"] - sheet[sheet["charge"] == chg]["transition energy"]
                elif (chg[0]*chg[1] < 0):
                    A_y = sheet[sheet["charge"] == chg]["transition energy"]
                elif (chg[0]*chg[1] > 0):
                    A_y = sheet[sheet["charge"] == chg]["transition energy"]
                try:
                    A_result = linear_regression(A_X, A_y)
                    #plot
                    print(i)
                    area = dict(zip(areas, ["4X4", "5X5", "6X6"]))
                    colors = dict(zip(areas, ["forestgreen", "dodgerblue", "darkorange"]))
                    axes[idx].plot(A_X, A_y, "o", color=colors[i], label=area[i])
                    axes[idx].plot(np.append(np.array([[0]]), A_X, axis=0),
                                   A_result[-1](np.append(np.array([[0]]), A_X, axis=0)), color=colors[i],
                                   linestyle="dotted")
                except Exception as err:
                    print(err)
                    continue
                print(pd.DataFrame([{"IE":v, "lz":k} for k, v in zip(A_X, A_y)]))
                print("IE(A,0):{}, beta:{}".format(A_result[0], A_result[1]*i))
                IE_A_0.append({"area": i, "IE(A,0)": A_result[0], "beta": A_result[1]*i})
            tot_X = np.array([[1/math.sqrt(i["area"])] for i in IE_A_0])
            tot_y = np.array([[i["IE(A,0)"]] for i in IE_A_0])
            tot = linear_regression(tot_X, tot_y)
            results.append({"charge": chg, "IE0": tot[0], "alpha": tot[1], "Score":tot[2]})
            print(pd.DataFrame([{"IE(S,0)": v, "1/sqrt(A)": k} for k, v in zip(tot_X, tot_y)]))
            print("Intercept:{}, Slop:{}, Score:{}".format(tot[0], tot[1], tot[2]))

            # plot
            colors = ["forestgreen", "dodgerblue", "darkorange"]
            axes[idx+1].plot(np.append(np.array([[0]]), tot_X, axis=0),
                             tot[-1](np.append(np.array([[0]]), tot_X, axis=0)), linestyle="dotted", color="black")
            for x, y, color in zip(tot_X, tot_y, colors):
                axes[idx+1].plot(x, y, "o", color=color)

            axes[idx].set_xlabel(r"$\mathrm{L_z\;(\AA)}$")
            axes[idx].set_ylabel(r"$\mathrm{IE(S, L_z)\;(eV)}$")
            axes[idx].xaxis.set_minor_locator(AutoMinorLocator())
            axes[idx].yaxis.set_minor_locator(AutoMinorLocator())
            axes[idx].legend(loc=2, ncol=3)

            axes[idx+1].set_xlabel(r"$\mathrm{\frac{1}{\sqrt{S}}\;(\AA^{-1})}$")
            axes[idx+1].set_ylabel(r"$\mathrm{IE(S)\;(eV)}$")
            axes[idx+1].xaxis.set_minor_locator(AutoMinorLocator())
            axes[idx+1].yaxis.set_minor_locator(AutoMinorLocator())


        axes[0].set_xlim(left=0)
        axes[0].set_ylim(bottom=0, top=3)
        axes[0].set_title("Step 1. for donor state "+r"$\epsilon (+/0)$")
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(bottom=0, top=3)
        axes[1].set_title("Step 2. for donor state "+r"$\epsilon (+/0)$")
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(bottom=0.5, top=3)
        axes[2].set_title("Step 1. for acceptor state "+r"$\epsilon (0/-)$")
        axes[3].set_xlim(left=0)
        axes[3].set_ylim(bottom=0.5, top=2)
        axes[3].set_title("Step 2. for acceptor state "+r"$\epsilon (0/-)$")

        print("**"*20, ",Finally")
        print(pd.DataFrame(results))
        fig.suptitle(chemsys, fontsize=16)
        # fig.savefig("/Users/jeng-yuantsai/Research/qubit/plt/{}.eps".format(calc_name), format="eps")
        plt.show()
        return results


def main():
    def SP_Kpoints_antisite(collection_name, compounds, aexx=0.25):
        path = "/Users/jeng-yuantsai/Research/qubit/calculations/"
        K = [[-0.33333333, -0.33333333, 0.0], [0,0,0]]
        quarter = [0.25, 0.25, 0]
        results = []
        for compound in compounds:
            se_antisite = FormationEnergy2D(
                bulk_db_files=[os.path.join(path, collection_name, "db.json")],
                bulk_entry_filter={
                    "chemsys": compound,
                    "formula_anonymous": "AB2",
                    "calcs_reversed.run_type": "HSE06",
                    "input.structure.lattice.c": {"$in": [20, 30, 40]},
                    "calcs_reversed.input.kpoints.kpoints.0": {"$in": [quarter]},
                    "input.parameters.AEXX": aexx,
                    "charge_state": 0,
                    "task_id": {"$nin":[1306, 1315, 1295, 1297, 1298]}
                },
                bk_vbm_bg_db_files=[os.path.join(path, collection_name, "db.json")],
                bk_vbm_bg_filter={
                    "chemsys": compound,
                    "formula_anonymous": "AB2",
                    "calcs_reversed.run_type": "HSE06",
                    "calcs_reversed.input.kpoints.kpoints.0": {"$in": K},
                    "input.structure.lattice.c": {"$in": [20, 30, 40]},
                    "charge_state": 0,
                    "input.parameters.AEXX": aexx
                },
                defect_db_files=[os.path.join(path, collection_name, "db.json")],
                defect_entry_filter={
                    "charge_state": {"$in": [1, 0, -1]},
                    "chemsys": compound,
                    "formula_anonymous": {"$in": ["A50B97", "A37B71"]},
                    "calcs_reversed.input.kpoints.kpoints.0": quarter,
                    "input.incar.NSW": 0,
                    "input.incar.LASPH": True,
                    "calcs_reversed.run_type": "HSE06",
                    "input.structure.lattice.c": {"$in": [20, 30, 40]},
                    "input.parameters.AEXX": aexx,
                    "calcs_reversed.output.outcar.total_magnetization": {"$lte": 3.1, "$gte": 0},
                    # "task_id": {"$nin": [1564, 2042, 1472, 1579, 3664]} # 3656, 3645, 3664
                    "task_id": {"$nin": [3625, 3661, 3664]} #for ws2
                    #[3510, 3537] for mos2
                }
            )
            se_antisite.transition_levels()
            for a in se_antisite.ionized_energy():
                a.update({"name": compound})
                results.append(a)



    def regular_antisite(project, collection_name, compounds, aexx=0.25):
        path = "/Users/jeng-yuantsai/Research/project/qubit/calculations/"
        c = [20, 25]
        # bulk_k = [[0.25, 0.25, 0], [-0.5, -0.5, 0]]
        bulk_k = [[0,0,0]]
        vbm_k = [[0, 0, 0], [-0.33333333, -0.33333333, 0], [0.33333333, 0.33333333,0]]
        results = []
        for compound in compounds:
            se_antisite = FormationEnergy2D(
                bulk_db_files=[os.path.join(path, project, collection_name, "db.json")],
                bulk_entry_filter={
                    # "nsites": {"$ne":48},
                    "chemsys": compound,
                    "formula_anonymous": "AB2",
                    "calcs_reversed.run_type": "HSE06",
                    "input.structure.lattice.c": {"$in": c},
                    "calcs_reversed.input.kpoints.kpoints.0": {"$in": bulk_k},
                    "input.parameters.AEXX": aexx,
                    "charge_state": 0,
                    "defect_type": "host",
                    "task_label": "HSE_scf"
                },
                bk_vbm_bg_db_files=[os.path.join(path, project, collection_name, "db.json")],
                bk_vbm_bg_filter={
                    # "nsites": {"$ne":48},
                    "chemsys": compound,
                    "formula_anonymous": "AB2",
                    "calcs_reversed.run_type": "HSE06",
                    "calcs_reversed.input.kpoints.kpoints.0": {"$in": vbm_k},
                    "input.structure.lattice.c": {"$in": c},
                    "charge_state": 0,
                    "input.parameters.AEXX": aexx,
                    "defect_type": "vbm",
                    "task_label": "HSE_scf"
                },
                defect_db_files=[os.path.join(path, project, collection_name, "db.json")],
                defect_entry_filter={
                    # "nsites": {"$ne":48},
                    "charge_state": {"$in": [1, 0, -1]},
                    "chemsys": compound,
                    "formula_anonymous": {"$in": ["A26B49", "A37B71", "A17B31"]}, #"A17B31"
                    "calcs_reversed.input.kpoints.kpoints.0": {"$in": bulk_k},
                    "input.incar.NSW": 0,
                    "input.incar.LASPH": True,
                    "calcs_reversed.run_type": "HSE06",
                    "input.structure.lattice.c": {"$in": c},
                    "input.parameters.AEXX": aexx,
                    "defect_type": {"$in":["defect_new", "defect"]},
                    "task_label": "HSE_scf"
                    # "calcs_reversed.output.outcar.total_magnetization": {"$lte": 3.1, "$gte": 0},
                    # "task_id": {"$nin": [327]}
                     #for ws2
                    #[3510, 3537] for mos2
                }
            )
            se_antisite.transition_levels()
            for a in se_antisite.ionized_energy(collection_name, compound):
                a.update({"name": compound})
                results.append(a)
        # print("$$"*50, "all")
        # print(pd.DataFrame(results))

    # regular_antisite("antisiteQubit", "W_Se_Ef_gamma", ["Se-W"])
    regular_antisite("antisiteQubit", "Mo_Te_Ef_gamma", ["Mo-Te"])



if __name__ == '__main__':
    main()


