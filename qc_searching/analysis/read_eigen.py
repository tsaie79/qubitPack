from pymatgen.io.vasp.outputs import Vasprun, Procar
from pymatgen.electronic_structure.core import Spin
from atomate.vasp.database import VaspCalcDb
import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
from qubitPack.qc_searching.py_energy_diagram.application.defect_levels import EnergyLevel
from collections import defaultdict
from qubitPack.qc_searching.analysis.dos_plot_from_db import DB_CONFIG_LOCAL, FIG_SAVING_FILE_PATH


class EigenParse:
    def __init__(self, path):
        self.path = path
        self.vp = Vasprun(os.path.join(self.path, "vasprun.xml"))

    def eigenvalue(self, spin, k_index):
        global s
        if spin == 1:
            s = Spin.up
        elif spin == 2:
            s = Spin.down

        ev = self.vp.eigenvalues
        print(self.vp.actual_kpoints)
        print("=="*50)
        return ev[s][k_index]

    def usage(self):
        v = []
        for i in range(4):
            for j in [1, 2]:
                v.append(self.eigenvalue(j, i))
        comb = np.concatenate(tuple(v), 1)
        # pd.DataFrame(defect.eigenvalue(1, 0)).to_excel("./pic/defect-1.xlsx", index=False)
        # pd.DataFrame(host.eigenvalue(1, 0)).to_excel("./pic/host-1.xlsx", index=False)

        pd.DataFrame(comb).to_excel(os.path.join(self.path, "ev.xlsx"), index=False)

        return None


class ProcarParse:
    def __init__(self, path):
        self.path = path
        self.procar = Procar(os.path.join(self.path, "PROCAR"))
        self.vasp_run = Vasprun(os.path.join(self.path, "vasprun.xml"))
        self.total_up = None
        self.total_down = None
        self.up_energy = None
        self.down_energy = None

    def band(self, write_out, adjacent, k_index, show_nonadjacent=False):
        band = self.procar
        pro_up = []
        pro_down = []
        total_up = []
        total_down = []
        print(self.path.split("/")[-3])
        for j in range(2, 169+1):
            j = j - 2
            print("---band %d---" % (j+2))
            sum_up = 0
            sum_down = 0
            array_up = []
            array_down = []
            if not show_nonadjacent:
                for i in adjacent:
                    up = band.data[Spin.up][k_index][j][i]
                    sum_up += np.sum(up)
                    up = np.append(up, np.sum(up))
                    down = band.data[Spin.down][k_index][j][i]
                    sum_down += -np.sum(down)
                    down = np.append(down, np.sum(down))
                    print(up, down)
                    array_up.append((i, up.tolist()))
                    array_down.append((i, down.tolist()))
                pro_up.append((j+2, dict(array_up)))
                pro_down.append((j+2, dict(array_down)))
                total_up.append(sum_up)
                total_down.append(sum_down)

            elif show_nonadjacent:
                for i in set(range(band.nions))-set(adjacent):
                    up = band.data[Spin.up][k_index][j][i]
                    up = np.append(up, np.sum(up))
                    down = band.data[Spin.down][k_index][j][i]
                    down = np.append(down, np.sum(down))
                    print(up, down)
                    array_up.append((i, up.tolist()))
                    array_down.append((i, down.tolist()))
                pro_up.append((j+2, dict(array_up)))
                pro_down.append((j+2, dict(array_down)))
            else:
                print("show_nonadjacent = True if one wants to show ions except adjacent ones.")
        print(total_up, total_down)
        if write_out and not show_nonadjacent:
            with open(os.path.join(self.path, self.path.split("/")[-3]+"_up_procar.js"), "w") as f:
                json.dump(dict(pro_up), f, indent=4)
                f.close()
            with open(os.path.join(self.path, self.path.split("/")[-3]+"_down_procar.js"), "w") as f:
                json.dump(dict(pro_down), f, indent=4)
                f.close()

        elif write_out and show_nonadjacent:
            with open(os.path.join(self.path, self.path.split("/")[-3]+"_nonadjacent_up_procar.js"), "w") as f:
                json.dump(dict(pro_up), f, indent=4)
                f.close()
            with open(os.path.join(self.path, self.path.split("/")[-3]+"_nonadjacent_down_procar.js"), "w") as f:
                json.dump(dict(pro_down), f, indent=4)
                f.close()
        else:
            print("Write-out was not processed!")

        def print_total():
            y1 = total_up
            y2 = total_down
            x = range(2, 169+1)
            fig, ax = plt.subplots()
            ax.plot(x, y1, "-", label="up")
            ax.plot(x, y2, "-", label="down")
            ax.set_xlabel('Band index')
            ax.set_ylabel('Total projection')
            ax.set_title(self.path.split("/")[-3] + '-Net Orbital Projection on Adjacent Ions' + " (k index=%d)" % k_index)
            plt.grid(linestyle="--")
            # plt.axvline(x=80, color="k", linestyle="-")

            plotly_plot = tls.mpl_to_plotly(fig)
            plotly_plot['data'][0]['hovertext'] = \
                ["%.3f" % self.vasp_run.eigenvalues[Spin.up][k_index][i-2][0] for i in x]
            plotly_plot['data'][1]['hovertext'] = \
                ["%.3f" % self.vasp_run.eigenvalues[Spin.down][k_index][i-2][0] for i in x]
            plotly.offline.plot(plotly_plot, filename=os.path.join(self.path, "procar.html"))

        print_total()
        return total_up, total_down


class DetermineDefectState:
    def __init__(self, db, db_filter, cbm, vbm, save_fig_path, locpot=None, locpot_c2db=None):
        """
        locpot or locpot_c2db (3D tuple): (db_object, task_id/uid, displacement)
        """

        self.save_fig_path = save_fig_path

        self.entry = db.collection.find_one(db_filter)

        print("---task_id: %s---" % self.entry["task_id"])

        if locpot_c2db:
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.entry_host = locpot_c2db[0].collection.find_one({"uid": locpot_c2db[1]})
            self.vacuum_locpot_host = self.entry_host["evac"]
            self.cbm = self.entry_host["cbm_hse_nosoc"] + locpot_c2db[2]
            self.vbm = self.entry_host["vbm_hse_nosoc"] + locpot_c2db[2]
            efermi_to_defect_vac = self.entry["calcs_reversed"][0]["output"]["efermi"] - self.vacuum_locpot + locpot_c2db[2]
            self.efermi = efermi_to_defect_vac
            
        elif locpot:
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            db_host = locpot[0]
            if not locpot[1]:
                self.entry_host = db_host.collection.find_one({"task_id": int(self.entry["pc_from"].split("/")[-1])})
            else:
                self.entry_host = db_host.collection.find_one({"task_id": locpot[1]})

            self.vacuum_locpot_host = max(self.entry_host["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.cbm = self.entry_host["output"]["cbm"] - self.vacuum_locpot_host
            self.vbm = self.entry_host["output"]["vbm"] - self.vacuum_locpot_host
            self.search_range = (self.vbm + locpot[3] + locpot[2], self.cbm + locpot[4] + locpot[2])
      
            efermi_to_defect_vac = self.entry["calcs_reversed"][0]["output"]["efermi"]
            self.efermi = efermi_to_defect_vac
        
        else:
            print("No vacuum alignment!")
            self.cbm = cbm
            self.vbm = vbm
            self.efermi = self.entry["calcs_reversed"][0]["output"]["efermi"]


        # if locpot and self.entry["vacuum_locpot"]:
        #     self.vacuum_locpot = self.entry["vacuum_locpot"]["value"]
        # else:
        #     self.vacuum_locpot = 0

        # if locpot and self.entry["info_primitive"]["vacuum_locpot_primitive"]:
        #     self.vacuum_locpot_primitive = self.entry["info_primitive"]["vacuum_locpot_primitive"]["value"]
        # else:
        #     self.vacuum_locpot_primitive = 0

        self.nn = self.entry["NN"]
        self.proj_eigenvals = db.get_proj_eigenvals(self.entry["task_id"])
        self.eigenvals = db.get_eigenvals(self.entry["task_id"])

        print("total_mag:{:.3f}".format(self.entry["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]))
        print("cbm:{:.3f}, vbm:{:.3f}, efermi:{:.3f}".format(self.cbm,
                                                            self.vbm,
                                                            self.efermi,
                                                            ))

    def get_candidates(self, kpoint=0, threshold=0.2, select_bands=None):
        def get_eigenvals(spin, entry_eigenvals, energy_range):
            eigenvals = defaultdict(list)
            for band_idx, i in enumerate(entry_eigenvals[spin][kpoint]):
                # (band_index, [energy, occupied])
                if energy_range[1] > (i[0] - self.vacuum_locpot) > energy_range[0]:
                    eigenvals[spin].append((band_idx, i))
            return eigenvals
        def get_promising_state(spin, eigenvals):
            promising_band = defaultdict(list)
            band_up_proj = []
            up = defaultdict(tuple)
            if eigenvals.get(spin, None):
                for band in eigenvals[spin]:
                    total_proj = 0
                    adj_proj = 0
                    for ion_idx in self.nn:
                        total_proj += sum(self.proj_eigenvals[spin][kpoint][band[0]][ion_idx])
                    for ion_adj_idx in self.nn[:-1]:
                        adj_proj += sum(self.proj_eigenvals[spin][kpoint][band[0]][ion_adj_idx])
                    antisite_proj = sum(self.proj_eigenvals[spin][kpoint][band[0]][self.nn[-1]])
                    if total_proj >= threshold:
                        # print("band_index: {}".format(band[0]))
                        promising_band[spin].append((band[0], band[1], total_proj,
                                                    round(adj_proj/total_proj*100,2),
                                                    round(antisite_proj/total_proj*100,2)))


                        sheet_procar = defaultdict(list)
                        for idx, o in enumerate(['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']):
                            test = {}
                            for ion_idx in self.nn:
                                orbital_proj = self.proj_eigenvals[spin][kpoint][band[0]][ion_idx][idx]
                                if orbital_proj < 1e-3:
                                    orbital_proj = None
                                test.update({ion_idx: orbital_proj})
                            test.update({"band_index": band[0]}) #
                            test.update({"spin":spin}) #
                            test.update({"orbital":o})
                            band_up_proj.append(test)
                for i in promising_band[spin]:
                    up[i[0]] = (round(i[1][0], 3), (i[1][1] > 0.4, i[1][1], i[2], i[3], i[4]))
            return up, band_up_proj

        def sheet(band_info, band_proj, spin):
            sheet_up = defaultdict(list)
            sheet_up["band_index"] = list(band_info.keys())
            sheet_up["energy"] = [i[0] for i in band_info.values()]
            sheet_up["occupied"] = [i[1][0] for i in band_info.values()]
            sheet_up["tot_proj"] = [i[1][2] for i in band_info.values()]
            sheet_up["adj_proj"] = [i[1][3] for i in band_info.values()]
            sheet_up["antisite_proj"] = [i[1][4] for i in band_info.values()]
            sheet_up["n_occ_e"] = [i[1][1] for i in band_info.values()]
            sheet_up["spin"] = [spin for i in range(len(band_info.keys()))]
            df_up, df_up_proj = pd.DataFrame(sheet_up), pd.DataFrame(band_proj[spin])
            if len(sheet_up["band_index"]) != 0:
                df_up = df_up.set_index(["band_index"])
                df_up_proj = df_up_proj.set_index(["band_index"])
            return df_up, df_up_proj

        # find eigenstates with (band_index, [energy, occupation])
        spins = ["1", "-1"]
        if self.entry["input"]["incar"].get("ISPIN", None) == 1:
            spins = ["1"]
        elif self.entry["input"]["incar"].get("LSORBIT", None):
            spins = ["1"]

        eigenvals = {}
        promising_band = {}
        band_proj = {}
        df = {}
        df_proj = {}
        channel = {}
        levels = {}
        for spin in spins:
            band_info = {}
            try:
                eigenvals.update(get_eigenvals(spin, self.eigenvals, energy_range=self.search_range))
                band_detail, proj = get_promising_state(spin, eigenvals)
                band_proj[spin] = proj
                band_info.update(band_detail)
                df[spin], df_proj[spin] = sheet(band_info, band_proj, spin)
                if select_bands:
                    for spin, band in select_bands.items():
                        channel[spin] = df[spin].loc[band, :].sort_index(ascending=False)
                        band_proj[spin] = df_proj[spin].loc[band, :].sort_index(ascending=False)
                        levels[spin] = dict(zip(channel[spin].loc[band:, "energy"], channel[spin].loc[band:, "occupied"]))
                else:
                    channel[spin] = df[spin].sort_index(ascending=False)
                    band_proj[spin] = df_proj[spin].sort_index(ascending=False)
                    levels[spin] = dict(zip(channel[spin].loc[:, "energy"], channel[spin].loc[:, "occupied"]))
            except IndexError:
                print("Threshold of projection is too high!")

        # print(levels)

        state_df = pd.concat(list(channel.values()), ignore_index=False)
        proj_state_df = pd.concat(band_proj.values(), ignore_index=False)
        proj_state_df = proj_state_df.fillna(0)
        proj_state_df["adjacent"] = proj_state_df.iloc[:,0] + proj_state_df.iloc[:,1] + proj_state_df.iloc[:, 2]
        proj_state_df["antisite"] = proj_state_df.iloc[:,3]

        # proj_state_df.loc[proj_state_df["adjacent"] < 0.01, "adjacent"] = 0
        # proj_state_df.loc[proj_state_df["antisite"] < 0.01, "antisite"] = 0

        proj_state_df = proj_state_df.sort_values(["spin", "band_index", "orbital"], ascending=False)
        print("=="*20)
        print(proj_state_df)
        print("=="*20)
        print(state_df)

        # depict defate states


        ## up
        up_states = state_df.loc[state_df["spin"] == "1"]
        print("**"*20)
        print(up_states)
        up_dist_from_vbm = up_states["energy"] - self.vbm - self.vacuum_locpot #(self.vbm + self.vacuum_locpot)
        up_dist_from_vbm = up_dist_from_vbm.round(3)
        up_occ = up_states.loc[:, "n_occ_e"]
        up_band_index = up_states.index
        ## dn
        dn_states = state_df.loc[state_df["spin"] == "-1"]
        dn_dist_from_vbm = dn_states["energy"] - self.vbm - self.vacuum_locpot #(self.vbm + self.vacuum_locpot)
        dn_dist_from_vbm = dn_dist_from_vbm.round(3)
        dn_occ = dn_states.loc[:, "n_occ_e"]
        dn_band_index = dn_states.index


        d_df = pd.DataFrame(
            [
                {
                    "up_from_vbm": up_dist_from_vbm.to_list(),
                    "up_occ": up_occ.to_list(),
                    "dn_from_vbm": dn_dist_from_vbm.to_list(),
                    "dn_occ": dn_occ.to_list(),
                    "up_band_idx": up_band_index.to_list(),
                    "dn_band_idx": dn_band_index.to_list()
                }
            ]
        )

        # highest_level_from_cbm = (self.cbm + self.vacuum_locpot) - highest_level
        # lowest_level_from_vbm = lowest_level - (self.vbm + self.vacuum_locpot)
        # highest_occupied_level_from_cbm = (self.cbm + self.vacuum_locpot) - highest_occupied_level
        # highest_occupied_level_from_vbm = highest_occupied_level - (self.vbm + self.vacuum_locpot)
        # d_df = pd.DataFrame([{"highest_from_cbm": highest_level_from_cbm,
        #                       "lowest_from_vbm": lowest_level_from_vbm,
        #                       "occ_level_from_cbm": highest_occupied_level_from_cbm,
        #                       "occ_level_from_vbm": highest_occupied_level_from_vbm,
        #                       }]).round(3)

        return state_df, proj_state_df, d_df

    def get_degeneracy(self, threshold, energy_tolerence, kpoint=0):
        eigenvals = {"1": [], "-1": []}
        for i in self.eigenvals["1"][kpoint]:
            eigenvals["1"].append((self.eigenvals["1"][kpoint].index(i), i))

        for i in self.eigenvals["-1"][kpoint]:
            eigenvals["-1"].append((self.eigenvals["-1"][kpoint].index(i), i))

        # find promising states if total projection of nn is larger than 0.2
        promising_band = {"1": [], "-1": []}
        for band in eigenvals["1"]:
            total_proj = 0
            for ion_idx in self.nn:
                total_proj += sum(self.proj_eigenvals["1"][kpoint][band[0]][ion_idx])
            if total_proj >= threshold:
                promising_band["1"].append((band[0], band[1], total_proj))

        for band in eigenvals["-1"]:
            total_proj = 0
            for ion_idx in self.nn:
                total_proj += sum(self.proj_eigenvals["-1"][kpoint][band[0]][ion_idx])
            if total_proj >= threshold:
                promising_band["-1"].append((band[0], band[1], total_proj))

        candidate_up_band = np.array([[i[0], i[1][0]] for i in promising_band["1"]])
        diff_can_up_band = np.diff(candidate_up_band[:, 1])/np.average(candidate_up_band)
        index = np.where(diff_can_up_band <= np.min(diff_can_up_band)*energy_tolerence[0])
        for i in index[0]: print("up: %s" % candidate_up_band[i:i+2, :])

        candidate_dn_band = np.array([[i[0], i[1][0]] for i in promising_band["-1"]])
        diff_can_dn_band = np.diff(candidate_dn_band[:, 1])/np.average(candidate_dn_band)
        index = np.where(diff_can_dn_band <= np.min(diff_can_dn_band)*energy_tolerence[1])
        for i in index[0]: print("down: %s" % candidate_dn_band[i:i+2, :])


class NewDetermineDefectState:
    def __init__(self, db, db_filter, cbm, vbm, save_fig_path, locpot=None, locpot_c2db=None):
        """
        locpot or locpot_c2db (3D tuple): (db_object, task_id/uid, displacement)
        """

        self.save_fig_path = save_fig_path

        self.entry = db.collection.find_one(db_filter)

        print("---task_id: %s---" % self.entry["task_id"])

        if locpot_c2db:
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.entry_host = locpot_c2db[0].collection.find_one({"uid": locpot_c2db[1]})
            self.vacuum_locpot_host = self.entry_host["evac"]
            self.cbm = self.entry_host["cbm_hse_nosoc"] + locpot_c2db[2]
            self.vbm = self.entry_host["vbm_hse_nosoc"] + locpot_c2db[2]
            efermi_to_defect_vac = self.entry["calcs_reversed"][0]["output"]["efermi"] - self.vacuum_locpot + locpot_c2db[2]
            self.efermi = efermi_to_defect_vac

        elif locpot:
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            db_host = locpot[0]
            if not locpot[1]:
                self.entry_host = db_host.collection.find_one({"task_id": int(self.entry["pc_from"].split("/")[-1])})
            else:
                self.entry_host = db_host.collection.find_one({"task_id": locpot[1]})
            self.vacuum_locpot_host = max(self.entry_host["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.cbm = self.entry_host["output"]["cbm"] - self.vacuum_locpot_host + locpot[4] + locpot[2]
            self.vbm = self.entry_host["output"]["vbm"] - self.vacuum_locpot_host + locpot[3] + locpot[2]

            efermi_to_defect_vac = self.entry["calcs_reversed"][0]["output"]["efermi"] - self.vacuum_locpot_host
            self.efermi = efermi_to_defect_vac

        else:
            print("No vacuum alignment!")
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.cbm = cbm
            self.vbm = vbm
            self.efermi = self.entry["calcs_reversed"][0]["output"]["efermi"]


        # if locpot and self.entry["vacuum_locpot"]:
        #     self.vacuum_locpot = self.entry["vacuum_locpot"]["value"]
        # else:
        #     self.vacuum_locpot = 0

        # if locpot and self.entry["info_primitive"]["vacuum_locpot_primitive"]:
        #     self.vacuum_locpot_primitive = self.entry["info_primitive"]["vacuum_locpot_primitive"]["value"]
        # else:
        #     self.vacuum_locpot_primitive = 0

        self.nn = self.entry["NN"]
        self.proj_eigenvals = db.get_proj_eigenvals(self.entry["task_id"])
        self.eigenvals = db.get_eigenvals(self.entry["task_id"])

        print("total_mag:{:.3f}".format(self.entry["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]))
        print("cbm:{:.3f}, vbm:{:.3f}, efermi:{:.3f}".format(self.cbm,
                                                             self.vbm,
                                                             self.efermi,
                                                             ))

    def get_eigenvals(self, spin, entry_eigenvals, energy_range, kpoint=0):
        eigenvals = defaultdict(list)
        for band_idx, i in enumerate(entry_eigenvals[spin][kpoint]):
            # (band_index, [energy, occupied])
            if energy_range[1] > i[0] > energy_range[0]:
                eigenvals[spin].append((band_idx, i))
        return eigenvals

    def get_promising_state(self, spin, eigenvals, kpoint=0, threshold=0.1):
            promising_band = defaultdict(list)
            bulk_band = defaultdict(list)
            band_up_proj = []
            bulk_up_proj = []
            up = defaultdict(tuple)
            bulk_up = defaultdict(tuple)

            if eigenvals.get(spin, None):
                for band in eigenvals[spin]:
                    total_proj = 0
                    adj_proj = 0
                    for ion_idx in self.nn:
                        total_proj += sum(self.proj_eigenvals[spin][kpoint][band[0]][ion_idx])
                    for ion_adj_idx in self.nn[:-1]:
                        adj_proj += sum(self.proj_eigenvals[spin][kpoint][band[0]][ion_adj_idx])
                    antisite_proj = sum(self.proj_eigenvals[spin][kpoint][band[0]][self.nn[-1]])
                    if total_proj >= threshold:
                        # print("band_index: {}".format(band[0]))
                        promising_band[spin].append((band[0], band[1], total_proj,
                                                     round(adj_proj/total_proj*100,2),
                                                     round(antisite_proj/total_proj*100,2)))

                        sheet_procar = defaultdict(list)
                        for idx, o in enumerate(['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']):
                            test = {}
                            for ion_idx in self.nn:
                                orbital_proj = self.proj_eigenvals[spin][kpoint][band[0]][ion_idx][idx]
                                if orbital_proj < 1e-3:
                                    orbital_proj = None
                                test.update({ion_idx: orbital_proj})
                            test.update({"band_index": band[0]}) #
                            test.update({"spin":spin}) #
                            test.update({"orbital":o})
                            band_up_proj.append(test)

                    else:
                        bulk_band[spin].append((band[0], band[1], total_proj, 0, 0))
                                                # round(adj_proj/total_proj*100,2),
                                                # round(antisite_proj/total_proj*100,2)))
                        
                        for idx, o in enumerate(['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']):
                            test = {}
                            for ion_idx in self.nn:
                                orbital_proj = self.proj_eigenvals[spin][kpoint][band[0]][ion_idx][idx]
                                if orbital_proj < 1e-3:
                                    orbital_proj = None
                                test.update({ion_idx: orbital_proj})
                            test.update({"band_index": band[0]}) #
                            test.update({"spin":spin}) #
                            test.update({"orbital":o})
                            bulk_up_proj.append(test)
                            
                for i in promising_band[spin]:
                    up[i[0]] = (round(i[1][0], 5), (i[1][1] > 0.4, i[1][1], i[2], i[3], i[4]))
                    
                for i in bulk_band[spin]:
                    bulk_up[i[0]] = (round(i[1][0], 5), (i[1][1] > 0.4, i[1][1], i[2], i[3], i[4]))
                    
            return up, band_up_proj, bulk_up, bulk_up_proj

    def sheet(self, band_info, band_proj, spin):
        sheet_up = defaultdict(list)
        sheet_up["band_index"] = list(band_info.keys())
        sheet_up["energy"] = [i[0] for i in band_info.values()]
        sheet_up["occupied"] = [i[1][0] for i in band_info.values()]
        sheet_up["tot_proj"] = [i[1][2] for i in band_info.values()]
        sheet_up["adj_proj"] = [i[1][3] for i in band_info.values()]
        sheet_up["antisite_proj"] = [i[1][4] for i in band_info.values()]
        sheet_up["n_occ_e"] = [i[1][1] for i in band_info.values()]
        sheet_up["spin"] = [spin for i in range(len(band_info.keys()))]
        df_up, df_up_proj = pd.DataFrame(sheet_up), pd.DataFrame(band_proj[spin])
        if len(sheet_up["band_index"]) != 0:
            df_up = df_up.set_index(["band_index"])
            df_up_proj = df_up_proj.set_index(["band_index"])
        return df_up, df_up_proj
    
    def get_candidates(self, kpoint=0, threshold=0.2, select_bands=None):
        # find eigenstates with (band_index, [energy, occupation])
        spins = ["1", "-1"]
        if self.entry["input"]["incar"].get("ISPIN", None) == 1:
            spins = ["1"]
        elif self.entry["input"]["incar"].get("LSORBIT", None):
            spins = ["1"]

        eigenvals = {}
        
        band_proj = {}
        bulk_band_proj = {}
        
        df = {}
        df_proj = {}
        bulk_df = {}
        bulk_df_proj = {}
        
        channel = {}
        bulk_channel = {}
        
        levels = {}
        bulk_levels = {}
        for spin in spins:
            band_info = {}
            bulk_band_info = {}
            try:
                eigenvals.update(self.get_eigenvals(spin, self.eigenvals, energy_range=[self.vbm, self.cbm], kpoint=kpoint))
                band_detail, proj, bulk_band_detail, bulk_proj = self.get_promising_state(spin, eigenvals, kpoint, 
                                                                                      threshold)
                band_proj[spin] = proj
                bulk_band_proj[spin] = bulk_proj
                
                band_info.update(band_detail)
                bulk_band_info.update(bulk_band_detail)
                
                df[spin], df_proj[spin] = self.sheet(band_info, band_proj, spin)
                bulk_df[spin], bulk_df_proj[spin] = self.sheet(bulk_band_info, bulk_band_proj, spin)

                if select_bands:
                    for spin, band in select_bands.items():
                        channel[spin] = df[spin].loc[band, :].sort_index(ascending=False)
                        band_proj[spin] = df_proj[spin].loc[band, :].sort_index(ascending=False)
                        levels[spin] = dict(zip(channel[spin].loc[band:, "energy"], channel[spin].loc[band:, "occupied"]))
                else:
                    channel[spin] = df[spin].sort_index(ascending=False)
                    bulk_channel[spin] = bulk_df[spin].sort_index(ascending=False)

                    band_proj[spin] = df_proj[spin].sort_index(ascending=False)
                    bulk_band_proj[spin] = bulk_df_proj[spin].sort_index(ascending=False)

                    levels[spin] = dict(zip(channel[spin].loc[:, "energy"], channel[spin].loc[:, "occupied"]))
                    bulk_levels[spin] = dict(zip(bulk_channel[spin].loc[:, "energy"], bulk_channel[spin].loc[:, "occupied"]))

            except IndexError:
                print("Threshold of projection is too high!")

        print("defect_levels: {}".format(levels))
        print("bulk_levels: {}".format(bulk_levels))

        state_df = pd.concat(list(channel.values()), ignore_index=False)
        bulk_state_df = pd.concat(list(bulk_channel.values()), ignore_index=False)
        self.cbm = bulk_state_df.loc[bulk_state_df["n_occ_e"] == 0, "energy"].min() 
        self.vbm = bulk_state_df.loc[bulk_state_df["n_occ_e"] == 1, "energy"].max()
        print("perturbed band edges: (VBM, CBM): ({}, {})".format(self.vbm, self.cbm))
        
        proj_state_df = pd.concat(band_proj.values(), ignore_index=False)
        proj_state_df = proj_state_df.fillna(0)
        bulk_proj_state_df = pd.concat(bulk_band_proj.values(), ignore_index=False)
        bulk_proj_state_df = bulk_proj_state_df.fillna(0)        
  
        proj_state_df["adjacent"] = proj_state_df.iloc[:,0] + proj_state_df.iloc[:,1] + proj_state_df.iloc[:, 2]
        proj_state_df["antisite"] = proj_state_df.iloc[:,3]

        bulk_proj_state_df["adjacent"] = bulk_proj_state_df.iloc[:,0] + bulk_proj_state_df.iloc[:, 
                                                                        1] + bulk_proj_state_df.iloc[:, 2]
        bulk_proj_state_df["antisite"] = bulk_proj_state_df.iloc[:,3]

        # proj_state_df.loc[proj_state_df["adjacent"] < 0.01, "adjacent"] = 0
        # proj_state_df.loc[proj_state_df["antisite"] < 0.01, "antisite"] = 0

        proj_state_df = proj_state_df.sort_values(["spin", "band_index", "orbital"], ascending=False)
        bulk_proj_state_df = bulk_proj_state_df.sort_values(["spin", "band_index", "orbital"], ascending=False)

        print("D=="*20)
        print(proj_state_df)
        print("D=="*20)
        print(state_df)

        print("B=="*20)
        print(bulk_proj_state_df)
        print("B=="*20)
        print(bulk_state_df)

        # depict defate states


        ## up
        up_states = state_df.loc[(state_df["spin"] == "1") & 
                                 (state_df["energy"] > self.vbm) &
                                 (state_df["energy"] < self.cbm)]
        print("D**up_states"*20)
        print(up_states)
        up_dist_from_vbm = up_states["energy"] - self.vbm #(self.vbm + self.vacuum_locpot)
        up_dist_from_vbm = up_dist_from_vbm.round(3)
        up_occ = up_states.loc[:, "n_occ_e"]
        up_band_index = up_states.index

        bulk_up_states = bulk_state_df.loc[bulk_state_df["spin"] == "1"]
        print("B**up_states"*20)
        print(bulk_up_states)
        bulk_up_dist_from_vbm = bulk_up_states["energy"] - self.vbm #(self.vbm + 
        # self.vacuum_locpot)
        bulk_up_dist_from_vbm = bulk_up_dist_from_vbm.round(3)
        bulk_up_occ = bulk_up_states.loc[:, "n_occ_e"]
        bulk_up_band_index = bulk_up_states.index        

        
        ## dn
        dn_states = state_df.loc[(state_df["spin"] == "-1") &
                                 (state_df["energy"] > self.vbm) &
                                 (state_df["energy"] < self.cbm)]
        print("D**dn_states"*20)
        print(dn_states)
        dn_dist_from_vbm = dn_states["energy"] - self.vbm #(self.vbm + self.vacuum_locpot)
        dn_dist_from_vbm = dn_dist_from_vbm.round(3)
        dn_occ = dn_states.loc[:, "n_occ_e"]
        dn_band_index = dn_states.index

        bulk_dn_states = bulk_state_df.loc[bulk_state_df["spin"] == "-1"]
        print("B**db_states"*20)
        print(bulk_dn_states)
        bulk_dn_dist_from_vbm = bulk_dn_states["energy"] - self.vbm #(self.vbm + 
        # self.vacuum_locpot)
        bulk_dn_dist_from_vbm = bulk_dn_dist_from_vbm.round(3)
        bulk_dn_occ = bulk_dn_states.loc[:, "n_occ_e"]
        bulk_dn_band_index = bulk_dn_states.index

        d_df = pd.DataFrame(
            [
                {
                    "up_from_vbm": up_dist_from_vbm.to_list(),
                    "up_occ": up_occ.to_list(),
                    "dn_from_vbm": dn_dist_from_vbm.to_list(),
                    "dn_occ": dn_occ.to_list(),
                    "up_band_idx": up_band_index.to_list(),
                    "dn_band_idx": dn_band_index.to_list()
                }
            ]
        )

        bulk_d_df = pd.DataFrame(
            [
                {
                    "up_from_vbm": bulk_up_dist_from_vbm.to_list(),
                    "up_occ": bulk_up_occ.to_list(),
                    "dn_from_vbm": bulk_dn_dist_from_vbm.to_list(),
                    "dn_occ": bulk_dn_occ.to_list(),
                    "up_band_idx": bulk_up_band_index.to_list(),
                    "dn_band_idx": bulk_dn_band_index.to_list()
                }
            ]
        )

        return state_df, proj_state_df, d_df, bulk_state_df, bulk_proj_state_df, bulk_d_df

class DetermineDefectStateV3:
    def __init__(self, db, db_filter, cbm, vbm, save_fig_path, locpot=None, locpot_c2db=None):
        """
        locpot or locpot_c2db (3D tuple): (db_object, task_id/uid, displacement)
        """

        self.save_fig_path = save_fig_path

        self.entry = db.collection.find_one(db_filter)

        print("---task_id: %s---" % self.entry["task_id"])

        if locpot_c2db:
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.entry_host = locpot_c2db[0].collection.find_one({"uid": locpot_c2db[1]})
            self.vacuum_locpot_host = self.entry_host["evac"]
            self.cbm = self.entry_host["cbm_hse_nosoc"] + locpot_c2db[2]
            self.vbm = self.entry_host["vbm_hse_nosoc"] + locpot_c2db[2]
            efermi_to_defect_vac = self.entry["calcs_reversed"][0]["output"]["efermi"] - self.vacuum_locpot + locpot_c2db[2]
            self.efermi = efermi_to_defect_vac

        elif locpot:
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            db_host = locpot[0]
            if not locpot[1]:
                self.entry_host = db_host.collection.find_one({"task_id": int(self.entry["pc_from"].split("/")[-1])})
            else:
                self.entry_host = db_host.collection.find_one({"task_id": locpot[1]})
            self.vacuum_locpot_host = max(self.entry_host["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.cbm = self.entry_host["output"]["cbm"] - self.vacuum_locpot_host + locpot[4] + locpot[2]
            self.vbm = self.entry_host["output"]["vbm"] - self.vacuum_locpot_host + locpot[3] + locpot[2]

            efermi_to_defect_vac = self.entry["calcs_reversed"][0]["output"]["efermi"] - self.vacuum_locpot_host
            self.efermi = efermi_to_defect_vac

        else:
            print("No vacuum alignment!")
            self.vacuum_locpot = max(self.entry["calcs_reversed"][0]["output"]["locpot"]["2"])
            self.cbm = cbm
            self.vbm = vbm
            self.efermi = self.entry["calcs_reversed"][0]["output"]["efermi"]


        # if locpot and self.entry["vacuum_locpot"]:
        #     self.vacuum_locpot = self.entry["vacuum_locpot"]["value"]
        # else:
        #     self.vacuum_locpot = 0

        # if locpot and self.entry["info_primitive"]["vacuum_locpot_primitive"]:
        #     self.vacuum_locpot_primitive = self.entry["info_primitive"]["vacuum_locpot_primitive"]["value"]
        # else:
        #     self.vacuum_locpot_primitive = 0

        self.nn = self.entry["NN"]
        self.proj_eigenvals = db.get_proj_eigenvals(self.entry["task_id"])
        self.eigenvals = db.get_eigenvals(self.entry["task_id"])

        print("total_mag:{:.3f}".format(self.entry["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]))
        print("cbm:{:.3f}, vbm:{:.3f}, efermi:{:.3f}".format(self.cbm,
                                                             self.vbm,
                                                             self.efermi,
                                                             ))

    def get_eigenvals(self, spin, entry_eigenvals, energy_range, kpoint=0):
        eigenvals = defaultdict(list)
        for band_idx, i in enumerate(entry_eigenvals[spin][kpoint]):
            # (band_index, [energy, occupied])
            if energy_range[1] > i[0] > energy_range[0]:
                eigenvals[spin].append((band_idx, i))
        return eigenvals

    def get_promising_state(self, spin, eigenvals, kpoint=0, threshold=0.1):
        promising_band = defaultdict(list)
        bulk_band = defaultdict(list)
        band_up_proj = []
        bulk_up_proj = []
        up = defaultdict(tuple)
        bulk_up = defaultdict(tuple)

        if eigenvals.get(spin, None):
            for band in eigenvals[spin]:
                total_proj = 0
                adj_proj = 0
                for ion_idx in self.nn:
                    total_proj += sum(self.proj_eigenvals[spin][kpoint][band[0]][ion_idx])
                for ion_adj_idx in self.nn[:-1]:
                    adj_proj += sum(self.proj_eigenvals[spin][kpoint][band[0]][ion_adj_idx])
                antisite_proj = sum(self.proj_eigenvals[spin][kpoint][band[0]][self.nn[-1]])
                if total_proj >= threshold:
                    # print("band_index: {}".format(band[0]))
                    promising_band[spin].append((band[0], band[1], total_proj,
                                                 round(adj_proj/total_proj*100,2),
                                                 round(antisite_proj/total_proj*100,2)))

                    sheet_procar = defaultdict(list)
                    for idx, o in enumerate(['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']):
                        test = {}
                        for ion_idx in self.nn:
                            orbital_proj = self.proj_eigenvals[spin][kpoint][band[0]][ion_idx][idx]
                            if orbital_proj < 1e-3:
                                orbital_proj = None
                            test.update({ion_idx: orbital_proj})
                        test.update({"band_index": band[0]}) #
                        test.update({"spin":spin}) #
                        test.update({"orbital":o})
                        band_up_proj.append(test)

                else:
                    bulk_band[spin].append((band[0], band[1], total_proj, 0, 0))
                    # round(adj_proj/total_proj*100,2),
                    # round(antisite_proj/total_proj*100,2)))

                    for idx, o in enumerate(['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']):
                        test = {}
                        for ion_idx in self.nn:
                            orbital_proj = self.proj_eigenvals[spin][kpoint][band[0]][ion_idx][idx]
                            if orbital_proj < 1e-3:
                                orbital_proj = None
                            test.update({ion_idx: orbital_proj})
                        test.update({"band_index": band[0]}) #
                        test.update({"spin":spin}) #
                        test.update({"orbital":o})
                        bulk_up_proj.append(test)

            for i in promising_band[spin]:
                up[i[0]] = (round(i[1][0], 5), (i[1][1] > 0.4, i[1][1], i[2], i[3], i[4]))

            for i in bulk_band[spin]:
                bulk_up[i[0]] = (round(i[1][0], 5), (i[1][1] > 0.4, i[1][1], i[2], i[3], i[4]))

        return up, band_up_proj, bulk_up, bulk_up_proj

    def sheet(self, band_info, band_proj, spin):
        sheet_up = defaultdict(list)
        sheet_up["band_index"] = list(band_info.keys())
        sheet_up["energy"] = [i[0] for i in band_info.values()]
        sheet_up["occupied"] = [i[1][0] for i in band_info.values()]
        sheet_up["tot_proj"] = [i[1][2] for i in band_info.values()]
        sheet_up["adj_proj"] = [i[1][3] for i in band_info.values()]
        sheet_up["antisite_proj"] = [i[1][4] for i in band_info.values()]
        sheet_up["n_occ_e"] = [round(i[1][1], 2) for i in band_info.values()]
        sheet_up["spin"] = [spin for i in range(len(band_info.keys()))]
        df_up, df_up_proj = pd.DataFrame(sheet_up), pd.DataFrame(band_proj[spin])
        if len(sheet_up["band_index"]) != 0:
            df_up = df_up.set_index(["band_index"])
            df_up_proj = df_up_proj.set_index(["band_index"])
        return df_up, df_up_proj

    def get_candidates(self, kpoint=0, threshold=0.2, select_bands=None):
        # find eigenstates with (band_index, [energy, occupation])
        spins = ["1", "-1"]
        if self.entry["input"]["incar"].get("ISPIN", None) == 1:
            spins = ["1"]
        elif self.entry["input"]["incar"].get("LSORBIT", None):
            spins = ["1"]

        eigenvals = {}

        band_proj = {}
        bulk_band_proj = {}

        df = {}
        df_proj = {}
        bulk_df = {}
        bulk_df_proj = {}

        channel = {}
        bulk_channel = {}

        levels = {}
        bulk_levels = {}
        for spin in spins:
            band_info = {}
            bulk_band_info = {}
            try:
                eigenvals.update(self.get_eigenvals(spin, self.eigenvals, energy_range=[self.vbm, self.cbm], kpoint=kpoint))
                band_detail, proj, bulk_band_detail, bulk_proj = self.get_promising_state(spin, eigenvals, kpoint,
                                                                                          threshold)
                band_proj[spin] = proj
                bulk_band_proj[spin] = bulk_proj

                band_info.update(band_detail)
                bulk_band_info.update(bulk_band_detail)

                df[spin], df_proj[spin] = self.sheet(band_info, band_proj, spin)
                bulk_df[spin], bulk_df_proj[spin] = self.sheet(bulk_band_info, bulk_band_proj, spin)

                if select_bands:
                    for spin, band in select_bands.items():
                        channel[spin] = df[spin].loc[band, :].sort_index(ascending=False)
                        band_proj[spin] = df_proj[spin].loc[band, :].sort_index(ascending=False)
                        levels[spin] = dict(zip(channel[spin].loc[band:, "energy"], channel[spin].loc[band:, "occupied"]))
                else:
                    channel[spin] = df[spin].sort_index(ascending=False)
                    bulk_channel[spin] = bulk_df[spin].sort_index(ascending=False)

                    band_proj[spin] = df_proj[spin].sort_index(ascending=False)
                    bulk_band_proj[spin] = bulk_df_proj[spin].sort_index(ascending=False)

                    levels[spin] = dict(zip(channel[spin].loc[:, "energy"], channel[spin].loc[:, "occupied"]))
                    bulk_levels[spin] = dict(zip(bulk_channel[spin].loc[:, "energy"], bulk_channel[spin].loc[:, "occupied"]))

            except IndexError:
                print("Threshold of projection is too high!")

        print("defect_levels: {}".format(levels))
        print("bulk_levels: {}".format(bulk_levels))

        state_df = pd.concat(list(channel.values()), ignore_index=False)
        bulk_state_df = pd.concat(list(bulk_channel.values()), ignore_index=False)
        self.cbm = bulk_state_df.loc[bulk_state_df["n_occ_e"] == 0, "energy"].min()
        self.vbm = bulk_state_df.loc[bulk_state_df["n_occ_e"] == 1, "energy"].max()
        print("perturbed band edges: (VBM, CBM): ({}, {})".format(self.vbm, self.cbm))

        proj_state_df = pd.concat(band_proj.values(), ignore_index=False)
        proj_state_df = proj_state_df.fillna(0)
        bulk_proj_state_df = pd.concat(bulk_band_proj.values(), ignore_index=False)
        bulk_proj_state_df = bulk_proj_state_df.fillna(0)

        proj_state_df["adjacent"] = proj_state_df.iloc[:,0] + proj_state_df.iloc[:,1] + proj_state_df.iloc[:, 2]
        proj_state_df["antisite"] = proj_state_df.iloc[:,3]

        bulk_proj_state_df["adjacent"] = bulk_proj_state_df.iloc[:,0] + bulk_proj_state_df.iloc[:,
                                                                        1] + bulk_proj_state_df.iloc[:, 2]
        bulk_proj_state_df["antisite"] = bulk_proj_state_df.iloc[:,3]

        # proj_state_df.loc[proj_state_df["adjacent"] < 0.01, "adjacent"] = 0
        # proj_state_df.loc[proj_state_df["antisite"] < 0.01, "antisite"] = 0

        proj_state_df = proj_state_df.sort_values(["spin", "band_index", "orbital"], ascending=False)
        bulk_proj_state_df = bulk_proj_state_df.sort_values(["spin", "band_index", "orbital"], ascending=False)

        print("D=="*20)
        print(proj_state_df)
        print("D=="*20)
        print(state_df)

        print("B=="*20)
        print(bulk_proj_state_df)
        print("B=="*20)
        print(bulk_state_df)

        # depict defate states
        dist_from_vbm = state_df["energy"] - self.vbm
        state_df["dist_from_vbm"] = dist_from_vbm.round(3)
        dist_from_cbm = state_df["energy"] - self.cbm
        state_df["dist_from_cbm"] = dist_from_cbm.round(3)
        
        state_df["band_index"] = state_df.index

        return state_df, proj_state_df, bulk_state_df, bulk_proj_state_df


if __name__ == '__main__':
    pass

