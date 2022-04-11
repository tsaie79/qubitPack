from qubitPack.qc_searching.analysis.dos_plot_from_db import DosPlotDB
from qubitPack.qc_searching.analysis.read_eigen import DetermineDefectState, NewDetermineDefectState,  \
    DetermineDefectStateV3
from qubitPack.qc_searching.py_energy_diagram.application.defect_levels import EnergyLevel
from qubitPack.qc_searching.analysis.dos_plot_from_db import DB_CONFIG_PATH

import matplotlib.pyplot as plt
from glob import glob
import os
import pandas as pd
import numpy as np
from collections import Counter


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def get_eigen_plot_v1(tot, determine_defect_state_obj, top_texts, is_vacuum_aligment=False):
    levels = {"1": {}, "-1":{}}
    vbm, cbm = None, None
    for spin in ["1", "-1"]:
        energy = tot.loc[tot["spin"] == spin]["energy"]
        print("energy_temp:{}".format(energy))

        def drop_one_of_degenerate_levels(level_df, diff_en_threshold=2e-3):
            diff_band = np.diff(level_df)
            delete_band_indices = [level_df.index[i] for i in np.where(abs(diff_band) <= diff_en_threshold)[0]]
            print("remove_possible_degenerate_band_index:{}".format(delete_band_indices))
            level_df = level_df.drop(delete_band_indices)
            return level_df
        
        if is_vacuum_aligment:
            energy -= determine_defect_state_obj.vacuum_locpot
            energy = trunc(energy, 3)
            print(energy)
            energy = drop_one_of_degenerate_levels(energy)
            vbm = trunc(determine_defect_state_obj.vbm, 3)
            cbm = trunc(determine_defect_state_obj.cbm, 3)
        else:
            energy = trunc(energy, 3)
            print(energy)
            energy = drop_one_of_degenerate_levels(energy)
            vbm = trunc(determine_defect_state_obj.vbm, 3)
            cbm = trunc(determine_defect_state_obj.cbm, 3)
            
        occup = []
        for i in tot.loc[tot["spin"] == spin]["n_occ_e"]:
            if i > 0.4:
                occup.append(True)
            else:
                occup.append(False)
        levels[spin].update(dict(zip(energy, occup)))
        print("levels:{}".format(levels))
    eng = EnergyLevel(levels, top_texts=top_texts)
    fig = eng.plotting(vbm, cbm)
    levels.update(
        {
            "level_vbm": vbm,  "level_cbm": cbm,
            "level_up_deg": tuple([int(i.split("/")[1]) for i in top_texts["1"]]) if top_texts else None,
            "level_dn_deg": tuple([int(i.split("/")[1]) for i in top_texts["-1"]]) if top_texts else None,
            "level_up_ir": tuple(["".join(i.split("/")[-1].split(" ")) for i in top_texts["1"]]) if top_texts else 
            None,
            "level_dn_ir": tuple(["".join(i.split("/")[-1].split(" ")) for i in top_texts["-1"]]) if top_texts else None
        }
    )

    if determine_defect_state_obj.save_fig_path:
        fig.savefig(os.path.join(determine_defect_state_obj.save_fig_path, "defect_states", "{}_{}_{}.defect_states.png".format(
            determine_defect_state_obj.entry["formula_pretty"],
            determine_defect_state_obj.entry["task_id"],
            determine_defect_state_obj.entry["task_label"])))
    
    return levels, fig


def get_eigen_plot_v2(tot, determine_defect_state_obj, is_vacuum_aligment=False, edge_tol=(0.25, 0.25), 
                      eigen_plot_title=None):
    vbm, cbm = None, None

    if is_vacuum_aligment:
        tot["energy"] -= determine_defect_state_obj.vacuum_locpot
        tot["energy"] = trunc(tot["energy"], 3)
        print("Total states: {}".format(tot["energy"]))
        vbm = trunc(determine_defect_state_obj.vbm - determine_defect_state_obj.vacuum_locpot, 3)
        cbm = trunc(determine_defect_state_obj.cbm - determine_defect_state_obj.vacuum_locpot, 3)
    else:
        tot["energy"] = trunc(tot["energy"], 3)
        print("Total states: {}".format(tot["energy"]))
        vbm = trunc(determine_defect_state_obj.vbm, 3)
        cbm = trunc(determine_defect_state_obj.cbm, 3)

    def plotting(set_vbm, set_cbm, tot_df):
        from matplotlib.ticker import AutoMinorLocator
        plt.style.use(['grid'])

        up = tot_df.loc[tot_df["spin"] == "1", "energy"]
        dn = tot_df.loc[tot_df["spin"] == "-1", "energy"]
        
        up_deg = tot_df.loc[tot_df["spin"] == "1", "band_degeneracy"]
        dn_deg = tot_df.loc[tot_df["spin"] == "-1", "band_degeneracy"]

        up_occ = tot_df.loc[tot_df["spin"] == "1", "n_occ_e"]
        dn_occ = tot_df.loc[tot_df["spin"] == "-1", "n_occ_e"]

        up_ir = tot_df.loc[tot_df["spin"] == "1", "band_ir"]
        dn_ir = tot_df.loc[tot_df["spin"] == "-1", "band_ir"]

        up_band_id = tot_df.loc[tot_df["spin"] == "1", "band_id"]
        dn_band_id = tot_df.loc[tot_df["spin"] == "-1", "band_id"]

        fig, ax = plt.subplots(figsize=(12, 11), dpi=300)
        if up_ir.empty:
            up_ir = [None for i in up]
        if up_deg.empty:
            up_deg = [None for i in up]

        if dn_ir.empty:
            dn_ir = [None for i in dn]
        if dn_deg.empty:
            dn_deg = [None for i in dn]

        ax.bar(0, 2, 1.5, set_vbm-2, color="deepskyblue", align="edge")
        ax.bar(0, 2, 1.5, set_cbm, color="orange", align="edge")

        for level, occupied, deg, ir, band_id in zip(up, up_occ, up_deg, up_ir, up_band_id):
            # dx += 0.2
            color = None
            if deg == 1:
                color = "red"
            elif deg == 2:
                color = "blue"
            elif deg == 3:
                color = "green"
            elif deg == 4:
                color = "yellow"
            else:
                color = "black"

            if occupied:
                ax.hlines(level, 0, 0.5, colors=color)
                ax.text(0.5, level, "{}".format(band_id), fontsize=12)    
            else:
                ax.hlines(level, 0, 0.5, colors=color)
                ax.text(0.5, level, "{}*".format(band_id), fontsize=12)

        up_info = '\n'.join(
            [
                "{}/ {}/ {}/ {}/{}".format(level, occupied, deg, ir, band_id) for
                level, occupied, deg, ir, band_id in zip(up, up_occ, up_deg, up_ir, up_band_id)
            ]
        )
        ax.text(0.05, 0.81, up_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black'), size=12)

        for level, occupied, deg, ir, band_id in zip(dn, dn_occ, dn_deg, dn_ir, dn_band_id):
            color = None
            if deg == 1:
                color = "red"
            elif deg == 2:
                color = "blue"
            elif deg == 3:
                color = "green"
            elif deg == 4:
                color = "yellow"
            else:
                color = "black"

            if occupied:
                ax.hlines(level, 1, 1.5, colors=color)
                ax.text(0.9, level, "{}".format(band_id), fontsize=12)
            else:
                ax.hlines(level, 1, 1.5, colors=color)
                ax.text(0.9, level, "{}*".format(band_id), fontsize=12)

        dn_info = '\n'.join(
            [
                "{}/ {}/ {}/ {}/{}".format(level, occupied, deg, ir, band_id) for
                level, occupied, deg, ir, band_id in zip(dn, dn_occ, dn_deg, dn_ir, dn_band_id)
            ]
        )
        ax.text(0.75, 0.81, dn_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black'), size=12)


        ax.set_ylim(set_vbm-2, set_cbm+2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        for tick in ax.get_yticklabels():
            tick.set_fontsize(15)
    
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        
        if is_vacuum_aligment:
            ax.set_ylabel("Energy relative to vacuum (eV)", fontsize=15)
        else:
            ax.set_ylabel("Energy (eV)", fontsize=15)
        # ax.text(-0.1, 1.05, "{}-{}".format(uid, host_taskid), size=30, transform=ax.transAxes)
        return fig

    def plotting_v2(set_vbm, set_cbm, tot_df, edge_tol=edge_tol, eigen_plot_title=None):
        from matplotlib.ticker import AutoMinorLocator
        # plt.style.use(['grid'])

        in_gap_condition = (tot_df["energy"] >= vbm-edge_tol[0]) & (tot_df["energy"] <= cbm+edge_tol[1]) 
        up_condition = (tot_df["spin"] == "1") & in_gap_condition
        dn_condition = (tot_df["spin"] == "-1") & in_gap_condition
        
        up = tot_df.loc[up_condition, "energy"]
        dn = tot_df.loc[dn_condition, "energy"]
        
        # print colume names of tot_df
        print(tot_df.columns) 
        # if tot_df has band_degeneracy column, get the band_degeneracy of each band 
        if "band_degeneracy" in tot_df.columns:
            up_deg = tot_df.loc[up_condition, "band_degeneracy"]
            dn_deg = tot_df.loc[dn_condition, "band_degeneracy"]
        else:
            up_deg = [None for i in up]
            dn_deg = [None for i in dn] 
            print("band_degeneracy column not found in tot_df")
        
        up_occ = tot_df.loc[up_condition, "n_occ_e"]
        dn_occ = tot_df.loc[dn_condition, "n_occ_e"]

        if "band_ir" in tot_df.columns:
            up_ir = tot_df.loc[up_condition, "band_ir"]
            dn_ir = tot_df.loc[dn_condition, "band_ir"]
        else:
            up_ir = [None for i in up]
            dn_ir = [None for i in dn]
        
        if "band_id" in tot_df.columns:
            up_band_id = tot_df.loc[up_condition, "band_id"]
            dn_band_id = tot_df.loc[dn_condition, "band_id"]
        else:
            up_band_id = [None for i in up]
            dn_band_id = [None for i in dn]

        fig, ax = plt.subplots(figsize=(12, 11), dpi=300)
        if eigen_plot_title:
            ax.set_title(eigen_plot_title)

        ax.bar(0, 2, 1.5, set_vbm-2, color="deepskyblue", align="edge")
        ax.bar(0, 2, 1.5, set_cbm, color="orange", align="edge")

        for level, occupied, deg, ir, band_id in zip(up, up_occ, up_deg, up_ir, up_band_id):
            # dx += 0.2
            color = None
            if deg == 1:
                color = "red"
            elif deg == 2:
                color = "blue"
            elif deg == 3:
                color = "green"
            elif deg == 4:
                color = "yellow"
            else:
                color = "black"

            if occupied:
                if round(occupied, 1) == 0.5: 
                    ax.hlines(level, 0, 0.5, colors=color)
                    ax.text(0.5, level, "{}%".format(band_id), fontsize=12)
                else:
                    ax.hlines(level, 0, 0.5, colors=color)
                    ax.text(0.5, level, "{}".format(band_id), fontsize=12)
            else:
                ax.hlines(level, 0, 0.5, colors=color)
                ax.text(0.5, level, "{}*".format(band_id), fontsize=12)

        up_info = '\n'.join(
            [
                "{}/ {}/ {}/ {}/{}".format(level, occupied, deg, ir, band_id) for
                level, occupied, deg, ir, band_id in zip(up, up_occ, up_deg, up_ir, up_band_id)
            ]
        )
        ax.text(0.05, 0.81, up_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black'), size=12)

        for level, occupied, deg, ir, band_id in zip(dn, dn_occ, dn_deg, dn_ir, dn_band_id):
            color = None
            if deg == 1:
                color = "red"
            elif deg == 2:
                color = "blue"
            elif deg == 3:
                color = "green"
            elif deg == 4:
                color = "yellow"
            else:
                color = "black"

            if occupied:
                if round(occupied, 1) == 0.5:
                    ax.hlines(level, 1, 1.5, colors=color)
                    ax.text(0.9, level, "{}%".format(band_id), fontsize=12)
                else:
                    ax.hlines(level, 1, 1.5, colors=color)
                    ax.text(0.9, level, "{}".format(band_id), fontsize=12)
            else:
                ax.hlines(level, 1, 1.5, colors=color)
                ax.text(0.9, level, "{}*".format(band_id), fontsize=12)

        dn_info = '\n'.join(
            [
                "{}/ {}/ {}/ {}/{}".format(level, occupied, deg, ir, band_id) for
                level, occupied, deg, ir, band_id in zip(dn, dn_occ, dn_deg, dn_ir, dn_band_id)
            ]
        )
        ax.text(0.75, 0.81, dn_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black'), size=12)

        ax.text(0.625, set_vbm-0.5, "VBM:{}".format(set_vbm), fontsize=12)
        ax.text(0.625, set_cbm+0.5, "CBM:{}".format(set_cbm), fontsize=12)
        
        ax.set_ylim(set_vbm-2, set_cbm+2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        for tick in ax.get_yticklabels():
            tick.set_fontsize(15)

        ax.tick_params(axis="x", bottom=False, labelbottom=False)


        if is_vacuum_aligment:
            ax.set_ylabel("Energy relative to vacuum (eV)", fontsize=15)
        else:
            ax.set_ylabel("Energy (eV)", fontsize=15)
        # ax.text(-0.1, 1.05, "{}-{}".format(uid, host_taskid), size=30, transform=ax.transAxes)
        return fig


    fig = plotting_v2(vbm, cbm, tot, eigen_plot_title=eigen_plot_title)    
    
    if determine_defect_state_obj.save_fig_path:
        fig.savefig(os.path.join(determine_defect_state_obj.save_fig_path, "defect_states", "{}_{}_{}.defect_states.png".format(
            determine_defect_state_obj.entry["formula_pretty"],
            determine_defect_state_obj.entry["task_id"],
            determine_defect_state_obj.entry["task_label"])))

    levels = {}
    levels.update(
        {
            "level_vbm": vbm,
            "level_cbm": cbm,
            # if tot has a column of band_degeneracy, then transform it into tuple 
            "level_up_deg": tuple(tot.loc[tot["spin"] == "1", "band_degeneracy"]) if "band_degeneracy" in tot.loc[tot["spin"] == "1"].columns else (),
            "level_dn_deg": tuple(tot.loc[tot["spin"] == "-1", "band_degeneracy"]) if "band_degeneracy" in tot.loc[tot["spin"] == "-1"].columns else (),
            "level_up_ir": tuple(tot.loc[tot["spin"] == "1", "band_ir"]) if "band_ir" in tot.loc[tot["spin"] == "1"].columns else (),
            "level_dn_ir": tuple(tot.loc[tot["spin"] == "-1", "band_ir"]) if "band_ir" in tot.loc[tot["spin"] == "-1"].columns else (),
            "level_up_energy": tuple(tot.loc[tot["spin"] == "1", "energy"]),
            "level_dn_energy": tuple(tot.loc[tot["spin"] == "-1", "energy"]),
            "level_up_occ": tuple(tot.loc[tot["spin"] == "1", "n_occ_e"]),
            "level_dn_occ": tuple(tot.loc[tot["spin"] == "-1", "n_occ_e"]),
            "level_up_id": tuple(tot.loc[tot["spin"] == "1", "band_id"]) if "band_id" in tot.loc[tot["spin"] == "1"].columns else (),
            "level_up_index": tuple(tot.loc[tot["spin"] == "1", "band_index"]+1),
            "level_dn_id": tuple(tot.loc[tot["spin"] == "-1", "band_id"]) if "band_id" in tot.loc[tot["spin"] == "-1"].columns else (),
            "level_dn_index": tuple(tot.loc[tot["spin"] == "-1", "band_index"]+1),
        }
    )
    print("%%%%"*20)
    print(levels)
    return levels, fig

def get_ir_info(tot, ir_db, ir_entry_filter):
    # Locate idx in band_idex
    ir_entry = ir_db.collection.find_one(ir_entry_filter)
    if ir_entry:
        print("IR taskid:{}".format(ir_entry["task_id"]))
    else:
        print("IR taskid:{}".format(None))
        return tot, ir_entry
        
    input_sheet = tot["spin"]
    bd_idx_dict = ir_entry["irvsp"]["parity_eigenvals"]["single_kpt"]["(0.0, 0.0, 0.0)"]#["up"]["band_index"]
    band_id_list = []
    band_idxs = input_sheet.index
    for band_idx, spin in zip(tot.index+1, tot["spin"]):
        spin = "up" if spin == "1" else "down"
        bd_idx_sheet = bd_idx_dict[spin]["band_index"]
        if band_idx in bd_idx_sheet:
            band_id = bd_idx_sheet.index(band_idx)
            band_id_list.append(band_id)
        else:
            for degeneracy in range(10):
                if band_idx-degeneracy not in bd_idx_sheet:
                    continue
                else:
                    band_id = bd_idx_sheet.index(band_idx-degeneracy)
                    band_id_list.append(band_id)
                    break

    # Locate degeneracy from band_degeneracy
    bd_degen_dict = ir_entry["irvsp"]["parity_eigenvals"]["single_kpt"]["(0.0, 0.0, 0.0)"]
    band_degen_list = []
    for band_id, spin in zip(band_id_list, tot["spin"]):
        spin = "up" if spin == "1" else "down"
        band_degen_list.append(bd_degen_dict[spin]["band_degeneracy"][band_id])

    # Locate IR from irreducible_reps
    bd_ir_dict = ir_entry["irvsp"]["parity_eigenvals"]["single_kpt"]["(0.0, 0.0, 0.0)"]
    band_ir_list = []
    for band_id, spin in zip(band_id_list, tot["spin"]):
        spin = "up" if spin == "1" else "down"
        band_ir_list.append("".join(bd_ir_dict[spin]["irreducible_reps"][band_id].split("\n")))

    # Integrate info into tot
    ir_info_sheet = pd.DataFrame({"band_id": band_id_list,
                                  "band_degeneracy": band_degen_list,
                                  "band_ir": band_ir_list},
                                 index=tot.index)
    tot = pd.concat([tot, ir_info_sheet], axis=1)
    return tot, ir_entry

def get_in_gap_levels(tot_df, edge_tol):
    in_gap_levels = {}
    up_condition = (tot_df["spin"] == "1") & (tot_df["dist_from_vbm"] >= -1*edge_tol[0]) & (tot_df["dist_from_cbm"] <= 
                                                                                         edge_tol[1])

    dn_condition = (tot_df["spin"] == "-1") & (tot_df["dist_from_vbm"] >= -1*edge_tol[0]) & (tot_df["dist_from_cbm"] <=
                                                                                             edge_tol[1])

    if "band_degeneracy" in tot_df.columns and "band_ir" in tot_df.columns and "band_id" in tot_df.columns:
        up_levels = tot_df.loc[up_condition, ["band_id", "energy", "dist_from_vbm", "n_occ_e", "band_ir",
                                              "band_degeneracy", "band_index"]]
        dn_levels = tot_df.loc[dn_condition, ["band_id", "energy", "dist_from_vbm", "n_occ_e", "band_ir",
                                              "band_degeneracy", "band_index"]]
    else:
        up_levels = tot_df.loc[up_condition, ["energy", "dist_from_vbm", "n_occ_e", "band_index"]]
        dn_levels = tot_df.loc[dn_condition, ["energy", "dist_from_vbm", "n_occ_e", "band_index"]]



    in_gap_levels.update(
        {
            "up_in_gap_level": tuple(up_levels["energy"]),
            "up_in_gap_ir": tuple(up_levels["band_ir"]) if "band_ir" in up_levels.columns else (),
            "up_in_gap_occ": tuple(up_levels["n_occ_e"]),
            "up_in_gap_deg": tuple(up_levels["band_degeneracy"]) if "band_degeneracy" in up_levels.columns else (),
            "up_in_gap_band_id": tuple(up_levels["band_id"]) if "band_id" in up_levels.columns else (),
            "up_in_gap_band_index": tuple(up_levels["band_index"]+1)
        }
    )

    in_gap_levels.update(
        {
            "dn_in_gap_level": tuple(dn_levels["energy"]),
            "dn_in_gap_ir": tuple(dn_levels["band_ir"]) if "band_ir" in dn_levels.columns else (),
            "dn_in_gap_occ": tuple(dn_levels["n_occ_e"]),
            "dn_in_gap_deg": tuple(dn_levels["band_degeneracy"]) if "band_degeneracy" in dn_levels.columns else (),
            "dn_in_gap_band_id": tuple(dn_levels["band_id"]) if "band_id" in dn_levels.columns else (),
            "dn_in_gap_band_index": tuple(dn_levels["band_index"]+1)
        }
    )
    return in_gap_levels


def get_in_gap_transition(tot_df, edge_tol):
    # well-defined in-gap state: energetic difference of occupied states and vbm > 0.1   
    up_condition = (tot_df["spin"] == "1") & (tot_df["dist_from_vbm"] >= -1*edge_tol[0]) & (tot_df["dist_from_cbm"] <=
                                                                                           edge_tol[1])

    dn_condition = (tot_df["spin"] == "-1") & (tot_df["dist_from_vbm"] >= -1*edge_tol[0]) & (tot_df["dist_from_cbm"] <=
                                                                                             edge_tol[1])
    
    # if tot_df has band_id, band_ir, and band_degeneracy,
    if "band_degeneracy" in tot_df.columns and "band_id" in tot_df.columns and "band_ir" in tot_df.columns:
        up_tran_df = tot_df.loc[up_condition, ["band_id", "energy", "dist_from_vbm", "n_occ_e", "band_ir",
                                                  "band_degeneracy", "band_index"]]
        dn_tran_df = tot_df.loc[dn_condition, ["band_id", "energy", "dist_from_vbm", "dist_from_cbm", "n_occ_e", "band_ir",
                                               "band_degeneracy", "band_index"]]
    else:
        up_tran_df = tot_df.loc[up_condition, ["energy", "dist_from_vbm", "dist_from_cbm", "n_occ_e", "band_index"]]
        dn_tran_df = tot_df.loc[dn_condition, ["energy", "dist_from_vbm", "dist_from_cbm", "n_occ_e", "band_index"]]
    
    transition_dict = {}
    
    sum_occ_up = 0
    for en_up, occ_up in zip(up_tran_df["dist_from_vbm"], up_tran_df["n_occ_e"]):
        if occ_up > 0.2:
            sum_occ_up += occ_up
            
    sum_occ_dn = 0
    for en_dn, occ_dn in zip(dn_tran_df["dist_from_vbm"], dn_tran_df["n_occ_e"]):
        if occ_dn > 0.2:
            sum_occ_dn += occ_dn

    if sum_occ_up > sum_occ_dn:
        transition_dict.update({"triplet_from": "up"})
    else:
        transition_dict.update({"triplet_from": "dn"})

    # Calculate plausible optical transition energy
    if not up_tran_df["n_occ_e"].empty:
        dE_ups = -1*np.diff(up_tran_df["dist_from_vbm"])
        for idx, dE_up in enumerate(dE_ups):
            if (
                    round(dE_up, 1) != 0 and
                    round(up_tran_df["n_occ_e"].iloc[idx], 0) == 0 and
                    round(up_tran_df["n_occ_e"].iloc[idx+1], 1) >= 0.5
            ):
                transition_dict.update(
                    {
                        "up_tran_level": tuple(up_tran_df["energy"]),
                        "up_tran_from_vbm": tuple(up_tran_df["dist_from_vbm"]),
                        "up_tran_en": dE_up,
                        "up_tran_bottom": round(up_tran_df["dist_from_vbm"].iloc[idx+1], 3),
                        "up_tran_top": round(up_tran_df["dist_from_vbm"].iloc[idx], 3),
                        "up_tran_ir": tuple(up_tran_df["band_ir"]) if "band_ir" in up_tran_df.columns else (),
                        "up_tran_occ": tuple(up_tran_df["n_occ_e"]),
                        "up_tran_deg": tuple(up_tran_df["band_degeneracy"]) if "band_degeneracy" in up_tran_df.columns else (),
                        "up_tran_band_id": tuple(up_tran_df["band_id"]) if "band_id" in up_tran_df.columns else (),
                        "up_tran_band_index": tuple(up_tran_df["band_index"]+1),
                        
                        "up_tran_lumo_homo_energy": (round(up_tran_df["dist_from_vbm"].iloc[idx], 3),
                                                     round(up_tran_df["dist_from_vbm"].iloc[idx+1], 3)),
                        "up_tran_lumo_homo_band_id": (up_tran_df["band_id"].iloc[idx], up_tran_df["band_id"].iloc[
                            idx+1])  if "band_id" in up_tran_df.columns else (),
                        "up_tran_lumo_homo_band_index": (up_tran_df["band_index"].iloc[idx]+1,
                                                         up_tran_df["band_index"].iloc[idx+1]+1),
                        "up_tran_lumo_homo_deg": (up_tran_df["band_degeneracy"].iloc[idx],
                                                  up_tran_df["band_degeneracy"].iloc[idx+1]) if "band_degeneracy" in up_tran_df.columns else (),
                        "up_tran_lumo_homo_ir": (up_tran_df["band_ir"].iloc[idx],
                                                 up_tran_df["band_ir"].iloc[idx+1]) if "band_ir" in up_tran_df.columns else (),
                    }
                )
                break
            else:
                transition_dict.update({"up_tran_from_vbm": (), 
                                        "up_tran_en": None, # was 0,
                                        
                                        "up_tran_bottom": None, # was 0 
                                        "up_tran_top": None,  # was 0 
                                        
                                        "up_tran_ir": (),
                                        "up_tran_occ": (),
                                        "up_tran_deg": (),
                                        "up_tran_band_id": (),
                                        "up_tran_level": (),
                                        "up_tran_band_index": (),

                                        "up_tran_lumo_homo_energy": (),
                                        "up_tran_lumo_homo_band_id": (),
                                        "up_tran_lumo_homo_band_index": (),
                                        "up_tran_lumo_homo_deg": (),
                                        "up_tran_lumo_homo_ir": (),
                                       })
    else:
        transition_dict.update({"up_tran_from_vbm": (),
                                "up_tran_en": None, #was 0,
                                
                                "up_tran_bottom": None,# was 0 
                                "up_tran_top": None, # was 0
                                
                                "up_tran_occ": (),
                                "up_tran_deg": (),
                                "up_tran_band_id": (),
                                "up_tran_level": (),
                                "up_tran_band_index": (),

                                "up_tran_lumo_homo_energy": (),
                                "up_tran_lumo_homo_band_id": (),
                                "up_tran_lumo_homo_band_index": (),
                                "up_tran_lumo_homo_deg": (),
                                "up_tran_lumo_homo_ir": (),
                                })

    if not dn_tran_df["n_occ_e"].empty:
        dE_dns = -1*np.diff(dn_tran_df["dist_from_vbm"])
        for idx, dE_dn in enumerate(dE_dns):
            if (
                    round(dE_dn, 1) != 0 and
                    round(dn_tran_df["n_occ_e"].iloc[idx], 0) == 0 and
                    round(dn_tran_df["n_occ_e"].iloc[idx+1], 1) >= 0.5
            ):
                transition_dict.update(
                    {
                        "dn_tran_level": tuple(dn_tran_df["energy"]),
                        "dn_tran_from_vbm": tuple(dn_tran_df["dist_from_vbm"]),
                        "dn_tran_en": dE_dn,
                        "dn_tran_bottom": round(dn_tran_df["dist_from_vbm"].iloc[idx+1], 3),
                        "dn_tran_top": round(dn_tran_df["dist_from_vbm"].iloc[idx], 3),
                        "dn_tran_ir": tuple(dn_tran_df["band_ir"]) if "band_ir" in dn_tran_df.columns else (),
                        "dn_tran_occ": tuple(dn_tran_df["n_occ_e"]),
                        "dn_tran_deg": tuple(dn_tran_df["band_degeneracy"]) if "band_degeneracy" in dn_tran_df.columns else (),
                        "dn_tran_band_id": tuple(dn_tran_df["band_id"]) if "band_id" in dn_tran_df.columns else (),
                        "dn_tran_band_index": tuple(dn_tran_df["band_index"]+1),

                        "dn_tran_lumo_homo_energy": (round(dn_tran_df["dist_from_vbm"].iloc[idx], 3),
                                                     round(dn_tran_df["dist_from_vbm"].iloc[idx+1], 3)),
                        "dn_tran_lumo_homo_band_id": (dn_tran_df["band_id"].iloc[idx],
                                                      dn_tran_df["band_id"].iloc[idx+1]) if "band_id" in dn_tran_df.columns else (),
                        "dn_tran_lumo_homo_band_index": (dn_tran_df["band_index"].iloc[idx]+1,
                                                         dn_tran_df["band_index"].iloc[idx+1]+1),
                        "dn_tran_lumo_homo_deg": (dn_tran_df["band_degeneracy"].iloc[idx],
                                                  dn_tran_df["band_degeneracy"].iloc[idx+1]) if "band_degeneracy" in dn_tran_df.columns else (),
                        "dn_tran_lumo_homo_ir": (dn_tran_df["band_ir"].iloc[idx], dn_tran_df["band_ir"].iloc[idx+1]) if "band_ir" in dn_tran_df.columns else ()
                    }
                )
                break
            else:
                transition_dict.update({"dn_tran_from_vbm": (),
                                        "dn_tran_en": None, #was 0,
                                        
                                        "dn_tran_bottom": None, # was 0 
                                        "dn_tran_top": None, # was 0 
                                        
                                        "dn_tran_ir": (),
                                        "dn_tran_occ": (),
                                        "dn_tran_deg": (),
                                        "dn_tran_band_id": (),
                                        "dn_tran_level": (),
                                        "dn_tran_band_index": (),
                                        
                                        "dn_tran_lumo_homo_energy": (),
                                        "dn_tran_lumo_homo_band_id": (),
                                        "dn_tran_lumo_homo_band_index": (),
                                        "dn_tran_lumo_homo_deg": (),
                                        "dn_tran_lumo_homo_ir": (),
                                       })
    else:
        transition_dict.update({"dn_tran_from_vbm": (),
                                "dn_tran_en": None, #was 0
                                
                                "dn_tran_bottom": None, # was 0 
                                "dn_tran_top": None,  # was 0 
                                
                                "dn_tran_ir": (),
                                "dn_tran_occ": (),
                                "dn_tran_deg": (),
                                "dn_tran_band_id": (),
                                "dn_tran_level": (),
                                "dn_tran_band_index": (),

                                "dn_tran_lumo_homo_energy": (),
                                "dn_tran_lumo_homo_band_id": (),
                                "dn_tran_lumo_homo_band_index": (),
                                "dn_tran_lumo_homo_deg": (),
                                "dn_tran_lumo_homo_ir": (),
                               })        
    print(transition_dict)
    transition_df = pd.DataFrame([transition_dict])
    return transition_df


def get_defect_state_v1(db, db_filter, vbm, cbm, path_save_fig, plot="all", clipboard="tot", locpot=None,
                     threshold=0.1, locpot_c2db=None, ir_db=None, ir_entry_filter=None, 
                     is_vacuum_aligment_on_plot=False) -> object:
    """
    When one is using "db_cori_tasks_local", one must set ssh-tunnel as following:
    "ssh -f tsaie79@cori.nersc.gov -L 2222:mongodb07.nersc.gov:27017 -N mongo -u 2DmaterialQuantumComputing_admin -p
    tsaie79 localhost:2222/2DmaterialQuantumComputing"
    """

    can = DetermineDefectState(db=db, db_filter=db_filter, cbm=cbm, vbm=vbm, save_fig_path=path_save_fig, locpot=locpot,
                               locpot_c2db=locpot_c2db)
    tot, proj, d_df = can.get_candidates(
        0,
        threshold=threshold,
        select_bands=None
    )
    top_texts = None
    if ir_db and ir_entry_filter:
        tot, ir_entry = get_ir_info(tot, ir_db, ir_entry_filter)
        if ir_entry:
            top_texts = {"1": [], "-1": []}
            for spin in ["1", "-1"]:
                ir_info = tot.loc[tot["spin"] == spin]
                for band_id, band_degeneracy, band_ir in zip(ir_info["band_id"],
                                                             ir_info["band_degeneracy"],
                                                             ir_info["band_ir"]):
                    info = "{}/{}/{}".format(band_id, band_degeneracy, band_ir)
                    top_texts[spin].append(info)
                top_texts[spin] = list(dict.fromkeys(top_texts[spin]))

    print("top_texts:{}".format(top_texts))
    levels, eigen_plot = get_eigen_plot_v1(tot, can, top_texts, is_vacuum_aligment=is_vacuum_aligment_on_plot)
    tot["energy"] -= can.vacuum_locpot
    print("**"*20)
    print(d_df)
    e = {}
    e.update(
        {
            "up_band_idx": tuple(d_df["up_band_idx"][0]),
            "up_from_vbm": tuple(d_df["up_from_vbm"][0]),
            "up_occ": tuple(d_df["up_occ"][0]),
            "dn_band_idx": tuple(d_df["dn_band_idx"][0]),
            "dn_from_vbm": tuple(d_df["dn_from_vbm"][0]),
            "dn_occ": tuple(d_df["dn_occ"][0]),
        }
    )
    if top_texts:
        e.update({"up_ir": tuple(top_texts["1"])})
        e.update({"dn_ir": tuple(top_texts["-1"])})

    # well-defined in-gap state: energetic difference of occupied states and vbm > 0.1
    sum_occ_up = 0
    for en_up, occ_up in zip(d_df["up_from_vbm"][0], d_df["up_occ"][0]):
        if abs(en_up) > 0.1 and occ_up > 0.2:
            sum_occ_up += occ_up
    sum_occ_dn = 0
    for en_dn, occ_dn in zip(d_df["dn_from_vbm"][0], d_df["dn_occ"][0]):
        if abs(en_dn) > 0.1 and occ_dn > 0.2:
            sum_occ_dn += occ_dn

    e.update({"up_deg": tuple(Counter(np.around(e["up_from_vbm"], 2)).values()),
              "dn_deg": tuple(Counter(np.around(e["dn_from_vbm"], 2)).values())})
    if sum_occ_up > sum_occ_dn:
        e.update({"triplet_from": "up"})
    else:
        e.update({"triplet_from": "dn"})

    # Calculate plausible optical transition energy
    for en_up, occ_up, idx in zip(d_df["up_from_vbm"][0], d_df["up_occ"][0], range(len(d_df["up_occ"][0]))):
        if occ_up == 1 and idx != 0:
            h_occ_en = en_up
            l_unocc_en = d_df["up_from_vbm"][0][idx-1]
            e.update({"up_tran_en": round(l_unocc_en - h_occ_en, 3)})
            break
        elif occ_up > 0 and idx != 0 and d_df["up_occ"][0][0] == 0:
            h_occ_en = en_up
            l_unocc_en = d_df["up_from_vbm"][0][idx-1]
            e.update({"up_tran_en": round(l_unocc_en - h_occ_en, 3)})
            break
        elif occ_up > 0 and idx != 0 and d_df["up_occ"][0][0] != 0:
            h_occ_en = d_df["up_from_vbm"][0][idx+1]
            l_unocc_en = en_up
            e.update({"up_tran_en": round(l_unocc_en - h_occ_en, 3)})
            break
        else:
            e.update({"up_tran_en": 0})
    for en_dn, occ_dn, idx in zip(d_df["dn_from_vbm"][0], d_df["dn_occ"][0], range(len(d_df["dn_occ"][0]))):

        if occ_dn == 1 and idx != 0:
            h_occ_en = en_dn
            l_unocc_en = d_df["dn_from_vbm"][0][idx-1]
            e.update({"dn_tran_en": round(l_unocc_en - h_occ_en, 3)})
            break
        elif occ_dn > 0 and idx != 0 and d_df["dn_occ"][0][0] == 0:
            h_occ_en = en_dn
            l_unocc_en = d_df["dn_from_vbm"][0][idx-1]
            e.update({"dn_tran_en": round(l_unocc_en - h_occ_en, 3)})
            break
        elif occ_dn > 0 and idx != 0 and d_df["dn_occ"][0][0] != 0:
            h_occ_en = d_df["dn_from_vbm"][0][idx+1]
            l_unocc_en = en_dn
            e.update({"dn_tran_en": round(l_unocc_en - h_occ_en, 3)})
            break
        else:
            e.update({"dn_tran_en": 0})

    d_df = pd.DataFrame([e]).transpose()
    print(d_df)
    print("=="*20)
    print(proj)
    print("=="*20)
    print(tot)
    print("=="*20)
    print(levels)


    if type(clipboard) == tuple:
        if clipboard[0] == "tot":
            tot.loc[clipboard[1]].to_clipboard("\t")
        elif clipboard[0] == "proj":
            proj.loc[clipboard[1]].to_clipboard("\t")
    elif type(clipboard) == str:
        if clipboard == "tot":
            tot.to_clipboard("\t")
        elif clipboard == "proj":
            proj.to_clipboard("\t")
        else:
            d_df.to_clipboard("\t")

    if plot:
        cbm_set, vbm_set = None, None
        if locpot_c2db or locpot:
            cbm_set = can.cbm + can.vacuum_locpot
            vbm_set = can.vbm + can.vacuum_locpot
            efermi = can.efermi + can.vacuum_locpot
        else:
            cbm_set = cbm
            vbm_set = vbm
            efermi = can.efermi
        
        
        dos_plot = DosPlotDB(db=db, db_filter=db_filter, cbm=cbm_set, vbm=vbm_set, efermi=efermi, path_save_fig=path_save_fig)
        # dos_plot.nn = [25, 26, 31, 30, 29, 49, 45]
        if plot == "eigen":
            eigen_plot.show()
        if plot == "tdos":
            tdos_plt = dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
            tdos_plt.show()
            print(dos_plot.nn)
        if plot == "site":
            site_dos_plt = dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
            site_dos_plt.show()
        if plot == "spd":
            spd_dos_plt = dos_plot.spd_plots(energy_upper_bound=2, energy_lower_bound=2)
            spd_dos_plt.show()
        if plot == "orbital":
            orbital_dos_plt = dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
            orbital_dos_plt.show()
        if plot == "all":
            tdos_plt = dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
            tdos_plt.show()
            site_dos_plt = dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
            site_dos_plt.show()
            orbital_dos_plt = dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
            orbital_dos_plt.show()
            eigen_plot.show()


        if path_save_fig:
            for df, df_name in zip([tot, proj, d_df], ["tot", "proj", "d_state"]):
                path = os.path.join(path_save_fig, "xlsx", "{}_{}_{}_{}.xlsx".format(
                    can.entry["formula_pretty"],
                    can.entry["task_id"],
                    can.entry["task_label"],
                    df_name
                ))
                df.to_excel(path)

    return tot, proj, d_df, levels


def get_defect_state_v2(db, db_filter, vbm, cbm, path_save_fig, plot="all", clipboard="tot", locpot=None,
                     threshold=0.1, locpot_c2db=None, ir_db=None, ir_entry_filter=None,
                     is_vacuum_aligment_on_plot=False) -> object:
    """
    When one is using "db_cori_tasks_local", one must set ssh-tunnel as following:
    "ssh -f tsaie79@cori.nersc.gov -L 2222:mongodb07.nersc.gov:27017 -N mongo -u 2DmaterialQuantumComputing_admin -p
    tsaie79 localhost:2222/2DmaterialQuantumComputing"
    """

    can = NewDetermineDefectState(db=db, db_filter=db_filter, cbm=cbm, vbm=vbm, save_fig_path=path_save_fig, 
                                locpot=locpot,
                               locpot_c2db=locpot_c2db)
    perturbed_bandgap = can.cbm - can.vbm
    tot, proj, d_df, bulk_tot, bulk_proj, bulk_d_df = can.get_candidates(
        0,
        threshold=threshold,
        select_bands=None
    )
    top_texts = None
    top_texts_for_d_df = None
    if ir_db and ir_entry_filter:
        tot, ir_entry = get_ir_info(tot, ir_db, ir_entry_filter)
        if ir_entry:
            top_texts = {"1": [], "-1": []}
            top_texts_for_d_df = {"1": [], "-1": []}
            for spin in ["1", "-1"]:
                ir_info = tot.loc[tot["spin"] == spin]
                for band_id, band_degeneracy, band_ir in zip(ir_info["band_id"], ir_info["band_degeneracy"], 
                                                             ir_info["band_ir"]):
                    info = "{}/{}/{}".format(band_id, band_degeneracy, band_ir)
                    top_texts[spin].append(info)
                top_texts[spin] = list(dict.fromkeys(top_texts[spin]))
                
                ir_info_for_d_df = tot.loc[(tot["spin"] == spin) & (tot["energy"] > can.vbm) & (tot["energy"] <      
                                                                                                can.cbm)]
                for band_id, band_degeneracy, band_ir in zip(ir_info_for_d_df["band_id"], ir_info_for_d_df[
                    "band_degeneracy"], ir_info_for_d_df["band_ir"]):
                    info = "{}/{}/{}".format(band_id, band_degeneracy, band_ir)
                    top_texts_for_d_df[spin].append(info)
                top_texts_for_d_df[spin] = list(dict.fromkeys(top_texts_for_d_df[spin]))


    print("$$$"*20)
    print("top_texts:{}".format(top_texts))
    levels, eigen_plot = get_eigen_plot_v1(tot, can, top_texts, is_vacuum_aligment=is_vacuum_aligment_on_plot)
    print("**"*20)
    print("d_df:{}".format(d_df))
    e = {}
    e.update(
        {
            "up_band_idx": tuple(d_df["up_band_idx"][0]),
            "up_from_vbm": tuple(d_df["up_from_vbm"][0]),
            "up_occ": tuple(d_df["up_occ"][0]),
            "dn_band_idx": tuple(d_df["dn_band_idx"][0]),
            "dn_from_vbm": tuple(d_df["dn_from_vbm"][0]),
            "dn_occ": tuple(d_df["dn_occ"][0]),
        }
    )
    if top_texts_for_d_df:
        e.update({"up_ir": tuple(top_texts_for_d_df["1"])})
        e.update({"dn_ir": tuple(top_texts_for_d_df["-1"])})

    # well-defined in-gap state: energetic difference of occupied states and vbm > 0.1    
    sum_occ_up = 0
    for en_up, occ_up in zip(d_df["up_from_vbm"][0], d_df["up_occ"][0]):
        if occ_up > 0.2:
            sum_occ_up += occ_up
    sum_occ_dn = 0
    for en_dn, occ_dn in zip(d_df["dn_from_vbm"][0], d_df["dn_occ"][0]):
        if occ_dn > 0.2:
            sum_occ_dn += occ_dn
    e.update({"up_deg": tuple(Counter(np.around(e["up_from_vbm"], 2)).values()),
              "dn_deg": tuple(Counter(np.around(e["dn_from_vbm"], 2)).values())})
    if sum_occ_up > sum_occ_dn:
        e.update({"triplet_from": "up"})
    else:
        e.update({"triplet_from": "dn"})

    # Calculate plausible optical transition energy
    if len(d_df["up_occ"][0]) > 1:
        dE_ups = -1*np.diff(d_df["up_from_vbm"][0])
        for idx, dE_up in enumerate(dE_ups):
            if (
                    round(dE_up, 1) != 0 and
                    round(d_df["up_occ"][0][idx], 0) == 0 and
                    round(d_df["up_occ"][0][idx+1], 1) >= 0.5
            ):
                e.update(
                    {
                        "up_tran_en": dE_up, 
                        "up_tran_bottom": round(d_df["up_from_vbm"][0][idx+1], 3),
                        "up_tran_top": round(d_df["up_from_vbm"][0][idx], 3)
                    }
                )
                break
            else:
                e.update({"up_tran_en": 0, "up_tran_bottom": 0, "up_tran_top": 0})
    else:
        e.update({"up_tran_en": 0, "up_tran_bottom": 0, "up_tran_top": 0})

        
    if len(d_df["dn_occ"][0]) > 1:
        dE_dns = -1*np.diff(d_df["dn_from_vbm"][0])
        for idx, dE_dn in enumerate(dE_dns):
            if (
                    round(dE_dn, 1) != 0 and
                    round(d_df["dn_occ"][0][idx], 0) == 0 and
                    round(d_df["dn_occ"][0][idx+1], 1) >= 0.5
            ):
                e.update(
                    {
                        "dn_tran_en": dE_dn,
                        "dn_tran_bottom": round(d_df["dn_from_vbm"][0][idx+1], 3),
                        "dn_tran_top": round(d_df["dn_from_vbm"][0][idx], 3)
                    }
                )                
                break
            else:
                e.update({"dn_tran_en": 0, "dn_tran_bottom": 0, "dn_tran_top": 0})
    else:
        e.update({"dn_tran_en": 0, "dn_tran_bottom": 0, "dn_tran_top": 0})

    d_df = pd.DataFrame([e]).transpose()
    print(d_df)
    print("=="*20)
    print(proj)
    print("=="*20)
    print(tot)
    print("=="*20)
    print(levels)


    if type(clipboard) == tuple:
        if clipboard[0] == "tot":
            tot.loc[clipboard[1]].to_clipboard("\t")
        elif clipboard[0] == "proj":
            proj.loc[clipboard[1]].to_clipboard("\t")
    elif type(clipboard) == str:
        if clipboard == "tot":
            tot.to_clipboard("\t")
        elif clipboard == "proj":
            proj.to_clipboard("\t")
        else:
            d_df.to_clipboard("\t")

    if plot:
        cbm_set, vbm_set = None, None
        if locpot_c2db or locpot:
            cbm_set = can.cbm + can.vacuum_locpot
            vbm_set = can.vbm + can.vacuum_locpot
            efermi = can.efermi + can.vacuum_locpot
        else:
            cbm_set = can.cbm
            vbm_set = can.vbm
            efermi = can.efermi


        dos_plot = DosPlotDB(db=db, db_filter=db_filter, cbm=cbm_set, vbm=vbm_set, efermi=efermi, path_save_fig=path_save_fig)
        # dos_plot.nn = [25, 26, 31, 30, 29, 49, 45]
        if plot == "eigen":
            eigen_plot.show()
        if plot == "tdos":
            tdos_plt = dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
            tdos_plt.show()
            print(dos_plot.nn)
        if plot == "site":
            site_dos_plt = dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
            site_dos_plt.show()
        if plot == "spd":
            spd_dos_plt = dos_plot.spd_plots(energy_upper_bound=2, energy_lower_bound=2)
            spd_dos_plt.show()
        if plot == "orbital":
            orbital_dos_plt = dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
            orbital_dos_plt.show()
        if plot == "all":
            tdos_plt = dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
            tdos_plt.show()
            site_dos_plt = dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
            site_dos_plt.show()
            orbital_dos_plt = dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
            orbital_dos_plt.show()
            eigen_plot.show()


        if path_save_fig:
            for df, df_name in zip([tot, proj, d_df], ["tot", "proj", "d_state"]):
                path = os.path.join(path_save_fig, "xlsx", "{}_{}_{}_{}.xlsx".format(
                    can.entry["formula_pretty"],
                    can.entry["task_id"],
                    can.entry["task_label"],
                    df_name
                ))
                df.to_excel(path)

    return tot, proj, d_df, levels


def get_defect_state_v3(db, db_filter, vbm, cbm, path_save_fig, plot="all", clipboard="tot", locpot=None,
                        threshold=0.1, locpot_c2db=None, ir_db=None, ir_entry_filter=None,
                        is_vacuum_aligment_on_plot=False, edge_tol=(0.25, 0.25)) -> object:
    """
    When one is using "db_cori_tasks_local", one must set ssh-tunnel as following:
    "ssh -f tsaie79@cori.nersc.gov -L 2222:mongodb07.nersc.gov:27017 -N mongo -u 2DmaterialQuantumComputing_admin -p
    tsaie79 localhost:2222/2DmaterialQuantumComputing"
    """

    can = DetermineDefectStateV3(db=db, db_filter=db_filter, cbm=cbm, vbm=vbm, save_fig_path=path_save_fig,
                                  locpot=locpot,
                                  locpot_c2db=locpot_c2db)
    perturbed_bandgap = can.cbm - can.vbm
    # define defect states and bulk states
    tot, proj, bulk_tot, bulk_proj = can.get_candidates(
        0,
        threshold=threshold,
        select_bands=None
    )
    # print("checking!"*20, bulk_tot.loc[(bulk_tot["spin"] == "-1") & (bulk_tot.index>=209) & (bulk_tot.index<=216)])
    # print("checking!"*20, tot.loc[(tot["spin"] == "-1") & (tot.index>=209) & (tot.index<=216)])

    top_texts = None
    top_texts_for_d_df = None
    perturbed_bandedge_ir = []

    if ir_db and ir_entry_filter:
        tot, ir_entry = get_ir_info(tot, ir_db, ir_entry_filter)

        bandedge_bulk_tot = bulk_tot.loc[(bulk_tot.index == can.vbm_index[0]) | (bulk_tot.index == can.cbm_index[0])]
        bandedge_bulk_tot, bandedge_bulk_ir_entry = get_ir_info(bandedge_bulk_tot, ir_db, ir_entry_filter)
        print("==bandedge_ir"*20)
        print(bandedge_bulk_tot)

        perturbed_bandedge_ir.append(bandedge_bulk_tot.loc[bandedge_bulk_tot.index == can.vbm_index[0],
                                                           "band_ir"].iloc[0].split(" ")[0])
        perturbed_bandedge_ir.append(bandedge_bulk_tot.loc[bandedge_bulk_tot.index == can.cbm_index[0],
                                                           "band_ir"].iloc[0].split(" ")[0])
        print("%%"* 20)
        print(tot.columns)
        if ir_entry:
            top_texts = {"1": [], "-1": []}
            top_texts_for_d_df = {"1": [], "-1": []}
            for spin in ["1", "-1"]:
                ir_info = tot.loc[tot["spin"] == spin]
                for band_id, band_degeneracy, band_ir in zip(ir_info["band_id"], ir_info["band_degeneracy"],
                                                             ir_info["band_ir"]):
                    info = "{}/{}/{}".format(band_id, band_degeneracy, band_ir)
                    top_texts[spin].append(info)
                top_texts[spin] = top_texts[spin]

                ir_info_for_d_df = tot.loc[(tot["spin"] == spin) & (tot["energy"] > can.vbm) & (tot["energy"] <
                                                                                                can.cbm)]
                for band_id, band_degeneracy, band_ir in zip(ir_info_for_d_df["band_id"], ir_info_for_d_df[
                    "band_degeneracy"], ir_info_for_d_df["band_ir"]):
                    info = "{}/{}/{}".format(band_id, band_degeneracy, band_ir)
                    top_texts_for_d_df[spin].append(info)
                top_texts_for_d_df[spin] = top_texts_for_d_df[spin]


    print("top_texts:{}".format(top_texts))
    levels, eigen_plot = get_eigen_plot_v2(tot, can, is_vacuum_aligment=is_vacuum_aligment_on_plot, 
                                           edge_tol=edge_tol, eigen_plot_title=db_filter["task_id"])
    levels.update({"perturbed_level_edge_ir": tuple(perturbed_bandedge_ir)})
    d_df = get_in_gap_transition(tot, edge_tol) # need to fix
    print("Transition: {}".format(d_df))
    in_gap_levels = get_in_gap_levels(tot, edge_tol)
    print(tot)

    if type(clipboard) == tuple:
        if clipboard[0] == "tot":
            tot.loc[clipboard[1]].to_clipboard("\t")
        elif clipboard[0] == "proj":
            proj.loc[clipboard[1]].to_clipboard("\t")
    elif type(clipboard) == str:
        if clipboard == "tot":
            tot.to_clipboard("\t")
        elif clipboard == "proj":
            proj.to_clipboard("\t")
        else:
            d_df.to_clipboard("\t")

    if plot:
        cbm_set, vbm_set = None, None
        if locpot_c2db or locpot:
            cbm_set = can.cbm + can.vacuum_locpot
            vbm_set = can.vbm + can.vacuum_locpot
            efermi = can.efermi + can.vacuum_locpot
        else:
            cbm_set = can.cbm
            vbm_set = can.vbm
            efermi = can.efermi


        dos_plot = DosPlotDB(db=db, db_filter=db_filter, cbm=cbm_set, vbm=vbm_set, efermi=efermi, path_save_fig=path_save_fig)
        # dos_plot.nn = [25, 26, 31, 30, 29, 49, 45]
        if plot == "eigen":
            plt.show()
        if plot == "tdos":
            tdos_plt = dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
            plt.show()
            print(dos_plot.nn)
        if plot == "site":
            site_dos_plt = dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
            plt.show()
        if plot == "spd":
            spd_dos_plt = dos_plot.spd_plots(energy_upper_bound=2, energy_lower_bound=2)
            plt.show()
        if plot == "orbital":
            orbital_dos_plt = dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
            plt.show()
        if plot == "all":
            eig_plt = eigen_plot
            plt.show()
            tdos_plt = dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
            plt.show()
            site_dos_plt = dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
            plt.show()
            orbital_dos_plt = dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
            plt.show()


        if path_save_fig:
            for df, df_name in zip([tot, proj, d_df], ["tot", "proj", "d_state"]):
                path = os.path.join(path_save_fig, "xlsx", "{}_{}_{}_{}.xlsx".format(
                    can.entry["formula_pretty"],
                    can.entry["task_id"],
                    can.entry["task_label"],
                    df_name
                ))
                df.to_excel(path)

    return tot, proj, d_df, levels, in_gap_levels


class RunDefectState:
    @classmethod
    def get_defect_state_with_ir(cls):
        from qubitPack.tool_box import get_db

        from pymatgen import Structure
        import os
        from matplotlib import pyplot as plt

        defect_taskid = 6043
        SCAN2dDefect = get_db("Scan2dDefect", "calc_data",  user="Jeng_ro", password="qimin", port=12347)
        SCAN2dIR = get_db("Scan2dDefect", "ir_data", port=12347)
        # 
        # SCAN2dDefect = get_db("HSE_triplets_from_Scan2dDefect", "calc_data-pbe_pc", port=12347)
        # SCAN2dIR = get_db("HSE_triplets_from_Scan2dDefect", "ir_data-pbe_pc", port=12347)

        defect_db =SCAN2dDefect
        ir_db = SCAN2dIR

        defect = defect_db.collection.find_one({"task_id": defect_taskid})

        pc_from_id = defect["pc_from_id"]
        defect_name = defect["defect_name"]
        charge_state = defect["charge_state"]

        level_info, levels, defect_levels = None, None, None
        state = get_defect_state_v3(
            defect_db,
            {"task_id": defect_taskid},
            -10, 10,
            None,
            "all",
            None,
            None,  #(host_db, host_taskid, 0, vbm_dx, cbm_dx),
            0.2,  #0.2
            locpot_c2db=None,  #(c2db, c2db_uid, 0)
            is_vacuum_aligment_on_plot=True,
            edge_tol=(0.25, 0.25), # defect state will be picked only if it's above vbm by 0.025 eV and below
            # cbm by 0.025 eV
            ir_db=ir_db,
            ir_entry_filter={"pc_from_id": pc_from_id, "defect_name": defect_name, "charge_state": charge_state},
            # ir_entry_filter={"prev_fw_taskid": defect_taskid},
        )

        tot, proj, d_df, levels, defect_levels = state
        level_info = d_df.to_dict("records")[0]
        plt.show()
        return tot, proj, d_df, levels, defect_levels

    @classmethod
    def get_defect_state_without_ir(cls, defect_taskid, host_taskid, defect_col, host_col, vbm_dx=0, cbm_dx=0):
        from qubitPack.qc_searching.analysis.main import get_defect_state_v1
        from qubitPack.tool_box import get_db
        from pymatgen import Structure
        import os

        defect_db = get_db("antisiteQubit", defect_col, port=12345)
        host_db = get_db("owls", host_col, port=12345)
        c2db = get_db("2dMat_from_cmr_fysik", "2dMaterial_v1", port=12345, user="readUser", password="qiminyan")

        tk_id = defect_taskid

        # pc_from_id = defect_db.collection.find_one({"task_id": tk_id})["pc_from"]
        # c2db_uid = host_db.collection.find_one({"task_id": pc_from_id})["c2db_info"]["uid"]

        tot, proj, d_df, levels = get_defect_state_v1(
            defect_db,
            {"task_id": defect_taskid},
            -10, 10,
            None,
            "all",
            locpot=(host_db, host_taskid, 0, vbm_dx, cbm_dx),
            #(get_db("antisiteQubit", "W_S_Ef"), 312, 0.), 591, 593:WS2, 592, 594:WSe2 630:WTe2, 611, :MoS2, 610,
            # :MoSe2,
            # 631:MoTe2
            threshold=0.,
            locpot_c2db=None, #(c2db, "MoS2-MoS2-NM", 0.387, 1.5),
            is_vacuum_aligment_on_plot=True
        )
        return tot, proj, d_df, levels


if __name__ == '__main__':
    # tot, proj, d_df, levels  = RunDefectState.get_defect_state_without_ir(744, 4237, "vac_tmd", "mx2_antisite_pc")
    tot, proj, d_df, levels, defect_levels = RunDefectState.get_defect_state_with_ir()