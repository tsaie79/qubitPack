from qubitPack.qc_searching.analysis.dos_plot_from_db import DosPlotDB
from qubitPack.qc_searching.analysis.read_eigen import DetermineDefectState
from qubitPack.qc_searching.py_energy_diagram.application.defect_levels import EnergyLevel
from qubitPack.qc_searching.analysis.dos_plot_from_db import DB_CONFIG_PATH

import matplotlib.pyplot as plt
from glob import glob
import os
import pandas as pd
import numpy as np
from collections import Counter

def get_eigen_plot(tot, determine_defect_state_obj, top_texts):
    levels = {"1": {}, "-1":{}}
    for spin in ["1", "-1"]:
        energy = tot.loc[tot["spin"]==spin]["energy"]
        occup = []
        for i in tot.loc[tot["spin"]==spin]["n_occ_e"]:
            if i > 0.4:
                occup.append(True)
            else:
                occup.append(False)
        levels[spin].update(dict(zip(energy, occup)))
    print(levels)
    eng = EnergyLevel(levels, top_texts=top_texts)
    fig = eng.plotting(
        round(determine_defect_state_obj.vbm + determine_defect_state_obj.vacuum_locpot, 3),
        round(determine_defect_state_obj.cbm + determine_defect_state_obj.vacuum_locpot, 3)
    )

    if determine_defect_state_obj.save_fig_path:
        fig.savefig(os.path.join(determine_defect_state_obj.save_fig_path, "defect_states", "{}_{}_{}.defect_states.png".format(
            determine_defect_state_obj.entry["formula_pretty"],
            determine_defect_state_obj.entry["task_id"],
            determine_defect_state_obj.entry["task_label"])))

def get_ir_info(tot, ir_db, ir_entry_filter):
    # Locate idx in band_idex
    ir_entry = ir_db.collection.find_one(ir_entry_filter)
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
    return tot

def get_defect_state(db, db_filter, vbm, cbm, path_save_fig, plot=True, clipboard="tot", locpot=None,
                     threshold=0.1, locpot_c2db=None, ir_db=None, ir_entry_filter=None, top_texts=None) -> object:
    """
    When one is using "db_cori_tasks_local", one must set ssh-tunnel as following:
    "ssh -f tsaie79@cori.nersc.gov -L 2222:mongodb07.nersc.gov:27017 -N mongo -u 2DmaterialQuantumComputing_admin -p
    tsaie79 localhost:2222/2DmaterialQuantumComputing"
    """

    can = DetermineDefectState(db=db, db_filter=db_filter, cbm=cbm, vbm=vbm, save_fig_path=path_save_fig, locpot=locpot,
                               locpot_c2db=locpot_c2db)
    # can.nn = [25, 26, 31, 30, 29, 49, 45]
    tot, proj, d_df = can.get_candidates(
        0,
        threshold=threshold,
        select_bands=None
    )
    if ir_db and ir_entry_filter:
        tot = get_ir_info(tot, ir_db, ir_entry_filter)
        top_texts = {"1": [], "-1": []}
        for spin in ["1", "-1"]:
            ir_info = tot.loc[tot["spin"]==spin]
            for band_id, band_degeneracy, band_ir in zip(ir_info["band_id"], ir_info["band_degeneracy"], ir_info["band_ir"]):
                info = "{}/{}/{}".format(band_id, band_degeneracy, band_ir)
                top_texts[spin].append(info)
            top_texts[spin] = list(dict.fromkeys(top_texts[spin]))

    print(top_texts)
    get_eigen_plot(tot, can, top_texts)

    print("**"*20)
    print(d_df)
    e = {}
    e.update(
        {
            "up_band_idx": d_df["up_band_idx"][0],
            "up_from_vbm": d_df["up_from_vbm"][0],
            "up_occ": d_df["up_occ"][0],
            "dn_band_idx": d_df["dn_band_idx"][0],
            "dn_from_vbm": d_df["dn_from_vbm"][0],
            "dn_occ": d_df["dn_occ"][0]
        }
    )
    # well-defined in-gap state: energetic difference of occupied states and vbm > 0.1
    sum_occ_up = 0
    for en_up, occ_up in zip(d_df["up_from_vbm"][0], d_df["up_occ"][0]):
        if abs(en_up) > 0.1 and occ_up > 0.2:
            sum_occ_up += occ_up
    sum_occ_dn = 0
    for en_dn, occ_dn in zip(d_df["dn_from_vbm"][0], d_df["dn_occ"][0]):
        if abs(en_dn) > 0.1 and occ_dn > 0.2:
            sum_occ_dn += occ_dn

    e.update({"up_deg": list(Counter(np.around(e["up_from_vbm"], 2)).values()),
              "dn_deg": list(Counter(np.around(e["dn_from_vbm"], 2)).values())})
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
    # print("=="*20)
    # print(d_df)


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
        dos_plot.total_dos(energy_upper_bound=2, energy_lower_bound=2)
        print(dos_plot.nn)
        dos_plot.sites_plots(energy_upper_bound=2, energy_lower_bound=2)
        dos_plot.orbital_plot(dos_plot.nn[-1], 2, 2)
        # plt.show()

        if path_save_fig:
            for df, df_name in zip([tot, proj, d_df], ["tot", "proj", "d_state"]):
                path = os.path.join(path_save_fig, "xlsx", "{}_{}_{}_{}.xlsx".format(
                    can.entry["formula_pretty"],
                    can.entry["task_id"],
                    can.entry["task_label"],
                    df_name
                ))
                df.to_excel(path)

    return tot, proj, d_df

if __name__ == '__main__':
    from qubitPack.tool_box import get_db
    # db = get_db("owls", 'mx2_antisite_basic_aexx0.25_final')
    db = get_db("single_photon_emitter", "standard_defect")
    # db = get_db("antisiteQubit", "move_z")

    c2db = get_db("2dMat_from_cmr_fysik", "2dMaterial_v1", user="adminUser", password="qiminyan")
    tot, proj, d_df = get_defect_state(
        db,
        {"task_id": 266},
        1, -5,
        None,
        True,
        "proj",
        None, #(get_db("antisiteQubit", "W_S_Ef"), 312, 0.),
        0.2,
        locpot_c2db=None #(c2db, "WTe2-MoS2-NM", 0)
    )
    # proj = proj.loc[[317, 325, 326], [5, 6, 0, 25, "orbital", "spin", "adjacent", "antisite"]]
    # proj = proj.loc[(proj["spin"] == "1") & (proj["orbital"] == "dz2")]
    # proj = proj.loc[(proj["spin"] == "1") & (proj["orbital"] == "dxy") & (proj.index < 328) & (315 < proj.index)]
    # proj.sort_values(["adjacent", 25], inplace=True, ascending=False)
    # proj = proj[25]
    # proj = proj.round(3)
    # print(proj)
    # proj.loc[proj["spin"] == "1", :].to_clipboard("\t")

