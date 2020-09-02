#%%
from atomate.vasp.database import VaspCalcDb
import pandas as pd
from pymatgen import Structure, Element, Spin
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.vasp.outputs import Vasprun, Locpot, VolumetricData
import os
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from IPython.display import display

db_wse2_like = VaspCalcDb.from_db_file("db_WSe2_like_Ef_from_C2DB.json")
db_wse2_Ef = VaspCalcDb.from_db_file("db_wse2_Ef.json")
db_dk = VaspCalcDb.from_db_file("db_dk_local.json")
db_c2db_tmdc_bglg1 = VaspCalcDb.from_db_file("db_c2db_tmdc_bglg1.json")
db_mx2_antisite_basic_aexx035 = VaspCalcDb.from_db_file('/Users/jeng-yuantsai/Research/qubit/'
                                                        'calculations/mx2_antisite_basic_aexx0.35/db.json')
db_mx2_antisite_basic = VaspCalcDb.from_db_file('/Users/jeng-yuantsai/Research/qubit/'
                                                        'calculations/mx2_antisite_basic/db.json')


#%% Defect state plot
from matplotlib.ticker import AutoMinorLocator

def triplet_defect_state_plot(db, filter_label, up, down, save_fig):
    colors = {
        "cbm": "orange",
        "vbm": "deepskyblue"
    }
    evac = []
    for chemsys in filter_label:
        e = db.collection.find_one({"task_label":"HSE_scf", "chemsys":chemsys})
        print(e["task_id"])
        vac = max(e["calcs_reversed"][0]["output"]["locpot"]["2"])
        evac.append(vac)
    evac = np.array(evac)

    vbms_025 = []
    cbms_025 = []
    for chemsys in filter_label:
        filters = {
            "chemsys": chemsys,
            "task_label":"hse gap",
            "input.incar.AEXX": 0.25
        }
        e = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/"
                                    "mx2_antisite_basic_bandgap/db.json").collection.find_one(filters)
        vac = max(e["calcs_reversed"][0]["output"]["locpot"]["2"])
        cbms_025.append(e["output"]["cbm"] - vac)
        vbms_025.append(e["output"]["vbm"] - vac)
        
    # vbms_025.append(-0.921-evac[0]) for sandwich
    # cbms_025.append(1.145-evac[0]) for sandwich
    cbms_025 = np.array(cbms_025)
    vbms_025 = np.array(vbms_025)

    bandgap = pd.DataFrame({"bandgap":dict(zip(filter_labels, cbms_025-vbms_025))})
    print(bandgap)
    x = np.arange(len(vbms_025))
    emin_025 = min(vbms_025)
    emin = emin_025-0.5


    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1, 1, 1)

    #up
    up_band = []
    up_band_occ = []
    for chemsys, vac in zip(filter_label, evac):
        e = db.collection.find_one({"task_label":"HSE_scf", "chemsys":chemsys})
        up_ev = []
        up_ev_occ = []
        ev = db.get_eigenvals(e["task_id"])
        for band in up[e['chemsys']]:
            up_ev.append(ev["1"][0][band][0]-vac)
            if ev["1"][0][band][1] == 1:
                up_ev_occ.append(ev["1"][0][band][0]-vac)
        up_band.append(up_ev)
        up_band_occ.append(up_ev_occ)

    for compound_idx in range(len(up_band)):
        ax.hlines(up_band[compound_idx], x[compound_idx]-0.4, x[compound_idx]-0.05)

    # for occ_idx in range(len(up_band_occ)):
    #     for energy, pos in zip(up_band_occ[occ_idx], [0.35, 0.2]):
    #         ax.text(x[occ_idx]-pos, energy, "\u2b06")

    #dn
    dn_band = []
    dn_band_occ = []
    for chemsys, vac in zip(filter_label, evac):
        e = db.collection.find_one({"task_label":"HSE_scf", "chemsys":chemsys})
        ev = db.get_eigenvals(e["task_id"])
        dn_ev = []
        dn_ev_occ = []
        for band in down[e["chemsys"]]:
            print(band)
            if band:
                dn_ev.append(ev["-1"][0][band][0]-vac)
                if ev["-1"][0][band][1] == 1:
                    dn_ev_occ.append(ev["-1"][0][band][0]-vac)
        dn_band.append(dn_ev)
        dn_band_occ.append(dn_ev_occ)
        print(dn_band)
    print(dn_band)
    print(dn_band_occ)

    for compound_idx in range(len(dn_band)):
        ax.hlines(dn_band[compound_idx], x[compound_idx]+0.05, x[compound_idx]+0.4)

    for occ_idx in range(len(dn_band_occ)):
        for energy, pos in zip(dn_band_occ[occ_idx], [-0.1, 0]):
            ax.text(x[occ_idx]-pos, energy, "\u2b07")





    ax.bar(x, np.array(vbms_025) - emin, bottom=emin, color=colors["vbm"])
    ax.bar(x, -np.array(cbms_025), bottom=cbms_025, color=colors["cbm"])

    ax.set_ylim(emin, -3)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels_in_plot, rotation=0, fontsize=10)
    ax.set_xticks([])
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))


    # plt.ylabel('Energy relative to vacuum (eV)', fontsize=20)
    # plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join("/Users/jeng-yuantsai/Research/qubit/plt", "{}.eps".format(save_fig)),
                    img_format="eps")
    plt.show()

#%% Defect states
filter_labels = ["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"]
# labels_in_plot = ["${W_{S}}^0$", "${W_{Se}}^0$", "${Mo_{S}}^0$", "${Mo_{Se}}^0$"]
triplet_defect_state_plot(
    VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.25_final/db.json"),
    filter_labels,
    {"S-W":[224,225,226],"Se-W":[224,225,226],"Te-W":[224,225,226],
     "Mo-S":[302,303,304],"Mo-Se":[302,303,304], "Mo-Te":[302,303,304]},
    {"S-W":[None],"Se-W":[None],"Te-W":[None], "Mo-S":[None],"Mo-Se":[None], "Mo-Te":[None]},
    "mx2_defect_state_withTe"
)
#%%
filter_labels = ["S-W"]
labels_in_plot = ["${W_{S}}^0$"]
triplet_defect_state_plot(
    VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.25_final/db.json"),
    filter_labels,
    labels_in_plot,
    {"S-W":[224,225,226]},
    {"S-W":[None]}
)
#%% BN-WS2-BN defect states
filter_labels = ["B-N-S-W"]
labels_in_plot = ["${W_{S}}^0$", "${W_{Se}}^0$", "${W_{Te}}^0$", "${Mo_{S}}^0$", "${Mo_{Se}}^0$", "${Mo_{Te}}^0$"]
triplet_defect_state_plot(
    VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/sandwich_BN_mx2/db.json"),
    filter_labels,
    labels_in_plot,
    {"B-N-S-W":[367,368,369]},
    {"B-N-S-W":[None]})



#%% Trnasition levels figure
from matplotlib.ticker import AutoMinorLocator
from atomate.vasp.database import VaspCalcDb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

def transistion_levels(filter_label, tranl):
    colors = {
        "cbm": "orange",
        "vbm": "deepskyblue"
    }
    vbms_025 = []
    cbms_025 = []
    for chemsys in filter_label:
        filters = {
            "chemsys": chemsys,
            "task_label":"hse gap",
            "input.incar.AEXX": 0.25
        }
        e = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/"
                                    "mx2_antisite_basic_bandgap/db.json").collection.find_one(filters)
        vac = max(e["calcs_reversed"][0]["output"]["locpot"]["2"])
        cbms_025.append(e["output"]["cbm"] - vac)
        vbms_025.append(e["output"]["vbm"] - vac)
    cbms_025 = np.array(cbms_025)
    vbms_025 = np.array(vbms_025)

    bandgap = pd.DataFrame({"bandgap":dict(zip(filter_labels, cbms_025-vbms_025))})
    print(bandgap)
    x = np.arange(len(vbms_025))
    emin_025 = min(vbms_025)
    emin = emin_025-0.5

    # ppi = 100
    # figw = 800
    # figh = 800
    # fig = plt.figure(figsize=(figw / ppi, figh / ppi), dpi=ppi)
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(1, 1, 1)

    for trls, pos, vbm in zip(tranl, x, np.array(vbms_025)):
        text = ["$\epsilon(+/0)$", "$\epsilon(0/-)$"]
        if pos == x[-1]:# for MoTe2
            ax.text(pos-0.4, trls[0]+vbm-0.17, text[0], size=6.5)
            ax.text(pos-0.4, trls[1]+vbm+0.1, text[1], size=6.5)
            ax.hlines(trls[0]+vbm, pos-0.4, pos+0.4)
            ax.hlines(trls[1]+vbm, pos-0.4, pos+0.4)
        elif pos == x[2]:
            ax.text(pos-0.4, trls[0]+vbm-0.15, text[0], size=6.5)
            ax.text(pos-0.4, trls[1]+vbm-0.15, text[1], size=6.5)
            ax.hlines(trls[0]+vbm, pos-0.4, pos+0.4)
            ax.hlines(trls[1]+vbm, pos-0.4, pos+0.4)
        else:
            ax.text(pos-0.4, trls[0]+vbm+0.1, text[0], size=6.5)
            ax.text(pos-0.4, trls[1]+vbm+0.1, text[1], size=6.5)
            ax.hlines(trls[0]+vbm, pos-0.4, pos+0.4)
            ax.hlines(trls[1]+vbm, pos-0.4, pos+0.4)
            # ax.text(pos-0.35, trl+vbm+0.1, text + " {:.2f}".format(trl+vbm))

    # plot bars
    # ax.bar(x, np.array(vbms_c2db) - emin, bottom=emin, label="vbm_c2db")
    ax.bar(x, np.array(vbms_025) - emin, bottom=emin, label="vbm", color=colors["vbm"])
    ax.bar(x, -np.array(cbms_025), bottom=cbms_025, label="cbm", color=colors["cbm"])
    # ax.bar(x, -np.array(cbms_c2db), bottom=cbms_c2db, label="cbm_c2db")


    ax.set_ylim(-6.8, -3)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # ax.set_xticks(x)
    # ax.set_xticklabels(labels_in_plot, rotation=0, fontsize=10)
    ax.set_xticks([])


    #         ax.legend(loc="upper right")

    #         plt.title("Defect transition levels (eV)", fontsize=12)
    # plt.ylabel("$\epsilon(q/q')$ Transition levels (eV)", fontsize=20)
    # plt.tight_layout()
    plt.savefig(os.path.join("/Users/jeng-yuantsai/Research/qubit/plt", "mx2_tls_0.25.tempTe.eps"),
                                   img_format="eps")
    plt.show()



filter_labels = ["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"]
# labels_in_plot = ["${W_{S}}^0$", "${W_{Se}}^0$", "${Mo_{S}}^0$", "${Mo_{Se}}^0$"]
trls = [[2.4945-2.1, 1.82], [2.1977-1.43, 1.65], [1.6603-0.71, 1.43], [2.313-2.01, 1.31], [2.0595-1.93, 1.37], [1.6577-0.88, 0.97]]
transistion_levels(filter_labels, trls)



#%%
def ef_025():
    x1 = [0,0.689]
    y1 = [-4.925,-4.236]
    plt.plot(x1, y1, label = "charge state +1", color="purple")
    x2 = [0.689,1.119]
    y2 = [-4.236,-4.236]
    plt.plot(x2, y2, label = "charge state 0", color="purple")
    x3 = [1.119,2]
    y3 = [-4.236,-5.998]
    plt.plot(x3, y3, label="charge state -2", color="purple")
    plt.yticks([-6, -5, -4])
    plt.xticks([0, 0.69, 1.12, 2])
    plt.show()
    plt.savefig("/Users/jeng-yuantsai/Research/qubit/plt/wse2_Ef_0.25.eps", format="eps")


#%%
from glob import glob
import os
from pymatgen import Structure
import pandas as pd
def get_st_data(path):
    data = {}
    data1 = {}
    for s in glob(os.path.join(path, "*HSE*")):
        st = Structure.from_file(s)
        print(s)
        if "Te" in s:
            data1[s.split("/")[-1]] = {"M-M1":st.get_distance(74, 55), "M-M2":st.get_distance(74, 49),
                                      "M-M3":st.get_distance(74, 54), "M-X":st.get_distance(74, 24)}
        else:
            data1[s.split("/")[-1]] = {"M-M1":st.get_distance(0, 25), "M-M2":st.get_distance(6, 25),
                                      "M-M3":st.get_distance(5, 25), "M-X":st.get_distance(50, 25)}
            data[s.split("/")[-1]] = {"a": st.lattice.a, "b": st.lattice.b, "c": st.lattice.c}
            data = pd.DataFrame(data).round(3)
            data1 = pd.DataFrame(data1).round(3)
    return data, data1

lattice, bond = get_st_data('/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic/structures')
lattice35, bond35 = get_st_data('/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.35/structures')
print(lattice, bond)

#%%
from pymatgen import Structure
col = db_wse2_like.collection
data75 = {}
data108 = {}
for i in col.find({"task_id": {"$in":[1947, 1954, 1949, 1953, 1958, 1948, 1957, 1962, 1968]+
                                    [1877, 1881, 1900, 1889, 1886, 1909, 1876, 1895, 1903]+
                                     [2550, 2545, 2549]+[2547, 2551, 2562]}, "charge_state": -2}):
    st = Structure.from_dict(i["calcs_reversed"][0]["output"]["structure"])
    # st.to("poscar", "/Users/jeng-yuantsai/Research/qubit/calculations/WSe2_like_Ef_from_C2DB/structures/{}.{}.vasp".
    #       format(i["formula_pretty"], i["task_id"]))
    if i["nsites"] == 108:
        data108[i["calcs_reversed"][0]["output"]["structure"]["lattice"]["c"]] = {"M-M1":st.get_distance(7, 36), "M-M2":st.get_distance(0, 36),
                         "M-M3":st.get_distance(6, 36), "M-X":st.get_distance(72, 36)}
    elif i["nsites"] == 75:
        data75[i["calcs_reversed"][0]["output"]["structure"]["lattice"]["c"]] = {"M-M1":st.get_distance(0, 25), "M-M2":st.get_distance(6, 25),
                               "M-M3":st.get_distance(5, 25), "M-X":st.get_distance(50, 25)}

a = pd.DataFrame(data108).round(3)
b = pd.DataFrame(data75).round(3)
pd.concat([a,b], axis=0)

#%% PC
from pymatgen import Structure
import pandas as pd
col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_pc/db.json").collection
data = {}
data1 = {}
for i in col.find({
    "chemsys": {"$in":["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"]},
    "formula_anonymous": "AB2",
    "calcs_reversed.run_type": "HSE06",
    "nsites": 3,
    "input.parameters.AEXX": 0.25,
    "input.structure.lattice.c": {"$nin":[20]}
}):
    st = Structure.from_dict(i["calcs_reversed"][0]["output"]["structure"])

    # st.to("poscar", path+"{}.vasp".format(i["formula_pretty"]))
    data[i["chemsys"]] = {"a": round(st.lattice.a,3), "b": round(st.lattice.b,3),
                          "c": round(st.lattice.c,3), "gamma": round(st.lattice.gamma,3),
                          "space_gp": i["output"]["spacegroup"]["symbol"]
                          }
    if "Te" in i["chemsys"]:
        data1[i["chemsys"]] = {"x-x": round(st.get_distance(0, 1),3), "m-x": round(st.get_distance(2, 0),3),
                               "task_id":i["task_id"]}
    else:
        data1[i["chemsys"]] = {"x-x": round(st.get_distance(1, 2),3), "m-x": round(st.get_distance(0, 1),3),
                               "task_id":i["task_id"]}

data2 = pd.DataFrame(data)
data1 = pd.DataFrame(data1)
host = pd.concat([data2, data1], axis=0).reindex(columns=["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"])
host.to_clipboard()
print(host)


#%% Full Table
from atomate.vasp.database import VaspCalcDb
from pymatgen import Structure
import pandas as pd

col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.25_final/db.json")
# col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_perturb_pattern/db.json")
col = col.collection
perturb = {}
unperturb = {}
for i in col.find({
    "perturbed":{"$in":[0, 0.02, 1e-5]},
    "task_label":"HSE_scf",
    "nupdown_set":0
    #"task_id":{"$in":[3187, 3188, 3190, 3189, 3192, 3191]}
}):
    if i["perturbed"] == 1e-5:
        st = Structure.from_dict(i["calcs_reversed"][0]["output"]["structure"])
        if "Te" in i["chemsys"]:
            perturb[i["chemsys"]] = {"M-M1":st.get_distance(55, 74),
                                  "M-M2":st.get_distance(49, 74),
                                  "M-M3":st.get_distance(54, 74),
                                  "M-X":st.get_distance(24, 74),
                                  "energy":i["output"]["energy"]
                                  }
        else:
            perturb[i["chemsys"]] = {"M-M1":st.get_distance(0, 25),
                                  "M-M2":st.get_distance(6, 25),
                                  "M-M3":st.get_distance(5, 25),
                                  "M-X":st.get_distance(50, 25),
                                  "energy":i["output"]["energy"]
                                  }
    elif i["perturbed"] == 0:
        st = Structure.from_dict(i["calcs_reversed"][0]["output"]["structure"])
        if "Te" in i["chemsys"]:
            unperturb[i["chemsys"]] = {
                "M-M1":st.get_distance(55, 74),
                "M-M2":st.get_distance(49, 74),
                "M-M3":st.get_distance(54, 74),
                "M-X":st.get_distance(24, 74),
                "energy":i["output"]["energy"],
                "task_id": i["task_id"]
            }
        else:
            unperturb[i["chemsys"]] = {
                "M-M1":st.get_distance(0, 25),
                "M-M2":st.get_distance(6, 25),
                "M-M3":st.get_distance(5, 25),
                "M-X":st.get_distance(50, 25),
                "energy":i["output"]["energy"],
                "task_id": i["task_id"]
            }
perturb = pd.DataFrame(perturb)
unperturb = pd.DataFrame(unperturb)


perturb = perturb.reindex(columns=["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"]).round(3)
unperturb = unperturb.reindex(columns=["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"]).round(3)
unperturb.to_clipboard()
print(perturb)
print(unperturb)

#%%75
db = "mx2_antisite_basic_aexx0.25_final"
col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/{}/db.json".
                              format(db)).collection
chemsys = "Te-W"
task_label = "PBE_relax"
encut = [320]
if "Te" in chemsys:
    for i in col.find({"task_label": task_label, "input.incar.ENCUT":{"$in":encut}, "nsites":5*5*3, "chemsys":chemsys}):
        print("=="*5, i["perturbed"], round(i["output"]["energy"],3))
        init = Structure.from_dict(i["orig_inputs"]["poscar"]["structure"])
        final = Structure.from_dict(i["output"]["structure"])
        for site in [55, 49, 54, 24]:
            print("before:{}, {}, after:{}. {}".format(round(init.get_distance(74,site),3), round(init.lattice.a/5,3),
                                                       round(final.get_distance(74, site), 3), round(final.lattice.a/5,3)))


else:
    for i in col.find({"task_label": task_label, "input.incar.ENCUT":{"$in":encut}, "nsites":5*5*3, "chemsys":chemsys}):
        print("=="*5, i["perturbed"], round(i["output"]["energy"],3))
        init = Structure.from_dict(i["orig_inputs"]["poscar"]["structure"])
        final = Structure.from_dict(i["output"]["structure"])
        for site in [0, 6, 5, 50]:
            print("before:{}, {}, after:{}. {}".format(round(init.get_distance(25,site),3), round(init.lattice.a/5,3),
                                                       round(final.get_distance(25, site), 3), round(final.lattice.a/5,3)))
print(i["calcs_reversed"][0]["dir_name"])
print(i["task_id"])
print("encut:{}, ediffg:{}".format(i["input"]["incar"]["ENCUT"], i["input"]["incar"]["EDIFFG"]))
print(i["calcs_reversed"][0]["input"]["kpoints"]["kpoints"])
print(i["calcs_reversed"][0]["input"]["nkpoints"])



#%% 108
mx2_antisite_perturb_pattern = "mx2_antisite_perturb_pattern"
mx2_antisite_sp_aexx25 = "mx2_antisite_sp_aexx0.25"
col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/{}/db.json".
                              format(mx2_antisite_perturb_pattern))
col = col.collection

for i in col.find({"defect_pos":"side", "nsites":108}):
    print("=="*5, i["perturbed"], round(i["output"]["energy"],3))
    init = Structure.from_dict(i["orig_inputs"]["poscar"]["structure"])
    final = Structure.from_dict(i["output"]["structure"])
    c = 36
    for site in [0,6,7,72]:
        print("before:{}, {}, after:{}. {}".format(round(init.get_distance(c,site),3), round(init.lattice.a/5,3),
                                                   round(final.get_distance(c, site), 3), round(final.lattice.a/5,3)))
    print(i["dir_name"])
    print(i["task_id"])
    print(i["input"]["incar"]["ENCUT"])
    print(i["calcs_reversed"][0]["input"]["kpoints"]["kpoints"])
    print(i["calcs_reversed"][0]["input"]["nkpoints"])

#%% for antisite at center of supercell
col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_perturb_pattern/db.json")
col = col.collection

for i in col.find({"defect_pos":"center"}):
    print("=="*5, i["perturbed"], round(i["output"]["energy"],3))
    init = Structure.from_dict(i["orig_inputs"]["poscar"]["structure"])
    final = Structure.from_dict(i["output"]["structure"])
    for site in [17,18,12,62]:
        print("before:{}, after:{}".format(round(init.get_distance(25,site),3), round(final.get_distance(25, site), 3)))
    print(i["dir_name"])


#%% make structure
def get_rand_vec(distance): #was 0.001
    # deals with zero vectors.
    vector = np.random.randn(3)
    vnorm = np.linalg.norm(vector)
    return vector / vnorm * distance if vnorm != 0 else get_rand_vec(distance)

col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_pc/db.json").collection
wse2_HSE06_relax_host = col.find_one({"task_id":3150})
host_st = Structure.from_dict(wse2_HSE06_relax_host["output"]["structure"])
host_st.make_supercell([5,5,1])
host_st.to("poscar", "/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_pc/wse2_75.vasp")
host_st.replace(37, "W")
distort = 0.02
host_st.translate_sites(37, get_rand_vec(distort))
host_st.to("poscar", "/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_pc/w_se_75_{}.vasp".format(distort))

import subprocess
cmd = "scp /Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_pc/w_se_75_{}.vasp" \
      " tug03990@owlsnesttwo.hpc.temple.edu:/gpfs/work/tug03990/mx2_antisite_perturb_pattern/structures".format(distort)
subprocess.call(cmd.split(" "))


#%% symmetrize strucutre
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.analysis.structure_analyzer import RelaxationAnalyzer

mx2_antisite_basic_aexx25_new = "mx2_antisite_basic_aexx0.25_new"
col = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/{}/db.json".
                              format(mx2_antisite_basic_aexx25_new)).collection

init = Structure.from_dict(col.find_one({"task_id":3126})["output"]["structure"])
print(init.get_space_group_info())
sym = SpacegroupAnalyzer(init, 0.9)
final = sym.get_refined_structure()
# final.to("poscar","/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.25_new/refined_wse2.vaspl")
for site in [0, 6, 5, 50]:
    print("before:{}, {}, after:{}. {}".format(round(init.get_distance(25,site),3), round(init.lattice.a/5,3),
                                               round(final.get_distance(25, site), 3), round(final.lattice.a/5,3)))
asym = SpacegroupAnalyzer(final, 0.5)
sym = asym


#%% symmetrize structure but no relax
mx2_antisite_symmetrized_st_aexx25 = "mx2_antisite_symmetrized_st_aexx0.25"
mx2_antisite_c3v_mv_z = "mx2_antisite_c3v_mv_z"
sym = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/{}/db.json".
                              format(mx2_antisite_c3v_mv_z)).collection
mx2_antisite_basic_aexx25_new = "mx2_antisite_basic_aexx0.25_new"
asym = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/{}/db.json".
                               format(mx2_antisite_basic_aexx25_new)).collection

chemsys = "Se-W"
if "Te" in chemsys:
    entry = {}
    for s, a, o in zip(sym.find({"chemsys":chemsys, "task_label":"PBE_relax"}),
                    asym.find({"chemsys":chemsys, "task_label":"PBE_relax", "perturbed":0.02}),
                    asym.find({"chemsys":chemsys, "task_label":"PBE_relax", "perturbed":0})):
        print("=="*5, "s: p:{} id:{} | a: p:{} id:{}".format(s["perturbed"], s["task_id"], a["perturbed"], a["task_id"]))
        init = Structure.from_dict(s["output"]["structure"])
        final = Structure.from_dict(a["output"]["structure"])
        orig = Structure.from_dict(o["output"]["structure"])
        entry["symmetrize"] = dict(zip(["X1", "X2", "X3", "Z1"],
                                [init.get_distance(74,site) for site in [55, 49, 54, 24]]))
        entry["symmetrize"].update({"energy": s["output"]["energy"]})
        entry["symmetrize"].update({"task_id": s["task_id"]})

        entry["perturb"] = dict(zip(["X1", "X2", "X3", "Z1"],
                                 [final.get_distance(74,site) for site in [55, 49, 54, 24]]))
        entry["perturb"].update({"energy": a["output"]["energy"]})
        entry["perturb"].update({"task_id": a["task_id"]})

        entry["no_perturb"] = dict(zip(["X1", "X2", "X3", "Z1"],
                                 [orig.get_distance(74,site) for site in [55, 49, 54, 24]]))
        entry["no_perturb"].update({"energy": o["output"]["energy"]})
        entry["no_perturb"].update({"task_id": o["task_id"]})
else:
    entry = {}
    for s, a, o in zip(sym.find({"chemsys":chemsys, "task_label":"PBE_relax"}),
                    asym.find({"chemsys":chemsys, "task_label":"PBE_relax", "perturbed":0.02, "task_id":{"$nin":[3156]}}),
                    asym.find({"chemsys":chemsys, "task_label":"PBE_relax", "perturbed":0, "task_id":{"$nin":[3218]}})):
        print("=="*5, "s: p:{} id:{} | a: p:{} id:{}".format(s["perturbed"], s["task_id"], a["perturbed"], a["task_id"]))
        init = Structure.from_dict(s["output"]["structure"])
        final = Structure.from_dict(a["output"]["structure"])
        orig = Structure.from_dict(o["output"]["structure"])
        entry["symmetrize"] = dict(zip(["X1", "X2", "X3", "Z1"],
                                [init.get_distance(25,site) for site in [0, 6, 5, 50]]))
        entry["symmetrize"].update({"energy": s["output"]["energy"]})
        entry["symmetrize"].update({"task_id": s["task_id"]})

        entry["perturb"] = dict(zip(["X1", "X2", "X3", "Z1"],
                                 [final.get_distance(25,site) for site in [0, 6, 5, 50]]))
        entry["perturb"].update({"energy": a["output"]["energy"]})
        entry["perturb"].update({"task_id": a["task_id"]})
        entry["no_perturb"] = dict(zip(["X1", "X2", "X3", "Z1"],
                                 [orig.get_distance(25,site) for site in [0, 6, 5, 50]]))
        entry["no_perturb"].update({"energy": o["output"]["energy"]})
        entry["no_perturb"].update({"task_id": o["task_id"]})


pd.DataFrame(entry).round(3)

#%% Band Gap
db = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_bandgap/db.json")
data = {}
for e in db.collection.find({"task_label": "hse gap"}):
    data[e["chemsys"]] = {
        "bandgap": e["output"]["bandgap"],
        "direct": e["output"]["is_gap_direct"],
        "task_id": e["task_id"],
        "db": "mx2_antisite_basic_bandgap"
    }
# WSe2 doesnt have band gap with AEXX = 0.25 (only 0.3 and 0.35 exist)
df = pd.DataFrame(data)
df = df[["S-W", "Se-W", "Te-W", "Mo-S", "Mo-Se", "Mo-Te"]]
df.round(3)

#%% Mv_z
from atomate.vasp.database import VaspCalcDb
from pymatgen.io.vasp.inputs import Structure
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

class MovingZSplitting:
    @classmethod
    def sheet(cls, chemsys):
        db = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_splitting/db.json")
        mx2_antisite_mv_z = db.collection

        orig_db = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.25_final/db.json")


        # wse2_orig = Structure.from_dict(mx2_antisite_mv_z.find_one({"task_id":2646})["output"]["structure"])
        p = "/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_splitting/"
        # e_orig = orig_db.collection.find_one({"chemsys":chemsys, "task_label":"HSE_scf", "nupdown_set":2})
        # orig = Structure.from_dict(e_orig["output"]["structrue"])
        # dz = [orig[site].c for site in e_orig["NN"][:-1]]
        # if "Te" in chemsys:
        #     d0 = round(orig.get_distance(74, 24),3)
        sheet = []
        for e in list(mx2_antisite_mv_z.find({"chemsys":chemsys, "task_label":"HSE_scf", "lattice_constant":{"$nin":["PBE"]}}))+\
                 list(orig_db.collection.find({"chemsys":chemsys, "task_label":"HSE_scf", "nupdown_set":2})):
            print(e["task_id"])
            try:
                eig = db.get_eigenvals(e["task_id"])
            except TypeError:
                eig = orig_db.get_eigenvals(e["task_id"])
            entry = {}
            st = Structure.from_dict(e["output"]["structure"])
            avg_z = np.average([st[site].z for site in e["NN"][:-1]])
            entry["perturb"] = e["perturbed"]
            entry["formula_pretty"] = e["formula_pretty"]
            entry["bond_length_diff"] = round(st[e["NN"][-1]].z-avg_z,3)
            entry["energy"] = round(e["output"]["energy"],3)
            if "Mo" in chemsys:
                entry["∆E02"] = round(eig["1"][0][304][0]-eig["1"][0][302][0],3)
                entry["∆E12"] = round((eig["1"][0][304][0]-eig["1"][0][303][0]),3 )#/entry["∆E02"],3)
                entry["∆E01"] = round((eig["1"][0][303][0]-eig["1"][0][302][0]), 3)#/entry["∆E02"],3)
            else:
                entry["∆E02"] = round(eig["1"][0][226][0]-eig["1"][0][224][0],3)
                entry["∆E12"] = round((eig["1"][0][226][0]-eig["1"][0][225][0]), 3)#/entry["∆E02"],3)
                entry["∆E01"] = round((eig["1"][0][225][0]-eig["1"][0][224][0]),3) #/entry["∆E02"],3)
            entry["task_id"] = e["task_id"]
            sheet.append(entry)
        df = pd.DataFrame(sheet).set_index("task_id")
        df.sort_values(["bond_length_diff"], inplace=True)
        fig, ax = plt.subplots(3, sharex=True, figsize=(10,8))
        fig.suptitle(df["formula_pretty"].iloc[0])
        ax[0].plot(df["bond_length_diff"], df["∆E02"], "o", color="k", label="∆E02")
        ax[0].set(ylabel="∆E02 (eV)", xlim=[-0.6, 0.6])
        ax[1].plot(df["bond_length_diff"], df["∆E01"], "o", color="blue", label="∆E01")
        ax[1].plot(df["bond_length_diff"], df["∆E12"], "o", color="red", label="∆E12")
        ax[1].set(ylabel="∆E/∆E02", xlim=[-0.6, 0.6])
        ax[2].plot(df["bond_length_diff"], df["energy"], "o", color="k", label="Etot")
        ax[2].set(xlabel="displacement from initial z-position (Å)", ylabel="total energy (eV)", xlim=[-0.6, 0.6])
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        os.makedirs(os.path.join(p, chemsys, "plt"), exist_ok=True)
        fig.savefig(os.path.join(p, chemsys, "plt", "{}.jpg".format(df["formula_pretty"].iloc[0])), dpi=160, format="jpg")
        return df

    @classmethod
    def plot(cls):
        colors = ["powderblue", "deepskyblue", "red", "orange"]
        # colors = ["lightgreen", "limegreen", "red"]
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(1, 1, 1)

        c = [3.15, 3.28, 3.51]
        # c = [3.15, 3.28]

        w = 0.03
        maxy, miny = 1.49+1, 1.671-0.6


        mos2 = [1.48, 1.58, 1.68, 1.78, 1.88, 1.98, 2.08, 2.18, 2.28, 2.38, 2.48]
        ax.bar(c[0]-w/2*1.2,  miny-1.49, w, 1.49, color=colors[0], label="1-2 configuration")
        ax.bar(c[0]-w/2*1.2, maxy-1.49, w, 1.49, color=colors[1], label="2-1 configuration")
        ax.hlines(1.98, c[0]-w/2*1.2-w/2, c[0]-w/2*1.2+w/2, colors[-2])
        ax.hlines(1.49, c[0]-w/2*1.2-w/2, c[0]-w/2*1.2+w/2, colors[-1])

        # ax.plot([c[0]-w/2*1.2, c[0]+w/2*1.2], [0.8, 1.49-0.4], color="silver", linestyle="dashed")

        ws2 = [1.405, 1.505, 1.605, 1.705, 1.805, 1.905, 2.005, 2.105, 2.205, 2.305, 2.405]
        ax.bar(c[0]+w/2*1.2, miny-1.505, w, 1.505, color=colors[0])
        ax.bar(c[0]+w/2*1.2, maxy-1.505, w, 1.505, color=colors[1])
        ax.hlines(1.905, c[0]+w/2*1.2-w/2, c[0]+w/2*1.2+w/2, colors[-2])
        ax.hlines(1.505, c[0]+w/2*1.2-w/2, c[0]+w/2*1.2+w/2, colors[-1])

        # ax.plot([1.125, x2], [0.8, 1.505-0.5], color="silver", linestyle="dashed")

        # ax.plot([c[0]-w/2*1.2+w/2, c[0]+w/2*1.2-w/2], [1.98, 1.905], color="black", linestyle="dashed")
        # ax.plot([c[0]+w/2*1.2+w/2, c[1]-w/2*1.2-w/2], [1.905, 1.924], color="black", linestyle="dashed")
        # ax.plot([c[0]-w/2*1.2+w/2, c[0]+w/2*1.2-w/2], [1.49, 1.505], color="black", linestyle="dashed")
        # ax.plot([c[0]+w/2*1.2+w/2, c[1]-w/2*1.2-w/2], [1.505, 1.524], color="black", linestyle="dashed")

        mose2 = [1.424, 1.524, 1.624, 1.724, 1.824, 1.924, 2.024, 2.124, 2.224, 2.324, 2.424]
        ax.bar(c[1]-w/2*1.2, miny-1.524, w, 1.524, color=colors[0])
        ax.bar(c[1]-w/2*1.2, maxy-1.524, w, 1.524, color=colors[1])
        ax.hlines(1.924, c[1]-w/2*1.2-w/2, c[1]-w/2*1.2+w/2, colors[-2])
        ax.hlines(1.524, c[1]-w/2*1.2-w/2, c[1]-w/2*1.2+w/2, colors[-1])

        # ax.plot([x1+0.1, x1], [0.8, 1.49-0.4], color="silver", linestyle="dashed")

        wse2 = [1.336, 1.436, 1.536, 1.636, 1.736, 1.836, 1.936, 2.036, 2.136, 2.236, 2.336]
        ax.bar(c[1]+w/2*1.2, miny-1.636, w, 1.636, color=colors[0])
        ax.bar(c[1]+w/2*1.2, maxy-1.636, w, 1.636, color=colors[1])
        ax.hlines(1.836, c[1]+w/2*1.2-w/2, c[1]+w/2*1.2+w/2, colors[-2])
        ax.hlines(1.636, c[1]+w/2*1.2-w/2, c[1]+w/2*1.2+w/2, colors[-1])
        # ax.plot([1.125, x2], [0.8, 1.505-0.5], color="silver", linestyle="dashed")

        # ax.plot([c[1]-w/2*1.2+w/2, c[1]+w/2*1.2-w/2], [1.924, 1.836], color="black", linestyle="dashed")
        # ax.plot([c[1]+w/2*1.2+w/2, c[2]-w/2*1.2-w/2], [1.836, 1.402], color="black", linestyle="dashed")
        # ax.plot([c[1]-w/2*1.2+w/2, c[1]+w/2*1.2-w/2], [1.524, 1.636], color="black", linestyle="dashed")
        # ax.plot([c[1]+w/2*1.2+w/2, c[2]-w/2*1.2-w/2], [1.636, 1.902], color="black", linestyle="dashed")

        mote2 = [1.402, 1.502, 1.602, 1.702, 1.802, 1.902]
        ax.bar(c[2]-w/2*1.2, miny-1.902, w, 1.902, color=colors[0])
        ax.bar(c[2]-w/2*1.2, maxy-1.902, w, 1.902, color=colors[1])
        ax.hlines(1.402, c[2]-w/2*1.2-w/2, c[2]-w/2*1.2+w/2, colors[-2])
        ax.hlines(1.902, c[2]-w/2*1.2-w/2, c[2]-w/2*1.2+w/2, colors[-1])
        # ax.plot([2.625, x1], [0.8, 1.902-0.9], color="silver", linestyle="dashed")


        wte2 = [1.271, 1.371, 1.471, 1.571, 1.671, 1.771]
        ax.bar(c[2]+w/2*1.2, miny-1.671, w, 1.671, color=colors[0])
        ax.bar(c[2]+w/2*1.2, maxy-1.671, w, 1.671, color=colors[1])
        ax.hlines(1.271, c[2]+w/2*1.2-w/2, c[2]+w/2*1.2+w/2, colors[-2])
        ax.hlines(1.671, c[2]+w/2*1.2-w/2, c[2]+w/2*1.2+w/2, colors[-1])
        # ax.plot([2.625, x2], [0.8, 1.671-0.6], color="silver", linestyle="dashed")
        # ax.plot([c[2]-w/2*1.2+w/2, c[2]+w/2*1.2-w/2], [1.402, 1.271], color="black", linestyle="dashed")
        # ax.plot([c[2]-w/2*1.2+w/2, c[2]+w/2*1.2-w/2], [1.902, 1.671], color="black", linestyle="dashed")


        # pos = [1, 1.25, 1.75, 2.0, 2.5, 2.75]
        # ax.plot(pos, [1.98, 1.905, 1.924, 1.836, 1.402, 1.271], color="orangered", marker=".", linestyle="dotted")
        # ax.plot(pos, [1.49, 1.505, 1.524, 1.636, 1.902, 1.671], color="royalblue", marker=".", linestyle="dotted")


        # labels = ["${Mo_{S}}^0$", "${W_{S}}^0$",  "${Mo_{Se}}^0$", "${W_{Se}}^0$", "${Mo_{Te}}^0$", "${W_{Te}}^0$"]
        labels = ["3.15", "3.28", "3.51"]
        x1 = 0.25
        # c = []
        # for i in range(3):
        #     x1 += 0.75
        #     x2 = x1+0.25
        #     c.append(x1)
        #     c.append(x2)
        ax.set_xticks(c)
        # ax.set_xticklabels(labels)
        # ax.set_ylabel("Antisite position along z direction (Å)")
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_ylim([1, 2.6])
        # ax.legend()

        fig.show()
        fig.savefig('/Users/jeng-yuantsai/Research/qubit/plt/defect_state_configuration.eps', format="eps")

MovingZSplitting.plot()

#%%
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(212)
ax.bar(1, 2, 0.4, 1)
ax1.bar(2, 4, 0.4, 1)
ax.text(1, 3,"x")

#%%
from atomate.vasp.database import VaspCalcDb
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp.inputs import Element

db = VaspCalcDb.from_db_file('/Users/jeng-yuantsai/Research/qubit/calculations/sandwich_BN_mx2/db.json')
dos = db.get_dos(3951)
dos_plt = DosPlotter(stack=False, sigma=0.05)
# tdos = DosPlotter(stack=True, zero_at_efermi=False)
# tdos.add_dos("tdos", dos)
# fig2 = tdos.get_plot(xlim=[-5,5])
# fig2.show()
dos_plt.add_dos("Total DOS", dos)
for element, dos in dos.get_element_dos().items():
    if element == Element("B"):
        dos_plt.add_dos("{}".format(element), dos)
    elif element == Element("N"):
        dos_plt.add_dos("{}".format(element), dos)
    # elif element == Element("W"):
    #     dos_plt.add_dos("{}".format(element), dos)
    # elif element == Element("S"):
fig = dos_plt.get_plot(xlim=[-5,5])

fig.savefig("/Users/jeng-yuantsai/Research/qubit/plt/sandwich_ws2_bn.eps", format="eps")
fig.show()

#%%
from pymatgen import Structure
from atomate.vasp.database import VaspCalcDb
import pandas as pd
import numpy as np

db = VaspCalcDb.from_db_file('/Users/jeng-yuantsai/Research/qubit/calculations/antisiteQubit/MxC3vToChDeltaE/db.json')
col = db.collection

data = [
    {
        "task_id": e["task_id"],
        "chemsys": e["chemsys"],
        "perturbed":e["perturbed"],
        "energy":e["output"]["energy"],
        "dx":np.array([round(Structure.from_dict(e["input"]["structure"]).get_distance(e["NN"][-1], i),3)
                       for i in e["NN"][:-1]+[24 if "Te" in e['chemsys'] else 50]])
    }
    for e in col.find({"task_label":"HSE_scf"})]
raw = pd.DataFrame(data).sort_values(["chemsys", "perturbed"])
print(raw)

d = []
for chem in ["Mo-S", "S-W", "Mo-Se", "Se-W", "Mo-Te", "Te-W"]:
    a = raw.loc[raw["chemsys"]==chem]["energy"].diff().iloc[-1]
    b = raw.loc[raw["chemsys"]==chem]["dx"].diff().iloc[-1]
    e = {"chemsys": chem, "dE":round(a,3), "dx":np.linalg.norm(b).round(3)}
    d.append(e)
pd.DataFrame(d).set_index("chemsys")

#%%
from pymatgen import Structure
from atomate.vasp.database import VaspCalcDb
import pandas as pd
import numpy as np

db = VaspCalcDb.from_db_file('/Users/jeng-yuantsai/Research/qubit/calculations/antisiteQubit/scan_opt_test/db.json')
col = db.collection

data = [
    {
        "formula": e["formula_pretty"],
        "origin_a": e["input"]["structure"]["lattice"]["a"],
        "a":e["output"]["structure"]["lattice"]["a"],
        "da": (e["input"]["structure"]["lattice"]["a"] - e["output"]["structure"]["lattice"]["a"])/
              e["input"]["structure"]["lattice"]["a"],
        "origin_b": e["input"]["structure"]["lattice"]["b"],
        "b":e["output"]["structure"]["lattice"]["b"],
        "db": (e["input"]["structure"]["lattice"]["b"] - e["output"]["structure"]["lattice"]["b"])/
              e["input"]["structure"]["lattice"]["b"],
        "origin_c": e["input"]["structure"]["lattice"]["c"],
        "c":e["output"]["structure"]["lattice"]["c"],
        "dc": (e["input"]["structure"]["lattice"]["c"] - e["output"]["structure"]["lattice"]["c"])/
              e["input"]["structure"]["lattice"]["c"]
    }
    for e in col.find()]
raw = pd.DataFrame(data).round(3).sort_values(["formula"]).to_clipboard()
print(raw)

