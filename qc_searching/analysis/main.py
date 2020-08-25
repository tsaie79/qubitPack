from qc_searching.analysis.dos_plot_from_db import DosPlotDB
from qc_searching.analysis.read_eigen import DetermineDefectState
import matplotlib.pyplot as plt
from qc_searching.analysis.dos_plot_from_db import DB_CONFIG_PATH
from glob import glob
import os
import numpy as np


def main(db, db_filter, cbm, vbm, path_save_fig, plot=True, clipboard="tot"):
    """
    When one is using "db_cori_tasks_local", one must set ssh-tunnel as following:
    "ssh -f tsaie79@cori.nersc.gov -L 2222:mongodb07.nersc.gov:27017 -N mongo -u 2DmaterialQuantumComputing_admin -p
    tsaie79 localhost:2222/2DmaterialQuantumComputing"
    """
    can = DetermineDefectState(db=db, db_filter=db_filter, cbm=cbm, vbm=vbm, show_edges="band_edges",
                               save_fig_path=path_save_fig,  locpot=False)

    print(can.efermi)
    tot, proj = can.get_candidates(
        [1, 1],
        0,
        threshold=0.01,
        select_up=None,
        select_dn=None
    )
    if clipboard == "tot":
        tot.to_clipboard("\t")
    else:
        proj.to_clipboard("\t")

    if plot:
        dos_plot = DosPlotDB(db=db, db_filter=db_filter, cbm=cbm, vbm=vbm, path_save_fig=path_save_fig)
        dos_plot.sites_plots(energy_upper_bound=4, energy_lower_bound=4)
        print(dos_plot.nn)
        dos_plot.orbital_plot(dos_plot.nn[-1], 4, 4)
        plt.show()


if __name__ == '__main__':
    proj_path = "/Users/jeng-yuantsai/Research/qubit/calculations/mx2_antisite_basic_aexx0.25_final"
    save_path = os.path.join(proj_path)
    for dir_name in ["defect_states", "structures", "xlsx"]:
        os.makedirs(os.path.join(save_path, dir_name), exist_ok=True)
    # path = '/Users/jeng-yuantsai/Research/qubit/My_manuscript/mx2_antisite_basic/defect_states'
    # db_json = "/Users/jeng-yuantsai/Research/qubit/My_manuscript/WSe_2/support/c2db_TMDC_search"
    db_json = os.path.join(proj_path, "db.json")
    # db_json = '/Users/jeng-yuantsai/Research/qubit/My_manuscript/mx2_antisite_basic/db.json'
    # p1 = os.path.join(db_json, "db_WSe2_like_Ef_from_C2DB.json")
    # p2 = os.path.join(db_json, "db_c2db_tmdc_bglg1.json")

    main(
        db_json,
        {"task_id": 3292},
        4, -2,
        save_path,
        True,
        "tot"
    )

