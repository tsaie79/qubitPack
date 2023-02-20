import numpy as np
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure.core import Orbital, OrbitalType
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen import Structure
import os
from collections import defaultdict
from atomate.vasp.database import VaspCalcDb
from glob import glob
import subprocess
import plotly.tools as tls
import plotly
from matplotlib import pyplot as plt

FIG_SAVING_FILE_PATH = "/Users/jeng-yuantsai/quantumComputing/WSe2_doping/plot"
DB_CONFIG_PATH = "/Users/jeng-yuantsai/PycharmProjects/git/my_pycharm_projects/database/db_config/formation_energy_related"
DB_CONFIG_LOCAL = defaultdict(str)
for db_config in glob(DB_CONFIG_PATH+"/*"):
    DB_CONFIG_LOCAL[os.path.basename(db_config)] = db_config


class DosPlotDB:

    def __init__(self, db, db_filter, cbm, vbm, efermi, path_save_fig, sigma, vacuum_level, mark_vbm_cbm, mark_efermi):
        self.path_save_fig = path_save_fig
        self.e = db.collection.find_one(filter=db_filter)
        if self.path_save_fig:
            Structure.from_dict(self.e["output"]["structure"]).to(
                "POSCAR",
                os.path.join(
                    self.path_save_fig,
                    "structures", "{}_{}_{}.vasp".format(
                        self.e["formula_pretty"],
                        self.e["task_id"],
                        self.e["task_label"]
                    )
                )
            )
        print("DOS="*20)
        print(f"\nDOS activated!:{self.e['task_id']}")
        self.complete_dos1 = db.get_dos(self.e["task_id"])
        if vacuum_level:
            self.complete_dos1 = self.__align_dos_with_vacuum(vacuum_level)

        self.cbm_primitive = cbm
        self.vbm_primitive = vbm
        self.efermi = efermi
        self.cation = self.complete_dos1.structure.types_of_specie[0]
        self.anion = self.complete_dos1.structure.types_of_specie[1]
        self.nn = self.e["NN"]
        self.sigma = sigma
        self.mark_vbm_cbm = mark_vbm_cbm
        self.mark_efermi = mark_efermi

    def __align_dos_with_vacuum(self, vacuum_level):
        """
        Align the eigenvalues with vacuum level
        :param vacuum_level: vacuum level
        :return: aligned eigenvalues
        """
        dos = self.complete_dos1.as_dict()
        efemri = dos["efermi"]
        energies = dos["energies"]
        aligned_efemri = efemri - vacuum_level
        aligned_energies = np.array(energies) - vacuum_level
        # create a new dos object
        dos["efermi"] = aligned_efemri
        dos["energies"] = aligned_energies
        return CompleteDos.from_dict(dos)

    def band_gap(self):
        """
        If it returns null, the material could be METAL
        :return: band gap, cbm, kpoint(cbm), vbm, kpoint(vbm), efermi
        """
        fer = self.complete_dos0.efermi
        print("cbm: %s, vbm: %s, efermi: %s" % (str(self.cbm_primitive), str(self.vbm_primitive), str(fer)))
        # fermi = self.vp.efermi
        # print("bandgap=%.4f\ncbm=%.4f\nvbm=%.4f\nfermi=%.4f\n" % (bandgap, cbm, vbm, fermi))
        # return bandgap, cbm, vbm, fermi

    def total_dos(self, energy_upper_bound, energy_lower_bound):
        # 2. plot dos
        plotter = DosPlotter(zero_at_efermi=False, stack=False, sigma=self.sigma)
        # os.makedirs(self.name + "_figures/%s/" % "total", exist_ok=True)
        # 2.1 total dos
        # print(self.complete_dos1.get_site_spd_dos(self.complete_dos1.structure.sites[0])[OrbitalType.s])
        plotter.add_dos("tdos", self.complete_dos1, line_style="-")
        tdos_plt = plotter.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, 
                                           self.cbm_primitive+energy_upper_bound])

        # fig = plt.gcf()
        # fig.set_size_inches(18.5, 13.5, forward=True)
        tdos_plt.legend(loc=1)
        if self.mark_vbm_cbm:
            tdos_plt.axvline(x=self.cbm_primitive, color="k", linestyle="--")
            tdos_plt.axvline(x=self.vbm_primitive, color="k", linestyle="--")
        if self.mark_efermi:
            tdos_plt.axvline(x=self.efermi, color="k", linestyle="-.")
        tdos_plt.legend(loc=1)
        tdos_plt.title(self.e["formula_pretty"]+" total_dos")

        if self.path_save_fig:
            tdos_plt.savefig(os.path.join(self.path_save_fig, "defect_states", "{}_{}_{}.tdos.png".format(
                self.e["formula_pretty"],
                self.e["task_id"],
                self.e["task_label"],
            )), img_format="png")

        return tdos_plt

    def orbital_plot(self, index, energy_upper_bound, energy_lower_bound):

        plotter = DosPlotter(zero_at_efermi=False, stack=True, sigma=self.sigma)
        print("==" * 50)
        print(self.complete_dos1.structure.sites[index])
        projection = dict(zip(["s", "y", "z", "x", "xy", "yz", "z2", "xz", "x2-y2"],
                              [self.complete_dos1.get_site_orbital_dos(
                                  self.complete_dos1.structure.sites[index], Orbital(j)) for j in range(0, 9)]))
        plotter.add_dos_dict(projection)
        # plotter.add_dos_dict({"tdos": self.complete_dos1})
        orbital_plt = plotter.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, 
                                           self.cbm_primitive+energy_upper_bound])
        fig = plt.gcf()
        fig.set_size_inches(18.5, 13.5, forward=True)
        if self.mark_vbm_cbm:
            orbital_plt.axvline(x=self.cbm_primitive, color="k", linestyle="--")
            orbital_plt.axvline(x=self.vbm_primitive, color="k", linestyle="--")
        if self.mark_efermi:
            orbital_plt.axvline(x=self.efermi, color="k", linestyle="-.")

        orbital_plt.legend(loc=1)
        orbital_plt.title(self.e["formula_pretty"] + " site%d %s" % (index, self.complete_dos1.structure.sites[index].specie))

        if self.path_save_fig:
            orbital_plt.savefig(os.path.join(self.path_save_fig, "defect_states", "{}_{}.{}_idx{}.orbital_dos.png".format(
                self.e["formula_pretty"],
                self.e["task_id"],
                self.e["task_label"],
                index
            )), img_format="png")


        # plot.savefig(self.name + "_figures/%s/" % self.fig_name + self.name + "_%s_%d.eps" % (
        #     self.complete_dos.structure.sites[index].specie, index), img_format="eps")
        return orbital_plt

    def sites_plots(self, energy_upper_bound, energy_lower_bound):
        title = self.e["formula_pretty"]
        plotter = DosPlotter(zero_at_efermi=False, stack=False, sigma=self.sigma)
        for i in self.nn:
            plotter.add_dos(str(i), self.complete_dos1.get_site_dos(self.complete_dos1.structure[i]))
        # plotter.add_dos("total_dos", self.complete_dos1)
        site_dos_plt = plotter.get_plot(xlim=[self.vbm_primitive-energy_lower_bound,
                                       self.cbm_primitive+energy_upper_bound])

        site_dos_plt.legend(loc=1)
        if self.mark_vbm_cbm:
            site_dos_plt.axvline(x=self.cbm_primitive, color="k", linestyle="--")
            site_dos_plt.axvline(x=self.vbm_primitive, color="k", linestyle="--")
        if self.mark_efermi:
            site_dos_plt.axvline(x=self.efermi, color="k", linestyle="-.")

        site_dos_plt.title(title+" site PDOS"+" Charge state:%d" % self.e["charge_state"])
        if self.path_save_fig:
            site_dos_plt.savefig(os.path.join(self.path_save_fig, "defect_states", "{}_{}_{}.pdos.png".format(
                title,
                self.e["task_id"],
                self.e["task_label"])), img_format="png")
        return site_dos_plt

    def spd_plots(self, energy_upper_bound, energy_lower_bound):
        title = self.e["formula_pretty"]
        plotter = DosPlotter(zero_at_efermi=False, sigma=self.sigma)
        dos_dict = defaultdict(object)
        for site in self.e["NN"]:
            for i in [0,1,2]:
                dos_dict[
                    self.complete_dos1.structure.as_dict()["sites"][site]["label"]+str(site)+str(OrbitalType(i))
                    ] = \
                    self.complete_dos1.get_site_spd_dos(
                        self.complete_dos1.structure.sites[site]
                    )[OrbitalType(i)]
        dos_dict["total_dos"] = self.complete_dos1
        plotter.add_dos_dict(dos_dict)
        # plot = dosplot.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, self.cbm_primitive+energy_upper_bound])
        spd_plt = plotter.get_plot(xlim=[self.vbm_primitive-energy_lower_bound,
                                         self.cbm_primitive+energy_upper_bound])
        # fig = plt.gcf()
        # fig.set_size_inches(18.5, 13.5, forward=True)
        spd_plt.legend()
        if self.mark_vbm_cbm:
            spd_plt.axvline(x=self.cbm_primitive, color="k", linestyle="--")
            spd_plt.axvline(x=self.vbm_primitive, color="k", linestyle="--")
        if self.mark_efermi:
            spd_plt.axvline(x=self.efermi, color="k", linestyle="-.")
        
        return spd_plt
# def main():
#     d = DosPlotDB(db1="db_w1te2_040752_local", db2=None, db0=None,
#                   defect_type="cation", name="test", icsd_id="040752",
#                   target_compound=dict(task_id=144))
#     # d.sites_plots(False)
#     d.cbm_primitive = -0.78
#     d.vbm_primitive = -2.68
#     d.orbital_plot(36)
#     # d.sites_plots(energy_upper_bound=1, energy_lower_bound=1)
#     # d.total_dos()


if __name__ == "__main__":
    pass