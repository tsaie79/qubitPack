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

    def __init__(self, db, db_filter, cbm, vbm, efermi, path_save_fig):
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

        print("dos_1:%s" % self.e["task_id"])
        self.complete_dos1 = db.get_dos(self.e["task_id"])
        self.cbm_primitive = cbm
        self.vbm_primitive = vbm
        self.efermi = efermi
        self.cation = self.complete_dos1.structure.types_of_specie[0]
        self.anion = self.complete_dos1.structure.types_of_specie[1]
        self.nn = self.e["NN"]


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
        plotter = DosPlotter(zero_at_efermi=False, stack=False, sigma=0)
        # os.makedirs(self.name + "_figures/%s/" % "total", exist_ok=True)
        # 2.1 total dos
        # print(self.complete_dos1.get_site_spd_dos(self.complete_dos1.structure.sites[0])[OrbitalType.s])
        plotter.add_dos("tdos", self.complete_dos1, line_style="-")
        plot = plotter.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, self.cbm_primitive+energy_upper_bound])

        # plotly_plot = tls.mpl_to_plotly(fig)
        # plotly.offline.plot(plotly_plot, filename=os.path.join("/home/jengyuantsai/PycharmProjects/qc_searching/analysis/test_figures","procar.html"))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 13.5, forward=True)
        plot.legend(loc=1)
        plot.axvline(x=self.cbm_primitive, color="k", linestyle="--")
        plot.axvline(x=self.vbm_primitive, color="k", linestyle="--")
        plot.axvline(x=self.efermi+0.125, color="k", linestyle="-.")
        plot.legend(loc=1)
        plot.title(self.e["formula_pretty"]+" total_dos")

        if self.path_save_fig:
            plot.savefig(os.path.join(self.path_save_fig, "defect_states", "{}_{}_{}.tdos.png".format(
                self.e["formula_pretty"],
                self.e["task_id"],
                self.e["task_label"],
            )), img_format="png")
        plot.show()

        return plot

    def orbital_plot(self, index, energy_upper_bound, energy_lower_bound):

        plotter = DosPlotter(zero_at_efermi=False, stack=True, sigma=0)
        print("==" * 50)
        print(self.complete_dos1.structure.sites[index])
        projection = dict(zip(["s", "y", "z", "x", "xy", "yz", "z2", "xz", "x2-y2"],
                              [self.complete_dos1.get_site_orbital_dos(
                                  self.complete_dos1.structure.sites[index], Orbital(j)) for j in range(0, 9)]))
        plotter.add_dos_dict(projection)
        # plotter.add_dos_dict({"tdos": self.complete_dos1})
        plot = plotter.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, self.cbm_primitive+energy_upper_bound])
        fig = plt.gcf()
        fig.set_size_inches(18.5, 13.5, forward=True)
        plot.axvline(x=self.cbm_primitive, color="k", linestyle="--")
        plot.axvline(x=self.vbm_primitive, color="k", linestyle="--")
        plot.axvline(x=self.efermi+0.125, color="k", linestyle="-.")
        plot.legend(loc=1)
        plot.title(self.e["formula_pretty"] + " site%d %s" % (index, self.complete_dos1.structure.sites[index].specie))

        if self.path_save_fig:
            plot.savefig(os.path.join(self.path_save_fig, "defect_states", "{}_{}.{}_idx{}.orbital_dos.png".format(
                self.e["formula_pretty"],
                self.e["task_id"],
                self.e["task_label"],
                index
            )), img_format="png")

        plot.show()

        # plot.savefig(self.name + "_figures/%s/" % self.fig_name + self.name + "_%s_%d.eps" % (
        #     self.complete_dos.structure.sites[index].specie, index), img_format="eps")
        return plot

    def sites_plots(self, energy_upper_bound, energy_lower_bound, spd=False):
        title = self.e["formula_pretty"]

        def sites_total_plots():
            dosplot = DosPlotter(zero_at_efermi=False, stack=True)
            for i in self.nn:
                dosplot.add_dos(str(i), self.complete_dos1.get_site_dos(self.complete_dos1.structure[i]))
            dosplot.add_dos("total_dos", self.complete_dos1)
            # plot = dosplot.get_plot(xlim=[-2.5, 2.5])
            plot = dosplot.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, self.cbm_primitive+energy_upper_bound])
            fig = plt.gcf()
            fig.set_size_inches(18.5, 13.5, forward=True)
            plot.legend(loc=1)
            plot.axvline(x=self.cbm_primitive, color="k", linestyle="--")
            plot.axvline(x=self.vbm_primitive, color="k", linestyle="--")
            plot.axvline(x=self.efermi+0.125, color="k", linestyle="-.")
            # fig = plot.figure()
            # plotly_plot = tls.mpl_to_plotly(fig)
            # plotly.offline.plot(plotly_plot, filename=os.path.join("procar.html"))
            plot.title(title+" site PDOS"+" Charge state:%d" % self.e["charge_state"])
            if self.path_save_fig:
                plot.savefig(os.path.join(self.path_save_fig, "defect_states", "{}_{}_{}.pdos.png".format(
                    title,
                    self.e["task_id"],
                    self.e["task_label"])), img_format="png")
            plot.show()
            return plot

        def spd_plots():
            dosplot = DosPlotter(zero_at_efermi=False)
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
            plot = dosplot.add_dos_dict(dos_dict)
            # plot = dosplot.get_plot(xlim=[self.vbm_primitive-energy_lower_bound, self.cbm_primitive+energy_upper_bound])
            plot = dosplot.get_plot()
            fig = plt.gcf()
            fig.set_size_inches(18.5, 13.5, forward=True)
            plot.legend()
            plot.axvline(x=self.cbm_primitive, color="k", linestyle="--")
            plot.axvline(x=self.vbm_primitive, color="k", linestyle="--")
            plot.axvline(x=self.efermi+0.125, color="k", linestyle="-.")
            plot.show()
            return plot

        if spd:
            spd_plots()
        else:
            sites_total_plots()

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
    # os.chdir("/home/tug03990/work/research/hBN/hbnDefectStates/defect/SCAN/")
    # for i in ["nupdown2"]:
    #     ds = DosPlot(vasprun=i+"/dos/gamma/vasprun.xml", defect_type="cation", name="Monk_SCAN_"+i)
    #     ds.total_dos.show()
    #     ds.band_gap()

    main()