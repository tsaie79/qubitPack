from qc_searching.py_energy_diagram.application.energydiagram import ED


class EnergyLevel(ED):

    def __init__(self, up, down):
        super().__init__(aspect='equal')
        self.up = up
        self.down = down

    def plotting(self, cbm, vbm):

        for up, en in self.up.items():
            self.add_level(energy=up, position="last", occupied=en)

        for dn, en in self.down.items():
            if dn == list(self.down.keys())[0]:
                self.add_level(dn, None, occupied=en)
            else:
                self.add_level(dn, None, "last", occupied=en)

        self.plot(show_IDs=True, cbm=cbm, vbm=vbm)
        my_fig = self.fig
        my_fig.show()

        return my_fig


if __name__ == '__main__':
    up = {-2.7996:True, -2.7104:True, -2.5998:True, -2.2885:True, -1.9898:False, -1.8119:False}
    down = {-2.7732: True, -2.3932: True, -2.2017:False, -2.0271: False, -1.8982: False, -1.7163: False}
    eng = EnergyLevel(up=up, down=down)
    fig = eng.plotting(-2.2566, -0.5512)
    fig.savefig("040752_W15Se32.png")
