from qubitPack.qc_searching.py_energy_diagram.application.energydiagram import ED


class EnergyLevel(ED):

    def __init__(self, levels, top_texts):
        super().__init__(aspect='equal')
        self.up = levels.get("1", {})
        self.down = levels.get("-1", {})
        if not top_texts:
            top_texts = {}
        self.up_top_texts = top_texts.get("1", ["" for i in range(len(self.up))])
        self.dn_top_texts = top_texts.get("-1", ["" for i in range(len(self.down))])
        self.up_top_texts.extend(["" for i in range(len(self.up) - len(self.up_top_texts))])
        self.dn_top_texts.extend(["" for i in range(len(self.down) - len(self.dn_top_texts))])
        print(self.up_top_texts, self.dn_top_texts)
    def plotting(self, cbm, vbm):
            for eig, occup, top_text in zip(self.up.keys(), self.up.values(), self.up_top_texts):
                self.add_level(energy=eig, position="last", occupied=occup, top_text=top_text)

            for eig, occup, top_text in zip(self.down.keys(), self.down.values(), self.dn_top_texts):
                if eig == list(self.down.keys())[0]:
                    self.add_level(eig, None, occupied=occup, top_text=top_text)
                else:
                    self.add_level(eig, None, "last", occupied=occup, top_text=top_text)


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
