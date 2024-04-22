import numpy as np
import owlspec as owl

class settings():
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.active_folder = ''

    def get_instr(self, x):
        w = self.mw.mol_instr_w.value()
        mu = self.mw.mol_instr_mu.value()
        instr = owl.util.psd_voigt_function(x, np.mean(x), w, mu)
        return instr