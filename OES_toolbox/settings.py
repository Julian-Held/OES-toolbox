import numpy as np

def psd_voigt_function(x, xc, w, mu):
    return (1 * ( mu * (2/np.pi) * (w / (4*(x-xc)**2 + w**2)) +
        (1 - mu) * (np.sqrt(4*np.log(2)) / (np.sqrt(np.pi) * w)) *
        np.exp(-(4*np.log(2)/w**2)*(x-xc)**2) )) # pseudo voidt function copied from origin

class settings():
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.active_folder = ''

    def get_instr(self, x):
        w = self.mw.mol_instr_w.value()
        mu = self.mw.mol_instr_mu.value()
        instr = psd_voigt_function(x, np.mean(x), w, mu)
        return instr