import numpy as np

def psd_voigt_function(x, xc, w, mu):
    return (1 * ( mu * (2/np.pi) * (w / (4*(x-xc)**2 + w**2)) +
        (1 - mu) * (np.sqrt(4*np.log(2)) / (np.sqrt(np.pi) * w)) *
        np.exp(-(4*np.log(2)/w**2)*(x-xc)**2) )) # pseudo voidt function copied from origin

class settings():
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.active_folder = ''
        self.mw.mol_instr_gamma.setOpts(compactHeight=False)

    def get_instr(self, x):
        """Compute the instrumental/overal broadening profile, that needs to be convolved.
        
        # TODO: Deprecated and should be removed; calculation of instrumental profile should not happen in/by the UI.
        If a thread needs it, it should be computed locally, based on already provided values.
        Currently, the user can change the instrumental profile while work is already submitted.
        This is less eggregious for the broadening then it is for the database selection.
        """
        w = self.mw.mol_instr_w.value()
        mu = self.mw.mol_instr_mu.value()
        instr = psd_voigt_function(x, np.mean(x), w, mu)
        return instr