from cherab.openadas import OpenADAS
from cherab.core.atomic import neon
import numpy as np

eV_to_J = 1.602176634e-19
eV_to_K = 11604.51812155008

adas = OpenADAS()

charge = 7
te_eV = 1e3
te_K = te_eV * eV_to_K     # convert to Kelvin
ne_m3 = 1e19
n_ion = 1e17

log_te = np.log10(te_K)
log_ne = np.log10(ne_m3)

ion_rate_obj = adas.ionisation_rate(neon, charge)


sv = ion_rate_obj.evaluate(log_te, log_ne)  # m^3/s
"""
E_ion_J = neon.ionisation_energy[charge]
P_ionisation = n_ion * ne_m3 * sv * E_ion_J

print("Ionisation power density [W/m^3]:", P_ionisation)
"""
