"""Physical constants: which CODATA release the pinned SciPy ships, and the allowed names.

The release is read from the data itself (a constant whose value changed between
releases), so evidence can never be labelled with the wrong release. A SciPy upgrade
that ships a different release changes ``codata_release()``, and
tests/unit/test_phys_tools.py fails until the pin and this table are updated together.
"""
from __future__ import annotations

from functools import lru_cache

# Electron mass in kg per release: it changed in every recent CODATA adjustment.
_SIGNATURES = {
    "2022": 9.1093837139e-31,
    "2018": 9.1093837015e-31,
    "2014": 9.10938356e-31,
}

# name -> (scipy.constants key, pint name, quantity kind or None)
CONSTANTS: dict[str, tuple[str, str, str | None]] = {
    "c": ("speed of light in vacuum", "speed_of_light", "speed"),
    "h": ("Planck constant", "planck_constant", None),
    "hbar": ("reduced Planck constant", "hbar", None),
    "e": ("elementary charge", "elementary_charge", "charge"),
    "k_B": ("Boltzmann constant", "boltzmann_constant", None),
    "N_A": ("Avogadro constant", "avogadro_constant", None),
    "R": ("molar gas constant", "molar_gas_constant", None),
    "sigma": ("Stefan-Boltzmann constant", "stefan_boltzmann_constant", None),
    "epsilon_0": ("vacuum electric permittivity", "vacuum_permittivity", None),
    "mu_0": ("vacuum mag. permeability", "vacuum_permeability", None),
    "G": ("Newtonian constant of gravitation", "gravitational_constant", None),
    "g": ("standard acceleration of gravity", "standard_gravity", "acceleration"),
    "m_e": ("electron mass", "electron_mass", "mass"),
    "m_p": ("proton mass", "proton_mass", "mass"),
    "u": ("atomic mass constant", "atomic_mass_constant", "mass"),
    "F": ("Faraday constant", "faraday_constant", None),
}


@lru_cache(maxsize=1)
def codata_release() -> str:
    from scipy import constants

    m_e = constants.physical_constants["electron mass"][0]
    for release, value in _SIGNATURES.items():
        if m_e == value:
            return release
    return "unknown"


def scipy_version() -> str:
    import scipy

    return scipy.__version__
