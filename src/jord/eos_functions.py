"""
EOS Data and Functions
"""


def mie_gruneisen_debye(P, P0, rho0, K0, K0prime, gamma0, theta0, V0, T):
    """
    Calculates density from the Mie-Grüneisen-Debye EOS.
    """
    V = V0 * (P0 / (P + P0))**(1/K0prime)
    rho = rho0 * (V0 / V)

    # Thermal pressure correction (simplified)
    gamma = gamma0 * (V / V0)**1  # Assuming q = 1 for simplicity
    theta = theta0 * (V / V0)**(-gamma)
    P_thermal = (gamma * rho * 8.314 * (T - 300))  # Simplified thermal pressure

    return rho

def birch_murnaghan(P, P0, rho0, K0, K0prime, V0):
    """
    Calculates density from the 3rd order Birch-Murnaghan EOS.
    """
    eta = (3/2) * (K0prime - 4)
    V = V0 * (1 + (3/4) * (K0prime - 4) * ((P - P0)/K0))**(-2/(3 * (K0prime - 4)))
    
    density = rho0 * (V0 / V)

    return density