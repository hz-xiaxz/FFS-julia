import numpy as np  # general math functions
from quspin.basis import spinful_fermion_basis_general  # spin basis constructor
from quspin.operators import hamiltonian  # operators

"""
ED for 2d rectangular_lattice (half filled case)
"""


def ED(Lx, Ly, J, U, W, fixed):
    ###### define model parameters ######
    N_2d = Lx * Ly  # number of sites for spin 1
    #
    eps = np.random.uniform(-W / 2, W / 2, size=N_2d)
    if fixed:
        eps = np.zeros(N_2d) * W / 2

    #
    ###### setting up user-defined BASIC symmetry transformations for 2d lattice ######
    s = np.arange(N_2d)  # sites [0,1,2,...,N_2d-1] in simple notation
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites
    T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
    T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
    S = -(s + 1)  # fermion spin inversion in the simple case
    N_up = N_2d // 2
    N_down = N_2d - N_up
    #
    ###### setting up bases ######
    basis_2d = spinful_fermion_basis_general(
        N_2d, Nf=(N_up, N_down), kxblock=(T_x, 0), kyblock=(T_y, 0), sblock=(S, 0)
    )
    #
    ###### setting up hamiltonian ######
    # setting up site-coupling lists for simple case
    hopping_left = [[-J, i, T_x[i]] for i in range(N_2d)] + [
        [-J, i, T_y[i]] for i in range(N_2d)
    ]
    hopping_right = [[+J, i, T_x[i]] for i in range(N_2d)] + [
        [+J, i, T_y[i]] for i in range(N_2d)
    ]
    potential = [[eps[i], i] for i in range(N_2d)]
    interaction = [[U, i, i] for i in range(N_2d)]
    #
    static = [
        ["+-|", hopping_left],  # spin up hops to left
        ["-+|", hopping_right],  # spin up hops to right
        ["|+-", hopping_left],  # spin down hops to left
        ["|-+", hopping_right],  # spin up hops to right
        ["n|", potential],  # onsite potential, spin up
        ["|n", potential],  # onsite potential, spin down
        ["n|n", interaction],
    ]  # spin up-spin down interaction
    # build hamiltonian
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H = hamiltonian(static, [], basis=basis_2d, dtype=np.float64)
    # diagonalise H
    E=H.eigvalsh()
    return E 


if __name__ == "__main__":
    print(ED(2, 2, 1, 1, 1, True))
