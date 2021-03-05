import os
import torch
from dolfin import assign
from multiphenics import BlockFunction
from rbnics.backends import BasisFunctionsMatrix
from rbnics.backends.online import OnlineFunction
from rbnics.utils.io import OnlineSizeDict
from network import mu_range, Network, NormalizeInputs, NormalizeOutputs
from problem import components, generate_block_function_space, read_mesh, read_solution


def read_basis_functions(W, N):
    basis_functions = BasisFunctionsMatrix(W)
    basis_functions.init(components)
    loaded = basis_functions.load("basis", "basis")
    assert loaded
    N_dict = OnlineSizeDict()
    for c in components:
        N_dict[c] = N
    return basis_functions[:N_dict]


def reduced_solve(mu, N):
    normalize_inputs = NormalizeInputs(mu_range)
    mu_torch = normalize_inputs(mu)

    mu_torch = mu_torch.view(mu_torch.shape[1], -1)

    reduced_solution = dict()
    for c in components:
        network = Network(len(mu), c, N)
        network.load_state_dict(torch.load(os.path.join("networks", "network_" + c + "_" + str(N))))

        normalize_outputs = NormalizeOutputs(os.path.join("networks", "output_normalization_" + c + "_" + str(N)))

        reduced_solution_c = OnlineFunction(N)
        reduced_solution_c.vector()[:] = normalize_outputs.inv(network(mu_torch).detach().numpy()[0])
        reduced_solution[c] = reduced_solution_c
    return reduced_solution


def reconstruct_solution(reduced_solution, N):
    (mesh, _, _, restrictions) = read_mesh()
    W = generate_block_function_space(mesh, restrictions)
    reconstructed_solution = BlockFunction(W)

    basis_functions = read_basis_functions(W, N)
    for c in components:
        assign(reconstructed_solution.sub(c), (basis_functions[c] * reduced_solution[c]).sub(c))
    reconstructed_solution.apply("from subfunctions")
    return reconstructed_solution


def reduced_error(mu, N):
    (mesh, _, _, restrictions) = read_mesh()
    W = generate_block_function_space(mesh, restrictions)

    truth_solution = BlockFunction(W)
    read_solution(mu, "truth_solve", truth_solution)

    reduced_solution = BlockFunction(W)
    read_solution(mu, "reduced_solve", reduced_solution)

    return truth_solution - reduced_solution
