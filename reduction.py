import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ufl import dx, grad, inner
from multiphenics import block_assemble, BlockFunction, BlockTestFunction, BlockTrialFunction, block_split
from rbnics.backends import BasisFunctionsMatrix, ProperOrthogonalDecomposition, SnapshotsMatrix, transpose
from rbnics.backends.online import OnlineFunction, OnlineLinearSolver
from rbnics.sampling import ParameterSpaceSubset
from rbnics.sampling.distributions import EquispacedDistribution, UniformDistribution
from rbnics.utils.io import Folders
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from network import mu_range, Network, NormalizeInputs, NormalizeOutputs
from problem import components, generate_block_function_space, read_mesh, read_solution
from reduced_problem import read_basis_functions
import sys
import pickle
sys.stdout.flush()

def create_folders():
    folders = list()
    folders.append("basis")
    folders.append("networks")
    folders.append("reduced_solve")
    folders.append("reduced_error")
    folders.append("sets")
    folders.append("truth_solve")
    for folder in folders:
        Folders.Folder(folder).create()


def get_set(name):
    set_ = ParameterSpaceSubset()
    set_.load("sets", name)
    return set_


def get_parameter_from_set(entry, name):
    set_ = get_set(name)
    return set_[entry]


def initialize_set(cardinality, name):
    set_ = ParameterSpaceSubset()
    assert name in ("training_set", "testing_set")
    if name == "training_set":
        sampling = EquispacedDistribution()
    elif name == "testing_set":
        sampling = UniformDistribution()
    set_.generate(mu_range, cardinality, sampling=sampling)
    set_.save("sets", name)


def get_inner_products(W, for_):
    # Test and trial functions
    block_v = BlockTestFunction(W)
    v, q = block_split(block_v)
    block_u = BlockTrialFunction(W)
    u, p = block_split(block_u)

    # Inner products
    assert for_ in ("POD", "L2 projection")
    if for_ == "POD":
        # x = {
        #     "u": [[inner(grad(u), grad(v)) * dx]]
        # }
        x = {
            "u": [[inner(grad(u), grad(v)) * dx, 0], [0, 0]],
            "p": [[0, 0], [0, inner(p, q) * dx]]
        }
    elif for_ == "L2 projection":
        # x = {
        #     "u": [[inner(u, v) * dx]]
        # }
        x = {
            "u": [[inner(u, v) * dx, 0], [0, 0]],
            "p": [[0, 0], [0, inner(p, q) * dx]]
        }
    return {c: block_assemble(x[c]) for c in components}


def perform_POD(N):
    # export mesh - instead of generating mesh everytime
    (mesh, _, _, restrictions) = read_mesh()
    W = generate_block_function_space(mesh, restrictions)

    # POD objects
    X = get_inner_products(W, "POD")
    POD = {c: ProperOrthogonalDecomposition(W, X[c]) for c in components}

    # Solution storage
    solution = BlockFunction(W)

    # Training set
    training_set = get_set("training_set")

    # Read in snapshots
    for mu in training_set:
        print("Appending solution for mu =", mu, "to snapshots matrix")
        read_solution(mu, "truth_solve", solution)
        for c in components:
            POD[c].store_snapshot(solution, component=c)

    # Compress component by component
    basis_functions_component = dict()
    for c in components:
        _, _, basis_functions_component[c], N_c = POD[c].apply(N, tol=0.)
        assert N_c == N
        print("Eigenvalues for component", c)
        POD[c].print_eigenvalues(N)
        POD[c].save_eigenvalues_file("basis", "eigenvalues_" + c)

    # Collect all components and save to file
    basis_functions = BasisFunctionsMatrix(W)
    basis_functions.init(components)
    for c in components:
        basis_functions.enrich(basis_functions_component[c], component=c)
    basis_functions.save("basis", "basis")
    # Also save components to file, for the sake of the ParaView plugin
    with open(os.path.join("basis", "components"), "w") as file_:
        for c in components:
            file_.write(c + "\n")


class L2ProjectionSolver(object):
    def __init__(self, inner_product, basis_functions, N):
        self.N = N
        self.basis_functions = basis_functions
        self.inner_product = inner_product
        self.inner_product_N = transpose(self.basis_functions) * self.inner_product * self.basis_functions

    def solve(self, mu, c, snapshot):
        print("Performing L^2 projection for mu =", mu, "and component", c)
        #quit()
        projected_snapshot_N = OnlineFunction(self.N)
        solver = OnlineLinearSolver(self.inner_product_N, projected_snapshot_N,
                                    transpose(self.basis_functions) * self.inner_product * snapshot)
        solver.solve()
        return projected_snapshot_N.vector().__array__()


def init_weights(m):
    """Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution."""
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def train_data_driven(N):
    (mesh, _, _, restrictions) = read_mesh()
    W = generate_block_function_space(mesh, restrictions)

    # L2 projection object
    basis_functions = read_basis_functions(W, N)
    X = get_inner_products(W, "L2 projection")
    l2_projection = {c: L2ProjectionSolver(X[c], basis_functions[c], N) for c in components}

    # Solution storage
    solution = BlockFunction(W)

    # Training set
    training_set = get_set("training_set")
    mu_len = len(training_set[0])


    # Read in snapshots
    snapshots_matrix = SnapshotsMatrix(W)
    for i, mu in enumerate(training_set):
        print("Appending solution for mu =", mu, "to snapshots matrix")
        read_solution(mu, "truth_solve", solution)
        snapshots_matrix.enrich(solution)

        filename = os.path.join("dis_x","dis_x_"+str(i))
        write_file = open(filename, 'wb')
        pickle.dump(snapshots_matrix[-1][0].vector()[::3], write_file)
        write_file.close()

        filename = os.path.join("dis_y","dis_y_"+str(i))
        write_file = open(filename, 'wb')
        pickle.dump(snapshots_matrix[-1][0].vector()[1::3], write_file)
        write_file.close()

        filename = os.path.join("dis_z","dis_z_"+str(i))
        write_file = open(filename, 'wb')
        pickle.dump(snapshots_matrix[-1][0].vector()[2::3], write_file)
        write_file.close()

    quit()

    # Data driven training component by component
    normalize_inputs = NormalizeInputs(mu_range)
    for c in components:
        projected_snapshots = [l2_projection[c].solve(mu, c, snapshots_matrix[i])
                               for i, mu in enumerate(training_set)]
        inputs = torch.unsqueeze(torch.FloatTensor(training_set._list), dim=mu_len)
        inputs = normalize_inputs(inputs)
        outputs = torch.stack([
            torch.from_numpy(projected_snapshot) for projected_snapshot in projected_snapshots])
        with open(os.path.join("networks", "output_normalization_" + c + "_" + str(N)), "w") as file_:
            file_.write(str(torch.min(outputs).detach().numpy()) + "\n")
            file_.write(str(torch.max(outputs).detach().numpy()) + "\n")
        normalize_outputs = NormalizeOutputs(os.path.join("networks", "output_normalization_" + c + "_" + str(N)))
        outputs = normalize_outputs(outputs)
        # print(len(training_set[0]))
        # print(len(training_set))
        # print(mu_len)
        # print(inputs.shape)
        # print(outputs.shape)
        # quit()
        network = Network(mu_len, c, N)
        network.apply(init_weights)

        criterion = nn.MSELoss()
        learning_rate = 0.3
        optimizer = optim.Adam(network.parameters(), lr=learning_rate, eps=1.e-08)
        torch_dataset = TensorDataset(inputs.float(), outputs.float())

        n_snpashots = len(training_set)
        n_trainining = 4 * int(n_snpashots / 6)
        n_validation = n_snpashots - n_trainining
        batch_size_training = int(round(np.sqrt(n_snpashots)))
        batch_size_validation = int(round(np.sqrt(n_snpashots)))
        epochs = 10000
        n_epochs_stop = epochs

        training_dataset, validation_dataset = random_split(
            torch_dataset, [n_trainining, n_validation])
        training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size_training)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size_validation)

        training_losses = [None] * epochs
        validation_losses = [None] * epochs
        min_validation_loss = np.Inf

        for epoch in range(epochs):
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate / (1 + np.sqrt(epoch))

            total_training_loss = 0.0
            for batch_x, batch_y in training_loader:  # for each training step
                network.train()
                optimizer.zero_grad()
                batch_x_normalized = batch_x.squeeze(1)
                prediction = network(batch_x_normalized)
                loss = criterion(prediction, batch_y)
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()
            training_losses[epoch] = total_training_loss / len(training_loader)
            print("[%d] Training loss: %.10f" % (epoch + 1, training_losses[epoch]))

            network.eval()

            total_validation_loss = 0.0
            with torch.no_grad():
                for validation_x, validation_y in validation_loader:
                    validation_x_normalized = validation_x.squeeze(1)
                    network_y = network(validation_x_normalized)
                    loss = criterion(network_y, validation_y)
                    total_validation_loss += loss.item()
            validation_losses[epoch] = total_validation_loss / len(validation_loader)
            print("[%d] Validation loss: %.10f" % (epoch + 1, validation_losses[epoch]))

            # add less than or eq
            if validation_losses[epoch] <= min_validation_loss:
                epochs_no_improvement = 0
                min_validation_loss = validation_losses[epoch]
                torch.save(network.state_dict(), os.path.join("networks", "network_" + c + "_" + str(N)))
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement == n_epochs_stop:
                print("Early stopping!")
                break
