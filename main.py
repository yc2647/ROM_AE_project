import argparse
from problem import truth_solve, write_solution
from reduced_problem import reduced_solve, reconstruct_solution, reduced_error
from reduction import create_folders, get_parameter_from_set, initialize_set, perform_POD, train_data_driven
import sys
sys.stdout.flush()
parser = argparse.ArgumentParser()
parser.add_argument("--action", required=True, type=str)
parser.add_argument("--mu", nargs='+', type=float)
parser.add_argument("--number-basis", type=int)
parser.add_argument("--set-cardinality", type=int)
parser.add_argument("--set-entry", type=int)
parser.add_argument("--set-name", type=str)
args = parser.parse_args()
action = args.action
if args.mu is not None:
    mu = tuple(args.mu)
else:
    mu = None
if args.number_basis is not None:
    N = args.number_basis
else:
    N = None
if args.set_cardinality is not None:
    set_cardinality = args.set_cardinality
else:
    set_cardinality = None
if args.set_entry is not None:
    set_entry = args.set_entry
else:
    set_entry = None
if args.set_name is not None:
    set_name = args.set_name
else:
    set_name = None

create_folders()

if action == "initialize set":
    assert mu is None
    assert N is None
    assert set_entry is None
    assert set_cardinality is not None
    assert set_name is not None
    initialize_set(set_cardinality, set_name)
elif action in ("perform POD", "train data driven"):
    assert mu is None
    assert N is not None
    assert set_cardinality is None
    assert set_entry is None
    assert set_name is None
    if action == "perform POD":
        perform_POD(N)
    elif action == "train data driven":
        train_data_driven(N)
elif action in ("truth solve", "reduced solve", "reduced error"):
    assert set_cardinality is None
    if mu is not None:
        assert set_entry is None
        assert set_name is None
    else:
        assert set_entry is not None
        assert set_name is not None
        mu = get_parameter_from_set(set_entry, set_name)
    if action == "truth solve":
        assert N is None
        # print(mu)
        # quit()
        solution = truth_solve(mu)
        write_solution(mu, "truth_solve", solution)
    elif action == "reduced solve":
        assert N is not None
        reduced_solution = reduced_solve(mu, N)
        reconstructed_solution = reconstruct_solution(reduced_solution, N)
        write_solution(mu, "reduced_solve", reconstructed_solution)
    elif action == "reduced error":
        assert N is not None
        reconstructed_error = reduced_error(mu, N)
        write_solution(mu, "reduced_error", reconstructed_error)
else:
    raise ValueError("Invalid action")
