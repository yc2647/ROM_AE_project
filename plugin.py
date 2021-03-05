"""
Contains material from:
* https://gitlab.kitware.com/paraview/paraview/blob/master/Examples/Plugins/PythonAlgorithm/PythonAlgorithmExamples.py
* https://discourse.paraview.org/t/issues-with-load-state-using-a-custom-reader/1259/9
* https://discourse.paraview.org/t/help-on-a-python-programmable-filter-plugin/2104/5
"""

import importlib
import numpy as np
import os
import sys
import torch
from collections import defaultdict
from paraview.vtk import vtkFloatArray, vtkUnstructuredGrid
from paraview.vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from paraview.vtk.vtkIOXdmf3 import vtkXdmf3Reader
from paraview.util.vtkAlgorithm import smdomain, smhint, smproperty, smproxy, VTKPythonAlgorithmBase


@smproxy.reader(name="NonIntrusiveROM", label="Non-intrusive ROM",
                extensions="xdmf", file_description="XDMF files", support_reload=False)
class NonIntrusiveROM(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType="vtkUnstructuredGrid")

        # Common data
        self._folder = None
        self._action = None

        # Basis functions data
        self._components = None
        self._filenames = None
        self._basis_functions = None
        self._arrays = None

        # Online solve data
        self._mu = None
        self._mu_range = None
        self._normalize_inputs = None
        self._normalize_outputs = None
        self._networks = None
        self._N = None

    @smproperty.stringvector(name="FileName", panel_visibility="never")
    @smdomain.filelist()
    @smhint.filechooser(extensions="xdmf", file_description="XDMF files")
    def SetFileName(self, filename):
        if filename is not None and filename != "None":  # Skip initial default filename
            # Find the basis folder
            filename = os.path.abspath(filename)
            basis_folder = os.path.dirname(filename)
            assert os.path.basename(basis_folder) == "basis"
            root_folder = os.path.dirname(basis_folder)
            if self._folder != root_folder:
                self._folder = root_folder
                # Add root folder to path, for later imports
                if self._folder not in sys.path:
                    sys.path.insert(0, self._folder)
                # Get available components
                with open(os.path.join(basis_folder, "components"), "r") as infile:
                    self._components = [line.rstrip("\n") for line in infile]
                # Store basis functions file names
                self._filenames = dict()
                for c_int, c_str in enumerate(self._components):
                    filenames_c = list()
                    n = 0
                    while True:
                        basis_c_n = os.path.join(
                            basis_folder, "basis_" + c_str + "_" + str(n) + "_block_" + str(c_int) + ".xdmf")
                        if os.path.exists(basis_c_n):
                            filenames_c.append(basis_c_n)
                            n += 1
                        else:
                            break
                    self._filenames[c_str] = filenames_c
                # Reset basis functions storage
                self._basis_functions = None
                # Mark state as modified
                self.Modified()

    def _load_basis_functions(self):
        assert self._filenames is not None

        # Load basis functions from XDMF files
        basis_functions = dict()
        for (c, filenames) in self._filenames.items():
            basis_functions_c = list()
            for (n, filename) in enumerate(filenames):
                reader = vtkXdmf3Reader()
                reader.SetFileName(filename)
                reader.UpdateInformation()
                reader.Update()
                basis_functions_c.append(reader)
            basis_functions[c] = basis_functions_c

        # Load arrays
        arrays = dict()
        for (c, basis_functions_c) in basis_functions.items():
            arrays_c = list()
            for basis_function in basis_functions_c:
                basis_function = basis_function.GetOutputDataObject(0)
                assert basis_function.GetCellData().GetNumberOfArrays() == 0
                assert basis_function.GetPointData().GetNumberOfArrays() == 1
                arrays_c.append(basis_function.GetPointData().GetArray(0))
            arrays[c] = arrays_c

        # Make sure that the number of basis functions is the same across all components
        num_basis_functions = None
        for (c, basis_functions_c) in basis_functions.items():
            if num_basis_functions is None:
                num_basis_functions = len(basis_functions_c)
            else:
                assert num_basis_functions == len(basis_functions_c)

        # Store basis functions
        assert self._basis_functions is None
        self._basis_functions = list()
        self._arrays = defaultdict(list)
        for n in range(num_basis_functions):
            basis_function_n = vtkUnstructuredGrid()
            basis_function_n.DeepCopy(basis_functions[self._components[0]][n].GetOutputDataObject(0))
            basis_function_n.GetPointData().RemoveArray(0)
            for c in self._components:
                array_c_n_copy = vtkFloatArray()
                array_c_n_copy.DeepCopy(arrays[c][n])
                array_c_n_copy.SetName(c)
                basis_function_n.GetPointData().AddArray(array_c_n_copy)
                self._arrays[c].append(vtk_to_numpy(array_c_n_copy))
            self._basis_functions.append(basis_function_n)
        for c in self._components:
            self._arrays[c] = np.stack(self._arrays[c], axis=-1)

    def _init_mu(self):
        assert self._folder is not None

        # Load training set
        with open(os.path.join(self._folder, "sets", "training_set.txt"), "r") as infile:
            training_set = eval(infile.read(), {}, {})
        assert isinstance(training_set, list)
        assert isinstance(training_set[0], tuple)
        mu_len = len(training_set[0])
        assert all(len(mu) == mu_len for mu in training_set)

        # Set default mu
        self._mu = [0] * mu_len

        # Import mu_range
        network_module = importlib.import_module("network")
        self._mu_range = network_module.mu_range

    def _load_networks(self):
        assert self._folder is not None
        assert self._mu is not None

        # Import Network class
        network_module = importlib.import_module("network")
        Network = network_module.Network

        # Prepare input normalization
        NormalizeInputs = network_module.NormalizeInputs
        self._normalize_inputs = NormalizeInputs(self._mu_range)

        # Prepare output normailization
        NormalizeOutputs = network_module.NormalizeOutputs
        self._normalize_outputs = dict()
        for c in self._components:
            normalize_outputs_c = list()
            for (n, _) in enumerate(self._basis_functions):
                normalize_outputs_c_n = NormalizeOutputs(os.path.join(
                    self._folder, "networks", "output_normalization_" + c + "_" + str(n + 1)))
                normalize_outputs_c.append(normalize_outputs_c_n)
            self._normalize_outputs[c] = normalize_outputs_c

        # Load networks for all basis functions sizes
        self._networks = dict()
        for c in self._components:
            network_c = list()
            for (n, _) in enumerate(self._basis_functions):
                network_c_n = Network(len(self._mu), c, n + 1)
                network_c_n.load_state_dict(torch.load(os.path.join(
                    self._folder, "networks", "network_" + c + "_" + str(n + 1))))
                network_c.append(network_c_n)
            self._networks[c] = network_c

    def RequestData(self, request, in_info_vec, out_info_vec):
        if self._basis_functions is None:
            self._load_basis_functions()
        if self._mu is None:
            self._init_mu()

        if self._action == "View basis functions":
            # Get basis function number from time step
            out_info = out_info_vec.GetInformationObject(0)
            selection = self.GetExecutive().UPDATE_TIME_STEP()
            if out_info.Has(selection):
                basis_function_number = int(round(out_info.Get(selection)))
            else:
                basis_function_number = 0

            # Copy corresponding basis function to output
            output = vtkUnstructuredGrid.GetData(out_info_vec)
            output.ShallowCopy(self._basis_functions[basis_function_number])
        elif self._action == "Run non-intrusive ROM":
            if self._networks is None:
                self._load_networks()

            # Compute reduced solution with pytorch
            mu_torch = self._normalize_inputs(tuple(self._mu))
            reduced_solution = dict()
            for c in self._components:
                network_c = self._networks[c][self._N - 1]
                normalize_outputs_c = self._normalize_outputs[c][self._N - 1]
                reduced_solution[c] = normalize_outputs_c.inv(network_c(mu_torch).detach().numpy()[0])

            # Copy mesh to output
            output = vtkUnstructuredGrid.GetData(out_info_vec)
            output.DeepCopy(self._basis_functions[0])
            for (c, _) in enumerate(self._components):
                output.GetPointData().RemoveArray(c)

            # Combine basis functions with coefficients provided by network
            for c in self._components:
                array_c_np = np.dot(self._arrays[c][..., :self._N], reduced_solution[c])
                array_c_vtk = numpy_to_vtk(array_c_np, deep=1)
                array_c_vtk.SetName(c)
                output.GetPointData().AddArray(array_c_vtk)
        else:
            raise RuntimeError("Invalid action")

        return 1

    def RequestInformation(self, request, in_info_vec, out_info_vec):
        if self._basis_functions is None:
            self._load_basis_functions()
        if self._mu is None:
            self._init_mu()

        executive = self.GetExecutive()
        out_info = out_info_vec.GetInformationObject(0)
        out_info.Remove(executive.TIME_RANGE())
        out_info.Remove(executive.TIME_STEPS())

        if self._action == "View basis functions":
            # Update information stored in output with the number of basis functions
            out_info.Append(executive.TIME_RANGE(), 0)
            out_info.Append(executive.TIME_RANGE(), len(self._basis_functions))
            for n in range(len(self._basis_functions)):
                out_info.Append(executive.TIME_STEPS(), n)
        elif self._action == "Run non-intrusive ROM":
            pass
        else:
            raise RuntimeError("Invalid action")

        return 1

    @smproperty.stringvector(name="Actions", information_only="1")
    def GetActions(self):
        return ["View basis functions", "Run non-intrusive ROM"]

    @smproperty.stringvector(name="Action", number_of_elements=1, default_values=["View basis functions"])
    @smdomain.xml(
        """
        <StringListDomain name="list">
            <RequiredProperties>
                <Property name="Actions" function="Actions"/>
            </RequiredProperties>
        </StringListDomain>
        """
    )
    def SetAction(self, action):
        self._action = action
        self.Modified()

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        assert self._basis_functions is not None
        return list(range(len(self._basis_functions)))

    @smproperty.intvector(name="OnlineSizeRange", information_only="1")
    def GetOnlineSizeRange(self):
        assert self._basis_functions is not None
        return (1, len(self._basis_functions))

    @smproperty.intvector(name="OnlineSize", default_values=[0])
    @smdomain.xml(
        """
        <IntRangeDomain name="range" default_mode="max">
            <RequiredProperties>
                <Property name="OnlineSizeRange" function="OnlineSizeRange" />
            </RequiredProperties>
        </IntRangeDomain>
        """
    )
    def SetOnlineSize(self, N):
        self._N = N
        self.Modified()

    @smproperty.doublevector(name="Mu0Range", information_only="1")
    def GetMu0Range(self):
        assert self._mu_range is not None
        return self._mu_range[0]

    @smproperty.doublevector(name="Mu0", default_values=[0])
    @smdomain.xml(
        """
        <DoubleRangeDomain name="range" default_mode="min">
            <RequiredProperties>
                <Property name="Mu0Range" function="Mu0Range" />
            </RequiredProperties>
        </DoubleRangeDomain>
        """
    )
    def SetMu0(self, mu_0):
        if self._mu is not None:
            self._mu[0] = mu_0
            self.Modified()
