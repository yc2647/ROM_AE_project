from dolfin import *
from multiphenics import *

# Helper function to generate interface restriction based on a boundary id
def generate_interface_restriction(mesh, boundaries, boundary_id):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        test_function_d = MeshFunction("bool", mesh, d)
        test_function_d.set_all(False)
        restriction.append(test_function_d)
    # Mark restriction mesh functions based on subdomain ids (except the mesh function corresponding to dimension D, as it is trivially false)
    for f in facets(mesh):
        if boundaries[f] == boundary_id:
            restriction[D - 1][f] = True
            for d in range(D - 1):
                for e in entities(f, d):
                    restriction[d][e] = True
    # Return
    return restriction

# Read in mesh generated with gmsh
mesh = Mesh("hyperelastic_cube.xml")
subdomains = MeshFunction("size_t", mesh, "hyperelastic_cube_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "hyperelastic_cube_facet_region.xml")

# Write out new-style xml files
File("hyperelastic_cube.xml") << mesh
File("hyperelastic_cube_physical_region.xml") << subdomains
File("hyperelastic_cube_facet_region.xml") << boundaries

# Write out new-style xml files
# File("medium.xml") << mesh
# File("medium_physical_region.xml") << subdomains
# File("medium_facet_region.xml") << boundaries

# Write out for visualization
XDMFFile("hyperelastic_cube.xdmf").write(mesh)
XDMFFile("hyperelastic_cube_physical_region.xdmf").write(subdomains)
XDMFFile("hyperelastic_cube_facet_region.xdmf").write(boundaries)

# Generate restriction corresponding to the internal interface
interface_restriction = generate_interface_restriction(mesh, boundaries, 1)

# Write out for simulation import (.xml) and visualization (.xdmf)
File("hyperelastic_cube_interface_restriction.rtc.xml") << interface_restriction
XDMFFile("hyperelastic_cube_interface_restriction.rtc.xdmf").write(interface_restriction)
