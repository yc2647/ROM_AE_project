from ufl import div, dx, ds, FiniteElement, grad, inner, VectorElement, pi, sym
from dolfin import (Constant, DirichletBC, Expression, Mesh, MeshFunction, UnitCubeMesh, SubDomain,
                    near, Identity, tr, det, ln, dot, derivative, VectorFunctionSpace, assemble, File,
                    Measure, BoxMesh, Point, FunctionSpace, inv)
from multiphenics import (block_derivative, BlockDirichletBC, BlockElement, BlockFunction, BlockFunctionSpace,
                          BlockNonlinearProblem, BlockPETScSNESSolver, block_split, BlockTestFunction,
                          BlockTrialFunction, block_assemble, MeshRestriction)
from rbnics.backends.dolfin.wrapping.function_save import SolutionFileXDMF as SolutionFile
import rbnics.backends.multiphenics.wrapping.block_function_space  # noqa: F401

components = ["u", "p"]
# components = ["u"]


def mu_str(mu):
    return "_".join("{:.3f}".format(mu_) for mu_ in mu)


def solution_files(mu, solution_folder):
    return {c: SolutionFile(solution_folder, "solution_" + c + "__" + mu_str(mu)) for c in components}


def write_solution(mu, solution_folder, from_):
    files = solution_files(mu, solution_folder)
    for (c, f) in files.items():
        f.write(from_.sub(c), c, 0)


def read_solution(mu, solution_folder, to):
    files = solution_files(mu, solution_folder)
    for (c, f) in files.items():
        files[c].read(to.sub(c), c, 0)
    to.apply("from subfunctions")


def read_mesh():
    # mesh = Mesh("data/backward_facing_step.xml")
    # subdomains = MeshFunction("size_t", mesh, "data/backward_facing_step_physical_region.xml")
    # boundaries = MeshFunction("size_t", mesh, "data/backward_facing_step_facet_region.xml")
    mesh = Mesh("data/hyperelastic_cube.xml")
    subdomains = MeshFunction("size_t", mesh, "data/hyperelastic_cube_physical_region.xml")
    boundaries = MeshFunction("size_t", mesh, "data/hyperelastic_cube_facet_region.xml")
    restrictions = MeshRestriction(mesh, "data/hyperelastic_cube_interface_restriction.rtc.xml")
    return (mesh, subdomains, boundaries, restrictions)

def create_mesh():
    N = 20
    x0, y0, z0 = -1.5, -1.5, -0.25
    x1, y1, z1 = 1.5, 1.5, 0.25

    # mesh = UnitCubeMesh.create(N, N, N//2, CellType.Type.hexahedron)
    mesh = BoxMesh(Point(x0, y0, z0), Point(x1, y1, z1), N, N, N//2)
    # mesh = UnitCubeMesh(N, N, N)

    # mesh size is smaller near x=y=0
    # mesh.coordinates()[:, :2] = mesh.coordinates()[:, :2]**2
    # mesh size is smaller near z=0 and mapped to a [-1;0] domain along z
    # mesh.coordinates()[:, 2] = -mesh.coordinates()[:, 2]**2
    # left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    # right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], z1) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], z0) and on_boundary

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x0) and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x1) and on_boundary

    class Front(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], y0) and on_boundary

    class Back(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], y1) and on_boundary

    class Symmetry_x(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary

    class Symmetry_y(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0) and on_boundary



    # exterior facets MeshFunction
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    Top().mark(boundaries, 1)
    Bottom().mark(boundaries, 2)
    Left().mark(boundaries, 3)
    Right().mark(boundaries, 4)
    Front().mark(boundaries, 5)
    Back().mark(boundaries, 6)
    # Symmetry_x().mark(boundaries, 2)
    # Symmetry_y().mark(boundaries, 3)


    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    subdomains.set_all(0)
    File("hyperelastic_cube.xml") << mesh
    File("hyperelastic_cube_physical_region.xml") << subdomains
    File("hyperelastic_cube_facet_region.xml") << boundaries

    return (mesh, subdomains, boundaries)

def generate_block_function_space(mesh, restrictions):
    # Block function space
    #V_element = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    # boundary_restriction = MeshRestriction(mesh, restrictions)
    # Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    #W_element = BlockElement(V_element)
    # Q. components = components?
    # A. it's a kwarg: the keyword argument name is before the equal, the value
    #    (i.e., the list defined on line 11) is after the equal sign.
    #    It's a modification of FunctionSpace/BlockFunctionSpace that RBniCS
    #    carries out.
    return BlockFunctionSpace([V, V1], restrict=[None, restrictions], components=components)
    #return BlockFunctionSpace(mesh, V_element, components=components)


def truth_solve(mu_unkown):
    print("Performing truth solve at mu =", mu_unkown)
    (mesh, subdomains, boundaries, restrictions) = read_mesh()
    # (mesh, subdomains, boundaries, restrictions) = create_mesh()
    dx = Measure('dx', subdomain_data=subdomains)
    ds = Measure('ds', subdomain_data=boundaries)
    W = generate_block_function_space(mesh, restrictions)

    # Test and trial functions
    block_v = BlockTestFunction(W)
    v, q = block_split(block_v)
    block_du = BlockTrialFunction(W)
    du, dp = block_split(block_du)
    block_u = BlockFunction(W)
    u, p = block_split(block_u)

    # gap
    # V2 = FunctionSpace(mesh, "CG", 1)
    # gap = Function(V2, name="Gap")

    # obstacle
    R = 0.25
    d = 0.15
    x_0 = mu_unkown[0]
    y_0 = mu_unkown[1]
    obstacle = Expression("-d+(pow(x[0]-x_0,2)+pow(x[1]-y_0, 2))/2/R", d=d, R=R , x_0 = x_0, y_0 = y_0, degree=0)

    # Constitutive parameters
    E = Constant(10.0)
    nu = Constant(0.3)
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    B  = Constant((0.0, 0.0, 0.0))      # Body force per unit volume
    T  = Constant((0.0, 0.0, 0.0))      # Traction force on the boundary

    # Kinematics
    # -----------------------------------------------------------------------------
    mesh_dim = mesh.topology().dim()                    # Spatial dimension
    I = Identity(mesh_dim)                 # Identity tensor
    F = I + grad(u)                 # Deformation gradient
    C = F.T*F                       # Right Cauchy-Green tensor
    J = det(F)                      # 3rd invariant of the deformation tensor

    # Strain function
    def P(u):                       # P = dW/dF:
        return mu*(F - inv(F.T)) + lmbda*ln(J)*inv(F.T)
    def eps(v):
        return sym(grad(v))
    def sigma(v):
        return lmbda*tr(eps(v))*Identity(3) + 2.0*mu*eps(v)
    # Definition of The Mackauley bracket <x>+
    def ppos(x):
        return (x+abs(x))/2.
    # Define the augmented lagrangian
    def aug_l(x):
        return x + pen*(obstacle-u[2])

    pen = Constant(1e4)

    # Boundary conditions
    # bottom_bc = DirichletBC(W.sub(0), Constant((0., 0., 0.)), boundaries, 2)
    # left_bc = DirichletBC(W.sub(0), Constant((0., 0., 0.)), boundaries, 3)
    # right_bc = DirichletBC(W.sub(0), Constant((0., 0., 0.)), boundaries, 4)
    # front_bc = DirichletBC(W.sub(0), Constant((0., 0., 0.)), boundaries, 5)
    # back_bc = DirichletBC(W.sub(0), Constant((0., 0., 0.)), boundaries, 6)
    # # sym_x_bc = DirichletBC(W.sub(0).sub(0), Constant(0.), boundaries, 2)
    # # sym_y_bc = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, 3)
    # # bc = BlockDirichletBC([bottom_bc, sym_x_bc, sym_y_bc])
    # bc = BlockDirichletBC([bottom_bc, left_bc, right_bc, front_bc, back_bc])

    bottom_bc = DirichletBC(W.sub(0), Constant((0., 0., 0.)), boundaries, 2)
    left_bc_x = DirichletBC(W.sub(0).sub(0), Constant(0.), boundaries, 3)
    left_bc_y = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, 3)
    right_bc_x = DirichletBC(W.sub(0).sub(0), Constant(0.), boundaries, 4)
    right_bc_y = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, 4)
    front_bc_x = DirichletBC(W.sub(0).sub(0), Constant(0.), boundaries, 5)
    front_bc_y = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, 5)
    back_bc_x = DirichletBC(W.sub(0).sub(0), Constant(0.), boundaries, 6)
    back_bc_y = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, 6)
    # sym_x_bc = DirichletBC(W.sub(0).sub(0), Constant(0.), boundaries, 2)
    # sym_y_bc = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, 3)
    # bc = BlockDirichletBC([bottom_bc, sym_x_bc, sym_y_bc])
    bc = BlockDirichletBC([bottom_bc, left_bc_x, left_bc_y, \
                           right_bc_x, right_bc_y, front_bc_x, front_bc_y, \
                           back_bc_x, back_bc_y])

    # Variational forms
    # F = inner(sigma(u), eps(v))*dx + pen*dot(v[2], ppos(u[2]-obstacle))*ds(1)

    # F = [inner(sigma(u), eps(v))*dx - aug_l(l)*v[2]*ds(1) + ppos(aug_l(l))*v[2]*ds(1),
    #     (obstacle-u[2])*v*ds(1) - (1/pen)*ppos(aug_l(l))*v*ds(1)]

    # F_a = inner(sigma(u), eps(v))*dx
    # F_b = - aug_l(p)*v[2]*ds(1) + ppos(aug_l(p))*v[2]*ds(1)
    # F_c = (obstacle-u[2])*q*ds(1)
    # F_d = - (1/pen)*ppos(aug_l(p))*q*ds(1)
    #
    # block_F = [[F_a, F_b],
    #            [F_c, F_d]]

    F_a = inner(P(u), grad(v))*dx - dot(B, v)*dx - dot(T, v)*ds \
        - aug_l(p)*v[2]*ds(1) + ppos(aug_l(p))*v[2]*ds(1)

    F_b = (obstacle-u[2])*q*ds(1) - (1/pen)*ppos(aug_l(p))*q*ds(1)

    block_F = [F_a,
               F_b]
    J = block_derivative(block_F, block_u, block_du)

    # Setup solver
    problem = BlockNonlinearProblem(block_F, block_u, bc, J)
    solver = BlockPETScSNESSolver(problem)
    solver.parameters.update({
        "linear_solver": "mumps",
        "absolute_tolerance": 1E-4,
        "relative_tolerance": 1E-4,
        "maximum_iterations": 50,
        "report": True,
        "error_on_nonconvergence": True
    })

    # solver.parameters.update({
    #     "linear_solver": "cg",
    #     "absolute_tolerance": 1E-4,
    #     "relative_tolerance": 1E-4,
    #     "maximum_iterations": 50,
    #     "report": True,
    #     "error_on_nonconvergence": True
    # })

    # Perform a fake loop over time. Note how up will store the solution at the last time.
    # Q. for?
    # A. You can remove it, since your problem is stationary. The template was targeting
    #    a final application which was transient, but in which the ROM should have only
    #    described the final solution (when reaching the steady state).
    # for _ in range(2):
    #     solver.solve()
    a1 = solver.solve()
    print(a1)
        # save all the solution here as a function of time

    # Return the solution at the last time
    # Q. block_u or block
    # A. I think block_u, it will split split among the components elsewhere
    return block_u
