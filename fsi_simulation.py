import numpy as np
from mpi4py import MPI
import dolfinx.mesh
from dolfinx import io, fem, default_scalar_type
import dolfinx.fem.petsc
import ufl
from basix.ufl import element, mixed_element

# 1. Initialize MPI Environment and Load the Verified Mesh
comm = MPI.COMM_WORLD
with io.XDMFFile(comm, "simulation_mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# 2. Define Modern Mixed Function Spaces (Taylor-Hood Fluid-Solid Pair)
gdim = mesh.topology.dim
cell_shape = mesh.basix_cell()

v_el = element("Lagrange", cell_shape, 2, shape=(gdim,))
s_el = element("Lagrange", cell_shape, 1)

mixed_el = mixed_element([v_el, s_el])
W = fem.functionspace(mesh, mixed_el)

# 3. Define Trial, Test, and Solution Functions
w = fem.Function(W)
(u, p) = ufl.split(w)  # u = velocity/displacement, p = fluid/solid pressure
(v, q) = ufl.TestFunctions(W)

# 4. Set Up Physical Parameters
rho = fem.Constant(mesh, default_scalar_type(1060.0))  # Blood/Tissue Density (kg/m^3)
mu_f = fem.Constant(mesh, default_scalar_type(0.0035))  # Blood Dynamic Viscosity (Pa*s)
E_s = fem.Constant(mesh, default_scalar_type(1e5))  # Tissue Elastic Modulus (Pa)
nu_s = fem.Constant(mesh, default_scalar_type(0.45))  # Tissue Poisson's Ratio

mu_s = E_s / (2.0 * (1.0 + nu_s))
lam_s = (E_s * nu_s) / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))

# 5. Automated Multi-Phase Phase Indicator (Phi)
Q_phi = fem.functionspace(mesh, ("Lagrange", 1))
phi = fem.Function(Q_phi)
x_coords = mesh.geometry.x

center = np.mean(x_coords, axis=0)
phi.interpolate(lambda x: 1.0 / (1.0 + np.exp(10.0 * (np.linalg.norm(x - center[:, None], axis=0) - 0.5))))


# 6. Construct Unified Fluid-Structure Interaction Governing Equations
def fluid_stress(u, p):
    return 2.0 * mu_f * ufl.sym(ufl.grad(u)) - p * ufl.Identity(len(u))


def solid_stress(u, p):
    return 2.0 * mu_s * ufl.sym(ufl.grad(u)) + lam_s * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(
        len(u)) - p * ufl.Identity(len(u))


sigma_total = phi * fluid_stress(u, p) + (1.0 - phi) * solid_stress(u, p)

dx = ufl.dx
F_momentum = rho * ufl.dot(ufl.grad(u) * u, v) * dx + ufl.inner(sigma_total, ufl.grad(v)) * dx
F_continuity = ufl.div(u) * q * dx
F_total = F_momentum + F_continuity

# 7. Automated Boundary Conditions
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

W_collapsed, sub_to_parent_map = W.sub(0).collapse()
zero_displacement = fem.Function(W_collapsed)
zero_displacement.x.array[:] = 0.0

exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = fem.locate_dofs_topological((W.sub(0), W_collapsed), fdim, exterior_facets)
bc_solid = fem.dirichletbc(zero_displacement, boundary_dofs, W.sub(0))

# 8. Configure PETSc SNES Options
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-4,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

print("Executing fully-coupled Fluid-Structure Interaction SNES Solver...")

problem = dolfinx.fem.petsc.NonlinearProblem(
    F_total, w, bcs=[bc_solid],
    petsc_options=petsc_options,
    petsc_options_prefix="fsi_avm_"
)

num_iters = problem.solve()
converged = problem.solver.getConvergedReason()

if converged > 0:
    print(f"\nSuccess! Simulation converged cleanly via SNES in {num_iters} iterations.")

    # 9. FIX: Create separate, standard Degree 1 spaces matching the mesh degree for file output
    V_out = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))  # Vector field space (Degree 1)
    P_out = fem.functionspace(mesh, ("Lagrange", 1))  # Scalar field space (Degree 1)

    u_export = fem.Function(V_out, name="Velocity_Displacement")
    p_export = fem.Function(P_out, name="Pressure")

    # Extract raw sub-space vectors from your converged system solution
    u_res, p_res = w.split()

    # Interpolate the Degree 2 math down to the clean Degree 1 export containers
    u_export.interpolate(u_res)
    p_export.interpolate(p_res)

    # 10. Save Results for Visual Analysis
    with io.XDMFFile(comm, "fsi_results.xdmf", "w") as xdmf_out:
        xdmf_out.write_mesh(mesh)
        xdmf_out.write_function(u_export, 0.0)
        xdmf_out.write_function(p_export, 0.0)
    print("Results successfully exported to 'fsi_results.xdmf'.")
else:
    print(f"\n[Solver Warning] Execution completed, but failed to converge. Status reason code: {converged}")