from mpi4py import MPI
from dolfinx import io, plot
import pyvista as pv

# 1. Initialize MPI and load the mesh
comm = MPI.COMM_WORLD
with io.XDMFFile(comm, "simulation_mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

print("Mesh loaded. Extracting geometry for visualization...")

# 2. Extract plot data directly from FEniCSx mesh topology
# This fetches the 3D tetrahedral cells and their vertex coordinates
tdim = mesh.topology.dim
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)

# 3. Create a PyVista unstructured grid object
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# 4. Set up the interactive 3D plotter window
plotter = pv.Plotter(title="FEniCSx AVM Mesh Visualization")

# Add the mesh to the window (show edges so you can see the tetrahedrons)
plotter.add_mesh(grid, show_edges=True, color="lightgray", edge_color="black", line_width=0.5)

# Add ambient visual anchors (axes, background)
plotter.add_axes()
plotter.set_background("white")

print("Opening viewer window... (You can rotate, zoom, and pan the model)")
# 5. Render the interactive window
plotter.show()