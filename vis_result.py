import pyvista as pv

# 1. Load the XDMF file
# Note: Ensure you have 'meshio' and 'h5py' installed to read XDMF
reader = pv.get_reader("fsi_results.xdmf")

# 2. Print available time steps and point/cell data arrays
print("Available Time Steps:", reader.time_values)
print("Available Data Arrays:", reader.point_data_keys)

# 3. Read the first time step mesh
reader.set_active_time_value(reader.time_values[0])
mesh = reader.read()

# 4. Set up the interactive plotter window
plotter = pv.Plotter(title="FEniCS FSI Visualization")

# 5. Add mesh to plotter (Change 'velocity' or 'displacement' to your actual array name)
# scalar_bar_args configures the color legend
plotter.add_mesh(
    mesh,
    scalars=mesh.point_data_keys[0],
    cmap="jet",
    show_edges=True,
    scalar_bar_args={"title": "FSI Fields"}
)

# 6. Customize view and open the interactive window
plotter.view_xy()
plotter.show()