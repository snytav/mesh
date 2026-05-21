import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

h5_filename = "fsi_results.h5"
print(f"Opening binary data target: {h5_filename}")

# 1. Parse the exact HDF5 paths
with h5py.File(h5_filename, "r") as f:
    points = f["Mesh/Grid/geometry"][:]
    cells = f["Mesh/Grid/topology"][:]
    print(f"Loaded Mesh Geometry: {points.shape} vertices.")
    print(f"Loaded Mesh Topology: {cells.shape} elements.")

    pressure_values = f["Function/Pressure/0"][:]
    vel_disp_values = f["Function/Velocity_Displacement/0"][:]

# 2. Extract spatial coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2] if points.shape[1] > 2 else np.zeros_like(x)
pressure_scalars = pressure_values.flatten()

# 3. Convert 4-node elements (quads) into 3-node triangles
if cells.shape[1] == 4:
    print("Detected 4-node elements. Converting to triangles for rendering...")
    tri1 = cells[:, [0, 1, 2]]
    tri2 = cells[:, [0, 2, 3]]
    triangles = np.vstack((tri1, tri2))
else:
    triangles = cells

# 4. Create the 2D triangulation object with the fixed 3-node connectivity
triang = Triangulation(x, y, triangles)

# 5. Generate the fully interactive 3D viewer window
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Render continuous surface colors matching the FSI field scalars
surf = ax.plot_trisurf(
    x, y, z,
    triangles=triang.triangles,
    cmap='turbo',
    linewidth=0.1,
    antialiased=True,
    edgecolors='#2c3e50'
)

# Bind the Pressure scalar data to color the physical elements
surf.set_array(pressure_scalars)

# 6. Add color scheme legend bars and labels
cbar = fig.colorbar(surf, ax=ax, pad=0.08, shrink=0.55)
cbar.set_label("Pressure Scalar Field", fontsize=11, fontweight='bold')

ax.set_title("FEniCS Fully-Coupled FSI Field Dashboard", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("X Space Coordinate")
ax.set_ylabel("Y Space Coordinate")
ax.set_zlabel("Z Elevation Profile")

# FIX: Use 'elev' instead of 'elevation' for modern Matplotlib versions
ax.view_init(elev=30, azim=-60)

print("\nOpening interactive Matplotlib 3D Viewport layout...")
plt.show()