import nibabel as nib
import meshio
import SVMTK  # Capitalized module import
from skimage import measure, filters
import numpy as np

# 1. Load your automated segmentation (NIfTI format)
img = nib.load("automated_segmentation.nii.gz")
data = img.get_fdata()

# 2. Smooth out jagged voxel stairs
smoothed_data = filters.gaussian(data, sigma=1.0)

# Normalize the data back to a strict 0.0 - 1.0 range
data_min = np.min(smoothed_data)
data_max = np.max(smoothed_data)
if data_max > data_min:
    smoothed_data = (smoothed_data - data_min) / (data_max - data_min)

# 3. Extract the clean surface boundary
verts, faces, normals, values = measure.marching_cubes(smoothed_data, level=0.5)

# 4. Save clean surface to a temporary STL file
surface_mesh = meshio.Mesh(points=verts, cells=[("triangle", faces)])
meshio.write("temp_surface.stl", surface_mesh)

# 5. Generate the Volume Mesh using Capitalized SVMTK Classes
surface = SVMTK.Surface("temp_surface.stl")  # Capitalized SVMTK
domain = SVMTK.Domain(surface)               # Capitalized SVMTK

print("Starting meshing process...")
domain.create_mesh(16)  # Resolution 16 for a fast initial test
print("Meshing complete!")

domain.save("raw_mesh.mesh")

# 6. Convert to FEniCS-compatible XDMF using Meshio
mesh = meshio.read("raw_mesh.mesh")
tetra_mesh = meshio.Mesh(
    points=mesh.points,
    cells={"tetra": mesh.cells_dict["tetra"]}
)
meshio.write("simulation_mesh.xdmf", tetra_mesh)
print("Mesh successfully generated: simulation_mesh.xdmf")