import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh


def nifti_to_mesh(nifti_path, output_stl_path, threshold=0.5):
    """Generates a 3D STL mesh from a NIfTI segmentation file."""
    print(f"Loading NIfTI file from: {nifti_path}")
    # Load the NIfTI volume
    nii_img = nib.load(nifti_path)
    volume = nii_img.get_fdata()

    print(f"Extracting 3D surface mesh at threshold {threshold}...")
    # Generate 3D surface using Marching Cubes algorithm
    # step_size=1 provides the highest detail; increase for faster processing
    verts, faces, normals, values = measure.marching_cubes(
        volume, level=threshold, step_size=1
    )

    print(f"Applying spatial transformation to align with voxel spacing...")
    # Extract the affine matrix to transform voxel indices to physical coordinates
    affine = nii_img.affine
    # Transform vertices from voxel space to world coordinate space (RAS/LPS)
    verts = np.dot(
        np.hstack((verts, np.ones((verts.shape[0], 1)))), affine.T
    )[:, :3]

    print("Creating the mesh structure...")
    # Create the mesh object
    mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    out_mesh = mesh.Mesh(mesh_data)

    for i, f in enumerate(faces):
        for j in range(3):
            out_mesh.vectors[i][j] = verts[f[j], :]

    # Save the output file
    print(f"Saving STL mesh to: {output_stl_path}")
    out_mesh.save(output_stl_path)
    print("Mesh generation completed successfully!")


if __name__ == "__main__":
    # Replace these paths with your actual file locations
    input_file = "automated_segmentation.nii.gz"
    output_file = "output_model.stl"

    # For binary segmentations (0 and 1), a threshold of 0.5 is ideal
    nifti_to_mesh(input_file, output_file, threshold=0.5)