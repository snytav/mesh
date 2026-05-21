import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


def visualize_stl(stl_path):
    """Loads and visualizes an STL mesh using Matplotlib to bypass pyglet."""
    print(f"Loading mesh from: {stl_path}...")

    try:
        mesh = trimesh.load(stl_path)
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return

    print("Mesh loaded successfully!")
    print(f"Total Vertices: {len(mesh.vertices)}")
    print(f"Total Faces:    {len(mesh.faces)}")

    if len(mesh.vertices) == 0:
        print("Warning: The mesh is empty!")
        return

    # Print coordinate bounds to inspect the physical location
    bounds = mesh.bounds
    print(f"Mesh Bounds (Min X, Y, Z): {bounds[0]}")
    print(f"Mesh Bounds (Max X, Y, Z): {bounds[1]}")

    # Set up Matplotlib 3D Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Correct way to import and generate 3D triangle faces in modern Matplotlib
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_collection = Poly3DCollection(
        vertices[faces], alpha=0.6, edgecolor="k", linewidths=0.1
    )
    mesh_collection.set_facecolor([0.2, 0.6, 0.8])  # Soft blue color
    ax.add_collection3d(mesh_collection)

    # Normalize axes to maintain a perfect 1:1:1 aspect ratio
    all_x, all_y, all_z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array(
        [all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]
    ).max()

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.title(f"3D Preview: {stl_path}")

    print("Opening display window...")
    plt.show()


if __name__ == "__main__":
    visualize_stl("output_model.stl")