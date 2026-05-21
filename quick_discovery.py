import h5py

with h5py.File("fsi_results.h5", "r") as f:
    print("--- Exact HDF5 Internal File Structure ---")
    def print_node(name):
        print(f"  Node: {name}")
    f.visit(print_node)
    print("------------------------------------------")