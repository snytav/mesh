import SVMTK

# 1. Define 3D points
p1 = SVMTK.Point_3(0.0, 0.0, 0.0)
p2 = SVMTK.Point_3(1.0, 0.0, 0.0)
p3 = SVMTK.Point_3(0.0, 1.0, 0.0)

print("SVMTK Geometric Objects initiated successfully!")
print(f"Point 1 coordinates: {p1.x()}, {p1.y()}, {p1.z()}")
