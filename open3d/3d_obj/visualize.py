import open3d as o3d
import numpy as np

# armadillo_mesh = o3d.data.ArmadilloMesh()
# mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

file_path = "./CapturaPly3_200.ply"
mesh = o3d.io.read_triangle_mesh(file_path)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])

# print("Try to render a mesh with normals (exist: " +
#       str(mesh.has_vertex_normals()) + ") and colors (exist: " +
#       str(mesh.has_vertex_colors()) + ")")
# o3d.visualization.draw_geometries([mesh])
# print("A mesh with no normals and no colors does not look good.")


