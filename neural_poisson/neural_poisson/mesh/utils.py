import open3d as o3d
from pytorch3d.structures import Meshes


def pytorch3d_to_open3d(mesh: Meshes):
    verts = mesh.verts_packed().cpu().numpy()  # Convert to NumPy
    faces = mesh.faces_packed().cpu().numpy()  # Convert to NumPy
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()  # Compute normals for better visualization
    return o3d_mesh
