import numpy as np
import open3d as o3d


def visualize_scene(path, threshold: float = 1.0):

    REMOVE_DIST = threshold
    # Read the point cloud from file
    point_cloud = o3d.io.read_point_cloud(str(path.resolve()))

    # Convert point cloud to numpy array for easy processing
    points = np.asarray(point_cloud.points)

    # Calculate the distance from the origin (0, 0, 0)
    distances = np.linalg.norm(points, axis=1)

    # Filter out points that are more than 1 unit away from the origin
    filtered_points = points[distances <= REMOVE_DIST]

    # Get color information (if it exists) and filter it
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_colors = colors[distances <= REMOVE_DIST]
    else:
        filtered_colors = None

    # Get normal information (if it exists) and filter it
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        filtered_normals = normals[distances <= REMOVE_DIST]
    else:
        filtered_normals = None

    # Create a new point cloud with the filtered points
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    # Assign the filtered color information (if it exists)
    if filtered_colors is not None:
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Assign the filtered normal information (if it exists)
    if filtered_normals is not None:
        filtered_point_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)
    o3d.visualization.draw_plotly(
        [filtered_point_cloud], up=np.array([1.0, -1.0, -0.9])
    )
