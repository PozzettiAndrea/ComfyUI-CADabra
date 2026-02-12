"""
Surface sampling utilities for Point2CAD.
Generates trimesh objects from fitted primitive parameters.
"""

import numpy as np
import trimesh


def sample_plane_mesh(normal, distance, extent=1.0, resolution=20):
    """
    Sample a plane as a trimesh quad mesh.

    Args:
        normal: Plane normal vector (3,)
        distance: Distance from origin along normal
        extent: Half-size of the plane in local coordinates
        resolution: Number of subdivisions per side

    Returns:
        trimesh.Trimesh object
    """
    normal = np.array(normal).flatten()
    normal = normal / (np.linalg.norm(normal) + 1e-10)

    # Find two orthogonal vectors in the plane
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Point on plane
    center = normal * distance

    # Create grid
    lin = np.linspace(-extent, extent, resolution)
    uu, vv = np.meshgrid(lin, lin)
    uu = uu.flatten()
    vv = vv.flatten()

    # Generate vertices
    vertices = center + uu[:, None] * u + vv[:, None] * v

    # Generate faces (quads as triangles)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # Two triangles per quad
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def sample_sphere_mesh(center, radius, resolution=20):
    """
    Sample a sphere as a trimesh.

    Args:
        center: Sphere center (3,)
        radius: Sphere radius
        resolution: Number of subdivisions (latitude/longitude)

    Returns:
        trimesh.Trimesh object
    """
    center = np.array(center).flatten()
    radius = float(radius)

    # Use trimesh's built-in sphere
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.vertices += center

    return sphere


def sample_cylinder_mesh(axis, center, radius, height_extent=1.0, resolution=32):
    """
    Sample a cylinder as a trimesh.

    Args:
        axis: Cylinder axis direction (3,)
        center: Point on cylinder axis (3,)
        radius: Cylinder radius
        height_extent: Half-height of cylinder along axis
        resolution: Number of circumferential subdivisions

    Returns:
        trimesh.Trimesh object
    """
    axis = np.array(axis).flatten()
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    center = np.array(center).flatten()
    radius = float(radius)

    # Create cylinder along Z axis, then transform
    height = height_extent * 2
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=resolution)

    # Rotation from Z to target axis
    z_axis = np.array([0, 0, 1])
    rotation_matrix = rotation_matrix_from_vectors(z_axis, axis)

    # Apply rotation and translation
    cylinder.apply_transform(rotation_matrix)
    cylinder.vertices += center

    return cylinder


def sample_cone_mesh(apex, axis, theta, height_extent=1.0, resolution=32):
    """
    Sample a cone as a trimesh.

    Args:
        apex: Cone apex point (3,)
        axis: Cone axis direction (pointing away from apex) (3,)
        theta: Half-angle in radians
        height_extent: Height of cone along axis from apex
        resolution: Number of circumferential subdivisions

    Returns:
        trimesh.Trimesh object
    """
    apex = np.array(apex).flatten()
    axis = np.array(axis).flatten()
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    theta = float(theta)

    # Cone radius at height h: r = h * tan(theta)
    height = height_extent
    base_radius = height * np.tan(theta)

    # Create cone along Z axis (apex at origin, base at z=height)
    cone = trimesh.creation.cone(radius=base_radius, height=height, sections=resolution)

    # Rotation from Z to target axis
    z_axis = np.array([0, 0, 1])
    rotation_matrix = rotation_matrix_from_vectors(z_axis, axis)

    # Apply rotation and translation (apex at origin initially)
    cone.apply_transform(rotation_matrix)
    cone.vertices += apex

    return cone


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Create a 4x4 rotation matrix that rotates vec1 to vec2.
    """
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    vec1 = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2 = vec2 / (np.linalg.norm(vec2) + 1e-10)

    cross = np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)

    if np.linalg.norm(cross) < 1e-10:
        if dot > 0:
            return np.eye(4)
        else:
            # 180 degree rotation - find perpendicular axis
            if abs(vec1[0]) < 0.9:
                perp = np.cross(vec1, [1, 0, 0])
            else:
                perp = np.cross(vec1, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            # Rodrigues formula for 180 degree rotation
            K = np.array([
                [0, -perp[2], perp[1]],
                [perp[2], 0, -perp[0]],
                [-perp[1], perp[0], 0]
            ])
            R = np.eye(3) + 2 * K @ K
            T = np.eye(4)
            T[:3, :3] = R
            return T

    # Rodrigues rotation formula
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + K + K @ K * (1 - dot) / (np.linalg.norm(cross) ** 2 + 1e-10)

    T = np.eye(4)
    T[:3, :3] = R
    return T


def sample_surface_mesh(surface, segment_points=None, resolution=32):
    """
    Sample a surface as a trimesh mesh.

    Args:
        surface: Surface dict with type, params, normalization
        segment_points: Optional points to determine extent (in world coords)
        resolution: Mesh resolution

    Returns:
        trimesh.Trimesh in world coordinates, or None if unsupported
    """
    surf_type = surface.get("type")
    params = surface.get("params")
    normalization = surface.get("normalization", {})
    offset = np.array(normalization.get("offset", [0, 0, 0]))
    scale = float(normalization.get("scale", 1.0))

    if params is None:
        # INR surface - use pre-computed mesh if available
        mesh = surface.get("mesh")
        if mesh is not None:
            # The INR mesh is in normalized space - transform to world
            # using the same normalization as primitives
            mesh = mesh.copy()  # Don't modify original
            mesh.vertices = mesh.vertices * scale + offset
            return mesh
        return None

    # Determine extent from segment points if available
    if segment_points is not None and len(segment_points) > 0:
        # Transform points to normalized space
        pts_norm = (segment_points - offset) / scale
        extent = np.max(np.abs(pts_norm)) * 1.5
    else:
        extent = 1.5  # Default extent in normalized space

    mesh = None

    if surf_type == "plane":
        normal, distance = params
        mesh = sample_plane_mesh(normal, distance, extent=extent, resolution=resolution)

    elif surf_type == "sphere":
        center, radius = params
        mesh = sample_sphere_mesh(center, radius, resolution=resolution)

    elif surf_type == "cylinder":
        axis, center, radius = params
        mesh = sample_cylinder_mesh(axis, center, radius, height_extent=extent, resolution=resolution)

    elif surf_type == "cone":
        apex, axis, theta = params
        mesh = sample_cone_mesh(apex, axis, theta, height_extent=extent, resolution=resolution)

    # Transform from normalized to world coordinates
    if mesh is not None:
        mesh.vertices = mesh.vertices * scale + offset

    return mesh


def sample_all_surfaces(surface_list, segment_points_map=None, resolution=32):
    """
    Sample all surfaces as meshes.

    Args:
        surface_list: List of surface dicts
        segment_points_map: Dict mapping segment_id -> points (world coords)
        resolution: Mesh resolution

    Returns:
        List of (surface_index, mesh) tuples
    """
    meshes = []

    for i, surface in enumerate(surface_list):
        seg_id = surface.get("segment_id")
        seg_pts = None
        if segment_points_map and seg_id in segment_points_map:
            seg_pts = segment_points_map[seg_id]

        mesh = sample_surface_mesh(surface, seg_pts, resolution)
        if mesh is not None:
            meshes.append((i, mesh))

    return meshes
