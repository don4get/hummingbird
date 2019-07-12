import numpy as np


def rotate_points(points, R):
    """
    Rotate points by the rotation matrix R
    """
    rotated_points = R @ points
    return rotated_points


def translate_points(points, translation):
    """
    Translate points by the vector translation
    """
    translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
    return translated_points


def get_mav_points():
    """
    Points that define the mav, and the colors of the triangular mesh
    Define the points on the aircraft following diagram in Figure C.3
    """
    # define MAV body parameters
    unit_length = 0.25
    fuse_h = unit_length
    fuse_w = unit_length
    fuse_l1 = unit_length * 2
    fuse_l2 = unit_length
    fuse_l3 = unit_length * 4
    wing_l = unit_length
    wing_w = unit_length * 6
    tail_h = unit_length
    tail_l = unit_length
    tail_w = unit_length * 2

    # points are in NED coordinates
    #   define the points on the aircraft following diagram Fig 2.14
    points = np.array([[fuse_l1, 0, 0],  # point 1 [0]
                       [fuse_l2, fuse_w / 2.0, -fuse_h / 2.0],  # point 2 [1]
                       [fuse_l2, -fuse_w / 2.0, -fuse_h / 2.0],  # point 3 [2]
                       [fuse_l2, -fuse_w / 2.0, fuse_h / 2.0],  # point 4 [3]
                       [fuse_l2, fuse_w / 2.0, fuse_h / 2.0],  # point 5 [4]
                       [-fuse_l3, 0, 0],  # point 6 [5]
                       [0, wing_w / 2.0, 0],  # point 7 [6]
                       [-wing_l, wing_w / 2.0, 0],  # point 8 [7]
                       [-wing_l, -wing_w / 2.0, 0],  # point 9 [8]
                       [0, -wing_w / 2.0, 0],  # point 10 [9]
                       [-fuse_l3 + tail_l, tail_w / 2.0, 0],  # point 11 [10]
                       [-fuse_l3, tail_w / 2.0, 0],  # point 12 [11]
                       [-fuse_l3, -tail_w / 2.0, 0],  # point 13 [12]
                       [-fuse_l3 + tail_l, -tail_w / 2.0, 0],  # point 14 [13]
                       [-fuse_l3 + tail_l, 0, 0],  # point 15 [14]
                       [-fuse_l3, 0, -tail_h],  # point 16 [15]
                       ]).T

    # scale points for better rendering
    scale = 50
    points = scale * points

    #   define the colors for each face of triangular mesh
    red = np.array([1., 0., 0., 1])
    green = np.array([0., 1., 0., 1])
    blue = np.array([0., 0., 1., 1])
    yellow = np.array([1., 1., 0., 1])
    mesh_colors = np.empty((13, 3, 4), dtype=np.float32)
    mesh_colors[0] = yellow  # nose-top
    mesh_colors[1] = yellow  # nose-right
    mesh_colors[2] = yellow  # nose-bottom
    mesh_colors[3] = yellow  # nose-left
    mesh_colors[4] = blue  # fuselage-left
    mesh_colors[5] = blue  # fuselage-top
    mesh_colors[6] = blue  # fuselage-right
    mesh_colors[7] = red  # fuselage-bottom
    mesh_colors[8] = green  # wing
    mesh_colors[9] = green  # wing
    mesh_colors[10] = green  # horizontal tail
    mesh_colors[11] = green  # horizontal tail
    mesh_colors[12] = blue  # vertical tail
    return points, mesh_colors


def points_to_mesh(points):
    """"
    Converts points to triangular mesh
    Each mesh face is defined by three 3D points
      (a rectangle requires two triangular mesh faces)
    """
    points = points.T
    mesh = np.array([[points[0], points[1], points[2]],  # nose-top
                     [points[0], points[1], points[4]],  # nose-right
                     [points[0], points[3], points[4]],  # nose-bottom
                     [points[0], points[3], points[2]],  # nose-left
                     [points[5], points[2], points[3]],  # fuselage-left
                     [points[5], points[1], points[2]],  # fuselage-top
                     [points[5], points[1], points[4]],  # fuselage-right
                     [points[5], points[3], points[4]],  # fuselage-bottom
                     [points[6], points[7], points[9]],  # wing
                     [points[7], points[8], points[9]],  # wing
                     [points[10], points[11], points[12]],  # horizontal tail
                     [points[10], points[12], points[13]],  # horizontal tail
                     [points[5], points[14], points[15]],  # vertical tail
                     ])
    return mesh


def orbit_points(path):
    N = 10
    theta = 0
    theta_list = [theta]
    while theta < 2 * np.pi:
        theta += 0.1
        theta_list.append(theta)
    points = np.array([[path.orbit_center.item(0) + path.orbit_radius,
                        path.orbit_center.item(1),
                        path.orbit_center.item(2)]])
    for angle in theta_list:
        new_point = np.array([[path.orbit_center.item(0) + path.orbit_radius * np.cos(angle),
                               path.orbit_center.item(1) + path.orbit_radius * np.sin(angle),
                               path.orbit_center.item(2)]])
        points = np.concatenate((points, new_point), axis=0)
    # convert North-East Down to East-North-Up for rendering
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    points = points @ R.T
    return points


def straight_waypoint_points(waypoints):
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    wps = np.copy(waypoints.ned)
    wps = wps[~np.all(np.isinf(wps), 1)]
    points = R @ wps.T
    return points.T
