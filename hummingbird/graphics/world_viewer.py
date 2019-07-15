"""
mavsim_python: world viewer (for chapter 12)
    - Beard & McLain, PUP, 2012
    - Update history:
        4/3/2019 - BGM
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from hummingbird.tools.points_transformations import rotate_points, translate_points, get_mav_points, points_to_mesh, orbit_points, \
    straight_waypoint_points
from hummingbird.tools.rotations import Euler2Rotation
from hummingbird.guidance.dubin_parameters import DubinsParameters
from hummingbird.tools.wrap import mod


class WorldViewer:
    def __init__(self):
        self.scale = 4000
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Path Viewer')
        self.window.setGeometry(0, 0, 800, 600)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem()  # make a grid to represent the ground
        grid.scale(self.scale / 20, self.scale / 20,
                   self.scale / 20)  # set the size of the grid (distance between each line)
        self.window.addItem(grid)  # add grid to viewer
        self.window.setCameraPosition(distance=self.scale, elevation=50, azimuth=-90)
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_()  # bring window to the front
        self.plot_initialized = False  # has the mav been plotted yet?
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = get_mav_points()
        # dubins path parameters
        self.dubins_path = DubinsParameters()
        self.mav_body = []

    ###################################
    # public functions
    def update(self, map, waypoints, path, state):

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            self.draw_mav(state)
            self.draw_waypoints(waypoints, path.orbit_radius)
            self.draw_path(path)
            self.drawMap(map)
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            self.draw_mav(state)
            if waypoints.flag_waypoints_changed == True:
                self.draw_waypoints(waypoints, path.orbit_radius)
            if path.flag_path_changed == True:
                self.draw_path(path)

        # update the center of the camera view to the mav location
        # view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        # self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    def draw_mav(self, state):
        """
        Update the drawing of the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = rotate_points(self.mav_points, R)
        translated_points = translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points)
        if not self.plot_initialized:
            # initialize drawing of triangular mesh.
            self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                          vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                          drawEdges=True,  # draw edges between mesh elements
                                          smooth=False,  # speeds up rendering
                                          computeNormals=False)  # speeds up rendering
            self.window.addItem(self.mav_body)  # add body to plot
        else:
            # draw MAV by resetting mesh using rotated and translated points
            self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def draw_path(self, path):
        red = np.array([[1., 0., 0., 1]])
        if path.type == 'line':
            points = self.straight_line_points(path)
        elif path.type == 'orbit':
            points = orbit_points(path)
        else:
            raise TypeError("Wrong path type: {}".format(path.type))

        if not self.plot_initialized:
            path_color = np.tile(red, (points.shape[0], 1))
            self.path = gl.GLLinePlotItem(pos=points,
                                          color=path_color,
                                          width=2,
                                          antialias=True,
                                          mode='line_strip')
            # mode='line_strip')
            self.window.addItem(self.path)
        else:
            self.path.setData(pos=points)

    def straight_line_points(self, path):
        points = np.array([[path.line_origin.item(0),
                            path.line_origin.item(1),
                            path.line_origin.item(2)],
                           [path.line_origin.item(0) + self.scale * path.line_direction.item(0),
                            path.line_origin.item(1) + self.scale * path.line_direction.item(1),
                            path.line_origin.item(2) + self.scale * path.line_direction.item(2)]])
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        return points

    def draw_waypoints(self, waypoints, radius):
        blue = np.array([[30, 144, 255, 255]]) / 255.
        if waypoints.type == 'straight_line' or waypoints.type == 'fillet':
            points = straight_waypoint_points(waypoints)
        elif waypoints.type == 'dubins':
            points = self.dubins_points(waypoints, radius, 0.1)
        else:
            raise TypeError("Unexpected waypoint type: {}".format(waypoints.type))

        if not self.plot_initialized:
            waypoint_color = np.tile(blue, (points.shape[0], 1))
            self.waypoints = gl.GLLinePlotItem(pos=points,
                                               color=waypoint_color,
                                               width=2,
                                               antialias=True,
                                               mode='line_strip')
            self.window.addItem(self.waypoints)
        else:
            self.waypoints.setData(pos=points)

    def dubins_points(self, waypoints, radius, Del):
        initialize_points = True
        for j in range(0, waypoints.num_waypoints - 1):
            self.dubins_path.update(
                waypoints.ned[j],
                waypoints.course[j],
                waypoints.ned[j + 1],
                waypoints.course[j + 1],
                radius)

            # points along start circle
            th1 = np.arctan2(self.dubins_path.p_s.item(1) - self.dubins_path.center_s.item(1),
                             self.dubins_path.p_s.item(0) - self.dubins_path.center_s.item(0))
            th1 = mod(th1)
            th2 = np.arctan2(self.dubins_path.r1.item(1) - self.dubins_path.center_s.item(1),
                             self.dubins_path.r1.item(0) - self.dubins_path.center_s.item(0))
            th2 = mod(th2)
            th = th1
            theta_list = [th]
            if self.dubins_path.dir_s > 0:
                if th1 >= th2:
                    while th < th2 + 2 * np.pi:
                        th += Del
                        theta_list.append(th)
                else:
                    while th < th2:
                        th += Del
                        theta_list.append(th)
            else:
                if th1 <= th2:
                    while th > th2 - 2 * np.pi:
                        th -= Del
                        theta_list.append(th)
                else:
                    while th > th2:
                        th -= Del
                        theta_list.append(th)

            if initialize_points:
                points = np.array([[self.dubins_path.center_s.item(0) + self.dubins_path.radius * np.cos(theta_list[0]),
                                    self.dubins_path.center_s.item(1) + self.dubins_path.radius * np.sin(theta_list[0]),
                                    self.dubins_path.center_s.item(2)]])
                initialize_points = False
            for angle in theta_list:
                new_point = np.array([[self.dubins_path.center_s.item(0) + self.dubins_path.radius * np.cos(angle),
                                       self.dubins_path.center_s.item(1) + self.dubins_path.radius * np.sin(angle),
                                       self.dubins_path.center_s.item(2)]])
                points = np.concatenate((points, new_point), axis=0)

            # points along straight line
            sig = 0
            while sig <= 1:
                new_point = np.array([[(1 - sig) * self.dubins_path.r1.item(0) + sig * self.dubins_path.r2.item(0),
                                       (1 - sig) * self.dubins_path.r1.item(1) + sig * self.dubins_path.r2.item(1),
                                       (1 - sig) * self.dubins_path.r1.item(2) + sig * self.dubins_path.r2.item(2)]])
                points = np.concatenate((points, new_point), axis=0)
                sig += Del

            # points along end circle
            th2 = np.arctan2(self.dubins_path.p_e.item(1) - self.dubins_path.center_e.item(1),
                             self.dubins_path.p_e.item(0) - self.dubins_path.center_e.item(0))
            th2 = mod(th2)
            th1 = np.arctan2(self.dubins_path.r2.item(1) - self.dubins_path.center_e.item(1),
                             self.dubins_path.r2.item(0) - self.dubins_path.center_e.item(0))
            th1 = mod(th1)
            th = th1
            theta_list = [th]
            if self.dubins_path.dir_e > 0:
                if th1 >= th2:
                    while th < th2 + 2 * np.pi:
                        th += Del
                        theta_list.append(th)
                else:
                    while th < th2:
                        th += Del
                        theta_list.append(th)
            else:
                if th1 <= th2:
                    while th > th2 - 2 * np.pi:
                        th -= Del
                        theta_list.append(th)
                else:
                    while th > th2:
                        th -= Del
                        theta_list.append(th)
            for angle in theta_list:
                new_point = np.array([[self.dubins_path.center_e.item(0) + self.dubins_path.radius * np.cos(angle),
                                       self.dubins_path.center_e.item(1) + self.dubins_path.radius * np.sin(angle),
                                       self.dubins_path.center_e.item(2)]])
                points = np.concatenate((points, new_point), axis=0)

        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        return points

    def drawMap(self, map):
        # draw map of the world: buildings
        fullMesh = np.array([], dtype=np.float32).reshape(0, 3, 3)
        fullMeshColors = np.array([], dtype=np.float32).reshape(0, 3, 4)
        for i in range(0, map.num_city_blocks):
            for j in range(0, map.num_city_blocks):
                mesh, meshColors = self.buildingVertFace(map.building_north[i],
                                                         map.building_east[j],
                                                         map.building_width,
                                                         map.building_height[j, i])
                fullMesh = np.concatenate((fullMesh, mesh), axis=0)
                fullMeshColors = np.concatenate((fullMeshColors, meshColors), axis=0)
        self.map = gl.GLMeshItem(vertexes=fullMesh,  # defines the triangular mesh (Nx3x3)
                                 vertexColors=fullMeshColors,  # defines mesh colors (Nx1)
                                 drawEdges=True,  # draw edges between mesh elements
                                 smooth=False,  # speeds up rendering
                                 computeNormals=False)  # speeds up rendering
        self.window.addItem(self.map)

    def buildingVertFace(self, n, e, width, height):
        # define patches for a building located at (x, y)
        # vertices of the building
        points = np.array([[e + width / 2, n + width / 2, 0],  # NE 0
                           [e + width / 2, n - width / 2, 0],  # SE 1
                           [e - width / 2, n - width / 2, 0],  # SW 2
                           [e - width / 2, n + width / 2, 0],  # NW 3
                           [e + width / 2, n + width / 2, height],  # NE Higher 4
                           [e + width / 2, n - width / 2, height],  # SE Higher 5
                           [e - width / 2, n - width / 2, height],  # SW Higher 6
                           [e - width / 2, n + width / 2, height]])  # NW Higher 7
        mesh = np.array([[points[0], points[3], points[4]],  # North Wall
                         [points[7], points[3], points[4]],  # North Wall
                         [points[0], points[1], points[5]],  # East Wall
                         [points[0], points[4], points[5]],  # East Wall
                         [points[1], points[2], points[6]],  # South Wall
                         [points[1], points[5], points[6]],  # South Wall
                         [points[3], points[2], points[6]],  # West Wall
                         [points[3], points[7], points[6]],  # West Wall
                         [points[4], points[7], points[5]],  # Top
                         [points[7], points[5], points[6]]])  # Top

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        mesh_colors = np.empty((10, 3, 4), dtype=np.float32)
        mesh_colors[0] = green
        mesh_colors[1] = green
        mesh_colors[2] = green
        mesh_colors[3] = green
        mesh_colors[4] = green
        mesh_colors[5] = green
        mesh_colors[6] = green
        mesh_colors[7] = green
        mesh_colors[8] = yellow
        mesh_colors[9] = yellow
        return mesh, mesh_colors

