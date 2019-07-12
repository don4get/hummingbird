import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from hummingbird.tools.rotations import Euler2Rotation
from hummingbird.tools.points_transformations import rotate_points, translate_points, get_mav_points, points_to_mesh

class PathViewer:
    def __init__(self):
        self.scale = 4000
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Path Viewer')
        self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem()  # make a grid to represent the ground
        grid.scale(self.scale / 20, self.scale / 20,
                   self.scale / 20)  # set the size of the grid (distance between each line)
        self.window.addItem(grid)  # add grid to viewer
        self.window.setCameraPosition(distance=self.scale, elevation=90, azimuth=0)
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_()  # bring window to the front
        self.plot_initialized = False  # has the mav been plotted yet?
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.points, self.meshColors = get_mav_points()

    ###################################
    # public functions
    def update(self, path, state):
        """
        Update the drawing of the mav.

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
        rotated_points = rotate_points(self.points, R)
        translated_points = translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            if path.type == 'line':
                straight_line_object = self.straight_line_plot(path)
                self.window.addItem(straight_line_object)  # add straight line to plot
            else:  # path.type=='orbit
                orbit_object = orbit_plot(path)
                self.window.addItem(orbit_object)
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.meshColors)

        # update the center of the camera view to the mav location
        # view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        # self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    ###################################
    # private functions

    def straight_line_plot(self, path):
        points = np.array([[path.line_origin.item(0),
                            path.line_origin.item(1),
                            path.line_origin.item(2)],
                           [path.line_origin.item(0) + self.scale * path.line_direction.item(0),
                            path.line_origin.item(1) + self.scale * path.line_direction.item(1),
                            path.line_origin.item(2) + self.scale * path.line_direction.item(2)]])
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        red = np.array([[1., 0., 0., 1]])
        path_color = np.concatenate((red, red), axis=0)
        object = gl.GLLinePlotItem(pos=points,
                                   color=path_color,
                                   width=2,
                                   antialias=True,
                                   mode='lines')
        return object


def orbit_plot(path):
    N = 100
    red = np.array([[1., 0., 0., 1]])
    theta = 0
    points = np.array([[path.orbit_center.item(0) + path.orbit_radius,
                        path.orbit_center.item(1),
                        path.orbit_center.item(2)]])
    path_color = red
    for i in range(0, N):
        theta += 2 * np.pi / N
        new_point = np.array([[path.orbit_center.item(0) + path.orbit_radius * np.cos(theta),
                               path.orbit_center.item(1) + path.orbit_radius * np.sin(theta),
                               path.orbit_center.item(2)]])
        points = np.concatenate((points, new_point), axis=0)
        path_color = np.concatenate((path_color, red), axis=0)
    # convert North-East Down to East-North-Up for rendering
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    points = points @ R.T
    obj = gl.GLLinePlotItem(pos=points,
                            color=path_color,
                            width=2,
                            antialias=True,
                            mode='line_strip')
    return obj
