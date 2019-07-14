import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector
from hummingbird.tools.rotations import Euler2Rotation
from hummingbird.tools.points_transformations import rotate_points, translate_points, get_mav_points, points_to_mesh


class MavViewer:
    def __init__(self):
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('MAV Viewer')
        self.window.setGeometry(0, 0, 800, 600)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem()  # make a grid to represent the ground
        grid.scale(100, 100, 100)  # set the size of the grid (distance between each line)
        self.window.addItem(grid)  # add grid to viewer
        self.window.setCameraPosition(distance=400) # distance from center of plot to camera
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the spacecraft been plotted yet?
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.points, self.meshColors = get_mav_points()

    ###################################
    # public functions
    def update(self, state):
        """
        Update the drawing of the spacecraft.
        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed to be:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = rotate_points(self.points, R)
        translated_points = translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.meshColors, # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.meshColors)

        # update the center of the camera view to the spacecraft location
        view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    ###################################
    # private functions

