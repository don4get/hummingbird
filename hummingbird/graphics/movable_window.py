import sys
from PyQt5 import QtCore
from pyqtgraph import GraphicsWindow


class MovableWindow(GraphicsWindow):
    def __init__(self, *args, **kwargs):
        print("Press Ctrl-Q to exit...")
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            print("\nDetected 'Q' keypress: Exiting\n")
            sys.exit()

    def location_on_the_screen(self, x=0, y=0):
        # screen = QDesktopWidget().screenGeometry()
        # widget = self.geometry()
        # x = screen.width() - widget.width()
        # y = screen.height() - widget.height()
        self.move(x, y)
