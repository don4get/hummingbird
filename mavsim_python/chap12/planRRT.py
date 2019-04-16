import numpy as np
from message_types.msg_waypoints import msg_waypoints


class planRRT():
    def __init__(self):
        self.waypoints = msg_waypoints()
        self.segmentLength = 300 # standard length of path segments

    def planPath(self, wpp_start, wpp_end, map):
        # desired down position is down position of end node
        pd = wpp_end[2]

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start[0], wpp_start[1], pd, 0, 0, 0])
        end_node = np.array([wpp_end[0], wpp_end[1], pd, 0, 0, 0])

        # establish tree starting with the start node
        tree = start_node

        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[:3] - end_node[:3]) < self.segmentLength ) 
            and not self.collision(start_node, end_node, map)):
            self.waypoints.ned = end_node[:3]
        else:
            numPaths = 0
            while numPaths < 3:
                tree, flag = self.extendTree(tree, end_node, self.segmentLength, map, pd)
                numPaths = numPaths + flag

        self.extendTree(tree, end_node, self.segmentLength, map, pd)


        # find path with minimum cost to end_node
        path = self.findMinimumPath(tree, end_node)
        return self.smoothPath(path, map)

    def generateRandomPoint(map, pd):
        pass

    def collision(start_node, end_node, map):
        pass

    def pointsAlongPath(start_node, end_node, Del):
        pass

    def downAtNE(map, n, e):
        pass

    def extendTree(tree, end_node, segmentLength, map, pd):
        node = self.generateRandomNode(map, pd)
        

    def findMinimumPath(tree, end_node):
        pass

    def smoothPath(path, map):
        pass

