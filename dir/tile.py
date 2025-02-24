import math

from const import * 


class Tile:
    def __init__(self, i, j, radius):
        self.i = i
        self.j = j
        self.radius = radius
        self.tile_type = 0
        self.center = self.hexCenter()
        self.points = self.hexPoints()

    def hexCenter(self):
        x = self.j * 1.5 * self.radius + self.radius
        y = self.i * math.sqrt(3) * self.radius + self.radius
        if self.j % 2 == 1:
            y += math.sqrt(3) * self.radius / 2
        return (x, y)

    def hexPoints(self):
        points = []
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            x = self.center[0] + self.radius * math.cos(rad)
            y = self.center[1] + self.radius * math.sin(rad)
            points.append((x, y))
        return points

    def getNeighbors(self, rows, cols):
        neighbors = []
        if self.j % 2:
            directions = [(0, -1), (0, 1), (-1, 0), (1, -1), (1, 0), (1, 1)]
        else:
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]
        neighbors = [(self.i + x, self.j + y) for x, y in directions 
                     if (self.i + x >= 0 + GAP and self.i + x <= rows - GAP and self.j + y >= 0 + GAP and self.j + y <= cols - y)]
        return neighbors

    def getDeg(self, grid, type=MAP_VAL):
        deg = 0
        for i, j in self.getNeighbors(len(grid), len(grid[0])):
            if grid[i][j].tile_type == type:
                deg += 1
        return deg

    @property
    def color(self):
        return COLOR_MAPPING.get(self.tile_type, WHITE)
        

    @property
    def text_color(self):
        return WHITE if self.color in (BLACK, DARK_BLUE) else BLACK