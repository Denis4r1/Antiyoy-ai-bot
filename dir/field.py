import math
import random
from collections import deque

from const import *
from tile import Tile


class Field:
    def __init__(self, rows, cols, radius):
        self.rows = rows
        self.cols = cols
        self.radius = radius
        self.grid = [[Tile(i, j, radius) for j in range(cols)] for i in range(rows)]
        self.width = int(cols * math.sqrt(3) * radius * 0.89)
        self.height = int(rows * math.sqrt(3) * radius + 2 * radius)
        self.generateMap()
        

    def generateMap(self):
        islands = self.genRect()
        self.connectIslands(islands)
        self.addMoreTiles()
        
    def genRect(self):
        islands = []
        sizes = [((3,4), (3,4), 3), ((2,3), (2,3), 3), ((1, 2), (1, 2), 5)]
        
        for size_type in sizes:
            for _ in range(size_type[-1]):
                placed = False
                attempts = 0
                
                while not placed and attempts < 100:
                    width = random.randint(size_type[0][0], size_type[0][1])
                    height = random.randint(size_type[1][0], size_type[1][1])
                    start_i = random.randint(GAP, max(2, self.rows - height - GAP))
                    start_j = random.randint(GAP, max(2, self.cols - width - GAP))
                    
                    if self.isFree(start_i, start_j, width, height):
                        self.placeRect(start_i, start_j, width, height)
                        islands.append((start_i, start_j, width, height))
                        placed = True

                    attempts += 1

        return islands
        

    def isFree(self, start_i, start_j, width, height):
        check_start_i = max(0, start_i - 1)
        check_end_i = min(self.rows, start_i + height + 1)
        check_start_j = max(0, start_j - 1)
        check_end_j = min(self.cols, start_j + width + 1)
        
        for i in range(check_start_i, check_end_i):
            for j in range(check_start_j, check_end_j):
                if self.grid[i][j].tile_type != 0:
                    return False
        return True


    def placeRect(self, start_i, start_j, width, height):
        for i in range(start_i, start_i + height):
            for j in range(start_j, start_j + width):
                self.grid[i][j].tile_type = ISLAND_VAL


    def connectIslands(self, islands):
        centers = set([(island[0] + island[3] // 2, island[1] + island[2] // 2) for island in islands])
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        center = centers.pop()
        self.grid[center[0]][center[1]].tile_type = MAP_VAL 
        queue = deque([(center[0], center[1], [])])
        visited[center[0]][center[1]] = True
        def addBridge(path):
            for tile in path:
                self.grid[tile[0]][tile[1]].tile_type = MAP_VAL
        while queue:
            i, j, path = queue.popleft()
            for y, x in self.grid[i][j].getNeighbors(self.rows, self.cols):
                if visited[y][x]:
                    continue
                if self.grid[y][x].tile_type == ISLAND_VAL:
                    self.grid[y][x].tile_type = MAP_VAL
                if (y, x) in centers:
                    addBridge(path)
                    path = []
                visited[y][x] = True
                temp = path.copy()
                temp.append((y, x))
                queue.append((y, x, temp))


    def addMoreTiles(self):
        for row in self.grid:
            for tile in row:
                if tile.tile_type == MAP_VAL and tile.getDeg(self.grid, MAP_VAL) <= 2:
                    for i, j in tile.getNeighbors(self.rows, self.cols):
                        if self.grid[i][j].tile_type != MAP_VAL and 2 <= self.grid[i][j].getDeg(self.grid, MAP_VAL) <= 4:
                            self.grid[i][j].tile_type = MAP_VAL if random.randint(0, 10000) % 3 == 0 else self.grid[i][j].tile_type