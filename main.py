import math
import heapq
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle, Arrow

# each cell in the matrix
class Cell:
    def __init__(self):
        self.parent_i = 0  # parent cell's row index
        self.parent_j = 0  # parent cell's column index
        self.f = float('inf')  # total cost of the cell (g + h)
        self.g = float('inf')  # cost from start to this cell
        self.h = 0  # heuristic cost from this cell to destination

# for field setup
class VEXField:
    # cell types
    BLOCKED = 0
    EMPTY = 1
    RING = 2
    SCORING_ZONE = 3
    ROBOT = 4
    PATH = 5
    
    def __init__(self, rows=12, cols=12):
        self.rows = rows
        self.cols = cols
        self.grid = [[self.EMPTY for _ in range(cols)] for _ in range(rows)]
        self.rings = [] 
        self.scoring_zones = []  
        self.robot_pos = None
        
    def set_cell(self, row, col, cell_type):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = cell_type
            
            if cell_type == self.RING:
                self.rings.append((row, col))
            elif cell_type == self.SCORING_ZONE:
                self.scoring_zones.append((row, col))
            elif cell_type == self.ROBOT:
                self.robot_pos = (row, col)
                
    def get_cell(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return self.BLOCKED 
    
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.rows) and (col >= 0) and (col < self.cols)
    
    def is_unblocked(self, row, col):
        return self.get_cell(row, col) != self.BLOCKED
    
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]
    
    def calculate_h_value(self, row, col, dest):
        # euclidean distance heuristic
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5
    
    def visualize(self, path=None, title="VEX Field"):
        # create a colormap for the field
        cmap = ListedColormap(['black', 'white', 'gold', 'green', 'blue', 'red'])
        
        # create a copy of the grid for visualization
        vis_grid = np.array(self.grid, dtype=int)
        
        # mark the path on the grid
        if path:
            for row, col in path:
                if self.grid[row][col] != self.ROBOT and self.grid[row][col] != self.RING and self.grid[row][col] != self.SCORING_ZONE:
                    vis_grid[row][col] = self.PATH
        
        # create the figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # plot the grid
        ax.imshow(vis_grid, cmap=cmap, interpolation='nearest')
        
        # add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        
        # add labels for cells
        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = vis_grid[row][col]
                if cell_type == self.RING:
                    ax.add_patch(Circle((col, row), 0.3, fill=True, color='gold'))
                elif cell_type == self.SCORING_ZONE:
                    ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, alpha=0.5, color='green'))
                elif cell_type == self.ROBOT:
                    ax.add_patch(Rectangle((col-0.3, row-0.3), 0.6, 0.6, fill=True, color='blue'))
        
        # draw arrows for the path
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                start_row, start_col = path[i]
                end_row, end_col = path[i + 1]
                
                # calculate direction vector
                dy = end_row - start_row
                dx = end_col - start_col
                
                # draw an arrow
                ax.arrow(start_col, start_row, dx * 0.7, dy * 0.7, 
                         head_width=0.15, head_length=0.15, fc='red', ec='red', width=0.05)
        
        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        plt.savefig("path_plot.png") 
        plt.close()  
    
    def a_star_search(self, src, dest):
        # check if the source and destination are valid
        if not self.is_valid(src[0], src[1]) or not self.is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return None
        
        # check if the source and destination are unblocked
        if not self.is_unblocked(src[0], src[1]) or not self.is_unblocked(dest[0], dest[1]):
            print("Source or the destination is blocked")
            return None
        
        # check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return [src]
        
        # initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        
        # initialize the details of each cell
        cell_details = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]
        
        # initialize the start cell details
        i, j = src
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j
        
        # initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
        
        # initialize the flag for whether destination is found
        found_dest = False
        
        # main loop of A* search algorithm
        while open_list:
            # pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)
            
            # mark the cell as visited
            i, j = p[1], p[2]
            closed_list[i][j] = True
            
            # for each direction, check the successors
            # include diagonal moves for more flexible path planning
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            
            for dir_i, dir_j in directions:
                new_i, new_j = i + dir_i, j + dir_j
                
                # if the successor is valid, unblocked, and not visited
                if (self.is_valid(new_i, new_j) and 
                    self.is_unblocked(new_i, new_j) and 
                    not closed_list[new_i][new_j]):
                    
                    # if the successor is the destination
                    if self.is_destination(new_i, new_j, dest):
                        # set the parent of the destination cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # trace the path
                        path = self._trace_path(cell_details, dest)
                        return path
                    
                    else:
                        # calculate movement cost (diagonal moves cost more)
                        move_cost = 1.0 if (dir_i == 0 or dir_j == 0) else 1.414
                        
                        # add extra cost for moving through certain cells (optional)
                        cell_type = self.get_cell(new_i, new_j)
                        # may add ring stack extra cost because it takes longer to intake, but for path plannign purposes since the stacks are alr at the end its not necessary
                        
                        # calculate the new f, g, and h values
                        g_new = cell_details[i][j].g + move_cost
                        h_new = self.calculate_h_value(new_i, new_j, dest)
                        f_new = g_new + h_new
                        
                        # if the cell is not in the open list or the new f value is smaller
                        if (cell_details[new_i][new_j].f == float('inf') or 
                            cell_details[new_i][new_j].f > f_new):
                            
                            # add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            
                            # update the cell details
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j
        
        # if the destination is not found after visiting all cells
        print("Failed to find the destination cell")
        return None
    
    def _trace_path(self, cell_details, dest):
        print("Tracing path...")
        path = []
        row, col = dest
        
        # trace the path from destination to source using parent cells
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row, col = temp_row, temp_col
        
        # add the source cell to the path
        path.append((row, col))
        
        # reverse the path to get the path from source to destination
        path.reverse()
        
        print("Path found with", len(path), "steps")
        return path
    
    def plan_optimal_path(self, collect_rings=True):
        # start with the robot's position
        current_pos = self.robot_pos
        complete_path = [current_pos]
        
        # if we need to collect rings first
        if collect_rings:
            print(f"Planning to collect {len(self.rings)} rings")
            
            # greedy approach: collect nearest ring first
            rings_to_collect = self.rings.copy()
            
            while rings_to_collect:
                # find the nearest ring
                nearest_ring = min(rings_to_collect, key=lambda ring: 
                                  ((ring[0] - current_pos[0]) ** 2 + 
                                   (ring[1] - current_pos[1]) ** 2) ** 0.5)
                
                # find path to the nearest ring
                print(f"Finding path from {current_pos} to ring at {nearest_ring}")
                path_to_ring = self.a_star_search(current_pos, nearest_ring)
                
                if path_to_ring:
                    # add the path to the complete path (excluding the start point to avoid duplicates)
                    complete_path.extend(path_to_ring[1:])
                    # update current position
                    current_pos = nearest_ring
                    # remove the collected ring
                    rings_to_collect.remove(nearest_ring)
                else:
                    print(f"Could not find path to ring at {nearest_ring}")
                    # skip this ring
                    rings_to_collect.remove(nearest_ring)
        
        # find the nearest scoring zone
        nearest_zone = min(self.scoring_zones, key=lambda zone: 
                           ((zone[0] - current_pos[0]) ** 2 + 
                            (zone[1] - current_pos[1]) ** 2) ** 0.5)
        
        # find path to the nearest scoring zone
        print(f"Finding path from {current_pos} to scoring zone at {nearest_zone}")
        path_to_zone = self.a_star_search(current_pos, nearest_zone)
        
        if path_to_zone:
            # add the path to the complete path (excluding the start point to avoid duplicates)
            complete_path.extend(path_to_zone[1:])
        else:
            print(f"Could not find path to scoring zone at {nearest_zone}")
        
        return complete_path

def main():
    # create field
    field = VEXField(13, 13)
    field.set_cell(6, 0, VEXField.SCORING_ZONE)
    field.set_cell(10, 1, VEXField.RING)
    field.set_cell(2, 1, VEXField.RING)
    field.set_cell(11, 2, VEXField.RING)
    field.set_cell(10, 2, VEXField.RING)
    field.set_cell(8, 2, VEXField.SCORING_ZONE)
    field.set_cell(4, 2, VEXField.SCORING_ZONE)
    field.set_cell(2, 2, VEXField.RING)
    field.set_cell(1, 2, VEXField.RING)
    field.set_cell(10, 4, VEXField.RING)
    field.set_cell(8, 4, VEXField.RING)
    field.set_cell(4, 4, VEXField.RING)
    field.set_cell(2, 4, VEXField.RING)
    field.set_cell(1, 6, VEXField.RING)
    field.set_cell(6, 6, VEXField.RING)
    field.set_cell(11, 6, VEXField.RING)
    field.set_cell(0, 6, VEXField.SCORING_ZONE)
    field.set_cell(12, 6, VEXField.SCORING_ZONE)
    field.set_cell(10, 8, VEXField.RING)
    field.set_cell(8, 8, VEXField.RING)
    field.set_cell(4, 8, VEXField.RING)
    field.set_cell(2, 8, VEXField.RING)
    field.set_cell(1, 10, VEXField.RING)
    field.set_cell(2, 10, VEXField.RING)
    field.set_cell(10, 10, VEXField.RING)
    field.set_cell(11, 10, VEXField.RING)
    field.set_cell(6, 10, VEXField.SCORING_ZONE)
    field.set_cell(2, 11, VEXField.RING)
    field.set_cell(10, 11, VEXField.RING)
    field.set_cell(6, 0, VEXField.ROBOT)

    # visualize the initial field
    field.visualize(title="VEX Field - Initial State")
    
    # plan the path
    complete_path = field.plan_optimal_path(collect_rings=True)
    
    if complete_path:
        # visualize the field with the planned path
        field.visualize(path=complete_path, title="VEX Field - Planned Path")
        
        # animation of the path - saving each step as a separate image
        for i in range(1, len(complete_path) + 1):
            partial_path = complete_path[:i]
            field.visualize(path=partial_path, title=f"VEX Field - Step {i}/{len(complete_path)}")
            
    else:
        print("Could not plan a complete path")

if __name__ == "__main__":
    main()