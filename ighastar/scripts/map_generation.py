import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.ndimage import zoom
import cv2
import os
cwd = os.getcwd()

def generate_map(style=None, size=(20, 20), upscale=512, num_points=400):
    if style is None:
        raise ValueError("Style must be specified.")
    grid = np.zeros(size, dtype=int)

    if style == "lead_figure":
         # Define the central region for the "tree line"
        middle_start, middle_end = size[1]*0.4, 0.6 * size[1]

        # create rectangular block of size middle_start, middle_end, full height:
        
        # # Randomly distribute blocks in the middle region
        # num_blocks = int(size[0] * (middle_end - middle_start) * 0.05)  # Adjust density
        # x_positions = np.random.randint(0, size[0], num_blocks)
        # y_positions = np.random.randint(middle_start, middle_end, num_blocks)

        # for x, y in zip(x_positions, y_positions):
        #     grid[x, y] = 1  # Place "trees"
        grid[0:size[0], int(middle_start):int(middle_end)] = 1

        # Ensure some gaps exist by randomly clearing parts
        num_gaps = 2  # Adjust passage difficulty
        gap_positions = [np.random.choice(size[0]//10, 1, replace=False)[0], size[0]//2 + np.random.choice(size[0]//10, 1, replace=False)[0], size[0]]
        gap_size = [5, 3, 0]
        for i in range(len(gap_positions)-1):
            grid[gap_positions[i]:gap_positions[i]+gap_size[i], int(middle_start):int(middle_end)] = 0  # Force a clear path
            grid[gap_positions[i]+gap_size[i]+1:gap_positions[i+1]-1, int(middle_start):int(middle_end)-1] = 0

    elif style == "single_bottleneck":
        # Define the central region for the "tree line"
        middle_start, middle_end = size[1]*0.2, 0.3 * size[1]

        grid[0:size[0]//2, int(middle_start):int(middle_end)] = 1

        grid[size[0]//2-2:size[0]//2, 0:int(middle_end)] = 1

        # Ensure some gaps exist by randomly clearing parts
        gap_positions = np.random.choice(size[0]//2 - 2, 1, replace=False)
        gap_size = 1 
        for x in gap_positions:
            grid[x:x+gap_size, int(middle_start):int(middle_end)] = 0  # Force a clear path

        middle_start, middle_end = size[1]*0.5, 0.7 * size[1]

        grid[0:size[0], int(middle_start):int(middle_end)] = 1

        # Ensure some gaps exist by randomly clearing parts
        num_gaps = size[0] // 5  # Adjust passage difficulty
        gap_positions = np.random.choice(size[0], num_gaps, replace=False)
        gap_size = 4 
        for x in gap_positions:
            grid[x:x+gap_size, int(middle_start):int(middle_end)] = 0  # Force a clear path


    elif style == "multi_bottleneck":
        # Define the central region for the "tree line"
        middle_start, middle_end = size[1]*0.2, 0.3 * size[1]


        # create rectangular block of size middle_start, middle_end, full height:
        grid[0:size[0]//2, int(middle_start):int(middle_end)] = 1

        grid[size[0]//2-2:size[0]//2, 0:int(middle_end)] = 1

        # Ensure some gaps exist by randomly clearing parts
        gap_positions = np.random.choice(size[0]//2 - 2, 1, replace=False)
        gap_size = 1 
        for x in gap_positions:
            grid[x:x+gap_size, int(middle_start):int(middle_end)] = 0  # Force a clear path

        grid[size[0]//2:size[0], int(size[1]*0.6):int(size[1]*0.6) + 4] = 1

        grid[size[0]//2:size[0]//2+4, int(size[1]*0.6):size[1]] = 1

        # Ensure some gaps exist by randomly clearing parts
        gap_positions = 4 + size[0]//2 + np.random.choice(size[0]//2 - 2, 1, replace=False)
        gap_size = 1 
        for x in gap_positions:
            if x >= size[0]:
                x = size[0] - gap_size
            if x <= size[0]//2 - 4:
                x = size[0]//2 - 4 + gap_size
            grid[x:x+gap_size, int(size[1]*0.6):int(size[1]*0.6) + 4] = 0  # Force a clear path
        
        grid[0:size[0], int(size[1]*0.45):int(size[1]*0.45)+2] = 1

        # Ensure some gaps exist by randomly clearing parts
        gap_positions = int(size[0]*0.3)  + np.random.choice(int(size[0]*0.2), 1, replace=False)
        gap_size = 1 
        for x in gap_positions:
            grid[x:x+gap_size, int(size[1]*0.45):int(size[1]*0.45)+2] = 0  # Force a clear path


    # Nearest-neighbor upscale to ~512x512
    scale_factor = upscale / size[0]
    grid = zoom(grid, (scale_factor, scale_factor), order=0)
    # blur the image:
    grid = np.array(grid, dtype=np.uint8)
    grid = 255*(1 - grid)

    grid = cv2.GaussianBlur(grid, (11, 11), 0)
    grid[grid < 254] = 0
    # threshold the image:
    return grid


final_size = 512
num_points = 400


style = "single_bottleneck"
# style = "multi_bottleneck"
N = 10
for i in range(N):
    path = os.path.join(cwd, f"examples/Maps/generated_maps/{style}")
    if not os.path.exists(path):
        os.makedirs(path)
    map = generate_map(style, size=(40, 40), upscale=final_size, num_points=400)
    # cv2.imwrite(os.path.join(path,f"{i}_{final_size}.png"), map)