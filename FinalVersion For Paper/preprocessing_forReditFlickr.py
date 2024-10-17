import numpy as np
import math
from scipy.sparse import csr_matrix
from tqdm import tqdm

def read_npz(file_path):
    data = np.load(file_path)
    return data['matrix'].toarray()

def slice_matrix(matrix, fraction=0.25):
    rows, cols = matrix.shape
    new_rows = int(rows * fraction)
    return matrix[:new_rows, :]

def reblock_matrix(matrix, new_block_size):
    rows, cols = matrix.shape
    new_rows = math.ceil(rows / new_block_size)
    new_matrix = np.zeros((new_rows, cols))
    
    for i in range(rows):
        new_row = i // new_block_size
        new_matrix[new_row] += matrix[i]
    
    return new_matrix

def save_npy(matrix, file_path):
    np.save(file_path, matrix)

def process_and_save(file_path, block_sizes):
    original_matrix = read_npz(file_path)
    sliced_matrix = slice_matrix(original_matrix)
    
    for block_size in tqdm(block_sizes):
        new_matrix = reblock_matrix(sliced_matrix, block_size)
        new_file_path = file_path.replace('16x1', f'{block_size}x1').replace('.npz', '.npy')
        save_npy(new_matrix, new_file_path)

# 示例用法
file_path = 'path/to/your/16x1_file.npz'
block_sizes = [64, 128, 256, 512, 1024, 1600, 2048, 2560, 3200, 3840, 4480, 5120, 5760, 6400, 7040, 7680, 8192]  # 你可以根据需要调整这些大小
process_and_save(file_path, block_sizes)