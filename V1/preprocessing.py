import numpy as np
import yaml

def calculate_sparsity(row, col, npy_file_path):
    # 加载npy文件
    matrix = np.load(npy_file_path)

    # 获取矩阵的形状
    rows, cols = matrix.shape

    # 计算需要扩展的行数和列数
    pad_rows = row - rows % row if rows % row != 0 else 0
    pad_cols = col - cols % col if cols % col != 0 else 0

    # 扩展矩阵
    matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)))

    # 计算需要分割的块数
    num_blocks_row = matrix.shape[0] // row
    num_blocks_col = matrix.shape[1] // col

    # 分割矩阵
    blocks = np.split(matrix, num_blocks_row)
    blocks = [np.split(block, num_blocks_col, axis=1) for block in blocks]

    # 计算每个块的非0元个数
    sparsity = [[np.count_nonzero(block == 0) for block in row_blocks] for row_blocks in blocks]

    return sparsity

def flatten_and_save(sparsity, yaml_file_path):
    # 将二维列表拆成一维列表，竖着拆开
    # flattened_list = [item for sublist in zip(*sparsity) for item in sublist]

    # 将二维列表拆成一维列表，横着拆开
    #flattened_list = [item for sublist in sparsity for item in sublist]

    # 将结果存储到yaml文件中
    with open(yaml_file_path, 'w') as file:
        yaml.dump(sparsity, file)

if __name__ == '__main__':
    dataset = 'citeseer'
    path = '/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/data/adj_'+dataset+'.npy'
    row = 579
    col = 1
    sparsity = calculate_sparsity(row, col, path)
    flatten_and_save(sparsity, '/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/data/adj_'+dataset+'_'+str(row)+'_'+str(col)+'.yaml')
    print(sparsity)