import numpy as np
import yaml
from tqdm import tqdm

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

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

def cal_min_sparsity(data_set, tile_size):
    path = "/Users/sijin/Desktop/workspace/GTA_Code/dataset/adj_"+data_set+"_"+str(tile_size)+"_1.yaml"
    data = read(path)
    max_tile = 0
    arr = np.array(data)
    x,y = arr.shape
    for i in range(0,x):
        for j in range(0,y):
            if(data[i][j]>max_tile):
                max_tile = data[i][j]
    return max_tile

def generate_multiples_of_16(min_value, max_value):
    # 确保输入的两个数都是16的倍数
    if min_value % 16 != 0 or max_value % 16 != 0:
        raise ValueError("输入的两个数必须是16的倍数")
    
    # 生成以min_value和max_value为极小值和极大值的16的倍数列表
    multiples_of_16 = list(range(min_value, max_value + 1, 16))
    
    return multiples_of_16

def generate_multiples_of_32(min_value, max_value):
    # 确保输入的两个数都是32的倍数
    if min_value % 32 != 0 or max_value % 32 != 0:
        raise ValueError("输入的两个数必须是32的倍数")
    
    # 生成以min_value和max_value为极小值和极大值的32的倍数列表
    multiples_of_64 = list(range(min_value, max_value + 1, 32))
    
    return multiples_of_64

def generate_multiples_of_64(min_value, max_value):
    # 确保输入的两个数都是64的倍数
    if min_value % 64 != 0 or max_value % 64 != 0:
        raise ValueError("输入的两个数必须是64的倍数")
    
    # 生成以min_value和max_value为极小值和极大值的64的倍数列表
    multiples_of_64 = list(range(min_value, max_value + 1, 64))
    
    return multiples_of_64

if __name__ == '__main__':
    dataset = 'cora'
    #row_list = generate_multiples_of_16(16, 2048)
    #row_list = generate_multiples_of_64(2048, 8192)
    row_list = generate_multiples_of_64(64, 8192)
    path = '/Users/sijin/Desktop/workspace/GTA_Code/dataset/adj_'+dataset+'.npy'
    col = 1
    sp_list = []
    # 生成yaml文件
    # for row in row_list:
    #     sparsity = calculate_sparsity(row, col, path)
    #     flatten_and_save(sparsity, '/Users/sijin/Desktop/workspace/GTA_Code/dataset/adj_'+dataset+'_'+str(row)+'_'+str(col)+'.yaml')
    # 打印
    for tile_size in tqdm(row_list):
        sp_list.append(cal_min_sparsity(dataset,tile_size))
    print(sp_list)