import numpy as np
import yaml
from tqdm import tqdm
import argparse

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

def calculate_sparsity(row, col, npy_file_path):
    # 加载npy文件
    matrix = np.load(npy_file_path)

    # 去掉self-loop
    np.fill_diagonal(matrix, 0)

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
    sparsity = [[np.count_nonzero(block) for block in row_blocks] for row_blocks in blocks]

    return sparsity

def save(matrix, yaml_file_path):
    # 将二维列表拆成一维列表，竖着拆开
    # flattened_list = [item for sublist in zip(*sparsity) for item in sublist]

    # 将二维列表拆成一维列表，横着拆开
    #flattened_list = [item for sublist in sparsity for item in sublist]

    # 将结果存储到yaml文件中
    with open(yaml_file_path, 'w') as file:
        yaml.dump(matrix, file)

def cal_min_sparsity(data_set, tile_size):
    path = "dataset/"+data_set+"/adj_"+data_set+"_"+str(tile_size)+"_1.yaml"
    data = read(path)
    max_tile = 0
    arr = np.array(data)
    x,y = arr.shape
    for i in range(0,x):
        for j in range(0,y):
            if(data[i][j] > max_tile):
                max_tile = data[i][j]
    return max_tile

def gen_size(start,end):
    size = [start]
    i = 1
    while(size[-1] < end):
        i += 1
        size.append(start*i)
        
    return size

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compiler script")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset')
    parser.add_argument('--start', type=str, required=True, help='The start size of tile size, e.g. start size = 16, tile size = 16, 32, 48, 64, ...')
    parser.add_argument('--end', type=str, required=True, help='The end size of tile size')
    #parser.add_argument('--step', type=str, required=True, help='The step of tile size, e.g. step=2, tile size = 16, 32, 64, 128, ...')
    args = parser.parse_args()

    row_list = gen_size(int(args.start),int(args.end))
    print(row_list)
    path = 'dataset/'+args.dataset+'/adj_'+args.dataset+'.npy'
    col = 1
    sp_list = []
    #生成yaml文件
    for row in tqdm(row_list):
        sparsity = calculate_sparsity(row, col, path)
        save(sparsity, 'dataset/'+args.dataset+'/adj_'+args.dataset+'_'+str(row)+'_'+str(col)+'.yaml')
    #打印非零元最多的稀疏块的稀疏个数
    for tile_size in tqdm(row_list):
        sp_list.append(cal_min_sparsity(args.dataset,tile_size))
        save(row_list, 'dataset/'+args.dataset+'/sizelist_'+args.dataset+'.yaml')
        save(sp_list, 'dataset/'+args.dataset+'/maxlist_'+args.dataset+'.yaml')