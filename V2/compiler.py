import yaml
import numpy as np
import tqdm

#读取yaml文件
def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

#bit_length为整个算子连接边的个数
#skip_bits为需要跳过的边的编号，是一整个list
def generate_all_binaries(bit_length, skip_bits=None):
    # 初始化二进制数为全0
    binary = '0' * bit_length
    # 计算最大值，即全1的情况
    max_binary = '1' * bit_length
    # 转换为整数进行比较
    max_number = int(max_binary, 2)

    binaries = []
    while True:
        # 检查是否需要跳过当前二进制数
        if skip_bits is not None and any(binary[bit_length - skip_bit] == '1' for skip_bit in skip_bits):
            pass  # 如果指定位中的任何一个为1，则跳过本次递增
        else:
            binaries.append(binary)
        # 将当前二进制数转换为整数，然后递增
        number = int(binary, 2)
        if number >= max_number:
            break
        number += 1
        # 将递增后的整数转换回二进制字符串，确保长度与输入一致
        binary = bin(number)[2:].zfill(bit_length)
        gen_tile_info(binary, data, same_buffer_list, tile_size_list, is_pingpang=False)

    return binaries

#求出二进制对应的融合方式
def trans_binary_to_fused_array(binary):
    fused_array = [[0]]
    for i in range(1,len(binary)+1):
        if(binary[i-1]=='1'):
            fused_array[len(fused_array)-1].append(i)
        else:
            fused_array.append([i])
    return fused_array

#计算最稠密的块有多少元素
def cal_min_sparsity(path):
    data = read(path)
    max_tile = 0
    arr = np.array(data)
    x,y = arr.shape
    for i in range(0,x):
        for j in range(0,y):
            if(data[i][j]>max_tile):
                max_tile = data[i][j]
    return max_tile


#计算每个融合块的访存量，分块大小
def cal_size(data, fused_block, same_buffer_list, tile_size_list, is_pingpang):
    rw = 0
    total_size = 0
    weight_size   = 0   #W
    input_size    = 0   #X
    output_size   = 0   #Y
    internal_size = 0   #Z
    
    for i in range(0,len(fused_block)):
        #**************计算Weight****************
        if data[i]["INPUT"]["input_nong_num"]!=0:
            for w in data[i]["INPUT"]["input_size"]:
                weight_size += w
        #***************************************

        #*****计算第input, output, internal******
        #INPUT
        if(data[i]["INPUT"]["input_list"]==[]):   #第一个元素
            input_size += data[i]["INPUT"]["feature_number"][0]*data[i]["INPUT"]["size_per_feature"][0]
        else:
            for op in data[i]["INPUT"]["input_list"]:
                if op not in fused_block:
                    input_size += data[i]["INPUT"]["feature_number"][op]*data[i]["INPUT"]["size_per_feature"][op]
        #OUTPUT
        if(data[i]["OUTPUT"]["output_list"]==[]):
            output_size += data[i]["OUTPUT"]["output_number"]*data[i]["OUTPUT"]["size_per_feature"]
        else:
            for op in data[i]["OUTPUT"]["output_list"]:
                if op not in fused_block:
                    output_size += data[i]["OUTPUT"]["output_number"]*data[i]["OUTPUT"]["size_per_feature"]
                #中间结果只有一次              
                else:
                    internal_size += data[i]["OUTPUT"]["output_number"]*data[i]["OUTPUT"]["size_per_feature"]
                    break #多个输入，只存一次
        #***************************************

    rw = weight_size + input_size + output_size


    return rw, total_size

#求出tile_size，tile_num，rw和对应的融合方式
#INPUT:
#binary:            当前的融合方式,从左到右分编号别从0到n-1
#data:              所有算子信息的yaml文件
#same_buffer_list:  共用buffer的列表的输入输出信息
#tile_size_list:    可能的分块大小
#is_pingpang:       是否进行乒乓buffer
#OUTPUT:
#fused_array:       融合方式
#tile_size:         分块大小的list，list里每个大小对应融合方式的每个块
#rw:                总共的访存量
def gen_tile_info(binary, data, same_buffer_list, tile_size_list, is_pingpang=False):
    fused_array = trans_binary_to_fused_array(binary)
    rw = 0
    tile_size = []
    for fused_block in fused_array:
        per_rw, per_size = cal_size(data, fused_block, same_buffer_list, tile_size_list, is_pingpang)
        rw += per_rw
        tile_size.append(per_size)
    
    return fused_array, tile_size, rw


if __name__ == '__main__':
    # res = generate_all_binaries(4, skip_bits=[2,3])
    # print(res[3], trans_binary_to_fused_array(res[2]))
    c = cal_min_sparsity('/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/data/adj_citeseer_3327_1.yaml')
    print(c)