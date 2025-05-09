import yaml
import numpy as np
from tqdm import tqdm
import math

#buffer_size = 2*1024*1024 #byte
#buffer_size = 4*1024*1024 #byte
#GAT
#breakpoint = [[0,3],[2,5],[8,10]]
#GAT trans
#breakpoint = [[0,3],[2,5]]
#GCN
breakpoint = [[8,9],[4,5],[0,1]]

#breakpoint = [[0,3]]

#breakpoint = [[0,1],[7,8],[14,15]]
#breakpoint = [[0,1],[7,8],[14,15]]
#breakpoint = [[0,3]]
isSinput = True
is_pingpang = True

#读取yaml文件
def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

#求出二进制对应的融合方式
#例如: 000000010 -> [[0],[1],[2],[3],[4],[5],[6,7],[8]]
#TODO: op_list目前还是手动需要定义
def trans_binary_to_fused_array(op_connected_info,binary,op_num):
    
    visited = set()

    def dfs(node):
        stack = [node]
        component = []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        return sorted(component)  # 对算子进行排序

    # 构建图
    graph = {i: [] for i in range(op_num)}
    for i, (u, v) in enumerate(op_connected_info):
        if binary[i] == '1':
            graph[u].append(v)
            graph[v].append(u)

    connected_components = []
    for node in range(op_num):
        if node not in visited:
            component = dfs(node)
            connected_components.append(component)

    # 对连通分量进行排序，以确保唯一性
    connected_components = sorted(connected_components, key=lambda x: (len(x), x))

    return connected_components

def generate_multiples_of_64(min_value, max_value):
    # 确保输入的两个数都是64的倍数
    if min_value % 64 != 0 or max_value % 64 != 0:
        raise ValueError("输入的两个数必须是64的倍数")
    
    # 生成以min_value和max_value为极小值和极大值的64的倍数列表
    multiples_of_64 = list(range(min_value, max_value + 1, 64))
    
    return multiples_of_64

def generate_multiples_of_16(min_value, max_value):
    # 确保输入的两个数都是16的倍数
    if min_value % 16 != 0 or max_value % 16 != 0:
        raise ValueError("输入的两个数必须是16的倍数")
    
    # 生成以min_value和max_value为极小值和极大值的16的倍数列表
    multiples_of_16 = list(range(min_value, max_value + 1, 16))
    
    return multiples_of_16

#二分找到最大且小于buffer_size的分块大小，即读取数据次数最少的分块大小
#tile_size_list必须是从小到大排列
#edge_op_num是指输出是边数据的算子的个数
def binary_search(tile_size_range, buffer_size, max_tile_size_list, weight_size, edge_buffer_size_per_tile, row_node_buffer_size_per_tile, col_node_buffer_size_per_tile, node_num):
    
    # tile_size_list = generate_multiples_of_64(tile_size_range[0], tile_size_range[1])
    # tile_size_list.insert(0,16) #第一位插入16，因为833600是按tile size 16计算的
    tile_size_list = [16, 64, 256, 1024, 1600, 2048, 2400, 3072, 3200, 4000, 4096, 4800, 5120, 5600, 6144, 6400, 7168, 7200, 8192]

    left  = 0 
    right = len(tile_size_list) - 1

    for i, size in enumerate(tile_size_list):
        if size > node_num:
            right = i
            break

    while left < right:
        mid = (left + right + 1) // 2
        
        required_size = 0

        edge_tile_size = max_tile_size_list[mid]

        #如果是乒乓buffer，需要多一倍的buffer
        if is_pingpang:
            required_size = weight_size + ( row_node_buffer_size_per_tile * tile_size_list[mid] + col_node_buffer_size_per_tile * 1 + edge_buffer_size_per_tile * edge_tile_size) * 2 
        else:
            required_size = weight_size + ( row_node_buffer_size_per_tile * tile_size_list[mid] + col_node_buffer_size_per_tile * 1 + edge_buffer_size_per_tile * edge_tile_size)

        if required_size < buffer_size:
            left = mid
        else:
            right = mid - 1
    
    # 最后一次循环后检查required_size
    mid = left
    edge_tile_size = max_tile_size_list[mid]
    if is_pingpang:
        res_size = weight_size + ( row_node_buffer_size_per_tile * tile_size_list[mid] + col_node_buffer_size_per_tile * 1 + edge_buffer_size_per_tile * edge_tile_size) * 2 
    else:
        res_size = weight_size + ( row_node_buffer_size_per_tile * tile_size_list[mid] + col_node_buffer_size_per_tile * 1 + edge_buffer_size_per_tile * edge_tile_size)
    
    if res_size > buffer_size:
        return -1, -1
    else:
        return tile_size_list[left], 1

def judge_inst_pattern(pattern, comp_type, inst_fused_dict):
    key = (tuple(pattern), tuple(comp_type))
    if key in inst_fused_dict:
        if inst_fused_dict[key]['Is_Fused']:
            return key
    return -1

def judge_feature_size(size_per_feature, isValid=False):
    # 计算32乘4的倍数
    multiple = 32 * 4
    
    # 检查是否是32乘4的倍数
    if size_per_feature % multiple == 0 or isValid == False:
        return size_per_feature
    # 计算大于输入数字的最小32乘4的倍数
    upper_multiple = ((size_per_feature // multiple) + 1) * multiple
    
    return upper_multiple

#计算每个融合块的访存量，分块大小
#Test: Fused Buffer Not Tested
def cal_size(op_data, inst_fused_dict, fused_block, tile_size_list, buffer_size, max_tile_size_list, node_num, adjust_feature_size=False):
    
    rw = 0
    weight_size   = 0   #W
    input_size    = 0   #X
    output_size   = 0   #Y

    edge_buffer_size_per_tile   = 0
    row_node_buffer_size_per_tile   = 0
    col_node_buffer_size_per_tile   = 0

    special_scatter_input_size = 0  #特殊的scatter，需要多次读取邻接矩阵

    for i in fused_block:

        op = op_data[i]
        op_type =op["TYPE"]
        op_order = op["ORDER"]
        op_input = op["INPUT"]
        op_output = op["OUTPUT"]
        op_comp_type = op["COMP_TYPE"]

        op_output_size_per_feature = op["OUTPUT"]["size_per_feature"]
        op_load_size_per_feature = op["INPUT"]["size_per_feature"]

        #**************计算Weight****************
        if op_input["input_nong_num"]!=0:
            for w in op_input["input_size"]:
                weight_size += w
        #***************************************

        #******计算input, output, internal******
        #INPUT
        if op_type == "gather": #gather天生多一个LOAD_N
            input_size += op_output["output_number"]*judge_feature_size(op_output["size_per_feature"],adjust_feature_size)
        if(op_input["input_g_list"]==[]):   #所有算子的第一个，而非当前分块的第一个
            if op_type == "applynode" and op_comp_type =='MM' and isSinput and op['OP_NO'] == 0:
                input_size += 0 #sinput不进行load_N
            else: 
                input_size += op_input["feature_number"][0]*judge_feature_size(op_input["size_per_feature"][0],adjust_feature_size)
            #load buffer
            if op_type == "scatter" or op_type == "applynode":
                if op_order == "R":
                    row_node_buffer_size_per_tile += 1 * op_load_size_per_feature[0]
                else:
                    col_node_buffer_size_per_tile += 1 * op_load_size_per_feature[0]
            else:
                edge_buffer_size_per_tile += 1 * op_load_size_per_feature[0]
        else:
            for current_op in range(0,len(op_input["input_g_list"])):
                #load buffer
                if op_type == "scatter" or op_type == "applynode":
                    if op_order == "R":
                        row_node_buffer_size_per_tile += 1 * op_load_size_per_feature[current_op]
                    else:
                        col_node_buffer_size_per_tile += 1 * op_load_size_per_feature[current_op]
                else:
                    edge_buffer_size_per_tile += 1 * op_load_size_per_feature[current_op]

                if op_input["input_g_list"][current_op] not in fused_block:
                    if op_type == "scatter" and op_order == "C": #特殊的scatter，需要多次读取邻接矩阵
                        special_scatter_input_size += op_input["feature_number"][current_op]*judge_feature_size(op_input["size_per_feature"][current_op],adjust_feature_size)
                    else:
                        input_size += op_input["feature_number"][current_op]*judge_feature_size(op_input["size_per_feature"][current_op],adjust_feature_size)
                #***************指令融合*****************
                else:
                    current_input_op = op_data[op_input["input_g_list"][current_op]]
                    key = judge_inst_pattern([current_input_op["TYPE"],op_type],[current_input_op["COMP_TYPE"],op_comp_type],inst_fused_dict)
                    if key != -1:
                        if inst_fused_dict[key]['Is_Fused']:
                            if inst_fused_dict[key]['Buffer_Type'] == 'Edge':
                                edge_buffer_size_per_tile -= 1 * op_output_size_per_feature
                            else:
                                if op_order == "R":
                                    row_node_buffer_size_per_tile -= 1 * op_output_size_per_feature
                                else:
                                    col_node_buffer_size_per_tile -= 1 * op_output_size_per_feature
                #**************************************

        #OUTPUT
        if(op_output["output_list"]==[]):
            output_size += op_output["output_number"]*judge_feature_size(op_output["size_per_feature"],adjust_feature_size)
            #output buffer
            if op_type == "scatter" or op_type == "applyedge":
                edge_buffer_size_per_tile  += 1 * op_output_size_per_feature
            else:
                if op_order == "R":
                    row_node_buffer_size_per_tile += 1 * op_output_size_per_feature
                else:
                    col_node_buffer_size_per_tile += 1 * op_output_size_per_feature
        else:
            #output buffer
            if op_type == "scatter" or op_type == "applyedge":
                edge_buffer_size_per_tile  += 1 * op_output_size_per_feature
            else:
                if op_order == "R":
                    row_node_buffer_size_per_tile += 1 * op_output_size_per_feature
                else:
                    col_node_buffer_size_per_tile += 1 * op_output_size_per_feature
            for current_op in range(0,len(op_output["output_list"])):
                if op_output["output_list"][current_op] not in fused_block:
                    output_size += op_output["output_number"]*judge_feature_size(op_output["size_per_feature"],adjust_feature_size)
                    break
        #**************************************

    #计算分块大小
    tile_row_size, tile_col_size = binary_search(tile_size_list, buffer_size, max_tile_size_list, weight_size, edge_buffer_size_per_tile, row_node_buffer_size_per_tile, col_node_buffer_size_per_tile, node_num)
        
    #计算访存量
    #TODO：如果size_per_feature小于读取带宽的长度有padding的问题
    rw = weight_size + input_size + output_size + special_scatter_input_size * math.ceil(node_num / tile_row_size)

    return rw, tile_row_size, tile_col_size



#求出tile_size，tile_num，rw和对应的融合方式
#INPUT:
#binary:            当前的融合方式,从左到右分编号别从0到n-1
#data:              所有算子信息的yaml文件
#same_buffer_list:  共用buffer的列表的输入输出信息
#tile_size_list:    可能的分块大小 eg.[[16,1],[64,1],[128,1]]
#is_pingpang:       是否进行乒乓buffer
#OUTPUT:
#fused_array:       融合方式
#tile_size:         分块大小的list，list里每个大小对应融合方式的每个块
#rw:                总共的访存量
def gen_tile_info(op_connected_info, op_num, binary, data, inst_fused_dict, buffer_size, tile_size_list, max_tile_size_list, node_num, adjust_feature_size=False):

    fused_array = trans_binary_to_fused_array(op_connected_info, binary,op_num)
    rw = 0
    tile_size_res = []

    for fused_block in fused_array:
        per_rw, per_row_size, per_col_size = cal_size(data, inst_fused_dict, fused_block, tile_size_list, buffer_size, max_tile_size_list, node_num, adjust_feature_size)
        if per_row_size == -1:
            #print("No Solution")
            return [], [], -1
        rw += per_rw
        tile_size_res.append([per_row_size, per_col_size])

    return fused_array, tile_size_res, rw

def load_inst_fused(file_path):
    data = read(file_path)
    
    inst_fused_dict = {}
    for item in data:
        if 'Inst_fused' in item:
            for entry in item['Inst_fused']:
                key = (tuple(entry['Pattern']), tuple(entry['Compute_Type']))
                inst_fused_dict[key] = {
                    'Is_Fused': entry['Is_Fused'],
                    'Buffer_Type': entry.get('Buffer_Type', [])
                }

    return inst_fused_dict

def check_numbers_in_list(breakpoint, lst):
    # 检查 num1 和 num2 是否都在列表 lst 中
    for i in lst:
        for j in breakpoint:
            num1 = j[0]
            num2 = j[1]
            if num1 in i and num2 in i:
                return False
    return True

#path是op的yaml文件路径
#max_tile_size_list是数据集的名字
#buffer_size是buffer的大小
#tile_size_list是可能的分块大小
#bit_length为整个算子连接边的个数
#skip_bits为需要跳过的边的编号，是一整个list，但是和binary是相反的编号，且从1开始
#例如，bit_length=3, skip_bits=[1], 则会跳过001, 011, 101, 111
def generate_all_binaries(path, op_connected_info, op_num, max_tile_size_list, buffer_size, tile_size_list, bit_length, node_num, skip_bits=None, adjust_feature_size=False):

    data = read(path)
    inst_fused_dict = load_inst_fused('Code/hardware_info.yaml')

    # 初始化二进制数为全0
    binary = '0' * bit_length
    # 计算最大值，即全1的情况
    max_binary = '1' * bit_length
    # 转换为整数进行比较
    max_number = int(max_binary, 2)

    binaries = []

    results = []

    while True:

        # 检查是否需要跳过当前二进制数
        if skip_bits is not None and any(binary[bit_length - skip_bit] == '1' for skip_bit in skip_bits):
            #print('Skip:', binary)
            # 将当前二进制数转换为整数，然后递增
            number = int(binary, 2)
            if number >= max_number:
                break
            number += 1
            # 将递增后的整数转换回二进制字符串，确保长度与输入一致
            binary = bin(number)[2:].zfill(bit_length)
            continue
        
        binaries.append(binary)

        
        current_fused_array, current_tile_size, current_rw = gen_tile_info(op_connected_info, op_num, binary, data, inst_fused_dict, buffer_size, tile_size_list, max_tile_size_list, node_num, adjust_feature_size)
        
        # 存储结果
        if current_rw != -1:
            if breakpoint != []:
                if check_numbers_in_list(breakpoint, current_fused_array):
                    results.append((current_fused_array, current_tile_size, current_rw, binary))
            else:
                results.append((current_fused_array, current_tile_size, current_rw, binary))
        if binary == '000000000000000' or binary == '00111001001':
            print(results[-1])
        
        # 将当前二进制数转换为整数，然后递增
        number = int(binary, 2)
        if number >= max_number:
            break
        number += 1
        # 将递增后的整数转换回二进制字符串，确保长度与输入一致
        binary = bin(number)[2:].zfill(bit_length)

    # 根据rw从小到大排序
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results 

if __name__ == '__main__':
    
    path = 'Network/GCN/GCN-reddit/GCN-trans/GCN-alllayer-trans.yaml'

    #buffer_size = 833600
    buffer_size = 2*1024*1024
    #Cora
    #node_num = 2708
    #Pubmed
    # node_num = 19717
    #flickr 89250/4
    #reddit
    node_num = 232965

    #SimpleTest
    #op_list = [[0,1],[0,2],[1,3],[2,4],[3,5],[4,5],[5,6],[6,7],[7,8]]

    #GCN_Cora
    # op_connected_info = [[0,1],[1,2],[2,3]]
    # skip_bits= []
    # op_num = 4

    # #GCN_Cora Trans all layers
    op_connected_info = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
    skip_bits= [3,7,11]
    op_num = 12
    
    #GraphSAGE Trans
    # op_connected_info = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6]]
    # skip_bits = [6]
    # op_num = 7

    #GraphSAGE Trans all layers
    # op_connected_info = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 12], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 19], [18, 19], [19, 20]]
    # skip_bits = [6,13,20]
    # op_num = 21

    #PAN Original/trans
    # op_connected_info = [[0, 3], [1, 4], [3, 5], [4, 5], [2, 6], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
    # #[0], [1], [3,4,5], [2,6], [7,8], [9], [10,11] ]
    # #'00111001001'
    # skip_bits = [11]
    # #skip_bits = [11]
    # op_num = 12

    #GAT_Cora
    # op_connected_info = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,11],[4,6],[5,6],[6,7],[7,8],[7,9],[8,10],[9,11],[10,9],[11,12],[12,13]]
    # #手工融合： ’0000011101000111‘
    # skip_bits=[5,12,14]
    # op_num = 14

    #GAT_Cora_Trans
    # op_connected_info = [[0, 1],[0, 2],[0, 3],[1, 4],[2, 5],[3, 8],[4, 6],[5, 6],[6, 7],[7, 8],[7, 9],[8, 10],[9, 11],[10, 11],[11, 12]]
    # #手工融合： ’000001111011001‘
    # skip_bits=[11,13]
    # op_num = 13

    
    connected_edge_num = len(op_connected_info)

    #cora
    #sp_list = [39, 59, 39, 68, 62, 68, 77, 68, 52, 83, 75, 68, 59, 109, 104, 100, 95, 91, 86, 84, 90, 92, 94, 99, 105, 108, 128, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168]
    #pubmed
    # sp_list = [32, 7, 7, 7, 8, 8, 8, 10, 10, 10, 11, 13, 16, 14, 15, 14, 15, 18, 16, 18, 19, 19, 22, 23, 22, 21, 21, 24, 25, 23, 25, 24, 26, 30, 29, 29, 29, 31, 32, 32, 32, 33, 33, 33, 35, 35, 35, 34, 34, 36, 38, 38, 39, 40, 40, 42, 41, 41, 44, 44, 44, 44, 45, 46, 46, 46, 46, 46, 47, 49, 49, 50, 52, 52, 52, 52, 52, 52, 52, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 58, 58, 60, 60, 60, 60, 60, 60, 63, 63, 65, 65, 65, 66, 66, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 72, 73, 74, 74, 74, 75]
    #flickr
    #sp_list = [5, 6, 17, 22, 26, 28, 31, 29, 36, 41, 42, 43, 41, 49, 51, 51, 54, 65, 69]
    #Reddit
    sp_list = [7, 8, 8, 11, 13, 18, 19, 23, 27, 32, 35, 45, 42, 52, 58, 64, 70, 76, 81, 84, 89, 91]
    res = generate_all_binaries(path, op_connected_info, op_num, sp_list, buffer_size, [64,8192], connected_edge_num, node_num, skip_bits)
    
    for i in res:
        print(i)

    