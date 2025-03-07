import yaml
import numpy as np
from tqdm import tqdm
import math
import os

#读取yaml文件
def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

def write(data,file_name):

    path = 'Results/Fused'

    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, file_name)

    with open(filepath, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

#求出二进制对应的融合方式
#例如: 000000010 -> [[0],[1],[2],[3],[4],[5],[6,7],[8]]
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

#二分找到最大且小于buffer_size的分块大小，即读取数据次数最少的分块大小
#tile_size_list必须是从小到大排列
#edge_op_num是指输出是边数据的算子的个数
def binary_search(isPingpang, tile_size_list, buffer_size, max_tile_size_list, weight_size, edge_buffer_size_per_tile, row_node_buffer_size_per_tile, col_node_buffer_size_per_tile, node_num, flexibleBuffer = False):
    
    # tile_size_list = generate_multiples_of_64(tile_size_range[0], tile_size_range[1])
    # tile_size_list.insert(0,16) #第一位插入16，因为833600是按tile size 16计算的
    #tile_size_list = [16, 64, 256, 1024, 1600, 2048, 2400, 3072, 3200, 4000, 4096, 4800, 5120, 5600, 6144, 6400, 7168, 7200, 8192]

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
        node_buffer_size = row_node_buffer_size_per_tile * tile_size_list[mid] + col_node_buffer_size_per_tile * 1
        edge_buffer_size = edge_buffer_size_per_tile * edge_tile_size
        if isPingpang:
            if flexibleBuffer:
                required_size = max(weight_size, node_buffer_size*2, edge_buffer_size*2)
            else:
                required_size = weight_size + ( node_buffer_size + edge_buffer_size ) * 2 
        else:
            if flexibleBuffer:
                required_size = max(weight_size, node_buffer_size, edge_buffer_size)
            else:
                required_size = weight_size + ( node_buffer_size + edge_buffer_size )

        if required_size < buffer_size:
            left = mid
        else:
            right = mid - 1
    
    # 最后一次循环后检查required_size
    mid = left
    edge_tile_size = max_tile_size_list[mid]
    node_buffer_size = row_node_buffer_size_per_tile * tile_size_list[mid] + col_node_buffer_size_per_tile * 1
    edge_buffer_size = edge_buffer_size_per_tile * edge_tile_size
    res_size = 0
    if isPingpang:
        if flexibleBuffer:
            res_size = max(weight_size, node_buffer_size*2, edge_buffer_size*2)
        else:
            res_size = weight_size + ( node_buffer_size + edge_buffer_size ) * 2 
    else:
        if flexibleBuffer:
            res_size = max(weight_size, node_buffer_size, edge_buffer_size)
        else:
            res_size = weight_size + ( node_buffer_size + edge_buffer_size )
    
    if res_size > buffer_size:
        return -1, -1
    else:
        return tile_size_list[left], 1

def judge_inst_pattern(pattern, comp_type, inst_fused_dict):
    key = (tuple(pattern), tuple(comp_type))
    if key in inst_fused_dict:
        if inst_fused_dict[key]['Is_Fused']:
            return key

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
def cal_size(op_data, isSinput, isPingpang, inst_fused_dict, fused_block, tile_size_list, buffer_size, max_tile_size_list, node_num, flexibleBuffer,adjust_feature_size=False):
    
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
        #TODO：非gather这种特殊input，非
        #INPUT
        if op_type == "gather": #gather天生多一个LOAD_N
            input_size += op_output["output_number"]*judge_feature_size(op_output["size_per_feature"],adjust_feature_size)
        if(op_input["input_g_list"]==[]):   #所有算子的第一个，而非当前分块的第一个
            if op_type == "applynode" and op_comp_type =='MM' and isSinput and op['OP_NO'] == 0:
                input_size += 0 #sinput不进行load_N
            elif op_type == "scatter" and op_order == "C":
                special_scatter_input_size += op_input["feature_number"][0]*judge_feature_size(op_input["size_per_feature"][0],adjust_feature_size)
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
            for current_op_idx in range(0,len(op_input["input_g_list"])):
                #load buffer
                if op_type == "scatter" or op_type == "applynode":
                    if op_order == "R":
                        row_node_buffer_size_per_tile += 1 * op_load_size_per_feature[current_op_idx]
                    else:
                        col_node_buffer_size_per_tile += 1 * op_load_size_per_feature[current_op_idx]
                else:
                    edge_buffer_size_per_tile += 1 * op_load_size_per_feature[current_op_idx]

                if op_input["input_g_list"][current_op_idx] not in fused_block:
                    if op_type == "applynode" and op_comp_type =='MM' and isSinput and op['OP_NO'] == 0:
                        input_size += 0 #sinput不进行load_N
                    elif op_type == "scatter" and op_order == "C": #特殊的scatter，需要多次读取邻接矩阵
                        special_scatter_input_size += op_input["feature_number"][current_op_idx]*judge_feature_size(op_input["size_per_feature"][current_op_idx],adjust_feature_size)
                    else:
                        input_size += op_input["feature_number"][current_op_idx]*judge_feature_size(op_input["size_per_feature"][current_op_idx],adjust_feature_size)
                #***************指令融合*****************
                else:
                    current_input_op = op_data[op_input["input_g_list"][current_op_idx]]
                    #key = judge_inst_pattern([current_input_op["TYPE"],op_type],[current_input_op["COMP_TYPE"],op_comp_type],inst_fused_dict)
                    key = (tuple([current_input_op["TYPE"],op_type]), tuple([current_input_op["COMP_TYPE"],op_comp_type]))
                    if key in inst_fused_dict:
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
            for current_op_idx in range(0,len(op_output["output_list"])):
                if op_output["output_list"][current_op_idx] not in fused_block:
                    output_size += op_output["output_number"]*judge_feature_size(op_output["size_per_feature"],adjust_feature_size)
                    break
        #**************************************

    #计算分块大小
    tile_row_size, tile_col_size = binary_search(isPingpang, tile_size_list, buffer_size, max_tile_size_list, weight_size, edge_buffer_size_per_tile, row_node_buffer_size_per_tile, col_node_buffer_size_per_tile, node_num, flexibleBuffer)
        
    #计算访存量
    #TODO：如果size_per_feature小于读取带宽的长度有padding的问题
    rw = weight_size + input_size + output_size + special_scatter_input_size * math.ceil(node_num / tile_row_size)

    # if fused_block == [0,1,2,3]:
    #     print('0:\n',rw, weight_size, input_size, output_size, special_scatter_input_size, math.ceil(node_num / tile_row_size))
    # if fused_block == [1]:
    #     print('1:\n',rw, weight_size, input_size, output_size, special_scatter_input_size * math.ceil(node_num / tile_row_size))
    # if fused_block == [2]:
    #     print('2:\n',rw, weight_size, input_size, output_size, special_scatter_input_size * math.ceil(node_num / tile_row_size))
    # if fused_block == [3]:
    #     print('3:\n',rw, weight_size, input_size, output_size, special_scatter_input_size * math.ceil(node_num / tile_row_size))

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
def gen_tile_info(isSinput, isPingpang, op_connected_info, op_num, binary, data, inst_fused_dict, buffer_size, tile_size_list, max_tile_size_list, node_num, flexibleBuffer, adjust_feature_size=False):

    fused_array = trans_binary_to_fused_array(op_connected_info, binary,op_num)
    rw = 0
    tile_size_res = []

    #MODIFILED:
    # if fused_array == [[0,1,2,3]]:

    for fused_block in fused_array:
        per_rw, per_row_size, per_col_size = cal_size(data, isSinput, isPingpang, inst_fused_dict, fused_block, tile_size_list, buffer_size, max_tile_size_list, node_num, flexibleBuffer, adjust_feature_size)
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

#检查重复性的代码
def check_numbers_in_list(breakpoint, lst):
    # 检查 num1 和 num2 是否都在列表 lst 中
    for i in lst:
        for j in breakpoint:
            num1 = j[0]
            num2 = j[1]
            if num1 in i and num2 in i:
                return False
    return True

def is_subgraph_output_returning(graph, subgraph_nodes):
    """
    判断子图的输出是否最终会回到子图
    :param graph: 图的邻接表表示，字典形式 {节点: [相邻节点列表]}
    :param subgraph_nodes: 子图的节点列表
    :return: 如果子图的输出最终会回到子图返回 True，否则返回 False
    """
    subgraph_set = set(subgraph_nodes)

    def dfs(node, visited):
        if node in visited:
            return False
        if node in subgraph_set:
            return True

        visited.add(node)
        for neighbor in graph.get(node, []):
            if dfs(neighbor, visited):
                return True
        visited.remove(node)
        return False

    for node in subgraph_nodes:
        visited = set()
        for neighbor in graph.get(node, []):
            if neighbor not in subgraph_set:
                if dfs(neighbor, visited):
                    return True

    return False

#检查是否成环，成环范围True，不成环则返回False
def check_cycle(data,res_fused_array):
    #rule f，即同一个块内不能有断开的边，这点会自动满足
    #原因是binary在转换到op的时候，会自动加这两个，也就默认两个融合了
    #rule g，不能成环
    #graph_list = []

    #生成整体graph字典
    graph = {}
    for i in range(0,len(data)):
        current_op = data[i]['OP_NO']
        #current_op_input = data[i]['INPUT']['input_g_list']
        current_op_output = data[i]['OUTPUT']['output_list']
        graph[current_op] = []
        for j in current_op_output:
            graph[current_op].append(j)

    #检查每一个融合块是否成环
    for one_fused_array in res_fused_array:
        if is_subgraph_output_returning(graph, one_fused_array):
            return True

    return False
    #print(graph_list)


#path是op的yaml文件路径
#max_tile_size_list是数据集的名字
#buffer_size是buffer的大小
#tile_size_list是可能的分块大小
#bit_length为整个算子连接边的个数
#skip_bits为需要跳过的边的编号，是一整个list，但是和binary是相反的编号，且从1开始
#例如，bit_length=3, skip_bits=[1], 则会跳过001, 011, 101, 111
def generate_all_binaries(data, isSinput, isPingpang, breakpoint, op_connected_info, op_num, max_tile_size_list, buffer_size, tile_size_list, bit_length, node_num, flexibleBuffer, skip_bits=None, adjust_feature_size=False):

    inst_fused_dict = load_inst_fused('hardware_info.yaml')

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

        current_fused_array, current_tile_size, current_rw = gen_tile_info(isSinput, isPingpang, op_connected_info, op_num, binary, data, inst_fused_dict, buffer_size, tile_size_list, max_tile_size_list, node_num, flexibleBuffer, adjust_feature_size)
        
        # 存储结果
        if current_rw != -1:
            # print(check_cycle(data,current_fused_array))
            # print(current_fused_array)
            # break
            if breakpoint != []:
                if check_numbers_in_list(breakpoint, current_fused_array) and not check_cycle(data,current_fused_array):
                    results.append((current_fused_array, current_tile_size, current_rw, binary))
            else:
                results.append((current_fused_array, current_tile_size, current_rw, binary))
        # if binary == '000000000000000' or binary == '00111001001':
        #     print(results[-1])
        

        # 将当前二进制数转换为整数，然后递增
        number = int(binary, 2)
        if number >= max_number:
            break
        number += 1
        # 将递增后的整数转换回二进制字符串，确保长度与输入一致
        binary = bin(number)[2:].zfill(bit_length)

    # 根据rw从大到小排序
    #results.sort(key=lambda x: x[2], reverse=True)
    # 根据rw从小到大排序
    results.sort(key=lambda x: x[2], reverse=False)

    return results 

def find_value_index(arr, value):
    try:
        index = arr.index(value)
        return index
    except ValueError:
        return -1

def gen_op_connected_info(data):
    op_connected_info = []
    op_num = len(data)
    break_point = []
    for op in data:
        op_no = op['OP_NO']
        op_output_list = op['OUTPUT']['output_list']
        for output_op in op_output_list:
            op_connected_info.append([op_no, output_op])
            if (data[op_no]['TYPE'] == 'gather' and data[output_op]['TYPE'] == 'scatter') or (data[op_no]['ORDER'] != data[output_op]['ORDER'] and data[output_op]['TYPE'] == 'scatter'):
                break_point.append([op_no, output_op])
    
    skip_bits = []
    if break_point != []:
        for point in break_point:
            skip_bits.append(len(op_connected_info) - op_connected_info.index(point))

    return op_connected_info, op_num, sorted(skip_bits), break_point

#四个规则
#scatter gather 方向
#gather scatter
#f
#成环
def compile(dataset_name,network_name,layer_name,isReorder,isSinput,isPingpang, flexibleBuffer):

#if __name__ == '__main__':

    #W 1MB
    #N 0.5MB
    #E 0.5MB
    buffer_size = 2*1024*1024

    str_reorder = 'original'
    if isReorder:
        str_reorder = 'trans'
        
    path = 'Network/' + network_name + '/' + network_name + '-' + dataset_name + '/' + network_name + '-' + str_reorder + '/' + network_name + '-' + layer_name + '-' + str_reorder + '.yaml'
    op_info = read(path)

    node_num = 0
    sp_list = read('dataset/' + dataset_name + '/maxlist_' + dataset_name + '.yaml')
    tile_size_list = read('dataset/'+dataset_name+'/sizelist_'+dataset_name+'.yaml')
    if dataset_name == 'cora':
        node_num = 2708
    elif dataset_name == 'pubmed':
        node_num = 19717
    elif dataset_name == 'flickr':
        node_num = 89250
    elif dataset_name == 'reddit':
        node_num = 232965

    op_connected_info, op_num, skip_bits, breakpoint = gen_op_connected_info(op_info)

    connected_edge_num = len(op_connected_info)    
    
    res = generate_all_binaries(op_info, isSinput, isPingpang, breakpoint, op_connected_info, op_num, sp_list, buffer_size, tile_size_list, connected_edge_num, node_num, flexibleBuffer, skip_bits)
    
    for i in res:
        print(i)
    write(res, dataset_name + '-' + network_name + '-' + layer_name + '-' + str_reorder + '.txt')

    return res, op_info, buffer_size, node_num, sp_list, tile_size_list

if __name__ == '__main__':
    compile('cora','GCN','layer1',False,False,True, True)