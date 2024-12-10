from simulator import *
from interpreter import *
from compiler import *
import random
import copy

#TODO:用哈希模拟仿真

def minimal_mem_access(res,isPingpang):
    #([[0, 1, 2], [4, 5, 6, 7, 8], [3, 9, 10, 11, 12, 13]], [[176, 1], [2720, 1], [1200, 1]], 27155664, '1100011111001111')
    return {'fusion': res[0], 'tsize': res[1], 'pingpong': isPingpang, 'latency': 0, 'mem_access': res[2], 'pattern': res[3]}

def minimal_mem_access_small_tsize(res,isPingpang):
    tsize = copy.deepcopy(res[1])
    for i in tsize:
        i[0] = math.ceil(math.ceil(i[0]/2)/16)*16
    return {'fusion': res[0], 'tsize': tsize, 'pingpong': isPingpang, 'latency': 0, 'mem_access': res[2], 'pattern': res[3]}

def no_fusion_large_tsize(res,isPingpang):
    return {'fusion': res[0], 'tsize': res[1], 'pingpong': isPingpang, 'latency': 0, 'mem_access': res[2], 'pattern': res[3]}

def no_fusion_small_tsize(res,isPingpang):
    tsize = copy.deepcopy(res[1])
    for i in tsize:
        i[0] = math.ceil(math.ceil(i[0]/2)/16)*16
    return {'fusion': res[0], 'tsize': tsize, 'pingpong': isPingpang, 'latency': 0, 'mem_access': res[2], 'pattern': res[3]}

def random_fusion_random_tsize(res,isPingpang):
    random_sample = random.choice(res)
    return {'fusion': random_sample[0], 'tsize': random_sample[1], 'pingpong': isPingpang, 'latency': 0, 'mem_access': random_sample[2], 'pattern': random_sample[3]}


# functions for genetic search algorithm
def initialize(compiler_res,isPingpang):

    # samples in generation should be a struct to describe the design space:
    #sample可以加mem_access和latency，生成新sample需要归零
    #sample = {'fusion': [0,[1,2]], 'tsize':[[128,1],[256,1]], 'pingpong':[1,0], 'latency':0, 'mem_access':0}

    generation = []

    # parsed_res = parse_result(compiler_res)

    # for one_initial_sample in parsed_res:
    #     generation.append(one_initial_sample)

    #第一个进行compiler生成，其他的不改原先代码生成
    #sample:  {'fusion': [[0, 1, 2, 3]], 'tsize': [[16, 1]], 'pingpong': True, 'latency': 24644163, 'mem_access': 2716932960}
    generation.append({'fusion': [[0, 1, 2, 3]], 'tsize': [[16, 1]], 'pingpong': True, 'latency': 24644163, 'mem_access': 2716932960, 'pattern': '111'})
    #generation.append(minimal_mem_access            (compiler_res[0],isPingpang)    )# use original compiler.py ?
    
    #{'fusion': [[0, 1, 2, 3]], 'tsize': [[16, 1]], 'pingpong': True, 'latency': 24644163, 'mem_access': 5433865920}
    #generation.append(minimal_mem_access_small_tsize(compiler_res[0],isPingpang)    ) 
    
    #{'fusion': [[0], [1], [2], [3]], 'tsize': [[2720, 1], [1376, 1], [112, 1], [96, 1]], 'pingpong': True, 'latency': 6442571, 'mem_access': 3083677136}
    generation.append({'fusion': [[0], [1], [2], [3]], 'tsize': [[2720, 1], [1376, 1], [112, 1], [96, 1]], 'pingpong': True, 'latency': 6442571, 'mem_access': 3083677136, 'pattern': '000'})
    #generation.append(no_fusion_large_tsize         (compiler_res[-1],isPingpang)   )
    
    #generation.append(no_fusion_small_tsize         (compiler_res[-1],isPingpang)   )
    # generation.append(fuse_as_much_as_possible()        )# smallest tile size, maybe largest tile size can be appended
    # generation.append(common_fusion_large_tsize()       )
    # generation.append(common_fusion_small_tsize()       )
    #generation.append(random_fusion_random_tsize    (compiler_res,isPingpang)       )
    # ... more
    # combine design dimensions(fusion and tiling size for now) to generate samples
    # largest tiling size and smallest tiling size are used now, maybe more (e.g. medium) can be appended

    return generation

def check_access(best_latency, new_sample):
    if new_sample['mem_access']/BW > best_latency*1.1:
        return True
    return False

#True表示符合规则
def check_fusion_rules(breakpoint,new_sample):
    return check_numbers_in_list(breakpoint,new_sample['fusion'])

def prune(new_sample, best_latency):
    if check_access(best_latency,new_sample):
        return True
    return False

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result
    
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

#Ture出现dup
def check_dup(generation, sample):
    for one_gen in generation:
        if one_gen['tsize'] == sample['tsize'] and one_gen['fusion'] == sample['fusion']:
            return True
    return False


def isOverflow(op_data, inst_fused_dict, isSinput, isPingpang, fused_block, tile_size, buffer_size, adjust_feature_size=False):

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
    required_size = 0
    if isPingpang:
        required_size = weight_size + ( row_node_buffer_size_per_tile * tile_size[0] + col_node_buffer_size_per_tile * 1 + edge_buffer_size_per_tile * tile_size[0]) * 2 
    else:
        required_size = weight_size + ( row_node_buffer_size_per_tile * tile_size[0] + col_node_buffer_size_per_tile * 1 + edge_buffer_size_per_tile * tile_size[1])
    
    if required_size > buffer_size:
        return True
    return False
    

#COMBINE：
#combine检查规则：融合规则，检查规则->赋值tile size，检查tile size
#pruning检查性能
#1.按01二进制融合，compiler检查融合规则和找到最大分块大小，再从原来sample中选择tile size赋值给新的sample
#2.拆开compiler，一步一步check

#融合不行：
#1. 改变01融合的比特数，从1/2，11+9，12+8（先按第一种写）
#2. 不符合融合规则的地方修改，不一定断开，但是让他复合规则（成环必须断开；必须连起来）
#[[0,1],[2],[3]]
#[[0],[1],[2,3]]
#[[0,1],[2,3]]
#分块大小换两个


def combine_strings(str1, str2, retain_length):
    if len(str1) != len(str2):
        raise ValueError("两个字符串的长度必须相等")

    # 拼接前一部分和后一部分
    combined_str = str1[:retain_length] + str2[retain_length:]

    return combined_str

#改递归，外层循环，如果出现多次循环后仍无结果，直接返回，外层重新选两个sample
#重复的sample不要，跳过找新的，和overflow和check融合规则同等地位
def combine(params, sample1, sample2, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, breakpoint, pattern_ratio = 0.5):
    
    increasement = 0

    while True:
        combine_choice = [0,1] #0 combine fusion pattern; 1 combine tile size
        choice = random.choice(combine_choice)
        if choice == 0:
            retain_length = int(len(sample1['pattern']) * pattern_ratio) + increasement
            if retain_length >= len(sample1['pattern']):
                continue
            new_pattern = combine_strings(sample1['pattern'], sample2['pattern'], retain_length)
            if check_cycle(op_data, new_sample['fusion']) or not check_fusion_rules(breakpoint, new_sample['fusion']):
                increasement += 1
                continue
            #TODO:选tile size，从原先的sample中选，而不是最大值
            sizes = sample1['tsize'] + sample2['tsize']
            new_sample = gen_new_sample(new_pattern, params, sizes) #改函数名，计算最大的tile size
        else:
            change_num = int(len(sample1['tsize'])/2)
            if len(sample1['tsize']) >= len(sample2['tsize']):
                new_sample = copy.deepcopy(sample1)
                for sample_1_fusion in range(0,len(sample1['fusion'])):
                    for sample_2_size in sample2['tsize']:
                        if isOverflow(op_data, inst_fused_dict, isSinput, isPingpang, new_sample['fusion'][sample_1_fusion], sample_2_size, buffer_size):
                            continue
                        new_sample['tsize'][sample_1_fusion] = sample_2_size
                        change_num -= 1
                        if change_num == 0:
                            break
            else:
                new_sample = copy.deepcopy(sample2)
                for sample_2_fusion in range(0,len(sample2['fusion'])):
                    for sample_1_size in sample1['tsize']:
                        if isOverflow(op_data, inst_fused_dict, isSinput, isPingpang, new_sample['fusion'][sample_2_fusion], sample_1_size, buffer_size):
                            continue
                        new_sample['tsize'][sample_2_fusion] = sample_1_size
                        change_num -= 1
                        if change_num == 0:
                            break
        return new_sample

def append_combine(params, generation, best_latency, num_combine, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, breakpoint):
    while len(generation) < num_combine:
        # randomly choose two samples from generation
        #先不考虑不同迭代间的重复
        sample1 = random.choice(generation)
        sample2 = random.choice(generation)
        new_sample = combine(params, sample1, sample2, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, breakpoint)
        if check_dup(generation,new_sample):
            continue
        if not prune(new_sample,best_latency): #粗略估计latency
            generation.append(new_sample)

def flip_bit_at_index(binary_str, index):
    if index < 0 or index >= len(binary_str):
        raise ValueError("索引超出范围")

    # 将字符串转换为列表，因为字符串是不可变的
    binary_list = list(binary_str)

    # 反转指定位置的位
    binary_list[index] = '0' if binary_list[index] == '1' else '1'

    # 将列表转换回字符串
    flipped_str = ''.join(binary_list)

    return flipped_str

#按01变异新的融合方式
#tsize x 2或者 / 2， x2过大，计算是否超buffer size
#出现不符合条件，就要补充新的下一代
#随机选mutate01或者tile size

#不用递归？优先级中等
def mutate(params, sample, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, mutate_config=[0.5,4]):
    while True:
        new_sample = copy.deepcopy(sample)
        mutate_choice = [0,1] #0 mutate fusion pattern; 1 mutate tile size
        choice = random.choice(mutate_choice)
        if choice == 0:
            #随机选择一个位置，改变01
            change_bit_num = math.ceil(len(sample['pattern'])/4) #选择1/4pattern进行改变
            mutate_bits = []
            for i in range(0,change_bit_num):
                mutate_bits.append(random.choice(range(0,len(sample['pattern']))))
            for mutate_bit in mutate_bits:
                new_pattern = flip_bit_at_index(sample['pattern'], mutate_bit)
            new_sample = gen_new_sample(new_pattern,params,sample['tsize'])
            if check_cycle(op_data, new_sample['fusion']) or not check_fusion_rules(breakpoint, new_sample):
                continue
        else:
            #可能可以不符合就不改了，保存当前？优先级低
            #每个块分别随机
            change_tile_num = math.ceil(len(sample['tsize'])/2) #选择一半tile size进行改变
            current_mutate_config = random.choice(mutate_config) #随机选择x2还是/2
            tile_id = []
            for i in range(0,change_tile_num):
                tile_id.append(random.choice(range(0,len(sample['tsize']))))
            for i in tile_id:
                mutate_tile_size =  math.ceil(math.ceil(sample['tsize'][i][0]*current_mutate_config)/16)*16
                if current_mutate_config > 1:
                    if mutate_tile_size > 8192:
                        mutate_tile_size = 8192
                    if isOverflow(op_data, inst_fused_dict, isSinput, isPingpang, new_sample['fusion'][i],  [mutate_tile_size,1], buffer_size):
                        #如果超过了大小，递归下一次mutate，减少size
                        mutate_config = [0.5, mutate_config[1]-0.5]
                        continue
                    else:
                        new_sample['tsize'][i][0] = mutate_tile_size
                else:
                    new_sample['tsize'][i][0] = mutate_tile_size
        return new_sample

def append_mutate(params, generation, best_latency, num_mutate, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, breakpoint):
    while len(generation) < num_mutate:
        # randomly choose one sample from generation
        sample = random.choice(generation)
        new_sample = mutate(params, sample, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size)
        if check_dup(generation,new_sample):
            continue
        if not prune(new_sample,best_latency):  #只检查mem_access
            generation.append(new_sample)

# use simulator.py, maybe use NN model to predict in the future
def evaluate(data_set, network, isReorder, isSinput, layer, sample):
    op_array = sample['fusion']
    tile_size_list = sample['tsize']
    interpret(data_set, network, isReorder, layer, op_array, tile_size_list)
    res = simulate(tile_size_list, data_set, network, layer, isReorder, isSinput)
    sample['latency'] = res[0]
    sample['mem_access'] = res[1]


def choose_parents(generation, num_parents):

    # choose the best num_parents samples
    parents = []
    #目前先按latency排序

    generation.sort(key=lambda x: x['latency'])

    best_sample = generation[0]

    for i in range(0, num_parents):
        parents.append(generation[i])

    return parents, best_sample

def judge_stable(current_best,best):
    delta = best - current_best
    if delta <= 0:
        return True
    return False

def genetic_search(params, data_set, network, isReorder, isSinput, layer,compiler_res, max_it, num_parents, num_combine, num_mutate, max_stable_iter, op_data, inst_fused_dict, isPingpang, buffer_size, breakpoint):
    first_generation = initialize(compiler_res,isPingpang)
    generation = first_generation
    # for sample in generation:
    #     evaluate(data_set, network, isReorder, isSinput, layer, sample)
    #     print("sample: ", sample)
    best_sample = generation[0]
    best_latency = generation[0]['latency']
    best_mem_access = generation[0]['mem_access']
    stable_iter = 0

    for i in range(0,max_it):  

        parents, current_best_sample = choose_parents(generation, num_parents)

        if judge_stable(current_best_sample['latency'],best_latency):
            stable_iter += 1 #没有提升或负提升，则stalbe_iter+1
        else:
            stable_iter = 0 #提升过大则stable_iter清零
        
        if stable_iter >= max_stable_iter:
            break
        
        if best_latency > current_best_sample['latency']:
            best_sample = copy.deepcopy(current_best_sample)
            best_latency = best_sample['latency']
            best_mem_access = best_sample['mem_access']

        #append_combine(params, parents, best_latency, num_combine, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, breakpoint) #保证latency有提升
        #print("parents after combine: ", parents)
        append_mutate(params, parents, best_latency, num_mutate, op_data, inst_fused_dict, isSinput, isPingpang, buffer_size, breakpoint)
        print("parents after mutate: ", parents)
        generation = parents

        for sample in generation:
            evaluate(data_set, network, isReorder, isSinput, layer, sample)
        
    return best_sample

def gen_new_sample(pattern, params, sizes):
    new_tile_size = []
    current_fused_array, current_tile_size, current_rw = gen_tile_info(
        params['isSinput'], params['isPingpang'], params['op_connected_info'], params['op_num'], pattern, params['op_data'], params['inst_fused_dict'], params['buffer_size'], params['tile_size_list'], params['sp_list'], params['node_num'], False
    )
    #从原先的sample中选size
    for one_fused_size in current_tile_size:
        for one_sample_size in sizes:
            if one_fused_size[0] > one_sample_size[0]:
                new_tile_size.append(one_sample_size)
                break
            new_tile_size.append(one_fused_size)
    return {'fusion': current_fused_array, 'tsize': new_tile_size, 'pingpong': params['isPingpang'], 'latency': 0, 'mem_access': current_rw, 'pattern': pattern}

def genetic_compile(dataset_name,network_name,layer_name,isReorder,isSinput,isPingpang):
    max_it = 3
    num_parents = 2
    num_combine = 3
    num_mutate = 4
    max_stable_iter = 3

    inst_fused_dict = load_inst_fused('hardware_info.yaml')

    compiler_res, op_data, buffer_size, node_num, sp_list, tile_size_list = compile(dataset_name,network_name,layer_name,isReorder,isSinput,isPingpang)
    op_connected_info, op_num, skip_bits, breakpoint = gen_op_connected_info(op_data)

    params = {
            'isSinput': isSinput,
            'isPingpang': isPingpang,
            'op_connected_info': op_connected_info,
            'op_num': op_num,
            'op_data': op_data,
            'inst_fused_dict': inst_fused_dict,
            'buffer_size': buffer_size,
            'tile_size_list': tile_size_list,
            'sp_list': sp_list,
            'node_num': node_num
        }


    res = genetic_search(params, dataset_name, network_name, isReorder, isSinput, layer_name, compiler_res, max_it, num_parents, num_combine, num_mutate, max_stable_iter, op_data, inst_fused_dict, isPingpang, buffer_size, breakpoint)

    
    return res

if __name__ == '__main__':

    res = genetic_compile('cora','GCN','layer1',False,False,True)

    print(res)