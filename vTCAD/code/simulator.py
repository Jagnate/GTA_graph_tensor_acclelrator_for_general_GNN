import yaml
import math
import time
import os
import numpy as np
import scipy.sparse
import json

#每种硬件配置怎么跑

#硬件参数
#cycle: 1ns = 1cycle
# isH4 = False
# vec_parm = [8,16]
# sf_parm = [8,16]
# mm_parm = [8,16]

# isH4 = False
# vec_parm = [4,16]
# sf_parm = [8,16]
# mm_parm = [12,16]

# isH4 = False
# vec_parm = [2,16]
# sf_parm = [8,16]
# mm_parm = [14,16]

# isH4 = True
# vec_parm = [8,16]
# sf_parm = [8,16]
# mm_parm = [16,16]

#HyGCN
# isH4 = False
# vec_parm = [2,16]
# sf_parm = [2,16]
# mm_parm = [16,16]

#GCNAX
# isH4 = False
# vec_parm = [2,128]
# sf_parm = [2,128]
# mm_parm = [2,128]

#OPU
# isH4 = True
# vec_parm = [12,16]
# sf_parm = [8,16]
# mm_parm = [16,16]


#isH4 = False
#GTA[0][1][2];HyGCN[3];GCNAX[4];OPU[5]
vec_parm = [[8,16],[4,16], [2,16], [2,16], [2,128],[12,16]]
sf_parm =  [[8,16],[8,16], [8,16], [2,16], [2,128],[8,16]]
mm_parm =  [[8,16],[12,16],[14,16],[16,16],[2,128],[16,16]]

#BW    = 128*(1024**3)*(10**(-9)) #128GB/s = 128*10^-9 GB/cycle
BW = 128 #128 byte/cycle
hardware_performance = {
    'Memory_Access_Unit':   BW,
    'VEC_ALU':              vec_parm,
    'SF_ALU':               sf_parm,
    'MM':                   mm_parm,
    'Virtual_Loader':       BW, #虽然是BW，但实际执行不执行BW，指用来判断类型
}
hardware_unit_list = ['Memory_Access_Unit','VEC_ALU','SF_ALU','MM','Virtual_Loader']
#访存量
rw = 0
rw_record = []

tmp = 0

def read_csr_npz(file_path):
    with np.load(file_path) as data:
        # 打印文件中的所有数组名称
        print("Arrays in the .npz file:", data.files)
        
        # 读取稀疏矩阵的数据
        if 'data' in data and 'indices' in data and 'indptr' in data and 'shape' in data:
            csr_matrix = scipy.sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
            return csr_matrix
        else:
            raise KeyError("The required keys are not found in the .npz file.")

def csr_to_vector(csr_matrix):
    dense_matrix = csr_matrix.toarray()
    vector = dense_matrix.flatten()
    return vector

def group_vector(vector, group_size):
    # 计算每组的和
    grouped_vector = [np.sum(vector[i:i + group_size]) for i in range(0, len(vector), group_size)]
    return np.array(grouped_vector)

def process_and_save(file_path, group_size):
    # 读取 npz 文件
    csr_matrix = read_csr_npz(file_path)
    
    # 将稀疏矩阵展开成向量
    vector = csr_to_vector(csr_matrix)
    
    # 对每个分组大小进行处理并保存
    grouped_vector = group_vector(vector, group_size)

    return grouped_vector

def read(path):
    try:
        with open(path, 'r') as file:
            data = file.read()
            result = yaml.load(data, Loader=yaml.FullLoader)
            return result
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None
    
def aggregate_rw_record():

    global rw_record

    value_result = {}
    count_result = {}

    for record in rw_record:
        record_type = record[0]
        last_value = record[-1]

        if record_type not in value_result:
            value_result[record_type] = 0
            count_result[record_type] = 0
        
        value_result[record_type] += last_value
        count_result[record_type] += 1

    return value_result, count_result

def aggregate_timeline(timeline):
    type_count = {}
    total_cycles = {}

    for unit_timeline in timeline:
        for entry in unit_timeline:
            for one_inst in entry:
                entry_type = one_inst[0]
                start_cycle = one_inst[2]
                end_cycle = one_inst[3]
                cycle_duration = end_cycle - start_cycle

                if entry_type not in type_count:
                    type_count[entry_type] = 0
                    total_cycles[entry_type] = cycle_duration

                type_count[entry_type] += 1
                total_cycles[entry_type] += cycle_duration

    return type_count, total_cycles

def find_inst(full_inst_list, type, id):
    for i in range(0,len(full_inst_list)):
        for j in range(0,len(full_inst_list[i])):
            if type == full_inst_list[i][j]['TYPE'] and id == full_inst_list[i][j]['ID']:
                return j
    return 'wrong'+str(type)+str(id)

def gen_link(full_inst_list):
    previous_inst_list = []
    next_inst_list = []
    for i in range(0,len(full_inst_list)):
        next_inst_list.append([])
        previous_inst_list.append([])
        for j in range(0,len(full_inst_list[i])):
            next_inst_list[-1].append([])
            previous_inst_list[-1].append([])
            for k in range(0,len(full_inst_list[i][j]['Dependency']['RAW'])):
                previous_inst_list[-1][-1].append(find_inst(full_inst_list,full_inst_list[i][j]['Dependency']['RAW'][k]['TYPE'],full_inst_list[i][j]['Dependency']['RAW'][k]['ID']))
            for k in range(0,len(full_inst_list[i][j]['Dependency']['WAR'])):
                next_inst_list[-1][-1].append(find_inst(full_inst_list,full_inst_list[i][j]['Dependency']['WAR'][k]['TYPE'],full_inst_list[i][j]['Dependency']['WAR'][k]['ID']))
    return previous_inst_list, next_inst_list

#初始化依赖矩阵，全部war依赖
def init_dep_matrix(full_inst,next_inst_list):
    dep_matrix = []
    for i in range(0,len(full_inst)):
        dep_matrix.append([[0 for _ in range(len(full_inst[i]))] for _ in range(len(full_inst[i]))])
        for j in range(0,len(full_inst[i])):
            if next_inst_list[i][j] != []:
                for k in range(0,len(next_inst_list[i][j])):
                    #dep_matrix[i][j][next_inst_list[i][j][k]] = 2*full_inst[i][j]['Dependency']['WAR'][k]['Times'][1]
                    #dep_matrix[i][j][next_inst_list[i][j][k]] = 2*TC
                    dep_matrix[i][j][next_inst_list[i][j][k]] = 2 * full_inst[i][j]['Dependency']['WAR'][k]['Times'][0] * full_inst[i][j]['Dependency']['WAR'][k]['Times'][1]

    return dep_matrix

def init_hardware_unit_status(hardware_unit_list):
    hardware_unit_status = [] # IDLE / Occupied
    for i in hardware_unit_list:
        hardware_unit_status.append('IDLE')
    return hardware_unit_status

def init_timeline(hardware_unit_list):
    timeline = []
    for i in hardware_unit_list:
        timeline.append([])
    return timeline

def init_inst_status(inst_list):
    inst_status = []
    for inst in inst_list:
        inst_status.append({
            'state': 'WAITING', # WAITING / RUNNING / FINISHED
            'remain_times': inst['Tile_Times'],
            'remain_cycle': 0
        })
    return inst_status

def check_hardware(inst,hardware_unit_list,hardware_unit_status):
    if hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])] == 'IDLE':
        return True
    else:
        return False

def check_dependency(inst, inst_num,dep_matrix,next_inst,prev_inst):
    if inst['Dependency']['RAW'] != []:
        for i in range(0,len(prev_inst)):
            if dep_matrix[inst_num][prev_inst[i]] < inst['Dependency']['RAW'][i]['Times'][1]:
                return False
    if inst['Dependency']['WAR'] != []:
        for i in range(0,len(next_inst)):
            if dep_matrix[inst_num][next_inst[i]] < inst['Dependency']['WAR'][i]['Times'][1]:
                return False
    return True

def consume_dependency(inst, inst_num,dep_matrix,next_inst,prev_inst):
    for i in range(0,len(next_inst)):
        dep_matrix[inst_num][next_inst[i]] -= inst['Dependency']['WAR'][i]['Times'][1] 
    for i in range(0,len(prev_inst)): #raw
        dep_matrix[inst_num][prev_inst[i]] -= inst['Dependency']['RAW'][i]['Times'][1]

def enable_dependency(inst, inst_num,dep_matrix,next_inst,prev_inst):
    for i in range(0,len(next_inst)):
        dep_matrix[next_inst[i]][inst_num] += inst['Dependency']['WAR'][i]['Times'][1]
    for i in range(0,len(prev_inst)):
        dep_matrix[prev_inst[i]][inst_num] += inst['Dependency']['RAW'][i]['Times'][1]

def occupy_hardware(inst,hardware_unit_status,hardware_unit_list, isH4):
    if isH4:
        if inst['Hardware_Unit'] == 'VEC_ALU':
            hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])+2] = 'Occupied'
        elif inst['Hardware_Unit'] == 'MM':
            hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])-2] = 'Occupied'
    hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])] = 'Occupied'

def release_hardware(inst,hardware_unit_status,hardware_unit_list, isH4):
    if isH4:
        if inst['Hardware_Unit'] == 'VEC_ALU':
            hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])+2] = 'IDLE'
        elif inst['Hardware_Unit'] == 'MM':
            hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])-2] = 'IDLE'
    hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])] = 'IDLE'

def flatten_2d_list(two_d_list):
    return [item for sublist in two_d_list for item in sublist]

def get_boundary_indices(rows, cols):
    boundary_indices = []
    for row in range(rows):
        for col in range(cols):
            if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                boundary_indices.append(row * cols + col)
    return boundary_indices

#TODO: 找到mm返回true
def find_war_mm(inst):
    war_list = inst['Dependency']['WAR']
    for i in war_list:
        if i['TYPE'] == 'COMP_MM' and i['ID'].split('_')[0] == inst['ID'].split('_')[0]:
            return True
    return False

def calculate_running_cycle(inst,remain_times,data,node_num,sparse_op_list,sparsity,isSparseFeature,isSparseWeight,isLoadEValid,isLoadNValid,isSinput,fusionBlock_num):
    global rw
    global rw_record
    hardware_unit = inst['Hardware_Unit']
    performance = hardware_performance.get(hardware_unit)
    
    res = 0
    if inst['TYPE'] == 'LOAD_E' or inst['TYPE'] == 'STORE_E':
        if inst['TYPE'] == 'LOAD_E' and isLoadEValid:
            if inst['Feature_Length'] <= 16:
                return 0 #load_E如果feature长度小于等于4，不占用时间
        data_size = data[len(data) - remain_times]*inst['Feature_Length']
        res = math.ceil(data_size/performance)
        rw_record.append([inst['TYPE'],inst['ID'],data[len(data) - remain_times],data_size])
        rw += data_size
        return res
    elif inst['TYPE'] == 'LOAD_W' or inst['TYPE'] == 'LOAD_N' or inst['TYPE'] == 'STORE_N':
        s = 1
        data_size = inst['Tile_Size']*inst['Feature_Length']
        if hardware_unit == 'Virtual_Loader' and isLoadNValid:
            return 1 #Gather的LOAD_N直接return，只读1个周期
        if inst['TYPE'] == 'LOAD_W' or inst['TYPE'] == 'LOAD_N':
            #sinput不读N
            if inst['TYPE'] == 'LOAD_N' and '0_applynode' in inst['ID'] and isSinput and find_war_mm(inst): #sinput
                rw += math.ceil(inst['Tile_Size']*inst['Feature_Length']*sparsity)
                return math.ceil(inst['Tile_Size']*inst['Feature_Length']*sparsity/performance)
            # if inst['TYPE'] == 'LOAD_N' and data[len(data) - remain_times] == 0 and math.ceil(node_num/inst['Tile_Size'])*node_num == inst['Tile_Times']:
            #     return 0 #load_N如果是TRTC次，且feature长度为0，则不占用时间
            # if (int(inst['ID'].split('_')[0]) in sparse_op_list and isSparseFeature and inst['TYPE'] == 'LOAD_N') or (int(inst['ID'].split('_')[0]) in sparse_op_list and isSparseWeight and inst['TYPE'] == 'LOAD_W'):
            #     s = 0.6 #稀疏特征读取减少60%
        #存储的时候考虑了padding
        if (inst['TYPE'] == 'LOAD_N' or inst['TYPE'] == 'STORE_N') and remain_times == 1:
            block = node_num - math.floor(node_num/inst['Tile_Size'])*inst['Tile_Size']
            if block == 0:
                block = inst['Tile_Size']
            data_size = block*inst['Feature_Length']
        res = math.ceil(data_size*s/performance)
        rw += math.ceil(data_size*s)
        rw_record.append([inst['TYPE'],inst['ID'],data_size])
        return res
    elif inst['TYPE'] == 'COMP_MM':
        if '0_applynode' in inst['ID'] and isSinput: #sinput计算按照加法来进行
            res_n_cin = math.ceil(inst['Tile_Size']*inst['Feature_Length']*sparsity/hardware_performance.get('VEC_ALU')[fusionBlock_num][0]/4) #个数 * byte / 个数单位不统一,所以需要再/4
            res_out_feature = math.ceil(inst['Weight_Size']/inst['Feature_Length']/hardware_performance.get('VEC_ALU')[fusionBlock_num][1]) #byte / byte
            return res_n_cin * res_out_feature
        s = 1
        # if int(inst['ID'].split('_')[0]) in sparse_op_list and isSparseWeight:
        #     s = 0.5
        #1. 乘tile_size
        #2. feature_length单位仍为byte
        #3. 统一size和length的单位
        #4. 拆开新变量

        #TODO: 
        #1. inst['Tile_Size']如果是COMP_MM + node + 不是最后一块，按下面的去算
        #2. inst['Tile_Size']如果是COMP_MM + node + 是最后一块，加上吗的if判断，改tile size
        #3. inst['Tile_Size']如果是COMP_MM + edge，按data[len(data) - remain_times]去读
        res_n_cin = math.ceil((inst['Feature_Length'])/performance[fusionBlock_num][1]/4) * inst['Tile_Size'] #byte / 个数单位不统一，需要额外/4
        res_out_feature = math.ceil((inst['Weight_Size']/inst['Feature_Length'])/performance[fusionBlock_num][0]) #byte / byte
        res = res_n_cin*res_out_feature*s
        return res
    else: 
        if 'applyedge' in inst['ID'] or 'gather' in inst['ID']:
            s = 1
            # if int(inst['ID'].split('_')[0]) in sparse_op_list and isSparseFeature:
            #     s = 0.65
            if(remain_times == 1):
                print(data[len(data) - remain_times])
            n_res = math.ceil(data[len(data) - remain_times]/performance[fusionBlock_num][0])
            output_feature_res = math.ceil(inst['Feature_Length']/performance[fusionBlock_num][1]/4) #byte / 个数 / 4
            res = n_res*output_feature_res*s
        else:
            #TODO: 最后一块需要判断if
            res = math.ceil(inst['Tile_Size']/performance[fusionBlock_num][0])*math.ceil(inst['Feature_Length']/performance[fusionBlock_num][1]/4) #byte / 个数 / 4
        return res
        
def update_timeline(inst,timeline,cycle,remain_cycle,hardware_unit_list):
    timeline[hardware_unit_list.index(inst['Hardware_Unit'])].append([inst['TYPE'],inst['ID'],cycle,cycle+remain_cycle])

def save_timeline_to_json(hardware_units, timeline, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    events = []
    
    filename = os.path.join(folder_name, f"chrome_timeline.json")

    for i, unit in enumerate(hardware_units):
            
        for fused_block_timeline in timeline:
            for entry in fused_block_timeline[i]:
                events.append(
                    {
                        "name": entry[0],  # 事件名称
                        "cat": entry[1],  # 硬件单元
                        "ph": "X",  # 表示完整事件
                        "ts": entry[2],  # 开始时间
                        "dur": entry[3] - entry[2] ,  # 持续时间
                        "pid": unit,  # 进程 ID，可根据需要修改
                        "tid": unit   # 线程 ID，可根据需要修改
                    }
                )
    with open(filename, "w") as file:
        json.dump(events, file, indent=4)
    # 将事件列表转换为 JSON 格式

def save_timeline_to_files(hardware_units, timeline, folder_name):
    # 确保目录存在
    os.makedirs(folder_name, exist_ok=True)
    
    for i, unit in enumerate(hardware_units):
        filename = os.path.join(folder_name, f"{unit}_timeline.txt")

        with open(filename, 'w') as file:
            
            for fused_block_timeline in timeline:
                for entry in fused_block_timeline[i]:
                    file.write(f"{entry}\n")

def save_rw_record_to_file(rw_record, folder_name, rw_info, filename="rw_record.txt"):
    # 确保目录存在
    os.makedirs(folder_name, exist_ok=True)
    
    filepath = os.path.join(folder_name, filename)
    
    with open(filepath, 'w') as file:
        # 写入 rw_info 的信息
        file.write(f"不同指令的访存量: {rw_info[0]}\n")
        file.write(f"不同指令的条数: {rw_info[1]}\n")
        
        for record in rw_record:
            file.write(f"{record}\n")

def save_timeline_info(folder_name, timeline_info):
    # 确保目录存在
    os.makedirs(folder_name, exist_ok=True)
    
    filename = os.path.join(folder_name, "timeline_info.txt")
    
    with open(filename, 'w') as file:
        # 写入 timeline_info 中的统计信息，并在合适的位置断开换行
        file.write(f"不同指令的Cycle数: {timeline_info[1]}\n")
        file.write(f"不同指令的条数: {timeline_info[0]}")

def simulate(tile_size_list,dataset,network,layer,isReorder,isSinput, isFlexibleHardware, architecture_name):

    global rw
    global rw_record
    rw = 0
    rw_record = []

    print("Simulating...")
    
    #初始信息
    
    if dataset == 'cora':
        node_num = 2708
        sparsity = 0.012 #cora的稀疏度
    elif dataset == 'pubmed':
        node_num = 19717
        sparsity = 0.1 #pubmed的稀疏度
    elif dataset == 'flickr':
        node_num = 89250
        sparsity = 0.46 #flickr的稀疏度
    elif dataset == 'reddit':
        node_num = 232965
        sparsity = 1

    op_map = 'original'
    if isReorder:
        op_map = 'trans'

    instpath = 'Results/Insts/'+network+'-'+dataset+'-'+layer+'-'+op_map+'.yaml'
    full_inst = read(instpath)
    sparse_op_list = []
    isLoadEValid = True #loade siez小于等于4不占用时间
    isLoadNValid = True #gather的load_n只读1个周期
    isSparseWeight = False #稀疏weight计算减少50%
    isSparseFeature = False #稀疏特征读取减少60%
    #isSinput = False
    isH4 = False
    if(sparsity < 0.5):
        isSinput = True #applynode的稀疏表示

    #初始化依赖矩阵
    fused_list_index = 0
    previous_inst_list, next_inst_list = gen_link(full_inst)
    dep_matrix = init_dep_matrix(full_inst,next_inst_list) 

    #初始化硬件状态
    hardware_unit_status = init_hardware_unit_status(hardware_unit_list) 

    #初始化时间线
    timeline = []
    timeline.append(init_timeline(hardware_unit_list)) 

    #初始化
    cycle = 0
    
    inst_list = full_inst[fused_list_index]
    inst_status = init_inst_status(inst_list) #初始化指令状态
    data = []
    path = 'dataset/'+dataset+'/adj_'+dataset+'_'+str(tile_size_list[fused_list_index][0])+'_1.yaml'
    data = flatten_2d_list(read(path))
    #data = np.load('dataset-npz/Flickr/grouped_vector_'+str(tile_size_list[fused_list_index][0])+'.npy')
    start_time = time.time()  # 记录开始时间

    #用来记录灵活硬件
    total_hardware_num = 0
    current_hardware_num = 0
    if architecture_name == 'GTA':
        isH4 = False
        if(isFlexibleHardware):
            total_hardware_num = 3
    elif architecture_name == 'HyGCN':
        isH4 = False
        if(isFlexibleHardware):
            isFlexibleHardware = False
        current_hardware_num = 3
    elif architecture_name == 'GCNAX':
        isH4 = False
        if(isFlexibleHardware):
            isFlexibleHardware = False
        current_hardware_num = 4
    elif architecture_name == 'OPU':
        isH4 = True
        if(isFlexibleHardware):
            isFlexibleHardware = False
        current_hardware_num = 5
    else:
        print('Wrong architecture name')
        return -1
    
    start_cycle = 0
    fast_cycle = 0
    fast_timeline = []
    start_rw = 0

    while True:        

        #flag = True #用来解决cycle=0时，inst_status[inst_num]["remain_cycle"] = 0的问题

        if cycle % 100000 == 0:
            elapsed_time = time.time() - start_time  # 计算经过的时间
            # print('cycle:', cycle, 'fused_block:', fused_list_index), 
            # print('elapsed time:', elapsed_time, 'seconds')
            # print(inst_status)
            start_time = time.time()  # 重置开始时间
        
        unfinished = False
        for inst_num in range(0,len(inst_list)):
            #浅拷贝
            current_inst = inst_list[inst_num]
            current_next_inst = next_inst_list[fused_list_index][inst_num]
            current_prev_inst = previous_inst_list[fused_list_index][inst_num]
            state = inst_status[inst_num]["state"]
            
            if state == 'WAITING':
                unfinished = True
                if check_dependency(current_inst,inst_num,dep_matrix[fused_list_index],current_next_inst,current_prev_inst) and check_hardware(current_inst, hardware_unit_list, hardware_unit_status):
                    consume_dependency(current_inst,inst_num,dep_matrix[fused_list_index],current_next_inst,current_prev_inst)
                    remain_times = math.ceil(inst_status[inst_num]["remain_times"])
                    if 'COMP' in current_inst['TYPE'] and ('applyedge' in current_inst['ID'] or 'gather' in current_inst['ID']):
                        if data[len(data) - remain_times] > 0.9 * current_inst['Tile_Size']:
                            current_inst['Hardware_Unit'] = 'MM'
                        else:
                            current_inst['Hardware_Unit'] = 'VEC_ALU'
                    occupy_hardware(current_inst,hardware_unit_status,hardware_unit_list, isH4)
                    inst_status[inst_num]["state"] = 'RUNNING'
                    #TODO:
                    #1. 计算cycle，如果cycle=0，不会直接结束，而是多一个周期，再结束？
                    tmp_cycle = calculate_running_cycle(current_inst,remain_times,data,node_num,sparse_op_list,sparsity,isSparseFeature,isSparseWeight,isLoadEValid,isLoadNValid, isSinput, current_hardware_num)
                    #if tmp_cycle == 0 or tmp_cycle == 1:
                    #    flag = False
                    inst_status[inst_num]["remain_cycle"] = tmp_cycle
                    #if tmp_cycle != 0:
                    update_timeline(current_inst,timeline[-1],cycle,inst_status[inst_num]["remain_cycle"],hardware_unit_list)
            elif state == 'RUNNING':
                unfinished = True
                inst_status[inst_num]["remain_cycle"] -= 1
                if inst_status[inst_num]["remain_cycle"] == 0 or inst_status[inst_num]["remain_cycle"] == -1:
                #if inst_status[inst_num]["remain_cycle"] == 0 or inst_status[inst_num]["remain_cycle"] == 1 or inst_status[inst_num]["remain_cycle"] == -1:
                    #if inst_status[inst_num]["remain_cycle"] == -1:
                        #flag = False
                    enable_dependency(current_inst,inst_num,dep_matrix[fused_list_index],current_next_inst,current_prev_inst)
                    release_hardware(current_inst,hardware_unit_status,hardware_unit_list, isH4)
                    inst_status[inst_num]["remain_times"] -= 1
                    if inst_status[inst_num]["remain_times"] == 0:
                        inst_status[inst_num]["state"] = 'FINISHED'
                    else:
                        inst_status[inst_num]["state"] = 'WAITING'
            else: # FINISHED
                continue

        if not unfinished:
            if(isFlexibleHardware):
                if(current_hardware_num < total_hardware_num - 1):
                    current_hardware_num += 1
                    unfinished = False
                    if(fast_cycle == 0):
                        fast_cycle = cycle
                        fast_timeline = timeline[-1]
                    else:
                        if(cycle < fast_cycle):
                            fast_cycle = cycle
                            fast_timeline = timeline[-1]
                    inst_status = init_inst_status(inst_list)
                    timeline[-1] = init_timeline(hardware_unit_list)
                    rw = start_rw
                    cycle = start_cycle
                    continue
                else:
                    #灵活硬件
                    if(cycle > fast_cycle and fast_cycle != 0):
                        cycle = fast_cycle
                        timeline[-1] = fast_timeline
                    fast_cycle = 0
                    start_cycle = cycle
                    start_rw = rw
                    current_hardware_num = 0

                    fused_list_index += 1
                    if fused_list_index != len(full_inst):
                        inst_list = full_inst[fused_list_index]
                        if tile_size_list[fused_list_index][0] != tile_size_list[fused_list_index-1][0]:
                            path = 'dataset/'+dataset+'/adj_'+dataset+'_'+str(tile_size_list[fused_list_index][0])+'_1.yaml'
                            data = flatten_2d_list(read(path))
                            #data = np.load('dataset-npz/Flickr/grouped_vector_'+str(tile_size_list[fused_list_index][0])+'.npy')
                        inst_status = init_inst_status(inst_list) #初始化指令状态
                        timeline.append(init_timeline(hardware_unit_list))
                    else:
                        break
            else:
                fused_list_index += 1
                if fused_list_index != len(full_inst):
                    inst_list = full_inst[fused_list_index]
                    if tile_size_list[fused_list_index][0] != tile_size_list[fused_list_index-1][0]:
                        path = 'dataset/'+dataset+'/adj_'+dataset+'_'+str(tile_size_list[fused_list_index][0])+'_1.yaml'
                        data = flatten_2d_list(read(path))
                        #data = np.load('dataset-npz/Flickr/grouped_vector_'+str(tile_size_list[fused_list_index][0])+'.npy')
                    inst_status = init_inst_status(inst_list) #初始化指令状态
                    timeline.append(init_timeline(hardware_unit_list))
                else:
                    break
        
        #if flag:
        cycle += 1
        
        num = 0
        for i in inst_status:
            if i['state'] == 'RUNNING':
                break
            else:
                num += 1

    print('总cycle数',cycle-1)
    print('总访存量',rw)

    test_name = 'Results'
    timeline_info = aggregate_timeline(timeline)
    rw_info = aggregate_rw_record()
    save_timeline_to_files(hardware_unit_list,timeline,test_name)
    save_rw_record_to_file(rw_record,test_name,rw_info)
    save_timeline_info(test_name,timeline_info)
    save_timeline_to_json(hardware_unit_list,timeline,test_name)

    
    print('不同指令的Cycle数:',timeline_info[1])
    print('不同指令的访存量：',rw_info[0])
    print(tmp)
    
    return cycle-1,rw

if __name__ == '__main__':
    data_set = 'cora'
    network = 'GAT'
    isReorder = False
    layer = 'layer1'
    op_array = [[0], [2], [3], [5], [6], [7], [8], [9], [10], [11], [12], [13], [1, 4]]
    tile_size_list = [[96, 1], [1808, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [1872, 1], [1008, 1], [1616, 1]]
    #(5141848, 245730192)
    print(simulate(tile_size_list,data_set,network,layer,isReorder,False,True,'OPU'))