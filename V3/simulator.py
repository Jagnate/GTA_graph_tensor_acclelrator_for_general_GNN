import yaml
import math
import time

#硬件参数
#cycle: 1ns = 1cycle
BW    = 128*(1024**3)*(10**(-9)) #128GB/s = 128*10^-9 GB/cycle
PL_IN  = 16 #16*16 feature/cycle
PL_OUT = 16

#buffer_size = 2*1024*1024 #byte

rw = 0
rw_record = []

def read(path):
    try:
        with open(path, 'r') as file:
            data = file.read()
            result = yaml.load(data, Loader=yaml.FullLoader)
            return result
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None
    

def find_inst(full_inst_list, type, id):
    for i in range(0,len(full_inst_list)):
        for j in range(0,len(full_inst_list[i])):
            if full_inst_list[i][j]['TYPE'] == type and full_inst_list[i][j]['ID'] == id:
                return j

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

def occupy_hardware(inst,hardware_unit_status,hardware_unit_list):
    hardware_unit_status[hardware_unit_list.index(inst['Hardware_Unit'])] = 'Occupied'

def release_hardware(inst,hardware_unit_status,hardware_unit_list):
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

def calculate_running_cycle(inst,remain_times,data,node_num):
    global rw
    global rw_record
    global BW
    global PL_IN
    global PL_OUT
    res = 0
    if inst['TYPE'] == 'LOAD_E' or inst['TYPE'] == 'STORE_E':
        data_size = data[len(data) - remain_times]*inst['Feature_Length']
        res = math.ceil(data_size/BW)
        rw_record.append([inst['TYPE'],inst['ID'],data[len(data) - remain_times],data_size])
        rw += data_size
        return res
    elif inst['TYPE'] == 'LOAD_W' or inst['TYPE'] == 'LOAD_N' or inst['TYPE'] == 'STORE_N':
        if (inst['TYPE'] == 'LOAD_N' or inst['TYPE'] == 'STORE_N') and remain_times == 1:
            block = node_num - math.floor(node_num/inst['Tile_Size'])*inst['Tile_Size']
            if block == 0:
                block = inst['Tile_Size']
            data_size = block*inst['Feature_Length']
        else:
            data_size = inst['Tile_Size']*inst['Feature_Length']
        res = math.ceil(data_size/BW)
        rw += data_size
        rw_record.append([inst['TYPE'],inst['ID'],data_size,remain_times])
        return res
    elif inst['TYPE'] == 'COMP_MM':
        res = math.ceil(inst['Feature_Length']/(4*PL_IN))*math.ceil((inst['Weight_Size']/inst['Feature_Length'])/PL_OUT)
        return res
    else: 
        if 'applyedge' in inst['ID'] or 'gather' in inst['ID']:
            res = math.ceil(data[len(data) - remain_times]*inst['Feature_Length']/PL_IN)
        else:
            res = math.ceil(inst['Tile_Size']*inst['Feature_Length']/PL_IN)
        return res
        
def update_timeline(inst,timeline,cycle,remain_cycle,hardware_unit_list):
    timeline[hardware_unit_list.index(inst['Hardware_Unit'])].append([inst['TYPE'],inst['ID'],cycle,cycle+remain_cycle])

def save_timeline_to_files(timeline):
    hardware_units = ['LOAD', 'ALU', 'MM', 'STORE']
    for i, unit in enumerate(hardware_units):
        filename = f"{unit}_timeline.txt"
        with open(filename, 'w') as file:
            for fused_block_timeline in timeline:
                for entry in fused_block_timeline[i]:
                    file.write(f"{entry}\n")

def save_rw_record_to_file(rw_record, filename="rw_record.txt"):
    with open(filename, 'w') as file:
        for record in rw_record:
            file.write(f"{record}\n")

if __name__ == '__main__':

    tile_size_list = [[16, 1], [1536, 1], [768, 1], [1280, 1], [1408, 1]]
    node_num = 2708

    full_inst = read('/Users/sijin/Desktop/workspace/GTA_Code/dataset/inst.yaml')


    fused_list_index = 0
    previous_inst_list, next_inst_list = gen_link(full_inst)
    dep_matrix = init_dep_matrix(full_inst,next_inst_list) #初始化依赖矩阵

    hardware_unit_list = ['LOAD','ALU','MM','STORE']
    hardware_unit_status = init_hardware_unit_status(hardware_unit_list) #初始化硬件状态

    timeline = []
    timeline.append(init_timeline(hardware_unit_list)) #初始化时间线

    cycle = 0
    
    inst_list = full_inst[fused_list_index]
    inst_status = init_inst_status(inst_list) #初始化指令状态
    data = []
    path = '/Users/sijin/Desktop/workspace/GTA_Code/dataset/adj_cora_'+str(tile_size_list[fused_list_index][0])+'_1.yaml'
    data = flatten_2d_list(read(path))
    start_time = time.time()  # 记录开始时间

    while True:        

        if cycle % 100000 == 0:
            elapsed_time = time.time() - start_time  # 计算经过的时间
            print('cycle:', cycle, 'fused_block:', fused_list_index), 
            print('elapsed time:', elapsed_time, 'seconds')
            print(inst_status)
            start_time = time.time()  # 重置开始时间

        if cycle == 0:
            print(dep_matrix)
        
        unfinished = False
        for inst_num in range(0,len(inst_list)):
            current_inst = inst_list[inst_num]
            current_next_inst = next_inst_list[fused_list_index][inst_num]
            current_prev_inst = previous_inst_list[fused_list_index][inst_num]
            state = inst_status[inst_num]["state"]
            
            if state == 'WAITING':
                unfinished = True
                if check_dependency(current_inst,inst_num,dep_matrix[fused_list_index],current_next_inst,current_prev_inst) and check_hardware(current_inst, hardware_unit_list, hardware_unit_status):
                    consume_dependency(current_inst,inst_num,dep_matrix[fused_list_index],current_next_inst,current_prev_inst)
                    occupy_hardware(current_inst,hardware_unit_status,hardware_unit_list)
                    inst_status[inst_num]["state"] = 'RUNNING'
                    inst_status[inst_num]["remain_cycle"] = calculate_running_cycle(current_inst,inst_status[inst_num]["remain_times"],data,node_num)
                    update_timeline(current_inst,timeline[-1],cycle,inst_status[inst_num]["remain_cycle"],hardware_unit_list)
            elif state == 'RUNNING':
                unfinished = True
                inst_status[inst_num]["remain_cycle"] -= 1
                if inst_status[inst_num]["remain_cycle"] == 0 or inst_status[inst_num]["remain_cycle"] == -1:
                    enable_dependency(current_inst,inst_num,dep_matrix[fused_list_index],current_next_inst,current_prev_inst)
                    release_hardware(current_inst,hardware_unit_status,hardware_unit_list)
                    inst_status[inst_num]["remain_times"] -= 1
                    if inst_status[inst_num]["remain_times"] == 0:
                        inst_status[inst_num]["state"] = 'FINISHED'
                    else:
                        inst_status[inst_num]["state"] = 'WAITING'
            else: # FINISHED
                continue

        if not unfinished:
            fused_list_index += 1
            #full_inst.pop(0)
            if fused_list_index != len(full_inst):
                inst_list = full_inst[fused_list_index]
                path = '/Users/sijin/Desktop/workspace/GTA_Code/dataset/adj_cora_'+str(tile_size_list[fused_list_index][0])+'_1.yaml'
                data = flatten_2d_list(read(path))
                inst_status = init_inst_status(inst_list) #初始化指令状态
                timeline.append(init_timeline(hardware_unit_list))
            else:
                break
        
        cycle += 1
        
        num = 0
        for i in inst_status:
            if i['state'] == 'RUNNING':
                break
            else:
                num += 1

    save_timeline_to_files(timeline)
    print(rw)
    save_rw_record_to_file(rw_record)                                                                                                                                                                                                        ['STORE_E', '7_applyedge_0', 6981, 7123]