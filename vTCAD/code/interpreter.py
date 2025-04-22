import yaml
import math
import os

hardware_unit_mapping = {
    'ADD':      'VEC_ALU',
    'SF':       'SF_ALU',
    'MUL':      'VEC_ALU',
    'LOAD':     'Memory_Access_Unit',
    'LOAD_N':   'Virtual_Loader',
    'STORE':    'Memory_Access_Unit',
    'MM':       'MM',
    'Fused':    'MM'
}

# hardware_unit_mapping = {
#     'ADD':      'VEC_ALU',
#     'SF':       'SF_ALU',
#     'MUL':      'VEC_ALU',
#     'LOAD':     'Memory_Access_Unit',
#     'LOAD_N':   'Virtual_Loader',
#     'STORE':    'Memory_Access_Unit',
#     'MM':       'VEC_ALU',
#     'Fused':    'VEC_ALU'
# }

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def write(data,file_name):

    path = 'Results/Insts'

    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, file_name)

    with open(filepath, 'w') as file:
        yaml.dump(data, file, Dumper=NoAliasDumper)

def judge_comp_inst_type(op_type, compute_type):
    if op_type == 'scatter':
        return 'FETCH'
    else:
        return 'COMP_'+compute_type

def judge_load_inst_tile(op_type, op_order, load_type, TR, TC, SR, SC):
    tile_times = 0
    tile_size = 0

    if load_type == 'LOAD_W':
        return 1, 1

    if op_type == 'scatter':
        if op_order == 'R':
            tile_times = TR
            tile_size = SR
        else:
            tile_times = TR*TC
            tile_size = SC #TODO: check
    elif op_type == 'gather' or op_type == 'applyedge':
        tile_times = TR*TC
        tile_size = SR*SC
    elif op_type == 'applynode':
        if op_order == 'R':
            tile_times = TR
            tile_size = SR
        else:
            tile_times = TC
            tile_size = SC
    else:
        tile_times = -1
        tile_size = -1

    return tile_times, tile_size

def judge_comp_inst_tile(op_type, op_order, TR, TC, SR, SC):
    tile_times = 0
    tile_size = 0

    if op_type == 'scatter':
        tile_times = TR*TC #TODO: check TR or TR*TC
        tile_size = SR*SC
    elif op_type == 'gather' or op_type == 'applyedge':
        tile_times = TR*TC
        tile_size = SR*SC
    elif op_type == 'applynode':
        if op_order == 'R':
            tile_times = TR
            tile_size = SR
        else:
            tile_times = TC
            tile_size = SC
    else:
        tile_times = -1
        tile_size = -1

    return tile_times, tile_size

def judge_store_inst_tile(op_type, op_order, TR, TC, SR, SC):
    tile_times = 0
    tile_size = 0

    if op_type == 'scatter':
        tile_times = TR*TC
        tile_size = SR*SC
    elif op_type == 'gather' or op_type == 'applynode':
        if op_order == 'R':
            tile_times = TR
            tile_size = SR
        else:
            tile_times = TR*TC
            tile_size = SC
    elif op_type == 'applyedge':
        tile_times = TR*TC
        tile_size = SR*SC
    else:
        tile_times = -1
        tile_size = -1

    return tile_times, tile_size


def gen_comp_inst(op_info, op_id, TR, TC, SR, SC, w_size=0):

    op_type = op_info[op_id]['TYPE']
    compute_type = op_info[op_id]['COMP_TYPE']
    op_order = op_info[op_id]['ORDER']
    input_feature_size_list = op_info[op_id]['INPUT']['size_per_feature']

    inst_type = judge_comp_inst_type(op_type, compute_type)
    inst_ID = str(op_id)+'_'+op_type+'_'+str(0)
    hardware_unit = hardware_unit_mapping.get(compute_type)
    tile_times,tile_size = judge_comp_inst_tile(op_type, op_order, TR, TC, SR, SC)
    feature_length = input_feature_size_list[0] #无论何种compute，add或者mm，均为第一个输入的feature size

    comp_inst = {
        'TYPE': inst_type,
        'ID': inst_ID,
        'Hardware_Unit': hardware_unit,
        'Tile_Times': tile_times,
        'Tile_Size': tile_size,
        'Feature_Length': feature_length,
        'Weight_Size': w_size,
        'Dependency': {
            'RAW': [],
            'WAR': [],
        },
        'Enable': {
            'RAW': [],
            'WAR': [],
        } 
    }

    return comp_inst

def judge_dependency_times(op_type, op_order, output_op, TR, TC):

    #print(op_type, op_order, output_op, TR, TC)
    if op_type == 'scatter':
        if output_op == 'gather':
            return [1,1] #TODO: check
        elif output_op == 'applyedge':
            return [1,1]
    elif op_type == 'gather':
        #if output_op == 'scatter'  #不存在融合
        if output_op == 'applynode':
            if op_order == 'R':
                return [TC,1]
            else:
                return [TR,1]
    elif op_type == 'applyedge':
        if output_op == 'gather':
            return [1,1]
        elif output_op == 'applyedge':
            return [1,1]
    elif op_type == 'applynode':
        if output_op == 'scatter':
            if op_order == 'R':
                return [1,TC]
            else:
                return [1,TR]
        elif output_op == 'applynode':
            return [1,1]
    else:
        return [-1,-1]

#inst必须和op是同一个
#按照连接数序，op指向output_op
#例如A->B
#如果需要在A的dependency种加入war为B，则reverse=false
#如果需要在B的dependency种加入raw为A，则reverse=true
def add_dependency(inst,op,op_order,output_op,TR,TC,reverse=False):

    op_times = judge_dependency_times(op,op_order,output_op,TR,TC)[0]
    output_times = judge_dependency_times(op,op_order,output_op,TR,TC)[1]
    
    times = [op_times,output_times]

    if reverse:
        times = [output_times,op_times]

    return {'TYPE': inst.get('TYPE'),
            'ID': inst.get('ID'),
            'Times': times}
    
def gen_load_inst(op_info, op_id, input_num, load_type, TR, TC, SR, SC):

    op_type = op_info[op_id]['TYPE']
    compute_type = op_info[op_id]['COMP_TYPE']
    op_order = op_info[op_id]['ORDER']
    input_feature_size_list = op_info[op_id]['INPUT']['size_per_feature']
    output_feature_size = op_info[op_id]['OUTPUT']['size_per_feature']

    tile_times,tile_size = judge_load_inst_tile(op_type, op_order, compute_type, TR, TC, SR, SC)

    inst_type = load_type
    hardware_unit = hardware_unit_mapping.get('LOAD')
    if load_type == 'LOAD_W':
        feature_length = op_info[op_id]['INPUT']['input_size'][0]
        inst_ID = str(op_id)+'_'+op_type+'_'+str(op_info[op_id]['INPUT']['input_g_num'])
        tile_times = 1
        tile_size = 1
    elif load_type == 'LOAD_N' and op_type == 'gather':
        feature_length = output_feature_size
        inst_ID = str(op_id)+'_'+op_type+'_'+str(op_info[op_id]['INPUT']['input_g_num'])
        if op_order == 'R':
            hardware_unit = hardware_unit_mapping.get('LOAD_N')
            tile_times = TR
        else:
            tile_times = TC
    else:
        feature_length = input_feature_size_list[input_num]
        inst_ID = str(op_id)+'_'+op_type+'_'+str(input_num)

    load_inst = {
        'TYPE': inst_type,
        'ID': inst_ID,
        'Hardware_Unit': hardware_unit,
        'Tile_Times': tile_times,
        'Tile_Size': tile_size,
        'Feature_Length': feature_length,
        'Dependency': {
            'RAW': [],
            'WAR': [],
        },
        'Enable': {
            'RAW': [],
            'WAR': [],
        } 
    }

    return load_inst

def gen_store_inst(op_info, op_id, TR, TC, SR, SC):

    op_type = op_info[op_id]['TYPE']
    op_order = op_info[op_id]['ORDER']
    output_feature_size = op_info[op_id]['OUTPUT']['size_per_feature']

    data_type = ''
    if op_type == 'scatter' or op_type == 'applyedge':
        data_type = 'E'
    else:
        data_type = 'N'
        
    inst_type = 'STORE_'+data_type
    inst_ID = str(op_id)+'_'+op_type+'_'+str(0)
    hardware_unit = hardware_unit_mapping.get('STORE')
    tile_times,tile_size = judge_store_inst_tile(op_type, op_order, TR, TC, SR, SC)
    feature_length = output_feature_size

    store_inst = {
        'TYPE': inst_type,
        'ID': inst_ID,
        'Hardware_Unit': hardware_unit,
        'Tile_Times': tile_times,
        'Tile_Size': tile_size,
        'Feature_Length': feature_length,
        'Dependency': {
            'RAW': [],
            'WAR': [],
        },
        'Enable': {
            'RAW': [],
            'WAR': [],
        } 
    }

    return store_inst

def judge_load_inst_type(op_type, op_comp_type):
    load_inst_type_list = []
    if op_type == 'scatter':
        load_inst_type_list = ['LOAD_N']
    elif op_type == 'gather':
        load_inst_type_list = ['LOAD_E','LOAD_N']
    elif op_type == 'applyedge' or op_type == 'applynode' :
        if op_comp_type == 'MM':
            load_inst_type_list = ['LOAD_'+op_type[5].upper(),'LOAD_W']
        else:
            load_inst_type_list = ['LOAD_'+op_type[5].upper(),'LOAD_'+op_type[5].upper()]
    return load_inst_type_list

def gen_inst(op_info, op_id, fused_array, TR, TC, SR, SC):

    load_inst = []
    comp_inst = gen_comp_inst(op_info, op_id, TR, TC, SR, SC)
    store_inst = []

    op_type = op_info[op_id]['TYPE']
    op_order = op_info[op_id]['ORDER']
    op_comp_type = op_info[op_id]['COMP_TYPE']
    input_list = op_info[op_id]['INPUT']['input_g_list']
    input_num = op_info[op_id]['INPUT']['input_g_num']
    output_list = op_info[op_id]['OUTPUT']['output_list']

    #先将自带的输入load进来，即gather的LOAD_N和MM的LOAD_W
    outside_type = ''
    outside_times = []
    if op_type == 'gather':
        outside_type = 'LOAD_N'
        if op_order == 'R':
            outside_times = [1,TC]
        else:
            outside_times = [1,1]
    elif op_type == 'applyedge' and op_comp_type == 'MM':
        outside_type = 'LOAD_W'
        outside_times = [1,TR*TC]
    elif op_type == 'applynode' and op_comp_type == 'MM':
        outside_type = 'LOAD_W'
        if op_order == 'R':
            outside_times = [1,TR]
        else:
            outside_times = [1,TC]
    elif op_type == 'applyedge' and len(input_list) != input_num:
        outside_type = 'LOAD_E'
        outside_times = [1,1]
    elif op_type == 'applynode' and len(input_list) != input_num:
        outside_type = 'LOAD_N'
        if op_order == 'R':
            outside_times = [1,TR]
        else:
            outside_times = [1,TC]

    if outside_type != '':
        load_inst.append(gen_load_inst(op_info, op_id, input_num-1, outside_type, TR, TC, SR, SC))
        load_inst[-1]['Dependency']['WAR'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':outside_times})
        load_inst[-1]['Enable']['RAW'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':outside_times})
        if outside_type == 'LOAD_W':
            comp_inst['Weight_Size'] = load_inst[-1]['Feature_Length']
        comp_inst['Dependency']['RAW'].append({'TYPE':load_inst[-1].get('TYPE'),'ID':load_inst[-1].get('ID'),'Times':[outside_times[1],outside_times[0]]})
        comp_inst['Enable']['WAR'].append({'TYPE':load_inst[-1].get('TYPE'),'ID':load_inst[-1].get('ID'),'Times':[outside_times[1],outside_times[0]]})

    #外部的输入LOAD
    if input_list == []:
        load_inst_type_list = []
        load_inst_times = []
        if op_type == 'scatter':
            load_inst_type_list = ['LOAD_N']
            if op_order == 'R':
                load_inst_times = [1,TC]
            else:
                load_inst_times = [1,1]
        elif op_type == 'gather':
            load_inst_type_list = ['LOAD_E']
            load_inst_times = [1,1]
        elif op_type == 'applyedge' or op_type == 'applynode' :
            load_inst_times = [1,1]
            if op_comp_type == 'MM':
                load_inst_type_list = ['LOAD_'+op_type[5].upper()]
            else:
                for i in range(0,len(input_list)):
                    load_inst_type_list.append('LOAD_'+op_type[5].upper())
        
        for current_load_num in range(0,len(load_inst_type_list)):
            current_load_type = load_inst_type_list[current_load_num]
            #自己的load append 自己的compute
            load_inst.append(gen_load_inst(op_info, op_id, current_load_num, current_load_type, TR, TC, SR, SC))
            load_inst[-1]['Dependency']['WAR'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':load_inst_times})
            load_inst[-1]['Enable']['RAW'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':load_inst_times})
            
            #compute append是自己算子的load，如何判断次数
            comp_inst['Dependency']['RAW'].append({'TYPE':load_inst[-1].get('TYPE'),'ID':load_inst[-1].get('ID'),'Times':[load_inst_times[1],load_inst_times[0]]})
            comp_inst['Enable']['WAR'].append({'TYPE':load_inst[-1].get('TYPE'),'ID':load_inst[-1].get('ID'),'Times':[load_inst_times[1],load_inst_times[0]]})
    else:
        for current_input_num in range(0,len(input_list)):

            current_input_op = input_list[current_input_num]

            if current_input_op in fused_array:
                #compute append别人算子的append
                comp_inst['Dependency']['RAW'].append(add_dependency(gen_comp_inst(op_info,current_input_op,TR,TC,SR,SC),op_info[current_input_op]['TYPE'],op_info[current_input_op]['ORDER'],op_type,TR,TC,reverse=True))
                comp_inst['Enable']['WAR'].append(add_dependency(gen_comp_inst(op_info,current_input_op,TR,TC,SR,SC),op_info[current_input_op]['TYPE'],op_info[current_input_op]['ORDER'],op_type,TR,TC,reverse=True))
            else:
                load_inst_type_list = []
                load_inst_times = []
                if op_type == 'scatter':
                    load_inst_type_list = ['LOAD_N']
                    if op_order == 'R':
                        load_inst_times = [1,TC]
                    else:
                        load_inst_times = [1,1]
                elif op_type == 'gather':
                    load_inst_type_list = ['LOAD_E']
                    load_inst_times = [1,1]
                elif op_type == 'applyedge' or op_type == 'applynode' :
                    load_inst_times = [1,1]
                    if op_comp_type == 'MM':
                        load_inst_type_list = ['LOAD_'+op_type[5].upper()]
                    else:
                        for i in range(0,len(input_list)):
                            load_inst_type_list.append('LOAD_'+op_type[5].upper())
                
                #自己的load append 自己的compute
                load_inst.append(gen_load_inst(op_info, op_id, current_input_num, load_inst_type_list[current_input_num], TR, TC, SR, SC))
                load_inst[-1]['Dependency']['WAR'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':load_inst_times})
                load_inst[-1]['Enable']['RAW'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':load_inst_times})
                
                #compute append是自己算子的load，如何判断次数
                comp_inst['Dependency']['RAW'].append({'TYPE':load_inst[-1].get('TYPE'),'ID':load_inst[-1].get('ID'),'Times':[load_inst_times[1],load_inst_times[0]]})
                comp_inst['Enable']['WAR'].append({'TYPE':load_inst[-1].get('TYPE'),'ID':load_inst[-1].get('ID'),'Times':[load_inst_times[1],load_inst_times[0]]})
    
    #外部的输出STORE
    if output_list == []:
        store_times = []
        if op_type == 'gather':
            if op_order == 'R':
                store_times = [1,TC] #store一次需要TC次compute
            else:
                store_times = [1,1]
        else:
            store_times = [1,1]
        #自己的store append 自己算子的compute
        store_inst = gen_store_inst(op_info, op_id, TR, TC, SR, SC)
        store_inst['Dependency']['RAW'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':store_times})
        store_inst['Enable']['WAR'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':store_times})

        #自己的compute append 自己算子的store
        comp_inst['Dependency']['WAR'].append({'TYPE':store_inst.get('TYPE'),'ID':store_inst.get('ID'),'Times':[store_times[1],store_times[0]]})
        comp_inst['Enable']['RAW'].append({'TYPE':store_inst.get('TYPE'),'ID':store_inst.get('ID'),'Times':[store_times[1],store_times[0]]})
    else:
        isStored = False
        for current_output_num in range(0,len(output_list)):
            current_output_op = output_list[current_output_num]
            
            if current_output_op in fused_array:
                #自己的compute append别人算子的compute
                comp_inst['Dependency']['WAR'].append(add_dependency(gen_comp_inst(op_info,current_output_op,TR,TC,SR,SC),op_type,op_order,op_info[current_output_op]['TYPE'],TR,TC))
                comp_inst['Enable']['RAW'].append(add_dependency(gen_comp_inst(op_info,current_output_op,TR,TC,SR,SC),op_type,op_order,op_info[current_output_op]['TYPE'],TR,TC))
            else:
                if not isStored:
                    store_times = []
                    if op_type == 'gather':
                        if op_order == 'R':
                            store_times = [1,TC] #store一次需要TC次compute
                        else:
                            store_times = [1,1]
                    else:
                        store_times = [1,1]
                    #自己的store append 自己算子的compute
                    store_inst = gen_store_inst(op_info, op_id, TR, TC, SR, SC)
                    store_inst['Dependency']['RAW'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':store_times})
                    store_inst['Enable']['WAR'].append({'TYPE':comp_inst.get('TYPE'),'ID':comp_inst.get('ID'),'Times':store_times})

                    #自己的compute append 自己算子的store
                    comp_inst['Dependency']['WAR'].append({'TYPE':store_inst.get('TYPE'),'ID':store_inst.get('ID'),'Times':[store_times[1],store_times[0]]})
                    comp_inst['Enable']['RAW'].append({'TYPE':store_inst.get('TYPE'),'ID':store_inst.get('ID'),'Times':[store_times[1],store_times[0]]})
                    isStored = True

    return [load_inst, comp_inst, store_inst]

def find_inst(full_inst_list, type, id):
    for i in range(0,len(full_inst_list)):
        for j in range(0,len(full_inst_list[i])):
            if full_inst_list[i][j]['TYPE'] == type and full_inst_list[i][j]['ID'] == id:
                return j

def gen_link(full_inst_list):
    next_inst_list = []
    for i in range(0,len(full_inst_list)):
        next_inst_list.append([])
        for j in range(0,len(full_inst_list[i])):
            next_inst_list[-1].append([])
            for k in range(0,len(full_inst_list[i][j]['Dependency']['WAR'])):
                next_inst_list[-1][-1].append(find_inst(full_inst_list,full_inst_list[i][j]['Dependency']['WAR'][k]['TYPE'],full_inst_list[i][j]['Dependency']['WAR'][k]['ID']))
    return next_inst_list

def extract_between_underscores(input_string):

    # 使用 split 方法分割字符串
    parts = input_string.split('_')
    
    # 提取下划线之间的内容
    if len(parts) > 1:
        return parts[1]
    else:
        return None

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

def judge_inst_pattern(pattern, comp_type, inst_fused_dict):
    key = (tuple(pattern), tuple(comp_type))
    if key in inst_fused_dict:
        if inst_fused_dict[key]['Is_Fused']:
            return True
    return False

def judge_inst_fused_pattern(first_inst, last_inst,hardware_info):
    first_type = extract_between_underscores(first_inst['ID'])
    last_type = extract_between_underscores(last_inst['ID'])
    first_comp_type = extract_between_underscores(first_inst['TYPE'])
    last_comp_type =  extract_between_underscores(last_inst['TYPE'])
    if first_inst['TYPE'].split('_')[0] == 'COMP' and last_inst['TYPE'].split('_')[0] == 'COMP':
        return judge_inst_pattern([first_type,last_type],[first_comp_type,last_comp_type],hardware_info)
    return False


def gen_inst_fused_list(link_list, full_inst_list, hardware_info):
    
    fused_list = []
    
    for fused_inst_list_num in range(0,len(full_inst_list)):
        fused_inst_list = full_inst_list[fused_inst_list_num]
        fused_link_list = link_list[fused_inst_list_num]
        to_be_decided_list = list(range(len(fused_inst_list)))
        fused_list.append([])
        current_inst_num = 0
        while to_be_decided_list != []:
            current_inst = fused_inst_list[current_inst_num]
            if current_inst['TYPE'].split('_')[0] != 'COMP':
                to_be_decided_list.remove(current_inst_num)
            else:
                if len(fused_link_list[current_inst_num]) != 1:
                    to_be_decided_list.remove(current_inst_num)
                else:
                    next_inst_num = fused_link_list[current_inst_num][0]
                    next_inst = fused_inst_list[next_inst_num]
                    if judge_inst_fused_pattern(current_inst,next_inst,hardware_info):
                        fused_list[-1].append([current_inst_num,next_inst_num])
                        to_be_decided_list.remove(current_inst_num)
                        if len(fused_link_list[next_inst_num]) == 1:
                            third_inst_num = fused_link_list[next_inst_num][0]
                            third_inst = fused_inst_list[third_inst_num]
                            if judge_inst_fused_pattern(next_inst,third_inst,hardware_info):
                                fused_list[-1][-1].append(third_inst)
                                to_be_decided_list.remove(next_inst_num)
                    else:
                        to_be_decided_list.remove(current_inst_num)
            current_inst_num += 1

    return fused_list

def inst_fusion_x2(full_inst_list,inst_1,inst_2,block_num):

    inst_1_type = full_inst_list[block_num][inst_1]['TYPE']
    inst_2_type = full_inst_list[block_num][inst_2]['TYPE']
    inst_1_id = full_inst_list[block_num][inst_1]['ID']
    inst_2_id = full_inst_list[block_num][inst_2]['ID']

    inst_1_raw = full_inst_list[block_num][inst_1]['Dependency']['RAW']
    inst_2_raw = full_inst_list[block_num][inst_2]['Dependency']['RAW']
    inst_1_war = full_inst_list[block_num][inst_1]['Dependency']['WAR']
    inst_2_war = full_inst_list[block_num][inst_2]['Dependency']['WAR']

    raw = []
    for i in range(0,len(inst_1_raw)):
        currrent_raw = inst_1_raw[i]
        raw.append(currrent_raw)
    for i in range(0,len(inst_2_raw)):
        currrent_raw = inst_2_raw[i]
        if currrent_raw['TYPE'] == inst_1_type and currrent_raw['ID'] == inst_1_id:
            continue
        raw.append(currrent_raw)

    war = []
    for i in range(0,len(inst_1_war)):
        currrent_war = inst_1_war[i]
        if currrent_war['TYPE'] == inst_2_type and currrent_war['ID'] == inst_2_id:
            continue
        war.append(currrent_war)
    for i in range(0,len(inst_2_war)):
        currrent_war = inst_2_war[i]
        war.append(currrent_war)
    

    inst_type = inst_1_type+'_'+inst_2_type
    inst_ID = inst_1_id+'_'+inst_2_id

    hardware_unit = 'VEC_ALU'
    if(inst_1_type == 'COMP_MM' or inst_2_type == 'COMP_MM'):
        hardware_unit = 'MM'

    tile_times = full_inst_list[block_num][inst_1]['Tile_Times']
    tile_size = full_inst_list[block_num][inst_1]['Tile_Size']
    feature_length = full_inst_list[block_num][inst_1]['Feature_Length']

    fused_inst = {
        'TYPE': inst_type,
        'ID': inst_ID,
        'Hardware_Unit': hardware_unit,
        'Tile_Times': tile_times,
        'Tile_Size': tile_size,
        'Feature_Length': feature_length,
        'Dependency': {
            'RAW': raw,
            'WAR': war,
        },
        'Enable': {
            'RAW': war,
            'WAR': raw,
        } 
    }

    update_fused_dependency(full_inst_list[block_num],fused_inst)

    return fused_inst

def inst_fusion_x3(full_inst_list,inst_1,inst_2,inst_3,block_num):

    inst_1_type = full_inst_list[block_num][inst_1]['TYPE']
    inst_2_type = full_inst_list[block_num][inst_2]['TYPE']
    inst_3_type = full_inst_list[block_num][inst_3]['TYPE']
    inst_1_id = full_inst_list[block_num][inst_1]['ID']
    inst_2_id = full_inst_list[block_num][inst_2]['ID']
    inst_3_id = full_inst_list[block_num][inst_3]['ID']


    inst_1_raw = full_inst_list[block_num][inst_1]['Dependency']['RAW']
    inst_2_raw = full_inst_list[block_num][inst_2]['Dependency']['RAW']
    inst_3_raw = full_inst_list[block_num][inst_3]['Dependency']['RAW']
    inst_1_war = full_inst_list[block_num][inst_1]['Dependency']['WAR']
    inst_2_war = full_inst_list[block_num][inst_2]['Dependency']['WAR']
    inst_3_war = full_inst_list[block_num][inst_3]['Dependency']['WAR']

    raw = []
    for i in range(0,len(inst_1_raw)):
        currrent_raw = inst_1_raw[i]
        raw.append(currrent_raw)
    for i in range(0,len(inst_2_raw)):
        currrent_raw = inst_2_raw[i]
        if currrent_raw['TYPE'] == inst_1_type and currrent_raw['ID'] == inst_1_id:
            continue
        raw.append(currrent_raw)
    for i in range(0,len(inst_3_raw)):
        currrent_raw = inst_3_raw[i]
        if currrent_raw['TYPE'] == inst_2_type and currrent_raw['ID'] == inst_2_id:
            continue
        raw.append(currrent_raw)
    

    war = []
    for i in range(0,len(inst_1_war)):
        currrent_war = inst_1_war[i]
        if currrent_war['TYPE'] == inst_2_type and currrent_war['ID'] == inst_2_id:
            continue
        war.append(currrent_war)
    for i in range(0,len(inst_2_war)):
        currrent_war = inst_2_war[i]
        if currrent_war['TYPE'] == inst_3_type and currrent_war['ID'] == inst_3_id:
            continue
        war.append(currrent_war)
    for i in range(0,len(inst_3_war)):
        currrent_war = inst_3_war[i]
        war.append(currrent_war)
    

    inst_type = inst_1_type+'_'+inst_2_type+'_'+inst_3_type
    inst_ID = inst_1_id+'_'+inst_2_id+'_'+inst_3_id

    hardware_unit = 'VEC_ALU'
    if(inst_1_type == 'COMP_MM' or inst_2_type == 'COMP_MM' or inst_3_type == 'COMP_MM'):
        hardware_unit = 'MM'

    tile_times = full_inst_list[block_num][inst_1]['Tile_Times']
    tile_size = full_inst_list[block_num][inst_1]['Tile_Size']
    feature_length = full_inst_list[block_num][inst_1]['Feature_Length']

    fused_inst = {
        'TYPE': inst_type,
        'ID': inst_ID,
        'Hardware_Unit': hardware_unit,
        'Tile_Times': tile_times,
        'Tile_Size': tile_size,
        'Feature_Length': feature_length,
        'Dependency': {
            'RAW': raw,
            'WAR': war,
        },
        'Enable': {
            'RAW': war,
            'WAR': raw,
        } 
    }

    update_fused_dependency(full_inst_list[block_num],fused_inst)

    return fused_inst

def update_fused_dependency(full_inst_list,fused_inst):
    for j in full_inst_list:
        for k in j['Dependency']['RAW']:
            if k['ID'] in fused_inst['ID'] and k['TYPE'] in fused_inst['TYPE']:
                k['ID'] = fused_inst['ID']
                k['TYPE'] = fused_inst['TYPE']
        for k in j['Dependency']['WAR']:
            if k['ID'] in fused_inst['ID'] and k['TYPE'] in fused_inst['TYPE']:
                k['ID'] = fused_inst['ID']
                k['TYPE'] = fused_inst['TYPE']
        for k in j['Enable']['RAW']:
            if k['ID'] in fused_inst['ID'] and k['TYPE'] in fused_inst['TYPE']:
                k['ID'] = fused_inst['ID']
                k['TYPE'] = fused_inst['TYPE']
        for k in j['Enable']['WAR']:
            if k['ID'] in fused_inst['ID'] and k['TYPE'] in fused_inst['TYPE']:
                k['ID'] = fused_inst['ID']
                k['TYPE'] = fused_inst['TYPE']
                    

def update_inst(full_inst_list,fused_inst_list):

    temp_inst_list = []

    #[[],[],[[5,7]]]
    for i in range(0,len(fused_inst_list)): #i = [[5,7]] 
        temp_inst_list.append([]) #[[],[],[]]
        if fused_inst_list[i] != []:
            for j in fused_inst_list[i]: # j = [5,7]
                if len(j) == 2:
                    temp_inst_list[-1].append(inst_fusion_x2(full_inst_list,j[0],j[1],i))
                elif len(j) == 3:
                    temp_inst_list[-1].append(inst_fusion_x3(full_inst_list,j[0],j[1],j[2],i))

    for i in range(0,len(full_inst_list)):
        if temp_inst_list[i] != []:
            for j in range(len(fused_inst_list[i]) - 1, -1, -1):
                for k in sorted(fused_inst_list[i][j], reverse=True):
                    del full_inst_list[i][k]
                full_inst_list[i].append(temp_inst_list[i][j])

def find_fetch_index(current_inst,fetch_inst_dependency):
    for i in range(0,len(fetch_inst_dependency)):
        if current_inst['ID'] == fetch_inst_dependency[i]['ID'] and current_inst['TYPE'] == fetch_inst_dependency[i]['TYPE']:
            return i


def fuse_fetch(full_inst_list):
    record = []
    #寻找fetch
    for i in range(0,len(full_inst_list)):
        for j in range(0,len(full_inst_list[i])):
            if full_inst_list[i][j]['TYPE'] == 'FETCH':
                record.append([i,j])
                raw = full_inst_list[i][j]['Dependency']['RAW']
                war = full_inst_list[i][j]['Dependency']['WAR']
                fetch_id = full_inst_list[i][j]['ID']
                #更新fetch有关的dependency
                for n in range(0,len(full_inst_list[i])):
                    if full_inst_list[i][n]['Dependency']['RAW'] != []:
                        for current_raw_num in range(0,len(full_inst_list[i][n]['Dependency']['RAW'])):
                            current_raw = full_inst_list[i][n]['Dependency']['RAW'][current_raw_num]
                            current_enable_war = full_inst_list[i][n]['Enable']['WAR'][current_raw_num]
                            if current_raw['ID'] == fetch_id and current_raw['TYPE'] == 'FETCH':
                                index = find_fetch_index(full_inst_list[i][n],war)
                                current_raw['ID'] = raw[index]['ID']
                                current_raw['TYPE'] = raw[index]['TYPE']
                                current_raw['Times'] = raw[index]['Times']
                                current_enable_war['ID'] = raw[index]['ID']
                                current_enable_war['TYPE'] = raw[index]['TYPE']
                                current_enable_war['Times'] = raw[index]['Times']

                    if full_inst_list[i][n]['Dependency']['WAR'] != []:
                        for current_war_num in range(0,len(full_inst_list[i][n]['Dependency']['WAR'])):
                            current_war = full_inst_list[i][n]['Dependency']['WAR'][current_war_num]
                            current_enable_raw = full_inst_list[i][n]['Enable']['RAW'][current_war_num]
                            if current_war['ID'] == fetch_id and current_war['TYPE'] == 'FETCH':
                                index = find_fetch_index(full_inst_list[i][n],raw)
                                current_war['ID'] = war[index]['ID']
                                current_war['TYPE'] = war[index]['TYPE']
                                current_enable_raw['ID'] = war[index]['ID']
                                current_enable_raw['TYPE'] = war[index]['TYPE']
                    
    # 删除 fetch
    for i, j in sorted(record, reverse=True):
        del full_inst_list[i][j]

#if __name__ == '__main__':
def interpret(data_set, network, isReorder, layer, op_array, tile_size_list):

    node_num = 0
    if data_set == 'cora':
        node_num = 2708
    elif data_set == 'pubmed':
        node_num = 19717
    elif data_set == 'flickr':
        node_num = 89250
    elif data_set == 'reddit':
        node_num = 232965

    op_map = 'original'
    if isReorder:
        op_map = 'trans'

    op_info = read('Network/'+network+'/'+network+'-'+data_set+'/'+network+'-'+op_map+'/'+network+'-'+layer+'-'+op_map+'.yaml')
    hardware_info = load_inst_fused('code/hardware_info.yaml')
    inst_name = network+'-'+data_set+'-'+layer+'-'+op_map+'.yaml'
    
    full_inst_list = []
    for current_block_num in range(0,len(op_array)):

        TR = math.ceil(node_num/tile_size_list[current_block_num][0])
        TC = math.ceil(node_num/tile_size_list[current_block_num][1])
        full_inst_list.append([])

        for current_op in op_array[current_block_num]:
            res = gen_inst(op_info, current_op, op_array[current_block_num],TR, TC, tile_size_list[current_block_num][0], tile_size_list[current_block_num][1])
            if res[0] != []:
                for load_inst in res[0]:
                    full_inst_list[-1].append(load_inst)
            full_inst_list[-1].append(res[1])
            if res[2] != []:
                full_inst_list[-1].append(res[2])

    link_list = gen_link(full_inst_list)

    fused_inst_list = gen_inst_fused_list(link_list, full_inst_list, hardware_info)

    update_inst(full_inst_list,fused_inst_list)

    fuse_fetch(full_inst_list)

    write(full_inst_list,inst_name)


if __name__ == '__main__':
    data_set = 'cora'
    network = 'GAT'
    isReorder = False
    layer = 'layer1'
    op_array = [[0], [2], [3], [5], [6], [7], [8], [9], [10], [11], [12], [13], [1, 4]]
    tile_size_list = [[96, 1], [1808, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [2720, 1], [1872, 1], [1008, 1], [1616, 1]]
    interpret(data_set, network, isReorder, layer, op_array, tile_size_list)