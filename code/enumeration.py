import tqdm
import csv
import copy

from compiler import *
from interpreter import *
from simulator import *

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

def check_csv_for_sample(filename, pattern, tsize):
    # 检查文件是否存在
    if not os.path.exists(filename):
        return None

    # 打开 CSV 文件
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # 遍历 CSV 文件中的每一行
        for row in reader:
            # 检查 pattern 是否匹配
            if row['pattern'] == pattern:
                # 将 tsize 从字符串转换为列表
                tsize_from_csv = eval(row['tsize'])
                
                # 检查 tsize 是否匹配
                if tsize_from_csv == tsize:
                    # 返回 latency 和 mem_access
                    return int(row['latency']), int(row['mem_access'])
    
    # 如果没有匹配的样本，返回 None
    return None

def save_sample_to_csv(sample, filename):
    # 定义 CSV 文件的列名
    fieldnames = ['pattern', 'fusion', 'tsize', 'pingpong', 'latency', 'mem_access']
    
    # 检查文件是否存在
    file_exists = os.path.isfile(filename)
    
    # 以追加模式打开 CSV 文件，如果文件不存在则创建
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件不存在，写入列名
        if not file_exists:
            writer.writeheader()
        
        # 写入 sample 数据
        writer.writerow(sample)

def eval_sim(data_set, network, isReorder, isSinput, layer, sample):
    op_array = sample['fusion']
    tile_size_list = sample['tsize']
    pattern = sample['pattern']
    filename = network + '_' + data_set + '_' + layer + '.csv'
    
    result = check_csv_for_sample(filename, pattern, tile_size_list)
    if result:
        sample['latency'], sample['mem_access'] = result
    else:
        interpret(data_set, network, isReorder, layer, op_array, tile_size_list)
        res = simulate(tile_size_list, data_set, network, layer, isReorder, isSinput)
        print(res)
        sample['latency'] = res[0]
        sample['mem_access'] = res[1]
        print(sample['mem_access'])
        save_sample_to_csv(sample, filename)

def res_to_sample(res, isPingpang):
    return {'fusion': res[0], 'tsize': res[1], 'pingpong': isPingpang, 'latency': 0, 'mem_access': res[2], 'pattern': res[3]}

def enumerate_search(data_set, network, isReorder, isSinput, layer, compiler_res, isPingpang):
    best_sample = eval_sim(data_set, network, isReorder, isSinput, layer, res_to_sample(compiler_res[0], isPingpang))
    best_mem_access = compiler_res[0][2]
    for one_res in tqdm(compiler_res):
        if one_res[2] < best_mem_access*1.5:
            for size in [0.25, 0.5, 0.75, 1]:
                tile_size = copy.deepcopy(one_res[1])
                for j in tile_size:
                    j[0] = math.ceil(math.ceil(j[0]*size)/16)*16
                    sample = res_to_sample(one_res, isPingpang)
                    sample['tsize'] = tile_size
                    eval_sim(data_set, network, isReorder, isSinput, layer, sample)
            if sample['latency'] < best_sample['latency']:
                best_sample = sample
    return best_sample


def genetic_compile(dataset_name,network_name,layer_name,isReorder,isSinput,isPingpang):

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
            'node_num': node_num,
            'breakpoint': breakpoint
        }

    res = enumerate_search(dataset_name, network_name, isReorder, isSinput, layer_name, compiler_res,isPingpang)

    return res

if __name__ == '__main__':
    inst_fused_dict = load_inst_fused('hardware_info.yaml')

    res = genetic_compile('cora','GAT','layer1',False, False, True)
    print(res)