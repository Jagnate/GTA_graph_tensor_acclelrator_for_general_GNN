import yaml
import os

def gen_one_op(op_no, comp_type, type, order, feature_number, input_g_list, input_g_num, input_nong_num, input_nong_list, input_size, input_size_per_feature, output_list, output_number, output_size_per_feature):
    op = {
        "OP_NO": op_no,
        "COMP_TYPE": comp_type,
        "TYPE": type,
        "ORDER": order,
        "INPUT": {
            "input_g_list": input_g_list,
            "input_g_num": input_g_num,
            "input_nong_num": input_nong_num,
            "input_nong_list": input_nong_list,
            "input_size": input_size,
            "feature_number": feature_number,
            "size_per_feature": input_size_per_feature
        },
        "OUTPUT": {
            "output_list": output_list,
            "output_number": output_number,
            "size_per_feature": output_size_per_feature
        }
    }
    return op

def gen_yaml(path,node_num,edge_num,size_per_feature, network, layer, isReorder):
    
    data = []

    size_per_feature_list = [0, size_per_feature, 128, 64, 16]
    weight_size = [0, 128, 64, 16]

    if network == 'GCN' and not isReorder:
        data.append(gen_one_op(0, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [1], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(1, "MUL", "applyedge", "R", [edge_num, edge_num], [0, -1], 2, 0, [], [], [size_per_feature_list[layer]*4, size_per_feature_list[layer]*4], [2], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(2, "ADD", "gather", "R", [edge_num], [1], 1, 0, [], [], [size_per_feature_list[layer]*4], [3], node_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(3, "MM", "applynode", "R", [node_num], [2], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [], node_num, weight_size[layer]*4))
    
    elif network == 'GCN' and isReorder:
        data.append(gen_one_op(0, "MM", "applynode", "R", [node_num], [], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [1], node_num, weight_size[layer]*4))
        data.append(gen_one_op(1, "NONE", "scatter", "C", [node_num], [0], 1, 0, [], [], [weight_size[layer]*4], [1], edge_num, weight_size[layer]*4))
        #TODO: 两个feature一样？
        data.append(gen_one_op(2, "MUL", "applyedge", "R", [edge_num], [1, -1], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [2], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(3, "ADD", "gather", "R", [edge_num], [2], 1, 0, [], [], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))
        
    elif network == 'GAT' and not isReorder:
        #op_no, comp_type, type, order, feature_number, input_g_list, input_g_num, input_nong_num, input_nong_list, input_size, size_per_feature, output_list, output_number
        data.append(gen_one_op(0, "MM", "applynode", "R", [node_num], [], 1, 1, [], [weight_size[layer]*size_per_feature_list[layer]*4], [size_per_feature_list[layer]*4], [1, 2, 3], node_num, weight_size[layer]*4))
        data.append(gen_one_op(1, "MM", "applynode", "R", [node_num], [0], 1, 1, [], [weight_size[layer]*weight_size[3]*4], [weight_size[layer]*4], [4], node_num, weight_size[3]*4))
        data.append(gen_one_op(1, "MM", "applynode", "R", [node_num], [0], 1, 1, [], [weight_size[layer]*weight_size[3]*4], [weight_size[layer]*4], [5], node_num, weight_size[3]*4))
        data.append(gen_one_op(3, "NONE", "scatter", "C", [node_num], [0], 1, 0, [], [], [weight_size[layer]*4], [11], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(4, "NONE", "scatter", "R", [node_num], [1], 1, 0, [], [], [weight_size[3]*4], [6], edge_num, weight_size[3]*4))
        data.append(gen_one_op(5, "NONE", "scatter", "C", [node_num], [2], 1, 0, [], [], [weight_size[3]*4], [6], edge_num, weight_size[3]*4))
        data.append(gen_one_op(6, "ADD", "applyedge", "R", [edge_num, edge_num], [4, 5], 2, 0, [], [], [weight_size[3]*4, weight_size[3]*4], [7], edge_num, weight_size[3]*4))
        data.append(gen_one_op(7, "SF", "applyedge", "R", [edge_num], [6], 1, 0, [], [], [weight_size[3]*4], [8, 9], edge_num, weight_size[3]*4))
        data.append(gen_one_op(8, "ADD", "gather", "R", [edge_num], [7], 1, 0, [], [], [weight_size[3]*4], [10], node_num, weight_size[3]*4))
        data.append(gen_one_op(9, "MUL", "applyedge", "R", [edge_num, edge_num], [7, 10], 2, 0, [], [], [weight_size[3]*4, weight_size[3]*4], [11], edge_num, weight_size[3]*4))
        data.append(gen_one_op(10, "NONE", "scatter", "R", [node_num], [7], 1, 0, [], [], [weight_size[3]*4], [9], edge_num, weight_size[3]*4))
        data.append(gen_one_op(11, "MUL", "applyedge", "R", [edge_num, edge_num], [3, 9], 2, 0, [], [], [weight_size[layer]*4, weight_size[3]*4], [12], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(12, "ADD", "gather", "R", [edge_num], [11], 1, 0, [], [], [weight_size[layer]*4], [13], node_num, weight_size[layer]*4))
        data.append(gen_one_op(13, "SF", "applynode", "R", [node_num], [12], 1, 0, [], [], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))
        
    elif network == 'GAT' and isReorder:
        data.append(gen_one_op(0, "MM", "applynode", "R", [node_num], [], 1, 1, [], [weight_size[layer]*size_per_feature_list[layer]*4], [size_per_feature_list[layer]*4], [1, 2, 3], node_num, weight_size[layer]*4))
        data.append(gen_one_op(1, "MM", "applynode", "R", [node_num], [0], 1, 1, [], [weight_size[layer]*weight_size[3]*4], [weight_size[layer]*4], [4], node_num, weight_size[3]*4))
        data.append(gen_one_op(1, "MM", "applynode", "R", [node_num], [0], 1, 1, [], [weight_size[layer]*weight_size[3]*4], [weight_size[layer]*4], [5], node_num, weight_size[3]*4))
        data.append(gen_one_op(3, "NONE", "scatter", "C", [node_num], [0], 1, 0, [], [], [weight_size[layer]*4], [11], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(4, "NONE", "scatter", "R", [node_num], [1], 1, 0, [], [], [weight_size[3]*4], [6], edge_num, weight_size[3]*4))
        data.append(gen_one_op(5, "NONE", "scatter", "C", [node_num], [2], 1, 0, [], [], [weight_size[3]*4], [6], edge_num, weight_size[3]*4))
        data.append(gen_one_op(6, "ADD", "applyedge", "R", [edge_num, edge_num], [4, 5], 2, 0, [], [], [weight_size[3]*4, weight_size[3]*4], [7], edge_num, weight_size[3]*4))
        data.append(gen_one_op(7, "MUL", "applyedge", "R", [edge_num, edge_num], [3, 8], 2, 0, [], [], [weight_size[layer]*4, weight_size[3]*4], [10], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(8, "SF", "applyedge", "R", [edge_num], [6], 1, 0, [], [], [weight_size[3]*4], [9], edge_num, weight_size[3]*4))
        data.append(gen_one_op(9, "ADD", "gather", "R", [edge_num], [8], 1, 0, [], [], [weight_size[3]*4], [11], node_num, weight_size[3]*4))
        data.append(gen_one_op(10, "ADD", "gather", "R", [edge_num], [7], 1, 0, [], [], [weight_size[layer]*4], [11], node_num, weight_size[layer]*4))
        data.append(gen_one_op(11, "MUL", "applynode", "R", [node_num, node_num], [9, 10], 2, 0, [], [], [weight_size[3]*4, weight_size[layer]*4], [12], node_num, weight_size[layer]*4))
        data.append(gen_one_op(12, "SF", "applynode", "R", [node_num], [11], 1, 0, [], [], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))

    elif network == 'SGC':
        data.append(gen_one_op(0, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [1], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(1, "MUL", "applyedge", "R", [edge_num, edge_num], [0, -1], 2, 0, [], [], [size_per_feature_list[layer]*4, size_per_feature_list[layer]*4], [2], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(2, "ADD", "gather", "R", [edge_num], [1], 1, 0, [], [], [size_per_feature_list[layer]*4], [3], node_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(3, "NONE", "scatter", "C", [node_num], [2], 1, 0, [], [], [size_per_feature_list[layer]*4], [4], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(4, "MUL", "applyedge", "R", [edge_num, edge_num], [3, -1], 2, 0, [], [], [size_per_feature_list[layer]*4, size_per_feature_list[layer]*4], [5], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(5, "ADD", "gather", "R", [edge_num], [4], 1, 0, [], [], [size_per_feature_list[layer]*4], [6], node_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(6, "MM", "applynode", "R", [node_num], [5], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [], node_num, weight_size[layer]*4))

    elif network == 'GraphSAGE':
        data.append(gen_one_op(0, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [1], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(1, "MUL", "applyedge", "R", [edge_num, edge_num], [0, -1], 2, 0, [], [], [size_per_feature_list[layer]*4, size_per_feature_list[layer]*4], [2], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(2, "ADD", "gather", "R", [edge_num], [1], 1, 0, [], [], [size_per_feature_list[layer]*4], [3], node_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(3, "MM", "applynode", "R", [node_num], [2], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [5], node_num, weight_size[layer]*4))
        data.append(gen_one_op(4, "MM", "applynode", "R", [node_num], [], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [5], node_num, weight_size[layer]*4))
        data.append(gen_one_op(5, "ADD", "applynode", "R", [node_num, node_num], [3, 4], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [6], node_num, weight_size[layer]*4))
        data.append(gen_one_op(6, "SF", "applynode", "R", [node_num], [5], 1, 0, [], [], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))

    elif network == 'GIN':
        data.append(gen_one_op(0, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [1], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(1, "MUL", "applyedge", "R", [edge_num, edge_num], [0, -1], 2, 0, [], [], [size_per_feature_list[layer]*4, size_per_feature_list[layer]*4], [2], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(2, "ADD", "gather", "R", [edge_num], [1], 1, 0, [], [], [size_per_feature_list[layer]*4], [3], node_num, size_per_feature_list[layer]*4))
        #TODO:可能有问题
        #设成SF
        data.append(gen_one_op(3, "MUL", "applynode", "R", [node_num, node_num], [-1, -1], 2, 0, [], [], [size_per_feature_list[layer]*4,4], [4], node_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(4, "ADD", "applynode", "R", [node_num, node_num], [2,3], 2, 0, [], [], [size_per_feature_list[layer]*4,size_per_feature_list[layer]*4], [5], node_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(5, "MM", "applynode", "R", [node_num], [4], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [6], node_num, weight_size[layer]*4))
        data.append(gen_one_op(6, "SF", "applynode", "R", [node_num], [5], 1, 0, [], [], [weight_size[layer]*4], [7], node_num, weight_size[layer]*4))
        data.append(gen_one_op(7, "MM", "applynode", "R", [node_num], [6], 1, 1, [], [weight_size[layer]*weight_size[layer]*4], [weight_size[layer]*4], [8], node_num, weight_size[layer]*4))
        data.append(gen_one_op(8, "SF", "applynode", "R", [node_num], [7], 1, 0, [], [], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))
    
    elif network == 'DGN':
        data.append(gen_one_op(0, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [2], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(1, "NONE", "scatter", "R", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [2], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(2, "ADD", "applyedge", "R", [edge_num, edge_num], [0, 1], 2, 0, [], [], [size_per_feature_list[layer]*4, size_per_feature_list[layer]*4], [3], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(3, "MM", "applyedge", "R", [edge_num], [2], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [7], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(4, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [weight_size[layer]*4], [6], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(5, "NONE", "scatter", "R", [node_num], [], 1, 0, [], [], [weight_size[layer]*4], [6], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(6, "ADD", "applyedge", "R", [edge_num, edge_num], [4, 5], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [7], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(7, "ADD", "applyedge", "R", [edge_num, edge_num], [3, 6], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [8], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(8, "ADD", "gather", "R", [edge_num], [7], 1, 0, [], [], [weight_size[layer]*4], [9], node_num, weight_size[layer]*4))
        data.append(gen_one_op(9, "MUL", "applynode", "R", [node_num], [8], 1, 0, [], [], [weight_size[layer]*4], [10], node_num, weight_size[layer]*4))
        data.append(gen_one_op(10, "SF", "applynode", "R", [node_num], [9], 1, 0, [], [], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))

    elif network == 'PNA' and not isReorder:
        data.append(gen_one_op(0, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [3], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(1, "NONE", "scatter", "R", [node_num], [], 1, 0, [], [], [size_per_feature_list[layer]*4], [4], edge_num, size_per_feature_list[layer]*4))
        data.append(gen_one_op(2, "MM", "applyedge", "R", [edge_num], [], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [6], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(3, "MM", "applyedge", "R", [edge_num], [0], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [5], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(4, "MM", "applyedge", "R", [edge_num], [1], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [5], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(5, "ADD", "applyedge", "R", [edge_num, edge_num], [3,4], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [6], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(6, "ADD", "applyedge", "R", [edge_num, edge_num], [2,5], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [7], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(7, "SF", "applyedge", "R", [edge_num], [6], 1, 0, [], [], [weight_size[layer]*4], [8], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(8, "ADD", "gather", "R", [edge_num], [7], 1, 0, [], [], [weight_size[layer]*4], [9], node_num, weight_size[layer]*4))
        data.append(gen_one_op(9, "MUL", "applynode", "R", [node_num], [8], 1, 0, [], [], [weight_size[layer]*4], [10], node_num, weight_size[layer]*4))
        data.append(gen_one_op(10, "MM", "applynode", "R", [node_num], [9], 1, 1, [], [weight_size[layer]*weight_size[layer]*4], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))

    elif network == 'PNA' and isReorder:
        data.append(gen_one_op(0, "MM", "applynode", "R", [node_num], [0], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [3], node_num, weight_size[layer]*4))
        data.append(gen_one_op(1, "MM", "applynode", "R", [node_num], [1], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [4], node_num, weight_size[layer]*4))
        data.append(gen_one_op(2, "MM", "applyedge", "R", [edge_num], [], 1, 1, [], [size_per_feature_list[layer]*weight_size[layer]*4], [size_per_feature_list[layer]*4], [6], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(3, "NONE", "scatter", "C", [node_num], [], 1, 0, [], [], [weight_size[layer]*4], [5], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(4, "NONE", "scatter", "R", [node_num], [], 1, 0, [], [], [weight_size[layer]*4], [5], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(5, "ADD", "applyedge", "R", [edge_num, edge_num], [3,4], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [6], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(6, "ADD", "applyedge", "R", [edge_num, edge_num], [2,5], 2, 0, [], [], [weight_size[layer]*4, weight_size[layer]*4], [7], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(7, "SF", "applyedge", "R", [edge_num], [6], 1, 0, [], [], [weight_size[layer]*4], [8], edge_num, weight_size[layer]*4))
        data.append(gen_one_op(8, "ADD", "gather", "R", [edge_num], [7], 1, 0, [], [], [weight_size[layer]*4], [9], node_num, weight_size[layer]*4))
        data.append(gen_one_op(9, "MUL", "applynode", "R", [node_num], [8], 1, 0, [], [], [weight_size[layer]*4], [10], node_num, weight_size[layer]*4))
        data.append(gen_one_op(10, "MM", "applynode", "R", [node_num], [9], 1, 1, [], [weight_size[layer]*weight_size[layer]*4], [weight_size[layer]*4], [], node_num, weight_size[layer]*4))

    else:
        print("Error: No such network")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        yaml.safe_dump(data, file)

def generate_connections(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    connections = []
    
    for entry in data:
        op_id = entry['OP_NO']
        output_list = entry['OUTPUT']['output_list']
        
        for output_item in output_list:
            connections.append([op_id, output_item])
    
    return connections

if __name__ == '__main__':

    data_set = ['cora']
    #data_set = ['cora', 'pubmed', 'citeseer', 'reddit']
    network = ['GCN', 'GAT', 'SGC', 'GraphSAGE', 'GIN', 'DGN', 'PNA']
    layer = [1, 2, 3]
    isReorder = [True, False]

    node_num = 0
    edge_num = 0
    size_per_feature = 0

    for one_data_set in data_set:
        if one_data_set == 'cora':
            node_num = 2708
            edge_num = 10556
            size_per_feature = 1433
        elif one_data_set == 'pubmed':
            node_num = 19717
            edge_num = 88648
            size_per_feature = 500
        elif one_data_set == 'citeseer':
            node_num = 3327
            edge_num = 9104
            size_per_feature = 3703
        elif one_data_set == 'reddit':
            node_num = 232965
            edge_num = 114615892
            size_per_feature = 602
        
        for one_network in network:
            for one_layer in layer:
                for one_isReorder in isReorder:
                    file_path = ''
                    if one_isReorder:
                        file_path = 'Network/'+one_network+'/'+one_network+'-'+one_data_set+'/'+one_network+'-trans/'+one_network+'-layer'+str(one_layer)+'-trans.yaml'
                    else:
                        file_path = 'Network/'+one_network+'/'+one_network+'-'+one_data_set+'/'+one_network+'-original/'+one_network+'-layer'+str(one_layer)+'-original.yaml'
                    gen_yaml(file_path, node_num, edge_num, size_per_feature, one_network, one_layer, one_isReorder)
        print("Successful")