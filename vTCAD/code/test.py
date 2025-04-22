from compiler import compile
from interpreter import interpret
from simulator import simulate

if __name__ == '__main__':
    data_set = 'cora'
    network = 'GIN'
    layer = 'layer3'
    isReorder = False
    res = compile(data_set,network,layer,False,False,True,False)[0]
    print(res[0])
    op_array = res[0][0]
    tile_size_list = res[0][1]
    print(interpret(data_set, network, isReorder, layer, op_array, tile_size_list))
    print(simulate(tile_size_list,data_set,network,layer,isReorder,False,True,'GTA'))