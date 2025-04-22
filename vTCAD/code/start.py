from compiler import compile
from interpreter import interpret
from simulator import simulate
import argparse
import yaml

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compiler script")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset')
    parser.add_argument('--network', type=str, required=True, help='The name of the network')
    #parser.add_argument('--layer', type=str, required=True, help='The name of the network')
    parser.add_argument('--isReorder', type=bool, default=False, help='Boolean flag for reorder')
    parser.add_argument('--isSinput', type=bool, default=False, help='Boolean flag for Sinput')
    parser.add_argument('--isPingpang', type=bool, default=False, help='Boolean flag for Pingpang Buffer')
    parser.add_argument('--isFlexibleBuffer', type=bool, default=False, help='Boolean flag for Flexible Buffer')
    args = parser.parse_args()

    layers = []
    if args.network == 'GAT':
        layers = ['layer1','layer2','layer3']
    else:
        layers = ['alllayer']

    op_array = []
    tile_size_list = []
    cycle = 0
    rw = 0

    print("Starting compilation...")
    for i in range(0,len(layers)):
        layer = layers[i]
        res = compile(args.dataset, args.network, layer, args.isReorder, args.isSinput, args.isPingpang, args.isFlexibleBuffer)
        op_array.append(res[0])
        tile_size_list.append(res[1])
    print("Compilation Done\n")

    print("Generating instructions...")
    for i in range(0,len(layers)):
        layer = layers[i]
        interpret(args.dataset, args.network, args.isReorder, layer, op_array[i], tile_size_list[i])
    print("Inst Generated\n")

    print("Starting simulation...")
    for i in range(0,len(layers)):
        per_cycle, per_rw = simulate(tile_size_list[i], args.dataset, args.network, layer, args.isReorder, args.isSinput)
        cycle += per_cycle
        rw += per_rw
    print("Simulation Done\n")

    print('Latency:',(cycle-1)/10**9,'s')
    print('总访存量:',rw/10**6,'MB')

    if args.isReorder:
        print("Test Name:"+' '+args.dataset+'-'+args.network+'-'+"Reorder")
    else:
        print("Test Name:"+' '+args.dataset+'-'+args.network+'-'+"Original")