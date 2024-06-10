import yaml
import copy
import math
import numpy as np
import tqdm
#gather改规则和后面融合
#必须断开的scatter有反复读的问题
#分块都是Dx1

#算W在最大块
def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

def write(data):
    with open('res.yaml','w', encoding='utf-8') as file:
        yaml.dump(data,file)

def increment_binary(binary):
    number = int(binary, 2)
    number += 1
    return bin(number)[2:]

def binary_to_int_array(binary):
    return [int(bit) for bit in binary]

def check_bit(binary, k):
    if(binary[k]=='1'):
        return True
    else:
        return False

def judge_order(data,op_in,op_out):
    if(data[op_out]["TYPE"]=="scatter"):
        if(data[op_out]["ORDER"]!=data[op_in]["ORDER"]):
            return True
    elif(data[op_in]["TYPE"]=="gather"):
        if(data[op_in]["ORDER"]!=data[op_out]["ORDER"]):
            return True
    else:
        return False 

def fuse_pattern(data,src,dst, op_array, size, sg_list):
    fuse_array = []
    record_1 = []
    record_0 = []
    fuse_op = []
    each_size = []
    each_edge = []
    rw = 0
    for i in range(0,data[0]["INPUT"]["input_g_num"]):
        rw += data[0]["INPUT"]["size_per_feature"][i]*data[0]["INPUT"]["feature_number"][i]
    rw += data[len(data)-1]["OUTPUT"]["output_number"]*data[len(data)-1]["OUTPUT"]["size_per_feature"]
    for i in range(0,len(op_array)):
        if(op_array[i]==1): 
            record_1.append(i) 
        else:
            record_0.append(i)
    #融合的边添加非图结构的输入输出
    for i in record_1:
        find = 0
        for j in range(0,len(fuse_array)):
            if(dst[i] in fuse_op and src[i] in fuse_op):
            # if(dst[i] in fuse_array[j] and src[i] in fuse_array[j]):
                find=1
                break
            elif(src[i] not in fuse_array[j] and dst[i] in fuse_array[j] and src[i] not in fuse_op):
                find=1
                fuse_array[j].append(src[i])
                each_edge[j].append(i)
                fuse_op.append(src[i])
                for k in range(0,data[src[i]]["INPUT"]["input_nong_num"]):
                    rw += data[src[i]]["INPUT"]["input_size"][k]
                break
            elif(dst[i] not in fuse_array[j] and src[i] in fuse_array[j] and dst[i] not in fuse_op):
                find=1
                fuse_array[j].append(dst[i])
                each_edge[j].append(i)
                fuse_op.append(dst[i])
                for k in range(0,data[dst[i]]["INPUT"]["input_nong_num"]):
                    each_size[j] += data[dst[i]]["INPUT"]["input_size"][k]
                    if(each_size[j]>size):
                        return False
                #rw += data[dst[i]]["OUTPUT"]["output_number"]*data[dst[i]]["OUTPUT"]["size_per_feature"]
                break
            #print(rw)
        if(find==0):
            fuse_array.append([src[i],dst[i]])
            each_edge.append([i])
            fuse_op.append(src[i])
            fuse_op.append(dst[i])
            each_size.append(0)
            for k in range(0,data[src[i]]["INPUT"]["input_nong_num"]):
                each_size[len(each_size)-1] += data[src[i]]["INPUT"]["input_size"][k]
                if(each_size[len(each_size)-1]>size):
                    return False
                rw += data[src[i]]["INPUT"]["input_size"][k]
            for k in range(0,data[dst[i]]["INPUT"]["input_nong_num"]):
                each_size[len(each_size)-1] += data[dst[i]]["INPUT"]["input_size"][k]
                if(each_size[len(each_size)-1]>size):
                    return False
                rw += data[dst[i]]["INPUT"]["input_size"][k]
    #剪开的边添加非图结构的输入+输入op的输出
    for i in record_0:
        rw += data[src[i]]["OUTPUT"]["size_per_feature"]*data[src[i]]["OUTPUT"]["output_number"]
        if(src[i] not in fuse_op):
            fuse_op.append(src[i])
            fuse_array.append([src[i]])
            each_size.append(0)
            for k in range(0,data[src[i]]["INPUT"]["input_nong_num"]):
                rw += data[src[i]]["INPUT"]["input_size"][k]
        if(dst[i] not in fuse_op):
            fuse_op.append(dst[i])
            fuse_array.append([dst[i]])
            each_size.append(0) 
            for k in range(0,data[dst[i]]["INPUT"]["input_nong_num"]):
                rw += data[dst[i]]["INPUT"]["input_size"][k]
    #对不剪开点边添加输入+输出
    tile_size = []
    tile_num = []
    for i in range(0,len(fuse_array)):
        tile_size.append(1)
        tile_num.append(1)
        total = 0
        if(len(fuse_array[i])>1):
            flag = 0
            temp = copy.deepcopy(each_size[i])
            for j in each_edge[i]:
                for k in range(0,data[src[j]]["INPUT"]["input_g_num"]):
                    total += data[src[j]]["INPUT"]["size_per_feature"][k]
                if(data[src[j]]["TYPE"]!="scatter"):
                    total += data[src[j]]["OUTPUT"]["size_per_feature"]
                total += data[dst[j]]["INPUT"]["size_per_feature"][int(data[dst[j]]["INPUT"]["input_g_list"].index(src[j]))]
            num = math.floor((size-temp)/total)
            if(num<=0):
                return False
            else:
                each_size[i] += num*(total)
                #多少个数/num个数一个块=多少个块
                tile_num[i] = math.ceil(data[src[j]]["OUTPUT"]["output_number"]/num)
                tile_size[i] = num
            # for num in range(data[fuse_array[i][0]]["OUTPUT"]["output_number"],0,-1):
            #     if(temp+num*total>size):
            #         continue
            #     else:
            #         each_size[i] += num*(total)
            #         #多少个数/num个数一个块=多少个块
            #         tile_num[i] = math.ceil(data[src[j]]["OUTPUT"]["output_number"]/num)
            #         tile_size[i] = num*4
            #         flag = 1
            #         break
            # if(flag==0):
            #     return False
    #对剪开的边的输出op添加输入
    for j in record_0:
        if(j in sg_list):
            rw += data[dst[j]]["INPUT"]["feature_number"][int(data[dst[j]]["INPUT"]["input_g_list"].index(src[j]))]*data[dst[j]]["INPUT"]["size_per_feature"][int(data[dst[j]]["INPUT"]["input_g_list"].index(src[j]))]*tile_size[i]
        else:
            #print(dst[j],int(data[dst[j]]["INPUT"]["input_g_list"].index(src[j])))
            rw += data[dst[j]]["INPUT"]["feature_number"][int(data[dst[j]]["INPUT"]["input_g_list"].index(src[j]))]*data[dst[j]]["INPUT"]["size_per_feature"][int(data[dst[j]]["INPUT"]["input_g_list"].index(src[j]))]          
    #return [fuse_array,each_size,tile_size,tile_num,rw]
            #[融合方式，融合块所占空间,融合块大小（多少个数不是byte）,分块数量,访存量]
    return [fuse_array,each_size,tile_size,tile_num,rw]


def generate_binary(data,edge,sg_list,src,dst,size):
    res = []
    n = len(edge)
    binary = '0' * n
    num = 0
    #int_array = edge
    for i in tqdm.tqdm(range(2**n)):
        if(i!=0):
            binary = increment_binary(binary)
            binary = binary.zfill(n)
        #特判scatter和gather
        flag = 0
        for i in sg_list:
            if check_bit(binary, i):
                flag = 1
                break
        if(flag==1):
            continue
        else:
            int_array = binary_to_int_array(binary)
            #print(int_array)
            num += 1
            #print(num)
            #print(num,fuse_pattern(data,src,dst,int_array,size,sg_list))
            temp_res = fuse_pattern(data,src,dst,int_array,size,sg_list)
            if(temp_res==False):
                continue
            else:
                res.append(temp_res)
                specific_binary = '0000011101101111'
                if binary == specific_binary:
                    print(f"temp_res for binary {specific_binary}: {temp_res}")
                if(len(res)==1):
                    print(res[0])
                # print(res[len(res)-1])
                # time.sleep(0.5)
    arr = np.array(res,dtype=object)
    sorted_indices = np.argsort(arr[:,-1])
    sorted_res = arr[sorted_indices]
    return sorted_res


def create_optree(path,size):
    data = read(path)
    edge = []
    src  = []
    dst  = []
    sg_list = []
    num = 0
    for i in range(0,len(data)):
        for j in data[i]["OUTPUT"]["output_list"]:
            num += 1
            src.append(data[i]["OP_NO"])
            edge.append(1)
            dst.append(j)
            if(judge_order(data,data[i]["OP_NO"],j)):
                sg_list.append(len(edge)-1)
    print(src)
    print(dst)
    print(edge)
    print(sg_list)
    print("-----------------")
    res = generate_binary(data,edge,sg_list,src,dst,size)
    for i in range(0,10):
        print(res[i])
    for i in range(1,10):
        print(res[len(res)-i])
    # for i in range(len(res)-1,len(res)-10,-1):
    #     print(res[i])
    #print(res[0])

#[融合方式，融合块所占空间,融合块大小,分块数量,访存量]
if __name__ == '__main__':
    size = 2*1024*1024
    create_optree("/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/v1/GAT_Cora.yaml",size)


#官方GCN GAT Cora Pubmed（reddit）2x2四个实验
#优化前分开运行
#写个文档

#Cora:
    #Statistics:
    # - Nodes: 2708
    # - Node_size: 1433
    # - Edges: 10556
    # - Number of Classes: 7
#Pubmed:
    # Statistics:

    # - Nodes: 19717
    # - Node_size: 500
    # - Edges: 88651
    # - Number of Classes: 3
#Reddit
    # Statistics

    # - Nodes: 232,965
    # - Edges: 114,615,892
    # - Node feature size: 602
    # - Number of training samples: 153,431
    # - Number of validation samples: 23,831
    # - Number of test samples: 55,703
#Citeseer
    # Statistics:

    # - Nodes: 3327
    # - Edges: 9228
    # - Node feature size: 3703
    # - Number of Classes: 6

