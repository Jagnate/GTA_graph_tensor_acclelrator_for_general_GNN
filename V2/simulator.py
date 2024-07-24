import yaml
import math
import tqdm

#四个维度，mm input/output feature； ele 两个

#向上取整（）
#（feature_size/16）*（分块大小/16）
#（1433/16）* （128/16）
#2708*1433 1433*128

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

def load(data,isCycle):
    global load_p
    global save_p
    global total_p
    global save_start_p
    global c_p
    global bw
    global rw

    rw += data

    if isCycle == 0:
        latency = math.ceil(data/bw)*(10**(-9))
    else:
        latency = math.ceil(data/bw)

    if(load_p + data > save_start_p and save_start_p!=0):
        load_p = save_p + latency
    else:
        load_p += latency

    if(load_p > c_p and load_p > save_p):
        total_p = load_p

def compute(data, compute_type,isCycle):
    global load_p
    global save_p
    global compute_p
    global c_p
    global total_p
    global compute_perfom

    if isCycle == 0:
        latency = math.ceil(data[0]/compute_perfom[compute_type][0])*math.ceil(data[1]/compute_perfom[compute_type][1])*(10**(-9))
    else:
        latency = math.ceil(data[0]/compute_perfom[compute_type][0])*math.ceil(data[1]/compute_perfom[compute_type][1])

    if(compute_type == 0):
        if(load_p < compute_p[0]):
            compute_p[0] += latency
        else:
            compute_p[0] = load_p + latency
    elif(compute_type == 1):
        if(load_p < compute_p[1]):
            compute_p[1] += latency
        else:
            compute_p[1] = load_p + latency
    if(compute_p[0] > compute_p[1]):
        c_p = compute_p[0]
    else:
        c_p = compute_p[1]

    if(c_p > load_p and c_p > save_p):
        total_p = c_p

def save(data,isCycle):
    global load_p
    global save_p
    global c_p
    global save_start_p
    global total_p
    global bw
    global rw

    rw += data

    if isCycle == 0:
        latency = math.ceil(data/bw)*(10**(-9))
    else:
        latency = math.ceil(data/bw)

    if(c_p < save_p):
        save_p += latency
    else:
        save_start_p = c_p
        save_p = c_p + latency

    if(save_p > load_p and save_p > c_p):
        total_p = save_p

def decode(data,type,op,isCycle,optype,times1,times2,sprase_dataset):
    global load_p
    global save_p
    global compute_p
    global c_p
    global total_p
    global save_start_p
    global rw

    if type == "load":
        load_data = []
        for j in range(0,len(data[op]["load_list"])):
            load_data.append(0)
            if(load_data[len(load_data)-1]==0):
                continue
            if optype == "gather" or optype=="applyedge":
                load_data[len(load_data)-1] += data[op]["load_list"][j]*sprase_dataset[times1][times2]
            else:
                load_data[len(load_data)-1] += data[op]["load_list"][j]*data[op]["load_shape"][j][1]
            load(load_data[len(load_data)-1],isCycle)
    elif type == "compute":
        compute_data = []
        for j in range(0,len(data[op]["compute_list"])):
            compute_data.append([0,0])
            if(compute_data[len(compute_data)-1]==0):
                continue
            if optype == "gather" or optype=="applyedge":
                if data[op]["compute_type"][j] == 1: #ele-wise
                    compute_data[len(compute_data)-1][0] = data[op]["w_list"][j][1]/4
                    compute_data[len(compute_data)-1][1] += sprase_dataset[times1][times2]
                else: #mm
                    compute_data[len(compute_data)-1][0] = data[op]["w_list"][j][0]
                    compute_data[len(compute_data)-1][1] += data[op]["w_list"][j][1]/4*sprase_dataset[times1][times2]
            else:
                if data[op]["compute_type"][j] == 1: #ele-wise
                    compute_data[len(compute_data)-1] = [data[op]["w_list"][j][1]/4,data[op]["compute_shape"][j][0]*data[op]["compute_shape"][j][1]]
                    #compute_data.append([data[op]["w_list"][j][1]/4,data[op]["compute_shape"][j][0]*data[op]["compute_shape"][j][1]])
                else: #mm
                    compute_data[len(compute_data)-1] = [data[op]["w_list"][j][0],data[op]["w_list"][j][1]/4*data[op]["compute_shape"][j][0]*data[op]["compute_shape"][j][1]]
                    #compute_data.append([data[op]["w_list"][j][0],data[op]["w_list"][j][1]/4*data[op]["compute_shape"][j][0]*data[op]["compute_shape"][j][1]])
            compute(compute_data[len(compute_data)-1],data[op]["compute_type"][j],isCycle)
    else:
        save_data = []
        for j in range(0,len(data[op]["save_list"])):
            save_data.append(0)
            if(data[op]["save_list"][j]==0):
                continue
            if optype == "scatter" or optype=="applyedge":
                save_data[len(save_data)-1] += data[op]["save_list"][j]*sprase_dataset[times1][times2]
            else:
                save_data[len(save_data)-1] += data[op]["save_list"][j]*data[op]["save_shape"][j][1]
            save(save_data[len(save_data)-1],isCycle)

#isR = 1顺序相同
def pipeline(data,op_fused,isCycle):
    global load_p
    global save_p
    global compute_p
    global c_p
    global total_p
    global save_start_p
    global rw

    record = []
    sparse_dataset = []

    for i in tqdm.tqdm(range(0,len(op_fused))): #遍历大块
        for times1 in range(0,int(data[op_fused[i][0]]["times_1"])):
            for op in op_fused[i]:
                record.append(total_p)
                if data[op]["type"] == "scatter" and data[op]["isR"] == 1:
                    sparse_dataset = read(data[op]["sparse_path"])
                    decode(data,"load",op,isCycle,"scatter",times1, -1,sparse_dataset)
                    for times2 in range(0,int(data[op_fused[i][0]]["times_2"])):
                        decode(data, "save",op,isCycle,"scatter",times1, times2, sparse_dataset)
                elif data[op]["type"] == "scatter" and data[op]["isR"] == 0:
                    sparse_dataset = read(data[op]["sparse_path"])
                    for times2 in range(0,int(data[op_fused[i][0]]["times_2"])):
                        decode(data, "load",op,isCycle,"scatter",times1, times2, sparse_dataset)
                        decode(data, "save",op,isCycle,"scatter",times1, times2, sparse_dataset)
                elif data[op]["type"] == "gather" and data[op]["isR"] == 1:
                    sparse_dataset = read(data[op]["sparse_path"])
                    decode(data, "load",op,isCycle,"gather", times1, -1, sparse_dataset) #load node
                    for times2 in range(0,int(data[op_fused[i][0]]["times_2"])):
                        decode(data, "load",op,isCycle,"gather",times1, times2, sparse_dataset)
                        decode(data, "compute",op,isCycle,"gather",times1, times2, sparse_dataset)
                    decode(data, "save",op,isCycle,"gather",times1, -1, sparse_dataset)
                elif data[op]["type"] == "gather" and data[op]["isR"] == 0:
                    sparse_dataset = read(data[op]["sparse_path"])
                    for times2 in range(0,int(data[op_fused[i][0]]["times_2"])):
                        decode(data, "load",op,isCycle,"gather",times1, times2, sparse_dataset)
                        decode(data, "compute",op,isCycle,"gather",times1, times2, sparse_dataset)
                        decode(data, "save",op,isCycle,"gather",times1, times2, sparse_dataset)
                elif data[op]["type"] == "applynode":
                    if(times1==0): #load W
                        for w in data[op]["w_list"]:
                            load(w[0]*w[1],isCycle)
                    for times3 in range(0,int(data[op_fused[i][0]]["times_3"])):
                        decode(data, "load",op,isCycle,"applynode",times1, times3, sparse_dataset)
                        decode(data, "compute",op,isCycle,"applynode",times1, times3, sparse_dataset)
                        decode(data, "save",op,isCycle,"applynode",times1, times3, sparse_dataset)
                else:
                    sparse_dataset = read(data[op]["sparse_path"])
                    if(times1==0): #load W
                        for w in data[op]["w_list"]:
                            load(w[0]*w[1],isCycle)
                    for times2 in range(0,int(data[op_fused[i][0]]["times_2"])):
                        decode(data, "load",op,isCycle,"applyedge",times1, times2, sparse_dataset)
                        decode(data, "compute",op,isCycle,"applyedge",times1, times2, sparse_dataset)
                        decode(data, "save",op,isCycle,"applyedge",times1, times2, sparse_dataset)

    return total_p,record,rw

if __name__ == '__main__':

    #硬件参数
    #cycle: 1ns = 1cycle
    bw     = 128*(1024**3)*(10**(-9)) #128GB/s = 128*10^-9 GB/cycle
    pl_in  = 16 #16*16 feature/cycle
    pl_out = 16

    buffer_size = 2*1024*1024 #byte
    #结构
    compute_op     = ["mm","ele-wise"]
    compute_perfom = [[pl_in,pl_out],[pl_in,pl_out]]

    load_p = 0
    save_p = 0
    save_start_p = 0
    compute_p = [0,0] #mm, ele-wise
    c_p = 0
    total_p = 0
    rw = 0

    data = read("/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/v1/fused.yaml")
    #op_fused =  [[0, 1, 2], [5], [4, 6, 7, 8, 9, 10], [3, 11, 12, 13]]
    #op_fused = [[0],[1],[2], [4,5,6], [7,8,9,10], [3, 11, 12, 13]]
    #op_fused = [[0,1,2,3],[4,5],[6,7,8]]
    op_fused = [[0],[1],[2], [4],[5],[6], [7],[8],[9],[10], [3], [11], [12], [13]]

    #isCycle: 0 second; 1 cycle
    print(pipeline(data,op_fused,1))

# def pipeline(data,op_fused):
#     load_p = 0
#     save_p = 0
#     compute_p = [0,0] #mm, ele-wise
#     c_p = 0
#     total_p = 0
#     for i in tqdm.tqdm(range(0,len(op_fused))): #遍历大块
#         for j in range(0,int(data[op_fused[i][0]]["times_1"])):
#             for op in op_fused[i]: #遍历大块里的op,load
#                 if j == 0:
#                     #先读weight,只读一次
#                     load = data[op]["w_list"][0]/bw
#                     if(load_p + load > c_p):
#                         load_p = save_p + load
#                         total_p += load
#                     else:
#                         load_p += load
#                 #读feature
#                 for ll in range(0,len(data[op]["load_list"])):
#                     for ln in range(0,data[op]["load_num"][ll]):
#                         load = data[op]["load_list"][ll]/bw
#                         if(load_p + load > c_p):
#                             load_p = save_p + load
#                             total_p += load
#                         else:
#                             load_p += load
#             for op in op_fused[i]: #遍历大块里的op, compute
#                 compute = data[op]["compute_list"][cl]/pl
#                 if(data[op]["compute_type"][cl] == 0):
#                     if(load_p < compute_p[0]):
#                         compute_p[0] += + compute
#                     else:
#                         compute_p[0] += load_p + compute
#                 elif(data[op]["compute_type"][cl] == 1):
#                     if(load_p < compute_p[1]):
#                         compute_p[1] += + compute
#                     else:
#                         compute_p[1] += load_p + compute
#                 if(compute_p[0] > compute_p[1]):
#                     c_p = compute_p[0]
#                 else:
#                     c_p = compute_p[1]
#             for op in op_fused[i]: #遍历大块里的op, store
#                 for sl in range(0,len(data[op]["save_list"])):
#                     for sn in range(0,data[op]["save_num"][sl]):
#                         save = data[op]["save_list"][sl]/bw
#                         if(c_p < save_p):
#                             save_p += save
#                         else:
#                             save_p += c_p + save
#                         total_p = save_p    

#     return total_p

#实际存储大小为tile_size*size_per_feature(size_per_feature*4)
#tile_size为多少个数，多少个结点/边
# def create_list(data, op_list, tile_size):
#     load_list = [] #s
#     compute_list = [] #s
#     compute_type = [] #[0,1] 0:mm; 1:ele-wise
#     save_list = [] #s
#     load_num_list = [] #需要load多少次才能compute or save,一个下标代表一个大块
#     #comp_num_list = [] #需要compute多少次才能save 
#     for i in range(0,len(op_list)): #遍历大块
#         #load_num_list.append(0)
#         load_list.append([])
#         compute_list.append([])
#         compute_type.append([])
#         save_list.append([])
#         #comp_num_list.append(0)
#         for j in range(0,len(op_list[i])): #遍历大块里的op
#             #SCATTER
#             if data[op_list[i][j]]["TYPE"] == "scatter":
#                 #load
#                 if data[op_list[i][j]]["ORDER"] == "R":
#                     for k in data[op_list[i][j]]["INPUT"]["size_per_feature"]:
#                         load_list[len(load_list)-1].append(tile_size[i]*k/bw)
#                         load_num_list[len(load_num_list)-1] += 1
#                 else: #顺序不一致的时候，一个一个scatter
#                     for k in data[op_list[i][j]]["INPUT"]["size_per_feature"]:
#                         load_list[len(load_list)-1].append(1*k/bw)
#                         load_num_list[len(load_num_list)-1] += 1
#                 compute_type[len(compute_type)-1].append(-1)
#                 compute_list[len(compute_list)-1].append(0)
#                 #save
#                 save_list[len(save_list)-1].append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"]*tile_size[i]/bw)
#             #GATHER
#             elif data[op_list[i][j]]["TYPE"] == "gather":
#                 #load
#                 if data[op_list[i][j]]["ORDER"] == "R":
#                     for k in data[op_list[i][j]]["INPUT"]["size_per_feature"]:
#                             load_list[len(load_list)-1].append(tile_size[i]*k/bw)
#                             #load_num_list[len(load_num_list)-1] += 1
#                 else: #顺序不一致的时候，一个一个gather
#                     for k in data[op_list[i][j]]["INPUT"]["size_per_feature"]:
#                             load_list[len(load_list)-1].append(1*k/bw)
#                             #load_num_list[len(load_num_list)-1] += 1
#                 #compute - plus(ele-wise)
#                 compute_type[len(compute_type)-1].append(1)
#                 compute_list[len(compute_list)-1].append(tile_size[i]/pl)
#                 #save
#                 save_list[len(save_list)-1].append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"]*tile_size[i]/bw)
#             #APPLY
#             else:
#                 #权重读取
#                 if data[op_list[i][j]]["INPUT"]["input_nong_num"] != 0:
#                     compute_type[len(compute_type)-1].append(0)
#                     for k in data[op_list[i][j]]["INPUT"]["input_size"]:
#                         load_list[len(load_list)-1].append(k/bw)
#                         #load_num_list[len(load_num_list)-1] += 1
#                 else:
#                     compute_type[len(compute_type)-1].append(1)
#                 #特征读取
#                 for k in data[op_list[i][j]]["INPUT"]["size_per_feature"]:
#                     load_list[len(load_list)-1].append(tile_size[i]*k/bw)
#                     #load_num_list[len(load_num_list)-1] += 1
#                 #计算
#                 if compute_type[len(compute_type)-1][len(compute_type[len(compute_type)-1])-1] == 0: #mm
#                     #mm不确定怎么算的？
#                     compute_list[len(compute_list)-1].append(data[op_list[i][j]]["INPUT"]["input_size"][0]*tile_size[i]/(pl*data[op_list[i][j]]["OUTPUT"]["size_per_feature"]))
#                 else: #element-wise
#                     for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
#                         compute_list[len(compute_list)-1].append(tile_size[i]/pl)
#                 #save
#                 save_list[len(save_list)-1].append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"]*tile_size[i]/bw)

#     return load_list, load_num_list, compute_list, compute_type, save_list
            

# def pipeline(load_list, load_num_list, compute_list, compute_type, save_list, tile_num):
#     load_p = 0
#     save_p = 0
#     compute_p = [0,0] #mm, ele-wise
#     c_p = 0
#     total_p = 0
#     for i in range(0,len(tile_num)): #第i个大块
#         for j in range(0,tile_num[i]): #第i个块的分块数量
#             for load in load_list[i]:
#                 for each_load in load:
#                     if(load_p + load > c_p):
#                         load_p = save_p + load
#                         total_p += load
#                     else:
#                         load_p += load
#                 for k in range(0,compute_list[i]):
#                     if(compute_type[i][k] == 0):
#                         if(load_p < compute_p[0]):
#                             compute_p[0] += + compute_list[i][k]
#                         else:
#                             compute_p[0] += load_p + compute_list[i][k]
#                     elif(compute_type[i][k] == 1):
#                         if(load_p < compute_p[1]):
#                             compute_p[1] += + compute_list[i][k]
#                         else:
#                             compute_p[1] += load_p + compute_list[i][k]
#                     if(compute_p[0] > compute_p[1]):
#                         c_p = compute_p[0]
#                     else:
#                         c_p = compute_p[1]
#                     for save in save_list[i]:
#                         if(c_p < save_p):
#                             save_p += save
#                         else:
#                             save_p += c_p + save
#                         total_p = save_p    