import yaml

#单位byte
bw = 128*1024*1024*1024 #128GB/s
pl = 256 #16*16 feature/s
buffer_size = 2*1024*1024 #byte

#结构
compute_op     = ["mm","ele-wise"]
compute_perfom = [pl,pl]

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

def pipeline(data):
    load_p = 0
    save_p = 0
    compute_p = [0,0] #mm, ele-wise
    c_p = 0
    total_p = 0
    for i in range(0,len(data)): #遍历大块
        for j in range(0,len(data[i]["load_list"])): #大块里有几个op
            #先读weight
            load = data[i]["w_list"][j]/bw
            if(load_p + load > c_p):
                load_p = save_p + load
                total_p += load/bw
            else:
                load_p += load
            #读feature
            print(data[i]["load_num"][j])
            for num in range(0,data[i]["load_num"][j]): #第j个op需要读几次
                load = data[i]["load_list"][j]/bw
                if(load_p + load > c_p):
                    load_p = save_p + load
                    total_p += load
                else:
                    load_p += load
            for num in range(0,int(data[i]["compute_num"][j])): #第j个op需要读几次
                compute = data[i]["compute_list"][j]/pl
                if(data[i]["compute_type"][j] == 0):
                    if(load_p < compute_p[0]):
                        compute_p[0] += + compute
                    else:
                        compute_p[0] += load_p + compute
                elif(data[i]["compute_type"][j] == 1):
                    if(load_p < compute_p[1]):
                        compute_p[1] += + compute
                    else:
                        compute_p[1] += load_p + compute
                if(compute_p[0] > compute_p[1]):
                    c_p = compute_p[0]
                else:
                    c_p = compute_p[1]
            for num in range(0,int(data[i]["save_num"][j])):
                save = data[i]["save_list"][j]/bw
                if(c_p < save_p):
                    save_p += save
                else:
                    save_p += c_p + save
                total_p = save_p    
    return total_p
         

if __name__ == '__main__':

    data = read("/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/fused.yaml")

    print(pipeline(data))


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