import yaml


def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

def cal_perform(op_list,tile_size,tile_num,bandwidth,parallesim,path):
    data = read(path)
    cycle = 0
    #record = []
    for i in range(0,len(op_list)):
        if(len(op_list[i])>1):
            weight = 0
            input_g = 0
            output = 0
            compute = 0
            for k in op_list[i]:
                if(tile_size[i]==1):
                    temp_input_g = 0
                    temp_weight = 0
                    for j in range(0,data[k]["INPUT"]["input_nong_num"]):
                        weight += data[k]["INPUT"]["input_size"][j]
                        temp_weight += data[k]["INPUT"]["input_size"][j]
                    if(data[k]["INPUT"]["input_g_num"]==1 and len(data[k]["INPUT"]["input_list"])==0):
                        input_g += data[k]["INPUT"]["size_per_feature"][0]*data[k]["INPUT"]["feature_number"][0]
                        temp_input_g += data[k]["INPUT"]["feature_number"][0]
                    else:
                        for j in range(0,data[k]["INPUT"]["input_g_num"]):
                            if(data[k]["INPUT"]["input_list"][j] not in op_list[i]):
                                input_g += data[k]["INPUT"]["size_per_feature"][j]*data[k]["INPUT"]["feature_number"][j]
                            temp_input_g += data[k]["INPUT"]["feature_number"][j]
                    if(len(data[k]["OUTPUT"]["output_list"])==0):
                        output += data[k]["OUTPUT"]["output_number"]*data[k]["OUTPUT"]["size_per_feature"]
                    else:
                        for j in range(0,len(data[k]["OUTPUT"]["output_list"])):
                            if(data[k]["OUTPUT"]["output_list"][j] not in op_list[i]):
                                output += data[k]["OUTPUT"]["output_number"]*data[k]["OUTPUT"]["size_per_feature"]
                    if(temp_weight == 0 and data[k]["TYPE"]!="scatter"):
                        temp_weight = 1
                    compute += temp_weight*temp_input_g
                else:
                    temp_input_g = 0
                    temp_weight = 0
                    for j in range(0,data[k]["INPUT"]["input_nong_num"]):
                        weight += data[k]["INPUT"]["input_size"][j]
                        temp_weight += data[k]["INPUT"]["input_size"][j]
                    if(len(data[k]["INPUT"]["input_list"])==0):
                        input_g += tile_size[i]
                        temp_input_g += tile_size[i]
                    else:
                        for j in range(0,data[k]["INPUT"]["input_g_num"]):
                            if(data[k]["INPUT"]["input_list"][j] not in op_list[i]):
                                input_g += tile_size[i]
                            else:
                                temp_input_g += tile_size[i]
                    if(len(data[k]["OUTPUT"]["output_list"])==0):
                        output += data[k]["OUTPUT"]["output_number"]*data[k]["OUTPUT"]["size_per_feature"]
                    else:
                        for j in range(0,len(data[k]["OUTPUT"]["output_list"])):
                            if(data[k]["OUTPUT"]["output_list"][j] not in op_list[i]):
                                output += data[k]["OUTPUT"]["output_number"]*data[k]["OUTPUT"]["size_per_feature"]
                    if(temp_weight == 0 and data[k]["TYPE"]!="scatter"):
                        temp_weight = 4
                    compute += temp_weight*temp_input_g/4
            temp_cycle = pipline(weight,input_g,output,tile_num[i],bandwidth,parallesim,compute)
            # record.append(temp_cycle)
            cycle += temp_cycle
        else:
            weight = 0
            input_g = 0
            output = 0
            compute = 0
            if(data[op_list[i][0]]["INPUT"]["input_nong_num"]==0):
                for j in range(0,data[op_list[i][0]]["INPUT"]["input_g_num"]):
                    input_g += data[op_list[i][0]]["INPUT"]["size_per_feature"][j]*data[op_list[i][0]]["INPUT"]["feature_number"][j]
                compute = input_g
            else:
                for j in range(0,data[op_list[i][0]]["INPUT"]["input_nong_num"]):
                    weight += data[op_list[i][0]]["INPUT"]["input_size"][j]
                for j in range(0,data[op_list[i][0]]["INPUT"]["input_g_num"]):
                    compute += data[op_list[i][0]]["INPUT"]["input_size"][j]*data[op_list[i][0]]["INPUT"]["feature_number"][j]
            output = data[op_list[i][0]]["OUTPUT"]["output_number"]*data[op_list[i][0]]["OUTPUT"]["size_per_feature"]
                #     print(data[op_list[i][0]]["INPUT"]["input_size"][j])
                #     print(data[op_list[i][0]]["INPUT"]["feature_number"][j])
                # print(compute)
            # for j in range(0,data[op_list[i][0]]["INPUT"]["input_nong_num"]):
            #     weight += data[op_list[i][0]]["INPUT"]["input_size"][j]
            # for j in range(0,data[op_list[i][0]]["INPUT"]["input_g_num"]):
            #     if(weight==0):
            #         input_g += data[op_list[i][0]]["INPUT"]["size_per_feature"][j]*data[op_list[i][0]]["INPUT"]["feature_number"][j]
            #     else:
            #         input_g += data[op_list[i][0]]["INPUT"]["feature_number"][j]
            # output = data[op_list[i][0]]["OUTPUT"]["output_number"]*data[op_list[i][0]]["OUTPUT"]["size_per_feature"]
            # if(weight == 0 and data[op_list[i][0]]["TYPE"]!="scatter"):
            #     compute = input_g/4
            # else:
            #     compute = weight*input_g/4
            temp_cycle = pipline(weight,input_g,output,tile_num[i],bandwidth,parallesim,compute)
            #record.append(temp_cycle)
            cycle += temp_cycle
    return cycle*(10**3)


def pipline(weight,input_g,output,tile_num,bandwidth,parallesim,compute):
    load_w_duration = weight/bandwidth
    load_g_duration = 0
    compute_duration = 0
    save_duration = 0

    left_load_point = 0
    right_load_point = 0

    left_compute_point = 0
    right_compute_point = 0

    right_save_point = 0

    right_op_point = 0

    # if(tile_num!=1):
    #     compute = weight*tile_size/4
    #     input_g = tile_size
    #     output = tile_size

    for i in range(0,tile_num):

        left_load_point = right_load_point
        load_g_duration = input_g/bandwidth
        if(i==0):
            right_load_point = left_load_point + load_w_duration + load_g_duration
        else:
            right_load_point = left_load_point + load_g_duration
        
        if(right_load_point > right_op_point):
            right_op_point = right_load_point

        #print("Load: time:",i,"Latency:",right_op_point)
        
        if(right_compute_point < right_load_point):
            left_compute_point = right_load_point
        else:
            left_compute_point = right_compute_point
        compute_duration = compute/parallesim
        right_compute_point = left_compute_point + compute_duration
        if(right_compute_point > right_op_point):
            right_op_point = right_compute_point

        #print("Compute:",right_op_point)

        if(right_save_point < right_compute_point):
            left_save_point = right_compute_point
        else:
            left_save_point = right_save_point

        save_duration = output/bandwidth
        right_save_point = left_save_point + save_duration
        if(right_save_point > right_op_point):
            right_op_point = right_save_point

        #print("Save:",right_op_point)

    return right_op_point
        
#op_list,tile_size,tile_num,bandwidth,parallesim,path
if __name__ == '__main__':
    path = "/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/GAT_Cora.yaml"
    bandwidth = 128*1024*1024*1024
    parallesim = 256*4*(10**(9))
    # op_list = [[0,1]]
    # tile_size = [1]
    # tile_num = [1]
    
    # print("不融合不分块    Latency：",cal_perform(op_list,tile_size,tile_num,bandwidth,parallesim,path))



#[[[0], [1], [2], [3], [4], [5], [11], [6], [7], [8], [9], [10], [12], [13]], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],57527120]
    op_list = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]]
    tile_size = [1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1]
    tile_num = [1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1]
    
    print("不融合不分块    Latency：",cal_perform(op_list,tile_size,tile_num,bandwidth,parallesim,path))

# # [list([[0, 1, 2, 4, 6, 7, 8, 9, 10], [3, 11, 12, 13], [5]])
# #  list([2083120, 2097024, 0]) list([376, 2032, 1]) list([29, 6, 1])
# #  20841936]
#     op_list_1 = [[0, 1, 2, 4, 6, 7, 8, 9, 10], [3, 11, 12, 13], [5]]
#     tile_size_1 = [376, 2032, 1]
#     tile_num_1 = [29, 6, 1]


# # [list([[0, 1, 2], [3, 11, 12, 13], [4, 6, 7, 8, 9, 10], [5]])
# #  list([2092640, 2097024, 2097152, 0]) list([1212, 2032, 32768, 1])
# #  list([66, 39, 3, 1]) 74076080]
#     op_list_1 = [[0, 1, 2], [3, 11, 12, 13], [4, 6, 7, 8, 9, 10], [5]]
#     tile_size_1 = [1212, 2032, 32768, 1]
#     tile_num_1 = [66, 39, 3, 1]

# [list([[0, 1, 2], [3, 11, 12, 13], [4, 6, 7, 8, 9, 10], [5]])
#  list([2090064, 2097024, 2097152, 0]) list([24, 2032, 32768, 1])
#  list([555, 7, 1, 1]) 56796004]
    op_list_1 = [[0, 1, 2], [3, 11, 12, 13], [4, 6, 7, 8, 9, 10], [5]]
    tile_size_1 = [24, 2032, 32768, 1]
    tile_num_1 = [555, 7, 1, 1]

    
    print("融合分块最佳结果Latency：",cal_perform(op_list_1,tile_size_1,tile_num_1,bandwidth,parallesim,path))
    print("------------------------------------")
    print("不融合不分块   访存量：90722676")
    print("融合分块最佳结果访存量：56796004")



#BW = 128GB/s
#1ns 一个 cycle 1个cycle128byte
#FP32 4 byte x size_per_Feature x feature_number
#latency x30 每个读写都有30ns

#缓存大小 = 2MB = 2 x 1024 x 1024 byte
#权重的数量 4byte

#并行层：256=16x16
#2层
