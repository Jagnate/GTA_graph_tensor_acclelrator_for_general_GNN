import yaml
import math

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

#实际存储大小为tile_size[i]*size_per_feature(size_per_feature*4)
#tile_size[i]为多少个数，多少个结点/边
def create_list(path_r, path_w, op_list, tile_size):

    with open(path_r, 'r') as file:
        data = yaml.safe_load(file)

    with open(path_w, 'r') as file:
            res = yaml.safe_load(file)

    for i in range(0,len(op_list)): #遍历大块
        w_list = [] #byte
        load_list = [] #byte = tile_size[i]*size_per_feature
        load_num = [] #num，循环几次，几个块
        compute_list = [] #byte = tile_size[i]*size_per_feature
        compute_num = [] #num，循环几次，几个块
        compute_type = [] #[0,1] 0:mm; 1:ele-wise
        save_list = [] #byte = tile_size[i]*size_per_feature
        save_num = [] #num，循环几次，几个块
        multi_input = []
        if(len(op_list[i]) == 1):
            tile_num = math.ceil(data[0]["INPUT"]["feature_number"][0]/tile_size[i])
            if data[op_list[i][0]]["TYPE"] == "scatter":
                w_list.append(0)
                compute_type.append(-1)
                compute_list.append(0)
                compute_num.append(0)
                multi_input.append(0)
                #load
                if data[op_list[i][0]]["ORDER"] == "R":
                    for k in data[op_list[i][0]]["INPUT"]["size_per_feature"]:
                        load_list.append(tile_size[i]*k)
                        load_num.append(tile_num)
                        save_list.append(tile_size[i]*k)
                        save_num.append(math.ceil(data[op_list[i][0]]["OUTPUT"]["output_number"])/tile_size[i])
                else: #顺序不一致的时候，一个一个scatter
                    for k in data[op_list[i][0]]["INPUT"]["size_per_feature"]:
                        load_list.append(1*k)
                        load_num.append(data[op_list[i][0]]["INPUT"]["feature_number"][0])
                        save_list.append(1*k)
                        save_num.append(data[op_list[i][0]]["OUTPUT"]["output_number"])
            #GATHER
            elif data[op_list[i][0]]["TYPE"] == "gather":
                w_list.append(0)
                multi_input.append(0)
                #load
                if data[op_list[i][0]]["ORDER"] == "R":
                    for k in data[0]["INPUT"]["size_per_feature"]:
                        load_list.append(tile_size[i]*k)
                        load_num.append(tile_num)
                        compute_num.append(tile_num)
                        save_list.append(tile_size[i]*k)
                        save_num.append(math.ceil(data[op_list[i][0]]["OUTPUT"]["output_number"])/tile_size[i])
                else: #顺序不一致的时候，一个一个gather
                    for k in data[op_list[i][0]]["INPUT"]["size_per_feature"]:
                        load_list.append(1*k)
                        load_num.append(data[op_list[i][0]]["INPUT"]["feature_number"][0])
                        compute_num.append(data[op_list[i][0]]["INPUT"]["feature_number"][0])
                        save_list.append(1*k)
                        save_num.append(data[op_list[i][0]]["OUTPUT"]["output_number"])
                #compute - plus(ele-wise)
                compute_type.append(1)
                compute_list.append(tile_size[i])
            #APPLY
            else:
                multi_input.append(0)
                #权重读取
                if data[op_list[i][0]]["INPUT"]["input_nong_num"] != 0:
                    compute_type.append(0)
                    weight = 0
                    for k in data[op_list[i][0]]["INPUT"]["input_size"]:
                        weight = weight + k
                    w_list.append(weight)
                else:
                    compute_type.append(1)
                    w_list.append(0)
                #特征读取
                if data[op_list[i][0]]["INPUT"]["input_g_num"] == 1:
                    load_list.append(tile_size[i]*data[op_list[i][0]]["INPUT"]["size_per_feature"][0])
                    load_num.append(tile_num)
                else:
                    multi_input[len(multi_input)-1] = 1
                    load_list.append(tile_size[i]*data[op_list[i][0]]["INPUT"]["size_per_feature"][0]*2)
                    load_num.append(tile_num)
                #计算
                if compute_type[len(compute_type)-1] == 0: #mm
                    compute_list.append(data[op_list[i][0]]["INPUT"]["input_size"][0]*tile_size[i]/(data[op_list[i][0]]["OUTPUT"]["size_per_feature"]))
                    compute_num.append(tile_num)
                else: #element-wise
                    #for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
                    compute_list.append(tile_size[i])
                    compute_num.append(tile_num)
                save_list.append(tile_size[i]*data[op_list[i][len(op_list[i])-1]]["OUTPUT"]["size_per_feature"])
                save_num.append(tile_num)
        else:
            #****第一个op特判断*****
            tile_num = math.ceil(data[op_list[i][0]]["INPUT"]["feature_number"][0]/tile_size[i])
            if data[op_list[i][0]]["TYPE"] == "scatter":
                multi_input.append(0)
                w_list.append(0)
                compute_type.append(-1)
                compute_list.append(0)
                compute_num.append(0)
                save_list.append(0)
                save_num.append(0)
                #load
                if data[op_list[i][0]]["ORDER"] == "R":
                    for k in data[op_list[i][0]]["INPUT"]["size_per_feature"]:
                        load_list.append(tile_size[i]*k)
                        load_num.append(tile_num)
                else: #顺序不一致的时候，一个一个scatter
                    for k in data[op_list[i][0]]["INPUT"]["size_per_feature"]:
                        load_list.append(1*k)
                        load_num.append(data[op_list[i][0]]["INPUT"]["feature_number"][0])
            #GATHER
            elif data[op_list[i][0]]["TYPE"] == "gather":
                multi_input.append(0)
                w_list.append(0)
                #load
                if data[op_list[i][0]]["ORDER"] == "R":
                    for k in data[0]["INPUT"]["size_per_feature"]:
                        load_list.append(tile_size[i]*k)
                        load_num.append(tile_num)
                        compute_num.append(tile_num)
                else: #顺序不一致的时候，一个一个gather
                    for k in data[op_list[i][0]]["INPUT"]["size_per_feature"]:
                        load_list.append(1*k)
                        load_num.append(data[op_list[i][0]]["INPUT"]["feature_number"][0])
                        compute_num.append(data[op_list[i][0]]["INPUT"]["feature_number"][0])
                #compute - plus(ele-wise)
                compute_type.append(1)
                compute_list.append(tile_size[i])
                save_list.append(0)
                save_num.append(0)
            #APPLY
            else:
                multi_input.append(0)
                #权重读取
                if data[op_list[i][0]]["INPUT"]["input_nong_num"] != 0:
                    compute_type.append(0)
                    weight = 0
                    for k in data[op_list[i][0]]["INPUT"]["input_size"]:
                        weight = weight + k
                    w_list.append(weight)
                else:
                    compute_type.append(1)
                    w_list.append(0)
                #特征读取
                if data[op_list[i][0]]["INPUT"]["input_g_num"] == 1:
                    load_list.append(tile_size[i]*data[op_list[i][0]]["INPUT"]["size_per_feature"][0])
                    load_num.append(tile_num)
                else:
                    multi_input[len(multi_input)-1] = 1
                    load_list.append(tile_size[i]*data[op_list[i][0]]["INPUT"]["size_per_feature"][0]*2)
                    load_num.append(tile_num)
                #计算
                if compute_type[len(compute_type)-1] == 0: #mm
                    compute_list.append(data[op_list[i][0]]["INPUT"]["input_size"][0]*tile_size[i]/(data[op_list[i][0]]["OUTPUT"]["size_per_feature"]))
                    compute_num.append(tile_num)
                else: #element-wise
                    #for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
                    compute_list.append(tile_size[i])
                    compute_num.append(tile_num)
                save_list.append(0)
                save_num.append(0)
            #****后续op****
            if(len(op_list[i])>2):
                for j in range(1,len(op_list[i])-1): #遍历大块里的op
                    tile_num = math.ceil(data[op_list[i][j]]["INPUT"]["feature_number"][0]/tile_size[i])
                    #SCATTER
                    if data[op_list[i][j]]["TYPE"] == "scatter":
                        multi_input.append(0)
                        w_list.append(0)
                        compute_type.append(-1)
                        compute_list.append(0)
                        compute_num.append(0)
                        save_list.append(0)
                        save_num.append(0)
                        load_num.append(0)
                        load_list.append(0)
                    #GATHER
                    elif data[op_list[i][j]]["TYPE"] == "gather":
                        multi_input.append(0)
                        w_list.append(0)
                        load_list.append(0)
                        load_num.append(0)
                        save_list.append(0)
                        save_num.append(0)
                        if data[0]["ORDER"] == "R":
                            for k in data[0]["INPUT"]["size_per_feature"]:
                                compute_num.append(tile_num)
                        else: #顺序不一致的时候，一个一个gather
                            for k in data[op_list[i][j]]["INPUT"]["size_per_feature"]:
                                compute_num.append(data[0]["INPUT"]["feature_number"][0])
                        #compute - plus(ele-wise)
                        compute_type.append(1)
                        compute_list.append(tile_size[i])
                    #APPLY
                    else:
                        multi_input.append(0)
                        #权重读取
                        if data[op_list[i][j]]["INPUT"]["input_nong_num"] != 0:
                            compute_type.append(0)
                            weight = 0
                            for k in data[op_list[i][j]]["INPUT"]["input_size"]:
                                weight = weight + k
                            w_list.append(weight)
                        else:
                            compute_type.append(1)
                            w_list.append(0)
                        #特征读取
                        if data[op_list[i][j]]["INPUT"]["intpu_g_num"] == 1:
                            if(data[data[op_list[i][0]]["INPUT"]["input_g_list"][0]]["TYPE"]=="scatter"):
                                load_list.append(tile_size[i]*data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                                load_num.append(tile_num)
                            else:
                                load_list.append(0)
                                load_num.append(0)
                        else:
                            multi_input[len(multi_input)-1] = 1
                            if(data[data[op_list[i][0]]["INPUT"]["input_g_list"][0]]["TYPE"]=="scatter"):
                                #直接x2
                                load_list.append(tile_size[i]*data[op_list[i][j]]["INPUT"]["size_per_feature"][0]*2)
                                load_num.append(tile_num)
                            else:
                                load_list.append(0)
                                load_num.append(0)
                        #计算
                        if compute_type[len(compute_type)-1] == 0: #mm
                            compute_list.append(data[op_list[i][j]]["INPUT"]["input_size"][0]*tile_size[i][i]/(data[op_list[i][j]]["OUTPUT"]["size_per_feature"]))
                            compute_num.append(tile_num)
                        else: #element-wise
                            #for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
                            compute_list.append(tile_size[i])
                            compute_num.append(tile_num)
                        save_list.append(0)
                        save_num.append(0)
            #****最后一个op特判断****
            tile_num = math.ceil(data[op_list[i][len(op_list[i])-1]]["INPUT"]["feature_number"][0]/tile_size[i])
            if data[op_list[i][len(op_list[i])-1]]["TYPE"] == "scatter":
                multi_input.append(0)
                w_list.append(0)
                compute_type.append(-1)
                compute_list.append(0)
                compute_num.append(0)
                load_list.append(0)
                load_num.append(0)
                #save
                if data[op_list[i][len(op_list[i])-1]]["ORDER"] == "R":
                    for k in data[op_list[i][len(op_list[i])-1]]["INPUT"]["size_per_feature"]:
                        save_list.append(tile_size[i]*k)
                        save_num.append(math.ceil(data[op_list[i][len(op_list[i])-1]]["OUTPUT"]["output_number"])/tile_size[i])
                else: #顺序不一致的时候，一个一个scatter
                    for k in data[op_list[i][len(op_list[i])-1]]["INPUT"]["size_per_feature"]:
                        save_list.append(1*k)
                        save_num.append(data[op_list[i][len(op_list[i])-1]]["OUTPUT"]["output_number"])
            #GATHER
            elif data[op_list[i][len(op_list[i])-1]]["TYPE"] == "gather":
                multi_input.append(0)
                w_list.append(0)
                load_list.append(0)
                load_num.append(0)
                #load
                if data[op_list[i][len(op_list[i])-1]]["ORDER"] == "R":
                    for k in data[op_list[i][len(op_list[i])-1]]["INPUT"]["size_per_feature"]:
                        compute_num.append(tile_num)
                        save_list.append(tile_size[i]*k)
                        save_num.append(math.ceil(data[op_list[i][len(op_list[i])-1]]["OUTPUT"]["output_number"])/tile_size[i])
                else: #顺序不一致的时候，一个一个gather
                    for k in data[op_list[i][len(op_list[i])-1]]["INPUT"]["size_per_feature"]:
                        compute_num.append(data[op_list[i][len(op_list[i])-1]]["INPUT"]["feature_number"][0])
                        save_list.append(1*k)
                        save_num.append(data[op_list[i][len(op_list[i])-1]]["OUTPUT"]["output_number"])
                #compute - plus(ele-wise)
                compute_type.append(1)
                compute_list.append(tile_size[i])
            #APPLY
            else:
                multi_input.append(0)
                #权重读取
                if data[op_list[i][len(op_list[i])-1]]["INPUT"]["input_nong_num"] != 0:
                    compute_type.append(0)
                    weight = 0
                    for k in data[op_list[i][len(op_list[i])-1]]["INPUT"]["input_size"]:
                        weight = weight + k
                    w_list.append(weight)
                else:
                    compute_type.append(1)
                #特征读取
                if data[op_list[i][j]]["INPUT"]["intpu_g_num"] == 1:
                    if(data[data[op_list[i][len(op_list[i])-1]]["INPUT"]["input_g_list"][0]]["TYPE"]=="scatter"):
                        load_list.append(tile_size[i]*data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                        load_num.append(tile_num)
                    else:
                        load_list.append(0)
                        load_num.append(0)
                else:
                    multi_input[len(multi_input)-1] = 1
                    if(data[data[op_list[i][len(op_list[i])-1]]["INPUT"]["input_g_list"][0]]["TYPE"]=="scatter"):
                        #直接x2
                        load_list.append(tile_size[i]*data[op_list[i][j]]["INPUT"]["size_per_feature"][0]*2)
                        load_num.append(tile_num)
                    else:
                        load_list.append(0)
                        load_num.append(0)
                #计算
                if compute_type[len(compute_type)-1] == 0: #mm
                    compute_list.append(data[op_list[i][len(op_list[i])-1]]["INPUT"]["input_size"][0]*tile_size[i]/(data[len(op_list[i])-1]["OUTPUT"]["size_per_feature"]))
                    compute_num.append(tile_num)
                else: #element-wise
                    #for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
                    compute_list.append(tile_size[i])
                    compute_num.append(tile_num)
                save_list.append(tile_size[i]*data[op_list[i][len(op_list[i])-1]]["OUTPUT"]["size_per_feature"])
                save_num.append(tile_num)
        print(i)

        res[i]["load_list"] = load_list
        res[i]["load_num"] = load_num
        res[i]["compute_list"] = compute_list
        res[i]["compute_num"] = compute_num
        res[i]["compute_type"] = compute_type
        res[i]["save_list"] = save_list
        res[i]["save_num"] = save_num
        res[i]["w_list"] = w_list
        res[i]["multi_input"] = multi_input

    with open(path_w, 'w') as file:
        yaml.safe_dump(res, file)
        
if __name__ == '__main__':
    create_list("/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/simpletest.yaml", "/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/fused.yaml",[[0],[1],[2],[3],[4],[5],[6],[7],[8]], [1354,1354,1354,1354,1354,1354,1354,1354,1354])
