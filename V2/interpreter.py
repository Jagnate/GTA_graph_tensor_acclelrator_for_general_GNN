import yaml
import math

def read(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data,Loader=yaml.FullLoader)
        return result

    
#实际存储大小为tile_size[i]*size_per_feature(size_per_feature*4)
#tile_size[i]为多少个数，多少个结点/边
def create_list(dataset,path_r, path_w, op_list, tile_size, node_num):

    res = {}

    with open(path_r, 'r') as file:
        data = yaml.safe_load(file)

    count = 0

    for i in range(0,len(op_list)): #遍历大块
        for j in range(0,len(op_list[i])): #大块里有几个op
            w_list = [] #byte
            load_list = [] #byte = tile_size[i]*size_per_feature
            load_shape = [] #load一次load的维度
                            #load_shape = [[a,b],[c,d]] 表示第一组load的数据，load一次需要load a组数据，每组数据有b个node/edge
            
            compute_list = [] #byte = tile_size[i]*size_per_feature
            compute_shape = [] #compute一次compute的维度
            compute_type = [] #[0,1] 0:mm; 1:ele-wise

            save_list = [] #byte = tile_size[i]*size_per_feature
            save_shape = [] #save一次的维度
            type = ""
            sparse_path = ""

            isR = 1
            times_1 = math.ceil(node_num/tile_size[i])
            times_2 = node_num
            times_3 = tile_size[i]

            #SCATTER
            if data[op_list[i][j]]["TYPE"] == "scatter":
                w_list.append([0,0])
                compute_type.append(-1)
                compute_list.append(0)
                compute_shape.append(0)
                type = "scatter"
                sparse_path = '/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/data/adj_'+dataset+'_'+str(tile_size[i])+'_1.yaml'
                #load
                for k in range(0,len(data[op_list[i][j]]["INPUT"]["input_g_list"])):
                    if(data[op_list[i][j]]["INPUT"]["input_g_list"][k] not in op_list[i]):
                        load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][k])
                        if data[op_list[i][j]]["ORDER"] == "R":
                            load_shape.append([tile_size[i],1])     
                        else:
                            isR = 0
                            load_shape.append([1,1])
                    else:
                        load_list.append(0)
                        load_shape.append([0,0])
                #save
                if len(data[op_list[i][j]]["OUTPUT"]["output_list"]) == 0:
                    save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                    if(data[op_list[i][j]]["OUTPUT"]["output_list"][0] not in op_list[i]):
                        if data[op_list[i][j]]["ORDER"] == "R":
                            #save一次save nodenum组数
                            #每组中包括tile_size[i]个边
                            save_shape.append([node_num,tile_size[i]])
                        else:
                            isR = 0
                            save_shape.append([1,tile_size[i]])
                    else:
                        save_list.append(0)
                        save_shape.append([0,0])
                else:
                    for k in range(0,len(data[op_list[i][j]]["OUTPUT"]["output_list"])):
                        if(data[op_list[i][j]]["OUTPUT"]["output_list"][k] not in op_list[i]):
                            save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                            if data[op_list[i][j]]["ORDER"] == "R":
                                save_shape.append([node_num,tile_size[i]])
                            else:
                                isR = 0
                                save_shape.append([1,tile_size[i]])
                        else:
                            save_list.append(0)
                            save_shape.append([0,0])
            elif data[op_list[i][j]]["TYPE"] == "gather":
                w_list.append([0,0])
                type = "gather"
                sparse_path = '/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/data/adj_'+dataset+'_'+str(tile_size[i])+'_1.yaml'
                if len(data[op_list[i][j]]["INPUT"]["input_g_list"])==0:
                    if(data[op_list[i][j]]["INPUT"]["input_g_list"][k] not in op_list[i]):
                        if data[op_list[i][j]]["ORDER"] == "R":
                            load_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                            load_shape.append([tile_size[i],1])
                            load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                            load_shape.append([node_num,tile_size[i]])
                        else:
                            isR = 0
                            load_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                            load_shape.append([tile_size[i],1])
                            load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][0]*tile_size[i])
                            load_shape.append([node_num,tile_size[i]])
                    else:
                        load_list.append(0)
                        load_shape.append([0,0])
                else:
                    for k in range(0,len(data[op_list[i][j]]["INPUT"]["input_g_list"])):
                        if(data[op_list[i][j]]["INPUT"]["input_g_list"][k] not in op_list[i]):
                            if data[op_list[i][j]]["ORDER"] == "R":
                                load_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                                load_shape.append([tile_size[i],1])
                                load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][k])
                                load_shape.append([node_num,tile_size[i]])
                            else:
                                isR = 0
                                load_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                                load_shape.append([tile_size[i],1])
                                load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][k]*tile_size[i])
                                load_shape.append([node_num,tile_size[i]])
                        else:
                            load_list.append(0)
                            load_shape.append([0,0])
                #compute
                compute_type.append(1)
                compute_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                if data[op_list[i][j]]["ORDER"] == "R":
                    compute_shape.append([node_num,tile_size[i]])
                else:
                    isR = 0
                    compute_shape.append([1,tile_size[i]])
                #save
                if len(data[op_list[i][j]]["OUTPUT"]["output_list"]) == 0:
                    if(data[op_list[i][j]]["OUTPUT"]["output_list"][0] not in op_list[i]):
                        save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                        if data[op_list[i][j]]["ORDER"] == "R":
                            save_shape.append([1,tile_size[i]])
                        else:
                            isR = 0
                            save_shape.append([1,1])
                    else:
                        save_shape.append([0,0])
                        save_list.append(0)
                else:
                    for k in range(0,len(data[op_list[i][j]]["OUTPUT"]["output_list"])):
                        if(data[op_list[i][j]]["OUTPUT"]["output_list"][k] not in op_list[i]):
                            save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                            if data[op_list[i][j]]["ORDER"] == "R":
                                save_shape.append([1,tile_size[i]])
                            else:
                                isR = 0
                                save_shape.append([1,1])
                        else:
                            save_shape.append([0,0])
                            save_list.append(0)
            #APPLYNODE
            elif data[op_list[i][j]]["TYPE"] == "applynode":
                type = "applynode"
                #权重读取
                if data[op_list[i][j]]["INPUT"]["input_nong_num"] != 0:
                    compute_type.append(0)
                    for w in range(0,len(data[op_list[i][j]]["INPUT"]["input_size"])):
                        w_list.append([data[op_list[i][j]]["INPUT"]["input_size"][w]/data[op_list[i][j]]["INPUT"]["size_per_feature"][w],data[op_list[i][j]]["INPUT"]["size_per_feature"][w]])
                else:
                    w_list.append([0,0])
                    compute_type.append(1)
                #特征读取
                if(len(data[op_list[i][j]]["INPUT"]["input_g_list"])==0):
                    load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                    load_shape.append([1,1])
                else:  
                    for k in range(0,len(data[op_list[i][j]]["INPUT"]["input_g_list"])):
                        if(data[op_list[i][j]]["INPUT"]["input_g_list"][k] not in op_list[i]):
                            #load
                            load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][k])
                            load_shape.append([1,1])
                        else:
                            load_list.append(0)
                            load_shape.append([0,0])
                #compute
                if data[op_list[i][j]]["INPUT"]["input_nong_num"] != 0: #mm
                    #compute_list.append(data[op_list[i][j]]["INPUT"]["input_size"][j]*data[op_list[i][j]]["INPUT"]["size_per_feature"][k]/16)
                    #第一个0不能换成k
                    compute_list.append(data[op_list[i][j]]["INPUT"]["input_size"][0]/4)
                    compute_shape.append([1,1])
                else: #element-wise
                    #for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
                    compute_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                    compute_shape.append([1,1])
                #save
                if len(data[op_list[i][j]]["OUTPUT"]["output_list"]) == 0:
                    save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                    save_shape.append([1,1])
                else:
                    for k in range(0,len(data[op_list[i][j]]["OUTPUT"]["output_list"])):
                        if(data[op_list[i][j]]["OUTPUT"]["output_list"][k] not in op_list[i] or len(data[op_list[i][j]]["OUTPUT"]["output_list"])==0):
                            save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                            save_shape.append([1,1])
                        else:
                            save_list.append(0)
                            save_shape.append([0,0])
            #APPLYEDGE
            else:
                sparse_path = '/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/data/adj_'+dataset+'_'+str(tile_size[i])+'_1.yaml'
                type = "applyedge"
                #权重读取
                if data[op_list[i][j]]["INPUT"]["input_nong_num"] != 0:
                    compute_type.append(0)
                    for w in range(0,len(data[op_list[i][j]]["INPUT"]["input_size"])):
                        w_list.append([data[op_list[i][j]]["INPUT"]["input_size"][w]/data[op_list[i][j]]["INPUT"]["size_per_feature"][w],data[op_list[i][j]]["INPUT"]["size_per_feature"][w]])
                else:
                    w_list.append([0,0])
                    compute_type.append(1)
                #特征读取
                if(len(data[op_list[i][j]]["INPUT"]["input_g_list"])==0):
                    load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                    load_shape.append([node_num,tile_size[i]])
                else:
                    for k in range(0,len(data[op_list[i][j]]["INPUT"]["input_g_list"])):                
                        if(data[op_list[i][j]]["INPUT"]["input_g_list"][k] not in op_list[i]):
                            #load
                            load_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][k])
                            load_shape.append([node_num,tile_size[i]])
                        else:
                            load_list.append(0)
                            load_shape.append([0,0])
                #compute
                if data[op_list[i][j]]["INPUT"]["input_nong_num"] != 0: #mm
                    compute_list.append(data[op_list[i][j]]["INPUT"]["input_size"][0]/4)
                    compute_shape.append([node_num,tile_size[i]])
                else: #element-wise
                    #for k in range(0,data[op_list[i][j]]["INPUT"]["input_g_num"]):
                    compute_list.append(data[op_list[i][j]]["INPUT"]["size_per_feature"][0])
                    compute_shape.append([node_num,tile_size[i]])
                #save
                if len(data[op_list[i][j]]["OUTPUT"]["output_list"]) == 0:
                    save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                    save_shape.append([node_num,tile_size[i]])
                else:
                    for k in range(0,len(data[op_list[i][j]]["OUTPUT"]["output_list"])):
                        if(data[op_list[i][j]]["OUTPUT"]["output_list"][k] not in op_list[i] or len(data[op_list[i][j]]["OUTPUT"]["output_list"])==0):
                            save_list.append(data[op_list[i][j]]["OUTPUT"]["size_per_feature"])
                            save_shape.append([node_num,tile_size[i]])
                        else:
                            save_list.append(0)
                            save_shape.append([0,0])

            res[count] = {
                "OP_NO": op_list[i][j],
                "type": type,
                "load_list": load_list,
                "load_shape": load_shape,
                "compute_list": compute_list,
                "compute_shape": compute_shape,
                "compute_type": compute_type,
                "save_list": save_list,
                "save_shape": save_shape,
                "w_list": w_list,
                "times_1": times_1,
                "times_2": times_2,
                "times_3": times_3,
                "isR": isR,
                "sparse_path": sparse_path
            }

            count += 1
            
    with open(path_w, 'w') as file:
        yaml.safe_dump(res, file)
        
if __name__ == '__main__':

    node_num = 2708
    #node_num = 19717
    #node_num = 3327

    #op_list = [[0, 1, 2], [5], [4, 6, 7, 8, 9, 10], [3, 11, 12, 13]]
    #tile_size = [100, 2708, 2708, 579]
    #tile_size = [303, 19717, 579, 8738]
    #tile_size = [579, 3327, 3327, 3327]
    
    
    #op_list = [[0],[1],[2], [4,5,6], [7,8,9,10], [3, 11, 12, 13]]
    #tile_size = [2708,2708,2708,2708,2708,579]
    #tile_size = [19717, 19717, 19717, 19717, 16384, 579]
    #tile_size = [3327, 3327, 3327, 3327, 3327, 579]

    op_list = [[0],[1],[2], [4],[5],[6], [7],[8],[9],[10], [3], [11], [12], [13]]
    #tile_size = [2708,2708,2708,2708,2708,2708,2708,2708,2708,2708,2708,2708,2708,2708]
    #tile_size = [19717,19717,19717,19717,19717,19717,19717,19717,19717,19717,19717,19717,19717,19717]
    tile_size = [3327,3327,3327,3327,3327,3327,3327,3327,3327,3327,3327,3327,3327,3327]

    create_list("citeseer","/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/v1/GAT_Cora.yaml", "/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/v1/fused.yaml", op_list, tile_size, node_num)
