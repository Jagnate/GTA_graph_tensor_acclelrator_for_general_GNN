import yaml

def modify_yaml(path,node_num,edge_num,size_per_feature,num_of_classes):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    # for key, value in params.items():
    #     for i in data:
    #         if key in i["INPUT"]:
    #             i["INPUT"][key] = value
    data[0]["INPUT"]["size_per_feature"]    = [size_per_feature*4]
    data[0]["INPUT"]["feature_number"]      = [node_num]
    data[0]["INPUT"]["input_size"]          = [128*size_per_feature*4]
    data[0]["OUTPUT"]["size_per_feature"]   = 128*4
    data[0]["OUTPUT"]["output_number"]      = node_num

    data[1]["INPUT"]["size_per_feature"]    = [128*4]
    data[1]["INPUT"]["feature_number"]      = [node_num]
    data[1]["INPUT"]["input_size"]          = [4*128*4]
    data[1]["OUTPUT"]["size_per_feature"]   = 4*4
    data[1]["OUTPUT"]["output_number"]      = node_num

    data[2]["INPUT"]["size_per_feature"]    = [128*4]
    data[2]["INPUT"]["feature_number"]      = [node_num]
    data[2]["INPUT"]["input_size"]          = [4*128*4]
    data[2]["OUTPUT"]["size_per_feature"]   = 4*4
    data[2]["OUTPUT"]["output_number"]      = node_num

    data[3]["INPUT"]["size_per_feature"]    = [128*4]
    data[3]["INPUT"]["feature_number"]      = [node_num]
    data[3]["INPUT"]["input_size"]          = []
    data[3]["OUTPUT"]["size_per_feature"]   = 128*4
    data[3]["OUTPUT"]["output_number"]      = edge_num

    data[4]["INPUT"]["size_per_feature"]    = [4*4]
    data[4]["INPUT"]["feature_number"]      = [node_num]
    data[4]["INPUT"]["input_size"]          = []
    data[4]["OUTPUT"]["size_per_feature"]   = 4*4
    data[4]["OUTPUT"]["output_number"]      = edge_num

    data[5]["INPUT"]["size_per_feature"]    = [4*4]
    data[5]["INPUT"]["feature_number"]      = [node_num]
    data[5]["INPUT"]["input_size"]          = []
    data[5]["OUTPUT"]["size_per_feature"]   = 4*4
    data[5]["OUTPUT"]["output_number"]      = edge_num

    data[6]["INPUT"]["size_per_feature"]    = [4*4,4*4]
    data[6]["INPUT"]["feature_number"]      = [edge_num,edge_num]
    data[6]["INPUT"]["input_size"]          = []
    data[6]["OUTPUT"]["size_per_feature"]   = 4*4
    data[6]["OUTPUT"]["output_number"]      = edge_num

    data[7]["INPUT"]["size_per_feature"]    = [4*4]
    data[7]["INPUT"]["feature_number"]      = [edge_num]
    data[7]["INPUT"]["input_size"]          = []
    data[7]["OUTPUT"]["size_per_feature"]   = 4*4
    data[7]["OUTPUT"]["output_number"]      = edge_num

    data[8]["INPUT"]["size_per_feature"]    = [4*4]
    data[8]["INPUT"]["feature_number"]      = [edge_num]
    data[8]["INPUT"]["input_size"]          = []
    data[8]["OUTPUT"]["size_per_feature"]   = 4*4
    data[8]["OUTPUT"]["output_number"]      = node_num

    data[9]["INPUT"]["size_per_feature"]    = [4*4,4*4]
    data[9]["INPUT"]["feature_number"]      = [edge_num,edge_num]
    data[9]["INPUT"]["input_size"]          = []
    data[9]["OUTPUT"]["size_per_feature"]   = 4*4
    data[9]["OUTPUT"]["output_number"]      = node_num

    data[10]["INPUT"]["size_per_feature"]    = [4*4]
    data[10]["INPUT"]["feature_number"]      = [node_num]
    data[10]["INPUT"]["input_size"]          = []
    data[10]["OUTPUT"]["size_per_feature"]   = 4*4
    data[10]["OUTPUT"]["output_number"]      = edge_num

    data[11]["INPUT"]["size_per_feature"]    = [4*4,128*4]
    data[11]["INPUT"]["feature_number"]      = [edge_num,edge_num]
    data[11]["INPUT"]["input_size"]          = []
    data[11]["OUTPUT"]["size_per_feature"]   = 128*4
    data[11]["OUTPUT"]["output_number"]      = edge_num

    data[12]["INPUT"]["size_per_feature"]    = [128*4]
    data[12]["INPUT"]["feature_number"]      = [edge_num]
    data[12]["INPUT"]["input_size"]          = []
    data[12]["OUTPUT"]["size_per_feature"]   = 128*4
    data[12]["OUTPUT"]["output_number"]      = node_num

    data[13]["INPUT"]["size_per_feature"]    = [128*4]
    data[13]["INPUT"]["feature_number"]      = [edge_num]
    data[13]["INPUT"]["input_size"]          = []
    data[13]["OUTPUT"]["size_per_feature"]   = 128*4
    data[13]["OUTPUT"]["output_number"]      = node_num

    # data[14]["INPUT"]["size_per_feature"]    = [size_per_feature*4]
    # data[14]["INPUT"]["feature_number"]      = [node_num]
    # data[14]["INPUT"]["input_size"]          = [num_of_classes*edge_num*4]
    # data[14]["OUTPUT"]["size_per_feature"]   = num_of_classes*4
    # data[14]["OUTPUT"]["output_number"]      = node_num

    # data[15]["INPUT"]["size_per_feature"]    = [num_of_classes*4]
    # data[15]["INPUT"]["feature_number"]      = [node_num]
    # data[15]["INPUT"]["input_size"]          = [4*num_of_classes*4]
    # data[15]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[15]["OUTPUT"]["output_number"]      = node_num

    # data[16]["INPUT"]["size_per_feature"]    = [num_of_classes*4]
    # data[16]["INPUT"]["feature_number"]      = [node_num]
    # data[16]["INPUT"]["input_size"]          = [4*num_of_classes*4]
    # data[16]["OUTPUT"]["size_per_feature"]   = 4
    # data[16]["OUTPUT"]["output_number"]      = node_num

    # data[17]["INPUT"]["size_per_feature"]    = [num_of_classes*4]
    # data[17]["INPUT"]["feature_number"]      = [node_num]
    # data[17]["INPUT"]["input_size"]          = []
    # data[17]["OUTPUT"]["size_per_feature"]   = num_of_classes*4
    # data[17]["OUTPUT"]["output_number"]      = edge_num

    # data[18]["INPUT"]["size_per_feature"]    = [4*4]
    # data[18]["INPUT"]["feature_number"]      = [node_num]
    # data[18]["INPUT"]["input_size"]          = []
    # data[18]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[18]["OUTPUT"]["output_number"]      = edge_num

    # data[19]["INPUT"]["size_per_feature"]    = [4*4]
    # data[19]["INPUT"]["feature_number"]      = [node_num]
    # data[19]["INPUT"]["input_size"]          = []
    # data[19]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[19]["OUTPUT"]["output_number"]      = edge_num

    # data[20]["INPUT"]["size_per_feature"]    = [4*4,4*4]
    # data[20]["INPUT"]["feature_number"]      = [edge_num,edge_num]
    # data[20]["INPUT"]["input_size"]          = []
    # data[20]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[20]["OUTPUT"]["output_number"]      = edge_num

    # data[21]["INPUT"]["size_per_feature"]    = [4*4]
    # data[21]["INPUT"]["feature_number"]      = [edge_num]
    # data[21]["INPUT"]["input_size"]          = []
    # data[21]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[21]["OUTPUT"]["output_number"]      = edge_num

    # data[22]["INPUT"]["size_per_feature"]    = [4*4]
    # data[22]["INPUT"]["feature_number"]      = [edge_num]
    # data[22]["INPUT"]["input_size"]          = []
    # data[22]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[22]["OUTPUT"]["output_number"]      = node_num

    # data[23]["INPUT"]["size_per_feature"]    = [4*4,4*4]
    # data[23]["INPUT"]["feature_number"]      = [edge_num,edge_num]
    # data[23]["INPUT"]["input_size"]          = []
    # data[23]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[23]["OUTPUT"]["output_number"]      = node_num

    # data[24]["INPUT"]["size_per_feature"]    = [4*4]
    # data[24]["INPUT"]["feature_number"]      = [node_num]
    # data[24]["INPUT"]["input_size"]          = []
    # data[24]["OUTPUT"]["size_per_feature"]   = 4*4
    # data[24]["OUTPUT"]["output_number"]      = edge_num

    # data[25]["INPUT"]["size_per_feature"]    = [4,num_of_classes*4]
    # data[25]["INPUT"]["feature_number"]      = [edge_num,edge_num]
    # data[25]["INPUT"]["input_size"]          = []
    # data[25]["OUTPUT"]["size_per_feature"]   = num_of_classes*4
    # data[25]["OUTPUT"]["output_number"]      = edge_num

    # data[26]["INPUT"]["size_per_feature"]    = [num_of_classes*4]
    # data[26]["INPUT"]["feature_number"]      = [edge_num]
    # data[26]["INPUT"]["input_size"]          = []
    # data[26]["OUTPUT"]["size_per_feature"]   = num_of_classes*4
    # data[26]["OUTPUT"]["output_number"]      = node_num

    # data[27]["INPUT"]["size_per_feature"]    = [num_of_classes*4]
    # data[27]["INPUT"]["feature_number"]      = [edge_num]
    # data[27]["INPUT"]["input_size"]          = []
    # data[27]["OUTPUT"]["size_per_feature"]   = num_of_classes*4
    # data[27]["OUTPUT"]["output_number"]      = node_num

    with open(path, 'w') as file:
        yaml.safe_dump(data, file)

# node_num = 2708
# edge_num = 10556
# size_per_feature = 1433
# num_of_classes = 7
# node_num = 19717
# edge_num = 88651
# size_per_feature = 500
# num_of_classes = 7
node_num = 3327
edge_num = 9228
size_per_feature = 3703
num_of_classes = 7
modify_yaml("/Users/sijin/Desktop/RA/MPAD/Eva/Compiler/GAT.yaml",node_num,edge_num,size_per_feature,num_of_classes)
