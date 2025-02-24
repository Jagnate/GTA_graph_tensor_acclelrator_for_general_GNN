
#二分查找，找到访存量小于threshold的最大的元素的index
def binary_search(arr, value):
    left, right = 0, len(arr) - 1
    result = -1  # 如果没有找到符合条件的元素，返回-1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid][2] < value:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result

#删除这个index之后的所有元素，所以需要rw是从小到大排序
def remove_elements_after_index(arr, index):
    if index < 0 or index >= len(arr):
        raise ValueError("Index out of range")
    return arr[:index + 1]

def sift(res_list, value):
    left, right = 0, len(res_list) - 1
    result = -1  # 如果没有找到符合条件的元素，返回-1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < value:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return arr[:result + 1]

def check_difference(sifted_list):
    print()

def search(res_list, rw_threshold, max_it):
    generation = []
    it = 0
    #Step 1: 根据访存量，筛掉部分数据
    sifted_list = sift(res_list,rw_threshold) 
    #Step 2: 选出第一代种族
    generation.append(sifted_list[0])
    for i in range(1,len(sifted_list)):
        one_generation = sifted_list[i]
        if check_difference(generation, one_generation):
            generation.append(one_generation)
        if len(generation) == 6: # 第一代种族选6个
            break
    #Step 3: 变异
    while it < max_it: #超出最大的变异次数直接返回
        new_generation = []
        latency_fitnesses = simulate(generation)
        parents = choose_parents(latency_fitnesses) #选择三个最优的个体
        for parent in parents:
            child1,child2 = mutate(parent) #从list中选出两个类似的变异个体作为子代
            new_generation.append(child1)
            new_generation.append(child2)
        generation = new_generation #更新种群
    
    latency_fitnesses = simulate(generation) #最后一代种群的性能
    parents = choose_parents(latency_fitnesses) #选择最优的个体

    return parents[0] #返回最优个体

if __name__ == "__main__":
    arr = [0,1,2,5,8,10,14,15,16,20,40,43,87,100]
    value = 50
    print(sift(arr, value))