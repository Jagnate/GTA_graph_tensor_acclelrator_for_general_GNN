import csv

def move_pattern_to_front(input_filename, output_filename):
    # 读取 CSV 文件
    with open(input_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # 将 pattern 列移动到最前面
    fieldnames.remove('pattern')
    fieldnames.insert(0, 'pattern')

    # 写入新的 CSV 文件
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# 示例
input_filename = 'GCN_cora_layer1.csv'
output_filename = 'GCN_cora_layer1_modified.csv'
move_pattern_to_front(input_filename, output_filename)