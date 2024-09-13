import pandas as pd
import os

# 设置文件夹路径
folder_path = '/home/hmsun/LLM-Personality-Questionnaires/ipip50/Llama3.1-8b-instruct/combined-result-0909'

# 初始化结果字典
results = {}

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        # 提取文件名中的部分，去掉最后的 '-combined-ipip50'
        file_name_part = '-'.join(file_name.split('-')[:-9]).lower()  # 提取并转换为小写

        # 读取 CSV 文件
        data = pd.read_csv(file_path)

        # 统计匹配个数
        match_count = 0
        for index, row in data.iterrows():
            # 生成字符串，用于与文件名比较并转换为小写
            generated_string = f"E-{row['EXT_high_or_low']}-N-{row['EST_high_or_low']}-A-{row['AGR_high_or_low']}-C-{row['CSN_high_or_low']}-O-{row['OPN_high_or_low']}".lower()
            #generated_string = f"E-{row['EXT_high_or_low']}".lower()
            # generated_string = f"N-{row['EST_high_or_low']}".lower()
            # generated_string = f"A-{row['AGR_high_or_low']}".lower()
            # generated_string = f"C-{row['CSN_high_or_low']}".lower()
            # generated_string = f"O-{row['OPN_high_or_low']}".lower()

            # 打印 file_name_part 和 generated_string
            print(f"File Name Part: {file_name_part}, Generated String: {generated_string}")

            if generated_string == file_name_part:
                match_count += 1

        # 保存匹配个数到结果字典
        results[file_name] = match_count

# 输出统计结果
print("文件匹配个数统计结果:")
for file_name, count in results.items():
    print(f"{file_name}: {count}")