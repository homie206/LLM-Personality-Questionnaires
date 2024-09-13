import os
import re
import csv
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

def extract_adjectives_from_txt_files(folder_path):
    # 使用 defaultdict 来存储按第一个单词分组的形容词
    adjectives_by_group = defaultdict(list)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            first_word = filename.split('-')[0]  # 提取文件名的第一个单词
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # 使用正则表达式提取形容词列表
                matches = re.findall(r'^\d+\.\s*([^\n]+)', content, re.MULTILINE)
                if matches:
                    # 将匹配的形容词添加到对应第一个单词的列表中
                    adjectives_by_group[first_word].extend(matches)

    return adjectives_by_group

def save_adjectives_to_csv(adjectives_by_group, output_file):
    # 将形容词写入 CSV 文件，每个形容词单独一行
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入标题
        writer.writerow(['Group', 'Adjective'])

        # 写入每个组的形容词
        for group, adjectives in adjectives_by_group.items():
            for adjective in adjectives:
                writer.writerow([group, adjective])  # 每个形容词单独一行

# 使用示例
folder_path = '/home/hmsun/LLM-Personality-Questionnaires/words/gen_words_combined'  # 替换为你的文件夹路径
output_file = 'adjectives_by_group.csv'  # 输出 CSV 文件名
adjectives_by_group = extract_adjectives_from_txt_files(folder_path)

# 保存到 CSV 文件
save_adjectives_to_csv(adjectives_by_group, output_file)

print(f"Adjectives have been saved to {output_file}.")

# 初始化模型和分词器
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 如果没有设置填充标记，则使用结束标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 读取CSV文件
csv_file_path = output_file  # 使用刚刚生成的CSV文件路径
df = pd.read_csv(csv_file_path)

# 准备一个新的列来存储嵌入
embeddings = []
success_count = 0
# 为每个形容词生成嵌入
for adjective in df['Adjective']:
    inputs = tokenizer(adjective, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings_tensor = outputs.last_hidden_state
    # 获取第一个词的嵌入并转换为列表
    adjective_embedding = embeddings_tensor[0][0].tolist()
    embeddings.append(adjective_embedding)
    success_count += 1
    print(f"success {success_count}!")


# 将嵌入添加到DataFrame中
df['Embedding'] = embeddings

# 保存更新后的DataFrame到原CSV文件
df.to_csv(csv_file_path, index=False)

print(f"Embeddings have been added to {csv_file_path}.")