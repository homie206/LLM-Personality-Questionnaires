import os
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('/home/hmsun/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')


# 提取需要的列
dims = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
columns = [i + str(j) for j in range(1, 11) for i in dims]
df = df[columns]


def get_final_scores(columns, dim):
    score = 0
    if dim == 'EXT':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score += (6 - columns[9])
    if dim == 'EST':
        score += (6 - columns[0])
        score += columns[1]
        score += (6 - columns[2])
        score += columns[3]
        score += (6 - columns[4])
        score += (6 - columns[5])
        score += (6 - columns[6])
        score += (6 - columns[7])
        score += (6 - columns[8])
        score += (6 - columns[9])
    if dim == 'AGR':
        score += (6 - columns[0])
        score += columns[1]
        score += (6 - columns[2])
        score += columns[3]
        score += (6 - columns[4])
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score += columns[8]
        score += columns[9]
    if dim == 'CSN':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score += columns[9]
    if dim == 'OPN':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += columns[7]
        score += columns[8]
        score += columns[9]
    return score


# 计算每个维度的总分
for dim in dims:
    df[dim + '_all'] = df.apply(lambda r: get_final_scores([r[dim + str(j)] for j in range(1, 11)], dim), axis=1)


def cal_test_position(test_score, df):
    print("success start once!")
    positions = {}
    for cnt, dim in enumerate(dims):
        df_tmp = df.sort_values(by=dim + '_all')
        target_value = test_score[cnt]
        if target_value in df_tmp[dim + '_all'].values:
            index_position = df_tmp[dim + '_all'][df_tmp[dim + '_all'] == target_value].index[0]
            percentage_position = (index_position + 1) / len(df_tmp[dim + '_all']) * 100
            positions[dim + '_position'] = percentage_position
        else:
            positions[dim + '_position'] = None  # 如果目标值不在数据中
    return positions


# 遍历 result2 文件夹中的所有 CSV 文件



if __name__ == '__main__':
    result_dir = '/home/hmsun/LLM-Personality-Questionnaires/ipip50/Llama3.1-8b-instruct/new-combined-result'
    for filename in os.listdir(result_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(result_dir, filename)
            df2 = pd.read_csv(file_path)

            # 遍历 df2 的每一行，并将位置添加到 df2
            for index, row in df2.iterrows():
                test_score = row[['EXT_Score', 'EST_Score', 'AGR_Score', 'CSN_Score', 'OPN_Score']].values.tolist()
                positions = cal_test_position(test_score, df)
                print("success once!")

                # 将位置添加到 df2
                for key, value in positions.items():
                    df2.at[index, key] = value

            # 保存更新后的 df2
            df2.to_csv(file_path, index=False)
            print("success !!!!!")
