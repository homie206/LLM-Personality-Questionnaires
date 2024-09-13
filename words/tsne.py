import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# 读取 CSV 文件
df = pd.read_csv('/home/hmsun/LLM-Personality-Questionnaires/words/adjectives_by_group.csv')

# 处理嵌入数据
df['Embedding'] = df['Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=', '))

# 提取嵌入向量
embeddings = np.vstack(df['Embedding'].values)

# 进行 t-SNE 降维，调整 perplexity 参数
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # 可以尝试不同的 perplexity 值
reduced_embeddings = tsne.fit_transform(embeddings)

# 创建颜色映射
label_encoder = LabelEncoder()
df['Group_encoded'] = label_encoder.fit_transform(df['Group'])
colors = plt.colormaps['viridis']

# 绘制图形，增大图形尺寸
plt.figure(figsize=(24, 18))
for group in df['Group'].unique():
    idx = df['Group'] == group
    plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1],
                label=group, color=colors(label_encoder.transform([group])[0] / len(df['Group'].unique())), alpha=0.7)

    # 在每个点旁边添加 Adjective 名称
    '''
    for i in np.where(idx)[0]:
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1],
                 df['Adjective'].iloc[i], fontsize=8, ha='right', alpha=0.7)
    '''


plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()