# %% [markdown]
# # 金庸小说人物词向量分析
# 
# 本notebook将实现以下功能：
# 1. 使用Word2Vec训练词向量模型
# 2. 提取人物名称的词向量
# 3. 使用PCA降维并可视化

# %% [markdown]
# ## 1. 导入必要的库

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import pickle
import jieba
from gensim.models import Word2Vec
import re
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %% [markdown]
# ## 2. 加载数据

# %%
print("加载人物名称...")
# 定义角色别名映射
character_aliases = {
    # 射雕英雄传
    '靖哥哥': '郭靖',
    '蓉儿': '黄蓉',
    '黄老邪': '黄药师',
    '东邪': '黄药师',
    '西毒': '欧阳锋',
    '南帝': '段智兴',
    '北丐': '洪七公',
    '中神通': '王重阳',
    '老顽童': '周伯通',
    '七公': '洪七公',
    
    # 神雕侠侣
    '龙儿': '小龙女',
    '过儿': '杨过',
    '姑姑': '小龙女',
    
    # 倚天屠龙记
    '无忌': '张无忌',
    '敏敏': '赵敏',
    '芷若': '周芷若',
    '金毛狮王': '谢逊',
    '紫衫龙王': '黛绮丝',
    
    # 天龙八部
    '萧峰': '乔峰',
    
    # 其他
    '香香公主': '喀丝丽'
}

# 加载角色名称
with open('金庸小说全人物集.txt', 'r', encoding='utf-8') as f:
    character_names = set(f.read().splitlines())
    # 添加别名到角色集合中
    for alias, name in character_aliases.items():
        if name in character_names:
            character_names.add(alias)

print(f"成功加载 {len(character_names)} 个人物名称（包含别名）")

# 加载停用词
print("加载停用词...")
with open('stop_words.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())
print(f"成功加载 {len(stopwords)} 个停用词")

# %% [markdown]
# ## 3. 数据预处理

# %%
print("开始数据预处理...")
sentences = []
name_to_novels = {}  # 记录每个角色出现的小说列表

# 遍历小说文件
for file in os.listdir('金庸小说全集'):
    if file.endswith('.txt'):
        novel_name = os.path.splitext(file)[0]
        with open(os.path.join('金庸小说全集', file), 'r', encoding='utf-8') as f:
            text = f.read().replace('\n', '')
            # 只保留中文、英文、数字和基本标点
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！]+', '', text).strip()
            
            # 分割句子
            split_sentences = [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
            
            # 对每个句子进行处理
            for sent in split_sentences:
                # 先检查句子中是否包含角色名或别名
                found_names = [name for name in character_names if name in sent]
                if found_names:
                    # 如果有角色名，先标记这些角色名
                    marked_sent = sent
                    for name in found_names:
                        marked_sent = marked_sent.replace(name, f" {name} ")
                    
                    # 分词
                    words = jieba.lcut(marked_sent, cut_all=False)
                    filtered_words = []
                    
                    # 处理分词结果，将别名映射为正式名称
                    for word in words:
                        if word not in stopwords and len(word.strip()) > 0:
                            # 如果是别名，转换为正式名称
                            actual_name = character_aliases.get(word, word)
                            filtered_words.append(actual_name)
                    
                    # 记录角色出现的小说
                    for word in filtered_words:
                        if word in character_names:
                            if word not in name_to_novels:
                                name_to_novels[word] = set()
                            name_to_novels[word].add(novel_name)
                    
                    sentences.append(filtered_words)

# %% [markdown]
# ## 4. 训练Word2Vec模型

# %%
print("开始训练Word2Vec模型...")
# 检查模型文件是否存在
model_path = "word_embedding.model"
if os.path.exists(model_path):
    print("发现已存在的模型文件，直接加载...")
    model = Word2Vec.load(model_path)
else:
    print("未找到模型文件，开始训练新模型...")
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=10,
        min_count=5,
        workers=4,
        sg=1
    )
    # 保存完整模型
    model.save(model_path)
    print("模型已保存为 word_embedding.model")

print("模型加载/训练完成")

# %% [markdown]
# ## 5. 提取人物名称的词向量

# %%
print("提取人物名称的词向量...")
embeddings = []
valid_names = []
name_novels = {}  # 记录每个角色的主要小说

for name in character_names:
    if name in model.wv.key_to_index:
        valid_names.append(name)
        embeddings.append(model.wv[name])
        
        # 确定该角色的主要小说
        novel_counts = name_to_novels.get(name, set())
        if novel_counts:
            max_novel = max(novel_counts, key=lambda k: k)
            name_novels[name] = max_novel
        else:
            name_novels[name] = '未知'

embeddings = np.array(embeddings)
print(f"成功获取 {len(valid_names)} 个人物名称的词向量")

# %% [markdown]
# ## 6. 2D可视化

# %%
print("生成2D可视化结果...")
# 2D可视化
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 创建DataFrame
df_2d = pd.DataFrame({
    'Name': valid_names,
    'Novel': [name_novels[name] for name in valid_names],
    'PC1': reduced_embeddings[:, 0],
    'PC2': reduced_embeddings[:, 1]
})

# 使用Plotly创建交互式散点图
fig = px.scatter(df_2d, 
                 x='PC1', 
                 y='PC2',
                 color='Novel',
                 text='Name',
                 title='人物词向量可视化',
                 labels={'PC1': '第一主成分', 'PC2': '第二主成分'},
                 hover_name='Name',
                 hover_data={'Novel': True})

# 调整显示效果
fig.update_traces(
    textposition='top center',
    marker=dict(size=10, opacity=0.7),
    textfont=dict(size=10),
    hovertemplate="<b>%{Name}</b><br>" +
                  "小说: %{Novel}<br>" +
                  "PC1: %{x:.2f}<br>" +
                  "PC2: %{y:.2f}<br>" +
                  "<extra></extra>"
)

# 调整布局
fig.update_layout(
    showlegend=True,
    legend=dict(
        title='小说',
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    height=800,
    width=1000
)

# 保存为HTML文件
fig.write_html('characters_2d.html', include_plotlyjs=True)
fig.show()

# %% [markdown]
# ## 7. 3D可视化

# %%
print("生成3D可视化结果...")
# 3D可视化
pca = PCA(n_components=3)
reduced_embeddings_3d = pca.fit_transform(embeddings)

# 创建DataFrame
df_3d = pd.DataFrame({
    'Name': valid_names,
    'Novel': [name_novels[name] for name in valid_names],
    'PC1': reduced_embeddings_3d[:, 0],
    'PC2': reduced_embeddings_3d[:, 1],
    'PC3': reduced_embeddings_3d[:, 2]
})

# 使用Plotly创建交互式3D散点图
fig = px.scatter_3d(df_3d, 
                    x='PC1', 
                    y='PC2',
                    z='PC3',
                    color='Novel',
                    text='Name',
                    title='人物词向量可视化',
                    labels={'PC1': '第一主成分', 'PC2': '第二主成分', 'PC3': '第三主成分'},
                    hover_name='Name',
                    hover_data={'Novel': True})

# 调整显示效果
fig.update_traces(
    textposition='top center',
    marker=dict(size=8, opacity=0.7),
    textfont=dict(size=10),
    hovertemplate="<b>%{Name}</b><br>" +
                  "小说: %{Novel}<br>" +
                  "PC1: %{x:.2f}<br>" +
                  "PC2: %{y:.2f}<br>" +
                  "PC3: %{z:.2f}<br>" +
                  "<extra></extra>"
)

fig.update_layout(
    showlegend=True,
    legend=dict(
        title='小说',
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    height=800,
    width=1000,
    scene=dict(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
)

# 保存为HTML文件
fig.write_html('characters_3d.html', include_plotlyjs=True)
fig.show()

# %% [markdown]
# ## 8. 角色相关性分析

# %%
print("分析角色相关性...")
# 计算余弦相关度
def get_related_characters(name, top_n=5):
    if name not in model.wv.key_to_index:
        return []
    
    # 获取最相关的角色
    related = model.wv.most_similar(name, topn=top_n*2)  # 获取更多结果以便过滤
    
    # 过滤掉非角色名称，并应用别名映射
    filtered_related = []
    seen_names = set()  # 用于去重
    
    for word, correlation in related:
        # 如果是别名，转换为正式名称
        actual_name = character_aliases.get(word, word)
        
        # 只保留在角色列表中的名称，并且去重
        if actual_name in character_names and actual_name not in seen_names:
            filtered_related.append((actual_name, correlation))
            seen_names.add(actual_name)
            if len(filtered_related) >= top_n:  # 达到所需数量就停止
                break
    
    return filtered_related

# 示例：分析几个主要角色的相关性
main_characters = ['郭靖', '黄蓉', '黄药师', '杨过', '小龙女', '周伯通', '张无忌', '赵敏']
print("\n角色相关性分析结果：")
for character in main_characters:
    if character in model.wv.key_to_index:
        related = get_related_characters(character)
        print(f"\n{character} 最相关的角色：")
        for name, correlation in related:
            print(f"{name}: {correlation:.4f}")