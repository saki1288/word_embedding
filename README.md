# 金庸小说人物词向量分析

本项目使用Word2Vec模型对金庸小说中的人物进行词向量分析，并通过PCA降维进行可视化。

## 功能特点

1. 人物名称预处理
   - 支持角色别名映射（如"靖哥哥"->"郭靖"）
   - 自动识别并过滤停用词

2. 词向量训练
   - 使用Word2Vec模型训练词向量
   - 支持模型保存和加载

3. 可视化分析
   - 2D和3D人物关系可视化
   - 支持交互式查看（缩放、旋转等）
   - 按小说分类显示不同颜色

4. 角色相关性分析
   - 计算角色之间的相似度
   - 显示最相关的角色

## 环境要求

- Python 3.7+
- 依赖包：见requirements.txt

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据：
   - 将金庸小说文本放在`金庸小说全集`目录下
   - 确保有`金庸小说全人物集.txt`和`stop_words.txt`文件

3. 运行分析：
```bash
python word_embedding_analysis.py
```

4. 查看结果：
   - 2D可视化结果保存在`characters_2d.html`
   - 3D可视化结果保存在`characters_3d.html`

## 文件说明

- `word_embedding_analysis.py`: 主程序文件
- `金庸小说全人物集.txt`: 包含所有角色名称
- `stop_words.txt`: 停用词列表
- `金庸小说全集/`: 存放小说文本的目录
- `characters_2d.html`: 2D可视化结果
- `characters_3d.html`: 3D可视化结果

## 注意事项

1. 确保小说文本编码为UTF-8
2. 角色名称文件每行一个角色名
3. 停用词文件每行一个停用词 