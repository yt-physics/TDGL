import os  # for makedirs
import homcloud.interface as hc  # HomCloud 
import numpy as np  # Numerical array library
from tqdm import tqdm  # For progressbar
import matplotlib.pyplot as plt  # Plotting
import sklearn.linear_model as lm  # Machine learning
from sklearn.decomposition import PCA  # for PCA
from sklearn.model_selection import train_test_split
import pyvista as pv  # for 3D visualization


# ディレクトリの指定
dir_1 = os.path.dirname(os.getcwd()) + r'\\PHanalyze_1super'
dir_2 = os.path.dirname(os.getcwd()) + r'\\PHanalyze_1sub'
output_path = os.path.dirname(os.getcwd()) + r'\\Results\\PCA_visualization.png'  # 保存先
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ディレクトリがない場合は作成


# PIVectorizeSpec
vectorize_spec_super = hc.PIVectorizeSpec(
    x_range=(0, 0.03),
    xbins=128,
    sigma=0.002,
    weight=("atan", 0.01, 3),
    superlevel=True  # Superlevel フィルトレーション
)

vectorize_spec_sub = hc.PIVectorizeSpec(
    x_range=(0, 0.03),
    xbins=128,
    sigma=0.002,
    weight=("atan", 0.01, 3),
    superlevel=False  # Sublevel フィルトレーション
)


# ベクトルを格納するリスト
vectors = []
indices = []

# すべてのファイルに対して処理を行う
for i in range(500, 10000, 500):  # 500刻みでファイルを探索
    filename = f"grayscale_{i:05d}.pdgm"  # ファイル名の生成
    file_path_1 = os.path.join(dir_1, filename)  # super のパス
    file_path_2 = os.path.join(dir_2, filename)  # sub のパス

    if os.path.exists(file_path_1) and os.path.exists(file_path_2):
        # ファイルが存在する場合は読み込む
        pd_1 = hc.PDList(file_path_1).dth_diagram(0)  # 0次の PD を取得
        pd_2 = hc.PDList(file_path_2).dth_diagram(0)

        # ベクトル化
        vector_1 = vectorize_spec_super.vectorize(pd_1)
        vector_2 = vectorize_spec_sub.vectorize(pd_2)

        # ベクトルを統合
        combined_vector = np.hstack([vector_1, vector_2])

        # リストに保存
        vectors.append(combined_vector)
        indices.append(i)

# ベクトルを NumPy 配列に変換
vectors = np.array(vectors)

# ベクトルの正規化
vectors = vectors / vectors.max()

# PCA を適用
pca = PCA(n_components=2)  # 2次元に圧縮
pca.fit(vectors)
reduced = pca.transform(vectors)

# 可視化
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=indices, cmap='viridis')
plt.colorbar(label="Index")
plt.gca().set_aspect('equal')
plt.title("PCA Visualization of Combined Persistence Images")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 解像度を設定して保存
plt.show()