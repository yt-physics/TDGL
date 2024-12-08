import numpy as np
import homcloud.interface as hc
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# ステップの幅と範囲(適宜変更)
start = 500   # 開始番号
end = 9500    # 終了
step = 500    # ステップ幅
output_folder = os.path.dirname(os.getcwd()) + r'\\PHanalyze_1super'  # 保存先フォルダ

# 保存先フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

for num in range(start, end + 1, step):
    # ファイル名をゼロパディングで作成
    filename = os.path.dirname(os.getcwd()) + r'\\img_2d_TDGL_H=0.4_fin\\' + f"{num:05d}.npy"
    
    # ファイルを読み込み
    picture = np.load(filename)
    pict = np.array(picture)
    
    # パーシステンス図を生成して保存
    pdg_filename = f"{output_folder}/grayscale_{num:05d}.pdgm"
    hc.PDList.from_bitmap_levelset(pict, "superlevel", save_to=pdg_filename)
    # hc.PDList.from_bitmap_levelset(pict, "sublevel", save_to=pdg_filename)
    
    # パーシステンス図をプロットして保存
    pd = hc.PDList(pdg_filename).dth_diagram(0)
    histogram = pd.histogram(x_bins=64)
    
    # プロット画像の保存
    plot_filename = f"{output_folder}/histogram_{num:05d}.png"
    histogram.plot(colorbar={"type": "log"})
    plt.savefig(plot_filename)
    plt.close()  # メモリ解放のためプロットを閉じる

print("Save Complete")

# GIF 生成関数
def create_gif(in_dir, out_filename):
    ''' 指定フォルダ内の複数画像からgifを作成 '''
    path_list = sorted(glob.glob(os.path.join(in_dir, '*.png')))  # PNG ファイルのみを対象
    print("File Path List:", path_list)

    if not path_list:
        print("error: PNG not found")
        return

    imgs = [Image.open(path) for path in path_list]
    imgs[0].save(out_filename, save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0)
    print(f"GIF Generated: {out_filename}")

# GIF の生成
gif_folder = os.path.dirname(os.getcwd()) + r"\\PHgif"
gif_file = gif_folder + r"\\2d-TDGL_PH2.gif"
os.makedirs(gif_folder, exist_ok=True)
create_gif(output_folder, gif_file)