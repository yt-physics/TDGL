import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob


def create_gif(in_dir, filename):
    ''' フォルダの複数画像からGIF画像を作る '''

    path_list = sorted(glob.glob(os.path.join(in_dir, '*.png')))
    print("File Path List:", path_list)  # デバッグ用にパスを表示

    # 画像ファイルが見つからない場合，エラーメッセージを表示して終了
    if not path_list:
        print("エラー: 画像ファイルなし　dir,filenameを確認せよ")
        return

    imgs = []
    for path in path_list:
        img = Image.open(path)
        imgs.append(img)

    # GIF作成
    out_filename = os.path.join(in_dir, filename)
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0)
    print(f"GIFファイルが作成されました: {out_filename}")
    return


dir = os.path.dirname(os.getcwd()) + r'\PHanalyze'
filename = '2d-TDGL_PH.gif'

create_gif(dir, filename)
