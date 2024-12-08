import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob

def initial_field(x_max, y_max, dx, dy, H):
    ''' 初期場を用意する '''

    # 初期場(x方向をj, y方向をkとする行列を作成→2D画像のデータ構造を模擬)
    x = np.linspace(0, x_max, int(x_max / dx))
    y = np.linspace(0, y_max, int(y_max / dy))
    psi = np.full((len(y), len(x)), 0.9+0.0j , dtype=complex)
    X, Y = np.meshgrid(x, y)

    # ベクトルポテンシャルの計算
    A_x = - (H / 2) * (Y - y_max / 2)
    A_y =   (H / 2) * (X - x_max / 2)
    
    return x, y, psi ,A_x , A_y

# def boundary_condition(psi):
#     ''' Dirichlet boundary condition '''

#     # 境界条件(左右上下)
#     psi[:, 0]  = 0
#     psi[:, -1] = 0
#     psi[0, :]  = 0
#     psi[-1, :] = 0

#     return psi


def boundary_condition(psi):
    ''' Neumann boundary condition '''
    # 上下の境界
    psi[0, :] = psi[1, :]
    psi[-1, :] = psi[-2, :]
    # 左右の境界
    psi[:, 0] = psi[:, 1]
    psi[:, -1] = psi[:, -2]
    return psi





def sol_2d_diffusion(x, y, psi, dt, dx, dy, step, A_x , A_y ,dir, result_interval, kappa):
    ''' 2次元TDGL方程式を計算 '''

    # 漸化式を反復計算
    psi= psi.T
    for i in range(step):
        psi0 = psi.copy() #これで前の計算をコピー
        for j in range(1, len(psi) - 1):
            for k in range(1, len(psi.T) - 1):
                r = (dt / dx ** 2)
                s = (dt / dy ** 2)
                psi[j, k] = psi0[j, k] - dt  * ( abs(psi0[j, k]) ** 2 - 1 ) * psi0[j,k] + \
                            r * (psi0[j,k-1]* np.exp(-1j * dx * A_x[j,k-1]) + psi0[j,k+1] * np.exp(1j * dx * A_x[j,k]) - 2 * psi0[j,k]) + \
                            s * (psi0[j-1,k]* np.exp(-1j * dy * A_y[j-1,k]) + psi0[j+1,k] * np.exp(1j * dy * A_y[j,k]) - 2 * psi0[j,k])    
                          
        # 境界条件を設定
        psi = boundary_condition(psi)

        # # 指定した間隔で画像保存
        # if i>10000 and i % result_interval == 0:
        #     print('Iteration=', i)
        #     psi = psi.T
        #     plot(x, y, abs(psi), i, dir, 1)
        #     psi = psi.T

        # 一定のステップごとに自由エネルギーを計算して表示
        if i > 0 and i % result_interval == 0:
            free_energy = compute_free_energy(psi, A_x, A_y, dx, dy, kappa)  # 定期的に自由エネルギーを計算
            print(f'Iteration={i}, Free Energy={free_energy:.2f}')
            psi = psi.T
            plot(x, y, abs(psi), i, dir, 1, free_energy)
            psi = psi.T

    return

# def compute_free_energy(psi, A_x, A_y, dx, dy, kappa):
#     ''' 無次元化された自由エネルギーを計算する関数
#         psi: 超伝導秩序変数
#         A_x, A_y: 無次元化ベクトルポテンシャル
#         dx, dy: 格子幅
#         kappa: GL係数
#     '''

#     # 秩序変数の絶対値とその二乗
#     psi_abs2 = np.abs(psi) ** 2
#     psi_abs4 = psi_abs2 ** 2

#     # ψの勾配を計算
#     dpsi_dx = (np.roll(psi, -1, axis=1) - psi) / dx
#     dpsi_dy = (np.roll(psi, -1, axis=0) - psi) / dy

#     # 無次元化されたベクトルポテンシャル A' による運動エネルギー項
#     kinetic_energy_x = np.abs(dpsi_dx - 1j * A_x * psi) ** 2
#     kinetic_energy_y = np.abs(dpsi_dy - 1j * A_y * psi) ** 2

#     # 磁場エネルギー項を計算（B = ∇ × A）
#     B_z = (np.roll(A_y, -1, axis=1) - A_y) / dx - (np.roll(A_x, -1, axis=0) - A_x) / dy
#     magnetic_energy = B_z ** 2 / 2

#     # 自由エネルギー密度の各項の合計
#     free_energy_density = psi_abs2 - 0.5 * psi_abs4 + (kinetic_energy_x + kinetic_energy_y) / (2 * kappa ** 2) + magnetic_energy

#     # 自由エネルギーを空間積分して全体の自由エネルギーを計算
#     total_free_energy = np.sum(free_energy_density) * dx * dy

#     return total_free_energy


def compute_free_energy(psi, A_x, A_y, dx, dy, kappa):
    ''' 無次元化された自由エネルギーを計算する関数
        psi: 超伝導秩序変数
        A_x, A_y: 無次元化ベクトルポテンシャル
        dx, dy: 格子幅
        kappa: GL係数
    '''

    # 秩序変数の絶対値の二乗
    psi_abs2 = np.abs(psi) ** 2

    # 境界を除いた内側の領域で計算
    psi_center = psi[1:-1, 1:-1]

    # 勾配の計算（中央差分）
    dpsi_dx = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)
    dpsi_dy = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dy)

    # ベクトルポテンシャルの中央領域
    A_x_center = A_x[1:-1, 1:-1]
    A_y_center = A_y[1:-1, 1:-1]

    # 共変微分
    Dpsi_dx = dpsi_dx - 1j * A_x_center * psi_center
    Dpsi_dy = dpsi_dy - 1j * A_y_center * psi_center

    # 運動エネルギー項
    kinetic_energy = np.abs(Dpsi_dx) ** 2 + np.abs(Dpsi_dy) ** 2

    # 磁場の計算
    dA_y_dx = (A_y[1:-1, 2:] - A_y[1:-1, :-2]) / (2 * dx)
    dA_x_dy = (A_x[2:, 1:-1] - A_x[:-2, 1:-1]) / (2 * dy)
    B_z = dA_y_dx - dA_x_dy

    # 磁場エネルギー項
    magnetic_energy = (B_z) ** 2 / 2

    # ポテンシャルエネルギー項
    potential_energy = (1 - psi_abs2[1:-1, 1:-1]) ** 2 / 2

    # 自由エネルギー密度
    free_energy_density = potential_energy + kinetic_energy / (2 * kappa ** 2) + magnetic_energy

    # 自由エネルギーの空間積分
    total_free_energy = np.sum(free_energy_density) * dx * dy

    return total_free_energy


# def save_numpy(x, y, psi, i, dir, save_flag, free_energy):
#     psi_abs =abs(psi)
#     path = os.path.join(*[dir, str("{:05}".format(i))])
#     np.save(path, psi_abs)



def plot(x, y, psi, i, dir, save_flag, free_energy):
    ''' 関数をプロット '''
    
    psi_abs =abs(psi)
    # フォントの種類とサイズを設定
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    # 目盛を内側にする
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # グラフの入れ物を用意して上下左右に目盛線を付ける
    x_size = 8
    y_size = int(0.8 * x_size * (np.max(y) / np.max(x)))
    fig = plt.figure(figsize=(x_size, y_size))
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')


    # 軸のラベルを設定
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # データをプロット
    im = ax1.imshow(psi_abs,
                    vmin=0, vmax=1,
                    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                    aspect='auto',
                    cmap='gray' #gray scale
                    # cmap='jet' #color
                    )
    # 自由エネルギーを表示
    ax1.text(0.1, 0.1, f'Step={i}, Free Energy={free_energy:.2f}', color="white", transform=ax1.transAxes)

    # ax1.text(0.1, 0.1, 'Step='+str(i), color="white")

    # カラーバーを設定
    cbar = fig.colorbar(im)
    cbar.set_label('psi')

    # 画像を保存
    # dirフォルダが無い時に新規作成
    save_dir = dir
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    # 画像保存パスを準備
    path = os.path.join(*[save_dir, str("{:05}".format(i)) + '.png'])
    path_np = os.path.join(*[save_dir, str("{:05}".format(i))])
    if save_flag == 1:
        # 画像を保存する
        plt.savefig(path)
        np.save(path_np, psi_abs)
    else:
        # グラフを表示する。
        plt.show()
    plt.close()


def create_gif(in_dir, filename):
    ''' imgフォルダの複数画像からGIF画像を作る '''

    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*.png'])))
    imgs = []

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])
        imgs.append(img)

    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    out_filename = os.path.join(in_dir, filename)
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=0)
    return


if __name__ == '__main__':
    ''' 条件設定を行いシミュレーションを実行、GIF画像を作成する '''

    # 画像保存フォルダと動画ファイル名
    dir = os.path.dirname(os.getcwd()) + r'\img_2d_TDGL_H=0.4_fin'
    filename = '2d-TDGL_H=0.4_fin.gif'

    # 時間項の条件
    dt = 0.01

    # 空間項の条件
    dx = 0.5
    dy = 0.5
    x_max = 25
    y_max = 25
    H = 0.4
    

    # 初期場を用意する
    x, y, psi, A_x, A_y= initial_field(x_max, y_max, dx, dy, H)

    # 境界条件を設定する
    psi = boundary_condition(psi)
    plot(x, y, psi, 0, dir, 0, free_energy=0)
    # save_numpy(x, y, psi, 0, dir, 0, free_energy=0)
    

    # 安定性の確認
    nu_x =  dt / dx ** 2
    nu_y =  dt / dy ** 2
    print('nu_x, nu_y=', nu_x, nu_y)

    # 計算を実行
    sol_2d_diffusion(x, y, psi, dt, dx, dy, 10000, A_x , A_y, dir, 500, kappa=1)
    
    #(x, y, psi, dt, dx, dy, a, step, A_x , A_y ,dir, result_interval)

    # GIF動画を作成
    create_gif(dir, filename)