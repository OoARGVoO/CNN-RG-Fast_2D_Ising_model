import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "mathtext.fontset": "stix",
})


def calculate_sk_blocks(data, L, block_size=2048, num_blocks=8):

    print(f">>> 正在采样 {num_blocks} 个 {block_size}x{block_size} 子块计算 S(k)...")
    sk_avg = np.zeros((block_size, block_size))

    for i in range(num_blocks):
        y = np.random.randint(0, L - block_size)
        x = np.random.randint(0, L - block_size)
        block = data[y:y + block_size, x:x + block_size].astype(np.float32)
        fft_val = np.fft.fft2(block)
        sk_2d = np.abs(np.fft.fftshift(fft_val)) ** 2 / (block_size ** 2)
        sk_avg += sk_2d
        print(f"  [S(k)] 已完成子块 {i + 1}/{num_blocks}")

    return sk_avg / num_blocks


def analyze_criticality_ultra(file_path):
    print(f"\n>>> 启动超大尺度分析模式: {os.path.basename(file_path)}")
    data = np.load(file_path, mmap_mode='r')
    L = data.shape[0]
    print(f"检测到晶格尺寸 L: {L}")
    print(">>> 正在统计统计指标...")
    sample_indices = np.random.choice(L, 10000, replace=True)
    m_samples = [np.mean(data[idx, :].astype(np.float32)) for idx in sample_indices]
    m_abs = np.abs(np.mean(m_samples))
    print(f"采样平均磁化强度 |M|: {m_abs:.6f}")
    block_size = 2048
    sk_data = calculate_sk_blocks(data, L, block_size=block_size, num_blocks=5)
    print(">>> 正在采样关联函数 G(r)...")
    r_max = L // 2
    rs = np.unique(np.logspace(0, np.log10(r_max), 30).astype(int))
    gr = []
    num_samples_per_r = 8000
    for r in rs:
        y = np.random.randint(0, L, num_samples_per_r)
        x = np.random.randint(0, L - r, num_samples_per_r)
        s1 = data[y, x].astype(np.float32)
        s2 = data[y, x + r].astype(np.float32)
        gr_val = np.mean(s1 * s2)
        gr.append(gr_val)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    axes[0].loglog(rs, gr, 'o-', markersize=5, linewidth=1.5, label='Measured $G(r)$')
    theoretical_slope = rs ** (-0.25) * (gr[0] / rs[0] ** (-0.25))
    axes[0].loglog(rs, theoretical_slope, '--', label=r'Theory ($r^{-1/4}$)', alpha=0.8, linewidth=2)
    gr_array = np.array(gr)
    positive_gr = gr_array[gr_array > 0]
    if len(positive_gr) > 0:
        y_min = min(np.min(positive_gr) * 0.3, 1e-5)
    else:
        y_min = 1e-5
    axes[0].set_ylim(y_min, 2.0)
    axes[0].set_xlim(0.8, r_max)
    #

    axes[0].set_title(f"Spin Correlation $G(r)$\n($L={L}$)")
    axes[0].set_xlabel("Distance $r$")
    axes[0].set_ylabel("$G(r)$")
    axes[0].legend(frameon=True)
    axes[0].grid(True, which="both", ls="-", alpha=0.3)
    norm_sk = np.log10(sk_data + 1e-9)
    im = axes[1].imshow(norm_sk, cmap='magma', vmin=-3.0, vmax=5.0)

    axes[1].set_title(r"Structure Factor $S(\mathbf{k})$" + "\n(Scale: $\log_{10}$ -3.0 to 5.0)")
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label(r"$\log_{10} S(\mathbf{k})$", size=16)

    plt.tight_layout()
    save_fig = file_path.replace(".npy", "_Sk_Gr_Analysis.png")
    plt.savefig(save_fig, dpi=300)
    plt.show()
    print(f">>> 深度分析报告已保存: {save_fig}")


if __name__ == "__main__":
    NPY_FILE = r"E:\coding temp\TEST\scripts\RG_Results\SpinConfig_S00_20260207_163617.npy"

    if os.path.exists(NPY_FILE):
        analyze_criticality_ultra(NPY_FILE)
    else:
        print(f"未找到文件: {NPY_FILE}")