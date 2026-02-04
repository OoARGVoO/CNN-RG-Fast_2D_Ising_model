import os

# [2026-01-10] 指令：处理 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# --- 参数配置 ---
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = r"E:\coding temp\TEST\scripts\RG_MultiStage_Results"
CONFIG_DIR = os.path.join(OUTPUT_DIR, "spin_configs")
KERNEL_FILE = r"E:\coding temp\TEST\scripts\full_conn_proj_k9_s3_2048_26_2_3.pt"

START_SIZE = 512
STAGES = 3
SCALE_FACTOR = 3
BETA = 0.4406868
SEED_MC_STEPS = 1000
REFINE_MC_STEPS = 35

# 追踪区域大小设置：在起始 512x512 中选取的观察窗口
TRACK_WINDOW_INITIAL = 32

for path in [OUTPUT_DIR, CONFIG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)


# ==========================================
# --- 模型定义 ---
# ==========================================
class FullConnectProjectionRG(nn.Module):
    def __init__(self, k_size=9, s_factor=3):
        super().__init__()
        self.proj = nn.Conv2d(1, s_factor ** 2, k_size, padding=k_size // 2, bias=False)
        self.ps = nn.PixelShuffle(s_factor)
        self.w1 = nn.Parameter(torch.tensor([1.0]))
        self.A = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        m = self.proj(x)
        out = self.ps(m)
        return torch.clamp(self.w1 * out, -self.A, self.A)


# ==========================================
# --- 物理工具 ---
# ==========================================
def checkerboard_mc(spin, beta, steps):
    if steps <= 0: return spin
    B, C, H, W = spin.shape
    coords = torch.stack(torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij'))
    mask_even = ((coords[0] + coords[1]) % 2 == 0).float().unsqueeze(0).unsqueeze(0)
    for _ in range(steps):
        for mask in [mask_even, 1.0 - mask_even]:
            neighbors = (torch.roll(spin, 1, 2) + torch.roll(spin, -1, 2) +
                         torch.roll(spin, 1, 3) + torch.roll(spin, -1, 3))
            dE = 2.0 * spin * neighbors
            prob = torch.exp(-beta * dE)
            accept = (torch.rand(spin.shape, device=DEVICE) < prob).float()
            spin = torch.where(mask > 0, torch.where(accept > 0, -spin, spin), spin)
    return spin


def get_physics_metrics(spin_tensor):
    with torch.no_grad():
        c10 = (spin_tensor * torch.roll(spin_tensor, 1, 2)).mean().item()
        m_abs = torch.abs(spin_tensor.mean()).item()
        deviation = abs(c10 - 0.7071) / 0.7071 * 100
        return c10, m_abs, deviation


# ==========================================
# --- 主流程 ---
# ==========================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = FullConnectProjectionRG().to(DEVICE)
    if os.path.exists(KERNEL_FILE):
        model.load_state_dict(torch.load(KERNEL_FILE, map_location=DEVICE))
        print(f">>> 载入模型: {KERNEL_FILE}")
    model.eval()

    # 1. 种子初始化
    print(f">>> 生成起始种子 (L={START_SIZE})...")
    seed_spin = torch.where(torch.rand((1, 1, START_SIZE, START_SIZE), device=DEVICE) < 0.5, 1.0, -1.0)
    seed_spin = checkerboard_mc(seed_spin, BETA, SEED_MC_STEPS)

    current_spin = seed_spin.clone()
    comparison_data = []

    # 初始记录
    c10, m_abs, dev = get_physics_metrics(seed_spin)
    # 裁剪起始 32x32 区域 (中心区域)
    start_idx = START_SIZE // 2
    crop_data = seed_spin[
        0, 0, start_idx:start_idx + TRACK_WINDOW_INITIAL, start_idx:start_idx + TRACK_WINDOW_INITIAL].cpu().numpy()

    comparison_data.append({
        "label": f"Seed\n{TRACK_WINDOW_INITIAL}x{TRACK_WINDOW_INITIAL}",
        "img": crop_data, "c10": c10, "dev": dev
    })

    metrics_table = [("Seed", c10, m_abs, dev)]

    # 2. 多级演化：追踪区域像素随级数翻倍
    for s in range(1, STAGES + 1):
        print(f">>> 执行第 {s} 级演化...")
        with torch.no_grad():
            continuous_out = model(current_spin)
            rg_init = torch.where(torch.rand_like(continuous_out) < (1.0 + continuous_out) / 2.0, 1.0, -1.0)
            current_spin = checkerboard_mc(rg_init, BETA, REFINE_MC_STEPS)

        c10, m_abs, dev = get_physics_metrics(current_spin)
        metrics_table.append((f"Stage {s}", c10, m_abs, dev))

        # 核心逻辑：追踪区域的大小随 SCALE_FACTOR 增长
        # 比如 Stage 1 追踪 32*3=96, Stage 2 追踪 288...
        current_window = TRACK_WINDOW_INITIAL * (SCALE_FACTOR ** s)
        start_idx_s = current_spin.shape[-1] // 2
        crop_s = current_spin[
            0, 0, start_idx_s:start_idx_s + current_window, start_idx_s:start_idx_s + current_window].cpu().numpy()

        comparison_data.append({
            "label": f"Stage {s}\n{current_window}x{current_window}",
            "img": crop_s, "c10": c10, "dev": dev
        })

    # 3. 数据报告生成
    fig, axes = plt.subplots(1, len(comparison_data), figsize=(18, 6))
    for i, data in enumerate(comparison_data):
        # 使用 'nearest' 能够清晰看到像素格点的“变细”过程
        axes[i].imshow(data['img'], cmap='Greys', interpolation='nearest')
        axes[i].set_title(data['label'], fontsize=12, fontweight='bold')

        # 标注物理指标
        axes[i].text(0.05, 0.05, f"$c_{{10}}$:{data['c10']:.4f}\nDev:{data['dev']:.1f}%",
                     transform=axes[i].transAxes, color='red', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        axes[i].axis('off')

    plt.suptitle(f"RG Microstructure Evolution: Tracking Initial {TRACK_WINDOW_INITIAL}x{TRACK_WINDOW_INITIAL} Region",
                 fontsize=14)
    report_path = os.path.join(OUTPUT_DIR, f"RG_Tracking_Evolution_{timestamp}.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')

    # 打印表格
    print("\n" + "=" * 70)
    print(f"{'阶段':<15} | {'c10':<10} | {'偏差(%)'}")
    print("-" * 70)
    for row in metrics_table:
        print(f"{row[0]:<15} | {row[1]:.6f} | {row[3]:.2f}%")
    print("=" * 70)
    print(f"\n>>> 演化追踪图已保存: {report_path}")


if __name__ == "__main__":
    main()