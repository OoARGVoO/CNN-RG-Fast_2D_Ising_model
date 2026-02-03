import os

# [2026-01-10] 指令：处理 OpenMP 运行时冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
from datetime import datetime

# ==========================================
# --- 物理参数与路径配置 ---
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = r"E:\coding temp\TEST\scripts\RG_Upsampling_Results"
KERNEL_FILE = r"E:\coding temp\TEST\scripts\full_conn_proj_k9_s3_2048_26_2_3.pt"

LOW_RES_SIZE = 4096
SCALE_FACTOR = 3
BETA = 0.4406868
SEED_MC_STEPS = 16000
REFINE_MC_STEPS = 20

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ==========================================
# --- 核心模型定义 ---
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
# --- 引擎与工具函数 ---
# ==========================================
def checkerboard_mc(spin, beta, steps):
    B, C, H, W = spin.shape
    coords = torch.stack(torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij'))
    mask_even = ((coords[0] + coords[1]) % 2 == 0).float().unsqueeze(0).unsqueeze(0)
    mask_odd = 1.0 - mask_even
    for _ in range(steps):
        for mask in [mask_even, mask_odd]:
            neighbors = (torch.roll(spin, 1, 2) + torch.roll(spin, -1, 2) +
                         torch.roll(spin, 1, 3) + torch.roll(spin, -1, 3))
            dE = 2.0 * spin * neighbors
            prob = torch.exp(-beta * dE)
            accept = (torch.rand(spin.shape, device=DEVICE) < prob).float()
            spin = torch.where(mask > 0, torch.where(accept > 0, -spin, spin), spin)
    return spin


def get_physics_metrics(spin_tensor):
    """
    临界态多维度评估：
    1. c10: 能量项 (目标 0.7071)
    2. |M|: 序参量 (临界点应接近 0)
    3. Binder: 宾德累积量相关项 (1 - <m^4>/3<m^2>^2)，衡量分布形状
    4. Suscep: 局部涨落 (1 - m^2)，衡量系统响应
    """
    with torch.no_grad():
        # 1. 能量项与偏差
        c10 = (spin_tensor * torch.roll(spin_tensor, 1, 2)).mean().item()
        deviation = abs(c10 - 0.7071) / 0.7071 * 100

        # 2. 序参量 (绝对磁化强度)
        m_mean = spin_tensor.mean()
        m_abs = torch.abs(m_mean).item()

        # 3. 磁化率相关涨落 (临界点应具有最大涨落)
        suscep = (1.0 - m_abs ** 2)

        # 4. 宾德累积量 (Binder Cumulant) 简易评估
        # 在有限尺寸 L 下，临界点的 Binder Cumulant 具有不变量特性
        m2 = (spin_tensor ** 2).mean()
        m4 = (spin_tensor ** 4).mean()
        binder = (1.0 - m4 / (3 * m2 ** 2 + 1e-8)).item()

        return c10, m_abs, binder, suscep, deviation


# ==========================================
# --- 主流程 ---
# ==========================================
def main():
    # 1. 加载模型
    model = FullConnectProjectionRG().to(DEVICE)
    if os.path.exists(KERNEL_FILE):
        model.load_state_dict(torch.load(KERNEL_FILE, map_location=DEVICE))
        print(f">>> 成功载入 RG 算子")
    model.eval()

    # 2. 生成统一种子
    print(f">>> 正在生成 {LOW_RES_SIZE}x{LOW_RES_SIZE} 物理种子...")
    spin_small = torch.where(torch.rand((1, 1, LOW_RES_SIZE, LOW_RES_SIZE), device=DEVICE) < 0.5, 1.0, -1.0)
    spin_small = checkerboard_mc(spin_small, BETA, SEED_MC_STEPS)

    # 3. 路径执行
    print(">>> 执行 RG 算子推理与修复...")
    with torch.no_grad():
        rg_raw_continuous = model(spin_small)
    # RG 直出图 (概率采样离散化)
    rg_initial = torch.where(torch.rand_like(rg_raw_continuous) < (1.0 + rg_raw_continuous) / 2.0, 1.0, -1.0)
    # RG 修复图
    rg_final = checkerboard_mc(rg_initial.clone(), BETA, REFINE_MC_STEPS)

    # 直接放大对比 (仅用于报告)
    naive_initial = torch.nn.functional.interpolate(spin_small, scale_factor=SCALE_FACTOR, mode='nearest')
    naive_final = checkerboard_mc(naive_initial.clone(), BETA, REFINE_MC_STEPS)

    # 4. 临界态深度评估报告
    results = {
        "Seed (Input)": get_physics_metrics(spin_small),
        "RG Path (Initial)": get_physics_metrics(rg_initial),
        "RG Path (Final)": get_physics_metrics(rg_final),
        "Direct Path (Initial)": get_physics_metrics(naive_initial),
        "Direct Path (Final)": get_physics_metrics(naive_final)
    }

    print("\n" + "=" * 105)
    print(
        f"{'配置路径':<25} | {'c10 (能量)':<12} | {'|M| (序参量)':<12} | {'Binder':<10} | {'涨落感度':<10} | {'偏差 %'}")
    print("-" * 105)
    for label, (c10, m_abs, binder, sus, dev) in results.items():
        print(f"{label:<25} | {c10:.6f}     | {m_abs:.6f}     | {binder:.4f}   | {sus:.4f}   | {dev:.2f}%")
    print("-" * 105)

    # 控制台深度评价
    print(">>> 临界态诊断报告:")
    rg_m = results["RG Path (Initial)"][1]
    if rg_m < 0.05:
        print(f"    [OK] RG 直出图像序参量 |M|={rg_m:.4f}，成功保持了临界无序态。")
    else:
        print(f"    [!] RG 直出图像出现自发磁化 (|M|={rg_m:.4f})，可能正在流向有序固定点。")

    rg_dev = results["RG Path (Initial)"][4]
    if rg_dev < 1.0:
        print(f"    [OK] 能量项偏差极低 ({rg_dev:.2f}%)，RG 算子准确捕获了哈密顿量核心。")
    else:
        print(f"    [WNG] 能量项偏差为 {rg_dev:.2f}%，直出图能量密度不足，依赖 MC 修复。")
    print("=" * 105)

    # 5. 可视化部分
    def to_img(t):
        return ((t.squeeze().cpu().numpy() + 1) * 127.5).astype(np.uint8)

    crop = 512 * SCALE_FACTOR
    img_seed = to_img(torch.nn.functional.interpolate(spin_small, scale_factor=SCALE_FACTOR, mode='nearest'))[
        :crop, :crop]
    img_rg_init = to_img(rg_initial)[:crop, :crop]
    img_rg_final = to_img(rg_final)[:crop, :crop]

    sep = np.zeros((crop, 5), dtype=np.uint8)
    combined = np.concatenate([img_seed, sep, img_rg_init, sep, img_rg_final], axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"RG_Evolution_{timestamp}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    Image.fromarray(combined).save(save_path)
    print(f"\n>>> 可视化图已保存: {save_path}")
    print(">>> 布局: [左] 原始种子插值 | [中] RG 算子直出图 | [右] RG 修复完成图")


if __name__ == "__main__":
    main()