import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
from datetime import datetime

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

    with torch.no_grad():
        c10 = (spin_tensor * torch.roll(spin_tensor, 1, 2)).mean().item()
        deviation = abs(c10 - 0.7071) / 0.7071 * 100
        m_mean = spin_tensor.mean()
        m_abs = torch.abs(m_mean).item()
        suscep = (1.0 - m_abs ** 2)

        m2 = (spin_tensor ** 2).mean()
        m4 = (spin_tensor ** 4).mean()
        binder = (1.0 - m4 / (3 * m2 ** 2 + 1e-8)).item()

        return c10, m_abs, binder, suscep, deviation

def main():
    model = FullConnectProjectionRG().to(DEVICE)
    if os.path.exists(KERNEL_FILE):
        model.load_state_dict(torch.load(KERNEL_FILE, map_location=DEVICE))
        print(f">>> 成功载入 RG 算子")
    model.eval()
    print(f">>> 正在生成 {LOW_RES_SIZE}x{LOW_RES_SIZE} 物理种子...")
    spin_small = torch.where(torch.rand((1, 1, LOW_RES_SIZE, LOW_RES_SIZE), device=DEVICE) < 0.5, 1.0, -1.0)
    spin_small = checkerboard_mc(spin_small, BETA, SEED_MC_STEPS)
    print(">>> 执行 RG 算子推理与修复...")
    with torch.no_grad():
        rg_raw_continuous = model(spin_small)
    rg_initial = torch.where(torch.rand_like(rg_raw_continuous) < (1.0 + rg_raw_continuous) / 2.0, 1.0, -1.0)
    rg_final = checkerboard_mc(rg_initial.clone(), BETA, REFINE_MC_STEPS)
    naive_initial = torch.nn.functional.interpolate(spin_small, scale_factor=SCALE_FACTOR, mode='nearest')
    naive_final = checkerboard_mc(naive_initial.clone(), BETA, REFINE_MC_STEPS)
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