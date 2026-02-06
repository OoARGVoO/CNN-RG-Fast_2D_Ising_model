import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = r"E:\coding temp\TEST\scripts\RG_MultiStage_Results"
KERNEL_FILE = r"E:\coding temp\TEST\scripts\full_conn_proj_k9_s3_2048_26_2_3.pt"
START_SIZE = 512
STAGES = 3
SCALE_FACTOR = 3
BETA = 0.4406868
SEED_MC_STEPS = 1000
REFINE_MC_STEPS = 35

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
        m2 = (spin_tensor ** 2).mean()
        m4 = (spin_tensor ** 4).mean()
        binder = (1.0 - m4 / (3 * m2 ** 2 + 1e-8)).item()
        deviation = abs(c10 - 0.7071) / 0.7071 * 100
        return c10, m_abs, binder, deviation

def main():
    model = FullConnectProjectionRG().to(DEVICE)
    if os.path.exists(KERNEL_FILE):
        model.load_state_dict(torch.load(KERNEL_FILE, map_location=DEVICE))
        print(f">>> 成功载入 RG 算子")
    model.eval()
    print(f">>> 正在生成种子...")
    seed_spin = torch.where(torch.rand((1, 1, START_SIZE, START_SIZE), device=DEVICE) < 0.5, 1.0, -1.0)
    seed_spin = checkerboard_mc(seed_spin, BETA, SEED_MC_STEPS)

    current_spin = seed_spin.clone()
    stage_inits = [] 
    stage_refines = [] 
    stage_metrics = []

    stage_metrics.append(("Seed", *get_physics_metrics(seed_spin)))
    for s in range(1, STAGES + 1):
        print(f">>> 正在执行第 {s} 级超分辨率...")
        with torch.no_grad():
            continuous_out = model(current_spin)
        rg_init = torch.where(torch.rand_like(continuous_out) < (1.0 + continuous_out) / 2.0, 1.0, -1.0)
        stage_inits.append(rg_init.clone())
        stage_metrics.append((f"Stage {s} Init", *get_physics_metrics(rg_init)))
        current_spin = checkerboard_mc(rg_init, BETA, REFINE_MC_STEPS)
        stage_refines.append(current_spin.clone())
        stage_metrics.append((f"Stage {s} Refine", *get_physics_metrics(current_spin)))
    print("\n" + "=" * 90)
    print(f"{'阶段':<20} | {'c10':<12} | {'|M|':<12} | {'Binder':<10} | {'偏差'}")
    print("-" * 90)
    for m in stage_metrics:
        print(f"{m[0]:<20} | {m[1]:.6f}     | {m[2]:.6f}     | {m[3]:.4f}   | {m[4]:.2f}%")
    print("=" * 90)
    print(">>> 正在合成完整图对比 (两行输出)...")
    final_h, final_w = stage_refines[-1].shape[-2:]
    VIS_SIZE = (2048, 2048)

    def to_vis(t):
        t_resized = nn.functional.interpolate(t, size=VIS_SIZE, mode='nearest')
        return ((t_resized.squeeze().cpu().numpy() + 1) * 127.5).astype(np.uint8)
    row1 = [to_vis(seed_spin)] + [to_vis(img) for img in stage_inits]
    row2 = [to_vis(seed_spin)] + [to_vis(img) for img in stage_refines]

    sep = np.zeros((VIS_SIZE[0], 20), dtype=np.uint8)  
    h_sep = np.zeros((20, (VIS_SIZE[1] + 20) * (STAGES + 1) - 20), dtype=np.uint8)

    row1_combined = np.concatenate([np.concatenate([img, sep], axis=1) for img in row1[:-1]] + [row1[-1]], axis=1)
    row2_combined = np.concatenate([np.concatenate([img, sep], axis=1) for img in row2[:-1]] + [row2[-1]], axis=1)
    final_output = np.concatenate([row1_combined, h_sep, row2_combined], axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"RG_FullComparison_{timestamp}.png")
    Image.fromarray(final_output).save(save_path)

    print(f"\n>>> 结果已保存: {save_path}")
    print(">>> 布局: [第一行: 直出演化] / [第二行: 修复演化]")


if __name__ == "__main__":
    main()