import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================================
# 1. 全局物理参数
# ==========================================================
CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "PARAM_FILE": "rg_model_b3_free_poly.pt",

    "N_SEED": 1000,
    "BETA_C": 0.44068679,
    "SEED_STEPS": 3000,

    # 阶段 B: 棋盘格加速重构 (3000x3000)
    "GEN_BETA": 99999999,
    "RECON_PASSES": 20, # Colden
    "STEP_SIZE": 0.3,
    "OUTPUT_IMAGE": "ising_recon_beta_c.png"
}

# ==========================================================
# 2. 模型定义
# ==========================================================
class FreePolyRGModel(nn.Module):
    def __init__(self):
        super(FreePolyRGModel, self).__init__()
        self.kernel = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.w1 = nn.Parameter(torch.tensor([0.0]))
        self.w3 = nn.Parameter(torch.tensor([0.0]))
        self.A = nn.Parameter(torch.tensor([0.0]))

# ==========================================================
# 3. 核心功能函数
# ==========================================================

def get_physical_seed():
    n, beta_c = CONFIG["N_SEED"], CONFIG["BETA_C"]
    s = torch.randint(0, 2, (n, n), device=CONFIG["DEVICE"], dtype=torch.float32) * 2 - 1

    x, y = torch.meshgrid(torch.arange(n, device=CONFIG["DEVICE"]),
                          torch.arange(n, device=CONFIG["DEVICE"]), indexing='ij')
    mask_b = ((x + y) % 2 == 0)
    mask_w = ~mask_b

    print(f"🧬 正在生成 {n}x{n} 临界种子 (Beta_c={beta_c})...")
    for i in range(CONFIG["SEED_STEPS"]):
        for mask in [mask_b, mask_w]:
            neigh = torch.roll(s, 1, 0) + torch.roll(s, -1, 0) + torch.roll(s, 1, 1) + torch.roll(s, -1, 1)
            dE = 2 * s * neigh
            accept = (dE <= 0) | (torch.rand((n, n), device=CONFIG["DEVICE"]) < torch.exp(-dE * beta_c))
            s[mask & accept] *= -1
    return s.unsqueeze(0).unsqueeze(0)

def checkerboard_reconstruct(seed, model):
    # 此时有效温度 T = 1/0.4406 较高
    print(f"🚀 执行严格临界 Beta 重构 (Beta: {CONFIG['GEN_BETA']}, Step: {CONFIG['STEP_SIZE']})")

    with torch.no_grad():
        # 1. 空间投影
        s = F.conv_transpose2d(seed, model.kernel, stride=3)
        s = torch.clamp(model.w1 * s + model.w3 * torch.pow(s, 3), -model.A.item(), model.A.item())

        # 2. 掩码准备
        _, _, H, W = s.shape
        x, y = torch.meshgrid(torch.arange(H, device=CONFIG["DEVICE"]),
                              torch.arange(W, device=CONFIG["DEVICE"]), indexing='ij')
        mask_b = ((x + y) % 2 == 0).unsqueeze(0).unsqueeze(0)
        mask_w = ~mask_b

        w1, w3, A, b = model.w1.item(), model.w3.item(), model.A.item(), CONFIG["GEN_BETA"]
        step = CONFIG["STEP_SIZE"]

        # 3. 异步演化
        for p in range(CONFIG["RECON_PASSES"]):
            for mask in [mask_b, mask_w]:
                neigh = torch.roll(s, 1, 2) + torch.roll(s, -1, 2) + torch.roll(s, 1, 3) + torch.roll(s, -1, 3)

                # 产生扰动候选
                s_rand = (torch.rand_like(s) * 2 - 1) * step
                s_new = torch.clamp(s + s_rand, -A, A)

                # 能量差 (dV 和 dJ 现在会被较小的 b 缩放)
                dV = -0.5 * w1 * (s_new ** 2 - s ** 2) - 0.25 * w3 * (s_new ** 4 - s ** 4)
                dJ = -(s_new - s) * neigh

                # Metropolis 准则
                accept = torch.rand_like(s) < torch.exp(-b * (dV + dJ))
                s[mask & accept] = s_new[mask & accept]

            if (p + 1) % 25 == 0:
                print(f"   进度: {p + 1}/{CONFIG['RECON_PASSES']}")

    return torch.sign(s).squeeze().cpu().numpy()

# ==========================================================
# 4. 执行主程序
# ==========================================================
if __name__ == "__main__":
    model = FreePolyRGModel().to(CONFIG["DEVICE"])
    if os.path.exists(CONFIG["PARAM_FILE"]):
        model.load_state_dict(torch.load(CONFIG["PARAM_FILE"]))
        print(f"📂 载入预训练权重: {CONFIG['PARAM_FILE']}")
    else:
        print("❌ 错误: 未找到权重文件。")
        exit()

    start_t = time.time()
    seed_field = get_physical_seed()
    final_spins = checkerboard_reconstruct(seed_field, model)

    print(f"\n✨ 生成完成！耗时: {time.time() - start_t:.2f}s")

    custom_cmap = ListedColormap(['#93A5CB', '#F7A24F'])
    plt.figure(figsize=(15, 15), dpi=200)
    plt.imshow(final_spins, cmap=custom_cmap, interpolation='nearest')
    plt.axis('off')
    plt.savefig(CONFIG["OUTPUT_IMAGE"], bbox_inches='tight', pad_inches=0)
    plt.show()