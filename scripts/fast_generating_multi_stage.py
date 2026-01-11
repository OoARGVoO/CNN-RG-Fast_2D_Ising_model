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
# 1. 全局配置
# ==========================================================
CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "PARAM_FILE": "rg_model_b3_free_poly.pt",

    # 初始种子参数 (第一代)
    "N_SEED": 1000,
    "BETA_C": 0.44068679,
    "SEED_STEPS": 6000,

    # 迭代放大配置
    "N_ITERATIONS": 4,

    "GEN_BETA": 99999999,  # 极低温度锁定
    "RECON_PASSES": 20,
    "STEP_SIZE": 0.3,  # 固定步长
    "OUTPUT_PREFIX": "ising_iterative"
}


# ==========================================================
# 2. 模型定义 (Free-Poly 结构)
# ==========================================================
class FreePolyRGModel(nn.Module):
    def __init__(self):
        super(FreePolyRGModel, self).__init__()
        self.kernel = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.w1 = nn.Parameter(torch.tensor([0.0]))
        self.w3 = nn.Parameter(torch.tensor([0.0]))
        self.A = nn.Parameter(torch.tensor([0.0]))


# ==========================================================
# 3. 核心功能模块
# ==========================================================

def get_physical_seed():
    """生成第一代物理种子 (1000x1000)"""
    n, beta_c = CONFIG["N_SEED"], CONFIG["BETA_C"]
    s = torch.randint(0, 2, (n, n), device=CONFIG["DEVICE"], dtype=torch.float32) * 2 - 1

    x, y = torch.meshgrid(torch.arange(n, device=CONFIG["DEVICE"]),
                          torch.arange(n, device=CONFIG["DEVICE"]), indexing='ij')
    mask_b = ((x + y) % 2 == 0)
    mask_w = ~mask_b

    print(f"🧬 [种子代] 正在生成 {n}x{n} 临界种子...")
    for i in range(CONFIG["SEED_STEPS"]):
        for mask in [mask_b, mask_w]:
            neigh = torch.roll(s, 1, 0) + torch.roll(s, -1, 0) + torch.roll(s, 1, 1) + torch.roll(s, -1, 1)
            dE = 2 * s * neigh
            accept = (dE <= 0) | (torch.rand((n, n), device=CONFIG["DEVICE"]) < torch.exp(-dE * beta_c))
            s[mask & accept] *= -1
    return s.unsqueeze(0).unsqueeze(0)


def iterate_reconstruct(field, model, iter_idx):
    """单次放大重构：输入一个小图，输出一个放大 3 倍的二值化大图"""
    _, _, h_in, w_in = field.shape
    h_out, w_out = h_in * 3, w_in * 3
    print(f"🚀 [迭代 {iter_idx + 1}] 重构启动: {h_in}x{w_in} -> {h_out}x{w_out}")

    with torch.no_grad():
        # 1. 空间投影 (Upsampling)
        s = F.conv_transpose2d(field, model.kernel, stride=3)
        s = torch.clamp(model.w1 * s + model.w3 * torch.pow(s, 3), -model.A.item(), model.A.item())

        # 2. 掩码准备
        x, y = torch.meshgrid(torch.arange(h_out, device=CONFIG["DEVICE"]),
                              torch.arange(w_out, device=CONFIG["DEVICE"]), indexing='ij')
        mask_b = ((x + y) % 2 == 0).unsqueeze(0).unsqueeze(0)
        mask_w = ~mask_b

        w1, w3, A, b = model.w1.item(), model.w3.item(), model.A.item(), CONFIG["GEN_BETA"]
        step = CONFIG["STEP_SIZE"]

        # 3. 异步演化
        for p in range(CONFIG["RECON_PASSES"]):
            for mask in [mask_b, mask_w]:
                neigh = torch.roll(s, 1, 2) + torch.roll(s, -1, 2) + torch.roll(s, 1, 3) + torch.roll(s, -1, 3)
                s_rand = (torch.rand_like(s) * 2 - 1) * step
                s_new = torch.clamp(s + s_rand, -A, A)

                dV = -0.5 * w1 * (s_new ** 2 - s ** 2) - 0.25 * w3 * (s_new ** 4 - s ** 4)
                dJ = -(s_new - s) * neigh

                accept = torch.rand_like(s) < torch.exp(-b * (dV + dJ))
                s[mask & accept] = s_new[mask & accept]

            if (p + 1) % 25 == 0:
                print(f"   进度: {p + 1}/{CONFIG['RECON_PASSES']}")

    # 关键：返回二值化后的场，作为下一轮迭代的“离散种子”
    return torch.sign(s)


# ==========================================================
# 4. 执行流水线
# ==========================================================
if __name__ == "__main__":
    # A. 加载模型
    model = FreePolyRGModel().to(CONFIG["DEVICE"])
    if os.path.exists(CONFIG["PARAM_FILE"]):
        model.load_state_dict(torch.load(CONFIG["PARAM_FILE"]))
        print(f"📂 已加载预训练权重: {CONFIG['PARAM_FILE']}")
    else:
        print("❌ 错误: 未找到权重文件。")
        exit()

    start_total = time.time()

    # B. 生成初始种子 (Iteration 0)
    current_field = get_physical_seed()

    # C. 递归迭代放大
    for i in range(CONFIG["N_ITERATIONS"]):
        current_field = iterate_reconstruct(current_field, model, i)
        # 显存清理，防止大尺寸图堆积
        torch.cuda.empty_cache()

    # D. 最终结果处理
    final_res = current_field.squeeze().cpu().numpy()
    print(f"\n✨ 迭代生成完成！")
    print(f"最终尺寸: {final_res.shape} | 总耗时: {time.time() - start_total:.2f}s")

    # E. 可视化
    custom_cmap = ListedColormap(['#93A5CB', '#F7A24F'])
    plt.figure(figsize=(15, 15), dpi=300)
    plt.imshow(final_res, cmap=custom_cmap, interpolation='nearest')
    plt.axis('off')

    save_path = f"{CONFIG['OUTPUT_PREFIX']}_iter{CONFIG['N_ITERATIONS']}.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"🖼️ 图像已保存至: {save_path}")
    plt.show()