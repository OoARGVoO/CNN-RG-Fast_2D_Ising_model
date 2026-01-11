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
    "SEED_STEPS": 3000,

    # 迭代放大配置
    "N_ITERATIONS": 2,

    # --- 手动设置每一层的弛豫步数 ---
    "RECON_PASSES_LIST": [50,100],  # 退火需要足够的步数才有效果

    # 退火配置
    "BETA_START": 0.5,  # 初始高温 (打破块状结构)
    "BETA_END": 10.0,  # 最终低温 (锁定物理纹理)

    "STEP_SIZE": 0.3,
    "OUTPUT_PREFIX": "ising_iterative"
}


# ==========================================================
# 2. 模型定义 (保持不变)
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
    """生成第一代物理种子"""
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


def iterate_reconstruct(field, model, iter_idx, num_passes):
    """带退火策略的单次放大重构"""
    _, _, h_in, w_in = field.shape
    h_out, w_out = h_in * 3, w_in * 3
    print(f"🚀 [迭代 {iter_idx + 1}] 重构: {h_in}x{w_in} -> {h_out}x{w_out} | 步数: {num_passes}")

    with torch.no_grad():
        # 1. 转置卷积上采样
        s = F.conv_transpose2d(field, model.kernel, stride=3)

        # --- 在投影后注入微小噪声，打破 3x3 的完全对称 ---
        s = s + torch.randn_like(s) * 0.1

        s = torch.clamp(model.w1 * s + model.w3 * torch.pow(s, 3), -model.A.item(), model.A.item())

        x, y = torch.meshgrid(torch.arange(h_out, device=CONFIG["DEVICE"]),
                              torch.arange(w_out, device=CONFIG["DEVICE"]), indexing='ij')
        mask_b = ((x + y) % 2 == 0).unsqueeze(0).unsqueeze(0)
        mask_w = ~mask_b
        w1, w3, A = model.w1.item(), model.w3.item(), model.A.item()
        step = CONFIG["STEP_SIZE"]

        # 2. 退火 MCMC 演化
        for p in range(num_passes):
            # 线性插值计算当前 Beta (从 BETA_START 到 BETA_END)
            alpha = p / max(1, num_passes - 1)
            curr_beta = CONFIG["BETA_START"] * (1 - alpha) + CONFIG["BETA_END"] * alpha

            for mask in [mask_b, mask_w]:
                neigh = torch.roll(s, 1, 2) + torch.roll(s, -1, 2) + torch.roll(s, 1, 3) + torch.roll(s, -1, 3)
                s_rand = (torch.rand_like(s) * 2 - 1) * step
                s_new = torch.clamp(s + s_rand, -A, A)

                dV = -0.5 * w1 * (s_new ** 2 - s ** 2) - 0.25 * w3 * (s_new ** 4 - s ** 4)
                dJ = -(s_new - s) * neigh

                # 使用当前退火温度对应的 Beta
                accept = torch.rand_like(s) < torch.exp(-curr_beta * (dV + dJ))
                s[mask & accept] = s_new[mask & accept]

            if num_passes >= 50 and (p + 1) % 50 == 0:
                print(f"   进度: {p + 1}/{num_passes} | 当前 Beta: {curr_beta:.2f}")

    return torch.sign(s)


# ==========================================================
# 4. 执行流水线
# ==========================================================
if __name__ == "__main__":
    # --- 随机种子设定 (使用当前系统时间，确保每次不同) ---
    random_seed = int(time.time() * 1000) % 100000
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    # 允许非确定性算法以获得自然的随机演化
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = FreePolyRGModel().to(CONFIG["DEVICE"])
    if os.path.exists(CONFIG["PARAM_FILE"]):
        model.load_state_dict(torch.load(CONFIG["PARAM_FILE"]))
        print(f"📂 已加载权重 | 随机种子: {random_seed}")
    else:
        print("❌ 错误: 未找到权重文件。")
        exit()

    start_total = time.time()
    current_field = get_physical_seed()

    # C. 递归迭代放大
    for i in range(CONFIG["N_ITERATIONS"]):
        current_passes = CONFIG["RECON_PASSES_LIST"][i]
        current_field = iterate_reconstruct(current_field, model, i, current_passes)
        torch.cuda.empty_cache()

    final_res = current_field.squeeze().cpu().numpy()
    print(f"\n✨ 迭代完成！总耗时: {time.time() - start_total:.2f}s")

    custom_cmap = ListedColormap(['#93A5CB', '#F7A24F'])
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(final_res, cmap=custom_cmap, interpolation='nearest')
    plt.axis('off')

    # 保存文件名包含种子，方便区分
    save_path = f"{CONFIG['OUTPUT_PREFIX']}_seed_{random_seed}.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
    print(f"🖼️ 图像已保存至: {save_path}")
    plt.show()