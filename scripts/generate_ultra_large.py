import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc  # 导入垃圾回收机制
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datetime import datetime
import time

# [2026-01-10] 指令：处理 OpenMP 运行时冲突


# 必须添加：解除 PIL 的 1.7 亿像素限制
Image.MAX_IMAGE_PIXELS = None


# ==========================================
# --- 视觉艺术与基础配置 ---
# ==========================================
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


HEX_SPIN_UP = "#FFA430"
HEX_SPIN_DOWN = "#CCCCCC"
SPIN_UP_COLOR = hex_to_rgb(HEX_SPIN_UP)
SPIN_DOWN_COLOR = hex_to_rgb(HEX_SPIN_DOWN)

TILE_N = 1
PREVIEW_L = 41472

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = r"E:\coding temp\TEST\scripts\RG_MultiStage_Results_Ultra_Large"
KERNEL_FILE = r"E:\coding temp\TEST\scripts\full_conn_proj_k9_s3_2048_26_2_3.pt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

START_SIZE = 512
STAGES = 4
SCALE_FACTOR = 3
BETA = 0.4406868
SEED_MC_STEPS = 1000
REFINE_STEPS_LIST = [20, 35, 50, 75]

# 16GB 显存优化参数
MC_TILE_SIZE = 4096
INFER_TILE_SIZE = 8192
PAD = 24


# ==========================================
# --- 核心算子：分块 MC ---
# ==========================================
def checkerboard_mc(spin, beta, steps):
    if steps <= 0: return spin
    if spin.dtype != torch.int8:
        spin = spin.to(torch.int8)

    B, C, H, W = spin.shape
    for step in range(steps):
        start_time = time.time()
        for phase in [0, 1]:
            for i in range(0, H, MC_TILE_SIZE):
                for j in range(0, W, MC_TILE_SIZE):
                    i_end, j_end = min(i + MC_TILE_SIZE, H), min(j + MC_TILE_SIZE, W)
                    t_i = torch.arange(i - 1, i_end + 1) % H
                    t_j = torch.arange(j - 1, j_end + 1) % W

                    sub_grid = spin[:, :, t_i][:, :, :, t_j].to(DEVICE).float()
                    core = sub_grid[:, :, 1:-1, 1:-1]
                    neighbors = (sub_grid[:, :, 0:-2, 1:-1] + sub_grid[:, :, 2:, 1:-1] +
                                 sub_grid[:, :, 1:-1, 0:-2] + sub_grid[:, :, 1:-1, 2:])

                    r = torch.arange(i, i_end, device=DEVICE).view(-1, 1)
                    c = torch.arange(j, j_end, device=DEVICE).view(1, -1)
                    mask = ((r + c) % 2 == phase).float()

                    dE = 2.0 * core * neighbors
                    prob = torch.exp(-beta * dE)
                    accept = (torch.rand(core.shape, device=DEVICE) < prob).float()
                    new_core = torch.where(mask > 0, torch.where(accept > 0, -core, core), core)

                    spin[:, :, i:i_end, j:j_end] = new_core.to(torch.int8).cpu()

            torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - start_time
        if (step + 1) % 5 == 0 or step == 0:
            print(f"   [MC Progress] Step {step + 1}/{steps} | Time/Step: {elapsed:.2f}s")
    return spin


# ==========================================
# --- 内存优化版分块推理 + 投影 ---
# ==========================================
@torch.inference_mode()
def tiled_predict_and_project(model, x, tile_size, pad, s_factor):
    """
    [2026-02-03 优化]
    直接在 GPU 内完成投影，返回 int8 结果。
    解决 L=124416 时 float32 导致的内存溢出问题。
    """
    B, C, H, W = x.shape
    out_h, out_w = H * s_factor, W * s_factor
    # 直接在 CPU 申请 int8 空间 (约 14.4GB)
    output = torch.zeros((B, C, out_h, out_w), dtype=torch.int8, device='cpu')

    x_padded = torch.cat([x[:, :, -pad:], x, x[:, :, :pad]], dim=2)
    x_padded = torch.cat([x_padded[:, :, :, -pad:], x_padded, x_padded[:, :, :, :pad]], dim=3)

    num_tiles_h = (H + tile_size - 1) // tile_size
    num_tiles_w = (W + tile_size - 1) // tile_size
    total_tiles = num_tiles_h * num_tiles_w
    current_tile = 0

    print(f"   [Inference & Project] Total tiles: {total_tiles}")
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            i_start, i_end = i, min(i + tile_size, H)
            j_start, j_end = j, min(j + tile_size, W)

            # 1. 送入 GPU 进行卷积
            tile_in = x_padded[:, :, i:i_end + 2 * pad, j:j_end + 2 * pad].to(DEVICE).float()
            tile_out = model(tile_in)

            # 2. 在 GPU 上直接计算投影概率 P = (1 + phi) / 2
            probs = (1.0 + tile_out) / 2.0
            projected_tile = torch.where(torch.rand_like(probs) < probs,
                                         torch.tensor(1, dtype=torch.int8, device=DEVICE),
                                         torch.tensor(-1, dtype=torch.int8, device=DEVICE))

            # 3. 裁剪并存回 CPU
            cut = pad * s_factor
            output[:, :, i * s_factor:i_end * s_factor, j * s_factor:j_end * s_factor] = \
                projected_tile[:, :, cut:-cut, cut:-cut].cpu()

            current_tile += 1
            if current_tile % 5 == 0 or current_tile == total_tiles:
                print(f"   [Inference] Processed {current_tile}/{total_tiles} tiles...")

            del tile_in, tile_out, probs, projected_tile
            torch.cuda.empty_cache()

    return output


# ==========================================
# --- 模型与主流程 ---
# ==========================================
class FullConnectProjectionRG(nn.Module):
    def __init__(self, k_size=9, s_factor=3):
        super().__init__()
        self.proj = nn.Conv2d(1, s_factor ** 2, k_size, padding=k_size // 2, bias=False)
        self.ps = nn.PixelShuffle(s_factor)
        self.w1 = nn.Parameter(torch.tensor([1.0]))
        self.A = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return torch.clamp(self.w1 * self.ps(self.proj(x)), -self.A, self.A)




def main():
    print("=" * 60)
    print(f"Ising RG System | Start Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"16GB Memory Optimized Mode | [2026-02-02] Energy Target: Nearest Neighbor")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = FullConnectProjectionRG().to(DEVICE)
    if os.path.exists(KERNEL_FILE):
        model.load_state_dict(torch.load(KERNEL_FILE, map_location=DEVICE))
        print(f">>> [System] 成功载入算子")
    model.eval()

    # 1. 种子
    current_spin = torch.where(torch.rand((1, 1, START_SIZE, START_SIZE)) < 0.5,
                               torch.tensor(1, dtype=torch.int8),
                               torch.tensor(-1, dtype=torch.int8))
    current_spin = checkerboard_mc(current_spin, BETA, SEED_MC_STEPS)

    # 2. 演化
    for s in range(1, STAGES + 1):
        steps = REFINE_STEPS_LIST[s - 1]
        target_l = current_spin.shape[-1] * SCALE_FACTOR
        print(f"\n--- [Stage {s}/{STAGES}] L={target_l} ---")

        # [关键改动] 直接生成投影后的 int8 自旋
        # 修改后：加上 .clone() 彻底切断与推理模式的联系
        rg_init = tiled_predict_and_project(model, current_spin, INFER_TILE_SIZE, PAD, SCALE_FACTOR).clone()

        del current_spin
        gc.collect()

        print(f"   [Action] 正在执行物理修复 (Steps: {steps})...")
        current_spin = checkerboard_mc(rg_init, BETA, steps)

        del rg_init
        gc.collect()

    # 保存配置
    final_l = current_spin.shape[-1]
    config_path = os.path.join(OUTPUT_DIR, f"SpinConfig_L{final_l}_{timestamp}.npy")
    print(f"\n>>> 导出配置数据 (约 {final_l ** 2 / (1024 ** 3):.2f} GB)...")
    np.save(config_path, current_spin.squeeze().numpy())
    print(f">>> [Success] 已保存: {config_path}")



if __name__ == "__main__":
    main()