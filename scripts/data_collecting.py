import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

# [2026-01-10] 指令：处理环境冲突


# ==========================================
# --- 配置参数 ---
# ==========================================
N = 2048
BETA_CRITICAL = 0.44068679
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生产计划
NUM_SAMPLES = 80
WARMUP_STEPS = 50000
INTERVAL_STEPS = 3000
SAVE_DIR = f"ising_data{N}"


# ==========================================
# --- 物理诊断工具 (EMP 监控) ---
# ==========================================
def get_physics_metrics(spins):
    """计算核心物理指标用于控制台实时显示"""
    # 磁化强度 M (Magnetization)
    m_abs = torch.abs(spins.mean()).item()

    # 最近邻关联 c10 (Energy related)
    c10_h = (spins * torch.roll(spins, 1, 2)).mean()
    c10_v = (spins * torch.roll(spins, 1, 3)).mean()
    c10 = ((c10_h + c10_v) / 2.0).item()

    # 对角关联 c11 用于计算 r21
    c11 = (spins * torch.roll(torch.roll(spins, 1, 2), 1, 3)).mean().item()
    r21 = c11 / (c10 + 1e-8)

    # Ising 能量密度 E = -J * sum(nn)
    energy = -c10

    return m_abs, c10, r21, energy


# ==========================================
# --- 核心更新逻辑 ---
# ==========================================
def run_metropolis(spins, mask_a, mask_b, steps, beta):
    kernel = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], device=DEVICE).float()

    for _ in range(steps):
        for m in [mask_a, mask_b]:
            neighbors = F.conv2d(F.pad(spins, (1, 1, 1, 1), mode='circular'), kernel)
            dE = 2.0 * spins * neighbors
            prob = torch.exp(-dE * beta)
            accept = (dE <= 0) | (torch.rand_like(spins) < prob)
            spins.mul_(1.0 - 2.0 * (m * accept.float()))
    return spins


def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    print(f"\n🚀 启动纯 GPU 棋盘格生产线 | 尺寸: {N}x{N} | 设备: {DEVICE}")
    print(f">>> 目标临界值参考: c10 ~ 0.7071 | r21 ~ 0.6050")
    print("-" * 85)

    # 1. 初始化
    spins = torch.where(torch.rand((1, 1, N, N), device=DEVICE) > 0.5, 1.0, -1.0)
    grid = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    mask_a = ((grid[0] + grid[1]) % 2 == 0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    mask_b = (1.0 - mask_a).to(DEVICE)

    # 2. 初始热化
    start_time = time.time()
    print(f"🔥 [Stage 1] 正在进行大规模热化 ({WARMUP_STEPS} 步)...")
    spins = run_metropolis(spins, mask_a, mask_b, WARMUP_STEPS, BETA_CRITICAL)

    m_abs, c10, r21, eng = get_physics_metrics(spins)
    print(f"✅ 热化完成! 当前状态: M={m_abs:.4f} | c10={c10:.4f} | r21={r21:.4f} | E={eng:.4f}")
    print("-" * 85)

    # 3. 采样循环与详细输出 (EMP)
    # 表头: Sample | TotalTime | Magnetization | c10 (Energy) | r21 (Geometry) | Status
    header = f"{'Sample':>8} | {'Elapsed':>8} | {'|M| (EMP)':>10} | {'c10 (E)':>9} | {'r21 (G)':>9} | {'Status'}"
    print(header)
    print("-" * 85)

    for i in range(401,960):
        # 演化
        spins = run_metropolis(spins, mask_a, mask_b, INTERVAL_STEPS, BETA_CRITICAL)

        # 物理诊断
        m_abs, c10, r21, eng = get_physics_metrics(spins)
        elapsed = time.time() - start_time

        # 状态判定 (诊断数据质量)
        status = "OK"
        if r21 > 0.75:
            status = "BLOCKY!"
        elif c10 < 0.65:
            status = "TOO HOT"
        elif m_abs > 0.2:
            status = "BIASED"

        # 实时打印
        print(f"{i:8d} | {elapsed:7.1f}s | {m_abs:10.4f} | {c10:9.4f} | {r21:9.4f} | {status}")

        # 保存结果
        save_path = os.path.join(SAVE_DIR, f"sample_{i:03d}.pt")
        torch.save(spins.squeeze().cpu().to(torch.int8), save_path)

    print("-" * 85)
    print(f"🎉 数据生产完毕！所有文件已存入 {SAVE_DIR} | 总耗时: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()