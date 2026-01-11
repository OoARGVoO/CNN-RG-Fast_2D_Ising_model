import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time


# 物理参数
equisteps = 15000
N = 1500
TEMP_CRITICAL = 2.26918531421  # 精确临界温度
J = 1.0
BETA = 1.0 / TEMP_CRITICAL
NUM_SAMPLES = 100  # 收集100个样本
INTERVAL_STEPS = 3000  # 样本间的间隔步数（确保独立性）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_checkerboard_masks(n):
    x = torch.arange(n, device=device).reshape(-1, 1)
    y = torch.arange(n, device=device).reshape(1, -1)
    return (x + y) % 2 == 0, (x + y) % 2 == 1


def generate_samples():
    print(f"正在准备生成临界态样本，使用设备: {device}")
    spins = (torch.randint(0, 2, (N, N), device=device, dtype=torch.float32) * 2 - 1)
    mask_b, mask_w = get_checkerboard_masks(N)

    # 预热：让系统先达到平衡
    print("正在进行预热（平衡过程）...")
    for _ in range(equisteps):
        for mask in [mask_b, mask_w]:
            neighbors = torch.roll(spins, 1, 0) + torch.roll(spins, -1, 0) + \
                        torch.roll(spins, 1, 1) + torch.roll(spins, -1, 1)
            dE = 2 * J * spins * neighbors
            prob = torch.exp(-dE * BETA)
            accept = (dE <= 0) | (torch.rand((N, N), device=device) < prob)
            spins[mask & accept] *= -1

    # 正式收集样本
    samples = []
    for i in range(NUM_SAMPLES):
        for _ in range(INTERVAL_STEPS):
            for mask in [mask_b, mask_w]:
                neighbors = torch.roll(spins, 1, 0) + torch.roll(spins, -1, 0) + \
                            torch.roll(spins, 1, 1) + torch.roll(spins, -1, 1)
                dE = 2 * J * spins * neighbors
                prob = torch.exp(-dE * BETA)
                accept = (dE <= 0) | (torch.rand((N, N), device=device) < prob)
                spins[mask & accept] *= -1

        samples.append(spins.clone().cpu())
        print(f"进度: {i + 1}/{NUM_SAMPLES} 样本已收集")

    # 保存数据
    torch.save(torch.stack(samples), '../../critical_samples.pt')
    print("所有样本已保存至 critical_samples.pt")


if __name__ == "__main__":
    generate_samples()