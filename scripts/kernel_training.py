import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 配置参数 ---
BLOCK_SIZE = 3
STEPS = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_FILE = f"rg_model_b{BLOCK_SIZE}_free_poly.pt"
CONFIG_FILE = f"rg_config_b{BLOCK_SIZE}_free_poly.json"


# ==========================================
# 1. 自由参数模型定义
# ==========================================
class FreePolyRGModel(nn.Module):
    def __init__(self):
        super(FreePolyRGModel, self).__init__()
        # 初始卷积核：均值分布 + 微小扰动
        self.kernel = nn.Parameter(torch.ones(1, 1, 3, 3, device=DEVICE) / 9.0 +
                                   torch.randn(1, 1, 3, 3, device=DEVICE) * 0.01)

        # Beta=1 模式下，w1 初始设定略高以承载临界相互作用
        self.w1 = nn.Parameter(torch.tensor([2.0], device=DEVICE))
        self.w3 = nn.Parameter(torch.tensor([-0.15], device=DEVICE))
        self.A = nn.Parameter(torch.tensor([1.0], device=DEVICE))

    def forward(self, x):
        m = torch.nn.functional.conv2d(x, self.kernel, stride=3)
        # 这里的 w1 和 w3 将学习包含 Beta 在内的总物理强度
        m_poly = self.w1 * m + self.w3 * torch.pow(m, 3)
        return torch.clamp(m_poly, -self.A, self.A)


# ==========================================
# 2. 深度长程关联函数计算 (全距离保留)
# ==========================================
def get_deep_correlations(field):
    def corr(dx, dy):
        return (field * torch.roll(torch.roll(field, dx, 2), dy, 3)).mean()

    c10 = corr(1, 0) + 1e-8

    r21 = corr(1, 1) / c10  # (1.41格)
    r31 = corr(2, 0) / c10  # (2格)
    r41 = corr(2, 2) / c10  # (2.8格)
    r51 = corr(3, 0) / c10  # (3格)
    r61 = corr(3, 3) / c10  # (4.2格)
    r71 = corr(4, 0) / c10  # (4格)
    r81 = corr(4, 4) / c10  # (5.6格)

    return r21, r31, r41, r51, r61, r71, r81


# ==========================================
# 3. 训练主程序
# ==========================================
def run_fitting():
    if not os.path.exists('../../critical_samples.pt'):
        print("错误：未找到数据文件！")
        return

    # 加载数据
    raw_data = torch.load('../../critical_samples.pt').to(DEVICE).unsqueeze(1).float()

    with torch.no_grad():
        # 提取原始数据的全距离关联作为目标
        targets = get_deep_correlations(raw_data)
        t_r21, t_r31, t_r41, t_r51, t_r61, t_r71, t_r81 = targets

    print("-" * 65)
    print(f"模式: 【全关联拟合模式 - 深度物理特征锁定】")
    print(f"目标 R(dx,dy) 监控:")
    print(f"  短程 R21(1.41): {t_r21.item():.4f} | 中程 R51(3.0): {t_r51.item():.4f}")
    print(f"  长程 R71(4.0):  {t_r71.item():.4f} | 极远 R81(5.6): {t_r81.item():.4f}")
    print("-" * 65)

    model = FreePolyRGModel().to(DEVICE)

    if os.path.exists(PARAM_FILE):
        print(f"加载已有参数: {PARAM_FILE} ...")
        model.load_state_dict(torch.load(PARAM_FILE))

    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    start_time = time.time()
    for epoch in range(STEPS + 1):
        optimizer.zero_grad()

        s_prime = model(raw_data)
        res = get_deep_correlations(s_prime)
        r21, r31, r41, r51, r61, r71, r81 = res

        # 关联函数多级损耗计算
        loss_near = torch.pow(r21 - t_r21, 2) * 1000.0 + torch.pow(r31 - t_r31, 2) * 800.0
        loss_mid = torch.pow(r41 - t_r41, 2) * 400.0 + torch.pow(r51 - t_r51, 2) * 300.0
        loss_far = torch.pow(r71 - t_r71, 2) * 150.0 + torch.pow(r81 - t_r81, 2) * 100.0

        loss_norm = torch.pow(model.kernel.sum() - 1.0, 2) * 300.0
        loss_mag = torch.pow(torch.abs(s_prime).mean() - 0.7, 2) * 25.0

        total_loss = loss_near + loss_mid + loss_far + loss_norm + loss_mag
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # 核心硬归一化
        with torch.no_grad():
            model.kernel.copy_(model.kernel / (model.kernel.sum() + 1e-9))

        if epoch % 250 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.5f} | Time: {elapsed:.1f}s")
            print(
                f"  R21: {r21.item():.4f} (目标:{t_r21.item():.4f}) | R51: {r51.item():.4f} (目标:{t_r51.item():.4f})")
            print(
                f"  R71: {r71.item():.4f} (目标:{t_r71.item():.4f}) | R81: {r81.item():.4f} (目标:{t_r81.item():.4f})")
            print("-" * 45)

    # 保存权重
    torch.save(model.state_dict(), PARAM_FILE)

    final_k = model.kernel.detach().cpu().squeeze().numpy()
    print("-" * 65)
    print("【拟合完成】全量程优化卷积核:")
    print(final_k)

    config = {
        "kernel": final_k.tolist(),
        "w1": model.w1.item(),
        "w3": model.w3.item(),
        "A": model.A.item(),
        "beta": 1.0
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n物理配置文件已成功导出至 {CONFIG_FILE}")


if __name__ == "__main__":
    run_fitting()