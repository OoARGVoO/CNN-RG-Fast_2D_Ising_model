import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# [2026-02-03] 遵照记忆指令：在 import os 下直接加上环境设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# --- 核心可调参数 ---
# ==========================================
KERNEL_SIZE = 9
SCALE_FACTOR = 3
BATCH_SIZE = 8
CROP_SIZE = 2046  # 3的倍数
LEARNING_RATE = 5e-4
STEPS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_MIN = 1e-8
T_MAX = STEPS

DATA_DIR = r"E:\coding temp\TEST\scripts\ising_data2048_batched"
PARAM_FILE = f"full_conn_proj_k{KERNEL_SIZE}_s{SCALE_FACTOR}_2048_26_2_3.pt"


# ==========================================
# --- 模型定义 ---
# ==========================================
class FullConnectProjectionRG(nn.Module):
    def __init__(self, k_size, s_factor):
        super().__init__()
        self.k_size = k_size
        self.s_factor = s_factor
        self.proj = nn.Conv2d(1, s_factor ** 2, k_size, padding=k_size // 2, bias=False)
        self.ps = nn.PixelShuffle(s_factor)
        self.w1 = nn.Parameter(torch.tensor([2.0], device=DEVICE))
        self.A = nn.Parameter(torch.tensor([1.0], device=DEVICE))
        with torch.no_grad():
            center = k_size // 2
            self.proj.weight.fill_(0.01)
            self.proj.weight[:, :, center - 1:center + 2, center - 1:center + 2] += 0.15

    def forward(self, x):
        m = self.proj(x)
        out = self.ps(m)
        return torch.clamp(self.w1 * out, -self.A, self.A)

    @torch.no_grad()
    def apply_symmetry(self):
        w = self.proj.weight
        w_sym = (w + w.flip(2) + w.flip(3) + w.transpose(2, 3)) / 4.0
        self.proj.weight.copy_(w_sym)


def get_correlations(field):
    def corr(dx, dy):
        return (field * torch.roll(torch.roll(field, dx, 2), dy, 3)).mean()

    c10 = corr(1, 0) + 1e-8
    r21 = corr(1, 1) / c10
    return c10, r21


# ==========================================
# --- 训练主程序 ---
# ==========================================
def run_training():
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not files: return

    print(f">>> 正在预载入数据至内存...")
    all_batches = []
    for f in tqdm(files, desc="Loading Data"):
        batch_data = torch.load(f, map_location='cpu').to(torch.int8)
        all_batches.append(batch_data)
    print(f">>> 产线就绪 | 样本数: {len(all_batches) * 20} | 设备: {DEVICE}")

    model = FullConnectProjectionRG(KERNEL_SIZE, SCALE_FACTOR).to(DEVICE)
    if os.path.exists(PARAM_FILE):
        print("载入现有权重...")
        model.load_state_dict(torch.load(PARAM_FILE))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=LR_MIN)

    target_c10, target_r21 = 0.7066, 0.8999

    # --- Line Search 辅助函数 ---
    def calc_loss(m, inp, target):
        pred = m(inp)
        c10, r21 = get_correlations(pred)
        l_phys = torch.pow(c10 - target_c10, 2) * 5000.0 + torch.pow(r21 - target_r21, 2) * 0.0
        l_pixel = nn.functional.mse_loss(pred, target)
        return l_phys + l_pixel * 1, c10, r21

    try:
        from tqdm import trange
        for step in range(STEPS + 1):
            batch_high_res = []
            for _ in range(BATCH_SIZE):
                batch_idx = np.random.randint(len(all_batches))
                data_batch = all_batches[batch_idx]
                sample_idx = np.random.randint(data_batch.shape[0])
                data = data_batch[sample_idx].float()
                while data.dim() < 4: data = data.unsqueeze(0)
                data = data[:, :1, :, :]
                if data.max() <= 1.01: data = data * 2.0 - 1.0
                h_max, w_max = data.shape[-2:]
                sy, sx = np.random.randint(0, h_max - CROP_SIZE), np.random.randint(0, w_max - CROP_SIZE)
                batch_high_res.append(data[:, :, sy:sy + CROP_SIZE, sx:sx + CROP_SIZE].to(DEVICE))

            high_res_target = torch.cat(batch_high_res, dim=0)
            low_res_input = nn.functional.avg_pool2d(high_res_target, SCALE_FACTOR)

            # 1. 计算当前 Loss 并反向传播获取梯度
            optimizer.zero_grad()
            current_loss, c10, r21 = calc_loss(model, low_res_input, high_res_target)
            current_loss.backward()

            # 2. --- Line Search 核心逻辑 ---
            # 备份当前参数
            backup_params = [p.data.clone() for p in model.parameters()]

            # 尝试步进
            optimizer.step()

            # 检查更新后的 Loss
            with torch.no_grad():
                new_loss, _, _ = calc_loss(model, low_res_input, high_res_target)

            # 如果新 Loss 变大（说明步子迈大了），则回退
            if new_loss > current_loss:
                for p, b_p in zip(model.parameters(), backup_params):
                    p.data.copy_(b_p)
                # 可以在此处适当调低下一轮的学习率（可选）
            # -------------------------------

            scheduler.step()

            if step % 20 == 0: model.apply_symmetry()
            if step % 100 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Step {step:5d} | Loss: {current_loss.item():.4f} | c10: {c10.item():.4f} | r21: {r21.item():.4f} | LR: {curr_lr:.2e}")

    except KeyboardInterrupt:
        print("\n[!] 中断保存")

    torch.save(model.state_dict(), PARAM_FILE)
    print(f">>> 结果已保存至: {PARAM_FILE}")


if __name__ == "__main__":
    from tqdm import tqdm

    run_training()