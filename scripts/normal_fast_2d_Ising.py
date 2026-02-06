import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap

N = 13500
TEMP = 2.26918
NUM_STEPS = 600000
J_CONSTANT = 1.0
K_CONSTANT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"检测到设备: {device} ({torch.cuda.get_device_name(0)})")


class IsingGPU:
    def __init__(self, n, temp):
        self.n = n
        self.temp = temp
        self.beta = 1.0 / (K_CONSTANT * temp)
        self.spins = (torch.randint(0, 2, (n, n), device=device, dtype=torch.float32) * 2 - 1)

        x = torch.arange(n, device=device).reshape(-1, 1)
        y = torch.arange(n, device=device).reshape(1, -1)
        checkerboard = (x + y) % 2
        self.mask_black = (checkerboard == 0)
        self.mask_white = (checkerboard == 1)
        self.energy_history = []
        self.mag_history = []

    def get_total_energy(self):
        neighbors = torch.roll(self.spins, -1, dims=0) + torch.roll(self.spins, -1, dims=1)
        energy = -J_CONSTANT * (self.spins * neighbors).sum()
        return energy.item()

    def run_simulation(self, steps, sample_interval=100):
        print(f"开始蒙特卡洛模拟... 晶格大小: {self.n}x{self.n}, 步数: {steps}")
        torch.cuda.synchronize()
        start_t = time.time()

        for s in range(steps):
            for mask in [self.mask_black, self.mask_white]:
                neigh_sum = (
                        torch.roll(self.spins, 1, 0) + torch.roll(self.spins, -1, 0) +
                        torch.roll(self.spins, 1, 1) + torch.roll(self.spins, -1, 1)
                )
                dE = 2 * J_CONSTANT * self.spins * neigh_sum
                prob = torch.exp(-dE * self.beta)
                rand_vals = torch.rand((self.n, self.n), device=device)
                accept = (dE <= 0) | (rand_vals < prob)
                self.spins[mask & accept] *= -1

            if s % sample_interval == 0:
                mag = torch.mean(self.spins).item()
                self.mag_history.append(mag)
                if s % (sample_interval * 5) == 0:
                    print(f"进度: {s / steps * 100:>3.1f}% | 磁化强度: {mag:>7.4f}")

        torch.cuda.synchronize()
        end_t = time.time()
        print(f"\n模拟结束！GPU 总计算耗时: {end_t - start_t:.4f} 秒")
        return end_t - start_t

    def save_config(self, filename="final_spin_config.pt"):

        output_data = self.spins.cpu()
        torch.save(output_data, filename)
        print(f"最终自旋配置已保存至: {filename}")

    def plot_result(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        custom_cmap = ListedColormap(['#F7A24F', '#93A5CB'])
        plt.imshow(self.spins.cpu().numpy(), cmap=custom_cmap)
        plt.title(f"Final Configuration (T={self.temp})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(self.mag_history, color='#93A5CB', label='Magnetization')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title("Magnetization Evolution")
        plt.xlabel("Sample Point")
        plt.ylabel("M")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = IsingGPU(n=N, temp=TEMP)
    duration = sim.run_simulation(steps=NUM_STEPS)
    final_mag = torch.mean(sim.spins).item()
    print(f"最终平均磁化强度: {final_mag:.6f}")
    sim.save_config("final_spin_config_1000.pt")
    sim.plot_result()