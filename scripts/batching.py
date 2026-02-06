import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import glob
from tqdm import tqdm

SOURCE_DIR = r"E:\coding temp\TEST\scripts\ising_data2048"
OUTPUT_DIR = r"E:\coding temp\TEST\scripts\ising_data2048_batched"
SAMPLES_PER_BATCH = 20
TOTAL_BATCHES = 48


def combine_to_batches():
    file_list = sorted(glob.glob(os.path.join(SOURCE_DIR, "sample_*.pt")))

    if len(file_list) < SAMPLES_PER_BATCH * TOTAL_BATCHES:
        print(f"警告：样本数量不足 {SAMPLES_PER_BATCH * TOTAL_BATCHES} 个，仅找到 {len(file_list)} 个。")
        num_batches = len(file_list) // SAMPLES_PER_BATCH
    else:
        num_batches = TOTAL_BATCHES

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"📦 准备将样本合并为 {num_batches} 个 Batch 文件，每个含 {SAMPLES_PER_BATCH} 个样本...")

    for b in range(num_batches):
        batch_tensors = []
        start_idx = b * SAMPLES_PER_BATCH
        end_idx = (b + 1) * SAMPLES_PER_BATCH

        current_batch_files = file_list[start_idx:end_idx]

        for f in current_batch_files:
            try:
                data = torch.load(f, map_location='cpu').to(torch.int8)
                if data.dim() == 3:
                    data = data.squeeze(0)
                batch_tensors.append(data)
            except Exception as e:
                print(f"读取 {f} 失败: {e}")

        if len(batch_tensors) > 0:
            final_batch = torch.stack(batch_tensors, dim=0)
            save_path = os.path.join(OUTPUT_DIR, f"batch_{b:03d}.pt")
            torch.save(final_batch, save_path)
            print(f"已生成 {save_path} | 形状: {final_batch.shape}")

    print(f"\n 合并完成！所有 Batch 文件已存入 {OUTPUT_DIR}")


if __name__ == "__main__":
    combine_to_batches()