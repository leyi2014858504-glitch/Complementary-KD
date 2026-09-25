# 在 Kaggle 上运行 CIFAR-10 实验

脚本 `image_distill.py` 是自包含的（Kaggle 预装 torch/torchvision，无需安装）。核心特性：

- **断点续跑**：每完成一个 run（约 2~6 分钟）立即写入输出目录下的
  `image_results.csv`、`image_curves.pkl`、`progress.log`，会话中断不丢进度
- **AMP 自动适配**：RTX 40xx 走 bf16；Kaggle T4 自动走 fp16 + GradScaler
  （该路径已在本地用 `IMAGE_AMP_DTYPE=fp16` 回归验证）
- **显存自限**：进程上限 30% 显存，不会挤爆共享卡

## 第 1 步：创建 Notebook

1. Kaggle → Create → New Notebook。
2. 右侧 Settings：
   - **Accelerator: GPU T4 x2**
     （**不要选 TPU**：本脚本基于 CUDA 的 autocast/GradScaler 编写，移植 TPU 需要
     torch_xla 重写并重新调通，且我们的负载是 300 次顺序小训练，用不上 TPU 的
     大规模并行优势，不值得在时间紧的 revision 里冒这个风险）
   - **Internet: On**（脚本需下载 CIFAR-10 到 `./data`，约 170MB）
3. 上传脚本：右侧 Add-ons → Files → Upload Files → 选 `image_distill.py`。

## 第 2 步：smoke 验证（约 5 分钟，务必先做）

```bash
!python image_distill.py --smoke --datasets cifar10 && rm -rf results_images_smoke
```

确认最后打印 `done`、无报错。这一步同时验证 Kaggle 环境的 fp16 路径。

## 第 3 步：正式运行——双卡并行，先跑 seeds 0–9

⚠️ Kaggle 里 `!cmd &` 会报 `Background processes not supported.`（IPython 的 `!`
走 system_piped，明确不支持后台进程）。改用下面两个 Python 单元：

**Cell 1（启动两个后台进程，立即返回）**

```python
import subprocess, os, sys
def launch(gpu, seed_start, n_seeds, outdir):
    log = open(f'gpu{gpu}.log', 'w')
    return subprocess.Popen(
        ['timeout', '28800', sys.executable, 'image_distill.py', '--datasets', 'cifar10',
         '--seed_start', str(seed_start), '--n_seeds', str(n_seeds), '--output', outdir],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
        stdout=log, stderr=subprocess.STDOUT)
p0 = launch(0, 0, 5, 'results_images_gpu0')
p1 = launch(1, 5, 5, 'results_images_gpu1')
print('launched', p0.pid, p1.pid)
```

**Cell 2（守候并打印进度；放在最后一个单元，这样 Save & Run All 也能正常等完）**

```python
import time
while p0.poll() is None or p1.poll() is None:
    time.sleep(120)
    for f in ('gpu0.log', 'gpu1.log'):
        lines = open(f).read().strip().splitlines()
        print(f, '->', lines[-1] if lines else '(starting)')
    print('----')
print('both finished')
```

两个进程都结束后点 **Save Version** 持久化输出。

> 简化替代（单卡，慢一倍，不需要 Cell 1）：`!timeout 28800 python image_distill.py --datasets cifar10 --n_seeds 10`

## 第 4 步：合并两卡结果（在 notebook 里跑）

```bash
!python - <<'EOF'
import glob, pickle, pandas as pd
merged = {}
for p in glob.glob('results_images_gpu*/image_curves.pkl'):
    merged.update(pickle.load(open(p, 'rb')))
pickle.dump(merged, open('results_images/image_curves.pkl', 'wb'))
df = pd.concat([pd.read_csv(p) for p in glob.glob('results_images_gpu*/image_results.csv')],
               ignore_index=True)
df.to_csv('results_images/image_results.csv', index=False)
print('merged', len(df), 'rows')
EOF
```

## 第 5 步：续跑到 20 seeds（配额允许时）

1. 新建 Notebook → Add Input → **Your Work → 选上一个版本**。
2. 恢复两个目录的缓存：

```bash
!cp -n -r /kaggle/input/<上一版本输出名>/results_images_gpu0 . && cp -n -r /kaggle/input/<上一版本输出名>/results_images_gpu1 .
```

3. 用第 3 步的 Cell 1 / Cell 2，只把 seeds 段改成：GPU0 `--seed_start 10 --n_seeds 5`、
   GPU1 `--seed_start 15 --n_seeds 5`（断点续跑，已完成的 run 自动跳过）。

## 第 6 步：取回结果

在 Notebook Output 面板下载 `results_images/`（合并后的 csv + pkl）——或直接把
`results_images_gpu0/`、`results_images_gpu1/` 两个文件夹都下载回来，我知道怎么合并。
放回本地项目目录后告诉我路径。

## 注意事项

- Kaggle 不支持 `!cmd &` 后台语法，必须用第 3 步的 `subprocess.Popen` 方案。
- 开始前可先 `!nvidia-smi -L` 确认能看到 2 张 T4。
- **T4 比 RTX 4060 慢**（约 0.4~0.5 倍）：CIFAR-10 全 20 seeds 预计 25~30 GPU 小时；
  双卡跑 10 seeds 约 6~8 小时（一晚上量级）。
- 每 9 小时会话会被强制结束——所以统一用 8 小时 `timeout` + Save Version。
- Kaggle GPU 配额约 30 小时/周；TPU 配额（20h/周）更少且不建议用（见第 1 步）。
- 如果 CIFAR-10 下载失败：确认 Internet 已打开，重跑同一条命令即可。
- 任何报错把最后 30 行贴给我。