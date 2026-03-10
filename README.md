# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled entirely through `program.md`. This fork preserves every design rule — 5-minute wall-clock budget, single mutable `train.py`, one metric (`val_bpb`), keep/revert via git — and runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch or CUDA required.

## Quick start

Requirements: Apple Silicon Mac (M1/M2/M3/M4), Python 3.10+, uv.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (one-time)
uv run prepare.py

# Run a single training experiment (~7 min including compile + eval)
uv run train.py

# Start autonomous research
# Point Claude Code (or any agent) at program.md and let it go
```

## How it works

Same as the original. Three files that matter:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation. Not modified.
- **`train.py`** — model, optimizer, training loop. The agent edits this.
- **`program.md`** — agent instructions. Point your agent here.

The agent reads `program.md`, modifies `train.py`, runs a 5-minute experiment, checks `val_bpb`, and commits or reverts. Repeat overnight. Wake up to results.

## Results on M1 Mac Studio (48GB)

Starting from the upstream default configuration and running the autoresearch loop:

| Experiment | Change           | val_bpb | Action |
| ---------- | ---------------- | ------- | ------ |
| baseline   | default config   | 2.667   | keep   |
| 1          | halve batch size | 2.589   | keep   |
| 2          | 10x matrix LR    | 2.534   | keep   |
| 3          | depth 8 → 4      | 1.808   | keep   |

Key finding: Apple Silicon throughput in a 5-minute window favors smaller, faster-training models. The autoresearch loop discovered this automatically — more optimizer steps beat more parameters when compute time is fixed.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon, unified memory.
- **AdamW only.** Muon optimizer port is future work.
- **Smaller eval token budget.** Reduced for faster iteration (~52s eval vs ~11min on full budget). Same `evaluate_bpb` function from `prepare.py`.
- **~7 min experiment cycle.** 5 min training + ~11s compile + ~52s eval. Expect ~8-9 experiments/hour, ~70 overnight.
- **MFU reporting is placeholder.** No Apple Silicon FLOPs benchmark exists equivalent to H100_BF16_PEAK_FLOPS. `peak_vram_mb` reports MLX unified memory.

## Recommended Defaults

Based on overnight results across three machines:

```python
DEPTH = 4
TOTAL_BATCH_SIZE = 2**14
MLP_EXPANSION = 3
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.1
USE_MUON = False          # True for constrained hardware
NS_STEPS = 3              # if using Muon
```

Starting points. The loop will find better settings for your hardware.

---

## อธิบายศัพท์เทคนิค (Technical Glossary — ภาษาไทย)

### val_bpb (Bits Per Byte) — ตัวชี้วัดหลัก

**Bits Per Byte** คือหน่วยวัดว่า "โมเดลเก่งแค่ไหนในการทำนายตัวอักษรถัดไป"

- ลองนึกภาพว่าคุณเล่นเกมทายคำ — ถ้าเก่งมากก็ใช้คำใบ้น้อย ถ้าไม่เก่งก็ต้องการคำใบ้เยอะ
- BPB = จำนวน "bits" (หน่วยข้อมูล) ที่โมเดลต้องการเพื่อเข้ารหัส 1 byte ของข้อความ
- **ยิ่งต่ำยิ่งดี** — หมายความว่าโมเดล "เข้าใจ" ภาษาได้ดีขึ้น
- ตัวอย่าง: val_bpb ลดจาก 2.667 → 1.808 = โมเดลทำนายแม่นขึ้น ~32%

### AdamW Optimizer — วิธีปรับค่าน้ำหนัก

**Optimizer** คือกลยุทธ์ในการปรับค่าน้ำหนัก (weights) ของ neural network ให้ผลลัพธ์ดีขึ้นทีละนิด

ลองนึกภาพ: **คุณยืนบนเขาในความมืด ต้องหาทางลงไปจุดต่ำสุด (loss ต่ำสุด)**

**Adam** ทำ 2 อย่าง:
1. **จำทิศทาง (Momentum)** — ดูว่า gradient ชี้ไปทางไหนซ้ำๆ แล้วไปทางนั้น เหมือนลูกบอลกลิ้งลงเนิน ไม่หยุดกะทันหันเมื่อเจอหลุมเล็กๆ
2. **ปรับขนาดก้าว (Adaptive LR)** — parameter ที่ไม่ค่อยเปลี่ยน → ก้าวใหญ่ / parameter ที่เปลี่ยนถี่ → ก้าวเล็กลง

**AdamW** เพิ่ม **Weight Decay** — ทุกรอบจะ "หดค่า weight ลง x%" ก่อนแล้วค่อยปรับ เพื่อป้องกันไม่ให้โมเดลท่องจำข้อมูล (overfitting)

ในโปรเจ็คนี้ AdamW ถูก customize ให้แยก learning rate สำหรับ parameter แต่ละกลุ่ม — embeddings, weight matrices, scalars ต่างก็ได้ LR ที่เหมาะกับตัวเอง

### Muon Optimizer — ก้าวน้อยแต่แม่น

**Muon** (Momentum + Unitary) ออกแบบมาให้แต่ละ step "มีคุณภาพสูงขึ้น" แลกกับความช้าต่อ step

|           | AdamW                     | Muon                               |
| --------- | ------------------------- | ---------------------------------- |
| เปรียบเทียบ | รถยนต์ทั่วไป — วิ่งเร็ว เชื่อถือได้ | รถ F1 — ทุกโค้งแม่นกว่า แต่ซ่อมบำรุงแพงกว่า |
| จุดแข็ง     | train ได้หลาย step         | ทุก step ได้ค่ามากกว่า                 |
| เหมาะกับ   | hardware แรง (M4 Max)     | hardware จำกัด (Mac Mini)            |

Muon ใช้ **Newton-Schulz iteration** เพื่อ "ปรับทิศทาง" gradient ให้ดีที่สุด — จาก overnight results พบว่า NS_STEPS=3 ดีกว่า 5 เพราะทำเร็วกว่าจึง train ได้มากรอบกว่า

### MFU (Model FLOPs Utilization) — ประสิทธิภาพการใช้ hardware

**MFU** วัดว่า hardware ถูกใช้ได้กี่เปอร์เซ็นต์ของศักยภาพสูงสุด

```
MFU = (FLOPs ที่ model ใช้จริง/วินาที) ÷ (FLOPs สูงสุดของ hardware) × 100%
```

- MFU 50-60% = ใช้ hardware คุ้มค่า โค้ดมีประสิทธิภาพ
- MFU 10-20% = มี bottleneck (data loading ช้า, memory bandwidth จำกัด)
- ในโปรเจ็คนี้ MFU = 0.00 (placeholder) เพราะ **Apple Silicon ไม่มี benchmark FLOPs อย่างเป็นทางการ** ต่างจาก NVIDIA H100 ที่มีตัวเลขชัดเจน

### Autoresearch Loop — วิจัยอัตโนมัติข้ามคืน

กลไกหลักของโปรเจ็คนี้ — ให้ AI agent วนลูปทำการทดลองเอง:

```
แก้โค้ด → train 5 นาที → วัดผล (val_bpb) → ดีขึ้น? → keep / ไม่ดี? → revert → ทำซ้ำ
```

- ใช้เวลาราว **7 นาทีต่อ experiment** (5 นาที train + 2 นาที compile/eval)
- **8-9 experiments ต่อชั่วโมง**, **~70 experiments ข้ามคืน**
- ใช้ Git เป็น experiment tracker — keep = commit, discard = `git reset --hard`

**Key insight จากการทดลอง:** ในเวลา 5 นาทีบน Apple Silicon, โมเดลเล็ก (depth 4) ที่ train ได้หลาย step ชนะโมเดลใหญ่ (depth 8) ที่ train ได้น้อย step เสมอ

---

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. Original copyright preserved. See [LICENSE](LICENSE).
