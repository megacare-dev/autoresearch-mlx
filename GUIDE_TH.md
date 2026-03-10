# 📘 คู่มือ Autoresearch-MLX ฉบับภาษาไทย

## โปรเจ็คนี้คืออะไร?

**Autoresearch-MLX** คือระบบ "วิจัย AI อัตโนมัติ" ที่วิ่งบน Mac โดยเฉพาะ

แนวคิดหลัก: ให้ AI agent (เช่น Claude Code, Gemini CLI) ทำการทดลอง machine learning แทนเรา — ปรับโค้ด, train โมเดล 5 นาที, วัดผล, เก็บผลดี/ทิ้งผลแย่ — แล้วทำซ้ำไปเรื่อยๆ **ข้ามคืนได้โดยไม่ต้องดูแล**

ต้นฉบับมาจาก [Andrej Karpathy](https://github.com/karpathy/autoresearch) ซึ่งใช้ PyTorch + NVIDIA GPU — โปรเจ็คนี้ port มาวิ่งบน Apple Silicon ผ่าน [MLX](https://github.com/ml-explore/mlx) โดยไม่ต้องพึ่ง PyTorch หรือ CUDA เลย

---

## โครงสร้างโปรเจ็ค

มีไฟล์สำคัญแค่ 3 ไฟล์:

| ไฟล์          | หน้าที่                                     | แก้ได้?          |
| ------------ | ---------------------------------------- | -------------- |
| `prepare.py` | โหลดข้อมูล, สร้าง tokenizer, ประเมินผล       | ❌ ห้ามแก้        |
| `train.py`   | สร้างโมเดล, ตั้งค่า optimizer, วน train loop | ✅ agent แก้ไฟล์นี้ |
| `program.md` | คำสั่งสำหรับ AI agent (วิธีทำ experiment)        | ❌ อ่านอย่างเดียว  |

ไฟล์เสริม:
- `results.tsv` — ตารางบันทึกผลทดลอง
- `pyproject.toml` — dependencies ของโปรเจ็ค
- `README.md` — เอกสารประกอบ (EN + TH)

---

## อธิบายศัพท์เทคนิค

### val_bpb (Bits Per Byte) — ตัวชี้วัดหลัก

**ตัวชี้วัดเดียว** ที่โปรเจ็คนี้สนใจ — วัดว่า "โมเดลเก่งแค่ไหนในการทำนายข้อความ"

ลองนึกภาพเกมทายคำ:
- ถ้าโมเดลเก่ง → ใช้ข้อมูลน้อยก็ทำนายได้ → BPB ต่ำ
- ถ้าโมเดลแย่ → ต้องใช้ข้อมูลเยอะ → BPB สูง

**ยิ่งต่ำยิ่งดี** — ตัวอย่างจากผลทดลอง:

| สถานะ         | val_bpb | หมายความว่า        |
| ------------- | ------- | ----------------- |
| baseline      | 2.667   | จุดเริ่มต้น           |
| depth 4       | 1.808   | ดีขึ้น ~32%          |
| M4 Max ข้ามคืน  | 1.295   | ดีสุดที่ทำได้           |
| H100 (NVIDIA) | 0.998   | อ้างอิงจาก GPU แรงๆ |

BPB ดีกว่า loss ตรงที่ **เปรียบเทียบข้าม tokenizer ได้** — ไม่ว่า vocab size จะเป็น 8K หรือ 128K ก็เทียบกันได้

### Transformer — สถาปัตยกรรมของโมเดล

โมเดลในโปรเจ็คนี้คือ **GPT** (Generative Pre-trained Transformer) ขนาดเล็ก ทำงานแบบนี้:

```
ข้อความ → แปลงเป็นตัวเลข (token) → ผ่าน Transformer หลายชั้น → ทำนายคำถัดไป
```

ส่วนประกอบสำคัญ:

**1. Embedding (การแปลงคำเป็นเวกเตอร์)**
- คำแต่ละคำถูกแปลงเป็น "เวกเตอร์" (ลิสต์ตัวเลข) ที่มีความหมาย
- คำที่ใกล้เคียงกันจะมีเวกเตอร์ใกล้กัน เช่น "แมว" กับ "สุนัข" จะอยู่ใกล้กัน

**2. Attention (กลไกความสนใจ)**
- โมเดลจะ "มอง" คำก่อนหน้าทุกตัว แล้วตัดสินว่าคำไหนสำคัญต่อการทำนายคำถัดไป
- เช่น "ฉันไปซื้อ___ที่ตลาด" → โมเดลจะให้ความสนใจกับ "ตลาด" มากเป็นพิเศษ

ในโปรเจ็คนี้ใช้เทคนิคพิเศษหลายอย่าง:
- **RoPE** — เข้ารหัสตำแหน่งของคำด้วยการหมุนเวกเตอร์ (คิดเป็นเข็มนาฬิกาที่บอกว่าคำอยู่ตรงไหนในประโยค)
- **GQA (Grouped-Query Attention)** — ให้ head หลายตัวแชร์ key/value กัน ประหยัด memory
- **Sliding Window** — แทนที่จะมองทุกคำ บางชั้นมองแค่คำที่อยู่ใกล้ๆ (ลดการคำนวณ)

**3. MLP (Multi-Layer Perceptron)**
- ชั้นประมวลผลหลัก ทำหน้าที่ "คิด" หลังจาก attention บอกว่าควรสนใจอะไร
- ใช้ **ReGLU²** activation — `max(x, 0)²` — ช่วยให้โมเดลเรียนรู้ pattern ที่ซับซ้อน

**4. Logit Soft-capping**
- จำกัดค่า output ไม่ให้เกิน ±15 เพื่อความเสถียร
- เหมือน "กันกระแทก" ไม่ให้โมเดลมั่นใจมากเกินไป

### Optimizer — วิธีปรับน้ำหนักโมเดล

**Optimizer** = กลยุทธ์ในการปรับค่า weights ให้โมเดลเก่งขึ้นทีละนิด

ลองนึกภาพ: **คุณยืนบนเขาในความมืด ต้องหาทางลงไปจุดต่ำสุด**

#### AdamW (ใช้ในโปรเจ็คนี้)

ทำ 3 อย่าง:

1. **Momentum** — จำทิศทางจากก้าวก่อนๆ
   - เหมือนลูกบอลกลิ้งลงเนิน — ไม่หยุดกะทันหันเมื่อเจอหลุมเล็กๆ
   - ช่วยผ่าน local minima ได้

2. **Adaptive Learning Rate** — ปรับขนาดก้าวตาม parameter
   - Parameter ที่ไม่ค่อยเปลี่ยน → ก้าวใหญ่ (สำรวจมากขึ้น)
   - Parameter ที่เปลี่ยนถี่ → ก้าวเล็ก (ปรับละเอียด)

3. **Weight Decay** — หดค่า weight ลงทุกรอบ
   - ป้องกัน overfitting (โมเดลท่องจำแทนที่จะเรียนรู้)
   - คิดเหมือน "ภาษีความซับซ้อน" — ถ้า weight ใหญ่เกินจำเป็นจะถูกปรับลง

**Custom ในโปรเจ็คนี้:** แยก learning rate ตาม parameter group:
- **Embedding LR = 0.6** — ตัวแปลงคำ ใช้ LR สูง
- **Matrix LR = 0.04** — weight matrices ใน attention/MLP
- **Unembedding LR = 0.004** — ชั้นสุดท้าย ใช้ LR ต่ำ (ระวังมากกว่า)
- **Scalar LR = 0.5** — ค่า lambda ต่างๆ

#### Muon (ยังไม่ implement ในเวอร์ชันปัจจุบัน)

**Muon** (Momentum + Unitary) ออกแบบมาให้แต่ละ step "คุ้มค่า" มากขึ้น

|           | AdamW                            | Muon                                     |
| --------- | -------------------------------- | ---------------------------------------- |
| เปรียบเทียบ | รถยนต์ทั่วไป — วิ่งเร็ว เชื่อถือได้        | รถ F1 — ทุกโค้งแม่นกว่า                      |
| จุดแข็ง     | train ได้หลาย step ในเวลาจำกัด      | แต่ละ step ได้ค่ามากกว่า                     |
| เหมาะกับ   | hardware แรง (M4 Max) — ปริมาณชนะ | hardware กลางๆ (Mac Mini) — คุณภาพชนะ     |
| วิธีทำ       | ใช้ gradient ตรงๆ + momentum      | ใช้ Newton-Schulz "ปรับทิศทาง" gradient ก่อน |

Muon ใช้ **Newton-Schulz iteration** ซึ่งเป็นวิธีทางคณิตศาสตร์ที่ทำให้ gradient update มี "ทิศทางที่ดีที่สุด" — แต่แลกกับเวลาคำนวณที่มากขึ้นต่อ step

**ผลจากค่าเริ่มต้นอ้างอิง:** NS_STEPS=3 ดีกว่า 5 เพราะทำเร็วกว่า จึง train ได้มากรอบกว่าในเวลา 5 นาที

### MFU (Model FLOPs Utilization) — วัดประสิทธิภาพ Hardware

**FLOPs** = Floating Point Operations Per Second (จำนวนการคำนวณต่อวินาที)

**MFU** วัดว่า hardware ถูกใช้ได้กี่เปอร์เซ็นต์ของศักยภาพสูงสุด:

```
MFU = (FLOPs ที่ใช้จริง) ÷ (FLOPs สูงสุดของ hardware) × 100%
```

ตัวอย่าง:
- GPU ทำได้สูงสุด 100 TFLOPS, โมเดลใช้จริง 60 TFLOPS → **MFU = 60%**
- MFU 50-60% = ดีมาก, 10-20% = มี bottleneck

ในโปรเจ็คนี้ **MFU = 0.00 (placeholder)** เพราะ Apple ไม่ได้เปิดเผยตัวเลข peak TFLOPS อย่างเป็นทางการ ต่างจาก NVIDIA H100 ที่มี spec ชัดเจน (989.5 TFLOPS BF16)

### Learning Rate Schedule — กำหนดการความเร็ว

โมเดลไม่ได้ใช้ learning rate คงที่ตลอด แต่มี "กำหนดการ":

```
[--- Warmup ---][------ Constant ------][--- Warmdown ---]
     ↗️ เพิ่มขึ้น     → คงที่                ↘️ ลดลง
```

- **Warmup (0%)** — เริ่มจาก LR ต่ำแล้วค่อยเพิ่ม (ปัจจุบันข้ามไป)
- **Constant** — ใช้ LR เต็ม
- **Warmdown (50%)** — ค่อยๆ ลด LR เพื่อ "fine-tune" ในช่วงท้าย

### Gradient Accumulation — สะสม Gradient

เมื่อ batch size ใหญ่เกินกว่าจะ train ทีเดียว:

```
Forward + Backward (micro-batch 1) → สะสม gradient
Forward + Backward (micro-batch 2) → สะสม gradient
Forward + Backward (micro-batch 3) → สะสม gradient
...
รวม gradient ทั้งหมด → update weights ครั้งเดียว
```

ตัวอย่าง: `TOTAL_BATCH_SIZE = 65536` tokens, `DEVICE_BATCH_SIZE = 16` × `SEQ_LEN = 2048` = 32768 tokens/forward → ต้องทำ 2 forward passes แล้วรวม gradient

---

## วิเคราะห์สำหรับ Mac Mini M4 Pro 64GB

### เทียบกับเครื่องอื่นในโปรเจ็ค

| Spec      | M1 Studio 48GB | **M4 Pro 64GB (ของคุณ)** | M4 Max 128GB |
| --------- | -------------- | ----------------------- | ------------ |
| GPU Cores | 16             | **20**                  | 40           |
| Memory BW | ~200 GB/s      | **~273 GB/s**           | ~546 GB/s    |
| RAM       | 48 GB          | **64 GB**               | 128 GB       |
| ผลที่ได้     | val_bpb 1.808  | **~1.35-1.45 (ประมาณ)** | 1.295        |
| ลักษณะ     | entry          | **mid-range**           | top-tier     |

### สิ่งที่ทำได้

#### ✅ 1. รัน Autoresearch ข้ามคืน
- ปล่อย agent รันข้ามคืน ~8-10 ชั่วโมง → ได้ ~60-70 experiments
- แต่ละ experiment ใช้เวลา ~7 นาที (5 min train + 2 min overhead)
- agent จะค้นหา hyperparameters ที่เหมาะกับ hardware ของคุณโดยเฉพาะ

#### ✅ 2. ทดลอง Architecture ที่ใหญ่กว่า Default
RAM 64 GB เหลือเยอะ (default ใช้แค่ ~21-27 GB):
- ลอง **depth 5-6** (แทน 4) — โมเดลลึกกว่า
- ลอง **batch size ใหญ่กว่า** — `2**17` หรือ `2**18`
- ลอง **model_dim กว้างกว่า** — ปรับ `ASPECT_RATIO` ให้สูงขึ้น

#### ✅ 3. Implement Muon Optimizer
M4 Pro อยู่กลางๆ ระหว่าง Mac Mini กับ M4 Max — เป็นจุดที่น่าสนใจในการทดสอบว่า Muon หรือ AdamW จะชนะ

#### ✅ 4. เรียนรู้ LLM Training
- โค้ดรวม ~900 lines — อ่านเข้าใจได้หมดภายในไม่กี่ชั่วโมง
- มี GPT ครบทุก component: embedding, attention, MLP, optimizer, training loop
- ลองแก้ทีละส่วนแล้วดูผลลัวธ์ได้ทันที (5 นาที/experiment)

#### ✅ 5. เป็น Benchmark สำหรับ MLX
- วัดประสิทธิภาพ MLX แต่ละเวอร์ชันบน M4 Pro
- เปรียบเทียบ config ต่างๆ ภายใต้เงื่อนไขเดียวกัน (fixed 5-min budget)

### ข้อควรระวัง

- **Memory bandwidth เป็น bottleneck** — M4 Pro (~273 GB/s) น้อยกว่า M4 Max (~546 GB/s) ราว 2 เท่า, throughput (tok/sec) จะต่ำกว่าตามสัดส่วน
- **ความร้อน** — Mac Mini กล่องเล็ก ควรวางในที่อากาศถ่ายเทดีถ้าจะปล่อยรันข้ามคืน
- **Muon ยังไม่มี** — ต้อง implement เองถ้าต้องการ (README มีแนวทางไว้)

---

## วิธีเริ่มต้นใช้งาน

### ขั้นตอนที่ 1: ติดตั้ง

```bash
# ติดตั้ง uv (ถ้ายังไม่มี)
curl -LsSf https://astral.sh/uv/install.sh | sh

# ติดตั้ง dependencies
cd autoresearch-mlx
uv sync
```

### ขั้นตอนที่ 2: เตรียมข้อมูล

```bash
# ดาวน์โหลด data shards + train tokenizer (ทำครั้งเดียว ใช้เวลา ~5 นาที)
uv run prepare.py
```

ข้อมูลจะถูกเก็บไว้ที่ `~/.cache/autoresearch/`

### ขั้นตอนที่ 3: ทดสอบ

```bash
# รัน training ครั้งแรก (~7 นาที) เพื่อสร้าง baseline
uv run train.py
```

ผลลัพธ์จะแสดง val_bpb, memory usage, throughput ที่เป็นของเครื่องคุณ

### ขั้นตอนที่ 4: เริ่ม Autoresearch

ชี้ AI agent ไปที่ `program.md` — agent จะ:
1. อ่าน program.md เข้าใจกติกา
2. สร้าง branch ใหม่ (เช่น `autoresearch/mar10`)
3. รัน baseline แล้วเริ่ม experiment loop อัตโนมัติ
4. ทำงานไปเรื่อยๆ จนกว่าจะหยุดเอง

---

## สรุป

| คำถาม               | คำตอบ                                        |
| ------------------ | ------------------------------------------- |
| ต้องมี GPU ไหม?      | ❌ ไม่ต้อง — ใช้ Apple Silicon GPU ที่ฝังมาใน chip |
| ต้องใช้ PyTorch ไหม? | ❌ ใช้ MLX ซึ่งเป็น native Apple Silicon         |
| ใช้เวลานานไหม?      | ~7 นาที/experiment, ~60-70 experiments/คืน    |
| M4 Pro 64GB พอไหม? | ✅ เหลือเฟือ — default ใช้แค่ ~21-27 GB          |
| ต้อง code เองไหม?   | ❌ agent ทำให้ — แค่ชี้ไปที่ program.md             |
