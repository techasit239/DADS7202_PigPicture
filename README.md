# FreshCheck Deep Lab

> **Material-aware pork freshness classification under real-world Thai retail packaging conditions**

[![Course](https://img.shields.io/badge/Course-DADS%207202-1e3a5f)](https://nida.ac.th)
[![Institute](https://img.shields.io/badge/Institute-NIDA-c4583a)](https://nida.ac.th)
[![Deadline](https://img.shields.io/badge/Final%20Report-21%20Jun%202026-4a7c59)](#-timeline)
[![Phase 0](https://img.shields.io/badge/Phase%200-DONE-4a7c59)](FreshCheck_Phase0_Foundation_v3.ipynb)
[![Phase 1](https://img.shields.io/badge/Phase%201-DONE-4a7c59)](FreshCheck_Phase1_Classification_v1.ipynb)
[![Phase 2](https://img.shields.io/badge/Phase%202-Pending%20HF%20approval-d4943c)](#-phase-2--material-aware-segmentation)

---

## 📋 About

**Course:** DADS 7202 Deep Learning · NIDA · Graduate School of Applied Statistics
**Project Proposal:** [Phase 2 Proposal (submitted)](2_PROPOSAL_FreshCheck_manuscript.pdf)
**Interactive Workflow:** [GitHub Pages](https://techasit239.github.io/DADS7202_PigPicture/FreshCheck_Project_WorkFlow_v2.html)

### Team

| Member | Role |
|---|---|
| Jirapong Sirichotithanawong | **M** — Model Architect (lead) |
| Techasit Kanthajai | **E** — Evaluator + PM |
| Yingphan Hemchanphet | **D** — Data Engineer |

### Problem

ผู้บริโภคไม่มีเครื่องมือประเมินความสดของเนื้อหมูในซุปเปอร์มาร์เก็ตไทย เพราะ:
1. งานวิจัยเดิมเน้นเนื้อแบบ **unpackaged** (วางขายตามตลาดสด ไม่มีพลาสติกหุ้ม)
2. เนื้อบรรจุภัณฑ์ retail มี **occlusion** จากพลาสติก ฉลาก ป้ายราคา
3. โมเดล classification ที่ตรงๆ ทาย freshness อาจไป focus ที่ฉลากแทนเนื้อ

### Solution — 2-Step Pipeline

```
   Raw Image                Step 1                  Step 2
   ─────────         ─────────────────────       ─────────────────
   พลาสติก +    →    Material-aware       →    Freshness
   เนื้อ + ฉลาก     Segmentation              Classification
                    (เลือกเฉพาะเนื้อ)            (Fresh/HF/Spoiled)
```

---

## 🚀 Notebooks (Run in Colab)

| Phase | Status | Notebook | Open |
|---|---|---|---|
| 0 | ✅ Done | Foundation Setup | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase0_Foundation_v3.ipynb) |
| 1 | ✅ Done | Classification (EfficientNet-B0) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase1_Classification_v1.ipynb) |
| 2 — Foundation | 🟡 Ready | CVAT parser + GT masks | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase2_Foundation_v1.ipynb) |
| 2 — Exp 1 | ⏳ Pending HF | SAM 3 + text prompt | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase2_Exp1_SAM3_v1.ipynb) |
| 2 — Exp 2 | ⏳ Pending HF | Florence-2 → SAM 3 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase2_Exp2_Florence2_SAM3_v1.ipynb) |
| 2 — Exp 3 | ⏳ Pending HF | DINOv3 + seg head (5-fold) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase2_Exp3_DINOv3_v1.ipynb) |
| 2 — Compare | ⏳ Wait Exp1-3 | เลือก best mask method | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase2_Compare_v1.ipynb) |

> Phase 3 (final pipeline + 5-fold CV) จะมาทีหลัง — หลัง Phase 2 Compare เลือก best mask method ได้แล้ว

---

## 📅 Timeline

**5 weeks remaining** (Today: 17 May 2026 → Deadline: **Sun 21 Jun 2026**)

> 📝 **Note:** Deadline เลื่อนจาก 6 ก.ค. → 21 มิ.ย. = หายไป 2 สัปดาห์
> ผลกระทบ: **ตัด Phase 4 + NEW TRACK ออกจาก scope** (เก็บเป็น future work ใน Final Report)

| Week | วันที่ | งาน | Status |
|---|---|---|---|
| 1 | 18-24 พ.ค. | Phase 0 + 1 + Thai data + CVAT tags | 🟡 ทำอยู่ |
| 2 | 25-31 พ.ค. | Phase 2 Foundation + Exp 1 + Exp 2 | ⏳ Up next |
| 3 | 1-7 มิ.ย. | Phase 2 Exp 3 + Compare → เริ่ม Phase 3 | ⏳ |
| 4 | 8-14 มิ.ย. | Phase 3 (5-fold CV) + start Report writing | ⏳ |
| 5 | 15-21 มิ.ย. | Final Report + Turnitin + Presentation + **Submit** | ⏳ |

---

## 🎯 Scope (ตาม Comment อาจารย์)

### Step 1 — Material-aware Segmentation (Phase 2)

| Exp | Method | Train? |
|---|---|---|
| 1 | SAM 3 + text prompt | ❌ Zero-shot |
| 2 | Florence-2 → SAM 3 (box refine) | ❌ Zero-shot |
| 3 | DINOv3 (frozen) + segmentation head | ✅ Train head only |

→ เปรียบเทียบด้วย **IoU + Dice** vs CVAT ground truth → เลือก best mask method สำหรับ Phase 3

### Step 2 — Freshness Classification

| Exp | Backbone | Status |
|---|---|---|
| 1 | EfficientNet-B0 | ✅ Phase 1 done (Kaggle baseline) |
| 2 | Swin Transformer Tiny | ⏳ Phase 3 |
| 3 | DINOv3-ViT-S/16 (frozen) | ⏳ Phase 3 |

→ ใช้ best mask × รูปต้นฉบับ → train classifier
→ Test ด้วย **5-fold StratifiedGroupKFold CV** บน Thai 150 รูป

### ❌ Cut from Scope

เนื่องจาก deadline เลื่อน — ระบุใน Final Report Section 6 (Future Work):
- **Phase 4** — Patch sampling + confidence-weighted vote + 3-state Decision Triage
- **NEW TRACK** — DINOv3 end-to-end (single model for seg+cls)

---

## 📊 Datasets

| Dataset | Source | Used For | Status |
|---|---|---|---|
| Kaggle Meat Freshness | [vinayakshanawad/meat-freshness-image-dataset](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset) | Train + Val (Phase 1) | ✅ 2,266 รูป downloaded |
| Thai retail pork loin | Team-collected (ซุปเปอร์ + ตลาดสด) | Test (Phase 3, 5-fold CV) | 🟡 collecting 150 รูป |
| CVAT polygon annotations | Team-annotated | Ground truth masks (Phase 2) | 🟡 in progress |

### Master Filename Format (Thai data)

```
YYYYMMDD_HHMM_classCode_sourceType_pieceID.ext

Example: 20260518_1000_FR_PK_P01.jpg
                          ^^^^
                          piece_id (group key for anti-leakage)

classCode:  FR=Fresh, HF=Half-Fresh, SP=Spoiled
sourceType: PK=Packaged, UP=Unpackaged
```

---

## 📁 Repository Structure

```
DADS7202_PigPicture/
│
├── README.md                                            ← (this file)
├── 2_PROPOSAL_FreshCheck_manuscript.pdf                 ← Proposal (submitted)
├── FreshCheck_Project_WorkFlow_v2.html                  ← Interactive workflow (GitHub Pages)
│
├── 📓 Notebooks (Run in Colab)
│   ├── FreshCheck_Phase0_Foundation_v3.ipynb           ✅ Setup + Kaggle + group-aware split
│   ├── FreshCheck_Phase1_Classification_v1.ipynb       ✅ EfficientNet-B0 baseline
│   ├── FreshCheck_Phase2_Foundation_v1.ipynb           🟡 CVAT XML → GT masks (Phase 2 prereq)
│   ├── FreshCheck_Phase2_Exp1_SAM3_v1.ipynb            ⏳ SAM 3 + text prompt
│   ├── FreshCheck_Phase2_Exp2_Florence2_SAM3_v1.ipynb  ⏳ Florence-2 → SAM 3
│   ├── FreshCheck_Phase2_Exp3_DINOv3_v1.ipynb          ⏳ DINOv3 + seg head (5-fold)
│   └── FreshCheck_Phase2_Compare_v1.ipynb              ⏳ เลือก best mask method
│
└── Phase2_README.md                                    ← Detail วิธีรัน Phase 2
```

> **Artifacts ไม่ commit GitHub:** ทุก Phase สร้าง outputs (predictions, models, plots) ใน Google Drive ของผู้รัน ที่ `MyDrive/FreshCheck/`

---

## GitHub + Colab + Drive Workflow

แนะนำให้ใช้ workflow นี้เป็นค่าเริ่มต้น:

1. เก็บ source code, notebooks, `requirements.txt`, `README.md` ไว้บน GitHub
2. ใช้ Google Colab เป็นเครื่องรัน train / evaluate / inference
3. เก็บ dataset, CVAT XML, checkpoints, predictions, logs ไว้บน Google Drive
4. ถ้าแก้โค้ดบน GitHub ให้กลับไปรัน cell update repo ใน Colab แล้วทำงานต่อ

ไฟล์ที่ใช้เป็น entrypoint:
- `FreshCheck_Colab_Runner.ipynb` สำหรับ bootstrap จาก GitHub แล้วเรียก `run_freshcheck.py`
- `run_freshcheck.py` สำหรับ `prepare-splits`, `train`, `evaluate`, `predict`, `prepare-cvat`
- `freshcheck/` สำหรับ logic หลักของ data/model/training

โครงเก็บไฟล์บน Drive ที่แนะนำ:

```text
MyDrive/FreshCheck/
├── data/
│   ├── kaggle_meat/
│   ├── thai_retail/
│   └── thai_retail_cvat_annotations.xml
└── artifacts/
    ├── splits/
    ├── train/
    ├── eval/
    ├── predict/
    └── phase2/
```

---

## ⚙️ Setup

## Local CLI Runner

ผมเพิ่ม local runner สำหรับรันโค้ดโดยไม่ต้องเปิด notebook แล้ว:

```bash
pip install -r requirements.txt
python run_freshcheck.py --help
```

รองรับคำสั่งหลัก:

```bash
python run_freshcheck.py prepare-splits --data-dir <kaggle_images_dir> --output-dir artifacts/splits
python run_freshcheck.py train --train-csv artifacts/splits/kaggle_train.csv --val-csv artifacts/splits/kaggle_val.csv --output-dir artifacts/train --models all
python run_freshcheck.py evaluate --csv artifacts/splits/kaggle_val.csv --checkpoint-dir artifacts/train/checkpoints --output-dir artifacts/eval --models all
python run_freshcheck.py predict --input-path <image_or_folder> --checkpoint-dir artifacts/train/checkpoints --output-dir artifacts/predict --models all
python run_freshcheck.py prepare-cvat --thai-data-dir <thai_images_dir> --cvat-xml-path <annotations.xml> --output-dir artifacts/phase2
```

`--models all` จะรัน `efficientnet_b0`, `swin_t`, และ `convnext_tiny` ต่อเนื่องให้ในคำสั่งเดียว

ข้อจำกัดตอนนี้:
- Local CLI ครอบคลุม Phase 1 classifiers และ Phase 2 foundation (CVAT → masks/CSV)
- Notebook ที่พึ่ง gated Hugging Face models (`SAM3`, `Florence-2`, `DINOv3`) ยังไม่ได้ย้ายเป็น local runner เพราะต้องใช้ access/token จริงก่อน
  และ API ของแต่ละโมเดลควร implement จาก environment ที่ยืนยันแล้ว ไม่ควรเดา

### Compare Existing Checkpoints From Google Drive

ถ้ามี checkpoint เดิมจาก notebook เก่าอยู่แล้ว เช่น:
- `phase1_efficientnet_b0_best.pth`
- `phase2_swin_t_best.pth`

สามารถ evaluate หรือ predict ได้ทันทีโดยไม่ต้อง rename:

```bash
python run_freshcheck.py evaluate \
  --csv /content/drive/MyDrive/FreshCheck/thai_test/thai_test_manifest.csv \
  --checkpoint-paths \
    efficientnet_b0=/content/drive/MyDrive/FreshCheck/models/classification/phase1_efficientnet_b0_best.pth \
    swin_t=/content/drive/MyDrive/FreshCheck/models/classification/phase2_swin_t_best.pth \
  --output-dir /content/drive/MyDrive/FreshCheck/artifacts/eval \
  --models efficientnet_b0 swin_t
```

```bash
python run_freshcheck.py predict \
  --input-path /content/drive/MyDrive/FreshCheck/thai_test/images \
  --checkpoint-paths \
    efficientnet_b0=/content/drive/MyDrive/FreshCheck/models/classification/phase1_efficientnet_b0_best.pth \
    swin_t=/content/drive/MyDrive/FreshCheck/models/classification/phase2_swin_t_best.pth \
  --output-dir /content/drive/MyDrive/FreshCheck/artifacts/predict \
  --models efficientnet_b0 swin_t
```

### Train One More Model: ConvNeXt-Tiny

เพิ่ม baseline ใหม่ได้ทันทีโดยไม่ต้องรอ gated access:

```bash
python run_freshcheck.py train \
  --train-csv /content/drive/MyDrive/FreshCheck/artifacts/splits/kaggle_train.csv \
  --val-csv /content/drive/MyDrive/FreshCheck/artifacts/splits/kaggle_val.csv \
  --output-dir /content/drive/MyDrive/FreshCheck/artifacts/train \
  --models convnext_tiny \
  --epochs 15
```

## Colab Quick Start

เปิด [FreshCheck_Colab_Runner.ipynb](FreshCheck_Colab_Runner.ipynb) บน Colab แล้วรันตามลำดับ cell:

```bash
1) Clone / Update repository from GitHub
2) Mount Google Drive and define project paths
3) Install dependencies
4) Prepare group-aware Kaggle train/val splits
5) Train classifiers
6) Evaluate checkpoints
7) Predict on one image or one folder
8) Prepare CVAT masks for Phase 2 foundation
```

ถ้าจะรันผ่าน shell ใน Colab ตรง ๆ:

```bash
!git clone https://github.com/techasit239/DADS7202_PigPicture.git
%cd DADS7202_PigPicture
!pip install -r requirements.txt
!python run_freshcheck.py --help
```

### 1. Google Colab + T4 GPU

ทุก notebook รันบน Colab → `Runtime → Change runtime type → T4 GPU`

### 2. Kaggle API token (Phase 0 — ครั้งเดียวพอ)

- [kaggle.com](https://www.kaggle.com) → Settings → API → **Create New Token** → ได้ `kaggle.json`
- Upload ตอน Phase 0 ขอ (จะ backup เก็บใน Drive ให้ ไม่ต้องอัพซ้ำ)

### 3. HuggingFace Access (Phase 2 — gated models)

Phase 2 ใช้ 2 gated models ที่ต้อง **request access ก่อน**:

#### Step A: Request access

| Model | Link | สถานะปัจจุบัน |
|---|---|---|
| SAM 3 | [facebook/sam3](https://huggingface.co/facebook/sam3) | 🟡 PENDING |
| DINOv3-ViT-S/16 | [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 🟡 PENDING |

ทั้ง 2 ต้องกรอกฟอร์ม Meta (Affiliation, Job title) → กด Submit → รอ approve

⏱ approve ปกติ **5-30 นาที** บางครั้งถึง 2 ชั่วโมง

#### Step B: เช็คสถานะ

[huggingface.co/settings/gated-repos](https://huggingface.co/settings/gated-repos) — ต้องขึ้น `ACCEPTED` ทั้งคู่

#### Step C: สร้าง token + ใส่ Colab Secret

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token** (Read permission พอ)
2. ใน Colab notebook → ไอคอน 🔑 ทางซ้าย → **Add new secret**
   - Name: `HF_TOKEN` (ต้องสะกดตรงนี้เป๊ะ)
   - Value: token (`hf_...`)
   - ✅ Notebook access ON

---

## 🔄 Run Order

```
   ┌────────────────────────┐
   │  Phase 0 Foundation    │  ✅ DONE
   │  Kaggle + split        │
   └───────────┬────────────┘
               │
               ▼
   ┌────────────────────────┐
   │  Phase 1 Classification│  ✅ DONE
   │  EfficientNet-B0       │
   └───────────┬────────────┘
               │
               ▼
   ┌────────────────────────────────────────┐
   │  Phase 2 Foundation                    │  🟡 พร้อมรัน
   │  Parse CVAT → 150 GT masks             │
   └───────────┬────────────────────────────┘
               │
               ├──────────────────┬──────────────────┐
               ▼                  ▼                  ▼
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │  Phase 2 Exp 1 │ │  Phase 2 Exp 2 │ │  Phase 2 Exp 3 │
   │  SAM 3+prompt  │ │  Florence→SAM3 │ │  DINOv3+head   │
   │  ~20-30 นาที   │ │  ~40-50 นาที   │ │  ~40-60 นาที   │
   └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Phase 2 Compare       │
                  │  เลือก best mask       │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Phase 3 (TBD)         │
                  │  Pipeline + 5-fold CV  │
                  │  → Final Report        │
                  └────────────────────────┘
```

**Parallel-friendly tips:**
- Phase 2 Exp 1, 2, 3 รันขนานได้ — เปิด 3 Colab tabs แล้ว Run all พร้อมกัน
- ต้อง Phase 2 Foundation เสร็จก่อน ทั้ง 3 Exp อ่าน GT masks จากที่นี่

---

## ⚠️ Current Status & Blockers

### ✅ Done

- [x] Phase 0 — Foundation Setup (Kaggle download, group-aware split, Thai filename parser)
- [x] Phase 1 — EfficientNet-B0 baseline on Kaggle (val metrics + confusion matrix)
- [x] HF account setup: `Jirapong5556`
- [x] HF token: `colab-freshcheck` (read permission)
- [x] HF `transformers` upgraded เพื่อรองรับ SAM 3
- [x] Request access SAM 3 + DINOv3 (gated → submitted)
- [x] Phase 2: 5 notebooks built (Foundation + Exp 1/2/3 + Compare)
- [x] Workflow HTML v2 + GitHub Pages

### 🟡 In Progress

- [ ] **HF approval** (SAM 3 + DINOv3 = PENDING — รอ Meta review)
- [ ] Thai retail data collection (150 รูป)
- [ ] CVAT classification tags (3 คน majority vote)

### 🔴 Blockers

| Blocker | Impact | ETA |
|---|---|---|
| HF SAM 3 + DINOv3 approval = PENDING | Phase 2 Exp 1/2/3 รันไม่ได้ | 5 นาที-2 ชม. |
| Thai 150 images ยังไม่ครบ | Phase 2 Foundation รันไม่ได้ | ทีมเก็บใน Week 1 |
| CVAT XML export ยังไม่มี | Phase 2 Foundation รันไม่ได้ | หลังใส่ tags ครบ |

---

## 🐛 Troubleshooting

### Phase 0 / 1

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| `FileNotFoundError: kaggle_train.csv` | Phase 0 ยังไม่ได้รัน | กลับไปรัน Phase 0 v3 ครบทุก cell |
| `Out of Memory` ตอน train | Batch size ใหญ่เกิน T4 | ลด `BATCH_SIZE` จาก 32 → 16 หรือ 8 |
| `kaggle.json not found` | ยังไม่ได้ upload | ถ้าเคย backup ใน Drive → cell 0.2.1 ใช้ของเก่า; ถ้าไม่มี → upload ใหม่ |
| Notebook ขึ้น "Invalid Notebook" บน GitHub | `metadata.widgets` ขาด field | รัน script strip widgets แล้ว save ใหม่ (ไม่กระทบการรัน) |

### Phase 2

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| `403 Forbidden — gated repo` | HF access ยังไม่ approve | เช็ค `settings/gated-repos` → ต้อง ACCEPTED |
| `Could not import Sam3Processor` | transformers version เก่า | `!pip install -U transformers` แล้ว Runtime → Restart session |
| `assert os.path.exists(CVAT_XML)` fail | CVAT XML ยังไม่อัพ Drive | Export CVAT format `CVAT for images 1.1` → upload ขึ้น Drive |
| Token เก่าใช้ไม่ได้ | Token expire หรือ scope ผิด | สร้างใหม่ที่ `settings/tokens` → update Colab Secret `HF_TOKEN` |
| `whoami()` returns ผิด account | login ผิด account | logout HF → login account ที่ Submit request |

### GitHub

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| Workflow HTML แสดงเป็น code | GitHub ไม่ render HTML ตรงๆ | เปิด **GitHub Pages** ใน Settings → ใช้ลิงก์ `techasit239.github.io/...` |
| Notebook แสดง "Invalid Notebook" | tqdm widget metadata ขาด state | รัน strip-widgets script (ไม่กระทบการรัน — เปิดใน Colab ได้ปกติ) |

---

## 📐 Verification & Reproducibility

- **Random seed:** `SEED = 42` ทุก notebook
- **Anti-leakage:** `GroupShuffleSplit` (Phase 0) + `StratifiedGroupKFold` (Phase 2 Exp 3 + Phase 3) ตาม `piece_id`
- **Verification protocol:** ทุก notebook assert ว่าไฟล์ Phase ก่อนมีอยู่ + columns ครบ + class labels ตรง ก่อนเริ่มทำงาน
- **Strict validation:** Fail loud ด้วย `assert` แทน fail silent — Phase ก่อนหน้าไม่เสร็จ จะหยุดทันที พร้อมบอกชัดเจนว่าต้องไปทำอะไร

---

## 📚 References

ดู Proposal เต็มใน [2_PROPOSAL_FreshCheck_manuscript.pdf](2_PROPOSAL_FreshCheck_manuscript.pdf) — 24 references มาตรฐาน IEEE + DOI links + verified authors

Key references:
- **[10] Bramantyo et al.** — EfficientNet-B0 reference baseline (98.10% on packaged-meat)
- **[17] V. Shanawad** — Meat Freshness Image Dataset (Kaggle, 2,266 images)
- **[23] Siméoni et al.** — *DINOv3* (arXiv:2508.10104)
- **[24] Carion et al.** — *SAM 3: Segment Anything with Concepts*

---

## 📝 License & Acknowledgement

Educational project for **DADS 7202 Deep Learning** at NIDA.

Models used:
- SAM 3 (Meta, gated)
- DINOv3 (Meta, gated)
- Florence-2 (Microsoft, MIT)
- EfficientNet-B0 (PyTorch torchvision)

---

*Last updated: 17 May 2026 · 35 days to deadline*
