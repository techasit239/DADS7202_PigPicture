# FreshCheck Deep Lab — DADS 7202

Material-aware pork freshness classification under real-world Thai retail packaging conditions

**Course:** DADS 7202 Deep Learning · NIDA · Graduate School of Applied Statistics
**Team:** Jirapong Sirichotithanawong (M) · Techasit Kanthajai (E) · Yingphan Hemchanphet (D)
**Deadline:** Sun 21 Jun 2026 (35 days from 17 May)

---

## 📋 Project Work Flow (Interactive)

แผนการทำงานแบบ interactive สำหรับทีม:

👉 **[เปิดดู Workflow แบบ Interactive](https://techasit239.github.io/DADS7202_PigPicture/FreshCheck_Project_WorkFlow_v2.html)**

*ลิงก์ใช้ได้หลังเปิด GitHub Pages — ดูคำแนะนำใน [SETUP_PAGES.md](#-การตั้งค่า-github-pages-เพื่อให้-workflow-เปิดได้)*

ทางเลือก: [ดาวน์โหลด HTML](FreshCheck_Project_WorkFlow_v2.html) เปิดในเบราเซอร์เครื่องตัวเอง

---

## 📓 Notebooks

| Phase | Notebook | Open in Colab |
|---|---|---|
| Phase 0 | [Foundation Setup](FreshCheck_Phase0_Foundation_v3.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase0_Foundation_v3.ipynb) |
| Phase 1 | [Classification (EfficientNet-B0)](FreshCheck_Phase1_Classification_v1.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techasit239/DADS7202_PigPicture/blob/main/FreshCheck_Phase1_Classification_v1.ipynb) |

---

## 🚀 การรัน (ตามลำดับ)

1. **Phase 0** — เตรียม Kaggle dataset + group-aware split (~10 นาที, ทำครั้งเดียวพอ)
2. **Phase 1** — เทรน EfficientNet-B0 บน Kaggle (~20-30 นาที, T4 GPU)
3. **Phase 2** (กำลังพัฒนา) — Material-aware segmentation
4. **Phase 3** (กำลังพัฒนา) — Integration + 5-fold CV บน Thai retail data

---

## ⚙️ การตั้งค่า GitHub Pages เพื่อให้ Workflow เปิดได้

ทำครั้งเดียว เพื่อให้ HTML ในไฟล์เปิดได้แบบ interactive:

1. ไปที่ repo → **Settings**
2. Sidebar ซ้าย → **Pages**
3. ส่วน **Build and deployment**:
   - **Source:** Deploy from a branch
   - **Branch:** `main` / `/ (root)`
4. กด **Save**
5. รอ 1-2 นาที — refresh หน้า Pages จะเห็นลิงก์
   `https://techasit239.github.io/DADS7202_PigPicture/`
6. ทดสอบเปิด `https://techasit239.github.io/DADS7202_PigPicture/FreshCheck_Project_WorkFlow_v2.html`

---

## 📅 Timeline (5 weeks)

| Week | วันที่ | งาน |
|---|---|---|
| 1 | 18-24 พ.ค. | Phase 1 + เก็บ Thai data + CVAT tags |
| 2 | 25-31 พ.ค. | Phase 2: CVAT parser + Seg Exp 1+2 |
| 3 | 1-7 มิ.ย. | Phase 2: Seg Exp 3 + เริ่ม Phase 3 |
| 4 | 8-14 มิ.ย. | Phase 3: 5-fold CV + เริ่ม Report |
| 5 | 15-21 มิ.ย. | Final Report + Submit |
