# C+Patch A100 Runbook

ใช้กับ Google Colab ที่ได้ `A100 GPU` แล้ว และใช้ข้อมูลการทดลอง `C+Patch` แบบหลาย seed

## 1. แนวคิดที่ใช้

- `efficientnet_b0`: fine-tune ทั้ง backbone, แต่ freeze ช่วงต้นเล็กน้อยได้
- `swin_t`: freeze backbone ช่วงต้น แล้วค่อย unfreeze เพื่อให้ optimize เสถียรกว่า
- `dinov2_vits14`: ใช้เป็น frozen backbone + linear head
- patch sampling: เปิดด้วย `--patch-sampling --num-patches 2`
- acceleration:
  - `--amp --amp-dtype bf16`
  - `--allow-tf32`
  - `--persistent-workers`

## 2. ตัวแปรหลัก

```bash
EXP_ROOT=/content/drive/MyDrive/FreshCheck/artifacts
PATCH_ROOT=$EXP_ROOT/exp_cplus_patch_runs
TRAIN_CSV=/content/drive/MyDrive/FreshCheck/artifacts/experiments_manual/exp_c_train.csv
VAL_CSV=/content/drive/MyDrive/FreshCheck/artifacts/experiments_manual/exp_c_val.csv
TARGET_CSV=/content/drive/MyDrive/FreshCheck/artifacts/target_split/target_holdout.csv
SOURCE_CSV=/content/drive/MyDrive/FreshCheck/artifacts/experiments_auto/source_holdout.csv
SEEDS="42 52 62 72 82"
```

## 3. EfficientNet-B0

```bash
for SEED in $SEEDS; do
  OUT=$PATCH_ROOT/efficientnet_b0/seed_$SEED
  python run_freshcheck.py train \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --output-dir "$OUT/train" \
    --models efficientnet_b0 \
    --epochs 14 \
    --patience 4 \
    --lr 0.0003 \
    --weight-decay 0.005 \
    --dropout 0.25 \
    --img-size 224 \
    --batch-size 32 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --freeze-backbone-epochs 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED

  python run_freshcheck.py evaluate \
    --csv "$TARGET_CSV" \
    --output-dir "$OUT/eval_target" \
    --models efficientnet_b0 \
    --checkpoint-paths efficientnet_b0="$OUT/train/checkpoints/efficientnet_b0_best.pt" \
    --img-size 224 \
    --batch-size 32 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED

  python run_freshcheck.py evaluate \
    --csv "$SOURCE_CSV" \
    --output-dir "$OUT/eval_source" \
    --models efficientnet_b0 \
    --checkpoint-paths efficientnet_b0="$OUT/train/checkpoints/efficientnet_b0_best.pt" \
    --img-size 224 \
    --batch-size 32 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED
done
```

## 4. Swin-T

```bash
for SEED in $SEEDS; do
  OUT=$PATCH_ROOT/swin_t/seed_$SEED
  python run_freshcheck.py train \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --output-dir "$OUT/train" \
    --models swin_t \
    --epochs 12 \
    --patience 4 \
    --lr 0.0003 \
    --weight-decay 0.01 \
    --dropout 0.30 \
    --img-size 224 \
    --batch-size 16 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --freeze-backbone-epochs 3 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED

  python run_freshcheck.py evaluate \
    --csv "$TARGET_CSV" \
    --output-dir "$OUT/eval_target" \
    --models swin_t \
    --checkpoint-paths swin_t="$OUT/train/checkpoints/swin_t_best.pt" \
    --img-size 224 \
    --batch-size 16 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED

  python run_freshcheck.py evaluate \
    --csv "$SOURCE_CSV" \
    --output-dir "$OUT/eval_source" \
    --models swin_t \
    --checkpoint-paths swin_t="$OUT/train/checkpoints/swin_t_best.pt" \
    --img-size 224 \
    --batch-size 16 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED
done
```

## 5. DINOv2

```bash
for SEED in $SEEDS; do
  OUT=$PATCH_ROOT/dinov2_vits14/seed_$SEED
  python run_freshcheck.py train \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --output-dir "$OUT/train" \
    --models dinov2_vits14 \
    --epochs 10 \
    --patience 3 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --dropout 0.30 \
    --img-size 224 \
    --batch-size 16 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED

  python run_freshcheck.py evaluate \
    --csv "$TARGET_CSV" \
    --output-dir "$OUT/eval_target" \
    --models dinov2_vits14 \
    --checkpoint-paths dinov2_vits14="$OUT/train/checkpoints/dinov2_vits14_best.pt" \
    --img-size 224 \
    --batch-size 16 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --patch-sampling \
    --num-patches 2 \
    --amp \
    --amp-dtype bf16 \
    --allow-tf32 \
    --persistent-workers \
    --seed $SEED
done
```

## 6. สรุปผลหลาย seed

```bash
python summarize_seed_runs.py \
  --root-dir "$PATCH_ROOT/efficientnet_b0" \
  --model efficientnet_b0 \
  --eval-subdir eval_target \
  --output-dir "$PATCH_ROOT/_summary/efficientnet_b0"

python summarize_seed_runs.py \
  --root-dir "$PATCH_ROOT/swin_t" \
  --model swin_t \
  --eval-subdir eval_target \
  --output-dir "$PATCH_ROOT/_summary/swin_t"

python summarize_seed_runs.py \
  --root-dir "$PATCH_ROOT/dinov2_vits14" \
  --model dinov2_vits14 \
  --eval-subdir eval_target \
  --output-dir "$PATCH_ROOT/_summary/dinov2_vits14"
```

## 7. สิ่งที่ควรเก็บไปเขียนผล

- `train/checkpoints/*_best.pt`
- `train/metrics/*_history.csv`
- `eval_target/*_eval.json`
- `eval_target/*_confusion_matrix.csv`
- `eval_target/*_confusion_matrix.png`
- `eval_source/*_eval.json` สำหรับ source retention check
