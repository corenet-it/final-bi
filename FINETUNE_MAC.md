# Fine-tuning guide (Mac) — ASSBI vehicle detector

Bu branch (`finetune-clean-dataset`) tozalangan datasetni va fine-tune skriptini
o'z ichiga oladi. Maqsad: `yolo11n` modelni **to'liq kadr + to'g'ri bbox
koordinatalari** (x1,y1,x2,y2 → YOLO formatga aylantirilgan) bilan fine-tune qilish.

## Nega bu dataset to'g'ri

YOLO **detector** rasmda obyekt **qayerda** ekanini o'rganishi kerak. Shuning uchun:
- Har bir rasm — **to'liq kadr** (crop emas).
- Har bir label — `class cx cy w h` (normalize qilingan haqiqiy koordinatalar).

Eski xato usulda (faqat kesilgan crop + `0.5 0.5 1.0 1.0` label) detector hech narsa
o'rganmaydi — u olib tashlangan. Hozir dataset toza:

- `data/roboflow_dataset/images/train` — 993 rasm
- `data/roboflow_dataset/images/val`   — 254 rasm
- mos `labels/` papkalari bilan
- Klasslar: `0=car, 1=bus, 2=truck, 3=motorcycle, 4=bicycle`

> Eslatma: `bicycle` klassida atigi ~4 ta namuna bor — u klass deyarli o'rganilmaydi.
> Kerak bo'lsa ko'proq bicycle kadr yig'ish kerak.

## 1. Repo'ni olish

```bash
git clone git@github.com:corenet-it/final-bi.git
cd final-bi
git checkout finetune-clean-dataset
```

## 2. Muhitni tayyorlash (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Mac (Apple Silicon M1/M2/M3) `torch`'ni avtomatik `mps` (GPU) bilan ishlatadi —
CPU'dan ancha tez. `requirements.txt` dagi `ultralytics` `torch`'ni o'zi tortadi.

## 3. Fine-tune (boshidan-oxirigacha bitta buyruq)

```bash
python train_finetune.py
```

Skript o'zi:
1. `data.yaml` dagi `path:`ni shu Mac'ga moslab qayta yozadi (muhim — aks holda train ishlamaydi).
2. Eng tez qurilmani tanlaydi (`mps` → `cuda` → `cpu`).
3. `yolo11n.pt` ni avtomatik yuklab, 40 epoch (early-stop bilan) train qiladi.

Natija:
```
data/fine_tuning/runs/vehicle_surveillance/weights/best.pt
```
Bu papka `.gitignore`'da — git'ga ketmaydi, faqat lokal.

### Sozlamalarni o'zgartirish
`train_finetune.py` ichida `epochs`, `imgsz`, `batch` ni o'zgartirish mumkin.
Mac GPU bilan `epochs=60, batch=16` bemalol.

## 4. Validatsiya / metrikalar

Train tugagach metrikalar (mAP50, mAP50-95) terminalda va shu yerda chiqadi:
```
data/fine_tuning/runs/vehicle_surveillance/   (results.png, confusion_matrix.png, ...)
```

Alohida tekshirish:
```bash
yolo detect val model=data/fine_tuning/runs/vehicle_surveillance/weights/best.pt \
  data=data/roboflow_dataset/data.yaml imgsz=512
```

## 5. Modelni ishlatish (tracking)

Train qilingan modelni asosiy pipeline'ga ulash:
```bash
export ASSBI_YOLO_MODEL=$(pwd)/data/fine_tuning/runs/vehicle_surveillance/weights/best.pt
python run_tracker.py
```

## 6. Ko'proq ma'lumot yig'ish (ixtiyoriy)

Yangi video'dan to'g'ri (full-frame + koordinata) sample yig'ish uchun
`tracking.py` ishga tushiriladi — u GUI ochib, ROI tanlatadi va har bir yangi
mashina uchun **to'liq kadr + bbox**ni avtomatik `data/roboflow_dataset/` ga yozadi
(`ENABLE_ROBOFLOW_DATASET_EXPORT=1`, default yoqilgan):

```bash
python tracking.py
```

So'ng yana `python train_finetune.py`.

---
**Qisqa:** clone → `checkout finetune-clean-dataset` → venv → `pip install -r requirements.txt`
→ `python train_finetune.py`. Tamom.
