# Code Folder

Bu papka ichida turganda commandlar `python file.py` ko'rinishida yoziladi. Masalan `python tracking.py`. `python code/tracking.py` deb yozmang, aks holda path `code/code/tracking.py` bo'lib ketadi.

Asosiy tracking fayl:

```bash
python tracking.py
```

Admin analytics panel:

```bash
streamlit run app.py
```

`Ask Data` tabida chatbot SQLite `chat_queries` jadvalidagi short memory orqali oxirgi suhbatlarni eslab qoladi. Memoryni admin paneldagi `Clear memory` tugmasi bilan tozalash mumkin.

Fine tuning helper:

```bash
python fine_tuning.py init-dataset
python fine_tuning.py export-review-crops --limit 300
```

Roboflow dataset helper:

```bash
python roboflow_dataset.py init
python roboflow_dataset.py export-crops --limit 300
python roboflow_dataset.py summary
```

Logs:

```bash
tail -f code/data/logs/app.log
```

Admin URL:

```text
http://localhost:8501
```

SQLite database:

```text
code/data/surveillance.db
```

To'liq yo'riqnoma root README’da:

```text
../README.md
```
