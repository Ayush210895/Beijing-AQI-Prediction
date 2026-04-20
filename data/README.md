# Data Setup

Download the Beijing Multi-Site Air Quality dataset from UCI:

https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata

Or run the downloader from the project root:

```bash
python3 scripts/download_data.py
```

Use the file named:

```text
PRSA2017_Data_20130301-20170228.zip
```

Unzip it and place the extracted CSV files in:

```text
data/PRSA_Data_20130301-20170228/
```

The main notebook was originally written for Google Colab, so update the data path before running locally:

```python
path = "data/PRSA_Data_20130301-20170228/"
```

The raw dataset is intentionally ignored by Git so the repository stays lightweight.
