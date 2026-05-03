# Data

The notebook expects two CSV files in this folder:

```
data/
├── Train.csv   # 20,000 rows × 41 cols (V1–V40 + Target)
└── Test.csv    #  5,000 rows × 41 cols (V1–V40 + Target)
```

## Why aren't they in the repo?

The original sensor data shared by **ReneWind** is **confidential and ciphered**, and the project description asks contributors not to redistribute it. The CSVs are therefore intentionally **excluded from version control** via [`.gitignore`](../.gitignore).

## How to obtain the data

If you are reviewing this portfolio project and would like to reproduce the results, you can:

1. Use the `Train.csv` / `Test.csv` shared with you in the original course / project pack, **or**
2. Contact the repository author for guidance.

Once you have the files, drop them into this folder. The notebook reads them with relative paths:

```python
data       = pd.read_csv("../data/Train.csv")
data_test  = pd.read_csv("../data/Test.csv")
```

## Schema

| Column      | Type    | Description                                  |
|-------------|---------|----------------------------------------------|
| `V1` … `V40`| float   | Anonymized / ciphered sensor readings        |
| `Target`    | int (0/1) | `1` = generator failure, `0` = no failure  |
