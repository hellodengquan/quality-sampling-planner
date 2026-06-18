"""生成示例批次数据."""

import random

import pandas as pd

random.seed(42)

batches = [
    "BATCH-2024-05-01",
    "BATCH-2024-05-02",
    "BATCH-2024-05-03",
    "BATCH-2024-05-04",
    "BATCH-2024-05-05",
]
suppliers = ["供应商甲", "供应商乙", "供应商丙"]
products = ["产品A", "产品B", "产品C", "产品D"]
workshops = ["1号车间", "2号车间", "3号车间"]

rows = []
idx = 1
for b in batches:
    n = random.randint(80, 150)
    for _ in range(n):
        rows.append(
            {
                "sample_id": f"S{idx:05d}",
                "batch_id": b,
                "supplier": random.choice(suppliers),
                "product": random.choice(products),
                "workshop": random.choice(workshops),
                "weight": round(random.gauss(500, 20), 1),
                "thickness": round(random.gauss(2.5, 0.2), 2),
                "temperature": round(random.gauss(23, 1.5), 1),
                "defect_flag": 1 if random.random() < 0.04 else 0,
            }
        )
        idx += 1

df = pd.DataFrame(rows)
out = "examples/batches.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"生成 {len(df)} 条数据, {df['batch_id'].nunique()} 个批次 -> {out}")
