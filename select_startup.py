import pandas as pd
import random
import json

# 1️⃣ CSV 파일 불러오기
CSV_PATH = "ai_filtered_startups.csv"  # 파일 이름 맞게 수정 가능

df = pd.read_csv(CSV_PATH)
if df.empty:
    raise ValueError("❌ CSV 파일에 스타트업 데이터가 없습니다.")

# 2️⃣ 무작위로 1개 스타트업 선택
selected = df.sample(1).iloc[0]
startup_name = selected["startup_name"]

# 3️⃣ 선택된 스타트업 출력
print(f"🎯 무작위로 선택된 스타트업: {startup_name}")

