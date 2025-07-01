import os
import pandas as pd
import shutil
from pathlib import Path
import random

# === CONFIG ===
CSV_PATH = "nexar-collision-prediction/train.csv"
VIDEO_DIR = "nexar-collision-prediction/train"
OUTPUT_DIR = "final_dataset"
SPLIT_COUNTS = {"train": 250, "val": 50, "test": 80}

# === LOAD AND PREPARE CSV ===
df = pd.read_csv(CSV_PATH)
df["label"] = df["target"].map({1: "crash", 0: "no_crash"})

# Ensure enough clips exist
total_needed = sum(SPLIT_COUNTS.values())
crash_df = (
    df[df["label"] == "crash"]
    .sample(n=total_needed, random_state=42)
    .reset_index(drop=True)
)
no_crash_df = (
    df[df["label"] == "no_crash"]
    .sample(n=total_needed, random_state=42)
    .reset_index(drop=True)
)

# === SPLIT + COPY FILES ===
splits = {}

for split in SPLIT_COUNTS:
    n = SPLIT_COUNTS[split]

    # Get n rows from each class
    crash_split = crash_df[:n]
    no_crash_split = no_crash_df[:n]
    crash_df = crash_df[n:]
    no_crash_df = no_crash_df[n:]

    # Combine and store
    split_df = pd.concat([crash_split, no_crash_split]).sample(frac=1, random_state=42).reset_index(drop=True)

    splits[split] = split_df

    # Copy video files
    for _, row in split_df.iterrows():
        clip_id = str(row["id"]).zfill(5)  # Convert ID to match file
        label = row["label"]
        filename = f"{clip_id}.mp4"

        src_path = os.path.join(VIDEO_DIR, filename)
        dst_path = os.path.join(OUTPUT_DIR, split, label, filename)
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Missing: {src_path}")

    # Save CSV for this split
    split_df.to_csv(os.path.join(OUTPUT_DIR, f"{split}.csv"), index=False)
    print(f"{split.upper()}: {len(split_df)} total clips ({n} crash + {n} no_crash)")


print(
    "All done! Your data is now split into train / val / test under 'final_dataset'"
)
