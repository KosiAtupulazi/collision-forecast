import pandas as pd
import os

# Load the prediction CSV
csv_path = "/home/atupulazi/personal_projects/collision-detection/frames/test/demo_predictions.csv"
df = pd.read_csv(csv_path)

# Extract the numeric ID from clip_name and format as zero-padded 5-digit filenames
df["video_file"] = df["clip_name"].str.extract(r"(\d+)")[0].astype(int).apply(lambda x: f"{x:05d}.mp4")

# Add full video path assuming crash and no_crash folders under final_dataset/test/
df["video_path"] = df.apply(
    lambda row: f"final_dataset/test/{row['label']}/{row['video_file']}", axis=1
)

# Preview to make sure it worked
print(df[["clip_name", "label", "video_file", "video_path"]].head())


# Save the updated DataFrame
df.to_csv("demo_predictions_with_paths.csv", index=False)