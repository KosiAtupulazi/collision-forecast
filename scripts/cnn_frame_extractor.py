
# %%
import os
import cv2
import pandas as pd
import math
import random
import numpy as np

# %%
for dataset_split in ['train', 'val', 'test']:
    print(f"Processing {dataset_split} split...")
    main_dir = "/home/atupulazi/personal_projects/collision-forecast/final_dataset"
    video_dir = os.path.join(main_dir, dataset_split)
    csv_path = os.path.join(main_dir, f"{dataset_split}.csv")
    output_dir = os.path.join("/home/atupulazi/personal_projects/collision-forecast/frames", dataset_split)

    #margin = 1
    #how to print files and debug if i get errors???
    df = pd.read_csv(csv_path)

    clip_labels = [] #output of extracted clips and their labels; will write to csv later
    print(df.head())

    for event in ['crash', 'no_crash']: 
        eventfolder_path = os.path.join(video_dir, event) #the path to each event

        for filename in os.listdir(eventfolder_path):
            if not filename.endswith('.mp4'):
                continue

            raw_id = filename.split('.')[0] #seperates the file name ie. 00064.mp4 = 00064
            video_id = int(raw_id.lstrip('0') or '0') #converts the value to int if its not an empty string and 0 if it is

            event_video = os.path.join(eventfolder_path, filename) # builds file path to the speccific video
            
            df_row = df[df['id'] == video_id] #match the current video's id to the csv id
            if df_row.empty:
                print(f"No label found for {filename}")
                continue #if the id is not in the csv, skip it
            df_row = df_row.iloc[0] # gets the first matching row from the dataframe

            #open video
            video_capture = cv2.VideoCapture(event_video)

            #auto detect the video fps (Frames Per Second)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #how many frames in total for the whole video

            print(f"Video: {filename}, FPS: {fps}, Total Frames: {total_frames}")

            #how many consecutive clips at a time
            clip_len = 16

            #if we know where the system is supposed to alert the driver, extract 16 frames around that moment (8 before the center and 8 after)
            if df_row["label"] == "crash" and pd.notna(df_row["time_of_alert"]):
                center_time = float(df_row['time_of_alert']) #crash time saved in the dataframe
                forecast_time = center_time - 2 # 2 seconds before the crash
                if forecast_time <= 0:
                    continue # skip if the video is too short
                center_frame = int(forecast_time * fps) #tells the frame where the crash likely happened ie. 22.752 * 30 = 682.56 → int(682.56) = 682
                start_frame = max(0, center_frame - clip_len // 2) #start 8 frames before the crash ie center_frame - 8 = 682 - 8 = 674
                end_frame = min(total_frames, start_frame + clip_len) #begin from the start frame to the 16th clip ie so we end at 674 + 16 = 690
                clip_indices = list(range(start_frame, end_frame)) #a list of the the frames of curr vid range(674, 690) → [674, 675, 676, ..., 689]

            else:
                max_start = total_frames - clip_len #the latest possisble place to START getting a clip without going over the total frames
                if max_start <= 0:
                    continue #this means that this video is too short to extract a clip based on the clip_len so we skip it
                start_frame = random.randint(0, max_start)
                clip_indices = list(range(start_frame, start_frame + clip_len))

            frames = [] #stores each frame we extract from a video

            for frame_idx in clip_indices: #iterates the list of frames in clip_indices
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) #goes to the exact frame we want
                success, frame = video_capture.read() #reads that frame if successfull
                if not success:
                    print(f"[WARNING] Failed to read frame {frame_idx} in {filename}")
                    continue #if not successful in reading the frame, skip it
                frame = cv2.resize(frame, (224,224)) #resize to 224x224 pixels (input size for videomae on videos)
                # opencv = (height, width, channels) → (112, 112, 3)
                #pytorch = (channels, height, width) → (3, 112, 112)
                frame = frame.transpose(2, 0, 1) #converts the opencv format to pytorch format 9swaps the dimensions)
                frames.append(frame) #stores the processed frame in the frame list

            if len(frames) == clip_len: #only keep clips containing eaxctly 16 frames to ensure consistency
                tensor = np.stack(frames) # (16, 3, 224, 224) #each frame has shape(3, 112, 112), and you stack 16 of them into a 4D array
                tensor = tensor.transpose(1, 0, 2, 3)    # (3, 16, 112, 112) #convert to pytorch format (channels, time, height, width)


                save_dir = os.path.join(output_dir, df_row["label"]) #create seperate folder for each label
                os.makedirs(save_dir, exist_ok=True) #creates the folder if it doesn’t exist.
                clip_name = f"{raw_id}_clip.npy" #raw_id = "00822" → clip_name = "00822_clip.npy"
                save_path = os.path.join(save_dir, clip_name) #Full file path where the tensor will be saved.
                print(f"Saving to: {save_path}") #debug 

                np.save(save_path, tensor) #actually saves the tensor to disk as a .npy file 

                clip_labels.append((clip_name, df_row["label"])) # a list of tuples to track what label goes with which clip.
                print(f"[{dataset_split.upper()}] Processed {filename} → Saved: {clip_name}")


            video_capture.release()

    csv_save_path = os.path.join(output_dir, f"{dataset_split}_clip_labels.csv")
    label_df = pd.DataFrame(clip_labels, columns=pd.Index(["clip_name", "label"]))
    label_df.to_csv(csv_save_path, index=False)

    print(f"Finished extracting clips for: {dataset_split}")



# %%



