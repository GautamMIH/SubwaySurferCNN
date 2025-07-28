import pandas as pd
from PIL import Image
import os
import time
from tqdm import tqdm

INPUT_CSV_PATH = 'frames/supervised_data.csv'
OUTPUT_CSV_PATH = 'frames/augmented_data.csv'
FLIPPED_IMAGE_DIR = 'frames/flipped'

os.makedirs(FLIPPED_IMAGE_DIR, exist_ok=True)
print(f"Flipped images will be saved in: '{FLIPPED_IMAGE_DIR}'")

try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Successfully loaded {len(df)} records from '{INPUT_CSV_PATH}'")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
    print("Please ensure the path is correct and the file exists.")
    exit()

print("Processing images for flipping (left/right labels)...")

new_records = []

last_run_id = df['run_id'].max()
new_run_id = last_run_id + 1
print(f"Last run ID was {last_run_id}. New flipped images will be under run ID: {new_run_id}")

new_frame_idx_counter = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    label = row['label']

    if label in ['left', 'right']:
        original_image_path = row['frame_path']
        
        try:
            with Image.open(original_image_path) as img:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
               
                new_label = 'right' if label == 'left' else 'left'
                
                
                original_filename = os.path.basename(original_image_path)
                new_filename = f"flipped_{original_filename}"
                new_image_path = os.path.join(FLIPPED_IMAGE_DIR, new_filename)
                
                flipped_img.save(new_image_path)
                new_row = row.copy()
                new_row['run_id'] = new_run_id
                new_row['frame_idx'] = new_frame_idx_counter 
                new_row['frame_path'] = new_image_path.replace('\\', '/')
                new_row['label'] = new_label
                new_row['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S') 

                new_records.append(new_row)
                new_frame_idx_counter += 1

        except FileNotFoundError:
            print(f"\nWarning: Could not find image {original_image_path}")
        except Exception as e:
            print(f"\nAn error occurred processing {original_image_path}: {e}")

print(f"Generated {len(new_records)} new flipped images.")
if new_records:
    augmented_df = pd.DataFrame(new_records)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"Successfully created augmented dataset with {len(combined_df)} total records.")
    print(f"New dataset saved to: '{OUTPUT_CSV_PATH}'")
else:
    print("No images with 'left' or 'right' labels were found to augment.")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Copied original data to '{OUTPUT_CSV_PATH}'")

