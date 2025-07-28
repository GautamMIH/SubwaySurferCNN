import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import glob

# Config and Constants
SUPERVISED_CSV_PATH = 'frames/augmented_data.csv'
REINFORCEMENT_CSVS_DIR = 'old/frames'
IMAGE_DIR = 'frames' # This is still relevant for the supervised data
IMG_SIZE = (240, 240)
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 5


all_dataframes = []

#Load data
if os.path.exists(SUPERVISED_CSV_PATH):
    print(f"Found supervised CSV: {SUPERVISED_CSV_PATH}")
    all_dataframes.append(pd.read_csv(SUPERVISED_CSV_PATH))
else:
    print(f"Warning: Main CSV file not found at '{SUPERVISED_CSV_PATH}'")

#load additional data if you have in the reinforcement directory
if os.path.isdir(REINFORCEMENT_CSVS_DIR):
    additional_csvs = glob.glob(os.path.join(REINFORCEMENT_CSVS_DIR, '*.csv'))
    if additional_csvs:
        print(f"Found {len(additional_csvs)} additional CSVs in '{REINFORCEMENT_CSVS_DIR}'.")

        reinf_df = pd.concat((pd.read_csv(f) for f in additional_csvs), ignore_index=True)

        reinf_df['frame_path'] = reinf_df['frame_path'].str.replace(
            f'{IMAGE_DIR}/', f'{REINFORCEMENT_CSVS_DIR}/', n=1, regex=False
        )
        all_dataframes.append(reinf_df)
    else:
        print(f"Warning: No CSV files found in '{REINFORCEMENT_CSVS_DIR}'.")
else:
    print(f"Warning: Directory not found: '{REINFORCEMENT_CSVS_DIR}'")


if not all_dataframes:
    print("Error: No CSV data files found. Exiting.")
    exit()


df = pd.concat(all_dataframes, ignore_index=True)
print(f"\nTotal records before verification: {len(df)}")

# Making sure the file paths are in the correct format
df['frame_path'] = df['frame_path'].apply(lambda x: os.path.join(*str(x).replace('/', '\\').split('\\')))

initial_record_count = len(df)
path_exists_mask = df['frame_path'].apply(os.path.exists)
df = df[path_exists_mask] # Keep only rows with valid paths

removed_count = initial_record_count - len(df)
if removed_count > 0:
    print(f"Removed {removed_count} records ")
print(f" {len(df)} valid records remaining.")


# Label Mapping for Later Classification
label_map = {'nothing': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}
df['label_idx'] = df['label'].map(label_map)

if df['label_idx'].isnull().any():
    print("Error: One or more labels in the CSV could not be mapped.")
    exit()

label_counts = df['label_idx'].value_counts()
min_count = label_counts.min()
print(f"\nBalancing dataset: limiting all classes to {min_count} samples...")

# Random Undersampling to balance the dataset, Can be safely deleted if you dont need it
df_balanced = (
    df.groupby('label_idx', group_keys=False)
      .apply(lambda x: x.sample(n=min_count, random_state=42))
      .reset_index(drop=True)
)

print("Balanced class distribution:")
print(df_balanced['label_idx'].value_counts())

df = df_balanced

# Split the data
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label_idx'])
train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=42, stratify=train_val_df['label_idx'])

print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")


def create_dataset(dataframe):
    filepaths = dataframe['frame_path'].values
    labels = dataframe['label_idx'].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_image(image_string, channels=3, expand_animations=False)
        image_resized = tf.image.resize(image_decoded, IMG_SIZE)
        image_resized.set_shape([*IMG_SIZE, 3])
        image_rescaled = image_resized / 255.0
        return image_rescaled, label

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)


#Model Definition

def build_simple_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_simple_cnn(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


#Model Training

print(f"\n--- Starting Training ({EPOCHS} epochs) ---")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
)
print("\nTraining finished.")


print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

model.save('actionspred_cnn.keras')
print("\nModel saved")