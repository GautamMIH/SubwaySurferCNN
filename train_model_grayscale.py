import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

CSV_PATH = 'frames/supervised_data.csv' 
IMAGE_DIR = 'frames'          
IMG_SIZE = (224, 224)         
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5 

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"'{CSV_PATH}' was not found.")
    exit()

label_map = {'nothing': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}
df['label_idx'] = df['label'].map(label_map)

if df['label_idx'].isnull().any():
    print("Labels could not be mapped.")
    exit()
 
df['frame_path'] = df['frame_path'].apply(lambda x: os.path.join(*x.split('\\')))


train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label_idx'])
train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=42, stratify=train_val_df['label_idx']) # 0.11 of 0.9 is ~0.1

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")


def create_dataset(dataframe):
    filepaths = dataframe['frame_path'].values
    labels = dataframe['label_idx'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    def _parse_function(filename, label):
       
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        
        image_grayscale = tf.image.rgb_to_grayscale(image_decoded)
        
        image_3_channel = tf.image.grayscale_to_rgb(image_grayscale)

        image_resized = tf.image.resize(image_3_channel, IMG_SIZE)
        image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)
        
        return image_preprocessed, label

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,  
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dropout(0.2),             
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') 
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    ]
)

loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

model.save('actionspred_grayscale.keras')

