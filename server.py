import socket
import numpy as np
import tensorflow as tf
import cv2
import struct


HOST = '0.0.0.0'
PORT = 65432
MODEL_PATH = 'actionspred_cnn.keras'
IMG_SIZE = (240, 240)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    model.summary() 
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

LABEL_MAP = {0: 'nothing', 1: 'up', 2: 'down', 3: 'left', 4: 'right'}

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
])
def predict_step(image_tensor):
    predictions = model(image_tensor, training=False)
    return predictions

def preprocess_image(image_bytes):
    try:
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)


        if image is None:
            print("Error: failed to decode image.")
            return None

        image_resized = cv2.resize(image, IMG_SIZE)
        image_rescaled = image_resized / 255.0
        
        return np.expand_dims(image_rescaled, axis=0).astype(np.float32)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()
print(f" Server listening on {HOST}:{PORT}")

while True:
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    
    try:
        while True:
            data_size_bytes = conn.recv(4)
            if not data_size_bytes:
                break 
            
            data_size = struct.unpack('>I', data_size_bytes)[0]
            
   
            image_data = b''
            while len(image_data) < data_size:
                packet = conn.recv(data_size - len(image_data))
                if not packet:
                    break
                image_data += packet
            
            if len(image_data) != data_size:
                print("Error: Did not receive complete image data.")
                continue

  
            processed_image_np = preprocess_image(image_data)
            
            if processed_image_np is not None:         
                processed_image_tensor = tf.convert_to_tensor(processed_image_np)
                prediction_tensor = predict_step(processed_image_tensor)
                predicted_class_idx = np.argmax(prediction_tensor.numpy()[0])
                predicted_label = LABEL_MAP[predicted_class_idx]
                conn.sendall(predicted_label.encode('utf-8'))

    except ConnectionResetError:
        print(f"Connection with {addr} was lost.")
    except Exception as e:
        print(f"An error occurred with {addr}: {e}")
    finally:
        conn.close()