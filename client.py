import socket
import time
import numpy as np
import cv2
import mss
import struct
import pygetwindow as gw
from pynput.keyboard import Controller, Key

HOST = 'localhost'
PORT = 65432
WINDOW_TITLE = "SM-S918B" 
CROP_TOP = 150 
def get_game_window_rect(title_contains=WINDOW_TITLE):
    try:
        windows = gw.getWindowsWithTitle(title_contains)
        if windows:
            win = windows[0]
            # Return a dictionary for mss
            return {"top": win.top + CROP_TOP, "left": win.left, "width": win.width, "height": win.height - CROP_TOP}
    except Exception as e:
        print(f"Error finding window: {e}")
    return None

def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            print(f"Attempting to connect to server at {HOST}:{PORT}...")
            client_socket.connect((HOST, PORT))
            print("Successfully connected to the server.")
            return client_socket
        except ConnectionRefusedError:
            print("Connection refused")
            time.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}. ")
            time.sleep(5)

def execute_action(action, keyboard_controller):
    key_map = {
        'up': Key.up,
        'down': Key.down,
        'left': Key.left,
        'right': Key.right
    }
    
    if action in key_map:
        key_to_press = key_map[action]
        print(f"Executing action: {action.upper()}")
        keyboard_controller.press(key_to_press)
        time.sleep(0.05) 
        keyboard_controller.release(key_to_press)
    elif action == 'nothing':
        pass 
    else:
        print(f"Warning: Received unknown action '{action}'")

monitor = get_game_window_rect()
if not monitor:
    print(f"Game window with title containing '{WINDOW_TITLE}' not found. Exiting.")
    exit()

print(f"Game window found at: {monitor}")

sct = mss.mss()
client_socket = connect_to_server()
keyboard_controller = Controller()

try:
    print("Starting game client")
    while True:
        img_np = np.array(sct.grab(monitor))

        is_success, im_buf_arr = cv2.imencode(".jpg", img_np)
        if not is_success:
            print("Failed to encode image to JPG.")
            continue
        
        image_bytes = im_buf_arr.tobytes()
        
        try:
            start_time = time.time()
            size = len(image_bytes)
            client_socket.sendall(struct.pack('>I', size))
            client_socket.sendall(image_bytes)


            prediction = client_socket.recv(1024).decode('utf-8')
            execute_action(prediction, keyboard_controller)
            end_time = time.time()
            print(f"Received prediction: {prediction} (Time taken: {(end_time - start_time)*1000:.2f}ms)")


        except (ConnectionResetError, BrokenPipeError):
            print("Connection to server lost. Attempting to reconnect...")
            client_socket.close()
            client_socket = connect_to_server()
        except Exception as e:
            print(f"An error occurred during communication: {e}")
            break
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopping client as requested.")
finally:
    print("Closing client socket.")
    client_socket.close()
    cv2.destroyAllWindows()
