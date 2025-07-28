# 1% nothing capture logic
import os
import time
import cv2
import mss
import numpy as np
import pandas as pd
import random
from datetime import datetime
from pynput import keyboard
import pygetwindow as gw
from queue import SimpleQueue


WINDOW_TITLE = "SM-S918B"
BASE_DIR = "frames"
CROP_TOP = 150
POLL_INTERVAL = 0.1  

def get_game_window_rect(title_contains=WINDOW_TITLE):
    for w in gw.getWindowsWithTitle(title_contains):
        if w.visible:
            return w.left, w.top, w.width, w.height
    return None

def load_and_increment_run_counter(counter_file=os.path.join(BASE_DIR, "run_counter.txt")):
    os.makedirs(os.path.dirname(counter_file), exist_ok=True)
    run_id = 0
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            try:
                run_id = int(f.read().strip())
            except (ValueError, IOError):
                run_id = 0
    run_id += 1
    with open(counter_file, "w") as f:
        f.write(str(run_id))
    return run_id

def main():
    app_state = {'running': True}

    while app_state['running']:

        rect = get_game_window_rect()
        if rect is None:
            print("Game window not found")
            time.sleep(2)
            continue

        left, top, width, height = rect
        run_id = load_and_increment_run_counter()
        run_folder = os.path.join(BASE_DIR, f"run_{run_id:05d}")
        os.makedirs(run_folder, exist_ok=True)
        csv_path = os.path.join(BASE_DIR, "supervised_data.csv")

        records = []
        frame_idx = 0
        pressed_keys = set()
        event_q = SimpleQueue()
        run_state = {'active': False}

        def on_press(key):
            if key == keyboard.Key.esc:
                run_state['active'] = False
                app_state['running'] = False
                return False  

            if hasattr(key, 'char') and key.char == 'q':
                run_state['active'] = not run_state['active'] 

            if run_state['active']:
                if key in [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down]:
                    if key not in pressed_keys:
                        pressed_keys.add(key)
                        label = {
                            keyboard.Key.left: "left", keyboard.Key.right: "right",
                            keyboard.Key.up: "up", keyboard.Key.down: "down",
                        }[key]
                        event_q.put(label)

        def on_release(key):
            if key in pressed_keys:
                pressed_keys.remove(key)

        def capture(label, sct, monitor):
            nonlocal frame_idx
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame_filename = f"run_{run_id:05d}_frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(run_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            records.append({
                "run_id": run_id, "frame_idx": frame_idx,
                "frame_path": frame_path, "label": label,
                "timestamp": datetime.now().isoformat()
            })
            frame_idx += 1
            print(f"Captured: {label} ({frame_filename})")

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        print(f"\nReady for run {run_id:05d}. Press 'q' to start capturing")

        while not run_state['active'] and app_state['running']:
            time.sleep(0.1)
        
        if run_state['active']:
            print(f"Capturing run {run_id:05d}... Press 'q' to stop and save")

        with mss.mss() as sct:
            monitor = {
                "top": top + CROP_TOP, "left": left,
                "width": width, "height": height - CROP_TOP
            }
            next_nothing_check = time.time()

            while run_state['active'] and app_state['running']:

                while not event_q.empty():
                    label = event_q.get()
                    capture(label, sct, monitor)


                now = time.time()
                if now >= next_nothing_check:
                    if not pressed_keys:
                        if random.random() >= 0.90:
                            capture("nothing", sct, monitor)
                    next_nothing_check = now + POLL_INTERVAL
                
                time.sleep(0.01)


        listener.stop()
        listener.join()
        
        if records:
            df = pd.DataFrame(records)
            file_exists = os.path.exists(csv_path)
            df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
            print(f"Run {run_id:05d} finished. Saved {len(records)} entries to {csv_path}")



if __name__ == "__main__":
    main()


# #Manual Nothing Capture
# import os
# import time
# import cv2
# import mss
# import numpy as np
# import pandas as pd
# import random
# from datetime import datetime
# from pynput import keyboard
# import pygetwindow as gw
# from queue import SimpleQueue
# import json

# # ==== CONFIG ====
# WINDOW_TITLE = "SM-S918B"
# BASE_DIR = "frames"
# CROP_TOP = 150
# MAX_SAMPLES_PER_ACTION = 2000
# ACTION_COUNTS_FILE = os.path.join(BASE_DIR, "action_counts.json")


# def get_game_window_rect(title_contains=WINDOW_TITLE):
#     for w in gw.getWindowsWithTitle(title_contains):
#         if w.visible:
#             return w.left, w.top, w.width, w.height
#     return None

# def load_and_increment_run_counter(counter_file=os.path.join(BASE_DIR, "run_counter.txt")):
#     os.makedirs(os.path.dirname(counter_file), exist_ok=True)
#     run_id = 0
#     if os.path.exists(counter_file):
#         with open(counter_file, "r") as f:
#             try:
#                 run_id = int(f.read().strip())
#             except (ValueError, IOError):
#                 run_id = 0
#     run_id += 1
#     with open(counter_file, "w") as f:
#         f.write(str(run_id))
#     return run_id

# def load_action_counts(file_path):
#     defaults = {"left": 0, "right": 0, "up": 0, "down": 0, "nothing": 0}
#     if os.path.exists(file_path):
#         try:
#             with open(file_path, "r") as f:
#                 loaded_counts = json.load(f)
#             defaults.update(loaded_counts)
#         except (json.JSONDecodeError, IOError) as e:
#             print(f"Warning: Could not parse {file_path} ({e}). Starting with fresh counts.")
#     return defaults

# def save_action_counts(file_path, counts):
#     with open(file_path, 'w') as f:
#         json.dump(counts, f, indent=4)
#     print(f" Saved final action counts to {file_path}.")


# def main():
#     app_state = {'running': True}
    
#     os.makedirs(BASE_DIR, exist_ok=True)
#     action_counts = load_action_counts(ACTION_COUNTS_FILE)

#     all_done = True
#     for action, count in action_counts.items():
#         is_full = count >= MAX_SAMPLES_PER_ACTION
#         if not is_full: all_done = False
#         print(f"- {action.capitalize()}: {count}/{MAX_SAMPLES_PER_ACTION} {'(COMPLETE)' if is_full else ''}")
#     print("\n")

#     if all_done:
#         print("All actions have reached their sample limits")
#         return

#     while app_state['running']:
#         rect = get_game_window_rect()
#         if rect is None:
#             print("Game window not found")
#             time.sleep(2)
#             continue

#         left, top, width, height = rect
#         run_id = load_and_increment_run_counter()
#         run_folder = os.path.join(BASE_DIR, f"run_{run_id:05d}")
#         os.makedirs(run_folder, exist_ok=True)
#         csv_path = os.path.join(BASE_DIR, "supervised_data.csv")

#         records = []
#         frame_idx = 0
#         pressed_keys = set()
#         event_q = SimpleQueue()
#         run_state = {'active': False}
#         full_actions_reported_this_run = set()

#         def capture(label, sct, monitor):
#             nonlocal frame_idx, records

#             if action_counts[label] >= MAX_SAMPLES_PER_ACTION:
#                 if label not in full_actions_reported_this_run:
#                     print(f"Limit for '{label}' reached. No more will be recorded.")
#                     full_actions_reported_this_run.add(label)
#                 return

#             img = np.array(sct.grab(monitor))
#             frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#             frame_filename = f"run_{run_id:05d}_frame_{frame_idx:06d}.jpg"
#             frame_path = os.path.join(run_folder, frame_filename)
#             cv2.imwrite(frame_path, frame)
            
#             action_counts[label] += 1
            
#             records.append({
#                 "run_id": run_id, "frame_idx": frame_idx,
#                 "frame_path": frame_path, "label": label,
#                 "timestamp": datetime.now().isoformat()
#             })
#             frame_idx += 1
#             print(f"Captured: {label} ({action_counts[label]}/{MAX_SAMPLES_PER_ACTION}) ({frame_filename})")

#         def on_press(key):
#             if key == keyboard.Key.esc:
#                 run_state['active'] = False
#                 app_state['running'] = False
#                 return False

#             if hasattr(key, 'char') and key.char == 'q':
#                 run_state['active'] = not run_state['active']

#             if run_state['active']:
#                 if hasattr(key, 'char') and key.char == 'z':
#                     event_q.put("nothing")

#                 if key in [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down]:
#                     if key not in pressed_keys:
#                         pressed_keys.add(key)
#                         label = {
#                             keyboard.Key.left: "left", keyboard.Key.right: "right",
#                             keyboard.Key.up: "up", keyboard.Key.down: "down",
#                         }[key]
#                         event_q.put(label)

#         def on_release(key):
#             if key in pressed_keys:
#                 pressed_keys.remove(key)

#         listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#         listener.start()

#         print(f"\n--- Ready for run {run_id:05d}. Press 'q' to start capturing")

#         while not run_state['active'] and app_state['running']:
#             time.sleep(0.1)
        
#         if run_state['active']:
#             print(f"Capturing run {run_id:05d}... Press 'q' to stop.")

#         with mss.mss() as sct:
#             monitor = {
#                 "top": top + CROP_TOP, "left": left,
#                 "width": width, "height": height - CROP_TOP
#             }
            
#             while run_state['active'] and app_state['running']:

#                 while not event_q.empty():
#                     label = event_q.get()
#                     capture(label, sct, monitor)

#                 if all(c >= MAX_SAMPLES_PER_ACTION for c in action_counts.values()):
#                     run_state['active'] = False
#                     app_state['running'] = False
                
#                 time.sleep(0.01)

#         listener.stop()
#         listener.join()
        
#         if records:
#             df = pd.DataFrame(records)
#             file_exists = os.path.exists(csv_path)
#             df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
#             print(f"Run {run_id:05d} finished. Saved {len(records)} new entries to {csv_path}")
#         elif not app_state['running']:
#              pass
#         else:
#              print(f"Run {run_id:05d} stopped. No new frames were recorded.")


#     save_action_counts(ACTION_COUNTS_FILE, action_counts)


# if __name__ == "__main__":
#     main()