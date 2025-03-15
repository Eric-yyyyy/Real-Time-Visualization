import serial
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
reading_done = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from scipy.signal import butter, filtfilt
#output_folder = "rest_signal"

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def low_pass_filter(data, cutoff=20, fs=1000, order=4):

    nyquist = 0.5 * fs  
    normal_cutoff = cutoff / nyquist 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def read_for_duration(port, baud, out_filename):
    global reading_done

    try:
        ser = serial.Serial(port, baud, timeout=None)
    except Exception as e:
        print(f"Error opening serial port: {e}")
        reading_done = True
        return

    print(f"Reading from {port} at {baud} baud. Press Ctrl+C to stop.")

    header_seq = b"\xAA\xAA\x5F\x00"
    found_header = False
    buffer = b""

    with open(out_filename, 'w', encoding='utf-8') as file:
        try:
            while not reading_done:
                raw_data = ser.read(1024)
                if raw_data:
                    buffer += raw_data
                    if not found_header:
                        idx = buffer.find(header_seq)
                        if idx != -1:
                            buffer = buffer[idx:]
                            found_header = True

                    if found_header:
                        while len(buffer) >= 64:
                            packet = buffer[:64]
                            buffer = buffer[64:]
                            hex_str = ' '.join(f'{b:02X}' for b in packet)
                            file.write(hex_str + '\n')
                            file.flush()
        except KeyboardInterrupt:
            print("KeyboardInterrupt in read_for_duration: Exiting reader thread.")
        finally:
            reading_done = True
            ser.close()

    print(f"Data capture complete. Saved to {out_filename}")


def parse_emg_data_from_stream(file_path):
    channels = [[] for _ in range(8)]
    try:
        with open(file_path, 'r') as file:
            raw_data = file.read().replace('\n', ' ')
        hex_values = raw_data.strip().split()
    except Exception as e:
        print("Error reading file:", e)
        return channels

    i = 0
    while i < len(hex_values):
        # Look for the start sequence AA AA 5F 00
        if i + 3 < len(hex_values) and hex_values[i:i + 4] == ['AA', 'AA', '5F', '00']:
            start = i + 16
            end = start + (10 * 8) + 2
            if end <= len(hex_values):
                useful_data = hex_values[start:start + (10 * 8)]
                # Break the data into groups of 8 
                for j in range(0, len(useful_data), 8):
                    pack = useful_data[j:j + 8]
                    for k, value in enumerate(pack):
                        try:
                            channels[k].append(int(value, 16))
                        except ValueError:
                            channels[k].append(0)
                i = end
            else:
                break
        else:
            i += 1
        
    """for i in range(len(channels)):
        if len(channels[i]) > 0: 
            channels[i] = moving_average(np.array(channels[i])).tolist()"""
    
    """
    for i in range(len(channels)):
        if len(channels[i]) > 0:
            raw_signal = np.array(channels[i])
            filtered_signal = low_pass_filter(raw_signal, cutoff=20, fs=1000, order=4) 
            channels[i] = filtered_signal.tolist()  # Convert back to list 
    """      
    return channels

"""
def live_plot(file_path, refresh_delay=1):
    global reading_done

    plt.ion()
    fig, axs = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
    lines = []

    for i, ax in enumerate(axs):
        line, = ax.plot([], [], label=f'Channel {i + 1}', linewidth=0.8, alpha=0.7)
        lines.append(line)
        ax.set_ylabel('EMG Signal')
        ax.grid(True)
        ax.legend(loc='upper right')

        # Force a fixed Y-axis scale (adjust as needed)
        ax.set_ylim(100, 200) 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    axs[-1].set_xlabel('Sample Index')

    while not reading_done:
        channels = parse_emg_data_from_stream(file_path)
        # Update each subplot 
        for i, channel in enumerate(channels):
            lines[i].set_data(range(len(channel)), channel)
            axs[i].relim()
            axs[i].autoscale_view(scalex=True, scaley=False)  # Only autoscale X, keep Y fixed

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(refresh_delay)

    channels = parse_emg_data_from_stream(file_path)
    for i, channel in enumerate(channels):
        lines[i].set_data(range(len(channel)), channel)
        axs[i].relim()
        axs[i].autoscale_view(scalex=True, scaley=False) 

    plt.ioff() 
    plt.show()
"""    
'''
def live_plot(file_path, refresh_delay=1, window_size=10000):
    global reading_done

    plt.ion()
    fig, axs = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
    lines = []
    
    # Initialize data buffers for each channel
    buffers = [[] for _ in range(8)]

    for i, ax in enumerate(axs):
        line, = ax.plot([], [], label=f'Channel {i + 1}', linewidth=0.8, alpha=0.7)
        lines.append(line)
        ax.set_ylabel('EMG Signal')
        ax.grid(True)
        ax.legend(loc='upper right')

        # Force a fixed Y-axis scale
        ax.set_ylim(120, 160) 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    axs[-1].set_xlabel('Sample Index')

    while not reading_done:
        channels = parse_emg_data_from_stream(file_path)
        
        for i, channel in enumerate(channels):
            if not channel:
                continue  # Skip empty channels

            # Update buffer with latest data
            buffers[i].extend(channel)
            buffers[i] = buffers[i][-window_size:]  

            lines[i].set_data(range(len(buffers[i])), buffers[i])

            axs[i].relim()
            axs[i].autoscale_view(scalex=True, scaley=False)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(refresh_delay)

    plt.ioff()
    plt.show()
'''
"""
def live_plot_optimized(file_path, refresh_delay=1, window_size=10000):
    global reading_done

    plt.ion()
    fig, axs = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
    lines = []
    

    buffers = [[] for _ in range(8)]

    for i, ax in enumerate(axs):
        line, = ax.plot([], [], label=f'Channel {i + 1}', linewidth=0.8, alpha=0.7)
        lines.append(line)
        ax.set_ylabel('EMG Signal')
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_ylim(100, 140) 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    axs[-1].set_xlabel('Sample Index')

    last_position = 0  

    while not reading_done:
        try:
            with open(file_path, 'r') as file:
                file.seek(last_position)  # Move to the last read position
                new_data = file.read()  # Read only new data
                last_position = file.tell()  # Update last position
            
            if new_data.strip():  # Process only if new data is present
                new_channels = parse_emg_data_from_stream(file_path)
                
                for i, channel in enumerate(new_channels):
                    if not channel:
                        continue  # Skip empty channels

                    # Append new data to buffer and maintain window size
                    buffers[i].extend(channel)
                    buffers[i] = buffers[i][-window_size:]  # Keep only latest data

                    lines[i].set_data(range(len(buffers[i])), buffers[i])
                    axs[i].relim()
                    axs[i].autoscale_view(scalex=True, scaley=False)

                fig.canvas.draw()
                fig.canvas.flush_events()

        except Exception as e:
            print(f"Error reading file: {e}")

        time.sleep(refresh_delay)

    plt.ioff()
    plt.show()
"""
def extract_summary_features(channels):

    selected_channels = [channels[4]]  
    features = []
    
    for channel in selected_channels:
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.max(channel) - np.min(channel),  #applitude
           
        ])
    
    return np.array(features)
def train_classifier(model_type="svm"):
   
    X = []
    y = []

    #strength_folder = "Strength_data"
    strength_folder = "Pure_Strength"
    for file in os.listdir(strength_folder):
        if file.endswith(".txt"):
            channels = parse_emg_data_from_stream(os.path.join(strength_folder, file))
            features = extract_summary_features(channels)
            X.append(features)
            y.append(1)  # Label 1 = Strength

    #rest_folder = "Rest_data"
    rest_folder = "Pure_Rest"
    for file in os.listdir(rest_folder):
        if file.endswith(".txt"):
            channels = parse_emg_data_from_stream(os.path.join(rest_folder, file))
            features = extract_summary_features(channels)
            X.append(features)
            y.append(0) 


    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        clf = SVC(kernel="rbf", C=100.0, gamma="scale", probability=True)
    else:
        raise ValueError("Choose 'random_forest' or 'svm'.")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Model Accuracy ({model_type}): {accuracy:.2f}%")

    return clf


def live_plot(file_path, clf, refresh_delay=1, window_size=10000):
    global reading_done

    plt.ion()
    fig, axs = plt.subplots(9, 1, figsize=(12, 18), sharex=True)
    lines = []
    
    
    buffers = [[] for _ in range(8)]
    classify_buffer = [[] for _ in range(8)]

    classification_ax = axs[-1]
    classification_text = classification_ax.text(0.5, 0.5, "Classifying...", fontsize=14, ha='center', va='center')
    classification_ax.set_xticks([])
    classification_ax.set_yticks([])
    classification_ax.set_frame_on(False)

    for i, ax in enumerate(axs[:-1]): 
        line, = ax.plot([], [], label=f'Channel {i + 1}', linewidth=0.8, alpha=0.7)
        lines.append(line)
        ax.set_ylabel('EMG Signal')
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_ylim(100, 140) 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.set_xlim(0, window_size)
    
    
        ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
        


    axs[-2].set_xlabel('Sample Index')  
    last_position = 0  # Track the last read position

    while not reading_done:
        try:
            with open(file_path, 'r') as file:
                file.seek(last_position)  # Move to the last read position
                new_data = file.read()  # Read only new data
                last_position = file.tell()  
            
            if new_data.strip(): 
                new_channels = parse_emg_data_from_stream(file_path)
                
                for i, channel in enumerate(new_channels):
                    if not channel:
                        continue  

                   
                    buffers[i].extend(channel)
                    buffers[i] = buffers[i][-window_size:] 
                    classify_buffer[i].extend(channel)
                    classify_buffer[i] = classify_buffer[i][-100:] 

                    lines[i].set_data(range(len(buffers[i])), buffers[i])
                    axs[i].relim()
                    axs[i].autoscale_view(scalex=True, scaley=False)

                features = extract_summary_features(classify_buffer).reshape(1, -1)
                prediction = clf.predict(features)[0]
                label = ""
                if prediction == 1:
                    label = "Strength"
                    #print("Strength")
                else:
                    label = "Rest"
                    #print("Rest")
                with open(r"C:\Users\andre\OneDrive\Desktop\New folder\VirtualHands\VirtualHands\Assets\EMGData\prediction.txt", "w") as f:
                    #f.write(label)   
                    f.write(str(np.max(classify_buffer[4][-100:]) - np.min(classify_buffer[4][-100:]))) 

            
                classification_text.set_text(f"Prediction: {label}")

                fig.canvas.draw()
                fig.canvas.flush_events()

        except Exception as e:
            print(f"Error reading file: {e}")

        time.sleep(refresh_delay)

    plt.ioff()
    plt.show()

def main():
    global reading_done

    port = 'COM4'        
    baud = 115200        
    output_dir = r"C:\Users\andre\OneDrive\Desktop\New folder\VirtualHands\VirtualHands\Assets\EMGData"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #out_filename = os.path.join(output_dir, "emg_live_data.txt")
    out_filename = "emg_data_live.txt"

    
    print("\nTraining Classifier...")
    clf = train_classifier(model_type="random_forest") 
    
    try:
       
        read_thread = threading.Thread(
            target=read_for_duration,
            args=(port, baud, out_filename),
            daemon=True
        )
        read_thread.start()
        
       
        live_plot(out_filename, clf, refresh_delay=0.1)

        read_thread.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt in main: shutting down.")
        reading_done = True
    finally:
        print("Exiting program.")
    


if __name__ == '__main__':
    main()
