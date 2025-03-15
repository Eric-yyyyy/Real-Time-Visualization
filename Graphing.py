import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
#output_folder = "rest_signal"
def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=20, highcut=200, fs=1000, order=4):
    
    nyquist = 0.5 * fs 
    low = lowcut / nyquist
    high = highcut / nyquist


    b, a = butter(order, [low, high], btype='band')


    filtered_data = filtfilt(b, a, data)
    return filtered_data

#os.makedirs(output_folder, exist_ok=True)
def parse_emg_data_from_stream(file_path):
    

    channels = [[] for _ in range(8)]
    with open(file_path, 'r') as file:
        raw_data = file.read().replace('\n', ' ') 
    hex_values = raw_data.strip().split()
    #print(hex_values)
    i = 0
    while i < len(hex_values):
        if i + 3 < len(hex_values) and hex_values[i:i + 4] == ['AA', 'AA', '5F', '00']:
            start = i + 16
            end = start + (10 * 8) + 2
            if end <= len(hex_values):
                useful_data = hex_values[start:start + (10 * 8)]

                data_packs = []
                for j in range(0, len(useful_data), 8):
                    pack = useful_data[j:j + 8] 
                    data_packs.append(pack)

                for pack in data_packs:
                    for k, value in enumerate(pack):
                        channels[k].append(int(value, 16)) 
                
                
                i = end
            else:
                break 
        else:
            i += 1
    """        
    for i in range(len(channels)):
        if len(channels[i]) > 0: 
            channels[i] = moving_average(np.array(channels[i])).tolist()
    """
    """
    for i in range(len(channels)):
        if len(channels[i]) > 0:
            channels[i] = bandpass_filter(np.array(channels[i])).tolist()
    """       
    return channels
def extract_summary_features(channels):

    selected_channels = [channels[4], channels[5]]  
    features = []
    
    for channel in selected_channels:
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.max(channel) - np.min(channel),  #applitude
            
        ])
    
    return np.array(features)
"""
def train_classifier(strength_file, rest_file):
    X = []
    y = []

    # Load Strength Data
    channels = parse_emg_data_from_stream(strength_file)
    features = extract_summary_features(channels)
    X.append(features)
    y.append(1)  # Label 1 = Strength


    channels = parse_emg_data_from_stream(rest_file)
    features = extract_summary_features(channels)
    X.append(features)
    y.append(0)  # Label 0 = Rest

    X = np.array(X)
    y = np.array(y)

    # Train Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return clf
"""
def train_classifier(model_type="svm"):
   
    X = []
    y = []

    strength_folder = "Strength_data"
    for file in os.listdir(strength_folder):
        if file.endswith(".txt"):
            channels = parse_emg_data_from_stream(os.path.join(strength_folder, file))
            features = extract_summary_features(channels)
            X.append(features)
            y.append(1)  # Label 1 = Strength

    rest_folder = "Rest_data"
    for file in os.listdir(rest_folder):
        if file.endswith(".txt"):
            channels = parse_emg_data_from_stream(os.path.join(rest_folder, file))
            features = extract_summary_features(channels)
            X.append(features)
            y.append(0)  # Label 0 = Rest


    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    else:
        raise ValueError("Choose 'random_forest' or 'svm'.")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Model Accuracy ({model_type}): {accuracy:.2f}%")

    return clf

def plot_emg_channels(channels):

    plt.figure(figsize=(12, 8))
    for i, channel_data in enumerate(channels):
        plt.plot(channel_data, label=f'Channel {i + 1}', alpha=0.7, linewidth=0.8)

    plt.title('EMG Data for 8 Channels')
    plt.xlabel('Index')
    plt.ylabel('EMG Signal Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_emg_channels_subplots(channels):

    fig, axs = plt.subplots(8, 1, figsize=(12, 16), sharex=True)

    for i, ax in enumerate(axs):
        ax.plot(channels[i], label=f'Channel {i + 1}', alpha=0.7, linewidth=0.8)
        ax.set_title(f'Channel {i + 1}')
        ax.grid(True)
        #ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.arange(round(ymin, -1), round(ymax, -1) + 20, 20))
        ax.set_ylim(100, 130) 

    plt.xlabel('Index')
    plt.tight_layout()
    plt.show()
   
    """
def plot_emg_channels_subplots(channels, output_path):
    
    fig, axs = plt.subplots(8, 1, figsize=(12, 16), sharex=True)

    for i, ax in enumerate(axs):
        ax.plot(channels[i], label=f'Channel {i + 1}', alpha=0.7, linewidth=0.8)
        ax.set_title(f'Channel {i + 1}')
        ax.grid(True)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.arange(round(ymin, -1), round(ymax, -1) + 20, 20))

    plt.xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig(output_path)  
    plt.close() 
    """
def classify_new_file(clf, file_path):
    channels = parse_emg_data_from_stream(file_path)
    features = extract_summary_features(channels).reshape(1, -1)
    prediction = clf.predict(features)[0]
    label = "Strength" if prediction == 1 else "Rest"
    print(f"Prediction for {file_path}: {label}")
    
def clean_file(file_path):
    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            words = line.split()
            filtered_words = []
            for word in words:
                
                if len(word) == 3 and word.endswith(','):
                    word = word[:-1]  
                if len(word) < 3:
                    filtered_words.append(word)
            cleaned_line = " ".join(filtered_words)
            cleaned_lines.append(cleaned_line)
    
    with open(file_path, 'w') as file:
        file.write("\n".join(cleaned_lines))


def main():
    """
    for i in range(1, 21):
        input_file = os.path.join("Rest_data", f'emg_data_Rest{i}.txt')
        output_file = os.path.join("rest_signal", f'rest{i}_signal.png')

        if os.path.exists(input_file): 
            print(f"Processing {input_file} -> Saving to {output_file}")
            channels = parse_emg_data_from_stream(input_file)
            plot_emg_channels_subplots(channels, output_file)
        else:
            print(f"File {input_file} not found, skipping.")

    print("All files processed and saved.")
    """
    
    file_path = 'emg_live_data.txt' 
    #clean_file(file_path)
    channels = parse_emg_data_from_stream(file_path)
    
    print("Plotting channels in subplots...")
    plot_emg_channels_subplots(channels)
    
    
    """print("\nTraining SVM Classifier...")
    svm_clf = train_classifier(model_type="svm")

    Rest_file = "Test.txt"

    
    print("\nTesting SVM Model:")
    classify_new_file(svm_clf, Rest_file)
    
    Strength_file = "Strength_Base_data.txt"


    print("\nTesting SVM Model:")
    classify_new_file(svm_clf, Strength_file)"""


    
if __name__ == "__main__":
    main()