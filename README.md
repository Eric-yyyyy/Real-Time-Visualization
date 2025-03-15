# Real-Time EMG Data Capture and Visualization

This Python script reads EMG data from a serial port, extracts relevant data, and visualizes it in real time using Matplotlib.

## Features
- Reads EMG data from a serial port continuously.
- Detects and filters the header sequence (`AA AA 5F 00`).
- Extracts 8-channel EMG data and plots it in real-time.
- Saves the raw hex data into a file for post-analysis.
- Allows for manual interruption via `Ctrl+C`.


## Requirements
Ensure you have the following dependencies installed:
```bash
pip install pyserial
pip install matplotlib

```

## Usage
### 1. Connect Your EMG Device

By default, the script Concurrent.py:
- Reads data from `COM4` at `115200` baud.
- Writes raw data to `emg_data_live.txt`.
- Plots real-time EMG data from all 8 channels.

If you need to change the serial port, modify the `port` variable in `main()`.
If you want to plot subplots from a txt file, use the Graphing.py and change file_path in main().

### 3. Stopping the Script
To stop data collection and visualization, press:
- `Ctrl+C`

---

## How It Works
### **1. Reading Serial Data**
- Opens a serial connection.
- Reads data continuously in 1024-byte chunks.
- Detects the header (`AA AA 5F 00`) and discards previous data.
- Extracts and saves 64-byte packets to `emg_data_live.txt`.

### **2. Parsing EMG Data**
- Reads the saved file.
- Extracts EMG signals from valid packets.
- Groups data into 8 EMG channels.

### **3. Real-Time Plotting**
- Uses `matplotlib` interactive mode (`plt.ion()`).
- Reads the latest EMG data and updates the graph every second.
- Fixes the Y-axis range (`100-200`) while autoscaling the X-axis.

---

## Customization
### **Changing the Serial Port & Baud Rate**
Modify these variables in `main()`:
```python
port = 'COM4'        # Change to your serial port
baud = 115200        # Adjust as needed
```

### **Adjusting Graph Refresh Rate**
Modify `refresh_delay` in `live_plot()`:
```python
live_plot(out_filename, refresh_delay=0.5)  # Updates every 0.5 seconds
```

### **Changing Y-axis Scaling**
Modify `set_ylim()` inside `live_plot()`:
```python
ax.set_ylim(100, 200))  # Adjust Y-axis range
```

---

## Example Output
Below is an example of how the data appears on the real-time graph:
![Example](https://github.com/user-attachments/assets/c4f82fe1-b7cb-4a40-ad2e-d406ad3f5def)



