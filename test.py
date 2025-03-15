import serial
import time

def measure_throughput(port, baud, duration_seconds):
    ser = serial.Serial(port, baud, timeout=0.1)
    print(f"Measuring throughput on {port} for {duration_seconds} seconds...")

    total_bytes = 0
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration_seconds:
            raw_data = ser.read(1024)
            total_bytes += len(raw_data)
    except KeyboardInterrupt:
        print("Measurement interrupted.")
    finally:
        ser.close()

    elapsed_time = time.time() - start_time
    throughput = total_bytes / elapsed_time  # Bytes per second
    print(f"Total bytes read: {total_bytes}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput / 1024:.2f} KB/s ({(throughput * 8) / 1e6:.2f} Mbps)")

# Run the test
measure_throughput('COM4', 115200, 10)
