import serial
import time

import serial
import time
import numpy as np
def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def read_for_duration(port, baud, duration_seconds, out_filename):

    ser = serial.Serial(port, baud, timeout=None)
    print(f"Reading for {duration_seconds} seconds from {port} at {baud} baud")

    start_time = time.time()
    header_seq = b"\xAA\xAA\x5F\x00"
    found_header = False
    buffer = b""

    with open(out_filename, 'w', encoding='utf-8') as file:
        try:
            while (time.time() - start_time) < duration_seconds:
                # Read from the serial port
                raw_data = ser.read(1024)
                if raw_data:
                    buffer += raw_data

                    if not found_header:
                        idx = buffer.find(header_seq)
                        if idx != -1:
                            # Discard everything up to the header
                            buffer = buffer[idx:]
                            found_header = True

                    
                    if found_header:
                        while len(buffer) >= 64:
                            packet = buffer[:64]
                            buffer = buffer[64:]

                            # Convert packet bytes to a space-separated hex string
                            hex_str = ' '.join(f'{b:02X}' for b in packet)
                            file.write(hex_str + '\n')

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting early.")
        finally:
            ser.close()

    print(f"Data capture complete. Saved to {out_filename}")




def read_for_count(port, baud, chunk_count, out_filename):

    ser = serial.Serial(port, baud, timeout=0.1)
    print(f"Reading {chunk_count} chunks from {port} at {baud} baud")

    file = open(out_filename, 'w', encoding='utf-8')
    
    chunks = 0
    try:
        while chunks < chunk_count:
            raw_data = ser.read(1024)
            if raw_data:
                hex_str = ' '.join(f'{byte:02X}' for byte in raw_data)
                file.write(hex_str + '\n')
                file.flush()  
                chunks += 1
            
    except KeyboardInterrupt:
        print("Exit")
    
    finally:
        ser.close()
        file.close()
    
    print(f"Captured data at {out_filename}")

import os
"""
def reformat_txt_file(input_file, output_file):
   
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            hex_values = line.strip().split()
            
            # Combine into groups of two bytes (e.g., "AAAA")
            formatted_line = []
            for i in range(0, len(hex_values), 4):
                group = ''.join(hex_values[i:i + 2])  # Join 2 hex values into AAAA
                formatted_line.append(group)
            
            # J (AAAA AAAA BBBB BBBB)
            outfile.write(' '.join(formatted_line) + '\n')
"""
def clean_and_align_file(input_file, output_file):

    header = "AA AA 5F 00"
    header_found = False
    buffer = []

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            hex_values = line.strip().split()

            if not header_found:
                if header in ' '.join(hex_values): 
                    header_index = hex_values.index("AA")  
                    buffer += hex_values[header_index:]  
                    header_found = True
            else:

                buffer += hex_values

            while len(buffer) >= 64:
                packet = buffer[:64] 
                buffer = buffer[64:]  
                outfile.write(' '.join(packet) + '\n')

        if len(buffer) > 0:
            outfile.write(' '.join(buffer) + '\n')

    print(f"Cleaned and aligned data saved to {output_file}")


def main():

    read_for_duration('COM4', 115200, 10, 'StrengthZou4.txt')
    #clean_and_align_file('emg_data_rest21.txt', 'cleanedmg_data_rest21.txt')

main()
