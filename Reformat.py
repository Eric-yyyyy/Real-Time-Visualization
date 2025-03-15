import os

def reformat_txt_file(input_file, output_file):
    """
    Reads a text file with hex values and reformats it into 'AAAA AAAA BBBB BBBB' style.
    Writes the formatted data to a new output file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into hex values (space-separated)
            hex_values = line.strip().split()
            
            # Combine into groups of two bytes (e.g., "AAAA")
            formatted_line = []
            for i in range(0, len(hex_values), 4):
                group = ''.join(hex_values[i:i + 2])  # Join 2 hex values into AAAA
                formatted_line.append(group)
            
            # Join groups into 4-character pairs separated by spaces (AAAA AAAA BBBB BBBB)
            outfile.write(' '.join(formatted_line) + '\n')


def reformat_all_files():
    """
    Reformats all files in the 'With_Strength' folder named Strength1.txt to Strength20.txt.
    Saves them as reformatted_Strength1.txt to reformatted_Strength20.txt in the same folder.
    """
    folder = "With_Strength"  # Folder containing the files
    for i in range(1, 21):  # Loop from 1 to 20
        input_file = os.path.join(folder, f'emg_data_Strength{i}.txt')
        output_file = os.path.join(folder, f'reformatted_emg_data_Strength{i}.txt')
        print(f"Processing {input_file} -> {output_file}...")
        reformat_txt_file(input_file, output_file)


if __name__ == '__main__':
    reformat_all_files()
