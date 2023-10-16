import json
import pandas as pd

def lzw_compress(input_list):
    dictionary = {str(i): i for i in range(256)}
    next_code = 256
    string = ""
    compressed_data = []
    
    for symbol in input_list:
        symbol = str(symbol)
        new_string = string + symbol
        if new_string in dictionary:
            string = new_string
        else:
            compressed_data.append(dictionary[string])
            dictionary[new_string] = next_code
            next_code += 1
            string = symbol
    
    if string:
        compressed_data.append(dictionary[string])
    
    return compressed_data

def lzw_decompress(compressed_data):
    dictionary = {i: str(i) for i in range(256)}
    next_code = 256
    decompressed_data = []
    
    string = dictionary[compressed_data.pop(0)]
    decompressed_data.append(int(string))
    
    for code in compressed_data:
        if code == next_code:
            new_string = string + string[0]
        elif code in dictionary:
            new_string = dictionary[code]
        else:
            raise ValueError("Invalid code encountered during decompression.")
        
        decompressed_data.append(int(new_string))
        
        dictionary[next_code] = string + new_string[0]
        next_code += 1
        string = new_string
    
    return decompressed_data


# Sample Pitch data
pitch_data = [60, 62, 64, 65, 62, 64, 60, 67, 72, 71, 71, 72]  # This should be your full list of pitch data
# read in  the data\interim\sample_pitch_data.csv data and turn it into a list
df = pd.read_csv("../../data/interim/sample_pitch_data.csv")
pitch_data = df["Pitch"].tolist()


# Compress the pitch data using LZW
compressed_data = lzw_compress(pitch_data)

# Convert compressed data to JSON
compressed_json = json.dumps(compressed_data)

# Print compressed JSON data
print("Compressed JSON Data:", compressed_json)

# To decompress
compressed_data_from_json = json.loads(compressed_json)
decompressed_data = lzw_decompress(compressed_data_from_json)

# Print decompressed data
print("Decompressed Data:", decompressed_data)
