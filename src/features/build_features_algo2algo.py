import pandas as pd
from collections import Counter

from collections import Counter

# Original pitches (as an example)
original_pitches = [60, 62, 64, 60, 62, 64, 65, 67, 69, 65, 67, 69] * 16 + [70]

# Analyze the recurring patterns (3-note sequences for simplicity)
patterns = [tuple(original_pitches[i:i+3]) for i in range(len(original_pitches)-2)]
pattern_freq = Counter(patterns)

# Choose the most common pattern for encoding
most_common_pattern, _ = pattern_freq.most_common(1)[0]

# Generate the algorithm encoding
algorithm_encoding = '''
global result
result = []

def common_pattern():
    return list({})

for _ in range(16):
    result += common_pattern()

result.append(70)
'''.format(most_common_pattern)

# Test if the algorithm encoding is correct
exec(algorithm_encoding)

# Validate the result
if result == original_pitches:
    print("The algorithm encoding successfully regenerated the original pitch sequence.")
else:
    print("The algorithm encoding failed to regenerate the original pitch sequence.")

print(result)
# compare result with original_pitches
print(result == original_pitches)
# print the meaningful differences between result and original_pitches
print([i for i, (a, b) in enumerate(zip(result, original_pitches)) if a != b])
# print patterns, pattern_freq, most_common_pattern separated by blank lines and with a header
print("patterns")
print(patterns)
print()
print("pattern_freq")
print(pattern_freq)
print()
print("most_common_pattern")
print(most_common_pattern)







# def find_scale(pitches):
#     intervals = [2, 4, 5, 7, 9, 11, 12]  # Major scale
#     for root in range(12):  # Consider all 12 possible starting pitches
#         scale = [root + interval for interval in intervals]
#         if all(pitch % 12 in scale for pitch in pitches):
#             return root, 'Major'
#     # Add similar logic for other scales like minor, Dorian, etc.
#     return None, None

# def find_recurring_melodies(pitches):
#     melodies = Counter(tuple(pitches[i:i + 4]) for i in range(len(pitches) - 3))
#     common_melodies = [melody for melody, count in melodies.items() if count > 1]
#     return common_melodies

# def encode_pitches(pitches):
#     root, scale_type = find_scale(pitches)
#     common_melodies = find_recurring_melodies(pitches)

#     if root is not None:
#         print(f"Use {scale_type} scale starting at {root}")

#     if common_melodies:
#         for i, melody in enumerate(common_melodies):
#             print(f"Define melody_{i} = {melody}")

#     print("Encoded Sequence:")
#     i = 0
#     while i < len(pitches):
#         if any(tuple(pitches[i:i + 4]) == melody for melody in common_melodies):
#             print(f"Play melody_{common_melodies.index(tuple(pitches[i:i + 4]))}")
#             i += 4
#         else:
#             print(f"Play note {pitches[i]}")
#             i += 1

# # Test the function with a hypothetical sequence of 200 pitches.
# # This sequence includes repetitions of the melody (60, 62, 64, 65) and fits the C Major scale.
# # test_pitches = [60, 62, 64, 65] * 25 + [67, 69, 71, 72] * 25

# df = pd.read_pickle("../../data/interim/note_data_diff_encoded.pkl")
# pitch_data = df["Pitch_diff"].tolist()
# # read the first 200 pitches from pitch_data into a list called test_pitches
# test_pitches = pitch_data[0:200]
# from collections import deque

# def find_patterns(pitch_diffs, min_len=4, max_len=20):
#     n = len(pitch_diffs)
#     patterns = {}
#     for length in range(min_len, max_len+1):
#         for i in range(0, n - length + 1):
#             pattern = tuple(pitch_diffs[i:i + length])
#             if pattern not in patterns:
#                 patterns[pattern] = []
#             patterns[pattern].append(i)
#     return {k: v for k, v in patterns.items() if len(v) > 1}

# def encode_series(pitch_diffs, min_len=4, max_len=20):
#     patterns = find_patterns(pitch_diffs, min_len, max_len)
#     encoded = []
#     i = 0
#     n = len(pitch_diffs)
#     while i < n:
#         found = False
#         for pattern, indices in patterns.items():
#             if i in indices:
#                 encoded.append(f"pattern{len(pattern)}_{pattern}")
#                 i += len(pattern)
#                 found = True
#                 break
#         if not found:
#             encoded.append(f"pitch_{pitch_diffs[i]}")
#             i += 1
#     return ' '.join(encoded)

# # def decode_series(encoded_series):
# #     decoded = deque()
# #     commands = encoded_series.split(' ')
# #     for command in commands:
# #         cmd, value = command.split('_')
# #         if cmd == 'pitch':
# #             decoded.append(float(value))
# #         elif cmd.startswith('pattern'):
# #             length = int(cmd[7:])
# #             pattern = tuple(map(float, value[1:-1].split(', ')))
# #             decoded.extend(pattern)
# #     return list(decoded)

# def decode_series(encoded_series):
#     decoded = deque()
#     commands = encoded_series.split(' ')
#     for command in commands:
#         if '_' in command:
#             cmd, value = command.split('_')
#             if cmd == 'pitch':
#                 decoded.append(float(value))
#             elif cmd.startswith('pattern'):
#                 pattern = tuple(map(float, value[1:-1].split(', ')))
#                 decoded.extend(pattern)
#         else:
#             decoded.append(float(command))
#     return list(decoded)

# # Test the functions with the pitch_diff list provided

# df = pd.read_pickle("../../data/interim/note_data_diff_encoded.pkl")
# pitch_data = df["Pitch_diff"].tolist()
# # read the first 200 pitches from pitch_data into a list called test_pitches
# pitch_diffs = pitch_data[0:200]


# #pitch_diffs = [60.0, 2.0, 2.0, 1.0, -3.0, 2.0, -4.0, 7.0, 5.0, -1.0, 0.0, 1.0, 2.0, -7.0, 2.0, 2.0, 1.0, -3.0, 2.0, -4.0, 7.0, 5.0, -2.0, 0.0, 2.0, -3.0, 5.0, -2.0, -2.0, -1.0, 3.0, -2.0, 4.0, -2.0, -2.0, -1.0, -2.0, -2.0, 4.0, -2.0, 3.0, -1.0, -2.0, -2.0, -1.0, -2.0, 3.0, -1.0, 3.0, -2.0, -1.0, -2.0, -2.0, -1.0, 3.0, -2.0, 4.0, -2.0, -7.0, 10.0, 0.0, 2.0, -3.0, -2.0, -2.0, -1.0, -2.0, 3.0, -1.0, 3.0, -2.0, 4.0, -2.0, 3.0, -1.0, 3.0, -2.0, 4.0, -2.0, -3.0, 3.0, 5.0, -8.0, 0.0, -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 1.0, -3.0, 2.0, -4.0, -1.0, 3.0, 2.0, 1.0, 2.0, -3.0, 1.0, -3.0, 2.0, 3.0, -2.0, -1.0, -2.0, 3.0, -1.0, 3.0, -2.0, 4.0, -2.0, -2.0, -1.0, 3.0, -1.0, 3.0, -2.0, -1.0, 1.0, 2.0, 1.0, -8.0, 2.0, 2.0, 1.0, -8.0, 2.0, 1.0, 2.0, 1.0, 2.0, -10.0, 2.0, 2.0, 1.0, -3.0, 2.0, -4.0, 12.0, -2.0, -2.0, 4.0, -2.0, -2.0, -1.0, 3.0, -2.0, 9.0, -1.0, 3.0, -2.0, -5.0, 1.0, -3.0, -6.0, 9.0, -1.0, -2.0, -2.0, 0.0, -1.0

# # Continue from where the previous code left off

# # Encode the pitch differences to a compressed format
# encoded_series = encode_series(pitch_diffs)
# print(f"Encoded series: {encoded_series}")

# # Decode the compressed format back to the pitch differences
# decoded_series = decode_series(encoded_series)
# print(f"Decoded series: {decoded_series}")

# # Validate if the decoding is accurate
# if pitch_diffs == decoded_series:
#     print("The encoding and decoding were successful!")
# else:
#     print("There was an error in the encoding or decoding process.")
