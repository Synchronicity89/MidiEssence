from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from numpy import int64
from numpy import float64
from collections import defaultdict, Counter

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data and turn it into a list
pitch_data = []
use_diff = True
if not use_diff:
    df = pd.read_csv("../../data/interim/sample_pitch_data.csv")
    pitch_data = df["Pitch"].tolist()
else:
    df = pd.read_pickle("../../data/interim/note_data_diff_encoded.pkl")
    pitch_data = df["Pitch_diff"].tolist()

# Define the quality metric function (MSE)
def mse(original, reconstructed):
    return ((np.array(original) - np.array(reconstructed)) ** 2).mean()

# Define the maximum acceptable MSE
max_mse = 10.0

# Define tweakable variables
threshold = 69.5  # You can adjust this to optimize the compression ratio

# Perform FFT
fft_coefficients = np.fft.fft(pitch_data)

# Zero out coefficients with magnitude smaller than the threshold
compressed_coefficients = np.where(np.abs(fft_coefficients) > threshold, fft_coefficients, 0)

# Perform IFFT to recover the original data
reconstructed_data = np.fft.ifft(compressed_coefficients).real.round().astype(int)

# Calculate the quality metric (MSE)
current_mse = mse(pitch_data, reconstructed_data)
print(f"Current MSE: {current_mse}")

print("Number of incorrect values:", np.count_nonzero(pitch_data - reconstructed_data))

print("number of non-zero coefficients:", np.count_nonzero(compressed_coefficients))

print("Compression Ratio:", len(pitch_data) / len(compressed_coefficients))

# Define the optimal variables before the loop
optimal_threshold = threshold
optimal_coefficients = copy.deepcopy(compressed_coefficients)
optimal_reconstructed_data = copy.deepcopy(reconstructed_data)

# Loop through different thresholds to find the optimal value
while True:
    # Increase the threshold
    threshold += 0.1
    
    # Zero out coefficients with magnitude smaller than the threshold
    compressed_coefficients = np.where(np.abs(fft_coefficients) > threshold, fft_coefficients, 0)

    # Perform IFFT to recover the original data
    reconstructed_data = np.fft.ifft(compressed_coefficients).real.round().astype(int)
    # if the reconstructed data is not the same as the original data, print an error message including the threshold, and ask if you want to continue

    # Calculate the quality metric (MSE)
    current_mse = mse(pitch_data, reconstructed_data)

    # Save the variables in this loop as the optimal variables
    if current_mse <= max_mse:
        optimal_threshold = threshold
        optimal_coefficients = copy.deepcopy(compressed_coefficients)
        optimal_reconstructed_data = copy.deepcopy(reconstructed_data)
    else:
        break
    if not np.array_equal(pitch_data, reconstructed_data.tolist()):
        print(f"Error: Reconstructed data does not match original data for threshold: {threshold}")
        break

# Print the optimal threshold and number of non-zero coefficients
num_nonzero_coefficients = np.count_nonzero(optimal_coefficients)
print(f"Optimal threshold: {optimal_threshold}, number of nonzero coefficients: {num_nonzero_coefficients}")

# If the quality metric falls below the threshold, we have our compressed representation
if current_mse <= max_mse and np.array_equal(pitch_data, optimal_reconstructed_data.tolist()):
    print("Compression successful.")
    # Print the compression ratio, also check for any loss of data
    print("Compression Ratio:", len(pitch_data) / len(optimal_coefficients))
    # check for any loss of data, by comparing the original data with the reconstructed data
    print("Loss of Data:", np.array_equal(pitch_data, optimal_reconstructed_data.tolist()))
    # convert the lists to numpy arrays before subtracting them
    pitch_data_arr = np.array(pitch_data)
    reconstructed_data_arr = np.array(optimal_reconstructed_data)
    print("Difference:", np.trim_zeros(pitch_data_arr - reconstructed_data_arr))
    # print the number of incorrect values in the reconstructed data
    print("Number of incorrect values:", np.count_nonzero(pitch_data_arr - reconstructed_data_arr))
    # print the average difference between the original data and the reconstructed data
    print("Average difference:", np.mean(pitch_data_arr - reconstructed_data_arr))
else:
    print("Compression failed.")

# print all the same things as above regarding the fft compression
print("Original Pitch Data:", pitch_data)
print("FFT Coefficients:", fft_coefficients)
# print("Optimal FFT Coefficients:", optimal_coefficients)
# print("Reconstructed Pitch Data:", optimal_reconstructed_data.tolist())

# print the compression ratio, also check for any loss of data
print("Compression Ratio:", len(pitch_data) / len(fft_coefficients))
# check for any loss of data, by comparing the original data with the reconstructed data
print("Loss of Data:", np.array_equal(pitch_data, reconstructed_data.tolist()))
# convert the lists to numpy arrays before subtracting them
pitch_data_arr = np.array(pitch_data)
reconstructed_data_arr = np.array(reconstructed_data)
print("Difference:", np.trim_zeros(pitch_data_arr - reconstructed_data_arr))
# print the number of incorrect values in the reconstructed data
print("Number of incorrect values:", np.count_nonzero(pitch_data_arr - reconstructed_data_arr))
# print the average difference between the original data and the reconstructed data
print("Average difference:", np.mean(pitch_data_arr - reconstructed_data_arr))

# Plot the original and reconstructed data
fig, ax = plt.subplots()
ax.plot(pitch_data, label="Original")
ax.plot(reconstructed_data, label="Reconstructed")
ax.set_xlabel("Time")
ax.set_ylabel("Pitch")
ax.legend()
plt.show()


# create an player to play the original data and one to play the regenerated data
from music21 import instrument, note, stream, chord
import numpy as np

def play_music(data):
    """ 
    Play the music using the music21 library.
    """
    # Convert the data to a music21 stream.Score object
    offset = 0
    output_notes = []
    for pattern in data:
        # If the pattern is a chord, then create a chord object
        if ('.' in str(pattern)) or str(pattern).isdigit():
            notes_in_chord = str(pattern).split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If the pattern is a rest, then create a rest object
        elif ('rest' in str(pattern)):
            new_rest = note.Rest(str(pattern))
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            output_notes.append(new_rest)
        # If the pattern is a note, then create a note object
        else:
            new_note = note.Note(str(pattern))
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # Increase the offset each time
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='../../data/processed/generated_music.mid')
    midi_stream.show('midi')

# play the original data
play_music(map(str, pitch_data))

# play the regenerated data
play_music(map(str, reconstructed_data))


















def find_patterns_orig(lst):
    patterns = defaultdict(list)
    n = len(lst)
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            pattern = tuple(lst[i:i + length])
            patterns[pattern].append(i)

    return {k: v for k, v in patterns.items() if len(v) > 1}

def find_patterns(lst, no_overlapping_instances_of_same_pattern=True, favor_longer_patterns=True):
    all_patterns = find_patterns_orig(lst)
    filtered_patterns = {}
    
    if no_overlapping_instances_of_same_pattern:
        for pattern, indices in all_patterns.items():
            non_overlapping_indices = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] - non_overlapping_indices[-1] >= len(pattern):
                    non_overlapping_indices.append(indices[i])
            if len(non_overlapping_indices) > 1:
                filtered_patterns[pattern] = non_overlapping_indices
                
    if favor_longer_patterns:
        final_patterns = {}
        for p1, ind1 in filtered_patterns.items():
            remove_p1 = False
            for p2, ind2 in filtered_patterns.items():
                if set(ind1).issubset(set(ind2)) and all(p1 in p2[i:i+len(p1)] for i in range(len(p2) - len(p1) + 1)):
                    remove_p1 = True
                    break
            if not remove_p1:
                final_patterns[p1] = ind1
                
    for p, ind in final_patterns.items():
        print(f"Pattern: {p}, Count: {len(ind)}, Indices: {ind}")

# Test the function
#find_patterns([1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 4, 4, 4], no_overlapping_instances_of_same_pattern=True, favor_longer_patterns=True)


inputList = [1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 5, 6]
# counter, indices, patterns = find_patterns(inputList)

print (inputList, )

# regenerate the original list from the counter, indices and patterns, and keep track of which patterns were used and how much of the original list was regenerated by each pattern
def regenerate_list(inputList, counter, indices, patterns):
    # make a copy of the input list
    inputList = inputList.copy()
    # create an empty list to store the regenerated list
    regenerated_list = []
    # create an empty list to store the patterns used to regenerate the list
    patterns_used = []
    # create an empty list to store the number of elements in the original list that were regenerated by each pattern
    pattern_counts = []
    # create an empty list to store the number of elements in the regenerated list that were regenerated by each pattern
    regenerated_counts = []
    # create an empty list to store the number of elements in the original list that were regenerated by each pattern
    percent_regenerated = []
    
    # iterate over the patterns from longest to shortest
    for length in sorted(patterns.keys(), reverse=True):
        # iterate over the patterns of the current length
        for pattern in patterns[length]:
            # if the pattern was used to regenerate the list, then add it to the patterns_used list
            if pattern in regenerated_list:
                patterns_used.append(pattern)
                # get the indices of the pattern in the original list
                pattern_indices = indices[pattern]
                # get the number of elements in the original list that were regenerated by the pattern
                pattern_count = len(pattern_indices)
                # append the pattern_count to the pattern_counts list
                pattern_counts.append(pattern_count)
                # get the number of elements in the regenerated list that were regenerated by the pattern
                regenerated_count = counter[pattern]
                # append the regenerated_count to the regenerated_counts list
                regenerated_counts.append(regenerated_count)
                # get the number of elements in the original list that were regenerated by the pattern
                percent_regenerated.append(pattern_count / len(inputList))
                # remove the pattern from the original list
                for i in pattern_indices:
                    try:
                        inputList.pop(i)
                    except Exception as e:
                        print(f"Error occurred while removing pattern from inputList: {e}")
                # remove the pattern from the regenerated list
                if pattern in regenerated_list:
                    for i in range(regenerated_count):
                        try:
                            regenerated_list.pop(regenerated_list.index(pattern))
                        except Exception as e:
                            print(f"Error occurred while removing pattern from regenerated_list: {e}")
                # remove the pattern from the indices dictionary
                try:
                    del indices[pattern]
                except Exception as e:
                    print(f"Error occurred while removing pattern from indices dictionary: {e}")
                # remove the pattern from the counter dictionary
                try:
                    del counter[pattern]
                except Exception as e:
                    print(f"Error occurred while removing pattern from counter dictionary: {e}")
                # remove the pattern from the patterns dictionary
                try:
                    patterns[length].pop(patterns[length].index(pattern))
                except Exception as e:
                    print(f"Error occurred while removing pattern from patterns dictionary: {e}")
                # break out of the loop
                break
            # if the pattern was not used to regenerate the list, then append it to the regenerated_list list
            else:
                regenerated_list.append(pattern)
                
    # regenerate the original list from the counter, indices and patterns
    regenerated_inputList = []
    for pattern in regenerated_list:
        regenerated_inputList.extend([pattern] * counter[pattern])
    
    return regenerated_inputList, patterns_used, pattern_counts, regenerated_counts, percent_regenerated

# regenerated_inputList, patterns_used, pattern_counts, regenerated_counts, percent_regenerated = regenerate_list(len(inputList), counter, indices, patterns)

# print (regenerated_inputList, patterns_used, pattern_counts, regenerated_counts, percent_regenerated)

# Function to find patterns in the input list
def find_patterns(inputList, min_len=2, max_len=5):
    counter = Counter()
    indices = defaultdict(list)
    
    for length in range(min_len, max_len + 1):
        for i in range(0, len(inputList) - length + 1):
            sub_seq = tuple(inputList[i:i+length])
            counter[sub_seq] += 1
            indices[sub_seq].append(i)
            
    return counter, indices

# Measure the size of the encoding based on the number of elements
def measure_size(indices, counter):
    # return the sum of all the lengths of the patterns in the indices dictionary
    return sum([len(pattern) for pattern in indices.keys()])  

# # Function to make small, random changes to an encoding
# def make_small_change(indices, counter):
#     # Placeholder: Implement your logic to make a small, random change to the encoding
#     # This could involve replacing a less frequent pattern with a more frequent one,
#     # merging adjacent patterns, splitting a pattern into smaller pieces, etc.
#     # The function should return indices_copy and counter_copy, which are copies of the indices and counter dictionaries
#     # that represent the new encoding, but have some small changes compared to the original ones
#     indices_copy = copy.deepcopy(indices)
#     counter_copy = copy.deepcopy(counter)
#     # get a list of the patterns in the encoding
#     patterns = indices.keys()
#     # get the number of patterns in the encoding
#     num_patterns = len(patterns)
#     # sort the patterns by frequency * the length of the pattern in ascending order
#     patterns_sorted = sorted(patterns, key=lambda x: counter[x] * len(x))
#     # drop the first pattern from the sorted patterns
#     popped = patterns_sorted.pop(0)
#     # make sure the pattern that was dropped is no longer in the indices_copy dictionary and the counter_copy dictionary
#     try:
#         del indices_copy[popped]
#     except Exception as e:
#         print(f"Error occurred while removing pattern from indices_copy dictionary: {e}")
#     try:
#         del counter_copy[popped]
#     except Exception as e:
#         print(f"Error occurred while removing pattern from counter_copy dictionary: {e}")
#     # remove any redundant patterns from the encoding.  Remove shorter patterns which are fully eclipsed by larger patterns
#     for pattern in patterns_sorted:
#         for i in indices_copy[pattern]:
#             if all([j in indices_copy[other_pattern] for j in range(i, i+len(pattern)) for other_pattern in patterns_sorted if pattern != other_pattern]):
#                 try:
#                     indices_copy[pattern].remove(i)
#                 except Exception as e:
#                     print(f"Error occurred while removing index from indices_copy dictionary: {e}")
#                 try:
#                     counter_copy[pattern] -= 1
#                 except Exception as e:
#                     print(f"Error occurred while decrementing counter_copy dictionary: {e}")
#     return indices_copy, counter_copy

from collections import defaultdict
import copy

def make_small_change(indices, counter):
    indices_copy = copy.deepcopy(indices)
    counter_copy = copy.deepcopy(counter)
    
    patterns = list(indices.keys())
    num_patterns = len(patterns)
    
    patterns_sorted = sorted(patterns, key=lambda x: counter[x] * len(x))
    
    # Remove least frequent pattern
    popped = patterns_sorted.pop(0)
    del indices_copy[popped]
    del counter_copy[popped]

    # Check for redundant patterns
    for pattern in patterns_sorted:
        indices_to_remove = []
        for i in indices_copy[pattern]:
            for other_pattern in patterns_sorted:
                if pattern == other_pattern:
                    continue
                is_eclipsed = all(j in indices_copy[other_pattern] for j in range(i, i + len(pattern)))
                if is_eclipsed:
                    indices_to_remove.append(i)
                    break

        for i in indices_to_remove:
            indices_copy[pattern].remove(i)
            counter_copy[pattern] -= 1
            if counter_copy[pattern] == 0:
                del indices_copy[pattern]
                del counter_copy[pattern]
                
    return indices_copy, counter_copy

    
# # Function to shrink the encoding
# def shrink_encoding(indices, counter, max_iter=1000, learning_rate=0.1):
#     indices_best = indices
#     counter_best = counter
#     best_size = measure_size(indices, counter)
    
#     for i in range(max_iter):
#         # Generate a candidate encoding by making a small change to the best one found so far
#         indices_changed, counter_changed = make_small_change(copy.deepcopy(indices_best), copy.deepcopy(counter_best))
        
#         # Measure the size of the candidate encoding
#         candidate_size = measure_size(indices_changed, counter_changed)
        
#         # If the candidate encoding is smaller, update the best encoding
#         if candidate_size < best_size:
#             indices_best = indices_changed
#             counter_best = counter_changed
#             best_size = candidate_size
#             print(f"New best encoding found in iteration {i}, size: {best_size}")
#         else:
#             # Stochastic update (akin to simulated annealing)
#             if random.random() < learning_rate:
#                 indices_best = indices_changed
#                 counter_best = counter_changed
#                 best_size = candidate_size
#                 print(f"New best encoding found in iteration {i}, size: {best_size}")

#     return indices_best, counter_best

# Function to decode an encoding back into the original list
def decode(indices, length):
    # Placeholder: Implement your decoding logic here
    # Initialize an empty list to store the output
    output = [None] * length
    
    for pattern, locations in indices.items():
        for loc in locations:
            for i, val in enumerate(pattern):
                output[loc + i] = val
                
    # Check for None in the output (indicating unsuccessful decoding)
    if any(val is None for val in output):
        return None
    
    return output


# Modified shrink_encoding function
def shrink_encoding(indices, counter, inputList, max_iter=1000, learning_rate=0.1):
    indices_best = indices
    counter_best = counter
    best_size = measure_size(indices, counter)
    
    for i in range(max_iter):
        indices_changed, counter_changed = make_small_change(copy.deepcopy(indices_best), copy.deepcopy(counter_best))
        candidate_size = measure_size(indices_changed, counter_changed)
        
        # Try to decode the candidate encoding into the original list
        decoded_list = decode(indices_changed, len(inputList))
        
        # If successful and the size is smaller, update the best encoding
        if decoded_list == inputList and (candidate_size < best_size):
            indices_best = indices_changed
            counter_best = counter_changed
            best_size = candidate_size
            print(f"New best encoding found in iteration {i}, size: {best_size}")
        else:
            break
            # if random.random() < learning_rate:
            #     indices_best = indices_changed
            #     counter_best = counter_changed
            #     best_size = candidate_size
                
    return indices_best, counter_best

# Your existing code to define the sample input list, find initial patterns, and optimize the encoding

# Finding patterns in the input list
counter, indices = find_patterns(inputList)

# Initial encoding is just a list of individual numbers (placeholder)
initial_encoding = inputList

# Optimize the encoding
indices_final, counter_final = shrink_encoding(indices, counter, inputList)

print("Final encoding:", indices_final, counter_final)
print("Final encoding size:", measure_size(indices_final, counter_final))
print("Decoded list:", decode(indices_final, len(inputList)))


# Unpickle the diff dataframe stored by build_features.py and create an encoding for each Filename, i.e. using all the instruments in that filename that compresses the data to the smallest encoding possible without cheating
def encode_squish(df):
    df_copy = df.copy()
    # create an empty dataframe called df_encoded with the same columns as the input dataframe
    df_encoded_squish = pd.DataFrame(columns=df.columns)

    # Get a list of unique Filenames
    Filenames = df['Filename'].unique()

    for Filename in Filenames:
        # Get the data for the current Filename
        Filename_data = df[df['Filename'] == Filename]
        Filename_data.reset_index(drop=True, inplace=True)

        # Get a list of unique Instruments for the current Filename
        Instruments = Filename_data['Instrument'].unique()

        # create an empty numpy array called file_patterns to store the patterns found in the current Filename it will be appended by the patterns found in each Instrument with (5053,) shape
        file_PitchDiff_patterns = np.empty((0, 5053), int64)
        file_StartDiff_patterns = np.empty((0, 5053), float64)
        file_EndDiff_patterns = np.empty((0, 5053), float64)
        file_VelocityDiff_patterns = np.empty((0, 5053), int64)

        # first find the patterns as expressed by specific instruments
        for Instrument in Instruments:
            Instrument_data = Filename_data[Filename_data['Instrument'] == Instrument]
            Instrument_data.reset_index(drop=True, inplace=True)

            # given that the columns of Instrument_data are Filename, Instrument,  Pitch_diff,  Start_diff,  End_diff,  Velocity_diff find the patterns in all 4 diffs columns
            # find the patterns in the Pitch_diff column
            
            # find the patterns in the Pitch_diff column
            # Pitch_diff_patterns = np.diff(df['Pitch_diff'].values)
            Pitch_diff_patterns = df['Pitch_diff'].values
            Start_diff_patterns = df['Start_diff'].values
            End_diff_patterns = df['End_diff'].values
            Velocity_diff_patterns = df['Velocity_diff'].values

            # append the patterns found in the current Instrument to the 4 file patterns arrays
            file_PitchDiff_patterns = np.append(file_PitchDiff_patterns, Pitch_diff_patterns)
            file_StartDiff_patterns = np.append(file_StartDiff_patterns, Start_diff_patterns)
            file_EndDiff_patterns = np.append(file_EndDiff_patterns, End_diff_patterns)
            file_VelocityDiff_patterns = np.append(file_VelocityDiff_patterns, Velocity_diff_patterns)

        unique_file_PitchDiff_patterns, counts = np.unique(file_PitchDiff_patterns, return_counts=True)
        unique_file_StartDiff_patterns, counts = np.unique(file_StartDiff_patterns, return_counts=True)
        unique_file_EndDiff_patterns, counts = np.unique(file_EndDiff_patterns, return_counts=True)
        unique_file_VelocityDiff_patterns, counts = np.unique(file_VelocityDiff_patterns, return_counts=True)

        # print a little info about the patterns found
        # print("Filename: " + str(Filename))
        # print("Number of unique Pitch_diff patterns: " + str(len(unique_file_PitchDiff_patterns)))
        # print("Number of unique Start_diff patterns: " + str(len(unique_file_StartDiff_patterns)))
        # print("Number of unique End_diff patterns: " + str(len(unique_file_EndDiff_patterns)))
        # print("Number of unique Velocity_diff patterns: " + str(len(unique_file_VelocityDiff_patterns)))

        # # print 5 of the most repeated Pitch_diff patterns
        # print("5 most repeated Pitch_diff patterns: " + str(unique_file_PitchDiff_patterns[np.argsort(counts)[-5:]]))
        # print("5 most repeated Start_diff patterns: " + str(unique_file_StartDiff_patterns[np.argsort(counts)[-5:]]))
        # print("5 most repeated End_diff patterns: " + str(unique_file_EndDiff_patterns[np.argsort(counts)[-5:]]))
        # print("5 most repeated Velocity_diff patterns: " + str(unique_file_VelocityDiff_patterns[np.argsort(counts)[-5:]]))

        # # print the counts of the 5 most repeated Pitch_diff patterns
        # print("Counts of the 5 most repeated Pitch_diff patterns: " + str(counts[np.argsort(counts)[-5:]]))
        # print("Counts of the 5 most repeated Start_diff patterns: " + str(counts[np.argsort(counts)[-5:]]))
        # print("Counts of the 5 most repeated End_diff patterns: " + str(counts[np.argsort(counts)[-5:]]))
        # print("Counts of the 5 most repeated Velocity_diff patterns: " + str(counts[np.argsort(counts)[-5:]]))

        # print a little info about the patterns found
        print("Filename: " + str(Filename))
        print("Number of unique Pitch_diff patterns: " + str(len(unique_file_PitchDiff_patterns)))

        # convert the unique_file_PitchDiff_patterns array to int64
        unique_file_PitchDiff_patterns_int = unique_file_PitchDiff_patterns.astype(int)

        # define the target sequence
        target_sequence = [2, 2, 1, -3, 2, -4]

        # target_indices = [i for i, pattern in enumerate(unique_file_PitchDiff_patterns_int) if all(x == y for x, y in zip(pattern, target_sequence))]
        target_indices = find_indices(unique_file_PitchDiff_patterns_int, target_sequence)

        if len(target_indices) > 0:
            # print the pitch diff patterns that contain the target sequence
            print("Pitch_diff patterns that contain the sequence [2, 2, 1, -3, 2, -4, 7]:")
            for i in target_indices:
                print(unique_file_PitchDiff_patterns[i])
        else:
            print("No Pitch_diff patterns contain the sequence [2, 2, 1, -3, 2, -4, 7]")

        # print 5 of the most repeated Pitch_diff patterns
        print("5 most repeated Pitch_diff patterns: " + str(unique_file_PitchDiff_patterns[np.argsort(counts)[-5:]]))

        # print the counts of the 5 most repeated Pitch_diff patterns
        print("Counts of the 5 most repeated Pitch_diff patterns: " + str(counts[np.argsort(counts)[-5:]]))

        # create a dictionary to store the patterns and their corresponding values
        PitchDiff_patterns_dict = {}
        StartDiff_patterns_dict = {}
        EndDiff_patterns_dict = {}
        VelocityDiff_patterns_dict = {}

        # fill the dictionaries with the patterns and their corresponding values
        for i in range(len(unique_file_PitchDiff_patterns)):
            PitchDiff_patterns_dict[unique_file_PitchDiff_patterns[i]] = i
        for i in range(len(unique_file_StartDiff_patterns)):
            StartDiff_patterns_dict[unique_file_StartDiff_patterns[i]] = i
        for i in range(len(unique_file_EndDiff_patterns)):
            EndDiff_patterns_dict[unique_file_EndDiff_patterns[i]] = i
        for i in range(len(unique_file_VelocityDiff_patterns)):
            VelocityDiff_patterns_dict[unique_file_VelocityDiff_patterns[i]] = i
        
        # create a new column for each pattern type
        Filename_data['PitchDiff_pattern'] = 0
        Filename_data['StartDiff_pattern'] = 0
        Filename_data['EndDiff_pattern'] = 0
        Filename_data['VelocityDiff_pattern'] = 0

        # fill the new columns with the corresponding pattern values
        for index, row in Filename_data.iterrows():
            Filename_data.loc[index, 'PitchDiff_pattern'] = PitchDiff_patterns_dict[row['Pitch_diff']]
            Filename_data.loc[index, 'StartDiff_pattern'] = StartDiff_patterns_dict[row['Start_diff']]
            Filename_data.loc[index, 'EndDiff_pattern'] = EndDiff_patterns_dict[row['End_diff']]
            Filename_data.loc[index, 'VelocityDiff_pattern'] = VelocityDiff_patterns_dict[row['Velocity_diff']]

        # Append the Instrument_data rows to the df_encoded_squish dataframe
        df_encoded_squish = pd.concat([df_encoded_squish, Filename_data], ignore_index=True)

        # add anything else requied to the encoding that might be required to decode it later into the original dataframe

    # Reorder the columns to match the original dataframe
    df_encoded_squish = df_encoded_squish[['Filename', 'Instrument', 'PitchDiff_pattern', 'StartDiff_pattern', 'EndDiff_pattern', 'VelocityDiff_pattern']]

    return df_encoded_squish


def decode_squish(df_encoded):
    # Create a copy of the input dataframe
    df_decoded = pd.DataFrame(columns=[col.replace('_diff', '') for col in df_encoded.columns])

    # Get a list of unique Filenames
    Filenames = df_encoded['Filename'].unique()

    for Filename in Filenames:
        # Get the data for the current Filename
        Filename_data = df_encoded[df_encoded['Filename'] == Filename]
        Filename_data.reset_index(drop=True, inplace=True)

        # Get a list of unique Instruments for the current Filename
        Instruments = Filename_data['Instrument'].unique()

        for Instrument in Instruments:
            Instrument_data = Filename_data[Filename_data['Instrument'] == Instrument]
            Instrument_data.reset_index(drop=True, inplace=True)

            # Get the offset values for each column
            offset = Instrument_data.iloc[0:1].copy()

            # # drop the first row of Instrument_data
            # Instrument_data.loc[0, ["Pitch_diff", "Start_diff", "End_diff", "Velocity_diff"]] = 0

            # # Add the offset values to the diffs
            # Instrument_data.loc[:,"Pitch"] = Instrument_data["Pitch_diff"].cumsum() + offset.loc[0, "Pitch_diff"]
            # Instrument_data.loc[:,"Start"] = Instrument_data["Start_diff"].cumsum() + offset.loc[0, "Start_diff"]
            # Instrument_data.loc[:,"End"] = Instrument_data["End_diff"].cumsum() + offset.loc[0, "End_diff"]
            # Instrument_data.loc[:,"Velocity"] = Instrument_data["Velocity_diff"].cumsum() + offset.loc[0, "Velocity_diff"]

            # Append the Instrument_data rows to the df_decoded dataframe
            df_decoded = pd.concat([df_decoded, Instrument_data], ignore_index=True)

    # Reorder the columns to match the original dataframe
    df_decoded = df_decoded[['Filename', 'Instrument', 'Pitch', 'Start', 'End', 'Velocity']]

    # Convert the Pitch and Velocity columns to integers
    df_decoded["Pitch"] = df_decoded["Pitch"].astype(int64)
    df_decoded["Velocity"] = df_decoded["Velocity"].astype(int64)

    # Convert the Filename and Instrument columns to integers
    df_decoded["Filename"] = df_decoded["Filename"].astype(int64)
    df_decoded["Instrument"] = df_decoded["Instrument"].astype(int64)

    
    return df_decoded

# Unpickle the diff dataframe stored by build_features.py and create an encoding for each Filename, i.e. using all the instruments in that filename that compresses the data to the smallest encoding possible without cheating
df_diffs = pd.read_pickle("../../data/interim/note_data_diff_encoded.pkl")

# encode the diffs dataframe using the squish method
df_squish_encoded = encode_squish(df_diffs)
df_squish_encoded

# pickle the squish encoded dataframe to ../../data/interim/note_data_diff_squish_encoded.pkl
df_squish_encoded.to_pickle("../../data/interim/note_data_diff_squish_encoded.pkl")

