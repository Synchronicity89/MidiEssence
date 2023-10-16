import pandas as pd
from collections import Counter
from itertools import combinations
import math

class MidiEssence:
    
    # Hard-coded scales and chords
    SCALES = {
        'NormUp': [0, 2, 2, 1, 2, 2, 2, 1],
        'HarmUp': [0, 2, 1, 2, 2, 2, 2, 1]
    }
    CHORDS = {
        'Major': [0, 4, 7],
        'Minor': [0, 3, 7]
    }
    
    def __init__(self, original_pitches):
        self.original_pitches = original_pitches
        self.algorithm_encoding = ""
        
    def detect_scale(self, pitches):
        for scale_name, scale_pitches in self.SCALES.items():
            if set(pitches).issubset(set(scale_pitches)):
                return scale_name
        return None
    
    # def detect_chord(self, pitches):
    #     for chord_name, chord_pitches in self.CHORDS.items():
    #         if set(pitches) == set(chord_pitches):
    #             return chord_name
    #     return None
    
    def find_common_patterns(self, min_length=4, max_length=8):
        common_patterns = []
        for length in range(min_length, max_length + 1):
            patterns = [tuple(self.original_pitches[i:i+length]) for i in range(len(self.original_pitches)-length+1)]
            pattern_freq = Counter(patterns)
            common_patterns.extend(pattern_freq.most_common(3))
        return common_patterns
    
    def algorithm_encoding_generator(self):
        common_patterns = self.find_common_patterns()
        
        # Initialize result
        self.algorithm_encoding += '''
    SCALES = {
        'NormUp': [0, 2, 2, 1, 2, 2, 2, 1],
        'HarmUp': [0, 2, 1, 2, 2, 2, 2, 1]
    }
    
    '''


        self.algorithm_encoding += "global result\n"
        self.algorithm_encoding += "result = []\n\n"
        
        
        self.algorithm_encoding += "result = result[:200]\n"  # truncate to 200 elements

    def dia2(self, Scales, ps, off):
        # make sure that the scale is long enough
        Scales = Scales[:7] * 3
        off = off + 7

        ps_dia = []
        for i in range(len(ps)):
            mult = 1
            if ps[i] < 0: mult = -1
            d = abs(ps[i])
            if d in range(1, 1 + Scales[off + 0]):
                ps_dia.append(mult * 1)
            elif d in range(3, 3 + Scales[off + 1]):
                ps_dia.append(mult * 2)
            elif d in range(5, 5 + Scales[off + 2]):
                ps_dia.append(mult * 3)
            elif d in range(7, 7 + Scales[off + 3]):
                ps_dia.append(mult * 4)
            elif d in range(9, 9 + Scales[off + 4]):
                ps_dia.append(mult * 5)
            elif d in range(11, 11 + Scales[off + 5]):
                ps_dia.append(mult * 6)
            else:
                ps_dia += [0]
        return ps_dia

    def dia(self, Scales, ps, off):
        # return a list of pitches that are in the scale Scales, starting at pitch ps, and offset by off
        # Scales is a list of integers
        # ps is a list of pitch diffs
        # off is an integer
        # create a variation of the pattern ps that is the same shape but where the steps between notes can vary by 1 or 2
        # if ps goes up by 1 or 2 semitones, that's 1
        return self.dia2(Scales, ps, off)



    def example_algo(self, pit):
        # this is an example of an algorithm that can be generated, so it won't assembled into the algorithm_encoding
        # this is just to show how the algorithm_encoding can be used
        S = {
            'NUp': [0, 2, 2, 1, 2, 2, 2, 1],
            'HUp': [0, 2, 1, 2, 2, 2, 2, 1],
            'NDn': [-1 * x for x in reversed(['NUp'])],
            'HDn': [-1 * x for x in reversed(['HUp'])]
        }
        global result
        result = []
        P = {
            'p0' : S['NUp'][1:4] + [-3, 2, -4],
            'p1' : [7,5, -1, 1, 2],
            # 'p2' : [-7]+['p0'][1:],
            # 'p2' : [-7]+['p0']*-1
        }
        # after each change to result, print the result

        result += pit[0:1]
        print(result)
        result += P['p0']
        print(result)
        result += P['p1']
        print(result)
        result += P['p0']*-1
        print(result)
        result += [-7] + P['p0']
        print(result)
        result += P['p1']
        print(result)
        # result += dia2(S['NUp'], P['p1'], 3)
        # print(result)
        # print([-x for x in P['p0']])


        indx = 0
        for p in result:
            if p != pit[indx]:
                # print what's different about pit and result
                print(f"pit[{indx}], {p} != {pit[indx]}")
            indx += 1







# Example usage
# original_pitches = [60, 62, 64, 60, 65, 67, 69, 65, 67, 69] * 20

df = pd.read_pickle("../../data/interim/note_data_diff_encoded.pkl")
pitch_data = df["Pitch_diff"].astype('int').tolist()

# remove bogus data
# remove 10th element from pitch_data, and other cheap trills
pitch_data.pop(10)
pitch_data.pop(22)

# read the first 200 pitches from pitch_data into a list called original_pitches
original_pitches = pitch_data[0:200]
# print the first seven elements of original_pitches, and a header
print("original_pitches")
print(original_pitches[0:7])
# print next 30 elements of original_pitches with header
print("original_pitches continued")
print(original_pitches[7:37])

essence = MidiEssence(original_pitches)
essence.algorithm_encoding_generator()

# # Test the generated encoding
# exec(essence.algorithm_encoding)

# # Validate the result
# if result == original_pitches:
#     print("The algorithm encoding successfully regenerated the original pitch sequence.")
# else:
#     print("The algorithm encoding failed to regenerate the original pitch sequence.")
# print(result)
# compare result with original_pitches
# print(result == original_pitches)
# print the meaningful differences between result and original_pitches
# print([i for i, (a, b) in enumerate(zip(result, original_pitches)) if a != b])

print(essence.algorithm_encoding)

# set a variable to this [0, 2, 2, 1, 2, 2, 2, 1], except reverse the numbers and make them negative
new_scale = [-1 * x for x in reversed(MidiEssence.SCALES['NormUp'])]
print(new_scale)

essence.example_algo(original_pitches)
print("result:")
print(result)
print("pitch_data:")
print(pitch_data[:len(result)+1])

def dia2(Scales, ps, off):
    Scales = Scales * 3  # Repeat the pattern to ensure it is long enough
    off = off % len(Scales)  # Ensure offset is within scale length

    ps_dia = []
    for p in ps:
        mult = 1 if p >= 0 else -1
        d = abs(p)
        
        # Find the closest diatonic step for the chromatic distance
        closest_dia_step = min(Scales, key=lambda x:abs(x-d))
        
        # Add it to the diatonic pattern, considering the sign of the original pattern
        ps_dia.append(closest_dia_step * mult)
    
    return ps_dia

def dia3(Scales, ps, off):
    Scales = (Scales[:7]) * 3  # Repeat the pattern to ensure it is long enough
    off = off % len(Scales)  # Ensure offset is within scale length

    ps_dia = []
    for p in ps:
        mult = 1 if p >= 0 else -1
        d = abs(p)
        
        # Find the closest diatonic step for the chromatic distance
        closest_dia_step = min(Scales, key=lambda x:abs(x-d))
        
        # Add it to the diatonic pattern, considering the sign of the original pattern
        ps_dia.append(closest_dia_step * mult)
    
    return ps_dia

diaex = dia3(MidiEssence.SCALES['NormUp'], result[-5:], 7)
print ("diaex:")
print (diaex)

chromatic_scale_diffs = [2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2]
existing_chromatic_pattern_to_adapt = [7, 5, -1, 1, -3]
actual_pitch_diff_data_needing_a_diatonic_match = [7, 5, -2, 2, -3]

class PitchPatternAdaptor:
    
    def __init__(self):
        self.chromatic_scale_diffs = [2, 2, 1, 2, 2, 2, 1]
    
    def closest_diatonic(self, n, offset=0):
        print("n:", n, "offset:", offset)
        total = 0
        step = 0
        for diff in self.chromatic_scale_diffs * 3:  # repeat the scale 3 times
            if step >= offset:
                if total >= abs(n):
                    return total if n >= 0 else -total
            total += diff
            step += 1
        return None  # Note: this will happen if n is too large/small to match in the scale
    
    def adapt_pattern_to_diatonic(self, pattern, actual_pattern):
        adapted_pattern = []
        for p, a in zip(pattern, actual_pattern):
            if p * a < 0:  # signs don't match, adapting p to match the sign of a
                p = -p
            adapted_pattern.append(self.closest_diatonic(a, offset=(p-a) + 7))
        return adapted_pattern

# Initialize class
adaptor = PitchPatternAdaptor()

# Example usage
existing_chromatic_pattern_to_adapt = [7, 5, -1, 1, -3]
actual_pitch_diff_data_needing_a_diatonic_match = [7, 5, -2, 2, -3]

adapted_pattern = adaptor.adapt_pattern_to_diatonic(existing_chromatic_pattern_to_adapt,
                                                    actual_pitch_diff_data_needing_a_diatonic_match)

print("Adapted pattern:", adapted_pattern)

chromatic_scale_diffs = [2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2]
actual_pitch_diff_data_needing_a_diatonic_match = [7, 5, -2, 2, -3]
# find all the subsequences of chromatic_scale_diffs that matches each element of actual_pitch_diff_data_needing_a_diatonic_match
# for each subsequence, find the index of the first element of the subsequence in chromatic_scale_diffs. Just start with one for now
def get_matches(chromatic_scale_diffs, target):
    target_orig = target
    target = abs(target)
    matches = []
    i = 0
    j = i
    for i in range(len(chromatic_scale_diffs)):
        match = False
        if target_orig >= 0:
            for j in range(i+1, len(chromatic_scale_diffs)):
                if sum(chromatic_scale_diffs[i:j]) == target:
                    match = True
                    break
        else:
            if i > 0:
                for j in range(i-1, -1, -1):
                    if sum(chromatic_scale_diffs[j:i-1]) == target:
                        match = True
                        break
        if match:
            matches += [(i, j, target_orig)]
            i = j
    print("match:", match)
    print("i:", i, "j:", j) 
    print("matches:", matches)
    return matches

# Now, find the first match that matches the whole pattern
# for each match, check if the whole pattern matches
# if it does, then that's the index we want
# if it doesn't, then keep looking
# if we get to the end of the list of matches, then the pattern doesn't fit anywhere
index = 0
match_dict = {}
for pitch_diff in actual_pitch_diff_data_needing_a_diatonic_match:
    matches = get_matches(chromatic_scale_diffs, pitch_diff)
    match_dict[index] = matches
    print("matches:", matches)
    for start_idx, end_idx, target in matches:
        total = sum(chromatic_scale_diffs[start_idx:end_idx])
        if total == pitch_diff:
            print(f"The pattern can legally start from index {start_idx} in chromatic_scale_diffs.")
            break
    else:
        print("The pattern doesn't fit anywhere.")
    index += 1
# Create
print("match_dict:", match_dict)
# print match_dicts with headers for each key, or just a pretty way of printing dictionaries
for key, value in match_dict.items():
    print(key, ":", value)

# convert the match_dict to a list of tuples
match_list = []
for key, value in match_dict.items():
    for ij_coord in value:
        print('ij_coord', ij_coord, type(ij_coord))
        # append match_list with a new tuple that starts with the key, and then ij_coord[0] and ij_coord[1]
        # , actual_pitch_diff_data_needing_a_diatonic_match[key]
        match_list.append((key, ij_coord[0], ij_coord[1], ij_coord[2]))
print("match_list:", match_list)


# sort match_list, the list of tuples, by the first element of the value, followed by the sum of the next two elements

sorted_match_list = sorted(match_list, key=lambda x: (x[0], x[1]))
print('sorted_match_list:', sorted_match_list)

list_of_lists = []
last_added = None
# for every tuple in sorted_match_list append to list_of_lists a list of the remaining tuples that satisfy this constraint:
# Each successive tuple added must have both:, a 1st element that is one bigger than the first element of the previous tuple that was added, and
# the 2nd element must be one bigger than the third element of the previous tuple added
# if there are no more tuples that satisfy this constraint, then stop adding tuples to the list_of_lists
for i in range(len(sorted_match_list)):
    if sorted_match_list[i][0] > 0:
        break
    more = [sorted_match_list[i]]
    last_added = sorted_match_list[i]
    list_of_lists.append(more)
    for j in range(len(sorted_match_list)):
        # first_match = sorted_match_list[j][0] == sorted_match_list[i][0] + 1
        # second_match = sorted_match_list[j][1] == sorted_match_list[i][2] + 1
        first_match = sorted_match_list[j][0] == (last_added[0] + 1)
        target_p = 1 if sorted_match_list[j][3] > 0 else -1

        second_match = sorted_match_list[j][1] == (last_added[2] + target_p )
        if first_match:
            if second_match:
                more.append(sorted_match_list[j])
                last_added = sorted_match_list[j]
            # else:
            #     break
    
print("list_of_lists:", list_of_lists)
        
# sort list_of_lists by the length of each list, from longest to shortest
sorted_list_of_lists = sorted(list_of_lists, key=lambda x: len(x), reverse=True)
# keep the longest lists that have the same length, and print only them
longest_lists = []
for i in range(len(sorted_list_of_lists)):
    if len(sorted_list_of_lists[i]) == len(sorted_list_of_lists[0]):
        longest_lists.append(sorted_list_of_lists[i])
    else:
        break
print("longest_lists:", longest_lists)
# for each tuple in each list in longest_lists, use the second and third to select and sum up the elements of chromatic_scale_diffs in that range, even if the range goes backwards
# if the sum is equal to the absolute value of the fourth element of the tuple, then print the whole tuple, followd by " OK"
# if the sum is not equal to the absolute value of the fourth element, then print the whole tuple, followed by " NOT OK"
for i in range(len(longest_lists)):
    for k in range(len(longest_lists[i])):
        start_idx = longest_lists[i][k][1]
        end_idx = longest_lists[i][k][2]
        target = longest_lists[i][k][3]
        total = sum(chromatic_scale_diffs[start_idx:end_idx])
        if abs(total) == abs(target):
            print(longest_lists[i][k], "OK, target:", target, "total:", total)
        else:
            print(longest_lists[i][k], "NOT OK, target:", target, "total:", total)

