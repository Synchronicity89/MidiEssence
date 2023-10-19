import pandas as pd
from collections import Counter
from itertools import combinations
import itertools


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
    
    def __init__(self):
        self.algorithm_encoding = ""

    def algorithm_encoding_generator(self):
        
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

    def dia(self, Scales, p):
        """
        This function takes a list of scales and a list of chromatic pitch differences, 
        and returns a tuple of the legal indices and the equivalent diatonic pitch differences
        e.g. ([0, 5, 10], (1,3,-2,5,-3))
        """

        # make sure that the scale is long enough
        scale = list(itertools.accumulate([0] + (Scales[1:] * 5)))  
        comp_pitch3 = list(itertools.accumulate(p))
        backup = comp_pitch3.copy()

        # throw out duplicates from comp_pitch
        comp_pitch3 = list(dict.fromkeys(comp_pitch3))

        # throw out duplicates from comp_pitch
        comp_pitch3 = list(dict.fromkeys(comp_pitch3))

        # sort it ascending
        comp_pitch3.sort()

        comp_pitch3 = comp_pitch3.copy()
        # if any of the comp_pitch pitches are negative, keep adding 12 to all of them until they are all positive
        while any(i < 0 for i in comp_pitch3):
            comp_pitch3 = [i + 12 for i in comp_pitch3]
        

        len_scale = len(scale)
        len_comp_pitch = len(comp_pitch3)
        x = 0
        offsets = []
        indices = []
        new_comp_pitches = []
        while x < len_scale - len_comp_pitch:
            new_comp_pitch = [i + x for i in comp_pitch3]
            if all(i in scale for i in new_comp_pitch):
                new_comp_pitches.append(new_comp_pitch)
                offsets.append(x)
                indices.append(scale.index(new_comp_pitch[0]))
            x += 1
        # Go through all the new_comp_pitches lists, remove duplicates
        sums = []
        results_pre = []
        for l in new_comp_pitches:
            if sum([l2 % 12 for l2 in l] ) not in sums:
                results_pre.append(l)
                sums.append(sum([l2 % 12 for l2 in l]))
        results_pre.sort(key=lambda x: sum(x))
        results = {}
        # store the results in a dictionary with the index of the first pitch as the key
        for i in range(len(results_pre)):
            results[indices[i]] = results_pre[i]
        

        # go through the results dictionary and capture the keys.
        # go though p and generate the diatonic pitch differences
        # assemble the results as described in the documentation for the function, and convert the results into a tuple
        # crawl through p and generate the diatonic pitch differences
        scaleDiff = [0] + Scales[1:8] * 5

        indi = []
        # for i in range(len(example)):
        # find the index of  example[i] in scale
        # indi.append(scale.index(example[i]))
        for i in range(len(backup)):
            indi.append(scale.index([y+offsets[0] for y in backup][i]))
        
        return (results.keys(), tuple(indi))

    # =MID("CDEFGAB", 1 +MOD( ROUND($A2*7/12, 0), 7), 1)
    def chr(self, Scales, p, starti):
        # do the reverse of dia
        # take a list of scales, a list of diatonic pitch differences, and a starting index
        # return a list of chromatic pitch differences
        scale = list(itertools.accumulate([0] + (Scales[1:] * 5)))
        scaleDiff = [0] + Scales[1:8] * 5
        chr_pitch_diffs = []
        last = 0
        for i in range(len(p)-1):
            mult = 1
            if p[i] < last:
                mult = -1
                chr_pitch_diffs.append(sum(scaleDiff[starti + p[i+1]:starti + p[i]]) * mult)
            else:
                chr_pitch_diffs.append(sum(scaleDiff[starti + p[i]:starti + p[i+1]]))
            last = p[i]
        return chr_pitch_diffs
    

    def example_algo(self, pit):
        # this is an example of an algorithm that can be generated, so it won't assembled into the algorithm_encoding
        # this is just to show how the algorithm_encoding can be used
        S = {
            'NUp': [0, 2, 2, 1, 2, 2, 2, 1],
            'HUp': [0, 2, 1, 2, 2, 2, 2, 1],
            'NDn': [-1 * x for x in reversed(['NUp'])],
            'HDn': [-1 * x for x in reversed(['HUp'])]
        }
        scale = [0] + essence.SCALES['NormUp'][1:8] * 5
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
        # result += P['p1']
        # print(result)
        dia_res = self.dia(S['NUp'], P['p1'])
        chrs = self.chr(S['NUp'], dia_res[1], list(dia_res[0])[0])
        result += chrs
        print(result)
        # print([-x for x in P['p0']])


        indx = 0
        for p in result:
            if p != pit[indx]:
                # print what's different about pit and result
                print(f"pit[{indx}], {p} != {pit[indx]}")
            indx += 1

if __name__ == "__main__":

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
    # print("original_pitches")
    # print(original_pitches[0:7])
    # # print next 30 elements of original_pitches with header
    # print("original_pitches continued")
    # print(original_pitches[7:37])

    essence = MidiEssence(original_pitches)
    result = essence.dia(Scales=essence.SCALES['NormUp'], p=[7, 5, -2, 2, -3])
    scale = [0] + essence.SCALES['NormUp'][1:8] * 5
    print('scale', scale)
    print('results of dia', result)

    # extract from result the indices and the diatonic pitch differences
    indices = list(result[0])
    chr_p = essence.chr(Scales=essence.SCALES['NormUp'], p=result[1], starti=indices[0])
    print('chr_p', chr_p)


