import pandas as pd
from collections import Counter
from itertools import combinations
import itertools
import copy
import random



global S
S = {
        'NUp': (0, 2, 4, 5, 7, 9, 11),
        'MUp': (0, 2, 3, 5, 7, 9, 11),
        'HUp': (0, 2, 3, 5, 7, 8, 11),
        'NDn': (12, 11, 9, 7, 5, 4, 2),
        'MDn': (12, 10, 8, 7, 5, 3, 2),
        'HDn': (12, 11, 8, 7, 5, 3, 2)
    }

# The range is from C −2 (note #0) to G8 (note #127). Middle C is note #60 (known as C3 in MIDI terminology).
# here we arbitrarily stop at 120
def Up(sc = 'N', repeats=10):
    # throw an exception if sc is not in ['N', 'M', 'H']
    if sc not in ['N', 'M', 'H']:
        raise ValueError(f"Invalid argument: {sc}. Must be one of 'N', 'M', 'H'")
    # repeat the scale such as S['NUp'] repeat number of octaves, adding 12 each time
    scale = S[sc + 'Up']
    long_scale = list(scale)
    for i in range(repeats - 1):
        long_scale += [x + 12*(i+1) for x in scale] #if x+12 not in long_scale
    return long_scale

# Equivalent function for down replication of scale pitches, called Dn, keeping in mind it has to start with the highest pitch
def Dn(sc = 'N', repeats=10):
    up_scale =  Up(sc, repeats)
    return [repeats * 12] + list(reversed(up_scale))[:repeats*7-1]


class MidiEssence:
   
    def __init__(self):
        self.algorithm_encoding = ""
        self.zzz = None

    def make_z(self, data):
        for item in data:
            yield item

    @property
    def z(self):
        return next(self.zzz)        

    def algorithm_encoding_generator(self):

        self.algorithm_encoding += "global result\n"
        self.algorithm_encoding += "result = []\n\n"
        
        
        self.algorithm_encoding += "result = result[:200]\n"  # truncate to 200 elements

    def dia(self, Scales, p):
        """
        This function takes a list of scales and a list of chromatic pitches, 
        and returns a list of legal lists of diatonic pitches
        e.g. [9, 14, 12, 14, 11]
        However, because it often has to shift the pitches in order to find a match,
        the function also returns a list of offsets, which is the number of semitones shifted
        """

        # make sure that the scale is long enough
        scale = Scales
        comp_pitch3 = list(p)
        backup = comp_pitch3.copy()

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

        # find all the possible offsets for the comp_pitch3 pitches
        # find all the possible indices for the comp_pitch3 pitches
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

        indi = []
        if len(results) > 0:
            for i in range(len(backup)):
                indi.append(scale.index([y+offsets[0] for y in backup][i]))
        
        return (offsets[: len(sums)+1], indi)


    # =MID("CDEFGAB", 1 +MOD( ROUND($A2*7/12, 0), 7), 1)
    def chr(self, Scales, starti, p, offset=0):
        # do the reverse of dia
        # take a list of scales, a list of diatonic pitch differences, and a starting index
        # return a list of chromatic pitch differences
        output = [Scales[x + starti] + offset for x in p]
        return output

    def flatten_dict(self, d):
        flat_list = []
        struct = {}

        for key, value in d.items():
            struct[key] = []
            for i, item in enumerate(value):
                if isinstance(item, list):
                    struct[key].append((i, len(item)))
                    flat_list.extend(item)
                else:
                    struct[key].append((i, 1))
                    flat_list.append(item)

        return flat_list, struct

    def unflatten_dict(self, flat_list, struct):
        d = {}
        index = 0

        for key, value in struct.items():
            d[key] = []
            for item in value:
                i, length = item
                if length == 1:
                    d[key].insert(i, flat_list[index])
                    index += 1
                else:
                    d[key].insert(i, flat_list[index:index+length])
                    index += length

        return d

    def randomize_dict_values(self, dict):
        # d is a flat 1 dimensional list of numbers.
        # We want to subtly vary the integers in the list, but not too much, keeping them as integers
        # On average we want a 20% chance that a number will go up or down by up to 50%, or up or down by 4, whichever is smaller, and a 
        # 5% chance that a number will go up or down by up to 100% or up or down by 8, whichever is smaller
        # We want to keep the numbers in the same order, so we can't just shuffle the list
        # We want to keep the numbers within the range of the original numbers, in the sense that they are similar
        d, struct = self.flatten_dict(dict)
        d_copy = []
        factor = 0
        for x in d:
            # generate a random integer that has a 20% chance of being 50% bigger or smaller than x
            r = random.random()
            if r < 0.2:
                factor = 1.5
            # generate a random integer that has  a 5% chance of being 100% bigger or smaller than x
            elif r < 0.05:
                factor = 2
            else:
                d_copy.append(x)
                continue
            min = -abs(round(x - (x * factor)))
            max = abs(round( x + (x * factor)))
            if min == max: continue
            candidate = x + random.randint( min , max )
            d_copy.append(candidate)


        return self.unflatten_dict(d_copy, struct)
 
    def example_algo(self, pitches_list, mut=None):
        # this is an example of an algorithm that can be generated, so it won't assembled into the algorithm_encoding
        # this is just to show how the algorithm_encoding can be used
        pit = pitches_list[0]
        N = Up('N', 6)
        M = Up('M', 6)
        H = Up('H', 6)
        global result
        result = []
        result0 = []
        result1 = []
        result.append(result0)
        result.append(result1)
        P_orig = {
            'p0' : N[0:4] + [2, 4, 0],
            'p1' : [7, 12, 11, 9, 11, 12,14],
            'p2' : [55, 43],
            'p3' : [2, 12, 11, 12, 14, 11],
            'p4' : [59, 57, 55, 54, 57, 56, 59], # TODO: might only be used once
            'p5' : [64, 62, 60, 59, 62, 61, 64],  # TODO: needs to be used in RH somewhere
            'ms_5_0' :[[4, 0, -5], [6, -4, -5], [8, -4, -5]],
            'ms_11_12' :[[6, 4, 2], [4, 4, 2], [5, 4, -3], [9, 2, -3]],
            'ms_06' : [11, 12, 14, 19, 11, 12, 11, 9, 7, 7],
            'ms_07_09' : [[0, 7, -1], [1, 7, 2], [0, 7], 5, [13, 16, 14]],
            'ms_13_14' : [4, 7, 7, 11, [8, 17, 16, 14, 12, 14, 12, 11, 9, 9]],
            'ms_15_19' : [[9, 1], 7, [8, 2], [7, 1], [9, 1]],
            'ms_20_22' : [[11, 3], [19, 16, 17, 16, 14, 12, 12], [-1, 5], [11, 12, 4, 2, 12, 5, 11, 4]],
            'ms_01_02_lh' : [[4, 5]],
            'ms_02_03_lh' : [[6, 0], [4, 0], [-2, 7], [2, 7]],
            'ms_04_06_lh' : [[2, 5], [-5, 7], [-1, 7], [52, 47, 48, 50, 38]],
            'ms_07_12_lh' : [5, -7, 5, 12, 8, [1, 12], [6, 5], [4, 5], 5, [5, -3]],
            'ms_13_15_lh' : [[52, 62, 60, 62, 60], 4, 6, [64, 57, 64, 52, 57, 45]],
            'ms_16_22_lh' : [-2, [5, 2], -7, [-1, 7, 7], [1, 7, 7], [3, 7, 7, 2], -12, [53, 55, 43, 48]]
        }
        P = None
        if mut != None:
            P = mut(P_orig)
        else:
            P = P_orig

        # after each change to result, print the result
        # measures 1 and 2 of right hand of invention 1 by Bach
        # print(result0)
        result0 += P['p0']
        # print(result0)
        result0 += P['p1']
        result0 += [x + 7 for x in P['p0']]
        # set p1 to a P['p1], but leaving off the last element of that list
        p1 = P['p1'][:-1]

        p1_dia = self.dia(N, p1)
        chrs = self.chr(N, p1_dia[0][0] + 4, p1_dia[1])
        result0 += chrs

        # measures 3 and 4 of right hand of invention 1 by Bach

        result0 += [16]
        # invert the diatonic figure p1_dia into p1_dia_inv
        p0_dia = self.dia(N, P['p0'])
        p0_dia_inv = [max(self.xte(p0_dia[1], 1)) - x + 1 for x in self.xte(p0_dia[1], 1)]

        # entire pattern is repeated 1 diatonic step lower each time
        for do in self.chr(N, -2, p0_dia_inv[:3], 4):
            result0 += self.chr(N, do, p0_dia_inv)

        p0_dia_inv_chr0 = self.chr(N, 8, p0_dia_inv)
        result0 += [x - 10 for x in p0_dia_inv_chr0]

        # measures 5 and 6 of right hand of invention 1 by Bach

        result0 += P['p3']

        for do, st, co in P['ms_5_0']:        
            result0 += self.chr(N, do, p0_dia_inv[st:], co)
        
        result0 += P['ms_06']

        self.zzz = self.make_z(P['ms_07_09'])
        # measures 7 and 8 of right hand of invention 1 by Bach

        o = self.z
        result0 += self.xte(self.chr(N, o[0], p0_dia[1], o[1]), o[2])
        o = self.z

        # measures 9 and 10 of right hand of invention 1 by Bach

        result0 += self.xte(self.chr(N, o[0], p0_dia[1], o[1]), o[2])
        o = self.z
        p0_dia_5_inv_chr = self.chr(N, o[0], p0_dia_inv, o[1])
        result0 += p0_dia_5_inv_chr

        result0 += self.chr(N, self.z, p0_dia_inv)[:-3]
        result0 += self.z

        # measures 11 and 12 of right hand of invention 1 by Bach

        for do, ed, co in P['ms_11_12']:
            result0 += self.chr(M, do, p0_dia[1][:ed], co)

        # measures 13 and 14 of right hand of invention 1 by Bach
        self.zzz = self.make_z(P['ms_13_14'])
        result0 += self.chr(M, self.z, p0_dia[1],-3)
        result0 += self.chr(M, self.z, p0_dia_inv[:3],-3)
        result0 += self.chr(M, self.z, p0_dia_inv[:6],-3)
        result0 += self.chr(M, self.z, p0_dia_inv[-4:],-3)
        result0 += self.Tr(P['p0'][2:5], 12)
        result0 += self.z

        # measures 15 thru 17 of right hand of invention 1 by Bach

        self.zzz = self.make_z(P['ms_15_19'])
        result0 += p0_dia_inv_chr0
        o = self.z
        result0 += self.chr(N, o[0], self.xte(p0_dia[1], o[1]))
        result0 += self.chr(N, self.z, p0_dia_inv)

        # measures 18 thru 19 of right hand of invention 1 by Bach

        o = self.z
        useinLHtoo = self.xte(self.chr(N, o[0], p0_dia[1]), o[1])
        result0 += useinLHtoo
        o = self.z
        result0 += self.chr(N, o[0], self.xte(p0_dia[1], o[1]))
        o = self.z
        result0 += self.chr(N, o[0], self.xte(p0_dia[1], o[1]))

        # measures 20 thru 22 of right hand of invention 1 by Bach

        self.zzz = self.make_z(P['ms_20_22'])
        o = self.z
        result0 += self.chr(N, o[0], self.xte(p0_dia[1], o[1]))
        result0 += self.z
        o = self.z
        result0 += self.chr(N, o[0], p0_dia_inv, o[1])
        result0 += self.z

        # done with RH, now do left
        pit = pitches_list[1]
        # find the minimum pitch in pit
        pit_min = min(pit)
        # divide pit_min by 12 and throw away the remainder
        octave_factor = pit_min // 12


        # measures 1 and 2 of left hand of invention 1 by Bach

        offset = (1 + octave_factor) * 12
        offset_d = (1 + octave_factor) * 7        
        result1 += self.Tr(P['p0'], offset)
        result1 += P['p2']

        o = P['ms_01_02_lh'][0]
        result1 += self.xte(self.chr(N, o[0], p0_dia[1], offset), o[1])

        # measures 2 thru 6 of left hand of invention 1 by Bach
        index = 0
        p1_dia = self.dia(N, p1)
        for do, co in P['ms_02_03_lh']:
            result1 += self.chr(N, do + offset_d, p0_dia[1][:4 if index != 3 else 2], co)
            index += 1

        self.zzz = self.make_z(P['ms_04_06_lh'])
        o = self.z
        result1 += self.xte(self.Tr(P['p0'], o[0] + offset), o[1])
        o = self.z
        result1 += self.chr(N, o[0] + offset_d, p0_dia[1][:4], o[1])
        o = self.z
        result1 += self.chr(N, o[0] + offset_d, p0_dia[1][:2], o[1])
        result1 += self.z


        # measures 7 -

        self.zzz = self.make_z(P['ms_07_12_lh'])
        o = self.z
        temp = [x - o + offset for x in result0[:27]]
        # remove the trills
        # remove the 11th and 12th pitches of temp, plus remove the 3rd and 2nd to last pitches of temp
        result1 += self.xte(temp[:10] + temp[12:-5] + temp[-5:-3] + temp[-1:], self.z)
        result1 += self.Tr(p0_dia_5_inv_chr, self.z + offset)


        result1 += self.Tr(P['p0'][2:-1], offset + self.z)
        result1 += self.chr(N, self.z, p0_dia_inv, offset)

        # measures 10 -

        temp = self.dia(N, P['p0'][2:-1])
        o = self.z
        result1 += self.chr(N, o[0], temp[1], offset + o[1])
        o = self.z
        result1 += self.chr(N, o[0], p0_dia_inv, offset + o[1])

        # measures 11 -

        o = self.z
        result1 += self.chr(N, o[0], p0_dia_inv, offset + o[1])

        # measures 12 -

        result1 += self.chr(N, self.z, p0_dia_inv, offset)
        o = self.z
        result1 += self.chr(H, o[0], p0_dia_inv, offset + o[1])

        # measures 13 -

        self.zzz = self.make_z(P['ms_13_15_lh'])

        result1 += self.z
        #TODO: find how the RH did it and copy that
        result1 += P['p4']
        result1 += self.chr(N, self.z, p0_dia_inv[3:][:-1], offset)
        result1 += self.chr(N, self.z, p0_dia_inv[3:][:-1], offset)
        result1 += self.z

        # measures 15 -
        self.zzz = self.make_z(P['ms_16_22_lh'])

        result1 += self.xte(P['p5'], self.z)
        o = self.z
        result1 += self.xte(self.chr(N, o[0], self.dia(N, P['p0'])[1], offset), o[1])
        result1 += self.Tr(p0_dia_5_inv_chr, offset)
        result1 += self.Tr(useinLHtoo, offset + self.z)
        o = self.z
        result1 += self.chr(N, o[0], self.Tr(p0_dia_inv[:4], o[1]), offset - o[2])
        o = self.z
        result1 += self.chr(N, o[0], self.Tr(p0_dia_inv[:4], o[1]), offset - o[2])
        o = self.z
        result1 += self.xte(self.chr(N, o[0], self.Tr(p0_dia_inv[:3], o[1]), offset - o[2]), o[3])
        result1 += self.Tr(useinLHtoo, offset + self.z)
        result1 += self.Tr(P['p0'][:6], offset)
        result1 += self.z


        # subtract 12*octave_factor from all the numbers in pit
        pit_lowered = pit # [p - 12 * octave_factor for p in pit]

        print('-'*60)
        pit_l = pit_lowered[:len(result1)]
        print(pit_l[180:])

        indx = 0
        for p in result1:
            if p != pit_lowered[indx]:
                # print what's different about pit_lowered and result
                print(f"pit_lowered[{indx}], {p} != {pit_lowered[indx]}")
            indx += 1

        # Transpose the pitches of result0 to match correct octave of original
        # get the 
        octave_diff = (pitches_list[0][0] - result0[0])/12
        result0 = [x + 12 * octave_diff for x in result0]
        # force into result in case it somehow is only a copy
        result = [[int(x) for x in result0], result1]
        return result
    
    def Tr(self, p, offset):
        return [x + offset for x in p]

    def xt(self, p0_dia_1, extend):
        '''
        This function extends a typically diatonic figure with an additional element that is the 
        same as the last element of the figure plus the extend value
        '''
        return [p0_dia_1[-1]+extend]

    def xte(self, p0_dia_1, extend):
        '''
        This function extends a typically diatonic figure with an additional element that is the 
        same as the last element of the figure plus the extend value
        '''
        return p0_dia_1 + [p0_dia_1[-1]+extend]
        
import pandas as pd
import copy

def modify_note_data():
    df = pd.read_pickle("../../data/interim/note_data.pkl")
    pitch_data = df["Pitch"].astype('int').tolist()

    # Leave the one note 'trill' apoggiaturas in the data

    # read the first 200 pitches from pitch_data into a list called original_pitches
    original_pitches = pitch_data[0:250]

    # extract from df into a list of lists original_pitches_lists, one list of pitches for each the instruments of a given filename
    filename = df['Filename'].unique()[0]
    original_pitches_lists = df[df['Filename'] == filename].groupby('Instrument')['Pitch'].apply(list).tolist()

    essence = MidiEssence()

    # essence.example_algo(original_pitches_lists)
    for i in range(30):
        try:
            essence.example_algo(original_pitches_lists, essence.randomize_dict_values)
        except:
            print("exception\n")
            continue
        first_result = copy.deepcopy(result)

        # compare first_result[0] with result[1]
        print('-'*60)
        print("Comparing pitches in first_result[0] with result[0]")
        print('-'*60)
        for i in range(len(first_result[0])):
            if first_result[0][i] != result[0][i]:
                print(f"first_result[0][{i}], {first_result[0][i]} != {result[0][i]}")
                break
        print('-'*60)
        print("Comparing pitches in first_result[1] with result[1]")
        print('-'*60)
        for i in range(len(first_result[1])):
            if first_result[1][i] != result[1][i]:
                print(f"first_result[1][{i}], {first_result[1][i]} != {result[1][i]}")
                break

        # print first 80 pitches in result[0] with header and likewise for result[1]

        # print first 80 pitches in result[0] with header
        print("Result[0]:")
        print(result[0][:80])
        print('-'*60)
        print("original_pitches_lists[0]:")
        print(original_pitches_lists[0][:80])


        # print first 80 pitches in result[1] with header
        print("\nResult[1]:")
        print(result[1][:80])
        print('-'*60)
        print("original_pitches_lists[1]:")
        print(original_pitches_lists[1][:80])

        # compare the pitches in result[0] with original_pitches_lists[0] and print a message if they are not the same, and the index of the first difference.
        # only compare len(result[0]) pairs of pitches
        print('-'*60)
        print("Comparing pitches in result[0] with original_pitches_lists[0]")
        print('-'*60)
        for i in range(len(result[0])):
            if result[0][i] != original_pitches_lists[0][i]:
                print(f"result[0][{i}], {result[0][i]} != {original_pitches_lists[0][i]}")
                break
        print('-'*60)
        print("Comparing pitches in result[1] with original_pitches_lists[1]")
        print('-'*60)
        for i in range(len(result[1])):
            if result[1][i] != original_pitches_lists[1][i]:
                print(f"result[1][{i}], {result[1][i]} != {original_pitches_lists[1][i]}")
                break
        pitch_parts = [result[0].copy(), result[1].copy()]
        # go through the rows of df for the first filename and instrument and sequentially replace the pitch data from result[0]
        filename = df['Filename'].unique()[0]
        instruments = df[df['Filename'] == filename]['Instrument'].unique()
        index = 0
        for instrument in instruments:
            res0 = df.loc[(df['Filename'] == filename) & (df['Instrument'] == instrument), 'Pitch']
            len_res0 = len(res0)

            if len(pitch_parts[index]) < len_res0:
                padding = len_res0 - len(pitch_parts[index])
                pitch_parts[index].extend([60] * padding)
                print(f"Padding occurred: {padding} values were added to pitch_parts[{index}].")
            elif len(pitch_parts[index]) > len_res0:
                truncation = len(pitch_parts[index]) - len_res0
                pitch_parts[index] = pitch_parts[index][:len_res0]
                print(f"Truncation occurred: {truncation} values were removed from pitch_parts[{index}].")

            df.loc[(df['Filename'] == filename) & (df['Instrument'] == instrument), 'Pitch'] = pitch_parts[index]
            index += 1

        # save the modified dataframe to a new pickle file
        df.to_pickle(f"../../data/interim/note_data_modified{str(i).zfill(2)}.pkl")

if __name__ == "__main__":
    modify_note_data()




