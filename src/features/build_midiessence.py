import pandas as pd
from collections import Counter
from itertools import combinations
import itertools

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
    def chr(self, Scales, starti, p):
        # do the reverse of dia
        # take a list of scales, a list of diatonic pitch differences, and a starting index
        # return a list of chromatic pitch differences
        output = [Scales[x + starti] for x in p]
        return output
 
    def example_algo(self, pit):
        # this is an example of an algorithm that can be generated, so it won't assembled into the algorithm_encoding
        # this is just to show how the algorithm_encoding can be used
        N = Up('N', 4)
        M = Up('M', 4)
        global result
        result = []
        P = {
            'p0' : N[0:4] + [2, 4, 0],
            'p1' : [7, 12, 11, 9, 11, 12,14],
            # 'p2' : [-7]+['p0'][1:],
            # 'p2' : [-7]+['p0']*-1
        }
        # after each change to result, print the result
        # measures 1 and 2 of right hand of invention 1 by Bach
        print(result)
        result += P['p0']
        print(result)
        result += P['p1']
        print(result)
        result += [x + 7 for x in P['p0']]
        print(result)
        # set p1 to a P['p1], but leaving off the last element of that list
        p1 = P['p1'][:-1]

        p1_dia = self.dia(N, p1)
        chrs = self.chr(N, p1_dia[0][0] + 4, p1_dia[1])
        result += chrs
        print(result)

        # measures 3 and 4 of right hand of invention 1 by Bach
        result += [16]
        # invert the diatonic figure p1_dia into p1_dia_inv
        p0_dia = self.dia(N, P['p0'])
        p0_dia_inv = [max(p0_dia[1] + self.xt(p0_dia[1], 1)) - x + 1 for x in p0_dia[1] + self.xt(p0_dia[1], 1)]
        p0_dia_inv_chr0 = self.chr(N, p0_dia[0][0] + 8, p0_dia_inv)
        result += p0_dia_inv_chr0
        print(result)
        p0_dia_inv_chr1 = self.chr(N, p0_dia[0][0] + 6, p0_dia_inv)
        result += p0_dia_inv_chr1
        print(result)
        p0_dia_inv_chr2 = self.chr(N, p0_dia[0][0] + 4, p0_dia_inv)
        result += p0_dia_inv_chr2
        print(result)
        result += [x - 10 for x in p0_dia_inv_chr0]
        print(result)

        # measures 5 and 6 of right hand of invention 1 by Bach
        result += [2, 12, 11, 12, 14, 11]
        result += [x - 5 for x in self.chr(N, p0_dia[0][0] + 4, p0_dia_inv)]
        print(result)
        result += [x - 5 for x in self.chr(N, p0_dia[0][0] + 6, p0_dia_inv[-4:])]
        result += [x - 5 for x in self.chr(N, p0_dia[0][0] + 8, p0_dia_inv[-4:])]
        print(result)
        result += [11, 12, 14, 19, 11, 12, 11, 9, 7, 7]
        print(result[80:])
        p0_dia_5_chr = self.Tr(self.chr(N, 0, p0_dia[1]), 7)
        result += p0_dia_5_chr + self.xt(p0_dia_5_chr, -1)
        print(result[80:])
        p0_dia_6_chr = self.Tr(self.chr(N, 1, p0_dia[1]), 7)
        result += p0_dia_6_chr + self.xt(p0_dia_6_chr, 2)
        print(result[80:])
        p0_dia_5_inv_chr = self.Tr(self.chr(N, 0, p0_dia_inv), 7)
        result += p0_dia_5_inv_chr
        print(result[80:])
        p0_dia_6_inv_chr = self.chr(N, 5, p0_dia_inv)[:-3]
        result += p0_dia_6_inv_chr
        print(result[80:])
        result += [13, 16, 14]
        print(result[80:])
        result += self.chr(self.Tr(M, 2), -1, [x + 7 for x in p0_dia[1][:4]])
        print(result[80:])
        result += self.chr(self.Tr(M, 2), -3, [x + 7 for x in p0_dia[1][:4]])
        print(result[80:])
        result += self.chr(self.Tr(M, -3), -2, [x + 7 for x in p0_dia[1][:4]])
        print(result[80:])
        result += self.chr(self.Tr(M, -3), 2, [x + 7 for x in p0_dia[1][:2]])
        print(result[80:])
        result += self.chr(self.Tr(M, -3), 4, [x + 0 for x in p0_dia[1]])
        print(result[80:])
        result += self.chr(self.Tr(M, -3), 0, [x + 7 for x in p0_dia_inv[:3]])
        print(result[80:])
        result += self.chr(self.Tr(M, -3), 0, [x + 7 for x in p0_dia_inv[:6]])
        print(result[80:])
        result += self.chr(self.Tr(M, -3), -3, [x + 14 for x in p0_dia_inv[-4:]])
        print(result[80:])
        result += self.Tr(P['p0'][2:5], 12)
        result += [8, 17, 16, 14, 12, 14, 12, 11, 9, 9]
        print(result[160:])
        result += p0_dia_inv_chr0
        print(result[160:])
        interim = self.chr(N, 5, [x + 4 for x in p0_dia[1]])
        result += interim + self.xt(interim, 1)
        print(result[160:])
        interim = self.chr(N, p0_dia[0][0] + 7, p0_dia_inv)
        result += interim
        print(result[160:])
        interim = self.chr(N, 0, [x + 8 for x in p0_dia[1]])
        result += interim + self.xt(interim, 2)
        print(result[160:])
        interim = self.chr(N, 0, [x + 7 for x in p0_dia[1]])
        result += interim + self.xt(interim, 2)
        print(result[160:])
        interim = self.chr(N, 2, [x + 7 for x in p0_dia[1]])
        result += interim + self.xt(interim, 1)
        print(result[160:])
        interim = self.chr(N, 4, [x + 7 for x in p0_dia[1]])
        result += interim + self.xt(interim, 5)
        print(result[160:])
        result += [19, 16, 17, 16, 14, 12, 12]
        interim = self.chr(self.Tr(N, 5), -1, p0_dia_inv)
        result += interim
        print(result[200:])
        result += [11, 12, 4, 2, 12, 5, 11, 4]
        print(result[200:])

        # find the minimum pitch in pit
        pit_min = min(pit)
        # divide pit_min by 12 and throw away the remainder
        octave_factor = pit_min // 12
        # subtract 12*octave_factor from all the numbers in pit
        pit_lowered = [p - 12 * octave_factor for p in pit]

        print('-'*60)
        print(pit_lowered[:len(result)][200:])

        indx = 0
        for p in result:
            if p != pit_lowered[indx]:
                # print what's different about pit_lowered and result
                print(f"pit_lowered[{indx}], {p} != {pit_lowered[indx]}")
            indx += 1

    def Tr(self, p, offset):
        return [x + offset for x in p]

    def xt(self, p0_dia_1, extend):
        '''
        This function extends a typically diatonic figure with an additional element that is the 
        same as the last element of the figure plus the extend value
        '''
        return [p0_dia_1[-1]+extend]
        
if __name__ == "__main__":

    # Example usage
    # original_pitches = [60, 62, 64, 60, 65, 67, 69, 65, 67, 69] * 20

    df = pd.read_pickle("../../data/interim/note_data.pkl")
    pitch_data = df["Pitch"].astype('int').tolist()

    # Leave the one note 'trill' apoggiaturas in the data

    # read the first 200 pitches from pitch_data into a list called original_pitches
    original_pitches = pitch_data[0:250]
    
    essence = MidiEssence()

    essence.example_algo(original_pitches)

























    run_tests = False

    if run_tests:
        result = essence.dia(Scales=Up('N', 4), p=[7, 12, 10, 12, 9])
        scale = Up('N', 4)
        print('scale', scale)
        print('results of dia', result)

        chr_p = essence.chr(Up(), result, result[0])
        print('chr_p', chr_p)

        # generate 2 random test patterns from each of the 4 scales and convert to diatonic and back
        # keep the random patterns to 8 notes or less, and the range of the pattens to 4 octaves or less
        no_conversion_count = 0
        simple_error_count = 0
        tests = 60
        simple_output = ""
        import random
        random.seed(42)
        for j in range(4):
            for i in range(tests):
                scale = Up(list(S.keys())[j][0], 4)
                print('scale', scale)
                p = random.sample(range(0, len(scale)), random.randint(1, 7))
                print('p', p)
                result = essence.dia(scale, p)
                print('results of dia', result)
                if len(result[1]) > 0:
                    chr_p = essence.chr(scale, result, result[0])
                    print('chr_p', chr_p)
                    # compare the original p with the chr_p and print a message if they are not the same
                    if p != chr_p:
                        print(f"p != chr_p for scale {list(S.keys())[j]}")
                        print(f"p = {p}")
                        print(f"chr_p = {chr_p}")
                        result = essence.dia(scale, p)
                        chr_p = essence.chr(scale, result, result[0])
                    else:
                        print(f"p == chr_p for scale {list(S.keys())[j]}")
                    # what if a very simple formula had been used to convert chromatic to diatonic
                    # do a simplistic conversion of p to chr_p using ideas from this formula (here in Excel format): chr_p_simple = MID("CDEFGAB", 1 + MOD( ROUND(p*12/7, 0), 7), 1)
                    dia_p_simple = []
                    for x in p:
                        y = round((x)*7/12)
                        dia_p_simple += [y]
                    print('dia_p_simple', dia_p_simple)
                    # compare the original p with the chr_p_simple and append newlines and messages to simple_output if they are not the same
                    if result[1] != dia_p_simple:
                        simple_error_count += 1
                        print(f"p != dia_p_simple for scale {list(S.keys())[j]}")
                        print(f"p = {p}")
                        print(f"dia_p_simple = {dia_p_simple}")
                        simple_output += f"p != dia_p_simple for scale {list(S.keys())[j]}\n"
                        simple_output += f"p = {p}\n"
                        simple_output += f"dia_p_simple = {dia_p_simple}\n"
                    else:
                        print(f"p == chr_p_simple for scale {list(S.keys())[j]}")
                else:
                    no_conversion_count += 1
                    print(f"dia returned empty list for scale {list(S.keys())[j]}")


        # print the simple_output
        print('_' * 60 + '\n')
        print('_' * 60 + '\n')
        print(simple_output)

        # print the simple_error_count divided by the number of tests * 4, as a percentage with a header
        print('_' * 60 + '\n')
        print(f"simple_error_count = {simple_error_count} out of {tests*4} = {(no_conversion_count + simple_error_count)/(tests*4)*100}%")

        # running the code shows that the simple conversion is not good enough, failing almost 75% of the time

