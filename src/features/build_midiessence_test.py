import random
import unittest
from build_midiessence import MidiEssence, Up, S

class TestBuildMidiEssence(unittest.TestCase):
    def setUp(self):
        self.essence = MidiEssence()

    def test_conversion(self):
        # test for crashes, to see if something is broken, and compare the original p with the chr_p
        # generate 2 random test patterns from each of the 4 scales and convert to diatonic and back
        # keep the random patterns to 8 notes or less, and the range of the pattens to 4 octaves or less
        no_conversion_count = 0
        simple_error_count = 0
        tests = 60
        simple_output = ""
        random.seed(42)
        for j in range(4):
            for i in range(tests):
                scale = Up(list(S.keys())[j][0], 4)
                p = random.sample(range(0, len(scale)), random.randint(1, 7))
                result = self.essence.dia(scale, p)
                if len(result[1]) > 0:
                    chr_p = self.essence.chr(scale, result, result[0])
                    # compare the original p with the chr_p and print a message if they are not the same
                    self.assertEqual(p != chr_p)
                    # what if a very simple formula had been used to convert chromatic to diatonic
                    # do a simplistic conversion of p to chr_p using ideas from this formula (here in Excel format): chr_p_simple = MID("CDEFGAB", 1 + MOD( ROUND(p*12/7, 0), 7), 1)
                    dia_p_simple = []
                    for x in p:
                        y = round((x)*7/12)
                        dia_p_simple += [y]
                    # compare the original p with the chr_p_simple and append newlines and messages to simple_output if they are not the same
                    if result[1] != dia_p_simple:
                        simple_error_count += 1
                        simple_output += f"p != dia_p_simple for scale {list(S.keys())[j]}\n"
                        simple_output += f"p = {p}\n"
                        simple_output += f"dia_p_simple = {dia_p_simple}\n"
                else:
                    no_conversion_count += 1

if __name__ == '__main__':
    unittest.main()