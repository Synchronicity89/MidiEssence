import os
import unittest
# because the test file is in the same folder as the module, the dot was crucial for test explorer to be able to run it
# that's because the test explorer is running from the root folder
from .build_midiessence import MidiEssence, Up, S

class TestBuildMidiEssence(unittest.TestCase):
    def setUp(self):
        self.essence = MidiEssence()

    def test_chromatic_diatonic_conversion(self):
        scale = Up('N', 3)
        p = [0, 2, 4, 5, 2, 4, 0, 7]
        dia_p = self.essence.dia(scale, p)
        self.assertEqual(dia_p[1], [0, 1, 2, 3, 1, 2, 0, 4])
        chr_p = self.essence.chr(scale, 0, dia_p[1])
        self.assertEqual(chr_p, p)


if __name__ == '__main__':
    unittest.main()