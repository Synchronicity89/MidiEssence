import random
import unittest
from build_midiessence import MidiEssence, Up, S

class TestBuildMidiEssence(unittest.TestCase):
    def setUp(self):
        self.essence = MidiEssence()

    def test_conversion(self):
        self.assertEqual(1, 2)

if __name__ == '__main__':
    unittest.main()