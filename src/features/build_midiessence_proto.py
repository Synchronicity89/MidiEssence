from collections import Counter
from itertools import combinations

class MidiEssence:
    
    # Hard-coded scales and chords
    SCALES = {
        'C_MAJOR': [60, 62, 64, 65, 67, 69, 71, 72],
        'D_MINOR': [62, 64, 65, 67, 69, 70, 72, 74]
    }
    CHORDS = {
        'C_MAJOR_CHORD': [60, 64, 67],
        'D_MINOR_CHORD': [62, 65, 69]
    }
    
    def __init__(self, original_pitches):
        self.original_pitches = original_pitches
        self.algorithm_encoding = ""
        
    def detect_scale(self, pitches):
        for scale_name, scale_pitches in self.SCALES.items():
            if set(pitches).issubset(set(scale_pitches)):
                return scale_name
        return None
    
    def detect_chord(self, pitches):
        for chord_name, chord_pitches in self.CHORDS.items():
            if set(pitches) == set(chord_pitches):
                return chord_name
        return None
    
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
        self.algorithm_encoding += "global result\n"
        self.algorithm_encoding += "result = []\n\n"
        
        for idx, (pattern, _) in enumerate(common_patterns):
            self.algorithm_encoding += f"def pattern_{idx}():\n"
            self.algorithm_encoding += f"    return list({list(pattern)})\n\n"
            
            # Add music theory knowledge
            detected_scale = self.detect_scale(pattern)
            detected_chord = self.detect_chord(pattern)
            if detected_scale:
                self.algorithm_encoding += f"# Pattern {idx} is in {detected_scale} scale\n"
            if detected_chord:
                self.algorithm_encoding += f"# Pattern {idx} forms a {detected_chord}\n"
                
        # Generate result
        self.algorithm_encoding += "for _ in range(50):\n"
        for idx in range(len(common_patterns)):
            self.algorithm_encoding += f"    result += pattern_{idx}()\n"
        
        self.algorithm_encoding += "result = result[:200]\n"  # truncate to 200 elements

# Example usage
original_pitches = [60, 62, 64, 60, 65, 67, 69, 65, 67, 69] * 20
essence = MidiEssence(original_pitches)
essence.algorithm_encoding_generator()

# Test the generated encoding
exec(essence.algorithm_encoding)

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

print(essence.algorithm_encoding)