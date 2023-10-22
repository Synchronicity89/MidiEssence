import pandas as pd
# df0.to_csv("../../data/interim/sample_note_data.csv", index=False)
# df.to_pickle("../../data/interim/note_data.pkl")
# chdir to the data folder
import os
os.chdir("data")
import data.make_dataset

# read the note_data.pkl file into a dataframe
os.chdir("../features")
df = pd.read_pickle("../../data/interim/note_data.pkl")

from features.build_midiessence import MidiEssence, Up, Dn, S
df = pd.read_pickle("../../data/interim/note_data.pkl")
pitch_data = df["Pitch"].astype('int').tolist()

# not removing the apoggiaturas that are in place of trills

# read the first 200 pitches from pitch_data into a list called original_pitches
original_pitches = pitch_data[0:200]

essence = MidiEssence()

result = essence.dia(Scales=Up('N', 4), p=[7, 12, 10, 12, 9])
scale = Up('N', 4)
print('scale', scale)
print('results of dia', result)

chr_p = essence.chr(Up(), result, result[0])
print('chr_p', chr_p)
