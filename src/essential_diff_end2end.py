import pandas as pd
# df0.to_csv("../../data/interim/sample_note_data.csv", index=False)
# df.to_pickle("../../data/interim/note_data.pkl")
# chdir to the data folder
import os
os.chdir("data")
import data.make_dataset

from features.build_features_diff import encode_diffs, decode_diffs
# read the note_data.pkl file into a dataframe
os.chdir("../features")
df = pd.read_pickle("../../data/interim/note_data.pkl")

# encode the dataframe.  This will create a new column for each column in the dataframe with the suffix "_diff"
# it turned out to be confusing dealing with diff data, so this whole approach is probably being mothballed
df_encoded = encode_diffs(df)

# write the encoded dataframe to a pickle file called note_data_diff_encoded.pkl in the interim folder
df_encoded.to_pickle("../../data/interim/note_data_diff_encoded.pkl")

from features.build_midiessence_diff import MidiEssence_diff
df = pd.read_pickle("../../data/interim/note_data_diff_encoded.pkl")
pitch_data = df["Pitch_diff"].astype('int').tolist()

# remove bogus data
# remove 10th element from pitch_data, and other cheap trills
# There are trills in the original sheet music, but in these MIDI files, they are encoded as a single note appoggiatura
# an appoggiatura is a grace note that takes up a small fraction of the duration of the main note, but is played on the beat
# so I'm removing the appoggiatura from the data.  It was a very cheap trill anyway
pitch_data.pop(10)
pitch_data.pop(22)

essence = MidiEssence_diff()
result = essence.dia(Scales=essence.SCALES['NormUp'], p=[7, 5, -2, 2, -3])
scale = [0] + essence.SCALES['NormUp'][1:8] * 5
print('scale', scale)
print('results of dia', result)