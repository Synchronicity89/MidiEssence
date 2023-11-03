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

from features.build_midiessence import MidiEssence, Up, Dn, S,  modify_note_data
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

chr_p = essence.chr(Up(), result[0][0], result[0])
print('chr_p', chr_p)

# create a bunch of  note_data_modified??.pkl files in the folder ../../data/interim/
#  
for x in range(100):
    modify_note_data()


# get a list of all note_data_modified??.pkl files where the ?? is a number in the folder ../../data/interim/
list_of_files = os.listdir("../../data/interim/")
list_of_files = [file for file in list_of_files if file.startswith('note_data_modified')]
list_of_files.sort()
#new_file_name = f"note_data_modified{last_number:02d}.pkl"
from data.make_midis import process_midi_data
# loop through the list of files and process each one, using the number in the filename as midi_name.
# set make list_filenames contain just the first Filename from the dataframe
# Load the track metadata from the pickle file
# load a dataframe with the note data that has been modified
df_meta = pd.read_pickle("../../data/interim/track_metadata.pkl")
list_filenames = df['Filename'].unique()[0:1]
for file_name in list_of_files:
    df = pd.read_pickle("../../data/interim/" + file_name)
    midi_name = file_name[18:20]
    print(f"Processing {file_name} with midi_name {midi_name}")
    process_midi_data(df, df_meta, midi_name, list_filenames, fix_pitch = True)
    print(f"Finished processing {file_name}")


