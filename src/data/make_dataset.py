import os
import pandas as pd
# conda install -c conda-forge music21 
from music21 import converter, note, chord

# Define the folder containing the MIDI files
folder_path = "../../data/raw/BachInventions"

# if the folder does not exist, then create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    # also create the other folders in the data folder: interim, processed, and external
    os.makedirs("../../data/interim")
    os.makedirs("../../data/processed")
    os.makedirs("../../data/external")
    # download the MIDI files from the web based on this url pattern https://www.bachcentral.com/invent/invent1.mid, download 1-8
    for i in range(1, 9):
        url = "https://www.bachcentral.com/invent/invent" + str(i) + ".mid"
        print("Downloading " + url)
        os.system("curl -o " + folder_path + "/invent" + str(i) + ".mid " + url)


# Get a list of all files in the folder
file_list = os.listdir(folder_path)
fileNo = 0
note_data = []
for file_name in file_list:
    if file_name.endswith(".mid"):
        midi_file_path = os.path.join(folder_path, file_name)

        # Use music21 to parse MIDI and access note properties
        score = converter.parse(midi_file_path)

        # Iterate through parts and notes
        partNo = 0
        for part in score.parts:
            for element in part.flat:
                if isinstance(element, note.Note):
                    # Extract note properties for single notes
                    pitch = element.pitch.midi
                    start = element.offset
                    end = start + element.quarterLength
                    velocity = element.volume.velocity
                    # note_data.append([fileNo, partNo, pitch, brev(start), brev(end), velocity])
                    note_data.append([fileNo, partNo, pitch, float(start), float(end), velocity])
                elif isinstance(element, chord.Chord):
                    # Extract data from the top or bottom note of the chord
                    # if the partNo is odd, then extract the bottom note, otherwise extract the top note
                    if partNo % 2 == 0:
                        bottom_note = element.pitches[0]
                        pitch = bottom_note.midi
                    else:
                        top_note = element.pitches[-1]
                        pitch = top_note.midi
                    start = element.offset
                    end = start + element.quarterLength
                    velocity = element.volume.velocity
                    # note_data.append([fileNo, partNo, pitch, brev(start), brev(end), velocity])
                    note_data.append([fileNo, partNo, pitch, float(start), float(end), velocity])
            partNo += 1
        fileNo += 1
df = pd.DataFrame(note_data, columns=["Filename", "Instrument", "Pitch", "Start", "End", "Velocity"])
df.info()
# Save the first 100 rows from each part related to a given filename as a CSV file.  Include all the filenames from df
df00 = df.loc[(df["Filename"] == 0) & (df["Instrument"] == 0)].head(100)
df01 = df.loc[(df["Filename"] == 0) & (df["Instrument"] == 1)].head(100)
df0 = pd.concat([df00, df01], ignore_index=True)

for fileNo in range(1, 7):
    # get the first 100 rows from each part related to a given filename and append to df0
    df00 = df.loc[(df["Filename"] == fileNo) & (df["Instrument"] == 0)].head(100)
    df01 = df.loc[(df["Filename"] == fileNo) & (df["Instrument"] == 1)].head(100)
    df0 = pd.concat([df0, df00, df01], ignore_index=True)

# Save df0 as a CSV file
df0.to_csv("../../data/interim/sample_note_data.csv", index=False)

# Save the DataFrame as a pickle file
df.to_pickle("../../data/interim/note_data.pkl")
df