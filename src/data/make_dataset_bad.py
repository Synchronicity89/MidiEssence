import os
import pandas as pd
# conda install -c conda-forge music21 
from music21 import *
from music21.common.numberTools import addFloatPrecision
import urllib.request

# Define the folder containing the MIDI files
folder_path = "../../data/raw/BachInventions"
# Define the folder containing the cleaned MIDI files
folder_path2 = "../../data/interim/BachInventions"

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
        urllib.request.urlretrieve(url, folder_path + "/invent" + str(i) + ".mid")
if not os.path.exists(folder_path2):
    os.makedirs(folder_path2)

# Get a list of all files in the folder
file_list = os.listdir(folder_path)
fileNo = 0
note_data = []
# create a dataframe with columns Filename, Instrument, Pitch, Start, End, Velocity
# make the datatype for Start and End be objects, so they can be Fraction objects
# make the datatype for Pitch and Velocity be 64 bit integers
# make the datatype for Filename and Instrument be 64 bit integers
df = pd.DataFrame(note_data, columns=["Filename", "Instrument", "Pitch", "Start", "End", "Velocity"])
df["Start"] = df["Start"].astype(object)
df["End"] = df["End"].astype(object)
df["Pitch"] = df["Pitch"].astype(int)
df["Velocity"] = df["Velocity"].astype(int)
df["Filename"] = df["Filename"].astype(int)
df["Instrument"] = df["Instrument"].astype(int)


for file_name in file_list:
    if file_name.endswith(".mid"):
        midi_file_path = os.path.join(folder_path, file_name)

        # Use music21 to parse MIDI and access note properties
        score = converter.parse(midi_file_path, quantizePost=False)

        # Iterate through parts and notes
        partNo = 0
        for part in score.parts:
            start_trill = 0.0
            fixing_trill = False
            bad_quarterLength = 0.0            
            # The trills were quantized in an unfortunate manner, so we will remove them
            for element in part.flat:
                pitch = -1
                velocity = -1
                # qL = float(element.quarterLength)
                # oS = float(element.offset)
                qL = element.quarterLength
                oS = element.offset
                if isinstance(element, note.Note) or isinstance(element, note.Rest):
                    if isinstance(element, note.Note):
                        # Extract note properties for single notes
                        pitch = element.pitch.midi
                        velocity = element.volume.velocity
                    if fixing_trill:
                        start = start_trill
                    else:
                        start = oS
                    end = start + qL# + bad_quarterLength
                    if(end-start < 0.25):
                        print('end-start', end-start)
                    # note_data.append([fileNo, partNo, pitch, float(start), float(end), velocity])
                    note_data.append([fileNo, partNo, pitch, start, end, velocity])
                    df.loc[len(df)] = [fileNo, partNo, pitch, start, end, velocity]
                    if fixing_trill:
                        fixing_trill = False
                        start_trill = 0.0
                        bad_quarterLength = 0.0
                elif isinstance(element, chord.Chord):
                    # Extract data from the top or bottom note of the chord
                    # if the partNo is odd, then extract the bottom note, otherwise extract the top note
                    # first identify a badly quantized trill
                    # if(len(element.pitches) == 2 and abs(element.pitches[0].midi - element.pitches[1].midi) in [1, 2] and element.quarterLength < 0.25):
                        # print(element)
                    if(len(element.pitches) == 2 and abs(element.pitches[0].midi - element.pitches[1].midi) in [1, 2]) and False:
                        start_trill = oS
                        bad_quarterLength = qL
                        # print(file_name + 'start_trill', start_trill)
                        fixing_trill = True
                        # don't include the bad trill chord in the data
                        continue
                    if partNo % 2 == 0:
                        bottom_note = element.pitches[0]
                        pitch = bottom_note.midi
                    else:
                        top_note = element.pitches[-1]
                        pitch = top_note.midi
                    start = oS
                    end = start + qL
                    velocity = element.volume.velocity
                    # note_data.append([fileNo, partNo, pitch, float(start), float(end), velocity])
                    note_data.append([fileNo, partNo, pitch, start, end, velocity])
                    # add the same data to df
                    df.loc[len(df)] = [fileNo, partNo, pitch, start, end, velocity]
            partNo += 1
        fileNo += 1
# df = pd.DataFrame(note_data, columns=["Filename", "Instrument", "Pitch", "Start", "End", "Velocity"])
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
from IPython.display import display

# Print the entire dataframe
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df)
# Create a bach inventions folder in the data/interim folder
folder_path = "../../data/interim/BachInventions"
# write midi files into the bach_inventions folder by reading through the df dataframe
# while reading through df, create unique Filenames named invention1.mid through invention8.mid
# for each unique Filename, create a new score object
# for each unique Instrument, create a new part object
# for each row in df, create a new note object
# append the note object to the part object
# append the part object to the score object
# write the score object to a midi file
# or effectively do the same thing
Filenames = df['Filename'].unique()

for Filename in Filenames:
    # Get the data for the current Filename
    Filename_data = df[df['Filename'] == Filename]
    Filename_data.reset_index(drop=True, inplace=True)

    # Create a new score object
    score = stream.Score()

    # Get a list of unique Instruments for the current Filename
    Instruments = Filename_data['Instrument'].unique()
    for Instrument in Instruments:
        Instrument_data = Filename_data[Filename_data['Instrument'] == Instrument]
        Instrument_data.reset_index(drop=True, inplace=True)
        # Create a new part object
        part = stream.Part()

        # Iterate through the rows of the Instrument_data dataframe
        for index, row in Instrument_data.iterrows():
            # Create a new note object with corresponding pitch and duration
            new_note = None
            if row['Pitch'] == -1:
                new_note = note.Rest()
            else:
                new_note = note.Note()
                new_note.pitch.midi = row['Pitch']
                new_note.volume.velocity = row['Velocity']
            new_note.offset = row['Start']
            # if(row['End'] - row['Start'] < 0.25):
            #     print('row', row)
            new_note.duration.quarterLength = row['End'] - row['Start']

            # Add the note object to the part object
            part.append(new_note)

        # Add the part object to the score object
        score.append(part)

    # Write the score object to a MIDI file
    score.write('midi', fp=folder_path + '/invention' + str(Filename) + '.mid')

