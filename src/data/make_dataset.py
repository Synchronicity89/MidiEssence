import os
import mido
import pandas as pd
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


# Define the path to the directory containing the MIDI files
midi_dir_path = '../../data/raw/BachInventions'

# Create an empty list to store note data
note_data = []
track_metadata = []

# Iterate through the MIDI files in the directory
for filename in os.listdir(midi_dir_path):
    # Define the path to the current MIDI file
    midi_file_path = os.path.join(midi_dir_path, filename)

    # Use mido to parse the MIDI file
    midi_file = mido.MidiFile(midi_file_path)

    # Iterate through the tracks in the MIDI file and extract note data
    for i, track in enumerate(midi_file.tracks):
        # add the file related track metadata to track_metadata, using the filename and instrument as keys for relational integrity
        track_metadata.append({'Filename': filename, 'Track': i, 'MetaMessages': [], 'Type': midi_file.type, 'TicksPerBeat': midi_file.ticks_per_beat, 'Tempo': []})
        
        for meta in track:
            if meta.is_meta:
                track_metadata[-1]['MetaMessages'].append(meta.dict())
        tempo = []
        note_on_events = []
        for event in track:
            if event.type == 'set_tempo':
                tempo += [event.time, event.tempo]
            if event.type == 'note_on':
                note_on_events.append(event)
            elif event.type == 'note_off':
                for note_on_event in note_on_events:
                    if note_on_event.note == event.note and note_on_event.channel == event.channel:
                        pitch = note_on_event.note
                        velocity = note_on_event.velocity
                        start_time = note_on_event.time
                        end_time = start_time + event.time
                        channel = note_on_event.channel
                        note_data.append([filename, i, pitch, start_time, end_time, velocity])
                        # do any cleanup required, e.g. remove the note_on_event from the list
                        note_on_events.remove(note_on_event)
                        break
        # if tempo is not empty, append that info to track_metadata
        if tempo:
            track_metadata[-1]['Tempo'].append(tempo)

        

# Create a pandas dataframe from the note data
df = pd.DataFrame(note_data, columns=['Filename', 'Instrument', 'Pitch', 'Start', 'End', 'Velocity'])
df_meta = pd.DataFrame(track_metadata, columns=['Filename', 'Track', 'MetaMessages', 'Type', 'TicksPerBeat', 'Tempo'])
df.to_pickle('../../data/interim/note_data.pkl')
df_meta.to_pickle('../../data/interim/track_metadata.pkl')

# create an empty dataframe that has the same columns as df
df0 = pd.DataFrame(columns=['Filename', 'Instrument', 'Pitch', 'Start', 'End', 'Velocity'])
# set the types of the columns the same way as the types of the columns in df
df0 = df0.astype({'Filename': 'int64', 'Instrument': 'int64', 'Pitch': 'int64', 'Start': 'float64', 'End': 'float64', 'Velocity': 'int64'})

import mido
import pandas as pd

# Load the parsed MIDI data from the dataframe
df = pd.read_pickle("../../data/interim/note_data.pkl")

# Load the track metadata from the pickle file
df_meta = pd.read_pickle("../../data/interim/track_metadata.pkl")

# Get unique filenames
unique_filenames = df['Filename'].unique()

# Iterate through the unique filenames
for filename in unique_filenames:
    # # Create a dictionary to store the tracks for each filename
    # filename_tracks = {}

    # Create a new MidiFile object
    midi_file = mido.MidiFile()

    # Get the data for the current filename
    filename_data = df[df['Filename'] == filename]

    # Get unique instruments for the current filename
    unique_instruments = filename_data['Instrument'].unique()
    added_file_info = False
    # create a track 0 tempo map, which is not for note events but just for the tempo map
    track0 = midi_file.add_track()
    # add the track meta data into the track from the associated filename and instrument
    track_meta = df_meta[(df_meta['Filename'] == filename) & (df_meta['Track'] == 0)]
    for meta in track_meta['MetaMessages'].values[0]:
        track0.append(mido.MetaMessage.from_dict(meta))
        # Set the ticks per quarter note to the same value as the original MIDI files
        track0.ticks_per_beat = track_meta['TicksPerBeat'].values[0]
        if not added_file_info:
            midi_file.ticks_per_beat = track_meta['TicksPerBeat'].values[0]
            midi_file.type = track_meta['Type'].values[0]
            added_file_info = True

    # Iterate through the unique instruments for the current filename
    for instrument in unique_instruments:
        # Get the data for the current instrument
        instrument_data = filename_data[filename_data['Instrument'] == instrument]

        # concatenate only the first 100 rows of instrument_data to df0, making sure to keep what is already there
        df0 = pd.concat([df0, instrument_data.head(100)], ignore_index=True)

        track = midi_file.add_track()
        # add the track meta data into the track from the associated filename and instrument
        track_meta = df_meta[(df_meta['Filename'] == filename) & (df_meta['Track'] == instrument)]
        for meta in track_meta['MetaMessages'].values[0]:
            track.append(mido.MetaMessage.from_dict(meta))
            # Set the ticks per quarter note to the same value as the original MIDI files
            track.ticks_per_beat = track_meta['TicksPerBeat'].values[0]
            if not added_file_info:
                midi_file.ticks_per_beat = track_meta['TicksPerBeat'].values[0]
                midi_file.type = track_meta['Type'].values[0]
                added_file_info = True

        # Set the instrument for the track
        program_change = mido.Message('program_change', program=instrument, time=0)
        track.append(program_change)

        # Iterate through the notes, and add them to track
        for index, row in instrument_data.iterrows():
            # Create a note on event
            note_on = mido.Message('note_on', note=row['Pitch'], velocity=row['Velocity'], time=int(row['Start']))
            track.append(note_on)

            # Create a note off event
            note_off = mido.Message('note_off', note=row['Pitch'], velocity=row['Velocity'], time=int(row['End'] - row['Start']))
            track.append(note_off)

    # Save the recreated MIDI file for the current filename
    midi_file.save(f"../../data/interim/BachInventions/{filename}")

# number the filenames in df0 and replace the filenames in df0 with the new numbers
df0['Filename'] = df0.groupby('Filename').ngroup()
# change the type of the Filename column to int64
df0 = df0.astype({'Filename': 'int64'})
# rename the Start column to Start_mido in df0
df0 = df0.rename(columns={'Start': 'Start_mido'})
# create a new float64 column called Start full of zeros at first
df0['Start'] = 0.0
# do the same for End
df0 = df0.rename(columns={'End': 'End_mido'})
# create a new float64 column called Start full of zeros at first
df0['End'] = 0.0
# in df0, arrange the columns from left to right like this: Filename,Instrument,Pitch,Start,Start_mido,End,Velocity
df0 = df0[['Filename', 'Instrument', 'Pitch', 'Start', 'Start_mido', 'End', 'End_mido', 'Velocity']]


# # use accumulate to fill in the Start and End columns in df0 from the Start_mido and End_mido columns, respectively
# # import accumulate from itertools
# from itertools import accumulate
# # create a list of the Start_mido values in df0
# Start_mido_list = df0['Start_mido'].tolist()
# # create a list of the End_mido values in df0
# End_mido_list = df0['End_mido'].tolist()
# # create a list of the Start values in df0
# Start_list = df0['Start'].tolist()
# # create a list of the End values in df0
# End_list = df0['End'].tolist()
# # use accumulate to fill in the Start and End columns in df0 from the Start_mido and End_mido columns, respectively
# df0['Start'] = list(accumulate(Start_mido_list))
# df0['End'] = list(accumulate(End_mido_list))

curr_filename = 0
curr_instrument = 0
# iterate through df0 rows one by one creating a variable row to represent the current row, 
# and row_prev the row before it, I'll sum up Start_mido and End_mido to get the new Start values
for index, row in df0.iterrows():
    # if this is the first row, then set Start to Start_mido, because we don't care about the previous row because there isn't one
    if index == 0 or (row['Filename'] != curr_filename or row['Instrument'] != curr_instrument):
        # set curr_filename and curr_instrument to the current row's Filename and Instrument
        curr_filename = row['Filename']
        curr_instrument = row['Instrument']

        df0.at[index, 'Start'] = df0.at[index, 'Start_mido']
        # do the same for End
        df0.at[index, 'End'] = df0.at[index, 'End_mido']

    else:
        # get the previous row
        row_prev = df0.iloc[index - 1]
        # add to the Start value of the current row, which should be 0.0 the Start of the previous row
        df0.at[index, 'Start'] += row_prev['Start']
        # add the End_mido of the previous row - the Start_mido of the previous row to the Start value of the current row
        df0.at[index, 'Start'] += row_prev['End_mido'] - row_prev['Start_mido']
        # do the same for End
        df0.at[index, 'End'] += row_prev['End']
        df0.at[index, 'End'] += row_prev['End_mido'] - row_prev['Start_mido']






    # # if this is the first row, then set Start to 0.0
    # if index == 0:
    #     df0.at[index, 'Start'] = df0.at[index, 'Start_mido']
    #     # df0.at[index, 'Start'] = 0
    # else:
    #     # get the previous row
    #     row_prev = df0.iloc[index - 1]
    #     # set the Start value of the current row to the sum of the Start_mido of the previous row and the End_mido of the previous row
    #     df0.at[index, 'Start'] = row_prev['Start_mido'] + row_prev['End'] + 0.0 if index == 1 else row_prev['Start']





# Save df0 as a CSV file
df0.to_csv("../../data/interim/sample_note_data.csv", index=False)


