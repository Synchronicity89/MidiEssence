import mido
import pandas as pd

def process_midi_data(df, df_meta, midi_name, list_filenames = None, fix_pitch = False):
    df0 = pd.DataFrame(columns=['Filename', 'Instrument', 'Pitch', 'Start', 'End', 'Velocity'])
    df0 = df0.astype({'Filename': 'int64', 'Instrument': 'int64', 'Pitch': 'int64', 'Start': 'float64', 'End': 'float64', 'Velocity': 'int64'})

    # # Load the parsed MIDI data from the dataframe
    # df = pd.read_pickle("../../data/interim/note_data.pkl")

    # # Load the track metadata from the pickle file
    # df_meta = pd.read_pickle("../../data/interim/track_metadata.pkl")

    # Get unique filenames
    unique_filenames = df['Filename'].unique()

    # Iterate through the unique filenames
    for filename in unique_filenames:
        if list_filenames is not None:
            if filename not in list_filenames:
                continue        

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
                pitch = row['Pitch']
                if fix_pitch:
                    while pitch < 0:
                        pitch += 12
                    while pitch > 127:
                        pitch -= 12
                note_on = mido.Message('note_on', note=pitch, velocity=row['Velocity'], time=int(row['Start']))
                track.append(note_on)

                # Create a note off event
                note_off = mido.Message('note_off', note=pitch, velocity=row['Velocity'], time=int(row['End'] - row['Start']))
                track.append(note_off)

        # Save the recreated MIDI file for the current filename
        # check if filename ends with .mid, if not, add it.  Check upper and lower case
        if filename[-4:].lower() != '.mid':
            filename = filename + '.mid'
        midi_file.save(f"../../data/interim/BachInventions/{midi_name + filename}")
