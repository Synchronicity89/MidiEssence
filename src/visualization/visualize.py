import pandas as pd
import matplotlib.pyplot as plt
from music21 import *
import os

# Load the data
data = pd.read_csv('../../data/interim/sample_note_data.csv')

# Get a list of unique Filenames
Filenames = data['Filename'].unique()

# Iterate over each Filename and create a plot with a separate line for each Instrument
for Filename in Filenames:
    # Get the data for the current Filename
    Filename_data = data[data['Filename'] == Filename]

    # Get a list of unique Instruments for the current Filename
    Instruments = Filename_data['Instrument'].unique()

    # Create a plot with a separate line for each Instrument
    fig, ax = plt.subplots()
    for Instrument in Instruments:
        Instrument_data = Filename_data[Filename_data['Instrument'] == Instrument]
        ax.plot(Instrument_data['Start'], Instrument_data['Pitch'], label=Instrument)

    # Add labels and legend for the current plot
    ax.set_xlabel('Start ({})'.format(Filename))
    ax.set_ylabel('Pitch ({})'.format(Filename))
    ax.legend()

    # Show the plot for the current Filename
    plt.show()

folder_path = "../../data/raw/BachInventions"

file_list = os.listdir(folder_path)
for file_name in file_list:
    if file_name.endswith(".mid"):
        midi_file_path = os.path.join(folder_path, file_name)
    
    # Load the MIDI file
    score = converter.parse(midi_file_path)
    # Get all elements in the score up to the offset of the 50th note
    start = score.flat.notes[0].offset
    stop=score.flat.notes[50].offset
    print('start', start)
    print('stop', stop)
    
    # Get all elements in the score up to the offset of the 50th note
    elements = score.flat.getElementsByOffset(start, stop )
    print('len(elements)', len(elements))


    # Create a new score object from the filtered elements
    filtered_score = stream.Score()
    for element in elements:
        filtered_score.append(element)

    # Create a piano roll view of the filtered score
    if len(filtered_score) > 0:
        piano_roll = filtered_score.flat.plot('pianoroll')

        # Show the piano roll view
        piano_roll.show()

    break