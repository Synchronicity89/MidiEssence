import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load the sample_note_data.csv file into a dataframe
df = pd.read_csv('../../data/interim/sample_note_data.csv')

# Get the unique filenames and instruments in the dataframe
filenames = df['Filename'].unique()
instruments = df['Instrument'].unique()

# Iterate through the filenames and create a piano roll plot for each filename
for i, filename in enumerate(filenames):
    # Set the size of the figure
    fig, axs = plt.subplots(len(instruments), 1, figsize=(20, 20), sharex=True)

    # Iterate through the instruments and create a piano roll plot for each instrument
    for j, instrument in enumerate(instruments):
        # Get the note data for the current filename and instrument
        data = df[(df['Filename'] == filename) & (df['Instrument'] == instrument)]

        # Get the start and end times of the note data
        start_time = int(data['Start'].min())
        end_time = int(data['End'].max())

        # Get the start and end times of the note data
        start_time = int(0.0)
        end_time = int(data['End'].max())

        # Create a 2D numpy array to represent the piano roll
        piano_roll = np.zeros((128, int(end_time - start_time)), dtype=np.int16)

        # Iterate through the notes in the data and set the appropriate cells in the piano roll
        for index, row in data.iterrows():
            pitch = int(row['Pitch'])
            start = int(row['Start'] - start_time)
            end = int(row['End'] - start_time)
            velocity = int(row['Velocity'])
            piano_roll[pitch, int(start):int(end)] = velocity

        # Create the piano roll plot
        axs[j].imshow(piano_roll, aspect='auto', cmap='gray_r', origin='lower')

        # Set the title and axis labels for the plot
        axs[j].set_title(f"{filename} - Instrument {instrument}")
        axs[j].set_xlabel('Time (ticks)')
        axs[j].set_ylabel('Pitch')

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Show the plot    
    plt.show()

