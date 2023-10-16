from matplotlib import pyplot as plt
import pandas as pd


data = pd.read_csv('../../data/interim/note_data_diff_squish_encoded.pkl')

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

    # Add vertical lines for the pattern matches
    for match in unique_file_PitchDiff_patterns:
        ax.axvline(x=match, color='red', linestyle='--')

    # Add labels and legend for the current plot
    ax.set_xlabel('Start ({})'.format(Filename))
    ax.set_ylabel('Pitch ({})'.format(Filename))
    ax.legend()

    # Show the plot for the current Filename
    plt.show()