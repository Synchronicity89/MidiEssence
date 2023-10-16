
from numpy import int64
import pandas as pd

def CompareDataframes(df, df_decoded):
    if df.equals(df_decoded) == False:
        print("The decoded dataframe is different from the original dataframe.")
    # show the differences between the two dataframes
        print(df.compare(df_decoded))

    # show the differences between the two dataframes without using compare, but by iterating over the rows
        for index, row in df.iterrows():
            if row.equals(df_decoded.iloc[index]) == False:
                print("The decoded dataframe row {} is different from the original dataframe row {}.".format(index, index))
                print(row)
                print(df_decoded.iloc[index])
                print()
    else:
        print("The decoded dataframe is the same as the original dataframe.")

def encode_diffs(df):
    df_copy = df.copy()
    # create an empty dataframe called df_encoded with the same columns as the input dataframe
    df_encoded = pd.DataFrame(columns=df.columns)

    # Get a list of unique Filenames
    Filenames = df['Filename'].unique()

    for Filename in Filenames:
        # Get the data for the current Filename
        Filename_data = df[df['Filename'] == Filename]
        Filename_data.reset_index(drop=True, inplace=True)

        # Get a list of unique Instruments for the current Filename
        Instruments = Filename_data['Instrument'].unique()

        for Instrument in Instruments:
            Instrument_data = Filename_data[Filename_data['Instrument'] == Instrument]
            Instrument_data.reset_index(drop=True, inplace=True)

            # Get the diffs of the Pitch, Start, End, and Velocity values
            Instrument_data.loc[:,"Pitch_diff"] = Instrument_data["Pitch"].diff()
            Instrument_data.loc[:,"Start_diff"] = Instrument_data["Start"].diff()
            Instrument_data.loc[:,"End_diff"] = Instrument_data["End"].diff()
            Instrument_data.loc[:,"Velocity_diff"] = Instrument_data["Velocity"].diff()

            # Replace the NaN values in the diffs columns with the original data values from the Pitch, Start, End, and Velocity columns, using the first row of the Instrument_data dataframe
            Instrument_data.loc[0, "Pitch_diff"] = Instrument_data.loc[0,"Pitch"]
            Instrument_data.loc[0, "Start_diff"] = Instrument_data.loc[0,"Start"]
            Instrument_data.loc[0, "End_diff"] = Instrument_data.loc[0,"End"]
            Instrument_data.loc[0, "Velocity_diff"] = Instrument_data.loc[0,"Velocity"]

            # Append the Instrument_data rows to the df_encoded dataframe
            df_encoded = pd.concat([df_encoded, Instrument_data], ignore_index=True)

    # Drop the original Pitch, Start, End, and Velocity columns
    df_encoded.drop(columns=["Pitch", "Start", "End", "Velocity"], inplace=True)

    return df_encoded


def decode_diffs(df_encoded):
    # Create a copy of the input dataframe
    df_decoded = pd.DataFrame(columns=[col.replace('_diff', '') for col in df_encoded.columns])

    # Get a list of unique Filenames
    Filenames = df_encoded['Filename'].unique()

    for Filename in Filenames:
        # Get the data for the current Filename
        Filename_data = df_encoded[df_encoded['Filename'] == Filename]
        Filename_data.reset_index(drop=True, inplace=True)

        # Get a list of unique Instruments for the current Filename
        Instruments = Filename_data['Instrument'].unique()

        for Instrument in Instruments:
            Instrument_data = Filename_data[Filename_data['Instrument'] == Instrument]
            Instrument_data.reset_index(drop=True, inplace=True)

            # Get the offset values for each column
            offset = Instrument_data.iloc[0:1].copy()

            # drop the first row of Instrument_data
            Instrument_data.loc[0, ["Pitch_diff", "Start_diff", "End_diff", "Velocity_diff"]] = 0

            # Add the offset values to the diffs
            Instrument_data.loc[:,"Pitch"] = Instrument_data["Pitch_diff"].cumsum() + offset.loc[0, "Pitch_diff"]
            Instrument_data.loc[:,"Start"] = Instrument_data["Start_diff"].cumsum() + offset.loc[0, "Start_diff"]
            Instrument_data.loc[:,"End"] = Instrument_data["End_diff"].cumsum() + offset.loc[0, "End_diff"]
            Instrument_data.loc[:,"Velocity"] = Instrument_data["Velocity_diff"].cumsum() + offset.loc[0, "Velocity_diff"]

            # Append the Instrument_data rows to the df_decoded dataframe
            df_decoded = pd.concat([df_decoded, Instrument_data], ignore_index=True)

    # Reorder the columns to match the original dataframe
    df_decoded = df_decoded[['Filename', 'Instrument', 'Pitch', 'Start', 'End', 'Velocity']]

    # Convert the Pitch and Velocity columns to integers
    df_decoded["Pitch"] = df_decoded["Pitch"].astype(int64)
    df_decoded["Velocity"] = df_decoded["Velocity"].astype(int64)

    # Convert the Filename and Instrument columns to integers
    df_decoded["Filename"] = df_decoded["Filename"].astype(int64)
    df_decoded["Instrument"] = df_decoded["Instrument"].astype(int64)

    
    return df_decoded

# read the note_data.pkl file into a dataframe
df = pd.read_pickle("../../data/interim/note_data.pkl")

# encode the dataframe
df_encoded = encode_diffs(df)

# write the encoded dataframe to a pickle file called note_data_diff_encoded.pkl in the interim folder
df_encoded.to_pickle("../../data/interim/note_data_diff_encoded.pkl")

# decode the dataframe
df_decoded = decode_diffs(df_encoded)
print("df:")
print()
print(df)
print("df_encoded:")
print()
print(df_encoded)
print("df_decoded:")
print()
print(df_decoded)
# check that the decoded dataframe is the same as the original dataframe

CompareDataframes(df, df_decoded)

print(df_decoded.info())
print(df.info())

