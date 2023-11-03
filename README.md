MidiEssence
==============================


MidiEssence is incomplete.  

The purpose of MidiEssence is to create an encoding "essence" of a piece of symbolic music in the MIDI format.  The essence or encoding, captures patterns of melody, rhythm, and velocity (note volume), and also contains enough information so that the decoder can generate a MIDI file very similar to the original MIDI file.  It may lose some information such as precise velocity information, but pitch and rhythm should be preserved.

Current state of project:

- manually written code compresses Bach Invention No. 1. into an encoding. The encoding can be tweaked and a new invention generated with different pitches than the original.

- How to run it and see it work: 
-- src/essential_end2end.py: 
--- generates pitch-tweaked midi files in data/interim/BachInventions.  The tweaked ones' filenames start with 01 02 etc

- src/features/build_midiessence.py  does most of the work of converting between chromatic to diatonic and back

- src/data/make_midis.py writes tweaked midis

- src/data/make_dataset.py reads most of the Bach 2 part Invention MIDI files and extracts the data into a DataFrame

- src\visualization\visualize.py visualizes the first 100 notes of each Instrument part, some of which run longer than the other part.  Here the parts are the right hand part and the left hand part

- src\features\build_features.py has a function that encodes the changes as differences in the time series of Pitch, Start, End, and Velocity data.  It also has an decoder to restore the symbolic music exactly as it was

Future scope:

The data that the visualization runs on is a CSV file.  This CSV is short enough to paste into a GPT 4 prompt, so one could ask GPT 4 how to find the patterns and encode the symbolic music, thereby capturing the essence, however this seems to be too big of a job for the technology right now, even if carefully prompted.  
The next steps might be to polish the hand-coded encoding so that it becomes more of a language.  Also the rhythm and velociy need a similar treatment.  The data might become three dimensional at that point, in which case numpy might be employed.  We are already at a stage where a lot of training data could already be generated that could be used to train a deep learning model.




This is a project that conforms to the DataAlchemy project format

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
