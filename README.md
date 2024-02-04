# Sign Language Translator

## Overview
This Sign Language Translator is an innovative tool designed to break down communication barriers by translating sign language into text in real-time. Utilizing advanced machine learning techniques, this application offers an accessible and user-friendly interface this tool provides instantaneous translations of sign language gestures from video input.

## Features
- *Data Version Control (DVC)*: Easily manage and version large data files, facilitating collaboration and streamlining the data pipeline.
- *Customizable Model Building*: Utilize model/preprocessing.py to construct and train models, improving the accuracy of translations.
- *Real-Time Translation*: With asl_video_translator_gui.py, load sign language videos and receive text translations instantly.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or newer
- DVC
- Necessary Python libraries: pip install -r requirements.txt

### Data Management with DVC
To get started with DVC, follow these steps:
1. Install DVC using the instructions on the [official DVC documentation](https://dvc.org/doc/install).
2. Retrieve the data using the DVC command: dvc pull.
3. If you do not posess password, you can download the data directly from [kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed).

### Model Setup and Training
Follow these steps to set up and train the model:
1. Confirm you have at least 30GB of free disk space for feature extraction matrices from videos.
2. Ensure a substantial amount of RAM is available (ideally 16+ GB for smoother processing).
3. Navigate to the data folder and ensure it contains the necessary video files and the WLASL_v0.3.json file.
4. Run the preprocessing script:
   ```bash
   python model/preprocessing.py
   ```
   This will preprocess the video data and extract features necessary for model training.
5. To train the model, execute:
   ```bash
   python model/train_model.py
   ```
   The script will stream the data, perform incremental learning, and save the model and associated files in the model/ directory.

Please note that you should adjust the paths, commands, and details according to your project's actual configuration and file structure. The above is just a template and may need to be modified to fit your specific implementation.

### Running the Translator
To translate sign language videos into text:
1. Launch the GUI with the command:
   ```bash
   python asl_video_translator_gui.py
   ```
3. Use the interface to load sign language video files.
4. The translation output will be displayed within the GUI.

## Contribution
Your contributions are welcome! Together, we can enhance the Sign Language Translator and make the world more accessible for the deaf and hard-of-hearing community.
