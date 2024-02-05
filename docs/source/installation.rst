Installation
============

This section covers the steps required to install and set up the Sign Language Translator project on your machine. Follow these instructions to prepare your environment for running the application.

Prerequisites
-------------

Before you begin, ensure you have the following installed on your system:

- Python 3.8 or newer
- DVC (Data Version Control)
- Necessary Python libraries

Ensure you also have at least 30GB of free disk space for feature extraction matrices and ideally 16+ GB of RAM for smoother processing.

Installing Python Libraries
----------------------------

Install all required Python libraries by running the following command in your terminal:

.. code-block:: bash

    pip install -r requirements.txt

Data Management with DVC
------------------------

Data Version Control (DVC) is used for managing and versioning large data files. To get started with DVC:

1. Install DVC following the official [DVC documentation](https://dvc.org/doc/install).
2. Retrieve the project data by running:

.. code-block:: bash

    dvc pull

3. If you encounter any issues with `dvc pull` or do not have access to the DVC remote storage, you can download the data directly from the provided data source (e.g., Kaggle).

Model Setup and Training
------------------------

To set up and train the model, follow these steps:

1. Confirm the `data` folder contains necessary video files and `WLASL_v0.3.json`.
2. Run the preprocessing script to prepare the video data:

.. code-block:: bash

    python model/preprocessing.py

3. To train the model, execute:

.. code-block:: bash

    python model/train_model.py

This script will preprocess the data, perform incremental learning, and save the trained model and associated files in the `model/` directory.

Running the Translator
----------------------

Once installation and setup are complete, you can run the translator using the GUI. Start the application with:

.. code-block:: bash

    python asl_video_translator_gui.py

Follow the instructions in the "Usage" section to load sign language videos and receive text translations.



