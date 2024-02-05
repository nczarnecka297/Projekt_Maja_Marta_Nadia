API Reference
=============

This section provides detailed reference documentation for the API of the Sign Language Translator project. It covers the preprocessing of video data, feature extraction, and the translation process, offering insights into the functions, classes, and methods designed for translating sign language into text.

Preprocessing Module
--------------------

The preprocessing module includes functions for preparing video data by reducing noise, enhancing contrast, and detecting edges.

.. autofunction:: preprocessing.preprocessing

   Preprocesses a video file by applying grayscale conversion, CLAHE, Gaussian blur, median blur and Canny edge detection.

Feature Extraction Module
-------------------------

The feature extraction module details the methods used to extract meaningful features from processed video frames, which are crucial for the translation model.

.. autofunction:: preprocessing.extract_features

   Extracts features from a list of processed video frames using the ORB algorithm.

Translation Module
------------------

The translation module focuses on loading video files, processing them, and displaying the translated text in the GUI.

.. autofunction:: asl_video_translator_gui.browse_file

   Opens a file dialog to select an MP4 video file, processes it, and displays the translation.

Model Training and Evaluation
-----------------------------

Includes documentation on the scripts and methods used for training the sign language translation model and evaluating its performance.

.. autofunction:: train_model.split_batch_features_labels
.. autofunction:: train_model.process_batch
.. autofunction:: train_model.transform_features

   Includes information on the model training process, including incremental learning and performance metrics.

