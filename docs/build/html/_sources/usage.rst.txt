Usage
=====

The ASL Video Translator provides an easy-to-use graphical interface for translating American Sign Language from video files into text. Follow the steps below to use the translator.

Starting the GUI
----------------

To start the GUI, navigate to the project directory in your terminal and run the following command:

.. code-block:: bash

    python asl_video_translator_gui.py

This will open the ASL Video Translator window.

Selecting a Video
-----------------

1. Click on the "Choose video file ('\*.mp4')" button to open a file dialog.
2. Navigate to the directory containing your MP4 video file.
3. Select the video file you wish to translate and click "Open".

The selected video file name will be displayed in the GUI, indicating that it is ready for translation.

Translating a Video
-------------------

Once a video is selected, the translation process will start automatically. You can view the translation progress in the status bar at the bottom of the GUI. When the translation is complete, the translated text will be displayed in the GUI.

Error Handling
--------------

If an error occurs during the video selection or translation process, an error message will be displayed in the GUI. Common errors include selecting an unsupported file format or issues with video processing. Ensure your video file is in MP4 format and try again.

Closing the Application
-----------------------

To close the ASL Video Translator, simply close the window or press the "Exit" button if available.
