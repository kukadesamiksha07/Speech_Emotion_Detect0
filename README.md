🎙️ Speech Emotion Detector

An AI-powered Speech Emotion Detection system that analyzes audio features to identify emotions expressed in speech using Machine Learning.

📌 About the Project

Speech Emotion Detector is a Python-based project designed to recognize human emotions from speech recordings. It uses audio feature extraction and a trained machine learning model to classify emotional characteristics in speech.

The project includes a Python backend, model training scripts, and a web interface for interacting with the application.

✨ Features

🎧 Speech Emotion Recognition — Analyze speech audio to predict emotions.

🤖 Machine Learning — Use a trained model for emotion classification.

🔊 Audio Feature Processing — Work with extracted audio features.

🌐 Web Interface — Interact with the application through a browser.

🧠 Model Training — Train the emotion recognition model using the training script.

⚡ Python Backend — Process application requests and generate predictions.

🛠️ Technologies Used

Python

Flask — Web application framework

Machine Learning — Model training and prediction

JavaScript — Frontend interactions

HTML & CSS — Web interface

NumPy — Numerical processing and feature-cache files

Additional libraries may be required depending on the implementation in train_model.py and app.py. See requirements.txt.

📂 Project Structure

Speech_emotion_Dtector/
│
├── app.py
├── app.js
├── train_model.py
├── index.html
├── requirements.txt
├── README.md
├── features_cache.npz
└── robust_features_cache.npz

This structure reflects the files currently identified in the repository. Add any other directories or model files that are part of your actual project.

⚙️ Installation and Setup

1. Clone the Repository

git clone https://github.com/kukadesamiksha07/Speech_emotion_Dtector.git

2. Navigate to the Project Folder

cd Speech_emotion_Dtector

3. Create a Virtual Environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

macOS/Linux

source venv/bin/activate

4. Install Dependencies

pip install -r requirements.txt

🚀 Running the Application

Start the Flask application:

python app.py

Open the local URL displayed in your terminal in your browser. For a typical Flask development configuration, this may be http://127.0.0.1:5000/.

Note: The application may require trained model files, feature caches, or additional assets. Ensure these are available before running it.

🧠 Model Training

The project includes train_model.py, which contains the model-training workflow.

To run it:

python train_model.py

Check the script for the required dataset location, supported audio formats, feature-extraction process, and expected output files before training.

📊 How It Works

The general speech emotion recognition workflow is:

Audio Input — Provide a speech recording to the application.

Feature Extraction — Extract relevant information from the audio.

Model Processing — Pass the extracted features to the trained model.

Emotion Prediction — Generate the predicted emotion.

Result Display — Display the prediction through the web interface.

The exact emotions supported and the specific feature-extraction and classification methods depend on the implementation in the project files.

📁 Dataset

The dataset used for training should contain speech recordings with emotion labels.

Before distributing a dataset, check its license and usage conditions. Large datasets do not generally need to be uploaded directly to this repository. Instead, document the dataset source and explain how to prepare it.

Dataset used: Update this section with the actual dataset name and source.

🔮 Future Improvements

Improve emotion classification performance.

Support additional audio formats.

Enhance the user interface and user experience.

Add visualizations for predicted emotions and confidence scores.

Improve audio preprocessing and feature extraction.

Evaluate the model using suitable classification metrics.

⚠️ Limitations

Predictions may be affected by recording quality, background noise, and speaker variation.

Model performance depends on the training data and evaluation methodology.

Vocal emotion predictions are estimates and may not reflect a person's actual emotional state.

🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Fork the repository.

Create a new branch.

Make your changes.

Submit a pull request describing your improvements.

📜 License

No license has been specified yet. Add an appropriate license after deciding how you want others to use, modify, and distribute this project.

👩‍💻 Author

Samiksha Kukade

GitHub: @kukadesamiksha07
