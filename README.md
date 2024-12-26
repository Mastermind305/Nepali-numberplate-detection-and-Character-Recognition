
# Nepali Number Plate Detection and Character Recognition

This project uses YOLOv8 and YOLOv11 models for detecting Nepali number plates and performing character recognition. The project includes a real-time detection Streamlit app, a testing Jupyter notebook for character inference, and an inference video to demonstrate the performance.

[inference](https://github.com/Mastermind305/Nepali-numberplate-detection-and-Character-Recognition/blob/main/inferences%20of%20character%20detection/detected%20output.1png.png)


## Project Structure

- `app.py`: Streamlit app that performs real-time Nepali number plate detection.
- `character_inference.ipynb`: Jupyter notebook for testing character recognition.
- `app.mp4`: Inference video demonstrating number plate and character detection.
- `requirements.txt`: List of Python dependencies required to run the Streamlit app.

## Setup Instructions

### 1. Clone the Repository
Clone the project repository to your local machine.

```bash
git clone <repository_url>
cd <project_directory>
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to isolate the dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**
  ```bash
  .env\Scriptsctivate
  ```


  ```

### 3. Install the Required Libraries
Install the dependencies listed in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries such as Streamlit, YOLO, OpenCV, etc.

### 4. Run the Streamlit App
After setting up the environment and installing the dependencies, run the Streamlit app using the following command:

```bash
streamlit run app.py
```

This will launch a web application where you can upload a video stream and perform real-time Nepali number plate detection.

### 5. Character Inference Testing
For testing character recognition, open and run the `character_inference.ipynb` Jupyter notebook. This notebook uses YOLOv11 to perform character recognition on detected number plates.

### 6. Inference Video
The `app.mp4` video demonstrates the real-time detection and character recognition process. It showcases the number plate detection and character inference using the trained models.

## Model Information

1. **Plate Detection Model**:
   - Model: YOLOv8
   - Purpose: Detects Nepali number plates in images .

2. **Character Recognition Model**:
   - Model: YOLOv11
   - Purpose: Recognizes characters from the detected number plates.
