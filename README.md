# Hiragana Character Classification Project

## Project Overview
This project utilizes the Kuzushiji-49 dataset to train a convolutional neural network model for the recognition and classification of Hiragana characters. The project includes three main components: model training, image loading, and API implementation, which allows for model prediction and results validation.

## Project Directory Structure

```plaintext
project-root
├── checkpoints              # Directory for saving the trained model
├── data                     # Directory for storing raw Kuzushiji-49 dataset
├── image_loader.ipynb       # Notebook to download and extract test images
├── kmnistTraing.ipynb       # Model training script
├── kuzushiji49_data.npz     # Compressed file of Kuzushiji-49 dataset
├── kuzushiji49_images       # Directory for all extracted images
├── kuzushiji49_test_images  # Directory for test images extracted from dataset
├── kuzushiji49_train_images # Directory for train images extracted from dataset
├── __pycache__              # Python cache directory
├── README.md                # Project documentation
└── webapi.py                # Web API implementation
```

## File Descriptions

- **kmnistTraing.ipynb**: Contains code for loading data, defining the model, training, and saving the model. After running this notebook, the trained model will be saved in the `checkpoints` folder.
- **image_loader.ipynb**: Responsible for downloading and extracting Kuzushiji-49 dataset images into the `kuzushiji49_images` folder and then you can randomly selecting images to test the Web API’s prediction.
- **webapi.py**: Implements a Web API using FastAPI, which receives uploaded images and returns classification predictions. The API supports batch prediction.

## Running Instructions

### 1. Environment Setup

Before running the project, ensure the following dependencies are installed:

```bash
pip install fastapi==0.88.0
pip install uvicorn==0.18.3
pip install pillow==9.3.0
pip install torch==1.12.1 torchvision==0.13.1
pip install python-multipart
```

### 2. Project Execution Steps

#### Step 1: Train the Model
Run `kmnistTraing.ipynb` in the project’s root directory to train the model. The trained model will be saved as `checkpoints/best_model1.pth`.

#### Step 2: Extract Test Images
Run `image_loader.ipynb` to download and extract the Kuzushiji-49 dataset images and store them in the `kuzushiji49_images` folder, including sample images for testing the API.

#### Step 3: Start the Web API

1. First, check if port 8000 is in use to ensure it is free. To find any processes using port 8000 (Linux or MacOS), run:

   ```bash
   lsof -i :8000
   ```

2. If the port is occupied, terminate the process with the following command:

   ```bash
   kill -9 <PID>  # Replace <PID> with the actual process ID
   ```

3. In the root directory, start the Web API with the following command:

   ```bash
   uvicorn webapi:app --reload
   ```

   Once started, the API will be available at `http://127.0.0.1:8000`.

#### Step 4: Test the Web API

In another terminal, use the following `curl` command to test the API endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path/to/your/test_image.png"
```

- `file=@path/to/your/test_image.png`: Replace with the path to the image you want to test.
- The API will return the classification prediction for the image.

## Notes

1. **Port Check**: Ensure port 8000 is available before starting the API.
2. **Execution Order**: Run `kmnistTraing.ipynb` to generate the trained model, then `image_loader.ipynb` to extract test images, and finally start `webapi.py` to run the API.
3. **Batch Testing**: The `webapi.py` script supports batch image upload and prediction. Refer to FastAPI documentation for further testing details.

## Future Improvements

- **Pending Items**: Due to time constraints, some model parameter tuning has not been fully explored.
- **Future Enhancements**: In future versions, deeper network architectures and optimized hyperparameter configurations could be introduced to further improve model accuracy.
