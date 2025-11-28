# House Price Prediction Project

A machine learning project that predicts house prices using both FastAPI and Gradio interfaces.

## Project Structure

- `main.py` - FastAPI server for REST API predictions
- `app.py` - Gradio web interface for interactive predictions
- `house_model.pkl` - Pre-trained machine learning model
- `requirements.txt` - Python dependencies

## Features

- **REST API** (FastAPI): Fast, production-ready API endpoint
- **Web UI** (Gradio): User-friendly interactive interface
- **Pre-trained Model**: Ready-to-use housing price prediction model

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `house_model.pkl` is in the project directory

## Usage

### Option 1: Run Gradio Web Interface (Recommended for beginners)
```bash
python app.py
```
Then open your browser to `http://localhost:7860`

### Option 2: Run FastAPI Server
```bash
python main.py
```
The API will be available at `http://0.0.0.0:10000`

API endpoint: `POST /predict`

Example request:
```json
{
  "data": [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
}
```

## Model Input Features

1. **MSA** (Median Square Area) - e.g., 8.3252
2. **Median Age** - e.g., 41.0
3. **Average Rooms** - e.g., 6.98
4. **Average Bedrooms** - e.g., 1.02
5. **Population** - e.g., 322
6. **Households** - e.g., 2.55
7. **Median Income** - e.g., 37.88
8. **Latitude** - e.g., -122.23
9. **Longitude** - e.g., -122.23

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Gradio
- joblib
- scikit-learn
- Pydantic

## Notes

- The model file must be in the same directory as the scripts
- Both servers can run independently
- Gradio interface provides real-time predictions with a user-friendly UI
- FastAPI is suitable for integration with other applications

## Troubleshooting

If you get a "Model file not found" error:
1. Ensure `house_model.pkl` is in the project directory
2. Check the file name spelling matches exactly
3. Verify the file is not corrupted

## License

This project is provided as-is for educational purposes.
