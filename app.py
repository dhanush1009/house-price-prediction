import gradio as gr
import joblib
import os

# Load model with error handling
model_path = "house_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure house_model.pkl is in the working directory.")

model = joblib.load(model_path)

def predict_house_price(msa, median_age, avg_rooms, avg_bedrooms, population, households, median_income, latitude, longitude):
    """
    Predict house price based on input features.
    
    Features:
    - msa: Median square area
    - median_age: Median age of houses
    - avg_rooms: Average number of rooms
    - avg_bedrooms: Average number of bedrooms
    - population: Population in the area
    - households: Number of households
    - median_income: Median income
    - latitude: Latitude coordinate
    - longitude: Longitude coordinate
    """
    input_data = [msa, median_age, avg_rooms, avg_bedrooms, population, households, median_income, latitude, longitude]
    prediction = model.predict([input_data])
    return float(prediction[0])

# Create Gradio interface
interface = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Number(label="MSA (Median Square Area)", value=8.3252),
        gr.Number(label="Median Age", value=41.0),
        gr.Number(label="Average Rooms", value=6.98),
        gr.Number(label="Average Bedrooms", value=1.02),
        gr.Number(label="Population", value=322),
        gr.Number(label="Households", value=2.55),
        gr.Number(label="Median Income", value=37.88),
        gr.Number(label="Latitude", value=-122.23),
        gr.Number(label="Longitude", value=-122.23),
    ],
    outputs=gr.Number(label="Predicted Price"),
    title="House Price Prediction Model",
    description="Predict house prices using machine learning",
    examples=[
        [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23, -122.23],
    ]
)

if __name__ == "__main__":
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
