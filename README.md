# Wildly Backend

This project provides a Flask-based API that classifies dog breeds using a pre-trained machine learning model.

## Requirements

- Python 3.10.11

## Folder Structure

```
models/
  ├── classes.json                # JSON file containing breed class names
  ├── dog_classifier_breed.keras  # Trained model file
app.py                            # Main application
requirements.txt                  # Project dependencies
utils.py                          # Utility functions
```

## Features

- **Dog breed prediction**: Accepts an image file and returns the predicted breed along with the confidence level.
- **Top 3 predictions**: Provides the top 3 breed predictions with their confidence levels.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/1stZorua/wildly-backend.git
   ```

2. Navigate to the project folder:

   ```bash
   cd wildly-backend
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the Flask app:

   ```bash
   python app.py
   ```

6. The app will be available at `http://127.0.0.1:5000/`.

## License

This project is licensed under the MIT License.
