from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

app = Flask(__name__)

# Load the Gradient Boosting Regressor model
boost_model = pickle.load(open('Gradiantboosting_1.pkl', 'rb'))

# Function to standardize 'Speed' column
def standardize_speed(speed_column):
    scaler = StandardScaler()
    speed_standardized = scaler.fit_transform(speed_column.reshape(-1, 1))
    return speed_standardized.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Read the uploaded CSV file
        data = pd.read_csv(file)

        # Keep a copy of the original 'Speed' column
        original_speed_column = data['Speed (rpm)'].copy()

        # Standardize 'Speed' column
        if 'Speed (rpm)' in data.columns:
            data['Speed (rpm)'] = standardize_speed(data['Speed (rpm)'].values)

        # Exclude the 'exp no' column if it exists
        if 'Exp.No' in data.columns:
            data = data.drop(columns=['Exp.No'])

        # Make predictions
        predictions = boost_model.predict(data)

        # Convert input data to a DataFrame
        input_data_df = data.copy()

        # Concatenate original 'Speed' column and input data with predictions horizontally
        result_df = pd.concat([original_speed_column, input_data_df.iloc[:, 1:], pd.DataFrame({'Predicted_Roughness': predictions})],
                              axis=1)

        return render_template('index.html',
                               message='File uploaded successfully',
                               result=result_df.to_html(index=False),
                               predictions=None)

@app.route('/predict', methods=['POST'])
def predict_input():
    speed = request.form.get('speed')
    feed = request.form.get('feed')
    depth_of_cut = request.form.get('depth_of_cut')

    if speed and feed and depth_of_cut:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Speed (rpm)': [float(speed)],
            'Feed (rev/min)': [float(feed)],
            'Depth of cut (mm)': [float(depth_of_cut)]
        })

        # Standardize 'Speed' column
        input_data['Speed (rpm)'] = standardize_speed(input_data['Speed (rpm)'].values)

        # Make predictions
        prediction = boost_model.predict(input_data)
        
        return render_template('index.html', message=f'Prediction: {prediction[0]}', predictions=prediction, result=None)

    return render_template('index.html', message='Please enter all input values for prediction', predictions=None, result=None)

if __name__ == '__main__':
    app.run(debug=True)
