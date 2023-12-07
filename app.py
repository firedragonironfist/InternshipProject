from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/static')

model = pickle.load(open('model.pkl', 'rb'))

# Initialize an empty set to store unique 'fav_sub' values
fav_sub_values = set()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    olympiad_participation = int(request.form['olympiad_participation'])
    school = int(request.form['school'])
    fav_sub = request.form['fav_sub']
    projects = int(request.form['projects'])
    grasp_pow = int(request.form['grasp_pow'])
    time_sprt = int(request.form['time_sprt'])
    medals = int(request.form['medals'])
    career_sprt = int(request.form['career_sprt'])
    act_sprt = int(request.form['act_sprt'])
    fant_arts = int(request.form['fant_arts'])
    won_arts = int(request.form['won_arts'])
    time_art = int(request.form['time_art'])

    # Add the user input 'fav_sub' value to the set of unique values
    fav_sub_values.add(fav_sub)

    # Create a label encoder for the current set of unique 'fav_sub' values
    fav_sub_encoder = LabelEncoder()
    fav_sub_encoder.fit(list(fav_sub_values))

    # Transform 'fav_sub' using the current label encoder
    fav_sub_encoded = fav_sub_encoder.transform([fav_sub])

    input_data = np.array([[olympiad_participation, school, fav_sub_encoded[0], projects, grasp_pow, time_sprt, medals, career_sprt, act_sprt, fant_arts, won_arts, time_art]])
    print(f"Received input data: {input_data}")
    prediction = model.predict(input_data)

    # Determine the predicted hobby
    predicted_hobby = "Unable to Predict"
    if prediction[0] == 0:
        predicted_hobby = "Academics"
    elif prediction[0] == 1:
        predicted_hobby = "Sports"
    elif prediction[0] == 2:
        predicted_hobby = "Arts"

    return render_template('index.html', prediction=f'Predicted Hobby: {predicted_hobby}')

if __name__ == '__main__':
    app.run(debug=True)
