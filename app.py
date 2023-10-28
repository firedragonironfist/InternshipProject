from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    olympiad_participation = int(request.form['olympiad_participation'])
    school = int(request.form['school'])
    fav_sub = int(request.form['fav_sub'])
    projects = int(request.form['projects'])
    grasp_pow = int(request.form['grasp_pow'])
    time_sprt = int(request.form['time_sprt'])
    medals = int(request.form['medals'])
    career_sprt = int(request.form['career_sprt'])
    act_sprt = int(request.form['act_sprt'])
    fant_arts = int(request.form['fant_arts'])
    won_arts = int(request.form['won_arts'])
    time_art = int(request.form['time_art'])

    input_data = np.array([[olympiad_participation, school, fav_sub, projects, grasp_pow, time_sprt, medals, career_sprt, act_sprt, fant_arts, won_arts, time_art]])

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
