from flask import Flask, request, render_template
from joblib import load
from tensorflow.keras.models import load_model
import pandas as pd
import os

app = Flask(__name__)

# Carga del preprocesador y modelo previamente guardados
preprocessor = load('pipeline.joblib')
model = load_model('best_model_0.18.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    if request.method == 'POST':
        # Extracción de datos del formulario
        elemento = request.form['elemento']
        ecut = request.form['ecut']
        kpoints = request.form['kpoints']
        pseudopotencial = request.form['pseudopotencial']

        # Verifica que 'ecut' sea un entero
        try:
            ecut = int(ecut)
        except ValueError:
            return render_template('calculator.html', error="Ecut debe ser un entero", prediction=None)

        # Crear DataFrame para la entrada
        input_df = pd.DataFrame({
            'Elemento': [elemento],
            'Ecut': [ecut],
            'KPoints': [kpoints],
            'Pseudopotencial': [pseudopotencial]
        })

        # Preprocesamiento y predicción
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)
        prediction = '{:.5f}'.format(prediction[0][0])

        # Muestra el resultado
        return render_template('calculator.html', prediction=prediction)
    
    return render_template('calculator.html')

if __name__ == '__main__':
    app.run(debug=True)
