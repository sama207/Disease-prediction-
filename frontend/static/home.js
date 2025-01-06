document.getElementById('symptom-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const symptomsInput = document.getElementById('symptoms').value.trim();
    const symptoms = symptomsInput.split(',').map(symptom => symptom.trim());
    const resultsElement = document.getElementById('results');

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symptoms }),
        });

        if (response.ok) {
            const predictions = await response.json();
            resultsElement.innerHTML = `
                <p>Random Forest Prediction: ${predictions.rf_model_prediction}</p>
                <p>SVM Prediction: ${predictions.svm_model_prediction}</p>
                <p>Decision Tree Prediction: ${predictions.dt_model_prediction}</p>
                <p><strong>Final Prediction: ${predictions.final_prediction}</strong></p>
            `;
        } else {
            const error = await response.json();
            document.getElementById('results').innerText = `Error: ${error.error}`;
        }
    } catch (error) {
        document.getElementById('results').innerText = `Error: ${error.message}`;
    }
});
