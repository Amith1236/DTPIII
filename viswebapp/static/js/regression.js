async function populateOptions() {
    const response = await fetch("/get_columns");
    const { features, targets } = await response.json();

    const featuresDiv = document.getElementById("features");
    const targetSelect = document.getElementById("target");

    features.forEach(col => {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = col;
        checkbox.name = "feature";
        checkbox.id = `checkbox-${col}`; // Dynamically set an ID for each checkbox

        const label = document.createElement("label");
        label.textContent = col;
        label.setAttribute("for", checkbox.id); // Link the label to the checkbox



        // Add checkbox, label, and input field to the container
        featuresDiv.appendChild(checkbox);
        featuresDiv.appendChild(label);
        featuresDiv.appendChild(document.createElement("br"));
    });

    targets.forEach(col => {
        const option = document.createElement("option");
        option.value = col;
        option.textContent = col;
        targetSelect.appendChild(option);
    });
}



let regressionBetas = []; // To store the regression coefficients (betas)
let selectedFeatures = []; // To store the selected features for regression
let calculatedMeans = [];
let calculatedStds = [];

async function runRegression() {
    selectedFeatures = Array.from(document.querySelectorAll("input[name='feature']:checked"))
        .map(cb => cb.value);
    const target = [document.getElementById("target").value];

    const response = await fetch("/run_regression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: selectedFeatures, target })
    });

    const { metrics, equation, vs_graph, betas , means, stds} = await response.json();
    regressionBetas = betas; // Store the regression coefficients for prediction
    calculatedMeans = means;
    calculatedStds = stds;
    console.log("Regression Betas:", regressionBetas);
    console.log("Regression means:", calculatedMeans);
    console.log("Regression stds:", calculatedStds);

    // Update the results section with the metrics and equation
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = `
        <h2>Regression Results</h2>
        <p><strong>Adjusted R2:</strong> ${metrics.AdjustedR2}</p>
        <p><strong>MSE:</strong> ${metrics.MSE}</p>
        <p><strong>MAE:</strong> ${metrics.MAE}</p>
        <p><strong>Equation:</strong> ${equation}</p>
        <img src="data:image/png;base64,${vs_graph}" alt="Prediction vs Actual Graph">
    `;

    // Dynamically create input fields for each selected feature
    const dynamicInputsDiv = document.getElementById("dynamicInputs");
    dynamicInputsDiv.innerHTML = ''; // Clear any previous inputs

    selectedFeatures.forEach(feature => {
        const inputDiv = document.createElement("div");
        inputDiv.classList.add("mb-3");
        inputDiv.innerHTML = `
            <label for="input-${feature}" class="form-label">${feature}:</label>
            <input type="number" id="input-${feature}" class="form-control" placeholder="Enter value for ${feature}">
        `;
        dynamicInputsDiv.appendChild(inputDiv);
    });
}


function predictOutput() {
    const inputValues = selectedFeatures.map((feature, idx) => {
        const inputElement = document.getElementById(`input-${feature}`);
        return parseFloat(inputElement.value);
    });

    if (inputValues.length !== calculatedMeans.length) {
        alert("Number of input values does not match the number of features.");
        return;
    }

    // Normalize inputs based on the means and stds
    const normalizedInputs = inputValues.map((value, idx) => {
        return (value - calculatedMeans[idx]) / calculatedStds[idx];
    });

    // Calculate the predicted output (y_hat = X * beta)
    let prediction = parseFloat(regressionBetas[0]); // Intercept should be a number
    for (let i = 0; i < normalizedInputs.length; i++) {
        prediction += normalizedInputs[i] * parseFloat(regressionBetas[i + 1]); // Ensure betas are numbers
    }

    // Display the prediction
    const predictionDiv = document.getElementById("predictionResult");
    predictionDiv.innerHTML = `
        <h4>Predicted Output: ${prediction}</h4>
    `;
}


window.onload = populateOptions; // This will populate the dropdowns when the page loads