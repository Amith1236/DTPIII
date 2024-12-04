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

        const label = document.createElement("label");
        label.textContent = col;

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

let regressionChartInstance; // Store the chart instance globally

async function runRegression() {
    const selectedFeatures = Array.from(document.querySelectorAll("input[name='feature']:checked"))
        .map(cb => cb.value);
    const target = [document.getElementById("target").value];

    const response = await fetch("/run_regression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: selectedFeatures, target })
    });

    const { metrics, equation, cost_graph } = await response.json();

    // Update the results section with the metrics and equation
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = `
        <h2>Regression Results</h2>
        <p><strong>Adjusted R2:</strong> ${metrics.AdjustedR2}</p>
        <p><strong>MSE:</strong> ${metrics.MSE}</p>
        <p><strong>MAE:</strong> ${metrics.MAE}</p>
        <p><strong>Equation:</strong> ${equation}</p>
        <img src="data:image/png;base64,${cost_graph}" alt="Cost Graph">
    `;
}


window.onload = populateOptions;
