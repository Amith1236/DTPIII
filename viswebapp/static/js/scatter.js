let chartInstance; // Store the chart instance globally

async function populateDropdowns() {
    const response = await fetch("/get_columns");
    const { features, targets } = await response.json();

    const xSelect = document.getElementById("x-select");
    const ySelect = document.getElementById("y-select");

    // Populate x-select with features only
    features.forEach(col => {
        const xOption = document.createElement("option");
        xOption.value = col;
        xOption.textContent = col;
        xSelect.appendChild(xOption);
    });

    // Populate y-select with targets only
    targets.forEach(col => {
        const yOption = document.createElement("option");
        yOption.value = col;
        yOption.textContent = col;
        ySelect.appendChild(yOption);
    });
}




async function updatePlot() {
    const x_col = document.getElementById("x-select").value;
    const y_col = document.getElementById("y-select").value;

    const response = await fetch("/scatterplot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x_col, y_col })
    });

    const data = await response.json();

    const ctx = document.getElementById("scatterPlot").getContext("2d");

    // Destroy previous chart instance if it exists
    if (chartInstance) {
        chartInstance.destroy();
    }

    // Create a new chart
    chartInstance = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                label: `${x_col} vs ${y_col}`,
                data: data.x.map((x, i) => ({ x, y: data.y[i] })),
                backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: x_col } },
                y: { title: { display: true, text: y_col } }
            }
        }
    });
}

window.onload = populateDropdowns;