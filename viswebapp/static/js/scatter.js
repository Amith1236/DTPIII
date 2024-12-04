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

    // Update the image source dynamically
    const plotImage = document.getElementById("scatterPlotImage");
    plotImage.src = data.plot_url; // Assuming the backend returns a URL or base64 data for the image

    // Show the plot container once the image is ready
    const scatterPlotContainer = document.getElementById("scatterPlotContainer");
    scatterPlotContainer.style.display = "block"; // Show the image container
}

window.onload = populateDropdowns;
