/**
 * Script for Crop Disease Prediction Flask App
 */

// Form submission
document.getElementById('predictionForm').addEventListener('submit', handlePrediction);
document.getElementById('sensitivityBtn').addEventListener('click', handleSensitivity);
document.getElementById('sensitivityCloseBtn').addEventListener('click', hideSensitivityModal);
document.getElementById('sensitivityCloseBackdrop').addEventListener('click', hideSensitivityModal);
document.getElementById('saveApiUrlBtn').addEventListener('click', saveApiUrl);
document.getElementById('syncNowBtn').addEventListener('click', syncLiveTelemetry);
document.getElementById('crop').addEventListener('change', syncLiveTelemetry);

const API_URL_STORAGE_KEY = 'cropDiseaseApiBaseUrl';
const LIVE_SYNC_INTERVAL_MS = 7000;
let liveSyncTimer = null;

const plotConfig = {
    responsive: true,
    displayModeBar: false
};

const plotTheme = {
    paper_bgcolor: 'rgba(0, 0, 0, 0)',
    plot_bgcolor: 'rgba(0, 0, 0, 0)',
    font: { color: '#17311f', family: 'Aptos, Segoe UI, sans-serif' },
    colorway: ['#2f8f4e', '#5cb85c', '#7bc96f', '#9fd18b', '#c7e6bc']
};

restoreApiUrl();
initializeLiveSync();

/**
 * Extract form data as object
 */
function getFormData() {
    const form = document.getElementById('predictionForm');
    const data = {
        crop: form.crop.value,
        nitrogen: parseFloat(form.nitrogen.value),
        phosphorus: parseFloat(form.phosphorus.value),
        potassium: parseFloat(form.potassium.value),
        ph_value: parseFloat(form.ph_value.value),
        soil_temp: parseFloat(form.soil_temp.value),
        humidity: parseFloat(form.humidity.value),
        rainfall: parseFloat(form.rainfall.value),
        weather_temp: parseFloat(form.weather_temp.value),
        weather_humidity: parseFloat(form.weather_humidity.value),
        cloud_cover: parseFloat(form.cloud_cover.value),
        wind_speed: parseFloat(form.wind_speed.value),
        rolling_rainfall: parseFloat(form.rolling_rainfall.value),
        temp_variance: parseFloat(form.temp_variance.value),
        favorability: parseFloat(form.favorability.value)
    };
    return data;
}

function getLiveFieldMap() {
    return {
        nitrogen: 'nitrogen',
        phosphorus: 'phosphorus',
        potassium: 'potassium',
        ph_value: 'ph_value',
        soil_temp: 'soil_temp',
        humidity: 'humidity',
        rainfall: 'rainfall',
        weather_temp: 'weather_temp',
        weather_humidity: 'weather_humidity',
        cloud_cover: 'cloud_cover',
        wind_speed: 'wind_speed',
        rolling_rainfall: 'rolling_rainfall',
        temp_variance: 'temp_variance',
        favorability: 'favorability'
    };
}

function setLiveStatus(statusText, updatedText) {
    document.getElementById('liveSyncStatus').textContent = statusText;
    document.getElementById('liveSyncUpdated').textContent = updatedText;
}

function flashField(element) {
    element.classList.remove('live-flash');
    void element.offsetWidth;
    element.classList.add('live-flash');
}

function syncFieldValue(fieldName, value) {
    const input = document.getElementById(fieldName);

    if (!input) {
        return;
    }

    const nextValue = typeof value === 'number' ? value.toString() : value;
    if (input.value !== nextValue) {
        input.value = nextValue;
        flashField(input);
    }
}

async function syncLiveTelemetry() {
    const apiUrl = getApiUrl();
    const crop = document.getElementById('crop').value;

    if (!apiUrl) {
        setLiveStatus('Live sync idle', 'Add an IoT URL to enable updates');
        return;
    }

    try {
        setLiveStatus('Syncing live data', `Pulling ${crop} telemetry...`);
        const response = await fetch(buildApiUrl(apiUrl, `/api/iot/readings?crop=${encodeURIComponent(crop)}`));

        if (!response.ok) {
            throw new Error(`Telemetry request failed (${response.status})`);
        }

        const telemetry = await response.json();
        const liveFields = getLiveFieldMap();

        Object.entries(liveFields).forEach(([formField, telemetryField]) => {
            if (telemetry[telemetryField] !== undefined) {
                syncFieldValue(formField, telemetry[telemetryField]);
            }
        });

        setLiveStatus('Live sync active', `Updated ${new Date().toLocaleTimeString()}`);
    } catch (error) {
        setLiveStatus('Live sync paused', error.message);
    }
}

function initializeLiveSync() {
    if (liveSyncTimer) {
        clearInterval(liveSyncTimer);
    }

    syncLiveTelemetry();
    liveSyncTimer = setInterval(syncLiveTelemetry, LIVE_SYNC_INTERVAL_MS);
}

/**
 * Handle prediction submission
 */
async function handlePrediction(event) {
    event.preventDefault();
    showLoading(true);
    clearError();

    try {
        const formData = getFormData();
        const apiUrl = getApiUrl();

        // Parallel API calls
        const [predictionRes, importanceRes, contributionsRes] = await Promise.all([
            fetch(buildApiUrl(apiUrl, '/api/predict'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            }),
            fetch(buildApiUrl(apiUrl, '/api/feature-importance'), {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            }),
            fetch(buildApiUrl(apiUrl, '/api/local-contributions'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            })
        ]);

        if (!predictionRes.ok || !importanceRes.ok || !contributionsRes.ok) {
            throw new Error('API request failed');
        }

        const prediction = await predictionRes.json();
        const importance = await importanceRes.json();
        const contributions = await contributionsRes.json();

        // Display results
        displayPrediction(prediction);
        displayTop3(prediction.top_3);
        displayProbabilities(prediction.probabilities);
        displayFeatureImportance(importance);
        displayLocalContributions(contributions);

        showLoading(false);
    } catch (error) {
        showError('Error during prediction: ' + error.message);
        showLoading(false);
    }
}

/**
 * Display prediction result
 */
function displayPrediction(prediction) {
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('predictedDisease').textContent = prediction.disease;
    document.getElementById('confidence').textContent = (prediction.confidence * 100).toFixed(2) + '%';
}

/**
 * Display top 3 predictions
 */
function displayTop3(top3) {
    const list = document.getElementById('top3List');
    list.innerHTML = '';

    top3.forEach((item, idx) => {
        const prob_percent = (item.probability * 100).toFixed(2);
        const html = `
            <div class="top3-item">
                <div class="top3-rank">#${idx + 1}</div>
                <div class="top3-disease">${item.disease}</div>
                <div class="top3-probability">${prob_percent}%</div>
            </div>
        `;
        list.innerHTML += html;
    });

    document.getElementById('top3Card').style.display = 'block';
}

/**
 * Display probability bar chart
 */
function displayProbabilities(probabilities) {
    const diseases = Object.keys(probabilities);
    const probs = Object.values(probabilities);

    const trace = {
        x: diseases,
        y: probs,
        type: 'bar',
        marker: {
            color: '#5cb85c',
            line: { color: '#2f8f4e', width: 1 }
        }
    };

    const layout = {
        title: '',
        height: 185,
        margin: { l: 42, r: 12, t: 6, b: 54 },
        xaxis: { title: 'Disease Class', automargin: true },
        yaxis: { title: 'Probability', range: [0, 1], automargin: true },
        hovermode: 'x unified',
        ...plotTheme,
        template: 'plotly_white'
    };

    Plotly.newPlot('probChart', [trace], layout, plotConfig);
}

/**
 * Display feature importance chart
 */
function displayFeatureImportance(importance) {
    if (importance.error) {
        console.warn('Feature importance error:', importance.error);
        return;
    }

    const features = Object.keys(importance).reverse();
    const importances = Object.values(importance).reverse();

    const trace = {
        y: features,
        x: importances,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: '#5cb85c',
            line: { color: '#2f8f4e', width: 1 }
        }
    };

    const layout = {
        title: '',
        height: 185,
        margin: { l: 120, r: 12, t: 6, b: 28 },
        xaxis: { title: 'Importance', automargin: true },
        yaxis: { automargin: true },
        hovermode: 'y unified',
        ...plotTheme,
        template: 'plotly_white'
    };

    Plotly.newPlot('importanceChart', [trace], layout, plotConfig);
}

/**
 * Display local contributions chart
 */
function displayLocalContributions(contributions) {
    if (!contributions.contributions || Object.keys(contributions.contributions).length === 0) {
        console.warn('No local contributions available');
        return;
    }

    const features = Object.keys(contributions.contributions);
    const contribs = Object.values(contributions.contributions);

    const colors = contribs.map(c => c >= 0 ? '#5cb85c' : '#c44949');

    const trace = {
        y: features,
        x: contribs,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: colors,
            line: { color: colors.map(c => c.replace('0.8', '1')), width: 1 }
        }
    };

    const layout = {
        title: '',
        height: 185,
        margin: { l: 120, r: 12, t: 6, b: 28 },
        xaxis: { title: 'Change in Predicted Probability', automargin: true },
        yaxis: { automargin: true },
        hovermode: 'y unified',
        annotations: [{
            text: `Predicted class: ${contributions.predicted_class}`,
            xref: 'paper',
            yref: 'paper',
            x: 0,
            y: 1.12,
            showarrow: false,
            font: { color: '#5f7665', size: 11 },
            align: 'left'
        }],
        ...plotTheme,
        template: 'plotly_white'
    };

    Plotly.newPlot('contributionsChart', [trace], layout, plotConfig);
}

/**
 * Handle sensitivity analysis
 */
async function handleSensitivity() {
    showLoading(true);
    clearError();

    try {
        const formData = getFormData();
        const apiUrl = getApiUrl();

        // First get local contributions to know top features
        const contribRes = await fetch(buildApiUrl(apiUrl, '/api/local-contributions'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!contribRes.ok) throw new Error('Failed to get contributions');
        
        const contribData = await contribRes.json();
        const topFeatures = contribData.top_features.slice(0, 3);

        // Get sensitivity for top 3 features
        const sensitivityCharts = document.getElementById('sensitivityCharts');
        sensitivityCharts.innerHTML = '';

        for (const feature of topFeatures) {
            const sensitivityRes = await fetch(buildApiUrl(apiUrl, '/api/sensitivity'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...formData, feature })
            });

            if (!sensitivityRes.ok) continue;
            
            const sensitivityData = await sensitivityRes.json();
            
            const chartDiv = document.createElement('div');
            chartDiv.className = 'sensitivity-chart';
            chartDiv.id = `sensitivity-${feature}`;
            sensitivityCharts.appendChild(chartDiv);

            const trace = {
                x: sensitivityData.values,
                y: sensitivityData.probabilities,
                type: 'scatter',
                mode: 'lines+markers',
                name: feature,
                line: { color: '#2f8f4e', width: 2 },
                marker: { size: 5 }
            };

            const layout = {
                title: '',
                height: 230,
                margin: { l: 46, r: 14, t: 24, b: 44 },
                xaxis: { title: `${feature} value`, automargin: true },
                yaxis: { title: 'Predicted Probability', range: [0, 1], automargin: true },
                hovermode: 'closest',
                annotations: [{
                    text: feature,
                    xref: 'paper',
                    yref: 'paper',
                    x: 0,
                    y: 1.13,
                    showarrow: false,
                    font: { color: '#5f7665', size: 11 },
                    align: 'left'
                }],
                ...plotTheme,
                template: 'plotly_white'
            };

            Plotly.newPlot(`sensitivity-${feature}`, [trace], layout, plotConfig);
        }

        showSensitivityModal(true);
        showLoading(false);
    } catch (error) {
        showError('Error during sensitivity analysis: ' + error.message);
        showLoading(false);
    }
}

/**
 * Utility functions
 */
function showLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('errorMsg');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function clearError() {
    document.getElementById('errorMsg').style.display = 'none';
}

function getApiUrl() {
    return document.getElementById('apiBaseUrl').value.trim();
}

function buildApiUrl(baseUrl, path) {
    if (!baseUrl) {
        return path;
    }

    return `${baseUrl.replace(/\/+$/, '')}${path}`;
}

function saveApiUrl() {
    const apiUrl = getApiUrl();

    if (apiUrl) {
        localStorage.setItem(API_URL_STORAGE_KEY, apiUrl);
        showError('IoT backend URL saved. Live telemetry sync is now enabled.');
        initializeLiveSync();
    } else {
        localStorage.removeItem(API_URL_STORAGE_KEY);
        showError('Using the local Flask API endpoint.');
        initializeLiveSync();
    }
}

function restoreApiUrl() {
    const savedApiUrl = localStorage.getItem(API_URL_STORAGE_KEY);

    if (savedApiUrl) {
        document.getElementById('apiBaseUrl').value = savedApiUrl;
    }
}

function showSensitivityModal(show) {
    document.getElementById('sensitivityCard').style.display = show ? 'block' : 'none';
}

function hideSensitivityModal() {
    showSensitivityModal(false);
}
