/**
 * Main JavaScript logic for Fake News Detection Web Application
 * Handles form submission, Chart.js visualization, and history updates.
 */

let predictionChart = null;

const sampleArticles = [
    "NASA confirms discovery of water ice deposits on the permanently shadowed craters of the Moon, paving the way for future lunar missions and sustainable space exploration.",
    "Shocking discovery: Drinking boiled seawater mixed with lemon cures all infectious diseases overnight according to secret leaked files!"
];

document.addEventListener("DOMContentLoaded", () => {
    const newsForm = document.getElementById("news-form");
    const newsInput = document.getElementById("news-input");
    const sampleBtn = document.getElementById("sample-btn");
    const clearBtn = document.getElementById("clear-btn");
    const refreshHistoryBtn = document.getElementById("refresh-history-btn");

    // Load initial history
    fetchHistory();

    // Event: Load Sample News
    sampleBtn.addEventListener("click", () => {
        const randomIndex = Math.floor(Math.random() * sampleArticles.length);
        newsInput.value = sampleArticles[randomIndex];
    });

    // Event: Clear Text
    clearBtn.addEventListener("click", () => {
        newsInput.value = "";
        hideResultCard();
    });

    // Event: Refresh History
    refreshHistoryBtn.addEventListener("click", () => {
        fetchHistory();
    });

    // Event: Form Submit
    newsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = newsInput.value.trim();
        if (!text) return;

        const analyzeBtn = document.getElementById("analyze-btn");
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = "<span>⏳ Analyzing...</span>";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ news_text: text })
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
                fetchHistory(); // Refresh history with new entry
            } else {
                alert(data.error || "An error occurred during prediction.");
            }
        } catch (error) {
            console.error("Prediction error:", error);
            alert("Failed to connect to the server.");
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = "<span>🔍 Analyze News</span>";
        }
    });
});

/**
 * Display prediction results and update the Chart.js visualizer
 */
function displayResult(result) {
    const resultCard = document.getElementById("result-card");
    const verdictBox = document.getElementById("verdict-box");
    const verdictText = document.getElementById("verdict-text");
    const confidenceText = document.getElementById("confidence-text");

    resultCard.style.display = "block";
    verdictText.textContent = result.prediction;
    confidenceText.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

    // Clear previous classes
    verdictBox.classList.remove("is-real", "is-fake");

    if (result.prediction.toUpperCase() === "REAL") {
        verdictBox.classList.add("is-real");
    } else {
        verdictBox.classList.add("is-fake");
    }

    renderChart(result.probabilities || {
        Real: result.prediction.toUpperCase() === "REAL" ? result.confidence : 1 - result.confidence,
        Fake: result.prediction.toUpperCase() === "FAKE" ? result.confidence : 1 - result.confidence
    });
}

function hideResultCard() {
    const resultCard = document.getElementById("result-card");
    resultCard.style.display = "none";
}

/**
 * Render or update Chart.js doughnut chart
 */
function renderChart(probabilities) {
    const ctx = document.getElementById("prediction-chart").getContext("2d");

    const realScore = ((probabilities.Real || probabilities.real || 0.5) * 100).toFixed(1);
    const fakeScore = ((probabilities.Fake || probabilities.fake || 0.5) * 100).toFixed(1);

    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Real", "Fake"],
            datasets: [{
                data: [realScore, fakeScore],
                backgroundColor: ["#10b981", "#ef4444"],
                borderColor: "#1e293b",
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#f8fafc" }
                }
            }
        }
    });
}

/**
 * Fetch and render recent predictions history
 */
async function fetchHistory() {
    try {
        const response = await fetch("/history");
        const historyData = await response.json();
        const tbody = document.getElementById("history-tbody");

        if (!historyData || historyData.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No predictions yet. Run an analysis above.</td></tr>`;
            return;
        }

        tbody.innerHTML = historyData.map((item, index) => {
            const isReal = item.prediction.toUpperCase() === "REAL";
            const tagClass = isReal ? "tag-real" : "tag-fake";
            const snippet = item.news_text.length > 60 
                ? item.news_text.substring(0, 60) + "..." 
                : item.news_text;

            return `
                <tr>
                    <td>${index + 1}</td>
                    <td title="${escapeHtml(item.news_text)}">${escapeHtml(snippet)}</td>
                    <td><span class="${tagClass}">${item.prediction}</span></td>
                    <td>${(item.confidence * 100).toFixed(1)}%</td>
                    <td>${item.created_at || "Just now"}</td>
                </tr>
            `;
        }).join("");
    } catch (err) {
        console.error("Error fetching history:", err);
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
