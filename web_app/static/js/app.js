// State variables
let currentOriginalBase64 = null;
let currentMaskBase64 = null;
let tissueChart = null;

document.addEventListener("DOMContentLoaded", () => {
    initSamples();
    initUpload();
    initOpacitySlider();
});

// 1. Load sample patient cases
async function initSamples() {
    try {
        const res = await fetch("/api/samples");
        const data = await res.json();
        const grid = document.getElementById("samplesGrid");
        grid.innerHTML = "";

        data.samples.forEach((sample, idx) => {
            const div = document.createElement("div");
            div.className = `sample-item ${idx === 0 ? "active" : ""}`;
            div.innerHTML = `
                <img src="${sample.url}" alt="${sample.label}">
                <div class="sample-label">${sample.label}</div>
            `;
            div.onclick = () => {
                document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
                div.classList.add("active");
                runSampleDiagnosis(sample.filename);
            };
            grid.appendChild(div);
        });

        // Auto-run first sample on startup
        if (data.samples.length > 0) {
            runSampleDiagnosis(data.samples[0].filename);
        }
    } catch (err) {
        console.error("Failed to load samples:", err);
    }
}

// 2. Handle File Upload
function initUpload() {
    const dropzone = document.getElementById("uploadDropzone");
    const fileInput = document.getElementById("fileInput");

    dropzone.onclick = () => fileInput.click();

    fileInput.onchange = (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    };

    dropzone.ondragover = (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--accent-cyan)";
    };

    dropzone.ondragleave = () => {
        dropzone.style.borderColor = "rgba(91, 192, 190, 0.3)";
    };

    dropzone.ondrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    };
}

async function uploadFile(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", "Patient_Uploaded");

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            renderResults(data);
        } else {
            alert(data.error || "Diagnosis failed");
        }
    } catch (err) {
        console.error("Upload failed:", err);
        alert("Server error during diagnosis.");
    } finally {
        showLoading(false);
    }
}

async function runSampleDiagnosis(sampleName) {
    showLoading(true);
    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sample_name: sampleName })
        });
        const data = await res.json();
        if (data.success) {
            renderResults(data);
        }
    } catch (err) {
        console.error("Sample diagnosis failed:", err);
    } finally {
        showLoading(false);
    }
}

// 3. Render Results into UI
function renderResults(data) {
    currentOriginalBase64 = data.images.original;
    currentMaskBase64 = data.images.mask;

    // Display original and overlay on canvas
    document.getElementById("originalImg").src = "data:image/png;base64," + currentOriginalBase64;
    document.getElementById("maskImg").src = "data:image/png;base64," + currentMaskBase64;
    drawOverlay(parseFloat(document.getElementById("opacitySlider").value));

    // Stats Cards
    const stats = data.tissue_stats;
    document.getElementById("valHI").innerText = (stats.healing_index * 100).toFixed(1) + "%";
    document.getElementById("valRisk").innerText = data.cdss.risk_score_100.toFixed(1) + "/100";
    document.getElementById("valSlough").innerText = (stats.slough_ratio * 100).toFixed(1) + "%";
    document.getElementById("valLatency").innerText = data.latency.inference_ms + " ms";

    // Quality Badge
    const qBadge = document.getElementById("qualityBadge");
    if (data.quality.status === "ACCEPTABLE") {
        qBadge.className = "quality-pill quality-ok";
        qBadge.innerHTML = `✓ عکس استاندارد (وضوح: ${data.quality.blur_score})`;
    } else {
        qBadge.className = "quality-pill quality-poor";
        qBadge.innerHTML = `⚠️ عکس تار یا کم‌نور (وضوح: ${data.quality.blur_score})`;
    }

    // CDSS Clinical Box
    const cdssBox = document.getElementById("cdssBox");
    cdssBox.className = `card cdss-box alert-${data.cdss.risk_tier}`;
    document.getElementById("cdssBadge").innerText = data.cdss.color_badge;
    document.getElementById("cdssUrgency").innerText = "فوریت بالینی: " + data.cdss.clinical_urgency;

    // TIME items
    const recs = data.cdss.time_framework_recommendations;
    document.getElementById("timeTissue").innerText = recs["Tissue"] || "N/A";
    document.getElementById("timeInfection").innerText = recs["Infection"] || "N/A";
    document.getElementById("timeMoisture").innerText = recs["Moisture"] || "N/A";
    document.getElementById("timeEdge").innerText = recs["Edge"] || "N/A";

    // Patient text
    document.getElementById("patientMsg").innerText = data.cdss.patient_instructions;

    // Donut Chart
    renderChart(stats.tissues);
}

// 4. Alpha Blending Overlay on Canvas
function initOpacitySlider() {
    const slider = document.getElementById("opacitySlider");
    const valText = document.getElementById("opacityVal");
    slider.oninput = (e) => {
        const val = parseFloat(e.target.value);
        valText.innerText = Math.round(val * 100) + "%";
        drawOverlay(val);
    };
}

function drawOverlay(opacity) {
    const canvas = document.getElementById("overlayCanvas");
    const ctx = canvas.getContext("2d");
    if (!currentOriginalBase64 || !currentMaskBase64) return;

    const baseImg = new Image();
    const maskImg = new Image();

    baseImg.onload = () => {
        canvas.width = baseImg.width;
        canvas.height = baseImg.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(baseImg, 0, 0);

        maskImg.onload = () => {
            ctx.globalAlpha = opacity;
            ctx.drawImage(maskImg, 0, 0);
            ctx.globalAlpha = 1.0;
        };
        maskImg.src = "data:image/png;base64," + currentMaskBase64;
    };
    baseImg.src = "data:image/png;base64," + currentOriginalBase64;
}

// 5. Chart.js Tissue Distribution
function renderChart(tissues) {
    const ctx = document.getElementById("tissuePieChart").getContext("2d");
    const labels = [];
    const values = [];
    const colors = {
        "Epithelial tissue": "#2ecc71",
        "Slough": "#f1c40f",
        "Granulation tissue": "#e056fd",
        "Necrotic tissue": "#e67e22",
        "Other": "#00ffff"
    };
    const bgColors = [];

    for (const [name, info] of Object.entries(tissues)) {
        if (info.pixels > 0) {
            labels.push(name);
            values.push(info.percentage_of_wound);
            bgColors.push(colors[name] || "#94a3b8");
        }
    }

    if (tissueChart) {
        tissueChart.destroy();
    }

    tissueChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: bgColors,
                borderWidth: 2,
                borderColor: '#11192e'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { size: 10 } }
                }
            },
            cutout: '65%'
        }
    });
}

function showLoading(show) {
    const sp = document.getElementById("loadingSpinner");
    if (sp) sp.style.display = show ? "inline-block" : "none";
}
