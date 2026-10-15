// State variables
let currentOriginalBase64 = null;
let currentMaskBase64 = null;
let currentOverlayBase64 = null;
let tissueChart = null;
let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
    initPreventWindowDrop();
    initSamples();
    initUpload();
    initOpacitySlider();
});

// Prevent browser from opening dragged files outside dropzone
function initPreventWindowDrop() {
    window.addEventListener("dragover", (e) => e.preventDefault(), false);
    window.addEventListener("drop", (e) => e.preventDefault(), false);
}

// Helper for status banner
function showAlert(message, type = "info") {
    const alertBox = document.getElementById("statusAlert");
    if (!alertBox) return;

    alertBox.style.display = "block";
    alertBox.innerText = message;

    if (type === "info") {
        alertBox.style.background = "rgba(14, 165, 233, 0.2)";
        alertBox.style.borderColor = "rgba(14, 165, 233, 0.4)";
        alertBox.style.color = "#7dd3fc";
    } else if (type === "success") {
        alertBox.style.background = "rgba(34, 197, 94, 0.2)";
        alertBox.style.borderColor = "rgba(34, 197, 94, 0.4)";
        alertBox.style.color = "#86efac";
    } else if (type === "error") {
        alertBox.style.background = "rgba(239, 68, 68, 0.2)";
        alertBox.style.borderColor = "rgba(239, 68, 68, 0.4)";
        alertBox.style.color = "#fca5a5";
    }
}

// 1. Load sample patient cases
async function initSamples() {
    try {
        const res = await fetch("/api/samples");
        const data = await res.json();
        const grid = document.getElementById("samplesGrid");
        if (!grid) return;
        grid.innerHTML = "";

        data.samples.forEach((sample, idx) => {
            const div = document.createElement("div");
            div.className = `sample-item ${idx === 0 ? "active" : ""}`;
            div.id = `sample-${sample.filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
            div.innerHTML = `
                <img src="${sample.url}" alt="${sample.label}">
                <div class="sample-label">${sample.label}</div>
            `;
            div.onclick = () => {
                document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
                div.classList.add("active");
                // Clear selected uploaded file info when switching to a sample
                clearUploadedFileInfo();
                runSampleDiagnosis(sample.filename);
            };
            grid.appendChild(div);
        });

        // Auto-run first sample on startup
        if (data.samples && data.samples.length > 0) {
            runSampleDiagnosis(data.samples[0].filename);
        }
    } catch (err) {
        console.error("Failed to load samples:", err);
    }
}

function clearUploadedFileInfo() {
    selectedFile = null;
    const infoBox = document.getElementById("fileSelectedInfo");
    if (infoBox) infoBox.style.display = "none";
}

// 2. Handle File Upload
function initUpload() {
    const dropzone = document.getElementById("uploadDropzone");
    const fileInput = document.getElementById("fileInput");
    const btnAnalyze = document.getElementById("btnAnalyze");

    if (!dropzone || !fileInput) return;

    // Click to open file dialog
    dropzone.onclick = () => fileInput.click();

    fileInput.onchange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            fileInput.value = ""; // Reset so same file can be re-selected
            handleFileSelected(file);
        }
    };

    // Drag and Drop styling
    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "var(--accent-cyan)";
            dropzone.style.background = "rgba(72, 202, 228, 0.12)";
        }, false);
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "rgba(91, 192, 190, 0.3)";
            dropzone.style.background = "rgba(17, 25, 46, 0.4)";
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    }, false);

    if (btnAnalyze) {
        btnAnalyze.onclick = () => {
            if (selectedFile) {
                uploadAndAnalyze(selectedFile);
            } else {
                fileInput.click();
            }
        };
    }
}

function handleFileSelected(file) {
    if (!file || !file.type.startsWith("image/")) {
        showAlert("⚠️ لطفاً یک فایل تصویر معتبر (JPG, PNG, WEBP) انتخاب کنید.", "error");
        return;
    }

    selectedFile = file;

    // Deselect any active sample case
    document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));

    // Update file selected notification badge
    const infoBox = document.getElementById("fileSelectedInfo");
    const nameSpan = document.getElementById("selectedFileName");
    if (infoBox && nameSpan) {
        infoBox.style.display = "block";
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        nameSpan.innerText = `${file.name} (${sizeMb} MB)`;
    }

    // Instant local preview in viewer
    const reader = new FileReader();
    reader.onload = (e) => {
        const origImg = document.getElementById("originalImg");
        const overlayImg = document.getElementById("overlayMaskImg");
        const maskImg = document.getElementById("maskImg");
        if (origImg) origImg.src = e.target.result;
        if (overlayImg) overlayImg.src = ""; // Clear old mask while new is computing
        if (maskImg) maskImg.src = "";
    };
    reader.readAsDataURL(file);

    // Update active case badge
    const badge = document.getElementById("activeCaseBadge");
    if (badge) badge.innerText = "📁 پرونده آپلودی: " + file.name;

    // Trigger analysis
    uploadAndAnalyze(file);
}

async function uploadAndAnalyze(file) {
    showLoading(true);
    setAnalyzeButtonState(false, "⏳ در حال پردازش...");
    showAlert("⏳ در حال ارسال تصویر و استنتاج مدل سبک هوش مصنوعی لبه (Edge UNet)...", "info");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", "Patient_Uploaded_" + file.name.substring(0, 12));

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            renderResults(data, file.name);
            showAlert("✅ تحلیل با موفقیت انجام شد! نتایج، زمان استنتاج و پروتکل درمانی در زیر نمایش داده شده است.", "success");
            
            // Smooth scroll to results
            const resultsSection = document.getElementById("mainResults");
            if (resultsSection) {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } else {
            showAlert("❌ خطا: " + (data.error || "خطا در پردازش تصویر"), "error");
        }
    } catch (err) {
        console.error("Upload failed:", err);
        showAlert("❌ خطای ارتباط با سرور در هنگام پردازش تصویر.", "error");
    } finally {
        showLoading(false);
        setAnalyzeButtonState(true, "🔍 تحلیل هوشمند زخم (Analyze Wound)");
    }
}

async function runSampleDiagnosis(sampleName) {
    showLoading(true);
    setAnalyzeButtonState(false, "⏳ در حال استنتاج...");
    showAlert("⏳ در حال تحلیل نمونه بالینی با مدل سبک لبه...", "info");

    const badge = document.getElementById("activeCaseBadge");
    if (badge) badge.innerText = "پرونده نمونه: " + sampleName;

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sample_name: sampleName })
        });
        const data = await res.json();
        if (data.success) {
            renderResults(data, sampleName);
            showAlert("✅ تحلیل نمونه با موفقیت انجام شد!", "success");
        } else {
            showAlert("❌ خطا: " + (data.error || "تحلیل نمونه ناموفق بود"), "error");
        }
    } catch (err) {
        console.error("Sample diagnosis failed:", err);
        showAlert("❌ خطای سرور در تحلیل نمونه.", "error");
    } finally {
        showLoading(false);
        setAnalyzeButtonState(true, "🔍 تحلیل هوشمند زخم (Analyze Wound)");
    }
}

function setAnalyzeButtonState(enabled, text) {
    const btn = document.getElementById("btnAnalyze");
    if (!btn) return;
    btn.disabled = !enabled;
    btn.innerHTML = `<span>${enabled ? "🔍" : "⏳"}</span> ${text}`;
    btn.style.opacity = enabled ? "1" : "0.7";
    btn.style.cursor = enabled ? "pointer" : "wait";
}

// 3. Render Results into UI
function renderResults(data, caseTitle = "") {
    currentOriginalBase64 = data.images.original;
    currentMaskBase64 = data.images.mask;
    currentOverlayBase64 = data.images.overlay_mask || data.images.mask;

    // Display original and transparent overlay mask
    const origImg = document.getElementById("originalImg");
    const maskImg = document.getElementById("maskImg");
    const overlayImg = document.getElementById("overlayMaskImg");

    if (origImg) origImg.src = "data:image/png;base64," + currentOriginalBase64;
    if (maskImg) maskImg.src = "data:image/png;base64," + currentMaskBase64;
    if (overlayImg) {
        overlayImg.src = "data:image/png;base64," + currentOverlayBase64;
        const slider = document.getElementById("opacitySlider");
        if (slider) overlayImg.style.opacity = slider.value;
    }

    if (caseTitle) {
        const badge = document.getElementById("activeCaseBadge");
        if (badge) badge.innerText = "پرونده: " + caseTitle;
    }

    // Stats Cards
    const stats = data.tissue_stats;
    const elHI = document.getElementById("valHI");
    const elRisk = document.getElementById("valRisk");
    const elSlough = document.getElementById("valSlough");
    const elLatency = document.getElementById("valLatency");

    if (elHI && stats) elHI.innerText = ((stats.healing_index || 0) * 100).toFixed(1) + "%";
    if (elRisk && data.cdss) elRisk.innerText = (data.cdss.risk_score_100 || 0).toFixed(1) + "/100";
    if (elSlough && stats) elSlough.innerText = ((stats.slough_ratio || 0) * 100).toFixed(1) + "%";
    if (elLatency && data.latency) elLatency.innerText = (data.latency.inference_ms || 0) + " ms";

    // Quality Badge
    const qBadge = document.getElementById("qualityBadge");
    if (qBadge && data.quality) {
        if (data.quality.status === "ACCEPTABLE") {
            qBadge.className = "quality-pill quality-ok";
            qBadge.innerHTML = `✓ عکس استاندارد (وضوح: ${Math.round(data.quality.blur_score || 0)})`;
        } else {
            qBadge.className = "quality-pill quality-poor";
            qBadge.innerHTML = `⚠️ عکس تار یا کم‌نور (وضوح: ${Math.round(data.quality.blur_score || 0)})`;
        }
    }

    // CDSS Clinical Box
    const cdssBox = document.getElementById("cdssBox");
    const cdssBadge = document.getElementById("cdssBadge");
    const cdssUrgency = document.getElementById("cdssUrgency");

    if (cdssBox && data.cdss) {
        cdssBox.className = `card cdss-box alert-${data.cdss.risk_tier}`;
    }
    if (cdssBadge && data.cdss) {
        cdssBadge.innerText = data.cdss.color_badge;
    }
    if (cdssUrgency && data.cdss) {
        cdssUrgency.innerText = "فوریت بالینی: " + data.cdss.clinical_urgency;
    }

    // TIME Framework Recommendations
    if (data.cdss && data.cdss.time_framework_recommendations) {
        const recs = data.cdss.time_framework_recommendations;
        const elT = document.getElementById("timeTissue");
        const elI = document.getElementById("timeInfection");
        const elM = document.getElementById("timeMoisture");
        const elE = document.getElementById("timeEdge");

        if (elT) elT.innerText = recs["Tissue"] || "N/A";
        if (elI) elI.innerText = recs["Infection"] || "N/A";
        if (elM) elM.innerText = recs["Moisture"] || "N/A";
        if (elE) elE.innerText = recs["Edge"] || "N/A";
    }

    // Patient text
    const patientMsg = document.getElementById("patientMsg");
    if (patientMsg && data.cdss) {
        patientMsg.innerText = data.cdss.patient_instructions;
    }

    // Tissue Distribution (Chart or Fallback Progress Bars)
    if (stats && stats.tissues) {
        renderTissueAnalytics(stats.tissues);
    }
}

// 4. Alpha Blending Overlay Slider
function initOpacitySlider() {
    const slider = document.getElementById("opacitySlider");
    const valText = document.getElementById("opacityVal");
    const overlayImg = document.getElementById("overlayMaskImg");

    if (!slider) return;

    slider.oninput = (e) => {
        const val = parseFloat(e.target.value);
        if (valText) valText.innerText = Math.round(val * 100) + "%";
        if (overlayImg) overlayImg.style.opacity = val;
    };
}

// 5. Tissue Breakdown: Try Chart.js, or Fallback to Native HTML Progress Bars
function renderTissueAnalytics(tissues) {
    const colors = {
        "Epithelial tissue": "#2ecc71",
        "Granulation tissue": "#ef4444",
        "Slough": "#f59e0b",
        "Necrotic tissue": "#1e293b",
        "Other": "#06b6d4"
    };

    const persianNames = {
        "Epithelial tissue": "اپیتلیال (پوششی)",
        "Granulation tissue": "گرانولاسیون (سرخ)",
        "Slough": "اسلاف / فیبرین (زرد)",
        "Necrotic tissue": "نکروتیک (سیاه)",
        "Other": "سایر بافت‌ها"
    };

    // Fallback HTML Bars (always reliable, works 100% offline)
    const fallbackBox = document.getElementById("fallbackTissueBars");
    if (fallbackBox) {
        let barsHtml = `<div style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">`;
        for (const [name, info] of Object.entries(tissues)) {
            const pct = (info.percentage_of_wound || 0).toFixed(1);
            const color = colors[name] || "#94a3b8";
            const faName = persianNames[name] || name;
            barsHtml += `
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 2px;">
                        <span style="color: #e2e8f0;">${faName}</span>
                        <span style="font-weight: 700; color: ${color};">${pct}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; overflow: hidden;">
                        <div style="background: ${color}; width: ${pct}%; height: 100%;"></div>
                    </div>
                </div>
            `;
        }
        barsHtml += `</div>`;
        fallbackBox.innerHTML = barsHtml;
        fallbackBox.style.display = "block";
    }

    // If Chart.js library is loaded, render doughnut chart
    if (typeof Chart !== 'undefined') {
        const canvas = document.getElementById("tissuePieChart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");

        const labels = [];
        const values = [];
        const bgColors = [];

        for (const [name, info] of Object.entries(tissues)) {
            if (info.percentage_of_wound > 0) {
                labels.push(persianNames[name] || name);
                values.push(info.percentage_of_wound);
                bgColors.push(colors[name] || "#94a3b8");
            }
        }

        if (tissueChart) {
            tissueChart.destroy();
        }

        try {
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
                            display: false
                        }
                    },
                    cutout: '65%'
                }
            });
        } catch (e) {
            console.warn("Chart.js render skipped:", e);
        }
    }
}

function showLoading(show) {
    const sp = document.getElementById("loadingSpinner");
    if (sp) sp.style.display = show ? "inline-block" : "none";
}
