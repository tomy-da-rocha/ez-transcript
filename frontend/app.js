/**
 * EZ Transcript — Frontend application logic
 * WaveSurfer.js powered NLE-style timeline editor
 */

import WaveSurfer from "/static/lib/wavesurfer.esm.js";
import RegionsPlugin from "/static/lib/regions.esm.js";
import TimelinePlugin from "/static/lib/timeline.esm.js";

(function () {
    "use strict";

    // --- State ---
    let currentFileId = null;
    let currentFileInfo = null;
    let currentResult = null;
    let wavesurfer = null;
    let regionsPlugin = null;
    let isVideo = false;
    let regionCounter = 0;

    // Region color palette
    const REGION_COLORS = [
        "rgba(108, 99, 255, 0.25)",
        "rgba(255, 107, 107, 0.25)",
        "rgba(78, 205, 196, 0.25)",
        "rgba(255, 195, 0, 0.25)",
        "rgba(162, 155, 254, 0.25)",
        "rgba(0, 206, 209, 0.25)",
        "rgba(255, 154, 162, 0.25)",
        "rgba(144, 238, 144, 0.25)",
    ];

    // --- DOM refs ---
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const stepUpload = $("#step-upload");
    const stepPreview = $("#step-preview");
    const stepProgress = $("#step-progress");
    const stepResult = $("#step-result");
    const stepHistory = $("#step-history");

    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");
    const uploadProgress = $("#upload-progress");
    const uploadProgressFill = $("#upload-progress-fill");
    const uploadProgressText = $("#upload-progress-text");
    const uploadList = $("#upload-list");

    const nleEditor = $("#nle-editor");
    const videoPanel = $("#video-panel");
    const nleVideo = $("#nle-video");
    const waveformContainer = $("#waveform-container");
    const segmentsEditor = $("#segments-editor");
    const segmentsList = $("#segments-list");

    const transcribeProgressFill = $("#transcribe-progress-fill");
    const transcribeProgressText = $("#transcribe-progress-text");
    const spinner = $("#spinner");

    const resultMeta = $("#result-meta");
    const resultText = $("#result-text");
    const resultSegments = $("#result-segments");

    const historyList = $("#history-list");
    const systemInfo = $("#system-info");

    // Transport
    const nlePlay = $("#nle-play");
    const nleStop = $("#nle-stop");
    const nleTimecode = $("#nle-timecode");
    const nleDuration = $("#nle-duration");
    const nleZoom = $("#nle-zoom");
    const nleZoomIn = $("#nle-zoom-in");
    const nleZoomOut = $("#nle-zoom-out");

    // --- Init ---
    async function init() {
        loadSystemInfo();
        loadHistory();
        setupEventListeners();
    }

    // --- System Info ---
    async function loadSystemInfo() {
        try {
            const res = await fetch("/api/system-info");
            const info = await res.json();
            let badge = "";
            if (info.gpu_available) {
                badge = `🟢 GPU: ${info.gpu_name} (${info.vram_total_mb} MB) — Modèle: ${info.selected_model}`;
            } else {
                badge = `🟡 CPU — Modèle: ${info.selected_model}`;
            }
            if (info.cuda_warning) {
                badge += ` | ⚠️ ${info.cuda_warning}`;
            }
            if (!info.ffmpeg_available) {
                badge += " | ⚠️ FFmpeg non détecté";
            }
            systemInfo.textContent = badge;

            // Populate model selector
            const modelSelect = $("#model");
            if (modelSelect && info.available_models) {
                // Keep the first "Auto" option
                while (modelSelect.options.length > 1) modelSelect.remove(1);
                for (const m of info.available_models) {
                    const opt = document.createElement("option");
                    opt.value = m;
                    opt.textContent = m === info.selected_model ? `${m} (recommandé)` : m;
                    modelSelect.appendChild(opt);
                }
            }
        } catch {
            systemInfo.textContent = "⚠️ Impossible de charger les infos système";
        }
    }

    // --- Event Listeners ---
    function setupEventListeners() {
        // Drop zone
        dropZone.addEventListener("click", () => fileInput.click());
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                handleFiles(e.dataTransfer.files);
            }
        });
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                handleFiles(fileInput.files);
            }
        });

        // Segment mode toggle
        $$('input[name="segment-mode"]').forEach((radio) => {
            radio.addEventListener("change", () => {
                const isSegments = radio.value === "segments" && radio.checked;
                segmentsEditor.classList.toggle("hidden", !isSegments);
                if (regionsPlugin) {
                    if (isSegments) {
                        regionsPlugin.enableDragSelection({
                            color: REGION_COLORS[regionCounter % REGION_COLORS.length],
                        });
                    } else {
                        regionsPlugin.clearRegions();
                        segmentsList.innerHTML = "";
                    }
                }
            });
        });

        // Add segment
        $("#btn-add-segment").addEventListener("click", addSegmentRow);

        // Navigation
        $("#btn-back-upload").addEventListener("click", () => {
            destroyWaveSurfer();
            showStep("upload");
        });
        $("#btn-transcribe").addEventListener("click", startTranscription);
        $("#btn-new").addEventListener("click", resetAll);

        // Export buttons
        $("#btn-copy").addEventListener("click", copyToClipboard);
        $("#btn-export-txt").addEventListener("click", () => exportFile("txt"));
        $("#btn-export-srt").addEventListener("click", () => exportFile("srt"));
        $("#btn-export-json").addEventListener("click", () => exportFile("json"));

        // Transport controls
        nlePlay.addEventListener("click", () => {
            if (wavesurfer) wavesurfer.playPause();
        });
        nleStop.addEventListener("click", () => {
            if (wavesurfer) {
                wavesurfer.stop();
                updateTimecode(0);
            }
        });
        nleZoom.addEventListener("input", () => {
            if (wavesurfer) wavesurfer.zoom(Number(nleZoom.value));
        });
        nleZoomIn.addEventListener("click", () => {
            nleZoom.value = Math.min(500, Number(nleZoom.value) + 25);
            if (wavesurfer) wavesurfer.zoom(Number(nleZoom.value));
        });
        nleZoomOut.addEventListener("click", () => {
            nleZoom.value = Math.max(1, Number(nleZoom.value) - 25);
            if (wavesurfer) wavesurfer.zoom(Number(nleZoom.value));
        });
    }

    // --- Step management ---
    function showStep(step) {
        [stepUpload, stepPreview, stepProgress, stepResult].forEach((s) =>
            s.classList.add("hidden")
        );
        stepUpload.classList.remove("active");

        switch (step) {
            case "upload":
                stepUpload.classList.remove("hidden");
                stepUpload.classList.add("active");
                break;
            case "preview":
                stepPreview.classList.remove("hidden");
                break;
            case "progress":
                stepProgress.classList.remove("hidden");
                break;
            case "result":
                stepResult.classList.remove("hidden");
                break;
        }

        // Always show history
        stepHistory.classList.remove("hidden");
    }

    // --- File Upload ---
    async function handleFiles(files) {
        for (const file of files) {
            await uploadFile(file);
        }
    }

    async function uploadFile(file) {
        uploadProgress.classList.remove("hidden");
        uploadProgressFill.style.width = "0%";
        uploadProgressText.textContent = `Upload de ${file.name}...`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const xhr = new XMLHttpRequest();

            const result = await new Promise((resolve, reject) => {
                xhr.upload.addEventListener("progress", (e) => {
                    if (e.lengthComputable) {
                        const pct = Math.round((e.loaded / e.total) * 100);
                        uploadProgressFill.style.width = pct + "%";
                        uploadProgressText.textContent = `Upload de ${file.name}... ${pct}%`;
                    }
                });

                xhr.addEventListener("load", () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        try {
                            const err = JSON.parse(xhr.responseText);
                            reject(new Error(err.detail || "Erreur upload"));
                        } catch {
                            reject(new Error("Erreur upload"));
                        }
                    }
                });

                xhr.addEventListener("error", () => reject(new Error("Erreur réseau")));
                xhr.open("POST", "/api/upload");
                xhr.send(formData);
            });

            uploadProgressFill.style.width = "100%";
            uploadProgressText.textContent = `${file.name} — uploadé avec succès`;

            currentFileId = result.file_id;
            currentFileInfo = result;

            // Show in upload list
            addUploadEntry(file.name, result);

            // Move to preview step
            setTimeout(() => showPreview(result), 500);

        } catch (err) {
            uploadProgressText.textContent = `❌ ${err.message}`;
            uploadProgressFill.style.width = "0%";
        }
    }

    function addUploadEntry(name, info) {
        uploadList.innerHTML = "";
        const div = document.createElement("div");
        div.className = "upload-entry";
        div.innerHTML = `
            <span class="upload-name">${escapeHtml(name)}</span>
            <span class="upload-meta">${info.duration_display} — ${info.size_mb} Mo</span>
        `;
        uploadList.appendChild(div);
    }

    // --- Preview / NLE Editor ---
    function showPreview(info) {
        showStep("preview");
        destroyWaveSurfer();

        isVideo = info.is_video;
        const mediaUrl = `/api/media/${info.file_id}`;

        // Show/hide video panel
        if (isVideo) {
            videoPanel.classList.remove("hidden");
            nleEditor.classList.add("nle-video-mode");
            nleVideo.src = mediaUrl;
            nleVideo.load();
        } else {
            videoPanel.classList.add("hidden");
            nleEditor.classList.remove("nle-video-mode");
            nleVideo.src = "";
        }

        // Create WaveSurfer
        initWaveSurfer(mediaUrl, isVideo);

        // Reset segments
        segmentsList.innerHTML = "";
        regionCounter = 0;
        $('input[name="segment-mode"][value="full"]').checked = true;
        segmentsEditor.classList.add("hidden");
    }

    function initWaveSurfer(url, syncVideo) {
        // Regions plugin
        regionsPlugin = RegionsPlugin.create();

        // Timeline plugin
        const timelinePlugin = TimelinePlugin.create({
            container: "#nle-ruler",
            timeInterval: 1,
            primaryLabelInterval: 5,
            style: {
                fontSize: "11px",
                color: "#8888aa",
            },
        });

        const wsOptions = {
            container: waveformContainer,
            waveColor: "#6c63ff",
            progressColor: "#a29bfe",
            cursorColor: "#ff6b6b",
            cursorWidth: 2,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            height: syncVideo ? 100 : 160,
            normalize: true,
            mediaControls: false,
            plugins: [regionsPlugin, timelinePlugin],
            minPxPerSec: 1,
        };

        if (syncVideo) {
            // For video: use the video element as the media backend
            wsOptions.media = nleVideo;
        } else {
            wsOptions.url = url;
        }

        wavesurfer = WaveSurfer.create(wsOptions);

        // For video, load the audio waveform from the media URL
        if (syncVideo) {
            wavesurfer.load(url);
        }

        // Events
        wavesurfer.on("ready", () => {
            const duration = wavesurfer.getDuration();
            nleDuration.textContent = formatTimecodeHMS(duration);
            nleZoom.value = 1;
        });

        wavesurfer.on("timeupdate", (time) => {
            updateTimecode(time);
        });

        wavesurfer.on("play", () => {
            nlePlay.textContent = "⏸";
        });

        wavesurfer.on("pause", () => {
            nlePlay.textContent = "▶";
        });

        // Region events
        regionsPlugin.on("region-created", (region) => {
            const segMode = document.querySelector('input[name="segment-mode"]:checked');
            if (!segMode || segMode.value !== "segments") {
                region.remove();
                return;
            }
            regionCounter++;
            region._color = REGION_COLORS[(regionCounter - 1) % REGION_COLORS.length];
            region.setOptions({ color: region._color });
            addSegmentFromRegion(region);
        });

        regionsPlugin.on("region-updated", (region) => {
            updateSegmentFromRegion(region);
        });

        // Double-click region to play it
        regionsPlugin.on("region-double-clicked", (region, e) => {
            e.stopPropagation();
            region.play();
        });
    }

    function destroyWaveSurfer() {
        if (wavesurfer) {
            wavesurfer.destroy();
            wavesurfer = null;
            regionsPlugin = null;
        }
        nlePlay.textContent = "▶";
        nleTimecode.textContent = "00:00:00.00";
        nleDuration.textContent = "00:00:00.00";
        nleZoom.value = 1;
    }

    function updateTimecode(time) {
        nleTimecode.textContent = formatTimecodeHMS(time);
    }

    // --- Region ↔ Segment sync ---
    function addSegmentFromRegion(region) {
        const idx = segmentsList.children.length;
        const div = document.createElement("div");
        div.className = "segment-row";
        div.dataset.regionId = region.id;

        const color = region._color || REGION_COLORS[0];
        const solidColor = color.replace(/[\d.]+\)$/, "0.7)");

        div.innerHTML = `
            <span class="seg-color-dot" style="background:${solidColor}"></span>
            <input type="text" class="seg-label" placeholder="Label" value="Segment ${idx + 1}">
            <label>Début:</label>
            <input type="text" class="seg-start" value="${formatTimecodeHMS(region.start)}">
            <label>Fin:</label>
            <input type="text" class="seg-end" value="${formatTimecodeHMS(region.end)}">
            <button class="nle-btn nle-btn-sm seg-play" title="Écouter ce segment">▶</button>
            <button class="btn btn-small btn-danger seg-remove" title="Supprimer">✕</button>
        `;

        // Play segment
        div.querySelector(".seg-play").addEventListener("click", () => {
            region.play();
        });

        // Remove segment + region
        div.querySelector(".seg-remove").addEventListener("click", () => {
            region.remove();
            div.remove();
        });

        // Update region when inputs change
        const startInput = div.querySelector(".seg-start");
        const endInput = div.querySelector(".seg-end");

        const updateRegionFromInput = () => {
            const start = parseTimecodeHMS(startInput.value);
            const end = parseTimecodeHMS(endInput.value);
            if (start >= 0 && end > start) {
                region.setOptions({ start, end });
            }
        };

        startInput.addEventListener("change", updateRegionFromInput);
        endInput.addEventListener("change", updateRegionFromInput);

        segmentsList.appendChild(div);
    }

    function updateSegmentFromRegion(region) {
        const row = segmentsList.querySelector(`[data-region-id="${region.id}"]`);
        if (!row) return;
        row.querySelector(".seg-start").value = formatTimecodeHMS(region.start);
        row.querySelector(".seg-end").value = formatTimecodeHMS(region.end);
    }

    // --- Manual Segment Editor ---
    function addSegmentRow() {
        // Create a region on the waveform
        if (wavesurfer && regionsPlugin) {
            const duration = wavesurfer.getDuration();
            const start = 0;
            const end = Math.min(duration, 10);
            const color = REGION_COLORS[regionCounter % REGION_COLORS.length];
            regionsPlugin.addRegion({
                start,
                end,
                color,
                drag: true,
                resize: true,
            });
        } else {
            // Fallback without wavesurfer
            const idx = segmentsList.children.length;
            const div = document.createElement("div");
            div.className = "segment-row";
            div.innerHTML = `
                <span class="seg-color-dot"></span>
                <input type="text" class="seg-label" placeholder="Label" value="Segment ${idx + 1}">
                <label>Début:</label>
                <input type="text" class="seg-start" placeholder="00:00:00" value="00:00:00">
                <label>Fin:</label>
                <input type="text" class="seg-end" placeholder="00:00:00" value="${currentFileInfo ? currentFileInfo.duration_display : '00:00:00'}">
                <button class="btn btn-small btn-danger seg-remove" title="Supprimer">✕</button>
            `;
            div.querySelector(".seg-remove").addEventListener("click", () => div.remove());
            segmentsList.appendChild(div);
        }
    }

    function getSegments() {
        const mode = document.querySelector('input[name="segment-mode"]:checked').value;
        if (mode === "full") return [];

        const rows = segmentsList.querySelectorAll(".segment-row");
        const segments = [];
        for (const row of rows) {
            const label = row.querySelector(".seg-label").value || "";
            const startStr = row.querySelector(".seg-start").value;
            const endStr = row.querySelector(".seg-end").value;
            segments.push({
                label: label,
                start: parseTimecodeHMS(startStr),
                end: parseTimecodeHMS(endStr),
            });
        }
        return segments;
    }

    // --- Transcription ---
    async function startTranscription() {
        if (!currentFileId) return;

        showStep("progress");
        transcribeProgressFill.style.width = "0%";
        transcribeProgressText.textContent = "Démarrage...";
        spinner.classList.add("active");

        const segments = getSegments();
        const language = $("#language").value || null;
        const model = $("#model").value || null;

        try {
            const response = await fetch(`/api/transcribe/${currentFileId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ segments, language, model }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const event = JSON.parse(line.slice(6));
                            handleSSEEvent(event);
                        } catch {
                            // Skip malformed events
                        }
                    }
                }
            }
        } catch (err) {
            transcribeProgressText.textContent = `❌ Erreur : ${err.message}`;
            spinner.classList.remove("active");
        }
    }

    function handleSSEEvent(event) {
        switch (event.type) {
            case "progress":
                transcribeProgressFill.style.width = event.percent + "%";
                transcribeProgressText.textContent = event.message;
                break;

            case "complete":
                spinner.classList.remove("active");
                currentResult = event.result;
                showResult(event.result);
                loadHistory();
                break;

            case "error":
                spinner.classList.remove("active");
                transcribeProgressText.textContent = `❌ ${event.message}`;
                break;

            case "heartbeat":
                break;
        }
    }

    // --- Results ---
    function showResult(result) {
        showStep("result");

        resultMeta.innerHTML = `
            <div class="meta-grid">
                <div class="meta-item"><strong>Langue :</strong> ${result.language || "Auto"}</div>
                <div class="meta-item"><strong>Moteur :</strong> ${result.engine}</div>
                <div class="meta-item"><strong>Modèle :</strong> ${result.model}</div>
                <div class="meta-item"><strong>Appareil :</strong> ${result.device.toUpperCase()}</div>
                <div class="meta-item"><strong>Durée audio :</strong> ${formatDuration(result.duration_seconds)}</div>
                <div class="meta-item"><strong>Temps de traitement :</strong> ${formatDuration(result.processing_time_seconds)}</div>
            </div>
        `;

        resultText.value = result.text;

        // Show segments
        if (result.segments && result.segments.length > 0) {
            resultSegments.innerHTML = "<h3>Segments détaillés</h3>";
            const table = document.createElement("div");
            table.className = "segments-table";

            for (const seg of result.segments) {
                const row = document.createElement("div");
                row.className = "segment-result-row";
                row.innerHTML = `
                    <span class="seg-time">${formatTime(seg.start)} → ${formatTime(seg.end)}</span>
                    <span class="seg-text">${escapeHtml(seg.text)}</span>
                `;
                table.appendChild(row);
            }
            resultSegments.appendChild(table);
        } else {
            resultSegments.innerHTML = "";
        }
    }

    // --- Export ---
    async function copyToClipboard() {
        if (!resultText.value) return;
        try {
            await navigator.clipboard.writeText(resultText.value);
            $("#btn-copy").textContent = "✅ Copié !";
            setTimeout(() => ($("#btn-copy").textContent = "📋 Copier le texte"), 2000);
        } catch {
            resultText.select();
            document.execCommand("copy");
        }
    }

    function exportFile(format) {
        if (!currentFileId) return;
        window.open(`/api/export/${currentFileId}/${format}`, "_blank");
    }

    // --- History ---
    async function loadHistory() {
        try {
            const res = await fetch("/api/history");
            const history = await res.json();

            if (history.length === 0) {
                historyList.innerHTML = '<p class="history-empty">Aucune transcription précédente.</p>';
                return;
            }

            historyList.innerHTML = "";
            for (const entry of history) {
                const div = document.createElement("div");
                div.className = "history-entry";
                div.innerHTML = `
                    <div class="history-info">
                        <span class="history-name">${escapeHtml(entry.original_name)}</span>
                        <span class="history-meta">
                            ${entry.timestamp ? new Date(entry.timestamp).toLocaleString("fr-FR") : ""} —
                            ${entry.language || "auto"} — ${entry.model} (${entry.device})
                        </span>
                    </div>
                    <div class="history-actions">
                        <button class="btn btn-small" onclick="window._viewHistory('${entry.file_id}')">👁️ Voir</button>
                        <button class="btn btn-small" onclick="window._exportHistory('${entry.file_id}', 'txt')">📄 TXT</button>
                        <button class="btn btn-small" onclick="window._exportHistory('${entry.file_id}', 'srt')">🎬 SRT</button>
                        <button class="btn btn-small btn-danger" onclick="window._deleteHistory('${entry.file_id}')">🗑️</button>
                    </div>
                `;
                historyList.appendChild(div);
            }
        } catch {
            historyList.innerHTML = '<p class="history-empty">Erreur de chargement de l\'historique.</p>';
        }
    }

    window._viewHistory = async function (fileId) {
        try {
            const res = await fetch(`/api/result/${fileId}`);
            const result = await res.json();
            currentFileId = fileId;
            currentResult = result;
            showResult(result);
        } catch {
            alert("Impossible de charger le résultat.");
        }
    };

    window._exportHistory = function (fileId, format) {
        window.open(`/api/export/${fileId}/${format}`, "_blank");
    };

    window._deleteHistory = async function (fileId) {
        if (!confirm("Supprimer cette transcription ?")) return;
        try {
            await fetch(`/api/history/${fileId}`, { method: "DELETE" });
            loadHistory();
        } catch {
            alert("Erreur lors de la suppression.");
        }
    };

    // --- Reset ---
    function resetAll() {
        destroyWaveSurfer();
        currentFileId = null;
        currentFileInfo = null;
        currentResult = null;
        uploadList.innerHTML = "";
        uploadProgress.classList.add("hidden");
        fileInput.value = "";
        showStep("upload");
    }

    // --- Helpers ---
    function formatDuration(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m ${s}s`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }

    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 100);
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
    }

    function formatTimecodeHMS(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const cs = Math.floor((seconds % 1) * 100);
        return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
    }

    function parseTimecodeHMS(str) {
        const parts = str.split(":");
        if (parts.length === 3) {
            const h = Number(parts[0]);
            const m = Number(parts[1]);
            const secParts = parts[2].split(".");
            const s = Number(secParts[0]);
            const cs = secParts[1] ? Number(secParts[1]) / 100 : 0;
            return h * 3600 + m * 60 + s + cs;
        }
        if (parts.length === 2) {
            const m = Number(parts[0]);
            const secParts = parts[1].split(".");
            const s = Number(secParts[0]);
            const cs = secParts[1] ? Number(secParts[1]) / 100 : 0;
            return m * 60 + s + cs;
        }
        return Number(str) || 0;
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // --- Boot ---
    document.addEventListener("DOMContentLoaded", init);
})();
