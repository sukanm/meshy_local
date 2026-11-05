const generateBtn = document.getElementById("generate");
const promptEl = document.getElementById("prompt");
const scaleEl = document.getElementById("scale");
const seedEl = document.getElementById("seed");
const imageModelEl = document.getElementById("imageModel");
const smoothingEl = document.getElementById("smoothing");
const smoothingValueEl = document.getElementById("smoothingValue");
const detailEl = document.getElementById("detail");
const detailValueEl = document.getElementById("detailValue");
const candidatesEl = document.getElementById("candidates");
const candidatesValueEl = document.getElementById("candidatesValue");
const wallModeEl = document.getElementById("wallMode");
const wallOffsetEl = document.getElementById("wallOffset");
const wallOffsetWrapEl = document.getElementById("wallOffsetWrap");
const baseModeEl = document.getElementById("baseMode");
const errorEl = document.getElementById("error");
const statusEl = document.getElementById("status");
const jobIdEl = document.getElementById("jobId");
const jobStatusEl = document.getElementById("jobStatus");
const stageEl = document.getElementById("stage");
const viewer = document.getElementById("viewer");
const viewerToggleEl = document.getElementById("viewerToggle");
const viewerCaptionEl = document.getElementById("viewerCaption");
const reportEl = document.getElementById("report");
const sliceEl = document.getElementById("slice");
const previewsEl = document.getElementById("previews");
const actionsEl = document.getElementById("actions");
const downloadEl = document.getElementById("download");
const download3mfEl = document.getElementById("download3mf");
const openBambuEl = document.getElementById("openBambu");
const openStatusEl = document.getElementById("openStatus");
const refPanelEl = document.getElementById("refPanel");
const refQueryEl = document.getElementById("refQuery");
const refSearchEl = document.getElementById("refSearch");
const refStatusEl = document.getElementById("refStatus");
const refResultsEl = document.getElementById("refResults");
const refUrlEl = document.getElementById("refUrl");
const refUseUrlEl = document.getElementById("refUseUrl");
const refFileEl = document.getElementById("refFile");
const refSelectedEl = document.getElementById("refSelected");
const refSelectedImgEl = document.getElementById("refSelectedImg");
const refSelectedInfoEl = document.getElementById("refSelectedInfo");
const refClearEl = document.getElementById("refClear");
const inputImageEl = document.getElementById("inputImage");
const inputImageImgEl = document.getElementById("inputImageImg");
const inputImageCaptionEl = document.getElementById("inputImageCaption");

const DEFAULT_WALL_AUTO_MM = 0.3;
const STAGE_LABELS = {
  reference_image: "using your reference image",
  text_to_image: "generating concept image",
  image_to_3d: "building 3D model",
  repair: "repairing mesh",
  print_prep: "print prep (thickening, base)",
  export: "exporting STL",
  preview: "rendering previews",
  slice: "slicing in Bambu Studio",
};

let pollTimer = null;
let currentJobId = null;
let viewerSources = { printable: null, raw: null };
let viewerMode = "raw";
let selectedRef = null; // { ref_id, image_url, width, height, label }
let searchResults = [];
let searchProvider = "";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function updateWallOffsetVisibility() {
  wallOffsetWrapEl.hidden = wallModeEl.value !== "custom";
}

(async function initDefaults() {
  let wallAuto = DEFAULT_WALL_AUTO_MM;
  try {
    const res = await fetch("/defaults");
    const d = await res.json();
    smoothingEl.value = d.smoothing_iterations;
    detailEl.value = d.voxel_resolution;
    candidatesEl.value = d.num_candidates;
    const fromServer = Number(d.wall_offset_auto_mm);
    if (Number.isFinite(fromServer) && fromServer > 0) wallAuto = fromServer;
    baseModeEl.value = ["flatten", "plinth", "none"].includes(d.base_mode) ? d.base_mode : "flatten";
  } catch {
    smoothingEl.value = 10;
    detailEl.value = 350;
    candidatesEl.value = 3;
    baseModeEl.value = "flatten";
  }
  wallOffsetEl.value = Math.min(2, Math.max(0.1, wallAuto));
  wallModeEl.value = "auto";
  const autoOption = wallModeEl.querySelector('option[value="auto"]');
  if (autoOption) autoOption.textContent = `Auto — ${wallAuto}mm only if walls are too thin (default)`;
  smoothingValueEl.textContent = smoothingEl.value;
  detailValueEl.textContent = detailEl.value;
  candidatesValueEl.textContent = candidatesEl.value;
  updateWallOffsetVisibility();
})();

smoothingEl.addEventListener("input", () => {
  smoothingValueEl.textContent = smoothingEl.value;
});
detailEl.addEventListener("input", () => {
  detailValueEl.textContent = detailEl.value;
});
candidatesEl.addEventListener("input", () => {
  candidatesValueEl.textContent = candidatesEl.value;
});
wallModeEl.addEventListener("change", updateWallOffsetVisibility);

// --- 3D viewer -------------------------------------------------------------

function setViewerSource(src) {
  // Only reload when the model actually changes (polling must not reset it).
  if (src && viewer.getAttribute("src") !== src) viewer.setAttribute("src", src);
  viewer.hidden = !src;
}

function showViewerMode(mode) {
  viewerMode = mode;
  setViewerSource(viewerSources[mode]);
  for (const btn of viewerToggleEl.querySelectorAll("button")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  viewerCaptionEl.textContent =
    mode === "printable"
      ? "Showing the printable model — what the STL and Bambu project contain."
      : "Showing the raw generated model — its colours and paper-thin sheets are NOT what will print.";
  viewerCaptionEl.hidden = !viewerSources[mode];
}

viewerToggleEl.addEventListener("click", (event) => {
  const btn = event.target.closest("button[data-mode]");
  if (btn) showViewerMode(btn.dataset.mode);
});

// --- Reference image (web search, link, upload) ------------------------------

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  return data;
}

function setRefStatus(text, isError = false) {
  refStatusEl.textContent = text;
  refStatusEl.classList.toggle("warn", isError);
}

function updateGenerateLabel() {
  generateBtn.textContent = selectedRef ? "Generate from reference image" : "Generate";
}

function selectRef(ref, label) {
  selectedRef = { ...ref, label };
  refSelectedImgEl.src = ref.image_url;
  refSelectedInfoEl.textContent = `${label} — ${ref.width}×${ref.height}px`;
  refSelectedEl.hidden = false;
  updateGenerateLabel();
}

function clearRef() {
  selectedRef = null;
  refSelectedEl.hidden = true;
  refSelectedImgEl.removeAttribute("src");
  updateGenerateLabel();
}

function renderSearchResults() {
  if (searchResults.length === 0) {
    refResultsEl.innerHTML = "";
    refResultsEl.hidden = true;
    return;
  }
  refResultsEl.innerHTML = searchResults
    .map((r, i) => {
      const title = escapeHtml(r.title || "image");
      const size = r.width && r.height ? ` (${Number(r.width)}×${Number(r.height)})` : "";
      return `<button type="button" data-index="${i}" title="${title}${size}"><img src="${escapeHtml(
        r.thumbnail_url
      )}" alt="${title}" loading="lazy" referrerpolicy="no-referrer" /></button>`;
    })
    .join("");
  refResultsEl.hidden = false;
}

async function runSearch() {
  const query = refQueryEl.value.trim() || promptEl.value.trim();
  if (!query) {
    setRefStatus("Type something to search for.", true);
    return;
  }
  refSearchEl.disabled = true;
  setRefStatus(`Searching for “${query}”…`);
  try {
    const res = await fetch(`/image-search?q=${encodeURIComponent(query)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    searchResults = data.results || [];
    searchProvider = data.provider || "";
    renderSearchResults();
    setRefStatus(
      searchResults.length
        ? `${searchResults.length} results from ${searchProvider}. Click one to use it — pick a single object, clearly visible.`
        : "No images found. Try different words."
    );
  } catch (e) {
    setRefStatus(`Search failed: ${e.message}`, true);
  } finally {
    refSearchEl.disabled = false;
  }
}

refSearchEl.addEventListener("click", runSearch);
refQueryEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    runSearch();
  }
});

refResultsEl.addEventListener("click", async (event) => {
  const btn = event.target.closest("button[data-index]");
  if (!btn) return;
  const result = searchResults[Number(btn.dataset.index)];
  if (!result) return;
  btn.classList.add("loading");
  setRefStatus("Downloading the full-size image…");
  try {
    const ref = await postJson("/reference-image/from-url", {
      url: result.image_url,
      fallback_url: result.thumbnail_url,
      title: result.title || null,
      page_url: result.page_url || null,
      provider: searchProvider || null,
    });
    selectRef(ref, result.title || "Web image");
    setRefStatus("");
  } catch (e) {
    setRefStatus(`Couldn't use that image: ${e.message}`, true);
  } finally {
    btn.classList.remove("loading");
  }
});

refUseUrlEl.addEventListener("click", async () => {
  const url = refUrlEl.value.trim();
  if (!url) return;
  refUseUrlEl.disabled = true;
  setRefStatus("Downloading the image…");
  try {
    const ref = await postJson("/reference-image/from-url", { url });
    selectRef(ref, "Image from link");
    setRefStatus("");
  } catch (e) {
    setRefStatus(`Couldn't use that link: ${e.message}`, true);
  } finally {
    refUseUrlEl.disabled = false;
  }
});

refFileEl.addEventListener("change", async () => {
  const file = refFileEl.files && refFileEl.files[0];
  refFileEl.value = "";
  if (!file) return;
  setRefStatus(`Uploading ${file.name}…`);
  try {
    const res = await fetch("/reference-image", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
    selectRef(data, file.name);
    setRefStatus("");
  } catch (e) {
    setRefStatus(`Couldn't use that file: ${e.message}`, true);
  }
});

refClearEl.addEventListener("click", clearRef);

// --- Report, slice summary, previews -----------------------------------------

function renderReport(report) {
  if (!report) {
    reportEl.innerHTML = "";
    return;
  }
  const actions = (report.actions_taken || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("");
  const issues = (report.remaining_issues || [])
    .map((i) => `<li>[${escapeHtml(i.severity)}] ${escapeHtml(i.message)}</li>`)
    .join("");
  const status = escapeHtml(report.status);
  reportEl.innerHTML = `
    <p class="${status}"><strong>${status}</strong></p>
    <p>Watertight: ${escapeHtml(report.is_watertight)} — Bounding box: ${
      report.bbox_mm ? report.bbox_mm.map((v) => Number(v).toFixed(1)).join(" × ") + "mm" : "n/a"
    } — Fits P2S build volume: ${escapeHtml(report.fits_build_volume)}</p>
    ${actions ? `<p>Actions taken:</p><ul>${actions}</ul>` : ""}
    ${issues ? `<p>Issues:</p><ul>${issues}</ul>` : ""}
  `;
}

function formatDuration(seconds) {
  const minutes = Math.round(Number(seconds) / 60);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function renderSlice(slice) {
  if (!slice) {
    sliceEl.innerHTML = "";
    sliceEl.hidden = true;
    return;
  }
  let body;
  if (slice.error) {
    body = `<p class="warn">Slicing failed: ${escapeHtml(slice.error)}</p>`;
  } else {
    const facts = [];
    if (slice.print_time_s != null) facts.push(`Print time: ${formatDuration(slice.print_time_s)}`);
    if (slice.filament_g != null) facts.push(`Filament: ${Number(slice.filament_g).toFixed(1)} g`);
    if (slice.layer_height_mm != null) facts.push(`Layers: ${Number(slice.layer_height_mm)} mm`);
    let features = "";
    if (Number(slice.features_total) > 0) {
      const share = slice.features_uncovered / slice.features_total;
      features = `<p class="${share > 0.2 ? "warn" : ""}">Small features the slicer prints no plastic for: ${
        Number(slice.features_uncovered)
      } of ${Number(slice.features_total)} (${Math.round(share * 100)}%)</p>`;
    }
    const warnings = (slice.warnings || []).map((w) => `<li>${escapeHtml(w)}</li>`).join("");
    body = `
      <p>${facts.map(escapeHtml).join(" · ")}</p>
      <p class="muted">${escapeHtml(slice.settings_summary)}</p>
      ${features}
      ${warnings ? `<ul class="warn">${warnings}</ul>` : ""}
    `;
  }
  sliceEl.innerHTML = `<h3>Bambu Studio slice (P2S)</h3>${body}`;
  sliceEl.hidden = false;
}

function renderPreviews(previewImages, jobId) {
  if (!Array.isArray(previewImages) || previewImages.length === 0) {
    previewsEl.innerHTML = "";
    previewsEl.hidden = true;
    return;
  }
  const t = Date.now();
  const items = previewImages
    .map((filename) => {
      const name = encodeURIComponent(filename);
      const caption = escapeHtml(
        String(filename).replace(/\.[a-z]+$/i, "").replace(/^preview_/, "").replace(/_/g, " ")
      );
      return `<figure><img src="/output/${jobId}/${name}?t=${t}" alt="${caption}" loading="lazy" /><figcaption>${caption}</figcaption></figure>`;
    })
    .join("");
  previewsEl.innerHTML = `<h3>Final STL preview</h3><div class="preview-grid">${items}</div>`;
  previewsEl.hidden = false;
}

function renderActions(job, jobId) {
  downloadEl.href = `/output/${jobId}/model.stl`;
  const project = job.slice_result && !job.slice_result.error ? job.slice_result.project_file : null;
  if (project) {
    download3mfEl.href = `/output/${jobId}/${encodeURIComponent(project)}`;
    download3mfEl.hidden = false;
  } else {
    download3mfEl.hidden = true;
  }
  openBambuEl.textContent = project ? "Open project in Bambu Studio" : "Open STL in Bambu Studio";
  openStatusEl.textContent = "";
  actionsEl.hidden = false;
}

function resetResults() {
  errorEl.textContent = "";
  reportEl.innerHTML = "";
  renderSlice(null);
  renderPreviews(null);
  actionsEl.hidden = true;
  download3mfEl.hidden = true;
  openStatusEl.textContent = "";
  viewerSources = { printable: null, raw: null };
  viewerToggleEl.hidden = true;
  viewerCaptionEl.hidden = true;
  viewer.removeAttribute("src");
  viewer.hidden = true;
  inputImageEl.hidden = true;
  inputImageImgEl.removeAttribute("src");
}

function renderInputImage(job, jobId) {
  if (!job.generated_image_path || !inputImageEl.hidden) return;
  const name = String(job.generated_image_path).split("/").pop();
  inputImageImgEl.src = `/output/${jobId}/${encodeURIComponent(name)}`;
  const ref = job.stage_config && job.stage_config.reference_image;
  inputImageCaptionEl.textContent = ref
    ? `Built from your reference image${ref.title ? `: ${ref.title}` : ""}`
    : "Built from this generated concept image";
  inputImageEl.hidden = false;
}

// --- Polling ----------------------------------------------------------------

async function poll(jobId) {
  const res = await fetch(`/status/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();

  jobStatusEl.textContent = job.status;
  stageEl.textContent = job.current_stage ? `stage: ${STAGE_LABELS[job.current_stage] || job.current_stage}` : "";
  renderInputImage(job, jobId);

  if (job.raw_mesh_path && !viewerSources.raw) {
    viewerSources.raw = `/output/${jobId}/raw_mesh.glb`;
    if (!viewerSources.printable) showViewerMode("raw");
  }

  if (job.status === "SUCCEEDED") {
    clearInterval(pollTimer);
    if (job.viewer_model) {
      viewerSources.printable = `/output/${jobId}/${encodeURIComponent(job.viewer_model)}`;
      viewerToggleEl.hidden = !viewerSources.raw;
      showViewerMode("printable");
    }
    renderReport(job.repair_report);
    renderSlice(job.slice_result);
    renderPreviews(job.preview_images, jobId);
    renderActions(job, jobId);
    generateBtn.disabled = false;
  } else if (job.status === "FAILED") {
    clearInterval(pollTimer);
    errorEl.textContent = `Job failed: ${job.error || "unknown error"}`;
    generateBtn.disabled = false;
  }
}

openBambuEl.addEventListener("click", async () => {
  if (!currentJobId) return;
  openBambuEl.disabled = true;
  openStatusEl.textContent = "Opening…";
  try {
    const res = await fetch(`/open-in-bambu/${currentJobId}`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    openStatusEl.textContent = `Opened ${data.opened} in Bambu Studio`;
  } catch (e) {
    openStatusEl.textContent = `Couldn't open: ${e.message}`;
  } finally {
    openBambuEl.disabled = false;
  }
});

generateBtn.addEventListener("click", async () => {
  resetResults();
  generateBtn.disabled = true;

  const wallOffset = parseFloat(wallOffsetEl.value);
  const body = {
    prompt: promptEl.value.trim(),
    scale_mm: parseFloat(scaleEl.value),
    image_model: imageModelEl.value,
    smoothing_iterations: parseInt(smoothingEl.value, 10),
    voxel_resolution: parseInt(detailEl.value, 10),
    num_candidates: parseInt(candidatesEl.value, 10),
    wall_offset_mm:
      wallModeEl.value === "auto" ? null :
      wallModeEl.value === "off" ? 0 :
      Number.isFinite(wallOffset) ? wallOffset : DEFAULT_WALL_AUTO_MM,
    base_mode: baseModeEl.value,
  };
  if (seedEl.value !== "") {
    body.seed = parseInt(seedEl.value, 10);
  }
  if (selectedRef) {
    body.reference_image_id = selectedRef.ref_id;
    if (!body.prompt) body.prompt = selectedRef.label.slice(0, 800);
  } else if (!body.prompt) {
    errorEl.textContent = "Enter a prompt or choose a reference image.";
    generateBtn.disabled = false;
    return;
  }

  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const { job_id } = await res.json();
    currentJobId = job_id;

    statusEl.hidden = false;
    jobIdEl.textContent = job_id;
    jobStatusEl.textContent = "PENDING";
    stageEl.textContent = "";

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => poll(job_id), 5000);
    poll(job_id);
  } catch (e) {
    errorEl.textContent = e.message;
    generateBtn.disabled = false;
  }
});

// Lets you reopen a finished job: http://127.0.0.1:8420/?job=<job_id>
(function loadJobFromUrl() {
  const jobId = new URLSearchParams(window.location.search).get("job");
  if (!jobId || !/^[a-f0-9]{6,32}$/.test(jobId)) return;
  currentJobId = jobId;
  statusEl.hidden = false;
  jobIdEl.textContent = jobId;
  pollTimer = setInterval(() => poll(jobId), 5000);
  poll(jobId);
})();
