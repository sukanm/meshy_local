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
const errorEl = document.getElementById("error");
const statusEl = document.getElementById("status");
const jobIdEl = document.getElementById("jobId");
const jobStatusEl = document.getElementById("jobStatus");
const stageEl = document.getElementById("stage");
const viewer = document.getElementById("viewer");
const reportEl = document.getElementById("report");
const downloadEl = document.getElementById("download");

let pollTimer = null;

(async function initDefaults() {
  try {
    const res = await fetch("/defaults");
    const d = await res.json();
    smoothingEl.value = d.smoothing_iterations;
    detailEl.value = d.voxel_resolution;
    candidatesEl.value = d.num_candidates;
  } catch {
    smoothingEl.value = 10;
    detailEl.value = 350;
    candidatesEl.value = 3;
  }
  smoothingValueEl.textContent = smoothingEl.value;
  detailValueEl.textContent = detailEl.value;
  candidatesValueEl.textContent = candidatesEl.value;
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

function renderReport(report) {
  if (!report) {
    reportEl.innerHTML = "";
    return;
  }
  const actions = (report.actions_taken || [])
    .map((a) => `<li>${a}</li>`)
    .join("");
  const issues = (report.remaining_issues || [])
    .map((i) => `<li>[${i.severity}] ${i.message}</li>`)
    .join("");
  reportEl.innerHTML = `
    <p class="${report.status}"><strong>${report.status}</strong></p>
    <p>Watertight: ${report.is_watertight} — Bounding box: ${
      report.bbox_mm ? report.bbox_mm.map((v) => v.toFixed(1)).join(" × ") + "mm" : "n/a"
    } — Fits P2S build volume: ${report.fits_build_volume}</p>
    ${actions ? `<p>Actions taken:</p><ul>${actions}</ul>` : ""}
    ${issues ? `<p>Issues:</p><ul>${issues}</ul>` : ""}
  `;
}

async function poll(jobId) {
  const res = await fetch(`/status/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();

  jobStatusEl.textContent = job.status;
  stageEl.textContent = job.current_stage ? `stage: ${job.current_stage}` : "";

  if (job.raw_mesh_path) {
    viewer.src = `/output/${jobId}/raw_mesh.glb?t=${Date.now()}`;
    viewer.hidden = false;
  }

  if (job.status === "SUCCEEDED") {
    clearInterval(pollTimer);
    renderReport(job.repair_report);
    downloadEl.href = `/output/${jobId}/model.stl`;
    downloadEl.style.display = "inline-block";
    generateBtn.disabled = false;
  } else if (job.status === "FAILED") {
    clearInterval(pollTimer);
    errorEl.textContent = `Job failed: ${job.error || "unknown error"}`;
    generateBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", async () => {
  errorEl.textContent = "";
  reportEl.innerHTML = "";
  downloadEl.style.display = "none";
  viewer.hidden = true;
  generateBtn.disabled = true;

  const body = {
    prompt: promptEl.value.trim(),
    scale_mm: parseFloat(scaleEl.value),
    image_model: imageModelEl.value,
    smoothing_iterations: parseInt(smoothingEl.value, 10),
    voxel_resolution: parseInt(detailEl.value, 10),
    num_candidates: parseInt(candidatesEl.value, 10),
  };
  if (seedEl.value !== "") {
    body.seed = parseInt(seedEl.value, 10);
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
