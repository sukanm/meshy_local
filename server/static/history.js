// History page: every job in output/, newest first, with a side-by-side
// compare of up to four. Compare views are addressable:
// history.html?compare=<id>,<id>[,...]

const MAX_COMPARE = 4;
const PREVIEW_ROWS = [
  ["preview_front.png", "Front"],
  ["preview_side.png", "Side"],
  ["preview_back.png", "Back"],
  ["preview_three_quarter.png", "Three-quarter"],
];

const gridEl = document.getElementById("grid");
const filterEl = document.getElementById("filter");
const onlyFinishedEl = document.getElementById("onlyFinished");
const countEl = document.getElementById("count");
const listStatusEl = document.getElementById("listStatus");
const compareBarEl = document.getElementById("compareBar");
const compareCountEl = document.getElementById("compareCount");
const compareGoEl = document.getElementById("compareGo");
const compareClearEl = document.getElementById("compareClear");
const listViewEl = document.getElementById("listView");
const compareViewEl = document.getElementById("compareView");
const compareTableEl = document.getElementById("compareTable");
const backToListEl = document.getElementById("backToList");
const toggle3dEl = document.getElementById("toggle3d");

let jobs = [];
let byId = new Map();
const chosen = []; // job ids, in the order picked
let show3d = false;

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fileUrl(job, name) {
  return `/output/${encodeURIComponent(job.job_id)}/${encodeURIComponent(name)}`;
}

function pct(fraction) {
  if (fraction == null) return null;
  const v = Number(fraction) * 100;
  // One decimal below 10%, so close values don't print the same.
  return v > 0 && v < 10 ? `${v.toFixed(1)}%` : `${Math.round(v)}%`;
}

function featureShare(job) {
  return Number(job.features_total) > 0 ? job.features_uncovered / job.features_total : null;
}

function formatDate(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatDuration(seconds) {
  if (seconds == null) return null;
  const minutes = Math.round(Number(seconds) / 60);
  const h = Math.floor(minutes / 60);
  return h > 0 ? `${h}h ${minutes % 60}m` : `${minutes}m`;
}

function shortModel(name) {
  if (!name) return null;
  if (/z-image/i.test(name)) return "Z-Image Turbo";
  if (/qwen-image/i.test(name)) return name.split("/").pop();
  return name.split("/").pop();
}

function statusTag(job) {
  if (job.status === "FAILED") return '<span class="tag bad">Failed</span>';
  if (job.status !== "SUCCEEDED") return `<span class="tag warn">${esc(job.status || "unknown")}</span>`;
  if (job.readiness === "NEEDS_MANUAL_REPAIR") return '<span class="tag bad">Needs repair</span>';
  return "";
}

// --- List -------------------------------------------------------------------

function matches(job, text) {
  if (onlyFinishedEl.checked && job.status !== "SUCCEEDED") return false;
  if (!text) return true;
  return `${job.prompt || ""} ${job.job_id}`.toLowerCase().includes(text);
}

function cardHtml(job) {
  const thumb = job.thumbnail
    ? `<img src="${esc(fileUrl(job, job.thumbnail))}" alt="" loading="lazy" />`
    : '<div class="none">No image</div>';
  const facts = [];
  const share = featureShare(job);
  if (share != null) {
    facts.push(
      `<span class="${share > 0.2 ? "warn" : ""}">${Number(job.features_uncovered)}/${Number(job.features_total)} unprinted</span>`
    );
  }
  if (job.below_floor_fraction != null) facts.push(`${pct(job.below_floor_fraction)} &lt;1.2mm`);
  const models = [shortModel(job.text_to_image_model), job.image_model].filter(Boolean).map(esc).join(" → ");
  const isChosen = chosen.includes(job.job_id);
  return `
    <article class="card${isChosen ? " chosen" : ""}" data-id="${esc(job.job_id)}">
      <a class="thumb" href="/?job=${encodeURIComponent(job.job_id)}" title="Open this job">${thumb}</a>
      <div class="body">
        <div class="prompt" title="${esc(job.prompt)}">${esc(job.prompt || "(no prompt)")}</div>
        <div class="facts">${esc(formatDate(job.created_at))}${job.scale_mm ? ` · ${Number(job.scale_mm)}mm` : ""}</div>
        ${models ? `<div class="facts">${models}</div>` : ""}
        ${facts.length ? `<div class="facts">${facts.join(" · ")}</div>` : ""}
        ${statusTag(job)}
        <label><input type="checkbox" data-compare="${esc(job.job_id)}" ${isChosen ? "checked" : ""} /> Compare</label>
      </div>
    </article>`;
}

function renderList() {
  const text = filterEl.value.trim().toLowerCase();
  const visible = jobs.filter((j) => matches(j, text));
  gridEl.innerHTML = visible.map(cardHtml).join("");
  countEl.textContent = `${visible.length} of ${jobs.length} jobs`;
  listStatusEl.textContent = visible.length ? "" : "No jobs match.";
  renderCompareBar();
}

function renderCompareBar() {
  compareBarEl.hidden = chosen.length === 0;
  compareCountEl.textContent =
    chosen.length >= MAX_COMPARE
      ? `${chosen.length} selected (the most that fit side by side)`
      : `${chosen.length} selected${chosen.length < 2 ? " — pick at least 2" : ""}`;
  compareGoEl.disabled = chosen.length < 2;
}

gridEl.addEventListener("change", (event) => {
  const box = event.target.closest("input[data-compare]");
  if (!box) return;
  const id = box.dataset.compare;
  const at = chosen.indexOf(id);
  if (box.checked && at === -1) {
    if (chosen.length >= MAX_COMPARE) {
      box.checked = false;
      compareCountEl.textContent = `You can compare up to ${MAX_COMPARE} at once.`;
      return;
    }
    chosen.push(id);
  } else if (!box.checked && at !== -1) {
    chosen.splice(at, 1);
  }
  box.closest(".card").classList.toggle("chosen", box.checked);
  renderCompareBar();
});

filterEl.addEventListener("input", renderList);
onlyFinishedEl.addEventListener("change", renderList);
compareClearEl.addEventListener("click", () => {
  chosen.length = 0;
  renderList();
});
compareGoEl.addEventListener("click", () => openCompare(chosen.slice()));

// --- Compare ----------------------------------------------------------------

// Marks the lowest value in a row (ties all marked) when at least two jobs have one.
function bestFlags(values) {
  const nums = values.filter((v) => v != null && Number.isFinite(v));
  if (nums.length < 2) return values.map(() => false);
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  if (min === max) return values.map(() => false);
  return values.map((v) => v != null && v === min);
}

function row(label, cells, best = null) {
  const tds = cells
    .map((html, i) => `<td${best && best[i] ? ' class="best"' : ""}>${html ?? '<span class="muted">—</span>'}</td>`)
    .join("");
  return `<tr><th scope="row">${esc(label)}</th>${tds}</tr>`;
}

function renderCompare(list) {
  const head = list
    .map(
      (j) => `<th scope="col">
        <div>${esc(j.prompt || "(no prompt)")}</div>
        <div class="muted"><a href="/?job=${encodeURIComponent(j.job_id)}">${esc(j.job_id)}</a> · ${esc(formatDate(j.created_at))}</div>
      </th>`
    )
    .join("");
  const rows = [];

  rows.push(
    row(
      "Input image",
      list.map((j) =>
        j.input_image ? `<img class="input" src="${esc(fileUrl(j, j.input_image))}" alt="Input image" loading="lazy" />` : null
      )
    )
  );
  if (show3d) {
    rows.push(
      row(
        "Printable model",
        list.map((j) =>
          j.viewer_model
            ? `<model-viewer src="${esc(fileUrl(j, j.viewer_model))}" camera-controls exposure="1" shadow-intensity="1"></model-viewer>`
            : null
        )
      )
    );
  }
  if (list.some((j) => j.previews.length)) {
    rows.push(
      row(
        "Final model",
        list.map((j) => {
          const views = PREVIEW_ROWS.filter(([file]) => j.previews.includes(file));
          if (!views.length) return null;
          return `<div class="views">${views
            .map(
              ([file, label]) =>
                `<figure><img src="${esc(fileUrl(j, file))}" alt="${esc(label)} view" loading="lazy" /><figcaption>${esc(label)}</figcaption></figure>`
            )
            .join("")}</div>`;
        })
      )
    );
  }

  const shares = list.map(featureShare);
  rows.push(
    row(
      "Unprinted features",
      list.map((j, i) =>
        shares[i] == null ? null : `${Number(j.features_uncovered)} of ${Number(j.features_total)} (${pct(shares[i])})`
      ),
      bestFlags(shares)
    )
  );
  const floor = list.map((j) => (j.below_floor_fraction == null ? null : Number(j.below_floor_fraction)));
  rows.push(row("Surface below 1.2mm", floor.map(pct), bestFlags(floor)));
  const nozzle = list.map((j) => (j.below_nozzle_fraction == null ? null : Number(j.below_nozzle_fraction)));
  rows.push(row("Surface below nozzle width", nozzle.map(pct), bestFlags(nozzle)));
  rows.push(
    row(
      "Result",
      list.map((j) => {
        if (j.status !== "SUCCEEDED") return `<span class="bad">${esc(j.status)}</span>${j.error ? `<div class="muted">${esc(j.error)}</div>` : ""}`;
        const cls = j.readiness === "PRINT_READY" ? "good" : j.readiness === "NEEDS_MANUAL_REPAIR" ? "bad" : "warn";
        return `<span class="${cls}">${esc((j.readiness || "").replace(/_/g, " ").toLowerCase())}</span>`;
      })
    )
  );
  rows.push(row("Wall thickening", list.map((j) => (j.status !== "SUCCEEDED" ? null : j.wall_offset_mm ? `${j.wall_offset_mm}mm` : "none"))));
  rows.push(row("Base", list.map((j) => (j.base ? (j.base.includes("skipped") ? `<span class="warn">${esc(j.base)}</span>` : esc(j.base)) : null))));
  rows.push(row("Size", list.map((j) => (j.bbox_mm ? `${j.bbox_mm.map((v) => Number(v).toFixed(0)).join(" × ")}mm` : null))));
  rows.push(row("Print time", list.map((j) => esc(formatDuration(j.print_time_s)) || null)));
  rows.push(row("Filament", list.map((j) => (j.filament_g != null ? `${Number(j.filament_g).toFixed(1)}g` : null))));
  rows.push(row("Image source", list.map((j) => esc(j.image_source) || null)));
  rows.push(row("Image model", list.map((j) => esc(shortModel(j.text_to_image_model)) || null)));
  rows.push(row("3D model", list.map((j) => (j.image_model ? `${esc(j.image_model)}${j.num_candidates ? ` (best of ${Number(j.num_candidates)})` : ""}` : null))));
  rows.push(row("Seed", list.map((j) => (j.seed != null ? esc(j.seed) : null))));
  rows.push(row("Target size", list.map((j) => (j.scale_mm ? `${Number(j.scale_mm)}mm` : null))));
  rows.push(
    row(
      "Warnings",
      list.map((j) => {
        const items = (j.issues || []).filter((i) => i.severity !== "info");
        if (j.slice_error) items.push({ severity: "warning", message: `Slicing failed: ${j.slice_error}` });
        return items.length
          ? `<ul>${items.map((i) => `<li class="${i.severity === "blocking" ? "bad" : ""}">${esc(i.message)}</li>`).join("")}</ul>`
          : "none";
      })
    )
  );

  compareTableEl.innerHTML = `<colgroup><col style="width:130px">${list.map(() => "<col>").join("")}</colgroup>
    <thead><tr><th scope="row"></th>${head}</tr></thead><tbody>${rows.join("")}</tbody>`;
  toggle3dEl.textContent = show3d ? "Hide 3D models" : "Show 3D models";
}

function openCompare(ids, push = true) {
  const list = ids.map((id) => byId.get(id)).filter(Boolean).slice(0, MAX_COMPARE);
  if (list.length < 2) return false;
  if (push) history.pushState(null, "", `?compare=${list.map((j) => encodeURIComponent(j.job_id)).join(",")}`);
  listViewEl.hidden = true;
  compareViewEl.hidden = false;
  renderCompare(list);
  window.scrollTo(0, 0);
  return true;
}

function showList(push = true) {
  if (push) history.pushState(null, "", "history.html");
  compareViewEl.hidden = true;
  listViewEl.hidden = false;
  renderList();
}

function routeFromUrl() {
  const ids = (new URLSearchParams(location.search).get("compare") || "").split(",").filter(Boolean);
  if (ids.length >= 2 && openCompare(ids, false)) {
    chosen.length = 0;
    chosen.push(...ids.filter((id) => byId.has(id)).slice(0, MAX_COMPARE));
    return;
  }
  showList(false);
}

backToListEl.addEventListener("click", () => showList());
toggle3dEl.addEventListener("click", () => {
  show3d = !show3d;
  const ids = (new URLSearchParams(location.search).get("compare") || "").split(",");
  renderCompare(ids.map((id) => byId.get(id)).filter(Boolean));
});
window.addEventListener("popstate", routeFromUrl);

(async function load() {
  try {
    const res = await fetch("/jobs");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    jobs = (await res.json()).jobs || [];
    byId = new Map(jobs.map((j) => [j.job_id, j]));
    routeFromUrl();
  } catch (e) {
    listStatusEl.textContent = `Couldn't load jobs: ${e.message}`;
  }
})();
