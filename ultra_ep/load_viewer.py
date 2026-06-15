from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UltraEP Load Viewer</title>
<style>
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --ink: #17202a;
  --muted: #64748b;
  --line: #d8dde6;
  --soft-line: #edf1f5;
  --panel: #ffffff;
  --before: #c4512b;
  --after: #136f63;
  --focus: #1f8a70;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
button, select, input { font: inherit; }
.app { display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: 100vh; }
aside { border-right: 1px solid var(--line); background: #fbfcfd; padding: 18px 16px; }
main { padding: 18px 22px 28px; min-width: 0; }
h1 { margin: 0 0 18px; font-size: 22px; line-height: 1.15; font-weight: 760; }
h2 { margin: 20px 0 10px; font-size: 15px; text-transform: uppercase; letter-spacing: 0; color: #334155; }
label { display: block; margin: 14px 0 6px; font-size: 12px; color: var(--muted); font-weight: 680; }
select, button { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: white; color: var(--ink); min-height: 34px; padding: 6px 9px; }
button { cursor: pointer; background: #eef7f4; border-color: #b9d9cf; color: #0e594e; font-weight: 700; }
button:hover { border-color: var(--focus); }
input[type="range"] { accent-color: var(--focus); }
.mode-toggle { display: inline-flex; overflow: hidden; border: 1px solid var(--line); border-radius: 6px; background: #f8fafc; }
.mode-toggle button { width: auto; min-height: 30px; border: 0; border-radius: 0; padding: 4px 10px; background: transparent; color: #475569; font-size: 12px; font-weight: 760; }
.mode-toggle button.active { background: #17202a; color: #ffffff; }
.mode-toggle button:hover { border-color: transparent; color: #17202a; }
.mode-toggle button.active:hover { color: #ffffff; }
.stats { display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); gap: 1px; border: 1px solid var(--line); background: var(--line); margin-bottom: 16px; }
.stat { background: var(--panel); padding: 10px 12px; min-height: 58px; }
.stat .k { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
.stat .v { margin-top: 4px; font-size: 18px; font-weight: 760; white-space: nowrap; }
.chart-row { display: grid; grid-template-columns: minmax(360px, 0.9fr) minmax(540px, 1.45fr); gap: 18px; align-items: start; }
.chart-shell { border-top: 1px solid var(--line); padding-top: 10px; min-width: 0; }
.chart-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; min-height: 34px; margin-bottom: 10px; }
.chart-head h2 { margin: 0; }
.stack-controls { display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 9px; min-width: 430px; }
.stack-controls label { margin: 0; white-space: nowrap; }
.stack-controls input { width: min(220px, 22vw); min-width: 140px; }
.stack-controls > button { width: auto; min-height: 30px; padding: 4px 10px; }
.zoom-readout { color: var(--muted); font-size: 12px; font-weight: 760; text-align: right; }
canvas { display: block; width: 100%; height: 300px; background: white; border: 1px solid var(--line); }
#histCanvas { height: 390px; }
#stackCanvas { height: 390px; cursor: grab; touch-action: none; }
#stackCanvas.dragging { cursor: grabbing; }
.viewport-bar { display: none; position: relative; height: 18px; margin-top: 8px; border-radius: 999px; background: #e6ebf1; border: 1px solid #d6dde6; }
.viewport-bar.active { display: block; }
.viewport-handle { position: absolute; top: 3px; bottom: 3px; min-width: 18px; border-radius: 999px; background: #1f8a70; box-shadow: 0 2px 8px rgba(31, 138, 112, 0.24); cursor: grab; }
.viewport-handle:active { cursor: grabbing; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.detail-card { min-width: 0; }
.detail-card h3 { margin: 0 0 8px; font-size: 13px; color: #334155; }
.detail-card canvas { height: 430px; }
.legend { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin: 12px 0 16px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.tooltip { position: fixed; pointer-events: none; display: none; z-index: 20; max-width: 260px; padding: 8px 9px; border: 1px solid #cbd5e1; border-radius: 6px; background: rgba(255,255,255,0.98); box-shadow: 0 8px 24px rgba(15,23,42,0.14); font-size: 12px; color: #1f2937; }
.tooltip .t { font-weight: 760; margin-bottom: 3px; }
.muted { color: var(--muted); }
.error { color: #a43b20; font-weight: 700; }
@media (max-width: 1100px) { .chart-row, .detail-grid { grid-template-columns: 1fr; } }
@media (max-width: 900px) { .app { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); } .stats { grid-template-columns: repeat(2, minmax(120px, 1fr)); } .stack-controls { min-width: 0; justify-content: flex-start; } .stack-controls input { width: min(220px, 48vw); } }
</style>
</head>
<body>
<div class="app">
  <aside>
    <h1>UltraEP Load Viewer</h1>
    <label for="groupSelect">EP Group</label>
    <select id="groupSelect"></select>
    <label for="layerSelect">Layer</label>
    <select id="layerSelect"></select>
    <label for="recordSelect">Microbatch</label>
    <select id="recordSelect"></select>
    <button id="reloadBtn" title="Reload traces from disk">Reload</button>
    <p id="sideInfo" class="muted"></p>
  </aside>
  <main>
    <div id="error" class="error"></div>
    <div id="stats" class="stats"></div>
    <div class="chart-row">
      <section class="chart-shell">
        <div class="chart-head"><h2>Imbalance Distribution</h2></div>
        <canvas id="histCanvas" width="900" height="320"></canvas>
        <div class="legend"><span><i class="swatch" style="background: var(--before)"></i>Before</span><span><i class="swatch" style="background: var(--after)"></i>After</span></div>
      </section>
      <section class="chart-shell">
        <div class="chart-head">
          <h2>Per-Rank Load Stack</h2>
          <div class="stack-controls">
            <div id="modeToggle" class="mode-toggle" aria-label="Per-rank load stack mode">
              <button type="button" data-mode="post" class="active" title="After Balancing">After</button>
              <button type="button" data-mode="pre" title="Before Balancing">Before</button>
            </div>
            <label for="zoomRange">Zoom</label>
            <input id="zoomRange" type="range" min="1" max="1" value="1" step="0.01">
            <span id="zoomValue" class="zoom-readout">Fit</span>
            <button id="fitStackBtn" title="Show the full stack">Fit</button>
          </div>
        </div>
        <canvas id="stackCanvas" width="1200" height="440"></canvas>
        <div id="viewportBar" class="viewport-bar" title="Drag visible window"><div id="viewportHandle" class="viewport-handle"></div></div>
        <div id="stackLegend" class="legend"></div>
      </section>
    </div>
    <section class="chart-shell">
      <div class="chart-head"><h2>Microbatch Detail</h2></div>
      <div id="recordSummary" class="stats"></div>
      <div class="detail-grid">
        <div class="detail-card"><h3>Before Balancing</h3><canvas id="preDetailCanvas" width="900" height="470"></canvas></div>
        <div class="detail-card"><h3>After Balancing</h3><canvas id="postDetailCanvas" width="900" height="470"></canvas></div>
      </div>
    </section>
  </main>
</div>
<div id="tooltip" class="tooltip"></div>
<script>
const state = { summary: null, group: null, layer: null, layerData: null, record: null, stackMode: "post", viewStart: 0, viewSize: 1, canvasDrag: null, timelineDrag: null, panRemainder: 0, stackBars: [], stackPlot: null, detailHit: [] };
const rankColors = ["#2f6fbb", "#d66f2d", "#138a72", "#8c5cc2", "#b5445a", "#6b7d2c", "#287c9f", "#b06a8f", "#5b6ee1", "#c08b2f", "#2f855a", "#a3459b"];
const expertColors = ["#2f6fbb", "#d66f2d", "#138a72", "#8c5cc2", "#b5445a", "#6b7d2c", "#287c9f", "#b06a8f", "#5b6ee1", "#c08b2f", "#2f855a", "#a3459b", "#7c3aed", "#0f766e", "#be123c", "#a16207"];
const $ = id => document.getElementById(id);
function fmt(x, digits=3) { return Number.isFinite(x) ? Number(x).toFixed(digits) : "-"; }
function fmtInt(x) { return String(Math.round(Number.isFinite(x) ? x : 0)); }
function sum(arr) { return arr.reduce((a, b) => a + b, 0); }
function max(arr) { return arr.length ? Math.max(...arr) : 0; }
async function api(path) { const res = await fetch(path); if (!res.ok) throw new Error(await res.text()); return res.json(); }
function setError(msg="") { $("error").textContent = msg; }
function fillSelect(el, rows, labelFn, valueFn) { el.innerHTML = ""; for (const row of rows) { const opt = document.createElement("option"); opt.value = valueFn(row); opt.textContent = labelFn(row); el.appendChild(opt); } }
function statBox(k, v) { return `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`; }
function setupCanvas(canvas) { const dpr = window.devicePixelRatio || 1; const rect = canvas.getBoundingClientRect(); canvas.width = Math.max(1, Math.floor(rect.width * dpr)); canvas.height = Math.max(1, Math.floor(rect.height * dpr)); const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, w: rect.width, h: rect.height }; }
function pathRoundRect(ctx, x, y, w, h, r) { const rr = Math.max(0, Math.min(r, w / 2, h / 2)); ctx.beginPath(); ctx.moveTo(x + rr, y); ctx.lineTo(x + w - rr, y); ctx.quadraticCurveTo(x + w, y, x + w, y + rr); ctx.lineTo(x + w, y + h - rr); ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h); ctx.lineTo(x + rr, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - rr); ctx.lineTo(x, y + rr); ctx.quadraticCurveTo(x, y, x + rr, y); ctx.closePath(); }
function drawAxes(ctx, w, h, pad, ymax, xlabel, opts={}) { const yFormat = opts.yFormat || fmtInt; const axisY = h - pad.b; const innerW = w - pad.l - pad.r; ctx.strokeStyle = "#edf1f5"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(w - pad.r, pad.t); ctx.stroke(); ctx.strokeStyle = "#d8dde6"; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, axisY); ctx.lineTo(w - pad.r, axisY); ctx.stroke(); ctx.fillStyle = "#64748b"; ctx.font = "13px system-ui"; if (opts.showYLabels !== false) { ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(yFormat(ymax), pad.l - 10, pad.t); ctx.fillText(yFormat(0), pad.l - 10, axisY); } ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; ctx.fillText(xlabel, pad.l + innerW / 2, Math.min(h - 6, axisY + (opts.xlabelOffset || 38))); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; }
function drawCenteredTicks(ctx, ticks, y, color="#64748b") { ctx.fillStyle = color; ctx.font = "12px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "top"; for (const tick of ticks) ctx.fillText(tick.label, tick.x, y); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; }
function niceStep(raw) { const power = Math.pow(10, Math.floor(Math.log10(Math.max(1, raw)))); const norm = raw / power; return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * power; }
function makeMicrobatchTicks(ctx, records, pad, bw) { if (!records.length) return []; ctx.font = "12px system-ui"; const axisLeft = pad.l, axisRight = pad.l + records.length * bw; const plotW = axisRight - axisLeft; let maxWidth = 0; const sampleStep = Math.max(1, Math.floor(records.length / 96)); for (let i = 0; i < records.length; i += sampleStep) maxWidth = Math.max(maxWidth, ctx.measureText(String(records[i].microbatch)).width); maxWidth = Math.max(maxWidth, ctx.measureText(String(records[records.length - 1].microbatch)).width); const minTickPx = Math.max(72, maxWidth + 32); const targetTicks = Math.max(2, Math.floor(plotW / minTickPx)); const step = niceStep(Math.ceil(records.length / targetTicks)); const candidates = new Set([0, records.length - 1]); for (let i = 0; i < records.length; i += step) candidates.add(i); const ticks = []; const gap = 16; function tickFor(i) { const label = String(records[i].microbatch); const half = ctx.measureText(label).width / 2; const rawX = pad.l + i * bw + bw / 2; const x = Math.min(Math.max(rawX, axisLeft + half), axisRight - half); return {x, label, left: x - half, right: x + half}; } for (const i of [...candidates].sort((a, b) => a - b)) { const tick = tickFor(i); if (i === records.length - 1) { while (ticks.length && tick.left < ticks[ticks.length - 1].right + gap) ticks.pop(); } if (!ticks.length || tick.left >= ticks[ticks.length - 1].right + gap) ticks.push(tick); } return ticks; }
function drawBarTopLabels(ctx, labels, barGap) { if (!labels.length) return; ctx.font = "12px system-ui"; const widths = labels.map(l => ctx.measureText(l.label).width); const maxWidth = Math.max(...widths); if (barGap < 16 || maxWidth > barGap * 2.4) return; const stride = Math.max(1, Math.ceil((maxWidth + 12) / Math.max(barGap, 1))); if (stride > 4) return; ctx.fillStyle = "#17202a"; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; let lastRight = -Infinity; for (let i = 0; i < labels.length; i++) { if (i % stride !== 0 && i !== labels.length - 1) continue; const half = widths[i] / 2; const left = labels[i].x - half, right = labels[i].x + half; if (left < lastRight + 10) continue; ctx.fillText(labels[i].label, labels[i].x, labels[i].y); lastRight = right; } ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; }
function showTip(html, x, y) { const tip = $("tooltip"); tip.innerHTML = html; tip.style.left = `${x + 12}px`; tip.style.top = `${y + 12}px`; tip.style.display = "block"; }
function hideTip() { $("tooltip").style.display = "none"; }
function currentGroup() { return state.summary?.groups.find(g => g.id === state.group); }
function selectedRecordIndex() { if (!state.layerData || !state.record) return -1; return state.layerData.records.findIndex(r => r.record_id === state.record.record_id); }
function clampView() { const n = state.layerData?.records.length || 1; state.viewSize = Math.max(1, Math.min(Math.round(state.viewSize || n), n)); state.viewStart = Math.max(0, Math.min(Math.round(state.viewStart || 0), n - state.viewSize)); }
function visibleRecords() { const data = state.layerData; if (!data) return []; clampView(); return data.records.slice(state.viewStart, state.viewStart + state.viewSize); }
function syncViewportBar() { const bar = $("viewportBar"), handle = $("viewportHandle"); const n = state.layerData?.records.length || 1; if (!bar || !handle) return; if (n <= 1 || state.viewSize >= n) { bar.classList.remove("active"); return; } bar.classList.add("active"); handle.style.left = `${state.viewStart / n * 100}%`; handle.style.width = `${state.viewSize / n * 100}%`; }
function syncStackControls(reset=false) { const n = state.layerData?.records.length || 1; if (reset) { state.viewStart = 0; state.viewSize = n; } clampView(); const zoomRange = $("zoomRange"); const zoom = Math.max(1, n / state.viewSize); zoomRange.min = 1; zoomRange.max = Math.max(1, n); zoomRange.value = zoom; $("zoomValue").textContent = state.viewSize >= n ? "Fit" : `${zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1)}x`; syncViewportBar(); }
function setViewAround(anchorIdx, newSize) { const n = state.layerData?.records.length || 1; const oldSize = state.viewSize; const anchor = oldSize > 0 ? (anchorIdx - state.viewStart + 0.5) / oldSize : 0.5; state.viewSize = Math.max(1, Math.min(Math.round(newSize), n)); state.viewStart = Math.round(anchorIdx + 0.5 - anchor * state.viewSize); clampView(); syncStackControls(); drawStack(); }
function panView(deltaRecords) { if (!deltaRecords) return; state.viewStart += deltaRecords; clampView(); syncStackControls(); drawStack(); }
function drawHist(group) { const canvas = $("histCanvas"); const {ctx, w, h} = setupCanvas(canvas); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "white"; ctx.fillRect(0, 0, w, h); if (!group || !group.hist_bins.length) return; const pad = {l: 28, r: 28, t: 24, b: 58}; const pre = group.hist_pre, post = group.hist_post, bins = group.hist_bins; const ymax = Math.max(1, Math.ceil(max(pre)), Math.ceil(max(post))); drawAxes(ctx, w, h, pad, ymax, "max / mean", {showYLabels: false}); const n = pre.length, innerW = w - pad.l - pad.r, innerH = h - pad.t - pad.b, axisY = h - pad.b; const bw = innerW / Math.max(n, 1); for (let i = 0; i < n; i++) { const x = pad.l + i * bw; const hp = innerH * pre[i] / ymax, ha = innerH * post[i] / ymax; ctx.fillStyle = "rgba(196, 81, 43, 0.55)"; ctx.fillRect(x + 1, axisY - hp, Math.max(1, bw * 0.46), hp); ctx.fillStyle = "rgba(19, 111, 99, 0.62)"; ctx.fillRect(x + bw * 0.50, axisY - ha, Math.max(1, bw * 0.46), ha); } const ticks = []; const step = Math.max(1, Math.ceil(bins.length / 5)); for (let i = 0; i < bins.length; i += step) ticks.push({x: pad.l + (i + 0.5) * bw, label: fmt(bins[i], 2)}); drawCenteredTicks(ctx, ticks, axisY + 8); }
function drawStack() { const canvas = $("stackCanvas"); const {ctx, w, h} = setupCanvas(canvas); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "white"; ctx.fillRect(0, 0, w, h); const data = state.layerData; if (!data || !data.records.length) return; syncStackControls(); const mode = state.stackMode; const allLoads = mode === "pre" ? data.pre_rank_loads : data.post_rank_loads; const records = visibleRecords(); const loads = allLoads.slice(state.viewStart, state.viewStart + state.viewSize); const pad = {l: 28, r: 28, t: 28, b: 58}; const totals = loads.map(row => sum(row)); const ymax = Math.max(1, Math.ceil(max(totals))); drawAxes(ctx, w, h, pad, ymax, "microbatch", {showYLabels: false}); const innerW = w - pad.l - pad.r, innerH = h - pad.t - pad.b, axisY = h - pad.b; const bw = innerW / Math.max(loads.length, 1); state.stackPlot = {pad, barStep: bw, axisY}; state.stackBars = []; const selected = selectedRecordIndex(); const selectedVisible = selected >= state.viewStart && selected < state.viewStart + state.viewSize; if (selectedVisible) { const local = selected - state.viewStart; const x = pad.l + local * bw; ctx.fillStyle = "rgba(15, 23, 42, 0.08)"; ctx.fillRect(x, pad.t, Math.max(1, bw - 1), innerH); }
  for (let i = 0; i < loads.length; i++) { let y = axisY; const x = pad.l + i * bw; for (let r = 0; r < loads[i].length; r++) { const segH = innerH * loads[i][r] / ymax; ctx.fillStyle = rankColors[r % rankColors.length]; ctx.fillRect(x, y - segH, Math.max(1, bw - 1), segH); if (segH > 13 && bw > 16) { ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.font = "10px system-ui"; ctx.fillText(String(loads[i][r]), x + 3, y - segH + 11); } state.stackBars.push({x, y: y - segH, w: Math.max(1, bw - 1), h: segH, record: records[i], rank: r, load: loads[i][r]}); y -= segH; } }
  if (selectedVisible) { const local = selected - state.viewStart; const x = pad.l + local * bw; const barW = Math.max(2, bw - 1); pathRoundRect(ctx, x + 1, pad.t + 1, Math.max(2, barW - 2), innerH - 2, Math.min(6, barW / 2)); ctx.strokeStyle = "rgba(255, 255, 255, 0.94)"; ctx.lineWidth = 4; ctx.stroke(); ctx.strokeStyle = "rgba(15, 23, 42, 0.72)"; ctx.lineWidth = 1.5; ctx.stroke(); const markerW = Math.min(30, Math.max(10, barW * 0.72)); pathRoundRect(ctx, x + (barW - markerW) / 2, pad.t - 10, markerW, 6, 3); ctx.fillStyle = "#f8fafc"; ctx.fill(); ctx.strokeStyle = "rgba(15, 23, 42, 0.72)"; ctx.lineWidth = 1; ctx.stroke(); }
  drawCenteredTicks(ctx, makeMicrobatchTicks(ctx, records, pad, bw), axisY + 8); renderStackLegend(loads[0].length); }
function drawDetail(canvasId, byRank, totals, title, globalExpertBase, colorOffset) { const canvas = $(canvasId); const {ctx, w, h} = setupCanvas(canvas); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "white"; ctx.fillRect(0, 0, w, h); const pad = {l: 66, r: 18, t: 28, b: 58}; const ymax = Math.max(1, Math.ceil(max(totals))); drawAxes(ctx, w, h, pad, ymax, "rank", {yFormat: fmtInt}); const innerW = w - pad.l - pad.r, innerH = h - pad.t - pad.b, axisY = h - pad.b; const gap = innerW / Math.max(byRank.length, 1); const bw = Math.max(3, gap * 0.72); const hits = [], topLabels = [];
  for (let r = 0; r < byRank.length; r++) { const x = pad.l + r * gap + (gap - bw) / 2; let y = axisY; for (let e = 0; e < byRank[r].length; e++) { const load = byRank[r][e]; if (load <= 0) continue; const segH = innerH * load / ymax; const color = expertColors[(e + colorOffset) % expertColors.length]; ctx.fillStyle = color; ctx.fillRect(x, y - segH, bw, segH); if (segH > 15 && bw > 22) { ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.font = "10px system-ui"; ctx.fillText(String(load), x + 4, y - segH + 12); } hits.push({canvasId, x, y: y - segH, w: bw, h: segH, rank: r, expert: globalExpertBase(r, e), localExpert: e, load, title, color}); y -= segH; }
    topLabels.push({x: x + bw / 2, y: Math.max(13, y - 6), label: String(totals[r])}); }
  drawBarTopLabels(ctx, topLabels, gap); const tickStep = Math.max(1, Math.ceil(28 / Math.max(gap, 1))); const ticks = []; for (let r = 0; r < byRank.length; r++) { if (r % tickStep === 0 || r === byRank.length - 1) ticks.push({x: pad.l + r * gap + gap / 2, label: String(r)}); } drawCenteredTicks(ctx, ticks, axisY + 8); state.detailHit = state.detailHit.filter(x => x.canvasId !== canvasId).concat(hits); }
function renderStackLegend(rankCount) { if (rankCount > 24) { const samples = [0, 1, 2, 3, 4, 5].filter(r => r < rankCount).map(r => `<span><i class="swatch" style="background:${rankColors[r % rankColors.length]}"></i>${r}</span>`).join(""); $("stackLegend").innerHTML = `<span>${rankCount} ranks stacked; hover for per-rank load</span>${samples}`; return; } $("stackLegend").innerHTML = Array.from({length: rankCount}, (_, r) => `<span><i class="swatch" style="background:${rankColors[r % rankColors.length]}"></i>rank ${r}</span>`).join(""); }
function renderDetailCharts() { const r = state.record; if (!r) return; const meta = currentGroup(); drawDetail("preDetailCanvas", r.pre_by_rank, r.pre_rank_loads, "Before Balancing", (rank, e) => rank * meta.num_local_master_experts + e, 0); drawDetail("postDetailCanvas", r.post_by_rank, r.post_rank_loads, "After Balancing", (rank, e) => rank * meta.num_local_physical_experts + e, 4); }
function renderStats(group) { if (!group) { $("stats").innerHTML = ""; return; } $("stats").innerHTML = [statBox("Records", group.records), statBox("Layers", group.layers.length), statBox("Ranks", group.ep_size), statBox("Before p50", fmt(group.pre_ratio.p50)), statBox("After p50", fmt(group.post_ratio.p50)), statBox("After p99", fmt(group.post_ratio.p99))].join(""); $("sideInfo").textContent = `${group.num_global_logical_experts} logical experts, ${group.num_global_physical_experts} physical experts`; }
async function loadSummary(reload=false) { setError(); state.summary = await api(`/api/summary${reload ? "?reload=1" : ""}`); const groups = state.summary.groups; fillSelect($("groupSelect"), groups, g => g.label, g => g.id); if (!groups.length) { setError("No trace chunks found."); return; } state.group = groups[0].id; $("groupSelect").value = state.group; await renderGroup(); }
async function renderGroup() { const group = currentGroup(); renderStats(group); drawHist(group); fillSelect($("layerSelect"), group.layers, x => `Layer ${x}`, x => x); state.layer = String(group.layers[0]); $("layerSelect").value = state.layer; await loadLayer(true); }
async function loadLayer(resetView=false) { const group = state.group, layer = $("layerSelect").value; state.layer = layer; state.layerData = await api(`/api/layer?group=${encodeURIComponent(group)}&layer=${encodeURIComponent(layer)}`); fillSelect($("recordSelect"), state.layerData.records, r => `#${r.microbatch}`, r => r.record_id); syncStackControls(resetView); drawStack(); if (state.layerData.records.length) await loadRecord(state.layerData.records[0].record_id); }
async function loadRecord(recordId) { state.record = await api(`/api/record?group=${encodeURIComponent(state.group)}&record=${encodeURIComponent(recordId)}`); $("recordSelect").value = recordId; renderRecord(); drawStack(); }
function renderRecord() { const r = state.record; if (!r) return; $("recordSummary").innerHTML = [statBox("Layer", r.layer), statBox("Microbatch", `#${r.microbatch}`), statBox("Before", fmt(r.pre_ratio)), statBox("After", fmt(r.post_ratio)), statBox("Before max", max(r.pre_rank_loads)), statBox("After max", max(r.post_rank_loads))].join(""); renderDetailCharts(); }
function stackIndexFromX(x, rect) { const data = state.layerData; if (!data || !data.records.length) return -1; const plot = state.stackPlot; const padL = plot?.pad.l ?? 28, padR = plot?.pad.r ?? 28; const innerW = rect.width - padL - padR; const local = Math.floor((x - padL) / (innerW / state.viewSize)); const idx = state.viewStart + local; return Math.max(0, Math.min(data.records.length - 1, idx)); }
function setStackMode(mode) { state.stackMode = mode; for (const btn of document.querySelectorAll("#modeToggle button")) btn.classList.toggle("active", btn.dataset.mode === mode); drawStack(); }
$("groupSelect").addEventListener("change", async e => { state.group = e.target.value; await renderGroup(); });
$("layerSelect").addEventListener("change", () => loadLayer(true));
for (const btn of document.querySelectorAll("#modeToggle button")) btn.addEventListener("click", () => setStackMode(btn.dataset.mode));
$("recordSelect").addEventListener("change", e => loadRecord(e.target.value));
$("reloadBtn").addEventListener("click", () => loadSummary(true).catch(err => setError(err.message)));
$("fitStackBtn").addEventListener("click", () => { syncStackControls(true); drawStack(); });
$("zoomRange").addEventListener("input", e => { const n = state.layerData?.records.length || 1; const zoom = Number(e.target.value); const selected = selectedRecordIndex(); const anchor = selected >= 0 ? selected : state.viewStart + Math.floor(state.viewSize / 2); setViewAround(anchor, Math.max(1, Math.round(n / Math.max(1, zoom)))); });
$("stackCanvas").addEventListener("mousedown", e => { const rect = e.currentTarget.getBoundingClientRect(); state.canvasDrag = {x: e.clientX, start: state.viewStart, moved: false, rectW: rect.width}; e.currentTarget.classList.add("dragging"); });
$("viewportBar").addEventListener("mousedown", e => { if (!state.layerData) return; const n = state.layerData.records.length; if (state.viewSize >= n) return; const bar = $("viewportBar"), handle = $("viewportHandle"), rect = bar.getBoundingClientRect(); if (e.target !== handle) { const center = (e.clientX - rect.left) / rect.width * n; state.viewStart = Math.round(center - state.viewSize / 2); clampView(); syncStackControls(); drawStack(); } state.timelineDrag = {x: e.clientX, start: state.viewStart, trackW: rect.width}; e.preventDefault(); });
window.addEventListener("mousemove", e => { if (state.timelineDrag && state.layerData) { const n = state.layerData.records.length; const dx = e.clientX - state.timelineDrag.x; state.viewStart = Math.round(state.timelineDrag.start + dx / Math.max(state.timelineDrag.trackW, 1) * n); clampView(); syncStackControls(); drawStack(); return; } if (!state.canvasDrag || !state.layerData) return; const dx = e.clientX - state.canvasDrag.x; const per = state.stackPlot?.barStep || ((state.canvasDrag.rectW - 84) / Math.max(state.viewSize, 1)); if (Math.abs(dx) > 3) state.canvasDrag.moved = true; state.viewStart = Math.round(state.canvasDrag.start - dx / Math.max(per, 1)); clampView(); syncStackControls(); drawStack(); });
window.addEventListener("mouseup", e => { if (state.timelineDrag) { state.timelineDrag = null; return; } if (!state.canvasDrag) return; const drag = state.canvasDrag; state.canvasDrag = null; $("stackCanvas").classList.remove("dragging"); if (!drag.moved) { const rect = $("stackCanvas").getBoundingClientRect(); const idx = stackIndexFromX(e.clientX - rect.left, rect); const rec = state.layerData?.records[idx]; if (rec) loadRecord(rec.record_id).catch(err => setError(err.message)); } });
$("stackCanvas").addEventListener("wheel", e => { if (!state.layerData) return; const absX = Math.abs(e.deltaX), absY = Math.abs(e.deltaY); const rect = e.currentTarget.getBoundingClientRect(); if (!e.ctrlKey && absX > absY * 1.15) { e.preventDefault(); const per = state.stackPlot?.barStep || 1; state.panRemainder += e.deltaX / Math.max(per, 1); const step = Math.trunc(state.panRemainder); if (step) { state.panRemainder -= step; panView(step); } return; } if (e.ctrlKey || absY > absX * 1.15) { e.preventDefault(); const n = state.layerData.records.length; const idx = stackIndexFromX(e.clientX - rect.left, rect); const oldSize = state.viewSize; const factor = 1.18; const nextSize = e.deltaY > 0 ? Math.min(n, Math.max(oldSize + 1, Math.ceil(oldSize * factor))) : Math.max(1, Math.min(oldSize - 1, Math.floor(oldSize / factor))); if (nextSize !== oldSize) setViewAround(idx, nextSize); } }, {passive: false});
$("stackCanvas").addEventListener("mousemove", e => { const rect = e.currentTarget.getBoundingClientRect(); const x = e.clientX - rect.left, y = e.clientY - rect.top; const hit = (state.stackBars || []).find(b => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h); if (!hit) { hideTip(); return; } showTip(`<div class="t">#${hit.record.microbatch}</div><div>rank ${hit.rank}</div><div>load ${hit.load}</div>`, e.clientX, e.clientY); });
$("stackCanvas").addEventListener("mouseleave", hideTip);
for (const canvasId of ["preDetailCanvas", "postDetailCanvas"]) { $(canvasId).addEventListener("mousemove", e => { const rect = e.currentTarget.getBoundingClientRect(); const x = e.clientX - rect.left, y = e.clientY - rect.top; const hit = state.detailHit.find(b => b.canvasId === canvasId && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h); if (!hit) { hideTip(); return; } const kind = canvasId === "preDetailCanvas" ? "logical" : "physical"; showTip(`<div class="t">${hit.title}</div><div>rank ${hit.rank}</div><div>${kind} expert ${hit.expert}</div><div>local slot ${hit.localExpert}</div><div>load ${hit.load}</div>`, e.clientX, e.clientY); }); $(canvasId).addEventListener("mouseleave", hideTip); }
window.addEventListener("resize", () => { drawHist(currentGroup()); drawStack(); renderDetailCharts(); });
loadSummary().catch(err => setError(err.message));
</script>
</body>
</html>
"""


def _safe_ratio(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1)
    out = np.zeros_like(mean, dtype=np.float64)
    mask = mean > 0
    out[mask] = values.max(axis=1)[mask] / mean[mask]
    return out


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: 0.0 for k in ("min", "p50", "p90", "p99", "max", "mean")}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _hist(
    before: np.ndarray, after: np.ndarray
) -> tuple[list[float], list[int], list[int]]:
    if before.size == 0 and after.size == 0:
        return [], [], []
    vmax = max(float(before.max(initial=1.0)), float(after.max(initial=1.0)), 1.0)
    bins = np.linspace(1.0, vmax, 33)
    pre, edges = np.histogram(before, bins=bins)
    post, _ = np.histogram(after, bins=bins)
    return (
        edges[:-1].round(4).tolist(),
        pre.astype(int).tolist(),
        post.astype(int).tolist(),
    )


class TraceStore:
    def __init__(self, root: Path):
        self.root = root
        self.groups: dict[str, dict[str, object]] = {}
        self.load()

    def load(self) -> None:
        groups: dict[str, dict[str, object]] = {}
        for path in sorted(self.root.glob("*.chunk*.npz")):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                group_id = metadata["trace_id"]
                group = groups.setdefault(
                    group_id,
                    {
                        "metadata": metadata,
                        "layers": [],
                        "virtual_layers": [],
                        "microbatches": [],
                        "pre": [],
                        "post": [],
                    },
                )
                group["layers"].append(data["layers"].astype(np.int32))
                group["virtual_layers"].append(data["virtual_layers"].astype(np.int32))
                group["microbatches"].append(data["microbatches"].astype(np.int64))
                group["pre"].append(data["pre_logical_loads"].astype(np.int32))
                group["post"].append(data["post_physical_loads"].astype(np.int32))
        for group in groups.values():
            for key in ("layers", "virtual_layers", "microbatches", "pre", "post"):
                group[key] = np.concatenate(group[key], axis=0)
        self.groups = groups

    def summary(self) -> dict[str, object]:
        rows = []
        for group_id, group in self.groups.items():
            meta = group["metadata"]
            pre_rank, post_rank = self._rank_loads(group)
            pre_ratio = _safe_ratio(pre_rank)
            post_ratio = _safe_ratio(post_rank)
            bins, hist_pre, hist_post = _hist(pre_ratio, post_ratio)
            layers = sorted(int(x) for x in np.unique(group["layers"]))
            group_index = len(rows)
            rows.append(
                {
                    "id": group_id,
                    "label": f"group {group_index} / size {meta.get('ep_size', 0)}",
                    "records": int(group["layers"].shape[0]),
                    "layers": layers,
                    "ep_size": int(meta["ep_size"]),
                    "num_local_master_experts": int(meta["num_local_master_experts"]),
                    "num_local_physical_experts": int(
                        meta["num_local_physical_experts"]
                    ),
                    "num_global_logical_experts": int(
                        meta["num_global_logical_experts"]
                    ),
                    "num_global_physical_experts": int(
                        meta["num_global_physical_experts"]
                    ),
                    "pre_ratio": _stats(pre_ratio),
                    "post_ratio": _stats(post_ratio),
                    "hist_bins": bins,
                    "hist_pre": hist_pre,
                    "hist_post": hist_post,
                }
            )
        return {"root": str(self.root), "groups": rows}

    def layer(self, group_id: str, layer: int) -> dict[str, object]:
        group = self._group(group_id)
        idx = np.nonzero(group["layers"] == layer)[0]
        idx = idx[np.argsort(group["microbatches"][idx], kind="stable")]
        pre_rank, post_rank = self._rank_loads(group)
        records = [
            {
                "record_id": int(i),
                "microbatch": int(group["microbatches"][i]),
                "pre_ratio": float(_safe_ratio(pre_rank[i : i + 1])[0]),
                "post_ratio": float(_safe_ratio(post_rank[i : i + 1])[0]),
            }
            for i in idx
        ]
        return {
            "records": records,
            "pre_rank_loads": pre_rank[idx].astype(int).tolist(),
            "post_rank_loads": post_rank[idx].astype(int).tolist(),
        }

    def record(self, group_id: str, record_id: int) -> dict[str, object]:
        group = self._group(group_id)
        i = int(record_id)
        if i < 0 or i >= group["layers"].shape[0]:
            raise KeyError(f"record not found: {record_id}")
        meta = group["metadata"]
        ep_size = int(meta["ep_size"])
        local_master = int(meta["num_local_master_experts"])
        local_physical = int(meta["num_local_physical_experts"])
        pre_by_rank = group["pre"][i].reshape(ep_size, local_master)
        post_by_rank = group["post"][i].reshape(ep_size, local_physical)
        pre_rank = pre_by_rank.sum(axis=1)
        post_rank = post_by_rank.sum(axis=1)
        return {
            "record_id": i,
            "layer": int(group["layers"][i]),
            "virtual_layer": int(group["virtual_layers"][i]),
            "microbatch": int(group["microbatches"][i]),
            "pre_ratio": float(_safe_ratio(pre_rank.reshape(1, -1))[0]),
            "post_ratio": float(_safe_ratio(post_rank.reshape(1, -1))[0]),
            "pre_logical_loads": group["pre"][i].astype(int).tolist(),
            "post_physical_loads": group["post"][i].astype(int).tolist(),
            "pre_by_rank": pre_by_rank.astype(int).tolist(),
            "post_by_rank": post_by_rank.astype(int).tolist(),
            "pre_rank_loads": pre_rank.astype(int).tolist(),
            "post_rank_loads": post_rank.astype(int).tolist(),
        }

    def _group(self, group_id: str) -> dict[str, object]:
        if group_id not in self.groups:
            raise KeyError(f"group not found: {group_id}")
        return self.groups[group_id]

    @staticmethod
    def _rank_loads(group: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        meta = group["metadata"]
        ep_size = int(meta["ep_size"])
        local_master = int(meta["num_local_master_experts"])
        local_physical = int(meta["num_local_physical_experts"])
        pre = group["pre"].reshape(-1, ep_size, local_master).sum(axis=2)
        post = group["post"].reshape(-1, ep_size, local_physical).sum(axis=2)
        return pre, post


def _send_json(
    handler: BaseHTTPRequestHandler, payload: object, status: int = 200
) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_text(
    handler: BaseHTTPRequestHandler,
    text: str,
    status: int = 200,
    content_type: str = "text/html",
) -> None:
    data = text.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", f"{content_type}; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(store: TraceStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    _send_text(self, INDEX_HTML)
                elif parsed.path == "/api/summary":
                    if query.get("reload", ["0"])[0] == "1":
                        store.load()
                    _send_json(self, store.summary())
                elif parsed.path == "/api/layer":
                    _send_json(
                        self,
                        store.layer(
                            query.get("group", [""])[0],
                            int(query.get("layer", ["0"])[0]),
                        ),
                    )
                elif parsed.path == "/api/record":
                    _send_json(
                        self,
                        store.record(
                            query.get("group", [""])[0],
                            int(query.get("record", ["0"])[0]),
                        ),
                    )
                else:
                    _send_json(self, {"error": "not found"}, 404)
            except Exception as exc:
                _send_json(self, {"error": str(exc)}, 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="View UltraEP expert traces")
    parser.add_argument("--path", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.path = args.path.expanduser().resolve()
    store = TraceStore(args.path)
    if not store.groups:
        raise SystemExit(f"No valid UltraEP trace chunks found in {args.path}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"UltraEP load viewer: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
