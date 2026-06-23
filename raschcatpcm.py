from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import random
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template_string, request, send_file, send_from_directory, session, url_for

DEFAULT_BUNDLE = Path(os.environ.get("CAT_BUNDLE", str(Path(__file__).with_name("replay_bundle.zip"))))
SECRET_KEY = os.environ.get("CAT_SECRET_KEY", "rasch-pcm-cat-demo-secret")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")

try:
    from gtts import gTTS
except Exception:
    gTTS = None

TTS_CACHE_DIR = Path(tempfile.gettempdir()) / "raschcat_pcm_tts_cache"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

HOME_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1300px; margin: 24px auto; padding: 0 16px; line-height: 1.55; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .stat { padding: 12px; background: #f8fafc; border-radius: 10px; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .btn-secondary { background:#475569; }
    .muted { color: #666; }
    .mono { font-family: Consolas, monospace; }
    select, input { padding: 8px; width: 100%; box-sizing: border-box; }
    .dash-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap:18px; align-items:start; }
    .wm-wrap { overflow:auto; border:1px solid #e5e7eb; border-radius: 12px; padding: 8px; background:white; }
    .map-dot { cursor:pointer; }
    .map-dot:hover { stroke-width: 2.4 !important; }
    .detail-img { max-width:100%; max-height:280px; border-radius:10px; border:1px solid #ddd; display:block; margin-top:10px; }
    .pill { display:inline-block; padding:4px 8px; background:#eef2ff; border-radius:999px; margin-right:8px; margin-bottom:6px; }
    .small { font-size: 0.92rem; }
    .ok { color:#166534; font-weight:700; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <p><strong>Polytomous Rasch-CAT (Partial Credit Model)</strong> using <span class="mono">replay_bundle.zip</span>.</p>
    <div class="grid">
      <div class="stat"><strong>Model</strong><br>{{ summary.model }}</div>
      <div class="stat"><strong>Items</strong><br>{{ summary.n_items }}</div>
      <div class="stat"><strong>Prior mean</strong><br>{{ '%.3f'|format(summary.prior_mean) }}</div>
      <div class="stat"><strong>Prior SD</strong><br>{{ '%.3f'|format(summary.prior_sd) }}</div>
      <div class="stat"><strong>Score range</strong><br>{{ summary.score_range }}</div>
      <div class="stat"><strong>PCM step source</strong><br>{{ summary.step_source }}</div>
    </div>
  </div>

  <div class="card">
    <h2>ReadMe</h2>
    <p class="muted">This Rasch PCM CAT uses <strong>item-specific step structures</strong> and selects the next CAT item by <strong>PCM polytomous item information</strong>, i.e. the score variance under that item's own category probabilities at the current theta. In non-CAT mode, all items are administered in fixed item-number order. In voice practice, items are synchronized one by one with their linked picture and server-generated Chinese / English MP3 audio.</p>
    <p class="muted">CAT vs non-CAT(1) uses <strong>one full-answer pattern generated once</strong>. Full non-CAT is estimated once from all items, then CAT is repeated <strong>n</strong> times by following the same full answers while randomizing the first CAT item and continuing with ordinary CAT item selection. CAT vs non-CAT(n) simulates <strong>n</strong> independent persons; each person receives one full non-CAT estimate and one CAT estimate under the same SE stop rule.</p>
  </div>

  <div class="card">
    <h2>CAT stopping criterion</h2>
    <div class="grid">
      <div class="stat"><strong>Sample-data Cronbach’s α</strong><br>{{ '%.3f'|format(cat_criterion.alpha) }}</div>
      <div class="stat"><strong>Response data source</strong><br>{{ cat_criterion.source }}</div>
      <div class="stat"><strong>Theta SD used</strong><br>{{ '%.3f'|format(cat_criterion.theta_sd) }}</div>
      <div class="stat"><strong>Computed stop SE</strong><br>{{ '%.3f'|format(cat_criterion.stop_se) }}</div>
      <div class="stat"><strong>Sample persons</strong><br>{{ cat_criterion.n_persons }}</div>
      <div class="stat"><strong>Items used for α / CAT</strong><br>{{ cat_criterion.n_items }} / {{ summary.n_items }}</div>
      <div class="stat"><strong>Full-bank SE median</strong><br>{{ '%.3f'|format(cat_criterion.full_se_median) }}</div>
      <div class="stat"><strong>Full-bank SE range</strong><br>{{ '%.3f'|format(cat_criterion.full_se_min) }}–{{ '%.3f'|format(cat_criterion.full_se_max) }}</div>
    </div>
    <p class="muted">The default CAT stopping rule is calculated from the provided sample response data using Cronbach’s alpha as the reliability target: <span class="mono">Stop SE = theta SD × sqrt(1 − α)</span>. Only the 30 questionnaire items are used for alpha and CAT; the class / melanoma-status outcome column is excluded. The start-test field below is pre-filled with this value, but it can still be edited for sensitivity analysis.</p>
    <p class="muted"><strong>Why smaller SE may not change CAT length:</strong> {{ cat_criterion.feasibility_note }}</p>
  </div>


  <div class="card">
    <h2>CAT stopping criterion based on Cronbach’s alpha</h2>
    <div class="grid">
      <div class="stat"><strong>Cronbach’s α</strong><br>{{ '%.4f'|format(cat_criterion.alpha) }}</div>
      <div class="stat"><strong>Theta SD</strong><br>{{ '%.4f'|format(cat_criterion.theta_sd) }}</div>
      <div class="stat"><strong>Computed stop SE</strong><br>{{ '%.4f'|format(cat_criterion.stop_se) }}</div>
      <div class="stat"><strong>Simulated persons</strong><br>{{ cat_criterion.n_persons }}</div>
    </div>
    <p class="muted">
      Cronbach’s alpha is used as the reliability target:
      <span class="mono">Stop SE = theta SD × sqrt(1 − alpha)</span>.
      If this SE cannot be reached by the 30-item bank, CAT stops at the maximum item limit.
    </p>
  </div>

  <div class="card">
    <h2>Start test</h2>
    <form method="post" action="{{ url_for('start_test') }}">
      <div class="grid">
        <div>
          <label>Language</label>
          <select name="language">
            <option value="en" selected>English</option>
            <option value="zh">Traditional Chinese</option>
          </select>
        </div>
        <div>
          <label>Mode</label>
          <select name="mode">
            <option value="cat">CAT</option>
            <option value="cat_seq_sim">CAT sequential simulation</option>
            <option value="linear">non-CAT</option>
            <option value="voice">Voice practice</option>
            <option value="compare">CAT vs non-CAT(1)</option>
            <option value="compare_n">CAT vs non-CAT(n)</option>
          </select>
        </div>
        <div>
          <label>Maximum CAT items</label>
          <input type="number" name="max_items" min="1" max="{{ summary.n_items }}" value="{{ summary.n_items }}">
        </div>
        <div>
          <label>Stop CAT when posterior SE ≤ (Cronbach’s α-based default)</label>
          <input type="number" name="stop_se" step="0.001" value="{{ '%.3f'|format(cat_criterion.stop_se) }}">
        </div>
        <div>
          <label>Starting theta</label>
          <input type="number" name="start_theta" step="0.1" value="{{ '%.2f'|format(summary.prior_mean) }}">
        </div>
        <div>
          <label>Starting item (non-CAT)</label>
          <input type="number" name="start_item" min="1" max="{{ summary.n_items }}" value="1">
        </div>
        <div>
          <label>Theta range +/- (voice only)</label>
          <input type="number" name="theta_range" min="0.2" max="4" step="0.1" value="1.0">
        </div>
        <div>
          <label>Simulated/candidate persons N</label>
          <input type="number" name="compare_n" min="1" max="1000" value="20">
        </div>
      </div>
      <div style="margin-top:14px;"><button class="btn" type="submit">Start</button></div>
    </form>
  </div>

  <div class="card">
    <h2>Interactive item dashboards</h2>
    <p class="muted">Hover or click a bubble in either dashboard to inspect the item, its linked picture, and server-generated Chinese / English MP3 audio. No fixed item is pre-selected.</p>
    <div class="dash-grid">
      <div>
        <h3 style="margin-top:0;">Person–item overview (Wright Map)</h3>
        <div class="wm-wrap" id="wrightMapWrap">{{ wright_svg|safe }}</div>
        <div style="margin-top:10px;">
          <button class="btn btn-secondary" type="button" onclick="downloadInlineSvgAsPng('wrightMapWrap', 'wright_map_homepage.png')">Download Wright Map PNG</button>
        </div>
      </div>
      <div>
        <h3 style="margin-top:0;">Reference-person KIDMAP</h3>
        <div class="wm-wrap" id="kidmapWrap">{{ kidmap_home_svg|safe }}</div>
        <div style="margin-top:10px;">
          <button class="btn btn-secondary" type="button" onclick="downloadInlineSvgAsPng('kidmapWrap', 'reference_person_kidmap_homepage.png')">Download KIDMAP PNG</button>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:18px;">
      <h3 style="margin-top:0;">Item inspector</h3>
      <div id="detailEmpty" class="muted">Hover or click a Wright Map or Reference-person KIDMAP bubble to inspect that item. The synchronized item-by-item display is also shown in Voice practice mode.</div>
      <div id="detailBox" style="display:none;">
        <div style="margin-bottom:10px;">
          <span class="pill" id="detailItemNo"></span>
          <span class="pill" id="detailItemId"></span>
        </div>
        <div class="small"><strong>Item label</strong></div>
        <div id="detailStemEn" style="margin-bottom:8px;"></div>
        <div class="small"><strong>Chinese</strong></div>
        <div id="detailStemZh" style="margin-bottom:12px;"></div>
        <div class="small"><strong>Statistics</strong></div>
        <div id="detailStats" class="small" style="margin-bottom:12px;"></div>
        <div style="margin-bottom:12px;">
          <button class="btn" type="button" onclick="speakSelected('zh')">Play Chinese MP3</button>
          <button class="btn btn-secondary" type="button" onclick="speakSelected('en')">Read English</button>
        </div>
        <div id="detailFigureWrap"></div>
      </div>
    </div>
  </div>

<script>
const dashboardItems = {{ wright_items|tojson }};
const itemMap = Object.fromEntries(dashboardItems.map(x => [x.item_id, x]));
let selectedItemId = null;
function fmtNum(v){ return (v === null || v === undefined || Number.isNaN(Number(v))) ? '' : Number(v).toFixed(3); }
function selectDashboardItem(itemId){
  const it = itemMap[itemId];
  if(!it) return;
  selectedItemId = itemId;
  document.getElementById('detailEmpty').style.display = 'none';
  document.getElementById('detailBox').style.display = '';
  document.getElementById('detailItemNo').textContent = `No. ${it.no}`;
  document.getElementById('detailItemId').textContent = it.item_id;
  document.getElementById('detailStemEn').textContent = it.stem_en || '';
  document.getElementById('detailStemZh').textContent = it.stem_zh || '';
  document.getElementById('detailStats').innerHTML = `Measure = <strong>${fmtNum(it.measure)}</strong> | Delta = <strong>${fmtNum(it.delta)}</strong> | INFIT MNSQ = <strong>${fmtNum(it.infit)}</strong> | OUTFIT MNSQ = <strong>${fmtNum(it.outfit)}</strong> | SE = <strong>${fmtNum(it.se)}</strong> | INFIT ZSTD = <strong>${fmtNum(it.infit_zstd)}</strong> | OUTFIT ZSTD = <strong>${fmtNum(it.outfit_zstd)}</strong>`;
  const fig = document.getElementById('detailFigureWrap');
  if(it.link_href && it.is_image_link){
    fig.innerHTML = `<img class="detail-img" src="${it.link_href}" alt="item image"><div class="muted small" style="margin-top:6px;">Picture linked from the response_category link column or /pic folder.</div>`;
  } else if(it.link_href){
    fig.innerHTML = `<a class="btn btn-secondary" href="${it.link_href}" target="_blank" rel="noopener">Open linked figure</a>`;
  } else {
    fig.innerHTML = `<div class="muted small">No linked picture matched this item.</div>`;
  }
  document.querySelectorAll('.map-dot').forEach(el => {
    el.setAttribute('fill-opacity', el.dataset.itemId === itemId ? '1.0' : '0.78');
    el.setAttribute('stroke-width', el.dataset.itemId === itemId ? '2.4' : '1.2');
  });
}
let detailAudio = null;
function stopDetailAudio(){
  if(detailAudio){ try{ detailAudio.pause(); detailAudio.currentTime = 0; }catch(e){} detailAudio = null; }
}
function downloadInlineSvgAsPng(wrapperId, filename){
  const wrap = document.getElementById(wrapperId);
  if(!wrap){ alert('Chart container not found.'); return; }
  const svg = wrap.querySelector('svg');
  if(!svg){ alert('No SVG chart is available for download.'); return; }
  const serializer = new XMLSerializer();
  let svgText = serializer.serializeToString(svg);
  if(!svgText.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)){
    svgText = svgText.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  const bboxW = Number(svg.getAttribute('width')) || Number(svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) || 1200;
  const bboxH = Number(svg.getAttribute('height')) || Number(svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height) || 600;
  const img = new Image();
  const blob = new Blob([svgText], {type:'image/svg+xml;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  img.onload = function(){
    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bboxW * scale));
    canvas.height = Math.max(1, Math.round(bboxH * scale));
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    canvas.toBlob(function(pngBlob){
      if(!pngBlob){ alert('PNG export failed.'); return; }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(pngBlob);
      a.download = filename || 'chart.png';
      document.body.appendChild(a);
      a.click();
      setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); }, 500);
    }, 'image/png');
  };
  img.onerror = function(){ URL.revokeObjectURL(url); alert('Unable to render SVG as PNG.'); };
  img.src = url;
}

function speakSelected(lang){
  if(!selectedItemId) return;
  const normalized = (String(lang||'').toLowerCase().startsWith('zh')) ? 'zh' : 'en';
  const url = `{{ url_for('voice_tts') }}?item_id=${encodeURIComponent(selectedItemId)}&language=${encodeURIComponent(normalized)}`;
  stopDetailAudio();
  try{
    detailAudio = new Audio(url);
    detailAudio.preload = 'auto';
    const pp = detailAudio.play();
    if(pp && typeof pp.catch === 'function'){ pp.catch(()=>{}); }
  }catch(e){}
}
document.addEventListener('DOMContentLoaded', ()=>{
  document.querySelectorAll('.map-dot').forEach(el => {
    el.addEventListener('click', ()=> selectDashboardItem(el.dataset.itemId));
    el.addEventListener('mouseenter', ()=> selectDashboardItem(el.dataset.itemId));
  });
});
</script>
<script>
function downloadInlineSvgAsPng(wrapperId, filename){
  const wrap = document.getElementById(wrapperId);
  if(!wrap){ alert('Chart container not found.'); return; }
  const svg = wrap.querySelector('svg');
  if(!svg){ alert('No SVG chart is available for download.'); return; }
  const serializer = new XMLSerializer();
  let svgText = serializer.serializeToString(svg);
  if(!svgText.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)) {
    svgText = svgText.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  const bboxW = Number(svg.getAttribute('width')) || Number(svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) || 1200;
  const bboxH = Number(svg.getAttribute('height')) || Number(svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height) || 600;
  const img = new Image();
  const blob = new Blob([svgText], {type:'image/svg+xml;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  img.onload = function(){
    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bboxW * scale));
    canvas.height = Math.max(1, Math.round(bboxH * scale));
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    canvas.toBlob(function(pngBlob){
      if(!pngBlob){ alert('PNG export failed.'); return; }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(pngBlob);
      a.download = filename || 'chart.png';
      document.body.appendChild(a);
      a.click();
      setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); }, 500);
    }, 'image/png');
  };
  img.onerror = function(){ URL.revokeObjectURL(url); alert('Unable to render SVG as PNG.'); };
  img.src = url;
}
</script>
</body>
</html>
"""

ITEM_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 920px; margin: 24px auto; padding: 0 16px; line-height: 1.6; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .muted { color: #666; }
    .option { display: block; padding: 12px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 10px; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .btn-secondary { background: #475569; }
    .stat { display: inline-block; margin-right: 14px; margin-bottom: 10px; padding: 10px 12px; background: #f8fafc; border-radius: 10px; }
    img { max-width: 100%; border-radius: 10px; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <div class="stat">Mode {{ progress.mode_name }}</div>
    <div class="stat">Answered {{ progress.answered }} / {{ progress.max_items }}</div>
    <div class="stat">Theta {{ '%.3f'|format(progress.theta) }}</div>
    <div class="stat">Posterior SE {{ '%.3f'|format(progress.se) }}</div>
    <div class="stat">Current item information {{ '%.3f'|format(progress.info_value) }}</div>
    <p class="muted">{{ progress.info_line }}</p>
  </div>

  <div class="card">
    <h2>{{ item.no }}. {{ item.stem }}</h2>
    <div style="margin-bottom:12px;">
      <button class="btn" type="button" onclick="playItemMp3('zh')">Play Chinese MP3</button>
      <button class="btn btn-secondary" type="button" onclick="playItemMp3('en')">Play English MP3</button>
      <span class="muted" style="margin-left:10px;">Mobile-friendly MP3 playback. Tap a button to play the current item.</span>
    </div>
    {% if item.link_href %}
      <div style="margin-bottom:14px;">
        {% if item.is_image_link %}
          <img src="{{ item.link_href }}" alt="item image">
        {% else %}
          <a href="{{ item.link_href }}" target="_blank" rel="noopener">Open reference</a>
        {% endif %}
      </div>
    {% endif %}

    <form method="post" action="{{ url_for('submit_answer') }}">
      {% for op in item.options %}
        <label class="option">
          <input type="radio" name="score" value="{{ op.score }}" required>
          <strong>{{ op.label }}</strong> — {{ op.text }}
        </label>
      {% endfor %}
      <button class="btn" type="submit">Submit response</button>
      <a class="btn btn-secondary" href="{{ url_for('reset') }}">Reset</a>
    </form>
  </div>
<script>
const itemId = {{ item.item_id|tojson }};
let itemAudio = null;
function stopItemAudio(){
  if(itemAudio){ try{ itemAudio.pause(); itemAudio.currentTime = 0; }catch(e){} itemAudio = null; }
}
function playItemMp3(lang){
  const normalized = (String(lang||'').toLowerCase().startsWith('zh')) ? 'zh' : 'en';
  const url = `{{ url_for('voice_tts') }}?item_id=${encodeURIComponent(itemId)}&language=${encodeURIComponent(normalized)}`;
  stopItemAudio();
  try{
    itemAudio = new Audio(url);
    itemAudio.preload = 'auto';
    const pp = itemAudio.play();
    if(pp && typeof pp.catch === 'function'){ pp.catch(()=>{}); }
  }catch(e){}
}
</script>
</body>
</html>
"""

VOICE_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1080px; margin: 24px auto; padding: 0 16px; line-height: 1.6; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; margin: 4px 8px 4px 0; }
    .secondary { background:#475569; }
    .warn { background:#b91c1c; }
    .muted { color:#666; }
    .stat { display: inline-block; margin-right: 14px; margin-bottom: 10px; padding: 10px 12px; background: #f8fafc; border-radius: 10px; }
    .detail-img { max-width:100%; max-height:320px; border-radius:10px; border:1px solid #ddd; display:block; }
    .grid2 { display:grid; grid-template-columns: minmax(360px, 1fr) minmax(300px, 0.9fr); gap:18px; align-items:start; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <div class="stat">Mode Voice practice</div>
    <div class="stat">Requested items {{ requested_max_items }}</div>
    <div class="stat">Actual items {{ item_count }}</div>
    <div class="stat">Start theta {{ '%.2f'|format(start_theta) }}</div>
    <div class="stat">Range +/- {{ '%.2f'|format(theta_range) }}</div>
    <p class="muted">Voice practice reads only the current item stem, one item after another, in the selected language. Because this Rasch PCM uses item-specific response categories, the audio cycle displays the current item's own response options while reading the current stem. Server-generated MP3 audio is used for all playback so that mobile browsers behave more consistently than browser speech synthesis. Use the buttons below to start, replay, pause, resume, stop, or move between items. The item picture and synchronized detail panel update item by item.</p>
    <p class="muted" id="statusLine">Ready. Press Start audio cycle to begin.</p>
  </div>

  <div class="card">
    <button class="btn" type="button" id="btnStart">Start audio cycle</button>
    <button class="btn secondary" type="button" id="btnReplay">Replay current item</button>
    <button class="btn secondary" type="button" id="btnPause">Pause audio</button>
    <button class="btn secondary" type="button" id="btnResume">Resume audio</button>
    <button class="btn warn" type="button" id="btnStop">Stop audio</button>
    <button class="btn secondary" type="button" id="btnPrev">Previous item</button>
    <button class="btn secondary" type="button" id="btnNext">Next item</button>
    <a class="btn secondary" id="homeBtn" href="{{ url_for('index') }}">Home</a>
  </div>

  <div class="grid2">
    <div class="card">
      <h2 id="itemHeader"></h2>
      <div id="itemStem"></div>
      <div id="itemOptions" style="margin-top:12px;"></div>
    </div>
    <div class="card">
      <h2 style="margin-top:0;">Current item</h2>
      <div style="margin-bottom:10px;">
        <span class="stat" id="itemNoPill"></span>
        <span class="stat" id="itemIdPill"></span>
      </div>
      <div><strong>English</strong></div>
      <div id="itemStemEn" style="margin-bottom:8px;"></div>
      <div><strong>Traditional Chinese</strong></div>
      <div id="itemStemZh" style="margin-bottom:12px;"></div>
      <div id="itemStats" class="muted small" style="margin-bottom:12px;"></div>
      <div style="margin-bottom:12px;">
        <button class="btn" type="button" id="btnReadZh">Read Chinese</button>
        <button class="btn secondary" type="button" id="btnReadEn">Read English</button>
      </div>
      <div id="itemFigure"></div>
    </div>
  </div>

<script>
const items = {{ items|tojson }};
const speechConfig = {{ speech_config|tojson }};
let currentIndex = 0;
let queueTimer = null;
let audioEl = null;
let cycleToken = 0;
let cycleActive = false;
let paused = false;

function normalizeLang(lang){
  const v = String(lang || '').toLowerCase();
  return v.startsWith('zh') ? 'zh' : 'en';
}
function estimateDurationMs(text, lang){
  const n = String(text || '').trim().length;
  if(n <= 0) return 2600;
  const perChar = normalizeLang(lang) === 'zh' ? 320 : 180;
  return Math.max(2800, Math.min(18000, n * perChar + 1800));
}
function setStatus(msg){ document.getElementById('statusLine').textContent = msg; }
function currentItem(){ return items[currentIndex] || null; }
function clearQueueTimer(){ if(queueTimer){ try{ clearTimeout(queueTimer); }catch(e){} queueTimer = null; } }
function stopAudioOnly(){
  clearQueueTimer();
  if(audioEl){ try{ audioEl.pause(); audioEl.currentTime = 0; }catch(e){} audioEl.onended = null; audioEl.onerror = null; audioEl = null; }
}
function renderCurrent(){
  const item = currentItem();
  if(!item){
    document.getElementById('itemHeader').textContent = 'Completed';
    document.getElementById('itemStem').textContent = '';
    document.getElementById('itemOptions').innerHTML='';
    document.getElementById('itemFigure').innerHTML='';
    document.getElementById('itemStemEn').textContent='';
    document.getElementById('itemStemZh').textContent='';
    document.getElementById('itemNoPill').textContent='';
    document.getElementById('itemIdPill').textContent='';
    document.getElementById('itemStats').textContent='';
    return;
  }
  document.getElementById('itemHeader').textContent = `No. ${item.no || currentIndex+1} · ${item.item_id} · Item ${currentIndex+1} / ${items.length}`;
  document.getElementById('itemStem').textContent = item.stem || '';
  document.getElementById('itemOptions').innerHTML = '<div class="muted">Voice practice plays the current item stem as MP3 audio. Response options are shown below for reference.</div>' + ((item.options||[]).length ? '<div style="margin-top:8px;">' + item.options.map(op => `<div><strong>${op.label}</strong> — ${op.text}</div>`).join('') + '</div>' : '');
  document.getElementById('itemNoPill').textContent = `No. ${item.no || currentIndex+1}`;
  document.getElementById('itemIdPill').textContent = item.item_id || '';
  document.getElementById('itemStemEn').textContent = item.stem_en || '';
  document.getElementById('itemStemZh').textContent = item.stem_zh || '';
  document.getElementById('itemStats').textContent = `Delta = ${Number(item.delta).toFixed(3)} | Measure = ${Number(item.measure).toFixed(3)} | INFIT MNSQ = ${Number(item.infit).toFixed(3)} | OUTFIT MNSQ = ${Number(item.outfit).toFixed(3)} | SE = ${Number(item.se).toFixed(3)}`;
  const fig = document.getElementById('itemFigure');
  if(item.link_href && item.is_image_link){ fig.innerHTML = `<img class="detail-img" src="${item.link_href}" alt="item image">`; }
  else if(item.link_href){ fig.innerHTML = `<a class="btn secondary" href="${item.link_href}" target="_blank" rel="noopener">Open linked figure</a>`; }
  else { fig.innerHTML = '<div class="muted">No linked picture for this item.</div>'; }
}
function composeItemText(item, lang){
  const useZh = normalizeLang(lang) === 'zh';
  return useZh ? (item.stem_zh || item.stem_en || item.stem || '') : (item.stem_en || item.stem_zh || item.stem || '');
}
function playItemAudio(item, lang, onDone){
  const normalized = normalizeLang(lang);
  if(!speechConfig.server_tts_enabled || !item || !item.item_id) return false;
  const url = `${speechConfig.tts_base_url}?item_id=${encodeURIComponent(item.item_id)}&language=${encodeURIComponent(normalized)}`;
  try{
    audioEl = new Audio(url);
    audioEl.preload = 'auto';
    audioEl.onended = ()=>{ if(typeof onDone === 'function') onDone(); };
    audioEl.onerror = ()=>{ if(typeof onDone === 'function') onDone(); };
    const pp = audioEl.play();
    if(pp && typeof pp.catch === 'function'){ pp.catch(()=>{ if(typeof onDone === 'function') onDone(); }); }
    return true;
  }catch(e){
    if(typeof onDone === 'function') onDone();
    return false;
  }
}
function scheduleNext(token){
  if(token !== cycleToken || !cycleActive || paused) return;
  currentIndex += 1;
  if(currentIndex >= items.length){
    cycleActive = false;
    paused = false;
    currentIndex = Math.max(0, items.length - 1);
    renderCurrent();
    setStatus(`Voice practice completed. Finished ${items.length} / ${items.length}`);
    return;
  }
  playCurrent(token, speechConfig.language_code || 'en');
}
function playCurrent(token, lang){
  if(token !== cycleToken || !cycleActive || paused) return;
  renderCurrent();
  const item = currentItem();
  if(!item){
    cycleActive = false;
    setStatus('Voice practice completed.');
    return;
  }
  const normalized = normalizeLang(lang || speechConfig.language_code || 'en');
  setStatus(`Playing MP3 ${currentIndex+1} / ${items.length}`);
  stopAudioOnly();
  const ok = playItemAudio(item, normalized, ()=> scheduleNext(token));
  if(!ok){
    setStatus('MP3 unavailable for this item.');
  }
}
function startCycle(){
  if(!items.length){ setStatus('No items available.'); return; }
  paused = false;
  cycleActive = true;
  cycleToken += 1;
  if(currentIndex >= items.length) currentIndex = 0;
  playCurrent(cycleToken, speechConfig.language_code || 'en');
}
function replayCurrent(){
  if(!items.length) return;
  cycleActive = false;
  paused = false;
  cycleToken += 1;
  stopAudioOnly();
  renderCurrent();
  const lang = speechConfig.language_code || 'en';
  const item = currentItem();
  setStatus(`Playing MP3 ${currentIndex+1} / ${items.length}`);
  playItemAudio(item, lang, ()=> setStatus(`Finished ${currentIndex+1} / ${items.length}`));
}
function pauseSpeech(){
  if(!items.length) return;
  paused = true;
  cycleActive = false;
  clearQueueTimer();
  if(audioEl){ try{ audioEl.pause(); }catch(e){} }
  setStatus('Paused.');
}
function resumeSpeech(){
  if(!items.length) return;
  if(audioEl){
    paused = false;
    cycleActive = true;
    const pp = audioEl.play();
    if(pp && typeof pp.catch === 'function'){ pp.catch(()=>{}); }
    setStatus(`Playing MP3 ${currentIndex+1} / ${items.length}`);
    return;
  }
  startCycle();
}
function stopSpeech(){
  cycleActive = false;
  paused = false;
  cycleToken += 1;
  stopAudioOnly();
  setStatus('Stopped.');
}
function prevItem(){ stopSpeech(); currentIndex = Math.max(0, currentIndex-1); renderCurrent(); setStatus(`Ready ${currentIndex+1} / ${items.length}`); }
function nextItem(){ stopSpeech(); currentIndex = Math.min(items.length-1, currentIndex+1); renderCurrent(); setStatus(`Ready ${currentIndex+1} / ${items.length}`); }
function readCurrent(lang){
  if(!items.length) return;
  stopSpeech();
  renderCurrent();
  const normalized = normalizeLang(lang);
  const item = currentItem();
  setStatus(`Playing MP3 ${currentIndex+1} / ${items.length}`);
  playItemAudio(item, normalized, ()=> setStatus(`Finished ${currentIndex+1} / ${items.length}`));
}
document.getElementById('btnStart').addEventListener('click', startCycle);
document.getElementById('btnReplay').addEventListener('click', replayCurrent);
document.getElementById('btnPause').addEventListener('click', pauseSpeech);
document.getElementById('btnResume').addEventListener('click', resumeSpeech);
document.getElementById('btnStop').addEventListener('click', stopSpeech);
document.getElementById('btnPrev').addEventListener('click', prevItem);
document.getElementById('btnNext').addEventListener('click', nextItem);
document.getElementById('btnReadZh').addEventListener('click', ()=> readCurrent('zh'));
document.getElementById('btnReadEn').addEventListener('click', ()=> readCurrent('en'));
document.addEventListener('DOMContentLoaded', ()=>{
  currentIndex = 0;
  renderCurrent();
  setStatus(items.length ? 'Ready. Press Start audio cycle to begin.' : 'No items available.');
});
</script>
</body>
</html>
"""

RESULT_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1120px; margin: 24px auto; padding: 0 16px; line-height: 1.6; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .stat { display: inline-block; margin-right: 16px; margin-bottom: 10px; padding: 10px 12px; background: #f8fafc; border-radius: 10px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <div class="stat">Mode {{ result.mode_name }}</div>
    <div class="stat">Final theta {{ '%.3f'|format(result.theta) }}</div>
    <div class="stat">Posterior SE {{ '%.3f'|format(result.se) }}</div>
    <div class="stat">Percentile {{ '%.1f'|format(result.percentile) }}</div>
    <div class="stat">Items used {{ result.n_answered }}</div>
    <div class="stat">Reason {{ result.stop_reason }}</div>
    {% if result.simulation_note %}
      <p class="muted" style="margin-top:12px;"><strong>Simulation source:</strong> {{ result.simulation_note }}</p>
    {% endif %}
  </div>

  <div class="card">
    <div class="stat">INFIT MNSQ {{ '%.3f'|format(result.infit_mnsq) }}</div>
    <div class="stat">OUTFIT MNSQ {{ '%.3f'|format(result.outfit_mnsq) }}</div>
    <div style="margin-top:12px;"><a class="btn" href="{{ url_for('index') }}">Start over</a></div>
  </div>

  <div class="card">
    <h2>Response history</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Item</th>
          <th>Difficulty</th>
          <th>Response</th>
          <th>Score</th>
          <th>Expected</th>
          <th>Theta after item</th>
          <th>SE after item</th>
          <th>ZSTD</th>
          {% if result.has_links %}<th>Reference</th>{% endif %}
        </tr>
      </thead>
      <tbody>
      {% for row in result.history %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ row.item_id }}</td>
          <td>{{ '%.3f'|format(row.delta) }}</td>
          <td>{{ row.answer }}</td>
          <td>{{ row.score }}</td>
          <td>{{ '%.3f'|format(row.expected) }}</td>
          <td>{{ '%.3f'|format(row.theta) }}</td>
          <td>{{ '%.3f'|format(row.se) }}</td>
          <td>{{ '%.3f'|format(row.zstd) }}</td>
          {% if result.has_links %}
            <td>{% if row.link_href %}<a href="{{ row.link_href }}" target="_blank" rel="noopener">Open</a>{% endif %}</td>
          {% endif %}
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  {% if result.comparison %}
  <div class="card">
    <h2>{{ result.comparison.title }}</h2>
    <div class="stat">Simulated CAT runs {{ result.comparison.n_reps }}</div>
    <div class="stat">Full non-CAT length {{ result.comparison.full_length }}</div>
    <div class="stat">CAT maximum items {{ result.comparison.cat_max_items }}</div>
    <div class="stat">CAT stopping rule posterior SE ≤ {{ '%.3f'|format(result.comparison.stop_se) }}</div>
    <div class="stat">Mean CAT length {{ '%.3f'|format(result.comparison.cat_length_summary.mean) }}</div>
    <div class="stat">CAT length SD {{ '%.3f'|format(result.comparison.cat_length_summary.sd) }}</div>

    <h3>Table 1. Item length</h3>
    <table>
      <thead>
        <tr>
          <th>Group</th><th>n</th><th>Mean length</th><th>SD length</th><th>Min</th><th>Median</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full non-CAT</td>
          <td>{{ result.comparison.full_summary.n }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_summary.mean_length) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_summary.sd_length) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_length_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_length_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_length_stats.max) }}</td>
        </tr>
        <tr>
          <td>CAT</td>
          <td>{{ result.comparison.n_reps }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_stats.max) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:10px;">One-sample t-test of CAT item length against the full non-CAT length ({{ result.comparison.full_length }} items): t = {{ '%.3f'|format(result.comparison.length_ttest.t) }}, p = {{ result.comparison.length_ttest.p_text }}, df = {{ result.comparison.length_ttest.df_text }}.</p>
    <div style="margin-top:12px;">{{ result.comparison.length_svg|safe }}</div>

    <h3>Table 2. Person measure</h3>
    <table>
      <thead>
        <tr>
          <th>Group</th><th>n</th><th>Mean theta</th><th>SD theta</th><th>Min</th><th>Median</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full non-CAT</td>
          <td>{{ result.comparison.full_theta_summary.n }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_stats.max) }}</td>
        </tr>
        <tr>
          <td>CAT</td>
          <td>{{ result.comparison.n_reps }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_stats.max) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:10px;">One-sample t-test of CAT person measure against the fixed full non-CAT measure ({{ '%.3f'|format(result.comparison.full_theta_value) }}): t = {{ '%.3f'|format(result.comparison.theta_diff_ttest.t) }}, p = {{ result.comparison.theta_diff_ttest.p_text }}, df = {{ result.comparison.theta_diff_ttest.df_text }}. Mean difference = {{ '%.3f'|format(result.comparison.theta_diff_summary.mean) }}, SD difference = {{ '%.3f'|format(result.comparison.theta_diff_summary.sd) }}.</p>
    <div style="margin-top:12px;">{{ result.comparison.theta_svg|safe }}</div>

    <p class="muted">Both figures are box plots. The left compares CAT and full non-CAT by item length; the right compares CAT and full non-CAT by person measure. {{ result.comparison.note_text }}</p>
    <h3>Actual CAT used items in this administration</h3>
    <table>
      <thead>
        <tr><th>Pos</th><th>Item</th><th>No</th><th>Delta</th><th>Reference</th></tr>
      </thead>
      <tbody>
        {% for row in result.comparison.actual_rows %}
        <tr>
          <td>{{ row.pos }}</td>
          <td>{{ row.item_id }}</td>
          <td>{{ row.no }}</td>
          <td>{{ '%.3f'|format(row.delta) }}</td>
          <td>{% if row.link_href %}<a href="{{ row.link_href }}" target="_blank" rel="noopener">Open</a>{% endif %}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <div class="card">
    <h2>CAT result trend chart</h2>
    <div id="trendChartWrap">{{ result.trend_svg|safe }}</div>
    <div style="margin-top:10px;">
      <button class="btn" type="button" onclick="downloadResultSvgAsPng('trendChartWrap', 'cat_result_trend_chart.png')">Download trend chart PNG</button>
    </div>
  </div>

  <div class="card">
    <h2>KIDMAP-style dashboard</h2>
    <div id="kidmapResultWrap">{{ result.kidmap_svg|safe }}</div>
    <div style="margin-top:10px;">
      <button class="btn" type="button" onclick="downloadResultSvgAsPng('kidmapResultWrap', 'kidmap_style_dashboard.png')">Download KIDMAP PNG</button>
    </div>
    <div class="card" style="background:#f8fafc; margin-top:14px;">
      <h3 style="margin-top:0;">Skin-cancer risk classification based on the overall KIDMAP person measure</h3>
      <div class="stat"><strong>Final theta</strong><br>{{ '%.3f'|format(result.theta) }}</div>
      <div class="stat"><strong>Classification</strong><br>{{ result.risk_classification.level }}</div>
      <p class="muted" style="clear:both;">
        This classification is based on the overall Rasch person measure shown in the KIDMAP title, not on any single item bubble and not as a clinical diagnosis:
        <span class="mono">theta &lt; -0.5 = low / mild risk; -0.5 to 0.5 = average / moderate risk; 0.5 to 1.5 = high risk; &gt; 1.5 = very high risk.</span>
      </p>
      <p class="muted">{{ result.risk_classification.explanation }}</p>
    </div>
    <p class="muted">Two red dotted horizontal lines mark the person measure ± 1 SE. Positive standardized residuals are shown in blue; negative residuals are shown in red. The KIDMAP supports risk interpretation and item-level review, but it should not be used alone as a clinical diagnosis.</p>
  </div>

  <div class="card">
    <h2>Category probability curves (CPC)</h2>
    <div id="cpcWrap">{{ result.cpc_svg|safe }}</div>
    <div style="margin-top:10px;">
      <button class="btn" type="button" onclick="downloadResultSvgAsPng('cpcWrap', 'category_probability_curves.png')">Download CPC PNG</button>
    </div>
    <p class="muted">This CPC uses the current <strong>reference item</strong>. The red dotted vertical lines indicate the adjacent-category intersections (step thresholds) for that item. The solid red vertical line marks the person measure, so the most probable response category can be read against the curves.</p>
    <p class="muted">At your final theta, the most probable response on the delta = 0 reference item is <strong>{{ result.cpc_pred_label }}</strong>.</p>
  </div>

<script>
(function(){
  function svgToDataUrl(svg){
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.querySelectorAll("*").forEach(function(el){
      const computed = window.getComputedStyle(el);
      if(!el.getAttribute("font-family")) el.setAttribute("font-family", computed.fontFamily || "Arial, sans-serif");
    });
    let svgText = new XMLSerializer().serializeToString(clone);
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgText);
  }

  window.downloadResultSvgAsPng = function(wrapperId, filename){
    const wrap = document.getElementById(wrapperId);
    if(!wrap){
      alert("Chart container not found: " + wrapperId);
      return;
    }
    const svg = wrap.querySelector("svg");
    if(!svg){
      alert("No SVG chart found in " + wrapperId + ".");
      return;
    }

    const viewBox = svg.getAttribute("viewBox");
    let width = parseFloat(svg.getAttribute("width"));
    let height = parseFloat(svg.getAttribute("height"));

    if((!width || !height) && viewBox){
      const parts = viewBox.trim().split(/\s+/).map(Number);
      if(parts.length === 4){
        width = width || parts[2];
        height = height || parts[3];
      }
    }
    width = width || svg.getBoundingClientRect().width || 1200;
    height = height || svg.getBoundingClientRect().height || 600;

    const img = new Image();
    img.onload = function(){
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const link = document.createElement("a");
      link.download = filename || "chart.png";
      link.href = canvas.toDataURL("image/png");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
    img.onerror = function(){
      alert("PNG export failed. Please try Chrome or Edge, or use the SVG directly.");
    };
    img.src = svgToDataUrl(svg);
  };
})();
</script>

</body>
</html>
"""


@dataclass
class CategoryOption:
    score: int
    label: str
    text: str


@dataclass
class ItemRecord:
    item_id: str
    no: int
    full_text_zh: str
    stem_zh: str
    full_text_en: str
    stem_en: str
    delta: float
    link: str = ""
    options_zh: List[CategoryOption] = field(default_factory=list)
    options_en: List[CategoryOption] = field(default_factory=list)
    raw_thresholds: List[float] = field(default_factory=list)

    def stem_for(self, language: str) -> str:
        return self.stem_en if language == "en" and self.stem_en else self.stem_zh

    def response_options(self, language: str = "en") -> List[CategoryOption]:
        lang = str(language or "en").lower()
        if lang == "zh" and self.options_zh:
            opts = [CategoryOption(score=int(op.score), label=str(op.label), text=str(op.text)) for op in self.options_zh]
        elif self.options_en:
            opts = [CategoryOption(score=int(op.score), label=str(op.label), text=str(op.text)) for op in self.options_en]
        elif self.options_zh:
            opts = [CategoryOption(score=int(op.score), label=str(op.label), text=str(op.text)) for op in self.options_zh]
        else:
            n_cat = max(1, len(self.raw_thresholds) + 1)
            opts = [CategoryOption(score=i, label=str(i), text=f"Category {i}") for i in range(n_cat)]
        if self.raw_thresholds:
            n_cat = max(1, len(self.raw_thresholds) + 1)
            if len(opts) > n_cat:
                opts = opts[:n_cat]
        return opts


    def score_values(self) -> np.ndarray:
        opts = self.response_options("en")
        vals = [int(op.score) for op in opts]
        if not vals:
            vals = list(range(max(1, len(self.raw_thresholds) + 1)))
        vals = sorted(dict.fromkeys(vals))
        return np.asarray(vals, dtype=float)



def compute_cat_stop_criterion(bank) -> dict:
    """Compute homepage CAT stopping SE from sample-data Cronbach's alpha."""
    # Prefer alpha computed from simulated response/sample data.
    alpha_candidates = [
        getattr(bank, "sample_cronbach_alpha", float("nan")),
        getattr(bank, "cronbach_alpha", float("nan")),
    ]
    alpha = float("nan")
    for v in alpha_candidates:
        try:
            vv = float(v)
            if math.isfinite(vv):
                alpha = vv
                break
        except Exception:
            pass

    # Use theta SD from sample person distribution when possible.
    try:
        vals = np.asarray(getattr(bank, "person_distribution", []), dtype=float)
        vals = vals[np.isfinite(vals)]
        theta_sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float(getattr(bank, "prior_sd", 1.0))
        n_persons = int(vals.size)
    except Exception:
        theta_sd = float(getattr(bank, "prior_sd", 1.0))
        n_persons = 0

    if not math.isfinite(alpha):
        alpha = 0.90
    alpha = min(max(alpha, 0.0), 0.9999)

    if not math.isfinite(theta_sd) or theta_sd <= 0:
        theta_sd = 1.0

    stop_se = theta_sd * math.sqrt(max(0.0, 1.0 - alpha))
    return {
        "alpha": alpha,
        "theta_sd": theta_sd,
        "stop_se": stop_se,
        "n_persons": n_persons,
    }


OPTION_LABELS = [str(i) for i in range(1, 10)]


def _zip_name_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    return {name.lower(): name for name in zf.namelist()}


def _zip_read_bytes(zf: zipfile.ZipFile, wanted_name: str) -> bytes:
    name_map = _zip_name_map(zf)
    hit = name_map.get(wanted_name.lower())
    if not hit:
        raise KeyError(f"{wanted_name} not found in ZIP. Available: {zf.namelist()}")
    return zf.read(hit)


def _read_csv_bytes_robust(raw: bytes, *, csv_name: str = "csv") -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.ParserError:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, engine="python")
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
    raise ValueError(f"Unable to read {csv_name}. Tried encodings: {encodings}. Last error: {last_err}")


def _read_text_bytes_robust(raw: bytes, *, text_name: str = "text") -> str:
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"Unable to decode {text_name}. Tried encodings: {encodings}. Last error: {last_err}")


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_item_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_option_string(raw: str, min_score: int = 0) -> List[CategoryOption]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\s*,\s*(?=\d+\s*=)", text) if p.strip()]
    out: List[CategoryOption] = []
    explicit_scores: List[int] = []
    parsed_parts: List[Tuple[Optional[int], str, str]] = []
    for part in parts:
        m = re.match(r"(\d+)\s*=\s*(.+)$", part)
        if m:
            explicit = int(m.group(1))
            explicit_scores.append(explicit)
            parsed_parts.append((explicit, m.group(1), m.group(2).strip()))
        else:
            parsed_parts.append((None, str(len(parsed_parts)), part))
    use_zero_based = bool(explicit_scores) and min(explicit_scores) == 0
    use_one_based = bool(explicit_scores) and min(explicit_scores) == 1 and min_score == 0
    next_score = min_score
    for explicit, label, desc in parsed_parts:
        if explicit is None:
            score = next_score
        elif use_zero_based:
            score = explicit
        elif use_one_based:
            score = explicit - 1
        else:
            score = explicit
        out.append(CategoryOption(score=int(score), label=str(label), text=str(desc)))
        next_score = int(score) + 1
    return out


def parse_step_string(raw: str) -> List[float]:
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", str(raw or ""))
    return [float(v) for v in vals]


def _copy_options(options: List[CategoryOption]) -> List[CategoryOption]:
    return [CategoryOption(score=int(op.score), label=str(op.label), text=str(op.text)) for op in (options or [])]


def _svg_wrap(width: int, height: int, inner: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">{inner}</svg>'


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    normalized = {re.sub(r'[^a-z0-9]+', '', str(col).lower()): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = re.sub(r'[^a-z0-9]+', '', cand.lower())
        if key in normalized:
            return normalized[key]
    return None


class RaschPCMBank:
    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path
        self.extract_dir = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "raschcatpcm_bundle_cache_v2"
        self.theta_grid = np.linspace(-6.0, 6.0, 2401)
        self.cat_title = "Polytomous Rasch-CAT (PCM)"
        self.model = "PCM"
        self.items: List[ItemRecord] = []
        self.item_lookup: Dict[str, ItemRecord] = {}
        self.prior_mean = 0.0
        self.prior_sd = 1.0
        self.person_distribution = np.array([0.0])
        self.person_df = pd.DataFrame()
        self.item_fit_df = pd.DataFrame()
        self.zscore_df = pd.DataFrame()
        self.category_options: List[CategoryOption] = []
        self.category_options_zh: List[CategoryOption] = []
        self.category_options_en: List[CategoryOption] = []
        self.score_to_option: Dict[int, CategoryOption] = {}
        self.step_thresholds: List[float] = []
        self.min_score = 0
        self.max_score = 1
        self.step_delta_df = pd.DataFrame()
        self._load()

    def _extract_selected_files(self, zf: zipfile.ZipFile) -> None:
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        needed = []
        for name in zf.namelist():
            low = name.lower()
            if low in {
                "response_category.csv", "fixed_item_delta.csv", "person_estimates.csv", "item_estimates.csv",
                "metadata.json", "step_thresholds.csv", "item_step_delta.csv", "zscore.csv", "readme.md"
            } or low.startswith("pic/"):
                needed.append(name)
        for name in needed:
            target = self.extract_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())

    def _load(self) -> None:
        with zipfile.ZipFile(self.bundle_path, "r") as zf:
            self._extract_selected_files(zf)
            response_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "response_category.csv"), csv_name="response_category.csv")
            person_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "person_estimates.csv"), csv_name="person_estimates.csv")
            item_fit_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "item_estimates.csv"), csv_name="item_estimates.csv")
            try:
                self.zscore_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "zscore.csv"), csv_name="zscore.csv")
            except Exception:
                self.zscore_df = pd.DataFrame()
            try:
                metadata = json.loads(_read_text_bytes_robust(_zip_read_bytes(zf, "metadata.json"), text_name="metadata.json"))
            except Exception:
                metadata = {}
            try:
                delta_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "fixed_item_delta.csv"), csv_name="fixed_item_delta.csv")
            except Exception:
                delta_df = pd.DataFrame(columns=["ITEM", "DELTA"])
            try:
                step_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "step_thresholds.csv"), csv_name="step_thresholds.csv")
            except Exception:
                step_df = pd.DataFrame(columns=["STEP", "THRESHOLD"])
            try:
                self.step_delta_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "item_step_delta.csv"), csv_name="item_step_delta.csv")
            except Exception:
                self.step_delta_df = pd.DataFrame()
            if not isinstance(metadata, dict) or not metadata:
                inferred_min = 0
                inferred_max = 1
                inferred_model = "PCM"
                option_texts_probe = [str(x).strip() for x in response_df.get("option", pd.Series(dtype=str)).fillna("") if str(x).strip()]
                if not option_texts_probe:
                    option_texts_probe = [str(x).strip() for x in response_df.get("option2", pd.Series(dtype=str)).fillna("") if str(x).strip()]
                probe_opts = parse_option_string(option_texts_probe[0] if option_texts_probe else "", min_score=0)
                if probe_opts:
                    inferred_min = int(min(op.score for op in probe_opts))
                    inferred_max = int(max(op.score for op in probe_opts))
                elif not self.step_delta_df.empty and "OBSERVED_CATEGORIES" in self.step_delta_df.columns:
                    cats = []
                    for raw in self.step_delta_df["OBSERVED_CATEGORIES"].fillna(""):
                        for tok in str(raw).split(","):
                            tok = tok.strip()
                            if tok.isdigit():
                                cats.append(int(tok))
                    if cats:
                        inferred_min = int(min(cats))
                        inferred_max = int(max(cats))
                metadata = {
                    "bundle_type": "replay_bundle",
                    "main_patch": "pcm_cat_metadata_fallback_v4",
                    "run_id": "runtime_inferred",
                    "model": inferred_model,
                    "min_cat": inferred_min,
                    "max_cat": inferred_max,
                    "warnings": ["metadata.json missing in replay bundle; inferred defaults were used"]
                }

        # Prefer an adjacent/root response_category.csv when it is richer than the one
        # inside replay_bundle.zip. This fixes the common deployment case where the
        # ZIP still has a minimal category-only CSV, while the project root CSV has
        # item stems, English options, PCM steps, and PNG filenames in the link column.
        adjacent_response = Path(__file__).with_name("response_category.csv")
        if adjacent_response.exists():
            try:
                adjacent_df = _read_csv_bytes_robust(adjacent_response.read_bytes(), csv_name="adjacent response_category.csv")
                current_cols = {str(c).strip().lower() for c in response_df.columns}
                adjacent_cols = {str(c).strip().lower() for c in adjacent_df.columns}
                current_has_links = "link" in current_cols and response_df.get("link", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").any()
                adjacent_has_links = "link" in adjacent_cols and adjacent_df.get("link", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").any()
                adjacent_has_items = {"no", "item2", "option2"}.issubset(adjacent_cols) or {"no", "item", "option"}.issubset(adjacent_cols)
                current_is_minimal = len(current_cols) <= 2 or current_cols == {"category"}
                if (adjacent_has_links and adjacent_has_items) and (current_is_minimal or not current_has_links):
                    response_df = adjacent_df
                    self.response_category_source = "adjacent response_category.csv"
                else:
                    self.response_category_source = "replay_bundle.zip/response_category.csv"
            except Exception:
                self.response_category_source = "replay_bundle.zip/response_category.csv"
        else:
            self.response_category_source = "replay_bundle.zip/response_category.csv"

        response_df = response_df.copy()
        for col in ["no", "link", "item", "item2", "option", "option2", "Title", "Step", "delta"]:
            if col not in response_df.columns:
                response_df[col] = ""
        response_df["link"] = response_df["link"].fillna("").astype(str).str.strip()
        response_df["item"] = response_df["item"].fillna("").astype(str)
        response_df["item2"] = response_df["item2"].fillna("").astype(str)
        response_df["no"] = pd.to_numeric(response_df["no"], errors="coerce")
        response_df = response_df.dropna(subset=["no"]).copy()
        response_df["no"] = response_df["no"].astype(int)

        self.model = str(metadata.get("model", "PCM") or "PCM").upper()
        if self.model not in {"PCM", "PARTIAL CREDIT MODEL"}:
            self.model = "PCM"
        self.min_score = int(metadata.get("min_cat", 0) or 0)
        option_texts_zh = [str(x).strip() for x in response_df.get("option", pd.Series(dtype=str)).fillna("") if str(x).strip()]
        option_texts_en = [str(x).strip() for x in response_df.get("option2", pd.Series(dtype=str)).fillna("") if str(x).strip()]
        self.category_options_zh = parse_option_string(option_texts_zh[0] if option_texts_zh else "", min_score=self.min_score)
        self.category_options_en = parse_option_string(option_texts_en[0] if option_texts_en else "", min_score=self.min_score)
        if self.category_options_zh and not self.category_options_en:
            self.category_options_en = [CategoryOption(score=op.score, label=op.label, text=op.text) for op in self.category_options_zh]
        if self.category_options_en and not self.category_options_zh:
            self.category_options_zh = [CategoryOption(score=op.score, label=op.label, text=op.text) for op in self.category_options_en]
        base_options = self.category_options_zh or self.category_options_en
        if not base_options:
            base_options = [CategoryOption(score=i, label=str(i + 1), text=f"Category {i + 1}") for i in range(5)]
        if len(self.category_options_zh) != len(base_options):
            self.category_options_zh = [CategoryOption(score=op.score, label=op.label, text=op.text) for op in base_options]
        if len(self.category_options_en) != len(base_options):
            self.category_options_en = [CategoryOption(score=op.score, label=op.label, text=op.text) for op in base_options]
        self.category_options = [CategoryOption(score=op.score, label=op.label, text=op.text) for op in base_options]
        self.max_score = self.category_options[-1].score
        self.score_to_option = {op.score: op for op in self.category_options_en}
        title_vals = [str(x).strip() for x in response_df.get("Title", pd.Series(dtype=str)).fillna("") if str(x).strip()]
        if title_vals:
            self.cat_title = title_vals[0]
        step_vals = [str(x).strip() for x in response_df.get("Step", pd.Series(dtype=str)).fillna("") if str(x).strip()]
        self.step_thresholds = parse_step_string(step_vals[0] if step_vals else "")
        if not self.step_thresholds and not step_df.empty:
            threshold_col = _find_column(step_df, ["THRESHOLD", "tau", "step_threshold", "step", "delta"])
            if threshold_col is not None:
                self.step_thresholds = [float(x) for x in pd.to_numeric(step_df[threshold_col], errors="coerce").dropna().tolist()]
        n_steps_expected = max(0, len(self.category_options) - 1)
        self.step_thresholds = self.step_thresholds[:n_steps_expected]
        while len(self.step_thresholds) < n_steps_expected:
            self.step_thresholds.append(0.0)

        delta_df = delta_df.copy()
        item_delta_item_col = _find_column(delta_df, ["ITEM", "item", "item_id", "entry"])
        item_delta_value_col = _find_column(delta_df, ["DELTA", "fixed_item_delta", "measure", "MEASURE", "delta"])
        if not delta_df.empty and item_delta_item_col is not None and item_delta_value_col is not None:
            delta_df = delta_df.rename(columns={item_delta_item_col: "ITEM", item_delta_value_col: "DELTA"}).copy()
            item_labels = delta_df["ITEM"].astype(str).str.strip()
            extracted_no = item_labels.str.extract(r"(\d+)(?!.*\d)", expand=False)
            delta_df["no"] = pd.to_numeric(extracted_no, errors="coerce")
            delta_df = delta_df.dropna(subset=["no"]).copy()
            delta_df["no"] = delta_df["no"].astype(int)
        else:
            delta_df = pd.DataFrame(columns=["no", "ITEM", "DELTA"])

        merged = response_df.merge(delta_df[[c for c in ["no", "ITEM", "DELTA"] if c in delta_df.columns]], on="no", how="left")
        merged["ITEM"] = merged.get("ITEM", pd.Series(index=merged.index, dtype=str)).fillna(merged["no"].map(lambda x: f"a{x}"))
        merged["DELTA"] = pd.to_numeric(merged.get("DELTA", merged.get("delta")), errors="coerce")
        if np.all(~np.isfinite(merged["DELTA"].to_numpy(dtype=float))):
            merged["DELTA"] = pd.to_numeric(merged.get("delta"), errors="coerce")
        med_delta = float(np.nanmedian(merged["DELTA"].to_numpy(dtype=float))) if merged["DELTA"].notna().any() else 0.0
        merged["DELTA"] = merged["DELTA"].fillna(med_delta)

        self.person_df = person_df.copy()
        self.item_fit_df = item_fit_df.copy()
        measures = pd.to_numeric(self.person_df.get("MEASURE"), errors="coerce").dropna().to_numpy(dtype=float)
        if measures.size > 5:
            self.prior_mean = float(np.mean(measures))
            self.prior_sd = max(float(np.std(measures, ddof=1)), 0.5)
            self.person_distribution = measures
        self.prior_sd = max(self.prior_sd, 0.5)

        step_delta_lookup: Dict[str, dict] = {}
        if not self.step_delta_df.empty:
            sdf = self.step_delta_df.copy()
            item_col = _find_column(sdf, ["ITEM", "item"])
            if item_col:
                for _, srow in sdf.iterrows():
                    key = str(srow.get(item_col, "")).strip()
                    if key:
                        step_delta_lookup[key] = dict(srow)

        items: List[ItemRecord] = []
        all_option_lens: List[int] = []
        for row in merged.itertuples(index=False):
            zh = parse_item_text(str(getattr(row, "item", "")))
            en = parse_item_text(str(getattr(row, "item2", "") or zh))
            item_id = str(getattr(row, "ITEM", f"a{int(row.no)}"))
            item_opts_zh = parse_option_string(str(getattr(row, "option", "") or ""), min_score=self.min_score)
            item_opts_en = parse_option_string(str(getattr(row, "option2", "") or ""), min_score=self.min_score)
            if item_opts_zh and not item_opts_en:
                item_opts_en = _copy_options(item_opts_zh)
            if item_opts_en and not item_opts_zh:
                item_opts_zh = _copy_options(item_opts_en)
            if not item_opts_zh and not item_opts_en:
                item_opts_zh = _copy_options(base_options)
                item_opts_en = _copy_options(base_options)
            raw_steps = parse_step_string(str(getattr(row, "Step", "") or ""))
            srow = step_delta_lookup.get(en) or step_delta_lookup.get(zh)
            observed_scores: List[int] = []
            if srow:
                obs_text = str(srow.get("OBSERVED_CATEGORIES", "") or "").strip()
                observed_scores = [int(x) for x in re.findall(r"-?\d+", obs_text)]
            if not raw_steps and srow:
                item_measure = pd.to_numeric(pd.Series([srow.get("ITEM_MEASURE")]), errors="coerce").iloc[0]
                step_vals = []
                for col in [c for c in srow.keys() if re.match(r"STEP_\d+_DELTA$", str(c))]:
                    v = pd.to_numeric(pd.Series([srow.get(col)]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        step_vals.append(float(v))
                real_steps = pd.to_numeric(pd.Series([srow.get("N_REAL_STEPS")]), errors="coerce").iloc[0]
                if pd.notna(item_measure) and step_vals:
                    if pd.notna(real_steps):
                        step_vals = step_vals[: max(0, int(real_steps))]
                    raw_steps = [float(item_measure) + float(v) for v in step_vals]
            if observed_scores:
                def _filter_scores(opts: List[CategoryOption], obs: List[int]) -> List[CategoryOption]:
                    keep = [op for op in opts if int(op.score) in set(obs)]
                    if len(keep) == len(obs):
                        keep.sort(key=lambda op: int(op.score))
                        return keep
                    return opts[:len(obs)] if len(opts) >= len(obs) else opts
                item_opts_zh = _filter_scores(item_opts_zh, observed_scores) if item_opts_zh else item_opts_zh
                item_opts_en = _filter_scores(item_opts_en, observed_scores) if item_opts_en else item_opts_en
            elif raw_steps and max(len(item_opts_zh), len(item_opts_en)) > len(raw_steps) + 1:
                trim_n = len(raw_steps) + 1
                item_opts_zh = item_opts_zh[:trim_n] if item_opts_zh else item_opts_zh
                item_opts_en = item_opts_en[:trim_n] if item_opts_en else item_opts_en
            target_n_cat = max(len(item_opts_zh), len(item_opts_en), len(raw_steps) + 1, 1)
            def _ensure_len(opts: List[CategoryOption], n: int) -> List[CategoryOption]:
                out = _copy_options(opts)
                if not out:
                    out = [CategoryOption(score=i, label=str(i), text=f"Category {i}") for i in range(n)]
                scores = [int(op.score) for op in out]
                start = min(scores) if scores else self.min_score
                if len(out) < n:
                    for idx in range(len(out), n):
                        score = start + idx
                        out.append(CategoryOption(score=score, label=str(score), text=f"Category {score}"))
                elif len(out) > n:
                    out = out[:n]
                return out
            item_opts_zh = _ensure_len(item_opts_zh, target_n_cat)
            item_opts_en = _ensure_len(item_opts_en, target_n_cat)
            if not np.isfinite(float(getattr(row, "DELTA"))):
                item_delta = float(np.mean(raw_steps)) if raw_steps else 0.0
            else:
                item_delta = float(getattr(row, "DELTA"))
            items.append(ItemRecord(
                item_id=item_id,
                no=int(getattr(row, "no")),
                full_text_zh=zh,
                stem_zh=zh,
                full_text_en=en,
                stem_en=en,
                delta=item_delta,
                link=str(getattr(row, "link", "") or "").strip(),
                options_zh=item_opts_zh,
                options_en=item_opts_en,
                raw_thresholds=[float(x) for x in raw_steps[: max(0, target_n_cat - 1)]],
            ))
            all_option_lens.append(target_n_cat)
        items.sort(key=lambda x: x.no)
        self.items = items
        self.item_lookup = {x.item_id: x for x in items}
        if items:
            longest = max(items, key=lambda it: len(it.response_options("en")))
            self.category_options_zh = _copy_options(longest.response_options("zh"))
            self.category_options_en = _copy_options(longest.response_options("en"))
            self.category_options = _copy_options(self.category_options_en or self.category_options_zh)
            self.max_score = int(max(max(op.score for op in it.response_options("en")) for it in items))

    def local_asset_path(self, raw_link: str) -> Optional[Path]:
        raw = (raw_link or "").strip().replace("\\", "/")
        if not raw or re.match(r"^[a-z]+://", raw, re.I):
            return None
        candidate = raw.lstrip("./")
        candidates = [candidate]
        stem, ext = os.path.splitext(candidate)
        if ext.lower() == '.jpt':
            candidates.extend([stem + e for e in IMG_EXTS])
        elif not ext:
            candidates.extend([candidate + e for e in IMG_EXTS])
        expanded = []
        for cand in candidates:
            expanded.append(cand)
            if "/" not in cand:
                expanded.append(f"pic/{cand}")
        search_roots = [self.extract_dir, Path(__file__).parent]
        for cand in expanded:
            for root in search_roots:
                candidate_path = root / cand
                if candidate_path.exists() and candidate_path.is_file():
                    return candidate_path
        # Windows users sometimes paste names with slightly different case.
        # Do a small case-insensitive fallback under /pic and bundle /pic.
        expanded_lower = {str(c).replace("\\", "/").lower() for c in expanded}
        for root in search_roots:
            for pic_root in [root, root / "pic"]:
                if not pic_root.exists():
                    continue
                for fp in pic_root.glob("*"):
                    rel1 = str(fp.relative_to(root)).replace("\\", "/").lower()
                    rel2 = fp.name.lower()
                    if rel1 in expanded_lower or rel2 in expanded_lower:
                        return fp
        return None

    def _resolve_item(self, item_or_delta) -> Optional[ItemRecord]:
        if isinstance(item_or_delta, ItemRecord):
            return item_or_delta
        if isinstance(item_or_delta, str):
            return self.item_lookup.get(str(item_or_delta))
        return None

    def response_options(self, language: str = "en", item_or_id=None) -> List[CategoryOption]:
        item = self._resolve_item(item_or_id)
        if item is not None:
            return item.response_options(language)
        lang = str(language or 'en').lower()
        if lang == 'zh' and self.category_options_zh:
            return _copy_options(self.category_options_zh)
        if lang == 'en' and self.category_options_en:
            return _copy_options(self.category_options_en)
        return _copy_options(self.category_options)

    def score_values(self, item_or_id=None) -> np.ndarray:
        item = self._resolve_item(item_or_id)
        if item is not None:
            return item.score_values()
        opts = self.response_options("en")
        vals = [int(op.score) for op in opts]
        if not vals:
            m = int(len(self.step_thresholds)) if getattr(self, "step_thresholds", None) is not None else max(0, self.max_score - self.min_score)
            vals = list(range(self.min_score, self.min_score + m + 1))
        vals = sorted(dict.fromkeys(vals))
        if getattr(self, "step_thresholds", None) is not None and len(self.step_thresholds) > 0:
            want = int(len(self.step_thresholds)) + 1
            if len(vals) > want:
                vals = vals[:want]
        return np.asarray(vals, dtype=float)


    def raw_thresholds(self, item_or_delta) -> np.ndarray:
        item = self._resolve_item(item_or_delta)
        if item is not None and item.raw_thresholds:
            return np.asarray(item.raw_thresholds, dtype=float)
        if item is not None:
            return np.asarray([], dtype=float)
        taus = np.asarray(self.step_thresholds, dtype=float)
        delta = float(item_or_delta)
        m = int(len(taus)) if taus.size else max(0, len(self.category_options) - 1)
        return delta + taus[:m]


    def option_text(self, score: int, language: str = "en", item_or_id=None) -> str:
        options = {int(op.score): op for op in self.response_options(language, item_or_id)}
        op = options.get(int(score))
        return f"{op.label} = {op.text}" if op else str(score)

    def category_probabilities(self, theta: np.ndarray | float, item_or_delta) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        item = self._resolve_item(item_or_delta)
        if item is not None:
            taus = np.asarray(self.raw_thresholds(item), dtype=float)
            m = int(len(taus))
            out_shape = theta_arr.shape + (m + 1,)
            eta = np.zeros(out_shape, dtype=float)
            if m > 0:
                for k in range(1, m + 1):
                    core = np.clip(theta_arr - float(taus[k - 1]), -35.0, 35.0)
                    eta[..., k] = eta[..., k - 1] + core
            eta -= np.max(eta, axis=-1, keepdims=True)
            p = np.exp(eta)
            p /= np.sum(p, axis=-1, keepdims=True)
            return p
        taus = np.asarray(self.step_thresholds, dtype=float)
        m = int(len(taus))
        delta = float(item_or_delta)
        core = np.clip(theta_arr - delta, -35.0, 35.0)
        out_shape = theta_arr.shape + (m + 1,)
        eta = np.zeros(out_shape, dtype=float)
        if m > 0:
            for k in range(1, m + 1):
                eta[..., k] = eta[..., k - 1] + core - taus[k - 1]
        eta -= np.max(eta, axis=-1, keepdims=True)
        p = np.exp(eta)
        p /= np.sum(p, axis=-1, keepdims=True)
        return p


    def expected_score(self, theta: float, item_or_delta) -> float:
        p = self.category_probabilities(theta, item_or_delta)
        scores = np.asarray(self.score_values(item_or_delta), dtype=float).reshape(-1)
        if scores.size != p.shape[-1]:
            scores = scores[: p.shape[-1]] if scores.size >= p.shape[-1] else np.arange(self.min_score, self.min_score + p.shape[-1], dtype=float)
        return float(np.sum(p * scores, axis=-1))

    def variance_score(self, theta: float, item_or_delta) -> float:
        p = self.category_probabilities(theta, item_or_delta)
        scores = np.asarray(self.score_values(item_or_delta), dtype=float).reshape(-1)
        if scores.size != p.shape[-1]:
            scores = scores[: p.shape[-1]] if scores.size >= p.shape[-1] else np.arange(self.min_score, self.min_score + p.shape[-1], dtype=float)
        ex = float(np.sum(p * scores, axis=-1))
        ex2 = float(np.sum(p * (scores ** 2), axis=-1))
        return max(ex2 - ex * ex, 1e-9)

    def information(self, theta: float, item_or_delta) -> float:
        return self.variance_score(theta, item_or_delta)


    def posterior(self, responses: List[Tuple[str, int]], start_theta: float | None = None) -> Tuple[float, float, np.ndarray]:
        grid = self.theta_grid
        mu = self.prior_mean if start_theta is None else float(start_theta)
        sd = max(self.prior_sd, 0.5)
        log_post = -0.5 * ((grid - mu) / sd) ** 2 - np.log(sd * math.sqrt(2.0 * math.pi))
        for item_id, score in responses:
            item = self.item_lookup.get(str(item_id))
            if item is None:
                continue
            probs = self.category_probabilities(grid, item)
            scores = self.score_values(item)
            if scores.size == 0:
                continue
            matches = np.where(np.isclose(scores, float(score)))[0]
            idx = int(matches[0]) if matches.size else int(np.argmin(np.abs(scores - float(score))))
            idx = max(0, min(idx, probs.shape[-1] - 1))
            p = np.clip(probs[..., idx], 1e-12, 1.0)
            log_post += np.log(p)
        log_post -= np.max(log_post)
        post = np.exp(log_post)
        den = _trapz_compat(post, grid)
        if not np.isfinite(den) or den <= 0:
            den = 1.0
        post = post / den
        mean = _trapz_compat(grid * post, grid)
        var = _trapz_compat(((grid - mean) ** 2) * post, grid)
        se = max(math.sqrt(max(var, 1e-10)), 1e-6)
        return mean, se, post

    def select_next_item(self, administered: List[str], theta: float) -> Optional[ItemRecord]:
        used = set(administered)
        remaining = [item for item in self.items if item.item_id not in used]
        if not remaining:
            return None
        return max(remaining, key=lambda item: (self.information(theta, item), -abs(item.delta - theta), -item.no))

    def next_linear_item(self, administered: List[str], start_no: int = 1) -> Optional[ItemRecord]:
        used = set(administered)
        ordered = sorted(self.items, key=lambda x: x.no)
        if not ordered:
            return None
        start_no = max(1, min(int(start_no), len(ordered)))
        start_idx = 0
        for idx, item in enumerate(ordered):
            if item.no >= start_no:
                start_idx = idx
                break
        rotated = ordered[start_idx:] + ordered[:start_idx]
        for item in rotated:
            if item.item_id not in used:
                return item
        return None

    def sample_voice_items(self, center_theta: float, theta_range: float, n_items: int) -> List[ItemRecord]:
        n_items = max(1, int(n_items))
        theta_range = max(0.05, float(theta_range))
        eligible = [item for item in self.items if abs(float(item.delta) - center_theta) <= theta_range]
        rng = random.SystemRandom()
        if len(eligible) > n_items:
            picked = rng.sample(eligible, n_items)
        else:
            picked = list(eligible)
        picked.sort(key=lambda item: (abs(float(item.delta) - center_theta), item.no))
        return picked

    def percentile(self, theta: float) -> float:
        vals = self.person_distribution
        if vals.size == 0:
            z = (theta - self.prior_mean) / self.prior_sd
            return 100.0 * (0.5 * (1 + math.erf(z / math.sqrt(2))))
        return 100.0 * float(np.mean(vals <= theta))


def make_home_wrightmap_svg(person_distribution: np.ndarray, item_df: pd.DataFrame, ref_theta: Optional[float] = None, ref_se: Optional[float] = None) -> str:
    if item_df is None or item_df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No item statistics available for the homepage Wright Map.</text>')
    df = item_df.copy()
    cols = {str(c).lower(): c for c in df.columns}
    item_col = cols.get('item_id') or cols.get('item') or cols.get('label') or cols.get('stem_en') or cols.get('stem_zh')
    measure_col = cols.get('measure') or cols.get('delta')
    infit_col = cols.get('infit') or cols.get('infit_mnsq')
    outfit_col = cols.get('outfit') or cols.get('outfit_mnsq')
    se_col = cols.get('se')
    no_col = cols.get('no') or cols.get('entry')
    stem_en_col = cols.get('stem_en')
    stem_zh_col = cols.get('stem_zh')
    if item_col is None or measure_col is None:
        return _svg_wrap(980, 360, '<text x="20" y="40">Missing item/item_id or measure columns for the homepage Wright Map.</text>')
    df['item_id'] = df[item_col].astype(str)
    df['measure_val'] = pd.to_numeric(df[measure_col], errors='coerce')
    if infit_col is not None:
        df['infit_val'] = pd.to_numeric(df[infit_col], errors='coerce')
    else:
        df['infit_val'] = 1.0
    df['outfit_val'] = pd.to_numeric(df[outfit_col], errors='coerce') if outfit_col is not None else np.nan
    df['se_val'] = pd.to_numeric(df[se_col], errors='coerce').fillna(0.12) if se_col is not None else 0.12
    stem_series = pd.Series([''] * len(df), index=df.index, dtype='object')
    if stem_en_col is not None:
        stem_series = df[stem_en_col].astype(str)
    if stem_zh_col is not None:
        zh = df[stem_zh_col].astype(str)
        stem_series = stem_series.mask(stem_series.eq('') | stem_series.eq('nan'), zh)
    df['stem_val'] = stem_series.fillna('').astype(str)
    df['no_val'] = pd.to_numeric(df[no_col], errors='coerce').fillna(0).astype(int) if no_col is not None else np.arange(1, len(df)+1)
    df = df.dropna(subset=['measure_val']).copy()
    df['infit_val'] = pd.to_numeric(df['infit_val'], errors='coerce').fillna(1.0)
    if df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No finite MEASURE rows available for the homepage Wright Map.</text>')

    width, height = 980, 430
    left, right, top, bottom = 46, 28, 28, 70
    hist_w, gap = 150, 30
    x0, y0 = left, height - bottom
    resid_x0 = x0 + hist_w + gap
    plot_w = width - resid_x0 - right
    plot_h = height - top - bottom

    measures = df['measure_val'].astype(float).to_numpy()
    infits = df['infit_val'].astype(float).to_numpy()
    ses = pd.to_numeric(df['se_val'], errors='coerce').fillna(0.12).astype(float).to_numpy()

    person_vals = np.asarray(person_distribution if person_distribution is not None else np.array([]), dtype=float)
    person_vals = person_vals[np.isfinite(person_vals)]
    if person_vals.size == 0:
        person_vals = np.array([0.0])
    theta_ref = float(np.nanmean(person_vals)) if ref_theta is None or not np.isfinite(ref_theta) else float(ref_theta)
    if ref_se is None or not np.isfinite(ref_se) or float(ref_se) <= 0:
        ref_se = 0.30
    se_band = max(float(ref_se), 0.0)

    ymin = float(min(np.min(measures), np.min(person_vals), theta_ref - se_band, theta_ref + se_band))
    ymax = float(max(np.max(measures), np.max(person_vals), theta_ref - se_band, theta_ref + se_band))
    if ymax <= ymin:
        ymax = ymin + 1.0
    ypad = max(0.4, (ymax - ymin) * 0.08)
    ymin -= ypad
    ymax += ypad

    xmin = float(np.nanmin(infits))
    xmax = float(np.nanmax(infits))
    if xmax <= xmin:
        xmax = xmin + 0.5
    xpad = max(0.08, (xmax - xmin) * 0.08)
    xmin -= xpad
    xmax += xpad

    def ymap(v: float) -> float:
        return y0 - (v - ymin) / (ymax - ymin) * plot_h

    def xmap(v: float) -> float:
        return resid_x0 + (v - xmin) / (xmax - xmin) * plot_w

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    bins = np.linspace(ymin, ymax, 22)
    counts, edges = np.histogram(person_vals, bins=bins)
    max_count = max(int(np.max(counts)), 1)
    parts.append(f'<rect x="{x0}" y="{top}" width="{hist_w}" height="{plot_h}" fill="#f3f4f6" stroke="#d1d5db"/>')
    for c, y1, y2 in zip(counts, edges[:-1], edges[1:]):
        if c <= 0:
            continue
        yy1, yy2 = ymap(y1), ymap(y2)
        bh = max(1.0, abs(yy1 - yy2) - 1.0)
        bw = c / max_count * (hist_w - 18)
        by = min(yy1, yy2) + 0.5
        parts.append(f'<rect x="{x0}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#5b7bd5" fill-opacity="0.9"/>')

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + hist_w}" y2="{y0}" stroke="#666"/>')
    parts.append(f'<line x1="{resid_x0}" y1="{y0}" x2="{resid_x0 + plot_w}" y2="{y0}" stroke="#666"/>')
    parts.append(f'<line x1="{resid_x0}" y1="{top}" x2="{resid_x0}" y2="{y0}" stroke="#666"/>')
    for frac in np.linspace(0.0, 1.0, 6):
        yv = ymin + (ymax - ymin) * frac
        yy = ymap(yv)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{resid_x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>')

    y_ref = ymap(theta_ref)
    y_hi = ymap(theta_ref + se_band)
    y_lo = ymap(theta_ref - se_band)
    parts.append(f'<line x1="{x0}" y1="{y_ref:.1f}" x2="{resid_x0 + plot_w}" y2="{y_ref:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>')
    parts.append(f'<line x1="{x0}" y1="{y_hi:.1f}" x2="{resid_x0 + plot_w}" y2="{y_hi:.1f}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="4,4"/>')
    parts.append(f'<line x1="{x0}" y1="{y_lo:.1f}" x2="{resid_x0 + plot_w}" y2="{y_lo:.1f}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{resid_x0 + plot_w - 4:.1f}" y="{max(top + 12, y_ref - 6):.1f}" text-anchor="end" font-size="11" fill="#b91c1c">Ref θ={theta_ref:.2f}</text>')
    parts.append(f'<text x="{resid_x0 + plot_w - 4:.1f}" y="{max(top + 24, y_hi - 6):.1f}" text-anchor="end" font-size="10" fill="#b91c1c">+1 SE={theta_ref + se_band:.2f}</text>')
    parts.append(f'<text x="{resid_x0 + plot_w - 4:.1f}" y="{min(y0 - 6, y_lo + 12):.1f}" text-anchor="end" font-size="10" fill="#b91c1c">-1 SE={theta_ref - se_band:.2f}</text>')

    xticks = sorted(set([round(v, 2) for v in np.linspace(xmin, xmax, 6)] + [1.5]))
    for xv in xticks:
        xx = xmap(xv)
        dash = '4,4' if abs(xv - 1.5) < 1e-9 else 'none'
        stroke = '#dc2626' if abs(xv - 1.5) < 1e-9 else '#d1d5db'
        sw = '2' if abs(xv - 1.5) < 1e-9 else '1'
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{y0}" stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash != 'none' else '') + '/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="11" fill="#111827">{xv:.2f}</text>')

    se_min = float(np.nanmin(ses)) if len(ses) else 0.12
    se_max = float(np.nanmax(ses)) if len(ses) else 0.12
    def rmap(se: float) -> float:
        if se_max <= se_min + 1e-9:
            return 7.5
        return 5.0 + (se - se_min) / (se_max - se_min) * 9.0

    for row in df.itertuples(index=False):
        item_id = html.escape(str(row.item_id))
        measure = float(row.measure_val)
        infit = float(row.infit_val)
        outfit = float(row.outfit_val) if pd.notna(row.outfit_val) else float('nan')
        se = float(row.se_val) if np.isfinite(float(row.se_val)) else 0.12
        x = xmap(infit)
        y = ymap(measure)
        r = rmap(se)
        over = infit > 1.5
        fill = '#ef4444' if over else '#2563eb'
        stroke = '#991b1b' if over else '#1e3a8a'
        label = html.escape(str(row.stem_val)[:80])
        no_txt = int(row.no_val)
        title = f'Item {no_txt} | {item_id} | {label} | Measure={measure:.3f} | INFIT MNSQ={infit:.3f} | OUTFIT MNSQ={outfit:.3f} | SE={se:.3f}'
        parts.append(f'<circle class="map-dot" data-item-id="{item_id}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="0.78" stroke="{stroke}" stroke-width="1.2"><title>{title}</title></circle>')

    high_n = int(np.sum(infits > 1.5))
    parts.append(f'<text x="{x0 + hist_w/2:.1f}" y="16" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Persons (distribution)</text>')
    parts.append(f'<text x="{resid_x0 + plot_w/2:.1f}" y="16" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Items by INFIT MNSQ</text>')
    parts.append(f'<text x="{resid_x0 + plot_w/2:.1f}" y="{height - 18:.1f}" text-anchor="middle" font-size="12">INFIT MNSQ</text>')
    parts.append(f'<text x="16" y="{top + plot_h/2:.1f}" transform="rotate(-90 16 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">Item / person measure</text>')
    parts.append(f'<text x="{resid_x0 + plot_w - 4:.1f}" y="{height - 40:.1f}" text-anchor="end" font-size="11" fill="#991b1b">Items beyond 1.5: {high_n}</text>')
    return _svg_wrap(width, height, ''.join(parts))



def _select_reference_person_from_zscore(person_df: pd.DataFrame, zscore_df: pd.DataFrame, target_theta: Optional[float] = None):
    if person_df is None or person_df.empty or zscore_df is None or zscore_df.empty:
        return None, None, None, None, None
    pdf = person_df.copy()
    zdf = zscore_df.copy()
    pdf_cols = {str(c).lower(): c for c in pdf.columns}
    zdf_cols = {str(c).lower(): c for c in zdf.columns}
    pdf_kid = pdf_cols.get('kid') or pdf_cols.get('id') or pdf_cols.get('entry')
    zdf_kid = zdf_cols.get('kid') or zdf_cols.get('id') or zdf_cols.get('entry')
    meas_col = pdf_cols.get('measure')
    se_col = pdf_cols.get('se')
    profile_pdf_col = pdf_cols.get('profile')
    profile_zdf_col = zdf_cols.get('profile')
    if pdf_kid is None or zdf_kid is None or meas_col is None:
        return None, None, None, None, None
    pdf = pdf.copy()
    pdf['_kid_str'] = pdf[pdf_kid].astype(str).str.strip()
    pdf['_measure_val'] = pd.to_numeric(pdf[meas_col], errors='coerce')
    pdf['_se_val'] = pd.to_numeric(pdf[se_col], errors='coerce').fillna(0.30) if se_col is not None else 0.30
    pdf = pdf.dropna(subset=['_measure_val']).copy()
    if pdf.empty:
        return None, None, None, None, None
    target = float(np.nanmean(pdf['_measure_val'])) if target_theta is None or not np.isfinite(target_theta) else float(target_theta)
    pdf['_dist'] = (pdf['_measure_val'] - target).abs()
    pdf = pdf.sort_values(['_dist', '_kid_str']).reset_index(drop=True)
    zdf = zdf.copy()
    zdf['_kid_str'] = zdf[zdf_kid].astype(str).str.strip()
    for _, prow in pdf.iterrows():
        kid = str(prow['_kid_str'])
        hit = zdf.loc[zdf['_kid_str'] == kid]
        if not hit.empty:
            zrow = hit.iloc[0]
            profile = None
            if profile_pdf_col is not None and pd.notna(prow.get(profile_pdf_col)):
                profile = prow.get(profile_pdf_col)
            elif profile_zdf_col is not None and pd.notna(zrow.get(profile_zdf_col)):
                profile = zrow.get(profile_zdf_col)
            return kid, float(prow['_measure_val']), float(prow['_se_val']), profile, zrow
    return None, None, None, None, None


def make_home_kidmap_svg(person_distribution: np.ndarray, item_df: pd.DataFrame, zscore_df: Optional[pd.DataFrame] = None, person_df: Optional[pd.DataFrame] = None, ref_theta: Optional[float] = None, ref_se: Optional[float] = None) -> str:
    if item_df is None or item_df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No item statistics available for the homepage KIDMAP.</text>')
    if zscore_df is None or zscore_df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">zscore.csv is missing, so the homepage KIDMAP cannot be drawn.</text>')

    df = item_df.copy()
    cols = {str(c).lower(): c for c in df.columns}
    item_col = cols.get('item_id') or cols.get('item') or cols.get('label') or cols.get('stem_en') or cols.get('stem_zh')
    measure_col = cols.get('measure') or cols.get('delta')
    se_col = cols.get('se')
    no_col = cols.get('no') or cols.get('entry')
    stem_en_col = cols.get('stem_en')
    stem_zh_col = cols.get('stem_zh')
    if item_col is None or measure_col is None:
        return _svg_wrap(980, 360, '<text x="20" y="40">Missing item/item_id or measure columns for the homepage KIDMAP.</text>')

    df['item_id'] = df[item_col].astype(str)
    df['measure_val'] = pd.to_numeric(df[measure_col], errors='coerce')
    df['se_val'] = pd.to_numeric(df[se_col], errors='coerce').fillna(0.12) if se_col is not None else 0.12
    stem_series = pd.Series([''] * len(df), index=df.index, dtype='object')
    if stem_en_col is not None:
        stem_series = df[stem_en_col].astype(str)
    if stem_zh_col is not None:
        zh = df[stem_zh_col].astype(str)
        stem_series = stem_series.mask(stem_series.eq('') | stem_series.eq('nan'), zh)
    df['stem_val'] = stem_series.fillna('').astype(str)
    df['no_val'] = pd.to_numeric(df[no_col], errors='coerce').fillna(0).astype(int) if no_col is not None else np.arange(1, len(df)+1)
    df = df.dropna(subset=['measure_val']).copy()
    if df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No finite item measures available for the homepage KIDMAP.</text>')

    person_vals = np.asarray(person_distribution if person_distribution is not None else np.array([]), dtype=float)
    person_vals = person_vals[np.isfinite(person_vals)]
    if person_vals.size == 0:
        person_vals = np.array([0.0])

    ref_kid, ref_person_theta, ref_person_se, ref_profile, zrow = _select_reference_person_from_zscore(person_df if isinstance(person_df, pd.DataFrame) else pd.DataFrame(), zscore_df, target_theta=ref_theta)
    if zrow is None:
        return _svg_wrap(980, 360, '<text x="20" y="40">Could not match zscore.csv to person_estimates.csv for the homepage KIDMAP.</text>')

    theta_ref = float(ref_person_theta if ref_person_theta is not None and np.isfinite(ref_person_theta) else (np.nanmean(person_vals) if person_vals.size else 0.0))
    if ref_se is not None and np.isfinite(ref_se) and float(ref_se) > 0:
        se_band = float(ref_se)
    else:
        se_band = float(ref_person_se if ref_person_se is not None and np.isfinite(ref_person_se) and float(ref_person_se) > 0 else 0.30)

    zcols_lower = {str(c).lower(): c for c in zscore_df.columns}
    skip_cols = {c for k, c in zcols_lower.items() if k in {'kid', 'id', 'entry', 'profile'}}
    z_item_cols = [c for c in zscore_df.columns if c not in skip_cols]
    item_to_z = {}
    for col in z_item_cols:
        item_to_z[str(col).strip()] = _safe_float(zrow.get(col), float('nan'))

    # Primary: exact item-name match.
    df['_item_key'] = df['item_id'].astype(str).str.strip()
    df['z_val'] = df['_item_key'].map(lambda x: item_to_z.get(x, float('nan')))

    # Fallback 1: ENTRY/no_val positional match to non-metadata zscore columns.
    if df['z_val'].notna().sum() == 0 and len(z_item_cols) > 0:
        order_map = {int(i + 1): _safe_float(zrow.get(col), float('nan')) for i, col in enumerate(z_item_cols)}
        df['z_val'] = df['no_val'].map(order_map)

    # Fallback 2: if zscore columns look like a1,a2,... but item labels are descriptive, map by extracted ordinal.
    if df['z_val'].notna().sum() == 0 and len(z_item_cols) > 0:
        token_map = {}
        for col in z_item_cols:
            m = re.search(r'(?:^|[^0-9])(\d+)(?:$|[^0-9])', str(col))
            if m:
                token_map[int(m.group(1))] = _safe_float(zrow.get(col), float('nan'))
        if token_map:
            df['z_val'] = df['no_val'].map(token_map)

    df['z_val'] = pd.to_numeric(df['z_val'], errors='coerce')
    df = df.dropna(subset=['z_val']).copy()
    if df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No finite z-scores available for the selected reference person after matching zscore.csv columns to item rows.</text>')
    df['z_val'] = df['z_val'].clip(-4.0, 4.0)

    width, height = 980, 430
    left, right, top, bottom = 46, 28, 28, 70
    hist_w, gap = 150, 30
    x0, y0 = left, height - bottom
    plot_x0 = x0 + hist_w + gap
    plot_w = width - plot_x0 - right
    plot_h = height - top - bottom

    measures = df['measure_val'].astype(float).to_numpy()
    zvals = np.clip(df['z_val'].astype(float).to_numpy(), -4.0, 4.0)
    ses = pd.to_numeric(df['se_val'], errors='coerce').fillna(0.12).astype(float).to_numpy()

    ymin = float(min(np.min(measures), np.min(person_vals), theta_ref - se_band, theta_ref + se_band))
    ymax = float(max(np.max(measures), np.max(person_vals), theta_ref - se_band, theta_ref + se_band))
    if ymax <= ymin:
        ymax = ymin + 1.0
    ypad = max(0.4, (ymax - ymin) * 0.08)
    ymin -= ypad
    ymax += ypad
    xmin, xmax = -4.0, 4.0

    def ymap(v: float) -> float:
        return y0 - (v - ymin) / (ymax - ymin) * plot_h
    def xmap(v: float) -> float:
        v = max(xmin, min(xmax, v))
        return plot_x0 + (v - xmin) / (xmax - xmin) * plot_w

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    bins = np.linspace(ymin, ymax, 22)
    counts, edges = np.histogram(person_vals, bins=bins)
    max_count = max(int(np.max(counts)), 1)
    parts.append(f'<rect x="{x0}" y="{top}" width="{hist_w}" height="{plot_h}" fill="#f3f4f6" stroke="#d1d5db"/>')
    for c, y1, y2 in zip(counts, edges[:-1], edges[1:]):
        if c <= 0:
            continue
        yy1, yy2 = ymap(y1), ymap(y2)
        bh = max(1.0, abs(yy1 - yy2) - 1.0)
        bw = c / max_count * (hist_w - 18)
        by = min(yy1, yy2) + 0.5
        parts.append(f'<rect x="{x0}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#5b7bd5" fill-opacity="0.9"/>')

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + hist_w}" y2="{y0}" stroke="#666"/>')
    parts.append(f'<line x1="{plot_x0}" y1="{y0}" x2="{plot_x0 + plot_w}" y2="{y0}" stroke="#666"/>')
    parts.append(f'<line x1="{plot_x0}" y1="{top}" x2="{plot_x0}" y2="{y0}" stroke="#666"/>')
    for frac in np.linspace(0.0, 1.0, 6):
        yv = ymin + (ymax - ymin) * frac
        yy = ymap(yv)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{plot_x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>')
    for xv in [-4,-2,0,2,4]:
        xx = xmap(float(xv))
        dash = '6,4' if abs(xv) == 2 else ('4,4' if xv == 0 else 'none')
        stroke = '#dc2626' if abs(xv) == 2 else ('#64748b' if xv == 0 else '#d1d5db')
        sw = '2' if abs(xv) == 2 else '1'
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{y0}" stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash != 'none' else '') + '/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="11" fill="#111827">{xv:.0f}</text>')

    se_min = float(np.nanmin(ses)) if len(ses) else 0.12
    se_max = float(np.nanmax(ses)) if len(ses) else 0.12
    def rmap(se: float) -> float:
        if se_max <= se_min + 1e-9:
            return 7.5
        return 5.0 + (se - se_min) / (se_max - se_min) * 9.0

    for row in df.itertuples(index=False):
        item_id = html.escape(str(row.item_id))
        measure = float(row.measure_val)
        zval = float(row.z_val)
        se = float(row.se_val) if np.isfinite(float(row.se_val)) else 0.12
        x = xmap(zval)
        y = ymap(measure)
        r = rmap(se)
        over = abs(zval) > 2.0
        fill = '#ef4444' if over else '#2563eb'
        stroke = '#991b1b' if over else '#1e3a8a'
        label = html.escape(str(row.stem_val)[:80])
        no_txt = int(row.no_val)
        title = f'Item {no_txt} | {item_id} | {label} | Measure={measure:.3f} | Person-level Z={zval:.3f} | Ref person={ref_kid} | SE={se:.3f}'
        parts.append(f'<circle class="map-dot" data-item-id="{item_id}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="0.78" stroke="{stroke}" stroke-width="1.2"><title>{title}</title></circle>')

    y_theta = ymap(theta_ref)
    parts.append(f'<line x1="{plot_x0:.1f}" x2="{plot_x0 + plot_w:.1f}" y1="{y_theta:.1f}" y2="{y_theta:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7,5" />')
    parts.append(f'<text x="{plot_x0 + plot_w - 6:.1f}" y="{y_theta - 6:.1f}" text-anchor="end" font-size="11" fill="#991b1b">Person measure {theta_ref:.2f}</text>')
    for band_val, band_lab in ((theta_ref + se_band, '+1 SE'), (theta_ref - se_band, '-1 SE')):
        yb = ymap(band_val)
        parts.append(f'<line x1="{plot_x0:.1f}" x2="{plot_x0 + plot_w:.1f}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="4,4" />')
        parts.append(f'<text x="{plot_x0 + plot_w - 6:.1f}" y="{yb - 4:.1f}" text-anchor="end" font-size="10" fill="#991b1b">{band_lab}</text>')

    profile_text = '' if ref_profile is None or (isinstance(ref_profile, float) and not np.isfinite(ref_profile)) else f'  Profile={html.escape(str(ref_profile))}'
    parts.append(f'<text x="{x0 + hist_w/2:.1f}" y="16" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Persons (distribution)</text>')
    parts.append(f'<text x="{plot_x0 + plot_w/2:.1f}" y="16" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Items by standardized residual Z</text>')
    parts.append(f'<text x="{plot_x0 + plot_w/2:.1f}" y="{height - 18:.1f}" text-anchor="middle" font-size="12">Standardized residual Z (reference person)</text>')
    parts.append(f'<text x="16" y="{top + plot_h/2:.1f}" transform="rotate(-90 16 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">Item / person measure</text>')
    parts.append(f'<text x="{plot_x0 + 4:.1f}" y="{height - 42:.1f}" text-anchor="start" font-size="11" fill="#991b1b">Reference person = {html.escape(str(ref_kid))}, θ = {theta_ref:.2f}, SE = {se_band:.2f}{profile_text}</text>')
    return _svg_wrap(width, height, ''.join(parts))

def make_trend_svg(history: List[dict]) -> str:
    if not history:
        return _svg_wrap(860, 240, '<text x="20" y="40">No CAT trend data available.</text>')
    width, height = 900, 300
    left, right, top, bottom = 58, 20, 24, 42
    x0, y0 = left, height - bottom
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = list(range(1, len(history) + 1))
    theta_vals = [float(row.get('theta', 0.0)) for row in history]
    delta_vals = [float(row.get('delta', 0.0)) for row in history]
    zstd_vals = [float(row.get('zstd', 0.0)) for row in history]
    all_vals = theta_vals + delta_vals + zstd_vals + [-2.0, 2.0]
    ymin, ymax = min(all_vals), max(all_vals)
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = max(0.4, (ymax - ymin) * 0.08)
    ymin -= pad
    ymax += pad

    def x_map(v: float) -> float:
        return x0 + (v - 1) / max(1, len(xs) - 1) * plot_w if len(xs) > 1 else x0 + plot_w / 2

    def y_map(v: float) -> float:
        return y0 - (v - ymin) / (ymax - ymin) * plot_h

    def series_path(vals: List[float]) -> str:
        pts = [f"{x_map(i+1):.1f},{y_map(v):.1f}" for i, v in enumerate(vals)]
        return "M " + " L ".join(pts) if pts else ""

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
             f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}" stroke="#444"/>',
             f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#444"/>']
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        yv = ymin + (ymax - ymin) * frac
        yy = y_map(yv)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>')
    colors = [(theta_vals, '#2563eb', 'Theta'), (delta_vals, '#dc2626', 'Item difficulty'), (zstd_vals, '#059669', 'ZSTD')]
    for vals, color, label in colors:
        path_d = series_path(vals)
        if path_d:
            parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for i, v in enumerate(vals, start=1):
            parts.append(f'<circle cx="{x_map(i):.1f}" cy="{y_map(v):.1f}" r="3.2" fill="{color}"/>')
    legend_x, legend_y = x0 + 8, top + 8
    for j, (_, color, label) in enumerate(colors):
        ly = legend_y + j * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 22}" y2="{ly}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{ly + 4}" font-size="11">{label}</text>')
    return _svg_wrap(width, height, ''.join(parts))


def make_combined_kidmap_svg(person_values: np.ndarray, theta: float, person_se: float, rows: List[Dict[str, object]], infit_mnsq: float = 1.0, outfit_mnsq: float = 1.0) -> str:
    values = np.asarray(person_values, dtype=float)
    values = values[np.isfinite(values)]
    item_deltas = np.array([float(r.get('delta', 0.0)) for r in rows], dtype=float) if rows else np.array([], dtype=float)
    se_band = max(float(person_se or 0.0), 0.0)
    ymin = min(values.min() if values.size else theta, item_deltas.min() if item_deltas.size else theta, theta - se_band, theta) - 0.6
    ymax = max(values.max() if values.size else theta, item_deltas.max() if item_deltas.size else theta, theta + se_band, theta) + 0.6
    if ymax <= ymin:
        ymax = ymin + 1.0

    width, height = 980, 680
    left, right, top, bottom = 68, 28, 56, 56
    strip_w, gap_w = 152, 18
    x0, y0 = left, top
    plot_h = height - top - bottom
    resid_x0 = x0 + strip_w + gap_w
    plot_w = width - resid_x0 - right
    xmin, xmax = -4.0, 4.0

    def xmap(val: float) -> float:
        val = max(xmin, min(xmax, val))
        return resid_x0 + (val - xmin) / (xmax - xmin) * plot_w

    def ymap(val: float) -> float:
        return y0 + (ymax - val) / (ymax - ymin) * plot_h

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    title = f'KIDMAP — Measure {theta:.2f} (SE {se_band:.2f})  INFIT {infit_mnsq:.2f}  OUTFIT {outfit_mnsq:.2f}'
    parts.append(f'<text x="{x0}" y="26" font-size="16" font-weight="700" fill="#111827">{html.escape(title)}</text>')

    for t in range(math.floor(ymin), math.ceil(ymax) + 1):
        y = ymap(float(t))
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{resid_x0+plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x0-12}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">{t}</text>')

    parts.append(f'<rect x="{x0}" y="{y0}" width="{strip_w}" height="{plot_h}" fill="#f8fafc" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{x0 + strip_w/2:.1f}" y="46" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Persons (distribution)</text>')
    if values.size:
        n_bins = max(14, min(28, int(round((ymax - ymin) * 4))))
        edges = np.linspace(ymin, ymax, n_bins + 1)
        counts, _ = np.histogram(values, bins=edges)
        max_count = max(int(counts.max()), 1)
        inner_left = x0 + 8
        usable_w = strip_w - 18
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            y_top = ymap(edges[i + 1])
            y_bot = ymap(edges[i])
            bar_h = max(2.0, y_bot - y_top - 1.5)
            cy = y_top + (y_bot - y_top) / 2.0
            bar_w = usable_w * (count / max_count)
            parts.append(f'<rect x="{inner_left:.1f}" y="{cy - bar_h/2:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="#4361c2" fill-opacity="0.82" stroke="#3148a5" stroke-opacity="0.55"/>')

    sep_x = resid_x0 - gap_w/2
    parts.append(f'<line x1="{sep_x:.1f}" y1="{y0}" x2="{sep_x:.1f}" y2="{y0+plot_h}" stroke="#cbd5e1" stroke-width="1"/>')
    parts.append(f'<text x="{resid_x0 + plot_w/2:.1f}" y="46" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">KIDMAP (cell ZSTD)</text>')

    for zv in (-2, 2):
        x = xmap(float(zv))
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+plot_h}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-20}" text-anchor="middle" font-size="11" fill="#dc2626">{zv}</text>')
    x_zero = xmap(0.0)
    parts.append(f'<line x1="{x_zero:.1f}" y1="{y0}" x2="{x_zero:.1f}" y2="{y0+plot_h}" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{x_zero:.1f}" y="{height-20}" text-anchor="middle" font-size="11" fill="#64748b">0</text>')
    parts.append(f'<line x1="{resid_x0}" y1="{y0+plot_h}" x2="{resid_x0+plot_w}" y2="{y0+plot_h}" stroke="#374151"/>')
    parts.append(f'<text x="{(resid_x0+resid_x0+plot_w)/2:.1f}" y="{height-16}" text-anchor="middle" font-size="12">ZSTD</text>')
    parts.append(f'<text x="{x0-52}" y="{y0-8}" font-size="12" fill="#4b5563">Logit</text>')

    theta_y = ymap(theta)
    parts.append(f'<line x1="{x0}" y1="{theta_y:.1f}" x2="{resid_x0+plot_w}" y2="{theta_y:.1f}" stroke="#dc2626" stroke-width="2"/>')
    parts.append(f'<text x="{resid_x0+plot_w-6}" y="{max(16, theta_y-6):.1f}" text-anchor="end" font-size="12" fill="#991b1b">Measure {theta:.2f}</text>')
    for band_val, band_lab in ((theta + se_band, '+1 SE'), (theta - se_band, '-1 SE')):
        by = ymap(band_val)
        parts.append(f'<line x1="{x0}" y1="{by:.1f}" x2="{resid_x0+plot_w}" y2="{by:.1f}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{resid_x0+plot_w-6}" y="{max(16, by-4):.1f}" text-anchor="end" font-size="11" fill="#991b1b">{band_lab} ({band_val:.2f})</text>')

    if rows:
        ses = [float(r.get('item_se', 0.12) or 0.12) for r in rows]
        max_se, min_se = max(ses), min(ses)
        def rmap(se: float) -> float:
            return 8.0 if max_se <= min_se + 1e-9 else 5.0 + (se - min_se) / (max_se - min_se) * 9.0
        for row in rows:
            delta = float(row.get('delta', 0.0))
            z = float(row.get('zscore', 0.0))
            se = float(row.get('item_se', 0.12) or 0.12)
            item_id = html.escape(str(row.get('item_id', '')))
            x, y, r = xmap(z), ymap(delta), rmap(se)
            fill = '#2563eb' if z >= 0 else '#dc2626'
            stroke = '#1e3a8a' if z >= 0 else '#7f1d1d'
            stroke_w = 1.5 if abs(z) > 2 else 1.1
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="0.78" stroke="{stroke}" stroke-width="{stroke_w}"/>')
            tx = min(resid_x0 + plot_w - 4, x + r + 5)
            parts.append(f'<text x="{tx:.1f}" y="{y+4:.1f}" font-size="10" fill="#111827">{item_id}</text>')
    return _svg_wrap(width, height, ''.join(parts))


def make_cpc_svg(bank: RaschPCMBank, theta_person: float, ref_item=None, ref_delta=None) -> Tuple[str, str]:
    grid = np.linspace(-6.0, 6.0, 901)
    item = bank._resolve_item(ref_item)
    if item is None and ref_delta is not None and getattr(bank, 'items', None):
        try:
            ref_val = float(ref_delta)
            item = min(bank.items, key=lambda it: (abs(float(it.delta) - ref_val), -len(bank.raw_thresholds(it)), it.no))
        except Exception:
            item = None
    if item is None and getattr(bank, 'items', None):
        item = max(bank.items, key=lambda it: (len(bank.raw_thresholds(it)), -abs(float(it.delta)), -it.no))
    ref_obj = item if item is not None else 0.0
    probs = bank.category_probabilities(grid, ref_obj)
    width, height = 980, 420
    left, right, top, bottom = 62, 24, 28, 52
    x0, y0 = left, height - bottom
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 0.0, 1.0
    xmin, xmax = float(grid.min()), float(grid.max())

    def xmap(v: float) -> float:
        return x0 + (v - xmin) / (xmax - xmin) * plot_w
    def ymap(v: float) -> float:
        return y0 - (v - ymin) / (ymax - ymin) * plot_h

    palette = ['#1d4ed8', '#059669', '#f59e0b', '#7c3aed', '#dc2626', '#0f766e', '#9333ea']
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
             f'<line x1="{x0}" y1="{y0}" x2="{x0+plot_w}" y2="{y0}" stroke="#444"/>',
             f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#444"/>']
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = ymap(frac)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{frac:.2f}</text>')
    for xv in range(-6, 7):
        xx = xmap(float(xv))
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+4}" stroke="#64748b"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0+18:.1f}" text-anchor="middle" font-size="11">{xv}</text>')

    for j in range(probs.shape[1]):
        pts = [f"{xmap(grid[i]):.1f},{ymap(float(probs[i,j])):.1f}" for i in range(len(grid))]
        parts.append(f'<path d="M ' + ' L '.join(pts) + f'" fill="none" stroke="{palette[j % len(palette)]}" stroke-width="2.3"/>')

    raw_taus = bank.raw_thresholds(ref_obj)
    for tau in np.asarray(raw_taus, dtype=float).reshape(-1):
        xx = xmap(float(tau))
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{y0}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top-8 if top>10 else 12}" text-anchor="middle" font-size="10" fill="#dc2626">{tau:.2f}</text>')

    xxp = xmap(theta_person)
    parts.append(f'<line x1="{xxp:.1f}" y1="{top}" x2="{xxp:.1f}" y2="{y0}" stroke="#b91c1c" stroke-width="2.4"/>')
    parts.append(f'<text x="{xxp:.1f}" y="{y0+34:.1f}" text-anchor="middle" font-size="11" fill="#b91c1c">Person θ = {theta_person:.2f}</text>')

    p_person = np.asarray(bank.category_probabilities(theta_person, ref_obj), dtype=float).reshape(-1)
    scores = np.asarray(bank.score_values(ref_obj), dtype=float).reshape(-1)
    if scores.size != p_person.size:
        scores = scores[: p_person.size] if scores.size >= p_person.size else np.arange(bank.min_score, bank.min_score + p_person.size, dtype=float)
    pred_idx = int(np.argmax(p_person))
    pred_score = int(scores[pred_idx]) if scores.size else bank.min_score + pred_idx
    pred_opt = {int(op.score): op for op in bank.response_options('en', ref_obj)}.get(pred_score)
    pred_label = f"{pred_opt.label} = {pred_opt.text}" if pred_opt else str(pred_score)
    title_item = getattr(item, 'item_id', 'reference item') if item is not None else 'reference item'
    parts.append(f'<text x="{width/2:.1f}" y="20" text-anchor="middle" font-size="16" font-weight="700">Category Probability Curves ({html.escape(str(title_item))})</text>')
    parts.append(f'<text x="{x0 + plot_w/2:.1f}" y="{height - 12:.1f}" text-anchor="middle" font-size="12">Theta for reference item</text>')
    parts.append(f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">Category probability</text>')
    return _svg_wrap(width, height, ''.join(parts)), pred_label


def compute_person_fit(bank: RaschPCMBank, responses: List[Tuple[str, int]], theta: float) -> Tuple[float, float]:
    if not responses:
        return 1.0, 1.0
    numer = denom = 0.0
    outfit_terms = []
    for item_id, score in responses:
        item = bank.item_lookup.get(item_id)
        if not item:
            continue
        exp_score = bank.expected_score(theta, item)
        var = bank.variance_score(theta, item)
        resid2 = (float(score) - exp_score) ** 2
        numer += resid2
        denom += var
        outfit_terms.append(resid2 / max(var, 1e-9))
    return (numer / denom if denom > 0 else 1.0), (float(np.mean(outfit_terms)) if outfit_terms else 1.0)



def skin_cancer_risk_classification(theta: float) -> dict:
    """Classify Rasch-estimated skin-cancer/melanoma risk from theta."""
    try:
        t = float(theta)
    except Exception:
        t = 0.0
    if t < -0.5:
        level = "Low / mild risk"
        explanation = "The estimated risk level is below the average range."
    elif t <= 0.5:
        level = "Average / moderate risk"
        explanation = "The estimated risk level is close to the average range."
    elif t <= 1.5:
        level = "High risk"
        explanation = "The estimated risk level is above the average range."
    else:
        level = "Very high risk"
        explanation = "The estimated risk level is far above the average range."
    return {
        "theta": t,
        "level": level,
        "explanation": explanation,
        "rule": "theta < -0.5 = low/mild; -0.5 to 0.5 = average/moderate; 0.5 to 1.5 = high; > 1.5 = very high",
    }


def mode_name(mode: str) -> str:
    return {
        "cat": "CAT",
        "cat_seq_sim": "CAT sequential simulation",
        "linear": "non-CAT",
        "voice": "Voice practice",
        "compare": "CAT vs non-CAT(1)",
        "compare_n": "CAT vs non-CAT(n)",
    }.get(str(mode), "CAT")


def synthesize_tts_path(text_value: str, language: str) -> Optional[Path]:
    if gTTS is None:
        return None
    payload = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not payload:
        return None
    lang_code = "zh-TW" if language == "zh" else "en"
    key = hashlib.sha1(f"{lang_code}|{payload}".encode("utf-8")).hexdigest()
    out_path = TTS_CACHE_DIR / f"{key}.mp3"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    try:
        tts = gTTS(text=payload, lang=lang_code, slow=False)
        with open(out_path, "wb") as fh:
            tts.write_to_fp(fh)
        return out_path if out_path.exists() else None
    except Exception:
        return None


def voice_part_texts(item: ItemRecord, bank: RaschPCMBank, language: str, item_number: int) -> str:
    stem = item.stem_for(language)
    if language == "zh":
        return f"第{item_number}題。{stem}。"
    return f"Question {item_number}. {stem}."




def _safe_float(val, default: float = float('nan')) -> float:
    try:
        out = float(val)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def summarize_numeric(values: List[float]) -> dict:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {'n': 0, 'mean_no': 0.0, 'sd_no': 0.0, 'mean_delta': 0.0, 'sd_delta': 0.0}
    return {'n': int(arr.size), 'mean': float(np.mean(arr)), 'sd': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}


def welch_ttest(a: List[float], b: List[float]) -> dict:
    a_arr = np.asarray([float(x) for x in a if np.isfinite(float(x))], dtype=float)
    b_arr = np.asarray([float(x) for x in b if np.isfinite(float(x))], dtype=float)
    if a_arr.size < 2 or b_arr.size < 2:
        return {'t': 0.0, 'p': None, 'df': None, 'p_text': 'NA', 'df_text': 'NA'}
    try:
        from scipy import stats
        res = stats.ttest_ind(a_arr, b_arr, equal_var=False, nan_policy='omit')
        sa2 = float(np.var(a_arr, ddof=1) / a_arr.size)
        sb2 = float(np.var(b_arr, ddof=1) / b_arr.size)
        df = (sa2 + sb2) ** 2 / ((sa2 ** 2) / (a_arr.size - 1) + (sb2 ** 2) / (b_arr.size - 1)) if (a_arr.size > 1 and b_arr.size > 1) else None
        p = float(res.pvalue) if res.pvalue is not None and np.isfinite(res.pvalue) else None
        return {'t': float(res.statistic), 'p': p, 'df': df, 'p_text': f'{p:.4g}' if p is not None else 'NA', 'df_text': f'{df:.2f}' if df is not None else 'NA'}
    except Exception:
        return {'t': 0.0, 'p': None, 'df': None, 'p_text': 'NA', 'df_text': 'NA'}


def build_dashboard_rows(bank: RaschPCMBank) -> List[dict]:
    rows = []
    fit_df = bank.item_fit_df.copy() if bank.item_fit_df is not None else pd.DataFrame()
    fit_map = {}
    if not fit_df.empty:
        item_col = _find_column(fit_df, ['ITEM', 'item'])
        meas_col = _find_column(fit_df, ['MEASURE', 'measure'])
        infit_col = _find_column(fit_df, ['INFIT_MNSQ', 'INFIT MNSQ', 'INFIT'])
        outfit_col = _find_column(fit_df, ['OUTFIT_MNSQ', 'OUTFIT MNSQ', 'OUTFIT'])
        infit_z_col = _find_column(fit_df, ['INFIT_ZSTD', 'INFIT ZSTD'])
        outfit_z_col = _find_column(fit_df, ['OUTFIT_ZSTD', 'OUTFIT ZSTD'])
        se_col = _find_column(fit_df, ['SE', 'MODEL_SE', 'MODEL SE', 'S.E.', 'S.E'])
        if item_col:
            for _, row in fit_df.iterrows():
                item_id = str(row.get(item_col, ''))
                fit_map[item_id] = {
                    'measure': _safe_float(row.get(meas_col)) if meas_col else float('nan'),
                    'infit': _safe_float(row.get(infit_col)) if infit_col else float('nan'),
                    'outfit': _safe_float(row.get(outfit_col)) if outfit_col else float('nan'),
                    'infit_zstd': _safe_float(row.get(infit_z_col)) if infit_z_col else float('nan'),
                    'outfit_zstd': _safe_float(row.get(outfit_z_col)) if outfit_z_col else float('nan'),
                    'se': _safe_float(row.get(se_col), 0.12) if se_col else 0.12,
                }
    for item in bank.items:
        stats = fit_map.get(item.item_id, {})
        link_href, is_image = resolve_link_href(item.link)
        rows.append({
            'item_id': item.item_id,
            'no': int(item.no),
            'stem_zh': item.stem_zh,
            'stem_en': item.stem_en,
            'delta': float(item.delta),
            'measure': stats.get('measure', float(item.delta)),
            'infit': stats.get('infit', float('nan')),
            'outfit': stats.get('outfit', float('nan')),
            'infit_zstd': stats.get('infit_zstd', float('nan')),
            'outfit_zstd': stats.get('outfit_zstd', float('nan')),
            'se': stats.get('se', 0.12),
            'link_href': link_href,
            'is_image_link': bool(is_image),
        })
    rows.sort(key=lambda x: x['no'])
    return rows


def linear_sequence(bank: RaschPCMBank, start_no: int, n: int) -> List[ItemRecord]:
    out: List[ItemRecord] = []
    used: List[str] = []
    n = max(0, min(int(n), len(bank.items)))
    for _ in range(n):
        nxt = bank.next_linear_item(used, start_no=start_no)
        if nxt is None:
            break
        out.append(nxt)
        used.append(nxt.item_id)
    return out




def onesample_ttest_against_constant(values: List[float], mu: float) -> dict:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size < 2:
        mean = float(np.mean(arr)) if arr.size else 0.0
        return {'n': int(arr.size), 'mean': mean, 'sd': 0.0, 't': 0.0, 'p': None, 'df': None, 'p_text': 'NA', 'df_text': 'NA'}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = max(sd / math.sqrt(arr.size), 1e-12)
    t = (mean - float(mu)) / se
    p = None
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), df=arr.size - 1))
    except Exception:
        p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return {'n': int(arr.size), 'mean': mean, 'sd': sd, 't': float(t), 'p': p, 'df': int(arr.size - 1), 'p_text': f'{p:.4g}' if p is not None else 'NA', 'df_text': str(arr.size - 1)}

def paired_ttest(a: List[float], b: List[float]) -> dict:
    n = min(len(a), len(b))
    if n <= 0:
        return {'n': 0, 'mean': 0.0, 'sd': 0.0, 't': 0.0, 'p': None, 'df': 0, 'p_text': 'NA', 'df_text': 'NA'}
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    return onesample_ttest_against_constant(diffs, 0.0)

def simulate_score_for_item(bank: RaschPCMBank, true_theta: float, item: ItemRecord, rng: random.Random) -> int:
    probs = np.asarray(bank.category_probabilities(true_theta, item), dtype=float).reshape(-1)
    idx = int(rng.choices(list(range(len(probs))), weights=[float(x) for x in probs], k=1)[0])
    scores = bank.score_values(item)
    return int(scores[idx])

def simulate_estimate_on_items(bank: RaschPCMBank, items: List[ItemRecord], true_theta: float, start_theta: float, rng: random.Random) -> dict:
    responses: List[Tuple[str, int]] = []
    for item in items:
        responses.append((item.item_id, simulate_score_for_item(bank, true_theta, item, rng)))
    theta, se, _ = bank.posterior(responses, start_theta=start_theta)
    return {'theta': float(theta), 'se': float(se), 'length': len(items), 'responses': responses}

def simulate_cat_administration(bank: RaschPCMBank, true_theta: float, start_theta: float, max_items: int, stop_se: float, rng: random.Random) -> dict:
    responses: List[Tuple[str, int]] = []
    theta, se, _ = bank.posterior([], start_theta=start_theta)
    while len(responses) < max_items:
        used = [item_id for item_id, _ in responses]
        item = bank.select_next_item(used, theta)
        if item is None:
            break
        score = simulate_score_for_item(bank, true_theta, item, rng)
        responses.append((item.item_id, score))
        theta, se, _ = bank.posterior(responses, start_theta=start_theta)
        if se <= stop_se:
            break
    return {'theta': float(theta), 'se': float(se), 'length': len(responses), 'responses': responses}

def simulate_random_full_answers(bank: RaschPCMBank, rng: random.Random) -> Dict[str, int]:
    answer_map: Dict[str, int] = {}
    for item in bank.items:
        categories = [int(x) for x in bank.score_values(item).tolist()]
        answer_map[item.item_id] = int(rng.choice(categories))
    return answer_map

def simulate_estimate_from_answer_map(bank: RaschPCMBank, items: List[ItemRecord], answer_map: Dict[str, int], start_theta: float) -> dict:
    responses: List[Tuple[str, int]] = []
    for item in items:
        score = int(answer_map.get(item.item_id, bank.score_values(item)[0]))
        responses.append((item.item_id, score))
    theta, se, _ = bank.posterior(responses, start_theta=start_theta)
    return {'theta': float(theta), 'se': float(se), 'length': len(items), 'responses': responses}

def simulate_cat_from_answer_map(bank: RaschPCMBank, answer_map: Dict[str, int], start_theta: float, max_items: int, stop_se: float, rng: Optional[random.Random] = None, first_item_random: bool = True) -> dict:
    responses: List[Tuple[str, int]] = []
    theta, se, _ = bank.posterior([], start_theta=start_theta)
    chosen_items: List[ItemRecord] = []
    while len(responses) < max_items:
        used = [item_id for item_id, _ in responses]
        if not used and first_item_random:
            remaining = [it for it in bank.items if it.item_id not in set(used)]
            if not remaining:
                break
            item = (rng or random).choice(remaining)
        else:
            item = bank.select_next_item(used, theta)
        if item is None:
            break
        score = int(answer_map.get(item.item_id, bank.score_values(item)[0]))
        responses.append((item.item_id, score))
        chosen_items.append(item)
        theta, se, _ = bank.posterior(responses, start_theta=start_theta)
        if se <= stop_se:
            break
    return {'theta': float(theta), 'se': float(se), 'length': len(responses), 'responses': responses, 'items': chosen_items}



def _normalize_sample_score_for_item(bank: RaschPCMBank, item: ItemRecord, raw_score) -> int:
    """Map a sample-data response to the nearest valid score for the current item."""
    valid_scores = [int(x) for x in bank.score_values(item).tolist()]
    if not valid_scores:
        return 0
    try:
        val = float(raw_score)
        if not np.isfinite(val):
            return int(valid_scores[0])
    except Exception:
        return int(valid_scores[0])
    return int(min(valid_scores, key=lambda x: abs(float(x) - val)))


def _load_sample_response_dataframe(bank: RaschPCMBank) -> Tuple[pd.DataFrame, str]:
    """Load sample response data from local files or replay_bundle.zip."""
    preferred = ["original_response.csv", "observed_response.csv", "simulated_response.csv"]
    base_dir = Path(__file__).parent
    for name in preferred:
        local = base_dir / name
        if local.exists():
            try:
                return _read_csv_bytes_robust(local.read_bytes(), csv_name=name), name
            except Exception:
                pass
    try:
        with zipfile.ZipFile(bank.bundle_path, "r") as zf:
            names = {n.lower(): n for n in zf.namelist()}
            for name in preferred:
                hit = names.get(name.lower())
                if hit:
                    try:
                        return _read_csv_bytes_robust(zf.read(hit), csv_name=hit), hit
                    except Exception:
                        pass
    except Exception:
        pass
    return pd.DataFrame(), "PCM parameter simulation fallback"


def load_sample_person_answer_candidates(bank: RaschPCMBank, start_theta: float, n_candidates: int = 20) -> List[dict]:
    """Select sample persons closest to the requested starting theta.

    The candidate pool is the N persons whose full-bank person MEASURE is closest
    to Starting theta. One candidate will then be selected randomly for the
    completed CAT sequential simulation.
    """
    df, source = _load_sample_response_dataframe(bank)
    if df is None or df.empty:
        return []

    item_by_no = {int(it.no): it for it in bank.items}
    col_by_no = {}
    for no, item in item_by_no.items():
        if str(no) in df.columns:
            col_by_no[no] = str(no)
        elif no in df.columns:
            col_by_no[no] = no

    if len(col_by_no) < 2:
        return []

    # Align person measures by row order when person_estimates.csv is available.
    measures = None
    try:
        if isinstance(bank.person_df, pd.DataFrame) and "MEASURE" in bank.person_df.columns and len(bank.person_df) >= len(df):
            measures = pd.to_numeric(bank.person_df["MEASURE"], errors="coerce").to_numpy(dtype=float)[:len(df)]
    except Exception:
        measures = None

    # If no person measure is available, estimate a rough ordering by total score.
    if measures is None or len(measures) < len(df):
        mat = []
        for no, col in sorted(col_by_no.items()):
            mat.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        arr = np.vstack(mat).T if mat else np.empty((0, 0))
        total = np.nanmean(arr, axis=1) if arr.size else np.zeros(len(df))
        total = np.where(np.isfinite(total), total, np.nanmean(total) if np.isfinite(np.nanmean(total)) else 0.0)
        sd = float(np.nanstd(total, ddof=1)) if len(total) > 1 else 1.0
        sd = sd if np.isfinite(sd) and sd > 1e-9 else 1.0
        measures = (total - float(np.nanmean(total))) / sd
        source = f"{source}; rough score-based theta ranking"

    candidates = []
    for ridx, row in df.iterrows():
        theta_val = float(measures[int(ridx)]) if int(ridx) < len(measures) and np.isfinite(measures[int(ridx)]) else float("nan")
        if not np.isfinite(theta_val):
            continue
        answer_map = {}
        for no, col in col_by_no.items():
            item = item_by_no[int(no)]
            answer_map[item.item_id] = _normalize_sample_score_for_item(bank, item, row.get(col))
        if len(answer_map) < 2:
            continue
        person_label = ""
        for label_col in ["KID", "kid", "name", "Name", "ID", "id"]:
            if label_col in df.columns:
                person_label = str(row.get(label_col, "")).strip()
                break
        if not person_label:
            person_label = f"row {int(ridx) + 1}"
        candidates.append({
            "row_index": int(ridx),
            "person_label": person_label,
            "theta": theta_val,
            "distance": abs(theta_val - float(start_theta)),
            "answer_map": answer_map,
            "source": source,
        })

    candidates.sort(key=lambda x: (x["distance"], x["row_index"]))
    n_candidates = max(1, min(int(n_candidates or 20), len(candidates)))
    return candidates[:n_candidates]


def _history_from_cat_run(bank: RaschPCMBank, cat_run: dict, start_theta: float, language: str) -> List[dict]:
    """Build the same response-history rows used by CAT mode, but without manual item-by-item input."""
    history: List[dict] = []
    responses = list(cat_run.get("responses", []))
    items = list(cat_run.get("items", []))
    for item, (_, score) in zip(items, responses):
        prev_responses = [(iid, sc) for iid, sc in responses[:len(history)]]
        prev_theta, _, _ = bank.posterior(prev_responses, start_theta=start_theta)
        exp_before = bank.expected_score(prev_theta, item)
        var_before = bank.variance_score(prev_theta, item)
        zstd = (int(score) - exp_before) / math.sqrt(max(var_before, 1e-9))
        theta_after, se_after, _ = bank.posterior(prev_responses + [(item.item_id, int(score))], start_theta=start_theta)
        link_href, _ = resolve_link_href(item.link)
        history.append({
            "item_id": item.item_id,
            "no": item.no,
            "delta": float(item.delta),
            "answer": bank.option_text(int(score), language, item),
            "score": int(score),
            "expected": float(exp_before),
            "theta_before": float(prev_theta),
            "theta": float(theta_after),
            "se": float(se_after),
            "zstd": float(zstd),
            "link_href": link_href,
        })
    return history


def _run_cat_sequential_simulation(language: str, start_theta: float, stop_se: float, start_item: int = 1, compare_n: int = 20, max_items: Optional[int] = None):
    """Run a completed CAT automatically for one randomly selected sample person.

    The N setting defines a candidate pool: the N sample persons whose full-bank
    MEASURE is closest to the user-specified Starting theta. One candidate is
    selected at random, then CAT is run automatically using the selected person's
    stored responses. This avoids the item-by-item manual CAT screen and goes
    directly to the completed CAT result page.
    """
    seed = random.randint(1, 10**9)
    rng = random.Random(seed)
    max_items = max(1, min(int(max_items or len(BANK.items)), len(BANK.items)))
    n_pool = max(1, int(compare_n or 20))

    candidates = load_sample_person_answer_candidates(BANK, start_theta=start_theta, n_candidates=n_pool)
    if candidates:
        selected = rng.choice(candidates)
        answer_map = {str(k): int(v) for k, v in selected["answer_map"].items()}
        selected_label = str(selected.get("person_label", "sample person"))
        selected_theta = float(selected.get("theta", float("nan")))
        selected_source = str(selected.get("source", "sample response data"))
        candidate_rank_note = f"randomly selected from the {len(candidates)} sample persons closest to Starting theta={start_theta:.3f}"
    else:
        # Fallback: generate N simulated persons centered on Starting theta and select one.
        generated = []
        for idx in range(n_pool):
            true_theta = float(rng.gauss(float(start_theta), max(0.25, float(getattr(BANK, "prior_sd", 1.0)) * 0.35)))
            amap = {}
            for item in BANK.items:
                amap[item.item_id] = simulate_score_for_item(BANK, true_theta, item, rng)
            generated.append({"person_label": f"generated person {idx + 1}", "theta": true_theta, "answer_map": amap})
        selected = rng.choice(generated)
        answer_map = {str(k): int(v) for k, v in selected["answer_map"].items()}
        selected_label = str(selected["person_label"])
        selected_theta = float(selected["theta"])
        selected_source = "PCM parameter simulation fallback"
        candidate_rank_note = f"randomly selected from {n_pool} generated persons centered on Starting theta={start_theta:.3f}"

    cat_run = simulate_cat_from_answer_map(
        BANK,
        answer_map,
        start_theta=start_theta,
        max_items=max_items,
        stop_se=float(stop_se),
        rng=rng,
        first_item_random=False,  # same CAT selection logic as manual CAT mode
    )
    responses = list(cat_run.get("responses", []))
    theta = float(cat_run.get("theta", start_theta))
    se = float(cat_run.get("se", getattr(BANK, "prior_sd", 1.0)))
    history = _history_from_cat_run(BANK, cat_run, start_theta=start_theta, language=language)

    if se <= float(stop_se):
        stop_reason = "target_se"
    elif len(responses) >= max_items:
        stop_reason = "max_items"
    else:
        stop_reason = "all_items"

    session["cat_state"] = {
        "mode": "cat_seq_sim",
        "max_items": max_items,
        "requested_max_items": max_items,
        "stop_se": float(stop_se),
        "start_theta": float(start_theta),
        "language": language,
        "start_item": start_item,
        "compare_n": n_pool,
        "simulation_seed": seed,
        "selected_person_label": selected_label,
        "selected_person_theta": selected_theta,
        "selected_person_source": selected_source,
        "simulation_note": f"{selected_label}; sample theta={selected_theta:.3f}; {candidate_rank_note}; source={selected_source}; seed={seed}",
        "responses": [list(x) for x in responses],
        "history": history,
        "theta": theta,
        "se": se,
        "stop_reason": stop_reason,
    }
    session.modified = True
    return redirect(url_for("show_result"))



def simulate_random_same_length(bank: RaschPCMBank, true_theta: float, start_theta: float, n_items: int, rng: random.Random) -> dict:
    n_items = max(1, min(int(n_items), len(bank.items)))
    picked = rng.sample(bank.items, n_items)
    return simulate_estimate_on_items(bank, picked, true_theta, start_theta, rng)

def summarize_mean_sd(values: List[float]) -> dict:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {'n': 0, 'mean': 0.0, 'sd': 0.0}
    return {'n': int(arr.size), 'mean': float(np.mean(arr)), 'sd': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}

def _box_stats(values: List[float]) -> Optional[dict]:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return None
    return {
        'min': float(np.min(arr)),
        'q1': float(np.percentile(arr, 25)),
        'med': float(np.percentile(arr, 50)),
        'q3': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
    }


def _draw_boxplot(parts: List[str], x: float, stats: dict, ymap, color: str, stroke: str, box_w: float = 70.0, label: str = '', show_points: Optional[List[float]] = None):
    q1, med, q3 = stats['q1'], stats['med'], stats['q3']
    vmin, vmax = stats['min'], stats['max']
    parts.append(f'<line x1="{x:.1f}" y1="{ymap(vmin):.1f}" x2="{x:.1f}" y2="{ymap(vmax):.1f}" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<line x1="{x-box_w/4:.1f}" y1="{ymap(vmin):.1f}" x2="{x+box_w/4:.1f}" y2="{ymap(vmin):.1f}" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<line x1="{x-box_w/4:.1f}" y1="{ymap(vmax):.1f}" x2="{x+box_w/4:.1f}" y2="{ymap(vmax):.1f}" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<rect x="{x-box_w/2:.1f}" y="{ymap(q3):.1f}" width="{box_w:.1f}" height="{max(1.2, ymap(q1)-ymap(q3)):.1f}" fill="{color}" fill-opacity="0.55" stroke="{stroke}"/>')
    parts.append(f'<line x1="{x-box_w/2:.1f}" y1="{ymap(med):.1f}" x2="{x+box_w/2:.1f}" y2="{ymap(med):.1f}" stroke="{stroke}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{x:.1f}" cy="{ymap(stats["mean"]):.1f}" r="4.2" fill="#2563eb" stroke="white" stroke-width="1"/>')
    if show_points:
        rng = random.Random(42 + int(x))
        for v in show_points:
            xx = x + rng.uniform(-box_w*0.34, box_w*0.34)
            parts.append(f'<circle cx="{xx:.1f}" cy="{ymap(float(v)):.1f}" r="2.9" fill="{stroke}" fill-opacity="0.55"/>')
    if label:
        parts.append(f'<text x="{x:.1f}" y="0" visibility="hidden">{html.escape(label)}</text>')


def make_length_efficiency_svg(full_length: int, cat_lengths: List[float]) -> str:
    width, height = 920, 320
    left, right, top, bottom = 58, 24, 32, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [float(full_length)] + [float(v) for v in cat_lengths] if cat_lengths else [float(full_length)]
    y_min = min(all_vals) - 1.0
    y_max = max(all_vals) + 1.0
    if y_max <= y_min:
        y_max = y_min + 2.0
    def ymap(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h
    x_nat = left + plot_w * 0.28
    x_cat = left + plot_w * 0.72
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    for yv in np.linspace(math.floor(y_min), math.ceil(y_max), 6):
        yy = ymap(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.0f}</text>')
    full_stats = {'min': float(full_length), 'q1': float(full_length), 'med': float(full_length), 'q3': float(full_length), 'max': float(full_length), 'mean': float(full_length)}
    _draw_boxplot(parts, x_nat, full_stats, ymap, '#f59e0b', '#b45309', box_w=72, show_points=[float(full_length)])
    if cat_lengths:
        _draw_boxplot(parts, x_cat, _box_stats(cat_lengths), ymap, '#fca5a5', '#b91c1c', box_w=78, show_points=cat_lengths)
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="18" text-anchor="middle" font-size="15" font-weight="700">a  Item length</text>')
    parts.append(f'<text x="{x_nat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">Full non-CAT</text>')
    parts.append(f'<text x="{x_cat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">CAT</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-size="12">Item length</text>')
    return _svg_wrap(width, height, ''.join(parts))


def make_theta_boxplot_svg(full_thetas: List[float], cat_thetas: List[float]) -> str:
    width, height = 920, 320
    left, right, top, bottom = 58, 24, 32, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [float(v) for v in (full_thetas + cat_thetas) if np.isfinite(float(v))]
    if not all_vals:
        all_vals = [0.0]
    y_min = min(all_vals) - 0.3
    y_max = max(all_vals) + 0.3
    if y_max <= y_min:
        y_max = y_min + 1.0
    def ymap(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h
    x_nat = left + plot_w * 0.28
    x_cat = left + plot_w * 0.72
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    for yv in np.linspace(y_min, y_max, 6):
        yy = ymap(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.1f}</text>')
    if full_thetas:
        _draw_boxplot(parts, x_nat, _box_stats(full_thetas), ymap, '#86efac', '#15803d', box_w=72, show_points=full_thetas)
    if cat_thetas:
        _draw_boxplot(parts, x_cat, _box_stats(cat_thetas), ymap, '#fecaca', '#b91c1c', box_w=78, show_points=cat_thetas)
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="18" text-anchor="middle" font-size="15" font-weight="700">b  Person measure</text>')
    parts.append(f'<text x="{x_nat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">Full non-CAT</text>')
    parts.append(f'<text x="{x_cat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">CAT</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-size="12">Person measure</text>')
    return _svg_wrap(width, height, ''.join(parts))


def build_cat_noncat_comparison(bank: RaschPCMBank, state: dict) -> Optional[dict]:
    mode = str(state.get('mode', 'cat'))
    if mode not in {'compare', 'compare_n'}:
        return None
    history = list(state.get('history', []))
    if not history:
        return None
    full_length = len(bank.items)
    cat_max_items = max(1, min(int(state.get('max_items', full_length) or full_length), full_length))
    if full_length <= 1:
        return None
    n_reps = max(1, int(state.get('compare_n', 20) or 20))
    start_theta = float(state.get('start_theta', bank.prior_mean))
    stop_se = float(state.get('stop_se', 0.32) or 0.32)

    actual_rows = []
    for idx, row in enumerate(history, start=1):
        actual_rows.append({
            'pos': idx,
            'item_id': row.get('item_id', ''),
            'no': row.get('no', ''),
            'delta': float(row.get('delta', 0.0)),
            'link_href': row.get('link_href', ''),
        })

    full_length_stats = {'min': float(full_length), 'q1': float(full_length), 'med': float(full_length), 'q3': float(full_length), 'max': float(full_length), 'mean': float(full_length)}

    if mode == 'compare':
        full_answer_map = {str(k): int(v) for k, v in dict(state.get('full_answer_map', {})).items()} if isinstance(state.get('full_answer_map', {}), dict) else {}
        if not full_answer_map:
            full_answer_map = simulate_random_full_answers(bank, random.Random(20260328))
        full_run = simulate_estimate_from_answer_map(bank, list(bank.items), full_answer_map, start_theta)
        full_theta = float(full_run['theta'])
        rng = random.Random(int(state.get('compare_seed', 20260329) or 20260329))
        cat_lengths: List[float] = []
        cat_thetas: List[float] = []
        for _ in range(n_reps):
            cat_run = simulate_cat_from_answer_map(bank, full_answer_map, start_theta, cat_max_items, stop_se, rng=rng, first_item_random=True)
            cat_lengths.append(float(cat_run['length']))
            cat_thetas.append(float(cat_run['theta']))
        full_thetas: List[float] = [full_theta]
        return {
            'title': 'CAT vs non-CAT(1) comparison (SE-based stopping)',
            'variant': 'compare',
            'n_reps': n_reps,
            'full_length': full_length,
            'cat_max_items': cat_max_items,
            'stop_se': float(stop_se),
            'full_summary': {'n': 1, 'mean_length': float(full_length), 'sd_length': 0.0},
            'full_theta_summary': summarize_mean_sd(full_thetas),
            'cat_length_summary': summarize_mean_sd(cat_lengths),
            'cat_theta_summary': summarize_mean_sd(cat_thetas),
            'theta_diff_summary': summarize_mean_sd([x - full_theta for x in cat_thetas]),
            'full_length_stats': full_length_stats,
            'cat_length_stats': _box_stats(cat_lengths) or full_length_stats,
            'full_theta_stats': _box_stats(full_thetas) or {'min': full_theta, 'q1': full_theta, 'med': full_theta, 'q3': full_theta, 'max': full_theta, 'mean': full_theta},
            'cat_theta_stats': _box_stats(cat_thetas) or {'min': 0.0, 'q1': 0.0, 'med': 0.0, 'q3': 0.0, 'max': 0.0, 'mean': 0.0},
            'length_ttest': onesample_ttest_against_constant(cat_lengths, full_length),
            'theta_diff_ttest': onesample_ttest_against_constant(cat_thetas, full_theta),
            'length_test_text': f'One-sample t-test of CAT item length against the full non-CAT length ({full_length} items):',
            'theta_test_text': f'One-sample t-test of CAT person measure against the fixed full non-CAT measure ({full_theta:.3f}):',
            'length_svg': make_length_efficiency_svg(full_length, cat_lengths),
            'theta_svg': make_theta_boxplot_svg(full_thetas, cat_thetas),
            'full_theta_value': full_theta,
            'actual_rows': actual_rows,
            'note_text': '{{ result.comparison.note_text }}',
        }

    # compare_n
    rng = random.Random(int(state.get('compare_seed', 20260329) or 20260329))
    full_thetas: List[float] = []
    cat_thetas: List[float] = []
    cat_lengths: List[float] = []
    for _ in range(n_reps):
        answer_map = simulate_random_full_answers(bank, rng)
        full_run = simulate_estimate_from_answer_map(bank, list(bank.items), answer_map, start_theta)
        cat_run = simulate_cat_from_answer_map(bank, answer_map, start_theta, cat_max_items, stop_se, rng=rng, first_item_random=True)
        full_thetas.append(float(full_run['theta']))
        cat_thetas.append(float(cat_run['theta']))
        cat_lengths.append(float(cat_run['length']))
    theta_diffs = [float(cat_thetas[i]) - float(full_thetas[i]) for i in range(min(len(cat_thetas), len(full_thetas)))]
    return {
        'title': 'CAT vs non-CAT(n) comparison (SE-based stopping)',
        'variant': 'compare_n',
        'n_reps': n_reps,
        'full_length': full_length,
        'cat_max_items': cat_max_items,
        'stop_se': float(stop_se),
        'full_summary': {'n': n_reps, 'mean_length': float(full_length), 'sd_length': 0.0},
        'full_theta_summary': summarize_mean_sd(full_thetas),
        'cat_length_summary': summarize_mean_sd(cat_lengths),
        'cat_theta_summary': summarize_mean_sd(cat_thetas),
        'theta_diff_summary': summarize_mean_sd(theta_diffs),
        'full_length_stats': full_length_stats,
        'cat_length_stats': _box_stats(cat_lengths) or full_length_stats,
        'full_theta_stats': _box_stats(full_thetas) or {'min': 0.0, 'q1': 0.0, 'med': 0.0, 'q3': 0.0, 'max': 0.0, 'mean': 0.0},
        'cat_theta_stats': _box_stats(cat_thetas) or {'min': 0.0, 'q1': 0.0, 'med': 0.0, 'q3': 0.0, 'max': 0.0, 'mean': 0.0},
        'length_ttest': onesample_ttest_against_constant(cat_lengths, full_length),
        'theta_diff_ttest': paired_ttest(cat_thetas, full_thetas),
        'length_test_text': f'One-sample t-test of CAT item length against the full non-CAT length ({full_length} items):',
        'theta_test_text': 'Paired t-test of person measure (CAT − full non-CAT):',
        'length_svg': make_length_efficiency_svg(full_length, cat_lengths),
        'theta_svg': make_theta_boxplot_svg(full_thetas, cat_thetas),
        'full_theta_value': float(np.mean(full_thetas)) if full_thetas else 0.0,
        'actual_rows': actual_rows,
        'note_text': 'Each of the n simulated persons receives one full non-CAT estimate and one CAT estimate; the CAT first item is randomized for each person.',
    }


def cronbach_alpha_from_matrix(score_matrix: np.ndarray) -> float:
    """Return Cronbach's alpha for an n_persons x n_items score matrix."""
    arr = np.asarray(score_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    # Drop items with all missing or zero variance to avoid unstable alpha.
    item_vars = np.nanvar(arr, axis=0, ddof=1)
    keep = np.isfinite(item_vars) & (item_vars > 1e-12)
    arr = arr[:, keep]
    if arr.shape[1] < 2:
        return 0.0
    item_vars = np.nanvar(arr, axis=0, ddof=1)
    total_scores = np.nansum(arr, axis=1)
    total_var = float(np.nanvar(total_scores, ddof=1))
    if not np.isfinite(total_var) or total_var <= 1e-12:
        return 0.0
    k = arr.shape[1]
    alpha = (k / (k - 1.0)) * (1.0 - float(np.nansum(item_vars)) / total_var)
    if not np.isfinite(alpha):
        return 0.0
    return float(max(0.0, min(alpha, 0.999)))


def _sample_response_matrix_from_bundle(bank: RaschRSMBank) -> Tuple[np.ndarray, str, int]:
    """Read sample response data and return only the 30 CAT item-score columns for Cronbach alpha.

    Non-item columns such as KID, Profile, name, class, outcome, or melanoma status are
    explicitly excluded. This prevents the criterion/outcome variable from being
    counted as a 31st CAT item.
    """
    preferred = ["original_response.csv", "observed_response.csv", "simulated_response.csv"]
    item_numbers = {str(int(it.no)) for it in getattr(bank, "items", []) if getattr(it, "no", None) is not None}
    non_item_keys = {"kid", "name", "profile", "class", "entry", "total_score", "count", "measure", "se", "outcome", "melanoma", "melanoma_status", "status"}

    def _matrix_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
        if df is None or df.empty:
            return None
        cols = [c for c in df.columns if str(c).strip() in item_numbers and re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") not in non_item_keys]
        if len(cols) < 2:
            # Fallback: numeric item-like columns, excluding known person/outcome fields.
            exclude = non_item_keys
            cols = []
            for c in df.columns:
                key = re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
                if key in exclude:
                    continue
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() >= max(3, int(0.5 * len(df))):
                    cols.append(c)
        if len(cols) < 2:
            return None
        mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Drop rows without any usable item scores.
        keep_rows = np.isfinite(mat).sum(axis=1) >= 2
        mat = mat[keep_rows, :]
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return None
        return mat

    # Prefer files placed next to raschcatpcm.py, then files inside replay_bundle.zip.
    base_dir = Path(__file__).parent
    for name in preferred:
        local = base_dir / name
        if local.exists():
            try:
                df = _read_csv_bytes_robust(local.read_bytes(), csv_name=name)
                mat = _matrix_from_df(df)
                if mat is not None:
                    return mat, name, int(mat.shape[1])
            except Exception:
                pass

    try:
        with zipfile.ZipFile(bank.bundle_path, "r") as zf:
            names = {n.lower(): n for n in zf.namelist()}
            for name in preferred:
                hit = names.get(name.lower())
                if not hit:
                    continue
                try:
                    df = _read_csv_bytes_robust(zf.read(hit), csv_name=hit)
                    mat = _matrix_from_df(df)
                    if mat is not None:
                        return mat, hit, int(mat.shape[1])
                except Exception:
                    pass
    except Exception:
        pass
    return np.empty((0, 0), dtype=float), "PCM parameter simulation fallback", 0


def _person_se_summary(bank: RaschRSMBank) -> dict:
    """Summarize full-bank person SE from person_estimates.csv, when available."""
    out = {
        'full_se_min': 0.0,
        'full_se_median': 0.0,
        'full_se_mean': 0.0,
        'full_se_max': 0.0,
    }
    try:
        if isinstance(bank.person_df, pd.DataFrame) and 'SE' in bank.person_df.columns:
            vals = pd.to_numeric(bank.person_df['SE'], errors='coerce').dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out.update({
                    'full_se_min': float(np.min(vals)),
                    'full_se_median': float(np.median(vals)),
                    'full_se_mean': float(np.mean(vals)),
                    'full_se_max': float(np.max(vals)),
                })
    except Exception:
        pass
    return out


def compute_alpha_based_stop_criterion(bank: RaschRSMBank, n_persons: int = 1000, seed: int = 20260623) -> dict:
    """Compute the CAT stopping SE from the provided sample data Cronbach alpha.

    Homepage formula:
        stop_se = theta_sd * sqrt(1 - alpha)

    The preferred alpha source is the provided sample response file in this order:
    original_response.csv, observed_response.csv, simulated_response.csv.
    If none is found, the function falls back to a PCM parameter simulation.
    """
    matrix, source, n_items_for_alpha = _sample_response_matrix_from_bundle(bank)
    if matrix.size:
        alpha = cronbach_alpha_from_matrix(matrix)
        n_persons_used = int(matrix.shape[0])
    else:
        # Last-resort fallback used only when no sample response file exists.
        rng = np.random.default_rng(int(seed))
        items = [it for it in bank.items if str(it.item_id).strip()]
        n_items_for_alpha = len(items)
        n_persons_used = max(50, int(n_persons))
        theta_sd_fallback = float(max(getattr(bank, 'prior_sd', 1.0), 0.5))
        thetas = rng.normal(float(getattr(bank, 'prior_mean', 0.0)), theta_sd_fallback, size=n_persons_used)
        matrix = np.zeros((n_persons_used, n_items_for_alpha), dtype=float)
        for p, theta in enumerate(thetas):
            for j, item in enumerate(items):
                probs = np.asarray(bank.category_probabilities(float(theta), item), dtype=float).reshape(-1)
                probs = np.where(np.isfinite(probs), probs, 0.0)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / max(len(probs), 1)
                scores = np.asarray(bank.score_values(item), dtype=float).reshape(-1)
                if scores.size != probs.size:
                    scores = scores[:probs.size] if scores.size >= probs.size else np.arange(probs.size, dtype=float)
                matrix[p, j] = scores[int(rng.choice(np.arange(probs.size), p=probs))]
        alpha = cronbach_alpha_from_matrix(matrix)

    theta_pool = np.asarray(getattr(bank, 'person_distribution', np.array([])), dtype=float)
    theta_pool = theta_pool[np.isfinite(theta_pool)]
    if theta_pool.size > 5 and float(np.std(theta_pool, ddof=1)) > 1e-9:
        theta_sd = float(np.std(theta_pool, ddof=1))
        theta_source = 'person_estimates.csv MEASURE SD'
    else:
        theta_sd = float(max(getattr(bank, 'prior_sd', 1.0), 0.5))
        theta_source = 'bank prior SD'

    stop_se = float(theta_sd * math.sqrt(max(0.0, 1.0 - alpha)))
    stop_se = float(max(0.05, min(stop_se, 1.50)))
    se_stats = _person_se_summary(bank)
    full_se_min = float(se_stats.get('full_se_min', 0.0) or 0.0)
    full_se_median = float(se_stats.get('full_se_median', 0.0) or 0.0)
    if full_se_min > 0 and stop_se < full_se_min:
        feasibility_note = (
            f"The current alpha-based criterion ({stop_se:.3f}) is smaller than the best available full-bank SE "
            f"({full_se_min:.3f}). It is therefore not reachable with the current 30-item bank. CAT will stop by the "
            "maximum-item rule, so making the SE criterion even smaller will not change CAT item length. "
            "Use a less strict SE value, such as 0.25-0.35, if you want the SE rule to shorten CAT length."
        )
    elif full_se_median > 0 and stop_se < full_se_median:
        feasibility_note = (
            f"The current alpha-based criterion ({stop_se:.3f}) is below the median full-bank SE "
            f"({full_se_median:.3f}). Many administrations may still stop at the maximum item count. "
            "When both the old and new SE criteria are below what the administered items can reach, CAT length remains identical. "
            "Use a less strict SE value, such as 0.25-0.35, or increase the maximum item limit if available."
        )
    elif full_se_median > 0:
        feasibility_note = (
            f"The criterion ({stop_se:.3f}) is at or above the median full-bank SE ({full_se_median:.3f}), "
            "so the stopping rule is more likely to affect CAT length. Smaller SE values will usually increase item length until the maximum-item limit is reached."
        )
    else:
        feasibility_note = "Full-bank sample SE was unavailable, so reachability could not be checked. If CAT length does not change after lowering SE, it is usually stopping at the maximum-item rule."

    return {
        'alpha': float(alpha),
        'theta_sd': float(theta_sd),
        'theta_source': theta_source,
        'stop_se': float(stop_se),
        'n_persons': int(n_persons_used),
        'n_items': int(n_items_for_alpha),
        'seed': int(seed),
        'source': source,
        'feasibility_note': feasibility_note,
        **se_stats,
    }



app = Flask(__name__)
app.secret_key = SECRET_KEY
BANK = RaschPCMBank(DEFAULT_BUNDLE)


def get_state() -> dict:
    return session.setdefault("cat_state", {})


def reset_state() -> None:
    session["cat_state"] = {}
    session.modified = True


def resolve_link_href(raw_link: str) -> Tuple[str, bool]:
    raw = (raw_link or "").strip()
    if not raw:
        return "", False
    if re.match(r"^[a-z]+://", raw, re.I):
        return raw, raw.lower().endswith(IMG_EXTS)
    local_path = BANK.local_asset_path(raw)
    if not local_path:
        return "", False
    try:
        rel = local_path.relative_to(BANK.extract_dir)
        rel_path = str(rel).replace("\\", "/")
        return url_for("bundle_asset", asset_path=rel_path), rel_path.lower().endswith(IMG_EXTS)
    except ValueError:
        pass
    try:
        rel = local_path.relative_to(Path(__file__).parent)
        rel_path = str(rel).replace("\\", "/")
        return url_for("bundle_asset", asset_path=rel_path), rel_path.lower().endswith(IMG_EXTS)
    except ValueError:
        # Fallback for absolute local files found by the asset resolver.
        return url_for("bundle_asset", asset_path="__abs__/" + str(local_path)), local_path.name.lower().endswith(IMG_EXTS)


@app.get("/bundle_asset/<path:asset_path>")
def bundle_asset(asset_path: str):
    asset_path = asset_path.replace("\\", "/")
    if asset_path.startswith("__abs__/"):
        fp = Path(asset_path[len("__abs__/"):])
        if fp.exists() and fp.is_file():
            return send_from_directory(str(fp.parent), fp.name)
        abort(404)
    for root in (BANK.extract_dir, Path(__file__).parent):
        fp = root / asset_path
        if fp.exists() and fp.is_file():
            return send_from_directory(str(root), asset_path)
    abort(404)


@app.get("/debug_assets")
def debug_assets():
    rows = []
    pic_root = Path(__file__).parent / "pic"
    local_pic_count = len(list(pic_root.glob("*.png"))) if pic_root.exists() else 0
    bundle_pic_count = len(list((BANK.extract_dir / "pic").glob("*.png"))) if (BANK.extract_dir / "pic").exists() else 0
    for item in BANK.items:
        link_href, is_image = resolve_link_href(item.link)
        rows.append({
            "no": item.no,
            "item_id": item.item_id,
            "link": item.link,
            "resolved_url": link_href,
            "is_image": bool(is_image),
            "exists": bool(link_href),
        })
    html_rows = "".join(
        f"<tr><td>{r['no']}</td><td>{html.escape(str(r['item_id']))}</td><td>{html.escape(str(r['link']))}</td><td>{'YES' if r['exists'] else 'NO'}</td><td>{html.escape(str(r['resolved_url']))}</td></tr>"
        for r in rows
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Asset debug</title>
    <style>body{{font-family:Arial,sans-serif;margin:24px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px}}</style></head><body>
    <h1>Asset debug</h1>
    <p><strong>response_category source:</strong> {html.escape(str(getattr(BANK, 'response_category_source', 'unknown')))}</p>
    <p><strong>root pic PNG count:</strong> {local_pic_count} | <strong>bundle-extracted pic PNG count:</strong> {bundle_pic_count}</p>
    <table><thead><tr><th>No</th><th>Item</th><th>link column</th><th>resolved?</th><th>resolved URL</th></tr></thead><tbody>{html_rows}</tbody></table>
    </body></html>"""


@app.get("/")
def index():
    cat_criterion = compute_alpha_based_stop_criterion(BANK)
    summary = {
        "model": BANK.model,
        "n_items": len(BANK.items),
        "prior_mean": BANK.prior_mean,
        "prior_sd": BANK.prior_sd,
        "score_range": f"{BANK.min_score} to {BANK.max_score}",
        "step_source": f"{sum(1 for it in BANK.items if it.raw_thresholds)} item-specific PCM step sets",
    }
    wright_items = build_dashboard_rows(BANK)
    ref_person_se = float(pd.to_numeric(BANK.person_df.get('SE'), errors='coerce').dropna().median()) if isinstance(BANK.person_df, pd.DataFrame) and 'SE' in BANK.person_df.columns and not pd.to_numeric(BANK.person_df.get('SE'), errors='coerce').dropna().empty else 0.30
    try:
        wright_svg = make_home_wrightmap_svg(BANK.person_distribution, pd.DataFrame(wright_items), ref_theta=float(BANK.prior_mean), ref_se=ref_person_se)
    except Exception as _e_home_wm:
        wright_svg = f"<div class='muted'>Wright Map unavailable: {html.escape(repr(_e_home_wm))}</div>"
    try:
        kidmap_home_svg = make_home_kidmap_svg(BANK.person_distribution, pd.DataFrame(wright_items), zscore_df=BANK.zscore_df, person_df=BANK.person_df, ref_theta=float(BANK.prior_mean), ref_se=ref_person_se)
    except Exception as _e_home_km:
        kidmap_home_svg = f"<div class='muted'>KIDMAP unavailable: {html.escape(repr(_e_home_km))}</div>"
        cat_criterion = compute_cat_stop_criterion(BANK)
    return render_template_string(HOME_TMPL, title=BANK.cat_title, bundle_name=BANK.bundle_path.name, summary=summary, options=BANK.category_options, thresholds=[f"{x:.4f}" for x in BANK.step_thresholds], wright_svg=wright_svg, kidmap_home_svg=kidmap_home_svg, wright_items=wright_items, cat_criterion=cat_criterion)


@app.post("/start")
def start_test():
    requested_max_items = int(request.form.get("max_items", 20))
    cat_criterion = compute_alpha_based_stop_criterion(BANK)
    try:
        stop_se = float(request.form.get("stop_se", cat_criterion.get("stop_se", 0.32)))
    except Exception:
        stop_se = float(cat_criterion.get("stop_se", 0.32))
    start_theta = float(request.form.get("start_theta", BANK.prior_mean))
    language = str(request.form.get("language", "en")).strip().lower()
    start_item = int(request.form.get("start_item", 1))
    theta_range = float(request.form.get("theta_range", 1.0))
    compare_n = int(request.form.get("compare_n", 20))
    mode = str(request.form.get("mode", "cat")).strip().lower()
    if language not in {"zh", "en"}:
        language = "en"
    theta, se, _ = BANK.posterior([], start_theta=start_theta)
    requested_max_items = max(1, min(requested_max_items, len(BANK.items)))
    start_item = max(1, min(start_item, len(BANK.items)))

    if mode == "cat_seq_sim":
        return _run_cat_sequential_simulation(language=language, start_theta=start_theta, stop_se=stop_se, start_item=start_item, compare_n=compare_n, max_items=requested_max_items)
    if mode == "compare":
        return _run_compare(language=language, start_theta=start_theta, stop_se=stop_se, start_item=start_item, compare_n=compare_n, max_items=requested_max_items)
    if mode == "compare_n":
        return _run_compare_n(language=language, start_theta=start_theta, stop_se=stop_se, start_item=start_item, compare_n=compare_n, max_items=requested_max_items)
    if mode == "voice":
        voice_items = BANK.sample_voice_items(center_theta=start_theta, theta_range=theta_range, n_items=requested_max_items)
        session["cat_state"] = {
            "mode": "voice",
            "language": language,
            "start_theta": start_theta,
            "theta_range": theta_range,
            "requested_max_items": requested_max_items,
            "start_item": start_item,
            "voice_item_ids": [it.item_id for it in voice_items],
        }
        session.modified = True
        return redirect(url_for("show_voice"))

    if mode == "linear":
        first_item = BANK.next_linear_item([], start_no=start_item)
        max_items = len(BANK.items)
        stop_se = 0.0
    else:
        first_item = BANK.select_next_item([], theta)
        max_items = requested_max_items
    if first_item is None:
        return redirect(url_for("index"))
    session["cat_state"] = {
        "mode": mode if mode in {"cat", "linear"} else "cat",
        "max_items": max_items,
        "requested_max_items": requested_max_items,
        "stop_se": stop_se,
        "start_theta": start_theta,
        "start_item": start_item,
        "language": language,
        "responses": [],
        "history": [],
        "current_item": first_item.item_id,
        "theta": theta,
        "se": se,
        "stop_reason": "",
    }
    session.modified = True
    return redirect(url_for("show_item"))


def _run_compare(language: str, start_theta: float, stop_se: float, start_item: int = 1, compare_n: int = 20, max_items: Optional[int] = None):
    seed = random.randint(1, 10**9)
    rng = random.Random(seed)
    history: List[dict] = []
    max_items = max(1, min(int(max_items or len(BANK.items)), len(BANK.items)))
    full_answer_map = simulate_random_full_answers(BANK, rng)
    cat_run = simulate_cat_from_answer_map(BANK, full_answer_map, start_theta, max_items, float(stop_se), rng=rng, first_item_random=True)
    responses = list(cat_run['responses'])
    theta = float(cat_run['theta'])
    se = float(cat_run['se'])
    for item, (_, score) in zip(cat_run.get('items', []), responses):
        prev_responses = [(iid, sc) for iid, sc in responses[:len(history)]]
        prev_theta, _, _ = BANK.posterior(prev_responses, start_theta=start_theta)
        exp_before = BANK.expected_score(prev_theta, item)
        var_before = BANK.variance_score(prev_theta, item)
        zstd = (score - exp_before) / math.sqrt(max(var_before, 1e-9))
        theta_after, se_after, _ = BANK.posterior(prev_responses + [(item.item_id, score)], start_theta=start_theta)
        link_href, _ = resolve_link_href(item.link)
        history.append({
            "item_id": item.item_id,
            "no": item.no,
            "delta": float(item.delta),
            "answer": BANK.option_text(score, language, item),
            "score": int(score),
            "expected": float(exp_before),
            "theta": float(theta_after),
            "se": float(se_after),
            "zstd": float(zstd),
            "link_href": link_href,
        })
    session["cat_state"] = {
        "mode": "compare",
        "max_items": max_items,
        "requested_max_items": max_items,
        "stop_se": float(stop_se),
        "start_theta": start_theta,
        "language": language,
        "start_item": start_item,
        "compare_n": max(1, int(compare_n or 20)),
        "compare_seed": seed if "seed" in locals() else random.randint(1, 10**9),
        "full_answer_map": {str(k): int(v) for k, v in full_answer_map.items()},
        "responses": [list(x) for x in responses],
        "history": history,
        "theta": theta,
        "se": se,
        "stop_reason": "posterior_se" if se <= float(stop_se) else "max_items",
    }
    session.modified = True
    return redirect(url_for("show_result"))


def _run_compare_n(language: str, start_theta: float, stop_se: float, start_item: int = 1, compare_n: int = 20, max_items: Optional[int] = None):
    seed = random.randint(1, 10**9)
    rng = random.Random(seed)
    history: List[dict] = []
    max_items = max(1, min(int(max_items or len(BANK.items)), len(BANK.items)))
    example_answer_map = simulate_random_full_answers(BANK, rng)
    cat_run = simulate_cat_from_answer_map(BANK, example_answer_map, start_theta, max_items, float(stop_se), rng=rng, first_item_random=True)
    responses = list(cat_run['responses'])
    theta = float(cat_run['theta'])
    se = float(cat_run['se'])
    for item, (_, score) in zip(cat_run.get('items', []), responses):
        prev_responses = [(iid, sc) for iid, sc in responses[:len(history)]]
        prev_theta, _, _ = BANK.posterior(prev_responses, start_theta=start_theta)
        exp_before = BANK.expected_score(prev_theta, item)
        var_before = BANK.variance_score(prev_theta, item)
        zstd = (score - exp_before) / math.sqrt(max(var_before, 1e-9))
        theta_after, se_after, _ = BANK.posterior(prev_responses + [(item.item_id, score)], start_theta=start_theta)
        link_href, _ = resolve_link_href(item.link)
        history.append({
            "item_id": item.item_id,
            "no": item.no,
            "delta": float(item.delta),
            "answer": BANK.option_text(score, language, item),
            "score": int(score),
            "expected": float(exp_before),
            "theta": float(theta_after),
            "se": float(se_after),
            "zstd": float(zstd),
            "link_href": link_href,
        })
    session["cat_state"] = {
        "mode": "compare_n",
        "max_items": max_items,
        "requested_max_items": max_items,
        "stop_se": float(stop_se),
        "start_theta": start_theta,
        "language": language,
        "start_item": start_item,
        "compare_n": max(2, int(compare_n or 20)),
        "compare_seed": seed,
        "responses": [list(x) for x in responses],
        "history": history,
        "theta": theta,
        "se": se,
        "stop_reason": "posterior_se" if se <= float(stop_se) else "max_items",
    }
    session.modified = True
    return redirect(url_for("show_result"))


@app.get("/item")
def show_item():
    state = get_state()
    if not state or "current_item" not in state:
        return redirect(url_for("index"))
    item = BANK.item_lookup[state["current_item"]]
    language = state.get("language", "en")
    link_href, is_image = resolve_link_href(item.link)
    progress = {
        "mode_name": mode_name(state.get("mode", "cat")),
        "answered": len(state.get("responses", [])),
        "max_items": state.get("max_items", len(BANK.items)),
        "theta": float(state.get("theta", BANK.prior_mean)),
        "se": float(state.get("se", BANK.prior_sd)),
        "info_value": float(BANK.information(float(state.get("theta", BANK.prior_mean)), item)),
        "info_line": "Selected by maximum polytomous information (score variance across categories) from the remaining items." if state.get("mode") == "cat" else "Presented sequentially from the selected starting item in fixed order.",
    }
    item_view = {
        "item_id": item.item_id,
        "no": item.no,
        "stem": item.stem_for(language),
        "stem_zh": item.stem_zh,
        "stem_en": item.stem_en,
        "auto_text": item.stem_for(language),
        "auto_lang": "zh-TW" if language == "zh" else "en-US",
        "options": BANK.response_options(language, item),
        "link_href": link_href,
        "is_image_link": is_image,
    }
    return render_template_string(ITEM_TMPL, title=BANK.cat_title, item=item_view, progress=progress)


@app.post("/answer")
def submit_answer():
    state = get_state()
    if not state or "current_item" not in state:
        return redirect(url_for("index"))
    item = BANK.item_lookup[state["current_item"]]
    language = state.get("language", "en")
    item_scores = [int(x) for x in BANK.score_values(item).tolist()]
    default_score = item_scores[0] if item_scores else 0
    score = int(request.form.get("score", str(default_score)))
    if item_scores:
        if score not in item_scores:
            score = min(item_scores, key=lambda x: abs(x - score))
    theta_before = float(state.get("theta", state.get("start_theta", BANK.prior_mean)))
    exp_before = BANK.expected_score(theta_before, item)
    var_before = BANK.variance_score(theta_before, item)
    zstd = (score - exp_before) / math.sqrt(max(var_before, 1e-9))

    responses = [tuple(x) for x in state.get("responses", [])]
    responses.append((item.item_id, score))
    theta, se, _ = BANK.posterior(responses, start_theta=state.get("start_theta", BANK.prior_mean))

    history = list(state.get("history", []))
    link_href, _ = resolve_link_href(item.link)
    history.append({
        "item_id": item.item_id,
        "no": item.no,
        "delta": float(item.delta),
        "answer": BANK.option_text(score, language, item),
        "score": int(score),
        "expected": float(exp_before),
        "theta_before": float(theta_before),
        "theta": float(theta),
        "se": float(se),
        "zstd": float(zstd),
        "link_href": link_href,
    })

    mode = state.get("mode", "cat")
    stop_reason = ""
    if mode == "linear":
        if len(responses) >= len(BANK.items):
            stop_reason = "all_items"
    else:
        if len(responses) >= int(state.get("max_items", len(BANK.items))):
            stop_reason = "max_items"
        elif se <= float(state.get("stop_se", 0.32)):
            stop_reason = "target_se"

    state.update({
        "responses": [list(x) for x in responses],
        "history": history,
        "theta": float(theta),
        "se": float(se),
        "stop_reason": stop_reason,
    })
    if stop_reason:
        session["cat_state"] = state
        session.modified = True
        return redirect(url_for("show_result"))
    used_ids = [i for i, _ in responses]
    next_item = BANK.next_linear_item(used_ids, start_no=state.get("start_item", 1)) if mode == "linear" else BANK.select_next_item(used_ids, theta)
    if next_item is None:
        state["stop_reason"] = "all_items"
        session["cat_state"] = state
        session.modified = True
        return redirect(url_for("show_result"))
    state["current_item"] = next_item.item_id
    session["cat_state"] = state
    session.modified = True
    return redirect(url_for("show_item"))


@app.get('/voice')
def show_voice():
    state = get_state()
    if not state or state.get('mode') != 'voice':
        return redirect(url_for('index'))
    voice_ids = [str(x) for x in state.get('voice_item_ids', [])]
    items_payload = []
    for idx, item_id in enumerate(voice_ids, start=1):
        item = BANK.item_lookup.get(item_id)
        if not item:
            continue
        link_href, is_image = resolve_link_href(item.link)
        fit_row = next((r for r in build_dashboard_rows(BANK) if r.get('item_id') == item_id), {})
        items_payload.append({
            'item_id': item_id,
            'no': item.no,
            'stem': item.stem_for(state.get('language', 'en')),
            'stem_zh': item.stem_zh,
            'stem_en': item.stem_en,
            'delta': float(item.delta),
            'measure': float(fit_row.get('measure', item.delta)),
            'infit': float(fit_row.get('infit', float('nan'))),
            'outfit': float(fit_row.get('outfit', float('nan'))),
            'se': float(fit_row.get('se', 0.12)),
            'options': [{'label': op.label, 'text': op.text, 'score': op.score} for op in BANK.response_options(state.get('language', 'en'), item)],
            'link_href': link_href,
            'is_image_link': bool(is_image),
        })
    speech_config = {
        'tts_base_url': url_for('voice_tts'),
        'server_tts_enabled': bool(gTTS is not None),
        'language_code': state.get('language', 'en'),
        'lang': 'zh-TW' if state.get('language', 'en') == 'zh' else 'en-US',
        'prefer_browser_cycle': False,
    }
    return render_template_string(VOICE_TMPL, title=BANK.cat_title, requested_max_items=int(state.get('requested_max_items', len(items_payload))), item_count=len(items_payload), start_theta=float(state.get('start_theta', BANK.prior_mean)), theta_range=float(state.get('theta_range', 1.0)), items=items_payload, speech_config=speech_config)


@app.get('/voice_tts')
def voice_tts():
    item_id = str(request.args.get('item_id', '')).strip()
    language = str(request.args.get('language', 'en')).strip().lower()
    item = BANK.item_lookup.get(item_id)
    if not item:
        abort(404)
    text_map = voice_part_texts(item, BANK, language, 1)
    audio_path = synthesize_tts_path(text_map, language)
    if not audio_path or not audio_path.exists():
        abort(404)
    return send_file(audio_path, mimetype='audio/mpeg', as_attachment=False, download_name=audio_path.name)


@app.get('/result')
def show_result():
    state = get_state()
    if not state or not state.get('history'):
        return redirect(url_for('index'))
    responses = [tuple(x) for x in state.get('responses', [])]
    final_theta = float(state.get('theta', BANK.prior_mean))
    final_se = float(state.get('se', BANK.prior_sd))
    infit_mnsq, outfit_mnsq = compute_person_fit(BANK, responses, final_theta)
    item_se_map = {}
    if not BANK.item_fit_df.empty:
        item_col = _find_column(BANK.item_fit_df, ['ITEM', 'item'])
        se_col = _find_column(BANK.item_fit_df, ['SE', 'MODEL_SE', 'MODEL SE', 'S.E.', 'S.E'])
        if item_col and se_col:
            item_se_map = dict(zip(BANK.item_fit_df[item_col].astype(str), pd.to_numeric(BANK.item_fit_df[se_col], errors='coerce').fillna(0.12)))
    residual_rows = []
    for row in state['history']:
        residual_rows.append({'item_id': row['item_id'], 'delta': float(row['delta']), 'zscore': float(row['zstd']), 'item_se': float(item_se_map.get(row['item_id'], 0.12))})
    _ref_item = None
    try:
        if state.get("history"):
            _last_id = str(state["history"][-1].get("item_id", "") or "")
            _ref_item = BANK.item_lookup.get(_last_id) or None
    except Exception:
        _ref_item = None
    cpc_svg, cpc_pred_label = make_cpc_svg(BANK, final_theta, ref_item=_ref_item)
    result = {
        'theta': final_theta,
        'se': final_se,
        'percentile': BANK.percentile(final_theta),
        'n_answered': len(responses),
        'stop_reason': str(state.get('stop_reason', 'finished') or 'finished'),
        'history': list(state['history']),
        'has_links': any(bool((row.get('link_href') or '').strip()) for row in state['history']),
        'mode_name': mode_name(state.get('mode', 'cat')),
        'infit_mnsq': infit_mnsq,
        'outfit_mnsq': outfit_mnsq,
        'trend_svg': make_trend_svg(list(state['history'])),
        'kidmap_svg': make_combined_kidmap_svg(BANK.person_distribution, final_theta, final_se, residual_rows, infit_mnsq, outfit_mnsq),
        'cpc_svg': cpc_svg,
        'cpc_pred_label': cpc_pred_label,
        'comparison': build_cat_noncat_comparison(BANK, state),
        'risk_classification': skin_cancer_risk_classification(float(state.get('theta', BANK.prior_mean))),
        'simulation_note': str(state.get('simulation_note', '') or ''),
    }
    return render_template_string(RESULT_TMPL, title=BANK.cat_title, result=result)


@app.get('/reset')
def reset():
    reset_state()
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True)
