"""
stream_detect.py — Веб-визуализация детекции людей с HLS-потока или папки с видео

Установка зависимостей:
    pip install ultralytics opencv-python flask

Запуск:
    python stream_detect.py --url "https://video2.interra.ru/glaz.naroda.125.../index.m3u8"
    
Открыть браузер: http://localhost:5000
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# HTML-шаблон
# ──────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Детекция людей — Live</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:      #0a0c0f;
    --surface: #111318;
    --border:  #1e2530;
    --accent:  #00ff88;
    --accent2: #00b8ff;
    --warn:    #ff4455;
    --amber:   #ffaa00;
    --text:    #c8d0dc;
    --dim:     #5a6375;
    --radius:  6px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-wrap: wrap;
  }
  .logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 3px;
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    color: var(--accent);
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
    letter-spacing: 0.08em;
  }
  #model-badge {
    position: relative;
    overflow: hidden;
    background: color-mix(in srgb, var(--amber) 12%, transparent);
    color: var(--amber);
    border-color: color-mix(in srgb, var(--amber) 30%, transparent);
    transition: color .3s, border-color .3s, background .3s;
  }
  #model-badge.loading::after {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,170,0,.35) 50%, transparent 100%);
    animation: shimmer .9s linear infinite;
  }
  @keyframes shimmer { from{transform:translateX(-100%)} to{transform:translateX(100%)} }

  .spacer { flex: 1; }
  #status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--dim); transition: background .4s;
  }
  #status-dot.live    { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 2s ease-in-out infinite; }
  #status-dot.error   { background: var(--warn); }
  #status-dot.loading { background: var(--amber); animation: pulse .6s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  #status-text { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dim); }

  /* ── Main layout ── */
  main {
    display: grid;
    grid-template-columns: 1fr 292px;
    flex: 1;
    overflow: hidden;
  }

  /* ── Video pane ── */
  .video-pane {
    position: relative; background: #000;
    display: flex; align-items: center; justify-content: center; overflow: hidden;
  }
  #stream {
    max-width: 100%; max-height: calc(100vh - 53px);
    object-fit: contain; display: block;
  }
  .overlay-src {
    position: absolute; top: 12px; left: 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: rgba(200,208,220,.55); background: rgba(0,0,0,.55);
    padding: 4px 8px; border-radius: 3px; letter-spacing: .06em;
    max-width: 60%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  #skip-overlay {
    position: absolute; bottom: 12px; right: 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: rgba(255,170,0,.8); background: rgba(0,0,0,.55);
    padding: 4px 8px; border-radius: 3px; letter-spacing: .06em;
    display: none;
  }

  /* ── Sidebar ── */
  aside {
    border-left: 1px solid var(--border);
    background: var(--surface);
    display: flex; flex-direction: column;
    overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
  .section { padding: 14px 16px; border-bottom: 1px solid var(--border); }
  .section-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: .12em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 11px;
  }

  /* ── Stats ── */
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
  .stat-card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 10px 12px;
  }
  .stat-card .val {
    font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600;
    color: var(--accent); line-height: 1; transition: color .3s;
  }
  .stat-card .lbl { font-size: 10px; color: var(--dim); margin-top: 3px; letter-spacing: .06em; }
  .stat-card.wide { grid-column: span 2; }
  #sparkline-wrap {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 8px;
  }
  canvas#sparkline { width: 100%; height: 56px; display: block; }
  .spark-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--dim); margin-top: 3px; text-align: right; }

  /* ── Model selector ── */
  .model-btns { display: flex; gap: 5px; }
  .model-btn {
    flex: 1; padding: 7px 4px; border-radius: var(--radius);
    border: 1px solid var(--border); background: var(--bg);
    color: var(--dim); font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 600; cursor: pointer;
    transition: all .2s; text-align: center;
  }
  .model-btn:hover { border-color: var(--accent2); color: var(--accent2); }
  .model-btn.active {
    border-color: var(--accent); color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    box-shadow: 0 0 8px color-mix(in srgb, var(--accent) 25%, transparent);
  }
  .model-btn.loading-btn {
    border-color: var(--amber); color: var(--amber);
    background: color-mix(in srgb, var(--amber) 10%, transparent);
    animation: pulse .6s ease-in-out infinite; pointer-events: none;
  }
  .model-info {
    margin-top: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: var(--dim); padding: 6px 8px; background: var(--bg);
    border: 1px solid var(--border); border-radius: var(--radius);
    display: flex; justify-content: space-between; align-items: center;
  }
  .model-info span { color: var(--text); }
  .model-sizes-hint {
    margin-top: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    color: var(--dim); line-height: 1.6;
  }

  /* ── Sliders ── */
  label {
    font-size: 11px; color: var(--dim); display: block;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: .05em;
  }
  input[type=range] {
    -webkit-appearance: none; width: 100%; height: 4px;
    border-radius: 2px; background: var(--border); outline: none; margin: 6px 0;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: var(--accent);
    cursor: pointer; border: 2px solid var(--bg); transition: transform .15s;
  }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }
  input[type=range].amber-thumb::-webkit-slider-thumb { background: var(--amber); }
  .range-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
  .range-val { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); }
  .range-val.amber { color: var(--amber); }

  .fpm-note {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dim);
    margin-top: 4px; padding: 5px 7px; background: var(--bg);
    border: 1px solid var(--border); border-radius: var(--radius);
  }
  .fpm-note b { color: var(--amber); }

  /* Source */
  input[type=text] {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 7px 10px; color: var(--text);
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; outline: none; transition: border-color .2s;
  }
  input[type=text]:focus { border-color: var(--accent); }
  button {
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    border-radius: var(--radius); padding: 8px 14px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .05em;
    cursor: pointer; transition: background .2s, transform .1s; white-space: nowrap;
  }
  button:hover { background: color-mix(in srgb, var(--accent) 25%, transparent); }
  button:active { transform: scale(.97); }
  button.full { width: 100%; }
  select {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 7px 10px; color: var(--text);
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; outline: none;
    margin-bottom: 8px; cursor: pointer;
  }
  select:focus { border-color: var(--accent); }

  /* Log */
  #det-log { list-style: none; max-height: 120px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
  #det-log li {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dim);
    padding: 3px 0; border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent);
    display: flex; gap: 8px;
  }
  #det-log li .ts { color: var(--dim); }
  #det-log li .cnt { color: var(--accent); font-weight: 600; min-width: 24px; text-align: right; }

  .footer-note {
    padding: 10px 16px; font-size: 10px; color: var(--dim);
    font-family: 'IBM Plex Mono', monospace; letter-spacing: .04em; margin-top: auto;
  }
  .gap { height: 10px; }
</style>
</head>
<body>

<header>
  <span class="logo">▣ PERSON DETECT</span>
  <span class="badge">YOLOv8</span>
  <span class="badge" id="model-badge">MODEL: {{model}}</span>
  <div class="spacer"></div>
  <div id="status-dot"></div>
  <span id="status-text">ИНИЦИАЛИЗАЦИЯ...</span>
</header>

<main>
  <div class="video-pane">
    <img id="stream" src="/video_feed" alt="stream"
         onload="onFrameLoad()" onerror="onFrameError()" />
    <div class="overlay-src" id="src-overlay">{{source_label}}</div>
    <div id="skip-overlay"></div>
  </div>

  <aside>

    <!-- Stats -->
    <div class="section">
      <div class="section-title">Статистика</div>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="val" id="stat-persons">0</div>
          <div class="lbl">ЛЮДЕЙ</div>
        </div>
        <div class="stat-card">
          <div class="val" id="stat-fps">—</div>
          <div class="lbl">STREAM FPS</div>
        </div>
        <div class="stat-card">
          <div class="val" id="stat-ms">—</div>
          <div class="lbl">INFER ms</div>
        </div>
        <div class="stat-card">
          <div class="val" id="stat-dfps">—</div>
          <div class="lbl">DETECT FPS</div>
        </div>
        <div class="stat-card wide" id="sparkline-wrap">
          <canvas id="sparkline"></canvas>
          <div class="spark-label">кол-во людей / время</div>
        </div>
      </div>
    </div>

    <!-- Model selector -->
    <div class="section">
      <div class="section-title">Модель YOLOv8</div>
      <div class="model-btns">
        {% for sz in ['n','s','m','l','x'] %}
        <div class="model-btn {% if sz == model_size %}active{% endif %}"
             id="mbtn-{{sz}}" onclick="switchModel('{{sz}}')">{{sz}}</div>
        {% endfor %}
      </div>
      <div class="model-info">
        <span style="color:var(--dim)">активна:</span>
        <span id="active-model-label">yolov8{{model_size}}.pt</span>
      </div>
      <div class="model-sizes-hint">
        n=nano · s=small · m=medium · l=large · x=xlarge<br>
        точность ↑ &nbsp;⟷&nbsp; скорость ↓
      </div>
    </div>

    <!-- Params -->
    <div class="section">
      <div class="section-title">Параметры детекции</div>

      <div class="range-row">
        <label>Порог уверенности</label>
        <span class="range-val" id="conf-val">{{conf}}</span>
      </div>
      <input type="range" id="conf-slider" min="0.1" max="0.9" step="0.05"
             value="{{conf}}" oninput="onConfChange(this.value)" onchange="applyConf()" />

      <div class="gap"></div>

      <div class="range-row">
        <label>Размер входа модели</label>
        <span class="range-val" id="imgsz-val">{{imgsz}}</span>
      </div>
      <input type="range" id="imgsz-slider" min="320" max="1920" step="320"
             value="{{imgsz}}" oninput="onImgszChange(this.value)" onchange="applyImgsz()" />

      <div class="gap"></div>

      <div class="range-row">
        <label>Лимит детекций / мин (FPM)</label>
        <span class="range-val amber" id="fpm-val">{{fpm}}</span>
      </div>
      <input type="range" class="amber-thumb" id="fpm-slider"
             min="1" max="300" step="1" value="{{fpm}}"
             oninput="onFpmChange(this.value)" onchange="applyFpm()" />
      <div class="fpm-note">
        анализируется каждый <b id="fpm-skip-note">—</b> кадр(ов) стрима
      </div>
    </div>

    <!-- Source -->
    <div class="section">
      <div class="section-title">Источник</div>
      {% if videos %}
      <select id="video-select">
        <option value="">— выбери файл —</option>
        {% for v in videos %}
        <option value="{{ v }}" {% if v == current_video %}selected{% endif %}>{{ v }}</option>
        {% endfor %}
      </select>
      <button class="full" onclick="switchVideo()">▷ Переключить</button>
      {% else %}
      <input type="text" id="url-input" value="{{ source_url }}"
             placeholder="rtsp:// или http://...m3u8" />
      <div style="margin-top:8px">
        <button class="full" onclick="reloadSource()">⟳ Перезапустить поток</button>
      </div>
      {% endif %}
    </div>

    <!-- Log -->
    <div class="section">
      <div class="section-title">Лог детекций</div>
      <ul id="det-log"></ul>
    </div>

    <div class="footer-note">Детекция: только люди (class 0)</div>
  </aside>
</main>

<script>
// ── Sparkline ──────────────────────────────────────────────────────────────
const canvas  = document.getElementById('sparkline');
const ctx     = canvas.getContext('2d');
const history = Array(60).fill(0);

function drawSparkline() {
  const W = canvas.offsetWidth * devicePixelRatio;
  const H = canvas.offsetHeight * devicePixelRatio;
  canvas.width = W; canvas.height = H;
  const max  = Math.max(...history, 1);
  const step = W / (history.length - 1);
  ctx.clearRect(0, 0, W, H);
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba(0,255,136,.25)'); grad.addColorStop(1,'rgba(0,255,136,0)');
  ctx.beginPath(); ctx.moveTo(0,H);
  history.forEach((v,i) => ctx.lineTo(i*step, H-(v/max)*(H-6)));
  ctx.lineTo(W,H); ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
  ctx.beginPath(); ctx.strokeStyle='#00ff88';
  ctx.lineWidth=1.5*devicePixelRatio; ctx.lineJoin='round';
  history.forEach((v,i) => i===0 ? ctx.moveTo(i*step,H-(v/max)*(H-6)) : ctx.lineTo(i*step,H-(v/max)*(H-6)));
  ctx.stroke();
}

// ── Stats polling ──────────────────────────────────────────────────────────
let errCount = 0;

async function pollStats() {
  try {
    const d = await fetch('/stats').then(r => r.json());
    errCount = 0;

    document.getElementById('stat-persons').textContent = d.persons ?? 0;
    document.getElementById('stat-fps').textContent     = d.fps  ? d.fps.toFixed(1)  : '—';
    document.getElementById('stat-ms').textContent      = d.ms   ? d.ms.toFixed(0)   : '—';
    document.getElementById('stat-dfps').textContent    = d.dfps ? d.dfps.toFixed(1) : '—';

    history.push(d.persons ?? 0); history.shift(); drawSparkline();
    if ((d.persons ?? 0) > 0) addLogEntry(d.persons, d.ts);

    // ── Model badge & buttons ──
    const loaded  = d.model_size;
    const loading = d.model_loading;

    document.getElementById('active-model-label').textContent = `yolov8${loaded}.pt`;

    const badge = document.getElementById('model-badge');
    if (loading) {
      badge.textContent = `ЗАГРУЗКА ${loading.toUpperCase()}…`;
      badge.classList.add('loading');
    } else {
      badge.textContent = `MODEL: ${loaded.toUpperCase()}`;
      badge.classList.remove('loading');
    }

    ['n','s','m','l','x'].forEach(sz => {
      const btn = document.getElementById(`mbtn-${sz}`);
      if (loading) {
        btn.className = 'model-btn' + (sz === loading ? ' loading-btn' : '');
      } else {
        btn.className = 'model-btn' + (sz === loaded  ? ' active'      : '');
      }
    });

    // ── FPM skip overlay ──
    const skip = d.frame_skip ?? 1;
    document.getElementById('fpm-skip-note').textContent = skip;
    const ov = document.getElementById('skip-overlay');
    if (skip > 1) { ov.style.display='block'; ov.textContent=`детекция: 1 из ${skip} кадров`; }
    else          { ov.style.display='none'; }

    setStatus(
      loading ? 'loading' : 'live',
      loading ? `ЗАГРУЗКА yolov8${loading}…` : `LIVE · ${(d.source_type||'').slice(0,32)}`
    );
  } catch {
    if (++errCount > 3) setStatus('error','НЕТ СИГНАЛА');
  }
}

function addLogEntry(cnt, ts) {
  const log = document.getElementById('det-log');
  const li  = document.createElement('li');
  const t   = ts ? new Date(ts*1000).toLocaleTimeString('ru') : '—';
  li.innerHTML = `<span class="ts">${t}</span><span class="cnt">${cnt}</span><span>чел.</span>`;
  log.prepend(li);
  while (log.children.length > 40) log.removeChild(log.lastChild);
}

function setStatus(s, text) {
  document.getElementById('status-dot').className   = s;
  document.getElementById('status-text').textContent = text;
}

function onFrameLoad()  {}
function onFrameError() { setStatus('error','НЕТ СИГНАЛА'); }

// ── Sliders ────────────────────────────────────────────────────────────────
function onConfChange(v)  { document.getElementById('conf-val').textContent  = parseFloat(v).toFixed(2); }
function onImgszChange(v) { document.getElementById('imgsz-val').textContent = v; }
function onFpmChange(v)   { document.getElementById('fpm-val').textContent   = v; }

async function applyConf()  { await post('/set_params', { conf:  parseFloat(document.getElementById('conf-slider').value) }); }
async function applyImgsz() { await post('/set_params', { imgsz: parseInt(document.getElementById('imgsz-slider').value)  }); }
async function applyFpm()   { await post('/set_params', { fpm:   parseInt(document.getElementById('fpm-slider').value)    }); }

// ── Model switch ───────────────────────────────────────────────────────────
async function switchModel(sz) {
  ['n','s','m','l','x'].forEach(s => {
    document.getElementById(`mbtn-${s}`).className = 'model-btn' + (s===sz ? ' loading-btn' : '');
  });
  setStatus('loading', `ЗАГРУЗКА yolov8${sz}…`);
  await post('/set_model', { model: sz });
}

// ── Source ─────────────────────────────────────────────────────────────────
async function reloadSource() {
  const url = document.getElementById('url-input')?.value?.trim();
  if (!url) return;
  document.getElementById('src-overlay').textContent = url;
  await post('/set_source', { url });
}
async function switchVideo() {
  const v = document.getElementById('video-select')?.value;
  if (!v) return;
  document.getElementById('src-overlay').textContent = v;
  await post('/set_source', { video: v });
}

async function post(url, body) {
  return fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
}

drawSparkline();
setInterval(pollStats, 700);
pollStats();
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Глобальное состояние
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "conf":          0.45,
    "imgsz":         640,
    "fpm":           60,        # лимит кадров детекции в минуту
    "persons":       0,
    "fps":           0.0,       # FPS всего стрима
    "dfps":          0.0,       # FPS только детекции
    "ms":            0.0,
    "frames":        0,
    "frame_skip":    1,
    "source_type":   "INIT",
    "ts":            time.time(),
    "model_size":    "m",
    "model_loading": None,
    "lock":          threading.Lock(),
}

_source_url:    str | None = None
_source_folder: str | None = None
_current_video: str | None = None
_restart_event      = threading.Event()
_model_reload_event = threading.Event()

_frame_queue: queue.Queue = queue.Queue(maxsize=2)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".webm", ".m4v"}

# ──────────────────────────────────────────────────────────────────────────────
# YOLO — горячая замена модели
# ──────────────────────────────────────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()


def load_yolo(size: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)
    name = f"yolov8{size}.pt"
    print(f"[INFO] Загружаем {name}…")
    return YOLO(name)


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            with state["lock"]:
                sz = state["model_size"]
            _model = load_yolo(sz)
    return _model


def _do_reload_model(size: str):
    """Фоновая загрузка новой модели с атомарной заменой."""
    global _model
    new_m = load_yolo(size)
    with _model_lock:
        _model = new_m
    with state["lock"]:
        state["model_size"]    = size
        state["model_loading"] = None
    _model_reload_event.set()
    print(f"[INFO] Модель переключена → yolov8{size}")


# ──────────────────────────────────────────────────────────────────────────────
# Детекция
# ──────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {0: (50, 205, 50)}


def detect_persons(model, frame: np.ndarray, conf: float, imgsz: int):
    t0 = time.perf_counter()
    results    = model(frame, classes=[0], conf=conf, iou=0.45, imgsz=imgsz, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    annotated = frame.copy()
    count = 0
    for box in results.boxes:
        cls_id     = int(box.cls)
        confidence = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        count += 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        lbl = f"person  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        pad = 4
        cv2.rectangle(annotated, (x1, y1-th-pad*2), (x1+tw+pad*2, y1), color, -1)
        cv2.putText(annotated, lbl, (x1+pad, y1-pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    _draw_legend(annotated, count, elapsed_ms)
    return annotated, count, elapsed_ms


def _draw_legend(frame: np.ndarray, persons: int, ms: float):
    with state["lock"]:
        model_sz = state["model_size"]
        fpm      = state["fpm"]
        skip     = state["frame_skip"]
    lines = [
        f"Inference: {ms:.0f} ms",
        f"Person:    {persons}",
        f"Model:     yolov8{model_sz}",
        f"FPM limit: {fpm}  (1/{skip})",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.52, 1
    pad, lh = 8, 20
    w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + pad*2
    h = len(lines) * lh + pad*2
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10+w, 10+h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    colors = [(180,180,180), (50,205,50), (90,130,255), (180,130,0)]
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(frame, line, (10+pad, 10+pad + lh//2 + i*lh),
                    font, scale, color, thick, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Frame-skip из FPM
# ──────────────────────────────────────────────────────────────────────────────
def compute_skip(stream_fps: float, fpm: int) -> int:
    if stream_fps <= 0:
        return 1
    fps_target = fpm / 60.0
    return max(1, int(round(stream_fps / fps_target)))


# ──────────────────────────────────────────────────────────────────────────────
# Поток захвата + детекции
# ──────────────────────────────────────────────────────────────────────────────
def _iter_video_sources():
    global _current_video
    if _source_folder:
        files = sorted(p for p in Path(_source_folder).iterdir()
                       if p.suffix.lower() in VIDEO_EXTS)
        if not files:
            print(f"[WARN] Нет видео в {_source_folder}"); return
        idx = 0
        while True:
            p = files[idx % len(files)]
            _current_video = p.name
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                print(f"[INFO] Воспроизводим: {p.name}")
                yield cap, p.name
            else:
                print(f"[WARN] Не удалось: {p}")
            idx += 1
    else:
        url = _source_url
        while True:
            if _restart_event.is_set():
                url = _source_url; _restart_event.clear()
            print(f"[INFO] Подключаемся: {url}")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if cap.isOpened():
                yield cap, url
            else:
                print("[WARN] Повтор через 3 с…"); time.sleep(3)


def capture_thread():
    get_model()   # прогрев до начала

    sfps_timer, sfps_cnt = time.perf_counter(), 0
    dfps_timer, dfps_cnt = time.perf_counter(), 0
    sfps_cur = 0.0

    for cap, label in _iter_video_sources():
        with state["lock"]:
            state["source_type"] = label[:50]

        frame_idx      = 0
        last_annotated = None

        while True:
            if _restart_event.is_set():
                cap.release(); break

            # Горячая замена модели
            if _model_reload_event.is_set():
                _model_reload_event.clear()

            ok, frame = cap.read()
            if not ok:
                print(f"[INFO] Конец: {label}")
                cap.release(); break

            frame_idx += 1

            # stream FPS
            sfps_cnt += 1
            el = time.perf_counter() - sfps_timer
            if el >= 1.0:
                sfps_cur = sfps_cnt / el
                sfps_cnt = 0; sfps_timer = time.perf_counter()
                with state["lock"]:
                    state["fps"] = sfps_cur

            with state["lock"]:
                fpm   = state["fpm"]
                conf  = state["conf"]
                imgsz = state["imgsz"]

            skip = compute_skip(sfps_cur if sfps_cur > 0 else 25.0, fpm)
            with state["lock"]:
                state["frame_skip"] = skip

            if frame_idx % skip == 0:
                with _model_lock:
                    mdl = _model
                annotated, persons, ms = detect_persons(mdl, frame, conf, imgsz)
                last_annotated = annotated

                dfps_cnt += 1
                del_t = time.perf_counter() - dfps_timer
                if del_t >= 1.0:
                    with state["lock"]:
                        state["dfps"] = dfps_cnt / del_t
                    dfps_cnt = 0; dfps_timer = time.perf_counter()

                with state["lock"]:
                    state["persons"] = persons
                    state["ms"]      = ms
                    state["frames"] += 1
                    state["ts"]      = time.time()
            else:
                annotated = last_annotated if last_annotated is not None else frame

            try:
                _frame_queue.put_nowait(annotated)
            except queue.Full:
                try:    _frame_queue.get_nowait()
                except queue.Empty: pass
                try:    _frame_queue.put_nowait(annotated)
                except queue.Full:  pass


# ──────────────────────────────────────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


def _gen_frames():
    blank = None
    while True:
        try:
            frame = _frame_queue.get(timeout=2.0)
        except queue.Empty:
            if blank is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Ожидание потока...", (160, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (55, 55, 55), 2)
            frame = blank
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    videos = []
    if _source_folder:
        videos = sorted(p.name for p in Path(_source_folder).iterdir()
                        if p.suffix.lower() in VIDEO_EXTS)
    with state["lock"]:
        sz    = state["model_size"]
        conf  = state["conf"]
        imgsz = state["imgsz"]
        fpm   = state["fpm"]
    return render_template_string(
        HTML_TEMPLATE,
        model=sz.upper(), model_size=sz,
        conf=conf, imgsz=imgsz, fpm=fpm,
        source_url=_source_url or "",
        source_label=(_source_url or _source_folder or ""),
        videos=videos, current_video=_current_video or "",
    )


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats_route():
    with state["lock"]:
        return jsonify({
            "persons":       state["persons"],
            "fps":           state["fps"],
            "dfps":          state["dfps"],
            "ms":            state["ms"],
            "frames":        state["frames"],
            "frame_skip":    state["frame_skip"],
            "source_type":   state["source_type"],
            "ts":            state["ts"],
            "model_size":    state["model_size"],
            "model_loading": state["model_loading"],
        })


@app.route("/set_params", methods=["POST"])
def set_params():
    d = request.get_json(force=True)
    with state["lock"]:
        if "conf"  in d: state["conf"]  = float(d["conf"])
        if "imgsz" in d: state["imgsz"] = int(d["imgsz"])
        if "fpm"   in d: state["fpm"]   = max(1, int(d["fpm"]))
    return jsonify({"ok": True})


@app.route("/set_model", methods=["POST"])
def set_model_route():
    d    = request.get_json(force=True)
    size = d.get("model", "m").strip().lower()
    if size not in {"n","s","m","l","x"}:
        return jsonify({"ok": False, "error": "invalid"}), 400
    with state["lock"]:
        if state["model_loading"] is not None:
            return jsonify({"ok": False, "error": "already loading"}), 409
        if state["model_size"] == size:
            return jsonify({"ok": True, "note": "already loaded"})
        state["model_loading"] = size
    threading.Thread(target=_do_reload_model, args=(size,), daemon=True,
                     name=f"reload-{size}").start()
    return jsonify({"ok": True})


@app.route("/set_source", methods=["POST"])
def set_source():
    global _source_url, _current_video
    d = request.get_json(force=True)
    if "url"   in d: _source_url    = d["url"].strip()
    if "video" in d: _current_video = d["video"]
    _restart_event.set()
    return jsonify({"ok": True})


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global _source_url, _source_folder

    parser = argparse.ArgumentParser(description="Веб-визуализация детекции людей (YOLOv8)")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",    type=str)
    group.add_argument("--folder", type=str)

    parser.add_argument("--model",  type=str,   default="m", choices=["n","s","m","l","x"],
                        help="Начальная модель (n/s/m/l/x). По умолчанию: m")
    parser.add_argument("--conf",   type=float, default=0.45)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--fpm",    type=int,   default=60,
                        help="Лимит детекций в минуту. По умолчанию: 60")
    parser.add_argument("--port",   type=int,   default=5000)

    args = parser.parse_args()

    with state["lock"]:
        state["model_size"] = args.model
        state["conf"]       = args.conf
        state["imgsz"]      = args.imgsz
        state["fpm"]        = max(1, args.fpm)

    _source_url    = args.url
    _source_folder = args.folder

    threading.Thread(target=capture_thread, daemon=True, name="capture").start()

    print(f"\n[✓] Открой браузер → http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()