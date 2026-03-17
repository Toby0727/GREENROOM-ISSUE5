// word-links.js
// Connects repeated words across all messages (user + AI) with straight pinned
// investigation-board lines that glow and spark.
(function () {
  'use strict';

  // ── config ────────────────────────────────────────────────────────────────
  const MIN_WORD_LEN   = 4;
  const LINE_WIDTH     = 2.2;
  const DOT_DUR_MIN    = 1.0;   // seconds per traversal (min)
  const DOT_DUR_MAX    = 2.2;   // seconds per traversal (max)
  const DOT_PAUSE_MIN  = 60;    // ms gap between traversals (min)
  const DOT_PAUSE_MAX  = 350;   // ms gap between traversals (max)
  const STOP_WORDS = new Set([
    'that','this','with','have','from','they','been','were','what','when',
    'where','your','their','will','would','could','should','about','there',
    'which','more','also','into','than','then','some','just','does','make',
    'even','only','very','these','those','such','here','each','much','most',
    'over','after','like','used','well','back','many','good','know','time',
    'long','come','look','them','said','need','feel','seem','work','call',
    'same','tell','help','want','give','show','keep','real','away','both',
    'life','left','next','open','being','because','through','before','between',
    'without','around','always','never','every','other','another','something',
    'nothing','anything','everything','still','while','where','again','under',
    'until','along','though'
  ]);

  const COLORS = [
    '#00d4ff', '#ffb800', '#c474ff',
    '#00ffd0', '#ff6b35', '#ff3f8e', '#7fff00',
  ];

  let colorIdx = 0;
  const wordColors = Object.create(null);
  function getColor(w) {
    if (!wordColors[w]) { wordColors[w] = COLORS[colorIdx++ % COLORS.length]; }
    return wordColors[w];
  }

  // ── text helpers ──────────────────────────────────────────────────────────
  function tokenizeUnique(text) {
    return [...new Set(
      String(text || '').toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)
        .filter(w => w.length >= MIN_WORD_LEN && !STOP_WORDS.has(w))
    )];
  }
  function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

  // ── ready elements (user msgs + completed AI replies) ─────────────────────
  function readyTextEls(container) {
    return [
      ...container.querySelectorAll('.msg-text:not(.ai)'),
      ...container.querySelectorAll('.msg-text.ai.ai-done'),
    ];
  }

  // ── highlight ─────────────────────────────────────────────────────────────
  function restoreEl(el) {
    if (el.dataset.wlOrig !== undefined) {
      el.textContent = el.dataset.wlOrig;
      delete el.dataset.wlOrig;
    }
  }

  function highlightEl(el, words) {
    const text = el.textContent;
    el.dataset.wlOrig = text;
    const re  = new RegExp(`\\b(${words.map(escRe).join('|')})\\b`, 'gi');
    const frag = document.createDocumentFragment();
    let last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const w    = m[1].toLowerCase();
      const c    = getColor(w);
      const mark = document.createElement('mark');
      mark.dataset.wlWord = w;
      mark.style.cssText  = [
        'background:transparent',
        `color:${c}`,
        `text-shadow:0 0 8px ${c}aa, 0 0 2px ${c}`,
        `border-bottom:1.5px solid ${c}`,
        'padding-bottom:1px',
        'border-radius:1px',
        'transition:text-shadow .3s',
      ].join(';');
      mark.textContent = m[1];
      frag.appendChild(mark);
      last = re.lastIndex;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    el.textContent = '';
    el.appendChild(frag);
  }

  // ── SVG helpers ───────────────────────────────────────────────────────────
  const NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  }

  // Thumbtack pin at (cx, cy)
  function drawPin(svg, cx, cy, c) {
    svg.appendChild(svgEl('circle', { cx, cy, r: 8,  fill: 'rgba(0,0,0,0.45)' }));
    svg.appendChild(svgEl('circle', { cx, cy, r: 7,  fill: c, 'fill-opacity': '0.18',
      stroke: c, 'stroke-width': '1.5', 'stroke-opacity': '0.85' }));
    svg.appendChild(svgEl('circle', { cx, cy, r: 4,  fill: c, 'fill-opacity': '0.9',
      filter: `drop-shadow(0 0 4px ${c})` }));
    svg.appendChild(svgEl('circle', { cx, cy, r: 1.5, fill: '#fff', 'fill-opacity': '0.9' }));
  }

  // Straight line with glow layers
  function drawLine(svg, x1, y1, x2, y2, c) {
    // Outer diffuse glow
    svg.appendChild(svgEl('line', { x1, y1, x2, y2,
      stroke: c, 'stroke-width': LINE_WIDTH * 5,
      'stroke-opacity': '0.07', 'stroke-linecap': 'round' }));
    // Mid glow
    svg.appendChild(svgEl('line', { x1, y1, x2, y2,
      stroke: c, 'stroke-width': LINE_WIDTH * 2.5,
      'stroke-opacity': '0.18', 'stroke-linecap': 'round' }));
    // Core dashed line
    const core = svgEl('line', { x1, y1, x2, y2,
      stroke: c, 'stroke-width': LINE_WIDTH,
      'stroke-opacity': '0.75', 'stroke-linecap': 'round',
      'stroke-dasharray': '5 4' });
    const anim = document.createElementNS(NS, 'animate');
    anim.setAttribute('attributeName', 'stroke-dashoffset');
    anim.setAttribute('from', '0'); anim.setAttribute('to', '-18');
    anim.setAttribute('dur', '1.6s'); anim.setAttribute('repeatCount', 'indefinite');
    core.appendChild(anim);
    svg.appendChild(core);
    return { x1, y1, x2, y2, c };
  }

  // Traveling dot: launches once, cleans up, then loops itself continuously
  let dotGen = 0;
  function clearDots() { dotGen++; }

  function launchDot(svg, ld, gen) {
    if (gen !== dotGen) return;                        // stale generation → stop
    const fwd         = Math.random() > 0.4;
    const { x1, y1, x2, y2, c } = ld;
    const [sx, sy, ex, ey] = fwd ? [x1, y1, x2, y2] : [x2, y2, x1, y1];
    const dur = DOT_DUR_MIN + Math.random() * (DOT_DUR_MAX - DOT_DUR_MIN);
    const r   = 1.8 + Math.random() * 1.4;

    const dot = svgEl('circle', {
      r, cx: sx, cy: sy,
      fill: '#fff', 'fill-opacity': '0',
      filter: `drop-shadow(0 0 3px ${c}) drop-shadow(0 0 6px #fff)`,
    });

    const mkAnim = (attr, from, to) => {
      const a = document.createElementNS(NS, 'animate');
      a.setAttribute('attributeName', attr);
      a.setAttribute('from', String(from));
      a.setAttribute('to',   String(to));
      a.setAttribute('dur',  `${dur.toFixed(2)}s`);
      a.setAttribute('fill', 'freeze');
      return a;
    };
    dot.appendChild(mkAnim('cx', sx, ex));
    dot.appendChild(mkAnim('cy', sy, ey));

    const fade = document.createElementNS(NS, 'animate');
    fade.setAttribute('attributeName', 'fill-opacity');
    fade.setAttribute('values', '0;0.9;0.9;0');
    fade.setAttribute('keyTimes', '0;0.1;0.85;1');
    fade.setAttribute('dur', `${dur.toFixed(2)}s`);
    fade.setAttribute('fill', 'freeze');
    dot.appendChild(fade);

    svg.appendChild(dot);

    setTimeout(() => {
      try { svg.removeChild(dot); } catch {}
      const pause = DOT_PAUSE_MIN + Math.random() * (DOT_PAUSE_MAX - DOT_PAUSE_MIN);
      setTimeout(() => launchDot(svg, ld, gen), pause);
    }, dur * 1000 + 60);
  }

  // ── main draw ─────────────────────────────────────────────────────────────
  function drawLines(container, svg, sharedWords) {
    svg.innerHTML = '';
    clearDots();
    svg.style.height = container.scrollHeight + 'px';

    const cRect  = container.getBoundingClientRect();
    const scrollT = container.scrollTop;
    const lines   = [];

    sharedWords.forEach(word => {
      const marks = [...container.querySelectorAll(`mark[data-wl-word="${word}"]`)];
      if (marks.length < 2) return;
      const c = getColor(word);

      for (let i = 0; i < marks.length - 1; i++) {
        const rA = marks[i].getBoundingClientRect();
        const rB = marks[i + 1].getBoundingClientRect();
        const x1 = Math.round(rA.right - cRect.left);
        const y1 = Math.round(rA.top   - cRect.top + scrollT + rA.height / 2);
        const x2 = Math.round(rB.left  - cRect.left);
        const y2 = Math.round(rB.top   - cRect.top + scrollT + rB.height / 2);
        if (Math.abs(y2 - y1) < 6 && Math.abs(x2 - x1) < 6) continue;

        const ld = drawLine(svg, x1, y1, x2, y2, c);
        lines.push(ld);
        drawPin(svg, x1, y1, c);
        drawPin(svg, x2, y2, c);
      }
    });

    // Continuous traveling dots per line — 2-3 staggered dots looping forever
    const gen = dotGen;
    lines.forEach(ld => {
      const numDots = 2 + Math.floor(Math.random() * 2);
      for (let k = 0; k < numDots; k++) {
        const stagger = k * (700 + Math.random() * 500);
        setTimeout(() => launchDot(svg, ld, gen), stagger);
      }
    });
  }

  // ── update cycle ──────────────────────────────────────────────────────────
  function update(container, svg) {
    const els = readyTextEls(container);
    els.forEach(restoreEl);

    const wordMap = Object.create(null);
    els.forEach(e => {
      tokenizeUnique(e.textContent).forEach(w => {
        if (!wordMap[w]) wordMap[w] = [];
        wordMap[w].push(e);
      });
    });

    const shared = Object.keys(wordMap).filter(w => wordMap[w].length >= 2);
    svg.innerHTML = '';
    clearDots();
    if (!shared.length) return;

    const elWords = new Map();
    shared.forEach(word => {
      wordMap[word].forEach(e => {
        if (!elWords.has(e)) elWords.set(e, []);
        elWords.get(e).push(word);
      });
    });

    elWords.forEach((words, e) => highlightEl(e, words));
    drawLines(container, svg, shared);
  }

  // ── init ──────────────────────────────────────────────────────────────────
  function init() {
    const container = document.getElementById('messages');
    if (!container) { setTimeout(init, 200); return; }

    const svg = document.createElementNS(NS, 'svg');
    svg.id = 'word-link-svg';
    svg.style.cssText = [
      'position:absolute', 'top:0', 'left:0',
      'width:100%', 'height:100%',
      'pointer-events:none', 'z-index:5', 'overflow:visible',
    ].join(';');
    container.appendChild(svg);

    let busy = false, rafId = null;
    function schedule() {
      if (busy) return;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        busy = true;
        try { update(container, svg); } finally { busy = false; }
      });
    }

    new MutationObserver(schedule).observe(container, {
      childList: true, subtree: true,
      attributes: true, attributeFilter: ['class'],
    });
    if (window.ResizeObserver) new ResizeObserver(schedule).observe(container);
    container.addEventListener('scroll', schedule, { passive: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
