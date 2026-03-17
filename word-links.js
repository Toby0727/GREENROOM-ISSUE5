// word-links.js
// Scans all rendered messages, finds words that appear in 2+ messages,
// highlights them in a shared colour, and draws animated dashed lines between them.
(function () {
  'use strict';

  // ── config ───────────────────────────────────────────────────────────────
  const MIN_WORD_LEN = 4;
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

  // Palette that fits the dark terminal aesthetic
  const COLORS = [
    '#00d4ff', // cyan
    '#ffb800', // amber
    '#c474ff', // violet
    '#00ffd0', // teal
    '#ff6b35', // orange
    '#ff3f8e', // pink
    '#7fff00', // chartreuse
  ];

  let colorIdx = 0;
  const wordColors = Object.create(null);

  function getColor(word) {
    if (!wordColors[word]) {
      wordColors[word] = COLORS[colorIdx % COLORS.length];
      colorIdx++;
    }
    return wordColors[word];
  }

  // ── text helpers ─────────────────────────────────────────────────────────
  function tokenizeUnique(text) {
    return [...new Set(
      String(text || '')
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length >= MIN_WORD_LEN && !STOP_WORDS.has(w))
    )];
  }

  function escapeRe(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // ── DOM helpers ───────────────────────────────────────────────────────────
  // Returns text elements that are fully rendered (user msgs + completed AI msgs)
  function readyTextEls(container) {
    return [
      ...container.querySelectorAll('.msg-text:not(.ai)'),
      ...container.querySelectorAll('.msg-text.ai.ai-done'),
    ];
  }

  function restoreEl(el) {
    if (el.dataset.wlOrig !== undefined) {
      el.textContent = el.dataset.wlOrig;
      delete el.dataset.wlOrig;
    }
  }

  function highlightEl(el, words) {
    const text = el.textContent;
    el.dataset.wlOrig = text;

    const re = new RegExp(`\\b(${words.map(escapeRe).join('|')})\\b`, 'gi');
    const frag = document.createDocumentFragment();
    let last = 0, m;

    while ((m = re.exec(text)) !== null) {
      if (m.index > last) {
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      }
      const w = m[1].toLowerCase();
      const c = getColor(w);
      const mark = document.createElement('mark');
      mark.dataset.wlWord = w;
      mark.style.cssText = [
        'background:transparent',
        `color:${c}`,
        `text-shadow:0 0 10px ${c}88`,
        `border-bottom:1px solid ${c}66`,
        'padding-bottom:1px',
        'border-radius:1px',
      ].join(';');
      mark.textContent = m[1];
      frag.appendChild(mark);
      last = re.lastIndex;
    }

    if (last < text.length) {
      frag.appendChild(document.createTextNode(text.slice(last)));
    }

    el.textContent = '';
    el.appendChild(frag);
  }

  // ── SVG line drawing ──────────────────────────────────────────────────────
  function drawLines(container, svg, sharedWords) {
    svg.innerHTML = '';
    // Expand SVG to cover the full scrollable height
    svg.style.height = container.scrollHeight + 'px';

    const cRect  = container.getBoundingClientRect();
    const scrollT = container.scrollTop;

    sharedWords.forEach(word => {
      const marks = [...container.querySelectorAll(`mark[data-wl-word="${word}"]`)];
      if (marks.length < 2) return;

      const color = getColor(word);

      for (let i = 0; i < marks.length - 1; i++) {
        const rA = marks[i].getBoundingClientRect();
        const rB = marks[i + 1].getBoundingClientRect();

        // Convert from viewport coords → SVG content coords
        const x1 = rA.right  - cRect.left;
        const y1 = rA.top    - cRect.top  + scrollT + rA.height / 2;
        const x2 = rB.left   - cRect.left;
        const y2 = rB.top    - cRect.top  + scrollT + rB.height / 2;

        // Skip lines that are near-flat (same block) or off-canvas
        if (Math.abs(y2 - y1) < 4) continue;

        // Bezier curve — control points fan out horizontally
        const bendX = Math.min(48, Math.abs(x2 - x1) * 0.45 + 16);
        const d = `M ${x1} ${y1} C ${x1 + bendX} ${y1}, ${x2 - bendX} ${y2}, ${x2} ${y2}`;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d);
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', '1');
        path.setAttribute('stroke-opacity', '0.38');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-dasharray', '4 3');

        // Flowing animated dash
        const anim = document.createElementNS('http://www.w3.org/2000/svg', 'animateTransform');
        anim.setAttribute('attributeName', 'stroke-dashoffset');
        anim.setAttribute('from', '0');
        anim.setAttribute('to', '-14');
        anim.setAttribute('dur', '2s');
        anim.setAttribute('repeatCount', 'indefinite');

        // Use animate (not animateTransform) for stroke-dashoffset
        const animPlain = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
        animPlain.setAttribute('attributeName', 'stroke-dashoffset');
        animPlain.setAttribute('from', '0');
        animPlain.setAttribute('to', '-14');
        animPlain.setAttribute('dur', '2s');
        animPlain.setAttribute('repeatCount', 'indefinite');
        path.appendChild(animPlain);

        // Dot at each end
        [{ x: x1, y: y1 }, { x: x2, y: y2 }].forEach(pt => {
          const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          dot.setAttribute('cx', pt.x);
          dot.setAttribute('cy', pt.y);
          dot.setAttribute('r', '2.5');
          dot.setAttribute('fill', color);
          dot.setAttribute('fill-opacity', '0.55');
          svg.appendChild(dot);
        });

        svg.appendChild(path);
      }
    });
  }

  // ── main update cycle ─────────────────────────────────────────────────────
  function update(container, svg) {
    const els = readyTextEls(container);
    els.forEach(restoreEl);

    // word → [elements that contain it]
    const wordMap = Object.create(null);
    els.forEach(el => {
      tokenizeUnique(el.textContent).forEach(w => {
        if (!wordMap[w]) wordMap[w] = [];
        wordMap[w].push(el);
      });
    });

    const shared = Object.keys(wordMap).filter(w => wordMap[w].length >= 2);

    svg.innerHTML = '';
    if (!shared.length) return;

    // Build el → [words to highlight] map
    const elWords = new Map();
    shared.forEach(word => {
      wordMap[word].forEach(el => {
        if (!elWords.has(el)) elWords.set(el, []);
        elWords.get(el).push(word);
      });
    });

    elWords.forEach((words, el) => highlightEl(el, words));
    drawLines(container, svg, shared);
  }

  // ── init ──────────────────────────────────────────────────────────────────
  function init() {
    const container = document.getElementById('messages');
    if (!container) { setTimeout(init, 200); return; }

    // SVG sits inside the scroll container at position 0,0 covering full height
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.id   = 'word-link-svg';
    svg.style.cssText = [
      'position:absolute', 'top:0', 'left:0',
      'width:100%', 'height:100%',
      'pointer-events:none', 'z-index:5', 'overflow:visible',
    ].join(';');
    container.appendChild(svg);

    let busy  = false;
    let rafId = null;

    function schedule() {
      if (busy) return;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        busy = true;
        try { update(container, svg); } finally { busy = false; }
      });
    }

    // Observe DOM changes (new messages, ai-done class added)
    new MutationObserver(schedule).observe(container, {
      childList: true, subtree: true,
      attributes: true, attributeFilter: ['class'],
    });

    // Redraw when container is resized
    if (window.ResizeObserver) {
      new ResizeObserver(schedule).observe(container);
    }

    // Redraw on scroll (lines need repositioning relative to offset SVG)
    container.addEventListener('scroll', schedule, { passive: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
