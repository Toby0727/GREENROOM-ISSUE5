// word-links.js
// Highlights words that appear in multiple messages with a pulsing colour glow.
// 0.5 Hz pulse (2s cycle). No lines, no SVG.
(function () {
  'use strict';

  // ── config ────────────────────────────────────────────────────────────────
  const MIN_WORD_LEN  = 4;
  const PULSE_DUR_S   = 2.0;  // 1 full cycle = 0.5 Hz
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
    if (!wordColors[w]) wordColors[w] = COLORS[colorIdx++ % COLORS.length];
    return wordColors[w];
  }

  // ── inject keyframe CSS once ──────────────────────────────────────────────
  (function injectStyles() {
    if (document.getElementById('wl-styles')) return;
    const s = document.createElement('style');
    s.id = 'wl-styles';
    // Uses --wl-c custom property (set inline per mark) for reliable animation.
    // currentColor in @keyframes text-shadow is not consistently interpolated
    // across browsers — CSS custom properties are.
    s.textContent = `
@keyframes wl-pulse {
  0%,100% {
    text-shadow: 0 0 6px var(--wl-c), 0 0 14px var(--wl-c);
    filter: drop-shadow(0 0 3px var(--wl-c));
    opacity: 0.65;
  }
  50% {
    text-shadow: 0 0 10px var(--wl-c), 0 0 30px var(--wl-c), 0 0 60px var(--wl-c);
    filter: drop-shadow(0 0 8px var(--wl-c)) drop-shadow(0 0 16px var(--wl-c));
    opacity: 1;
  }
}
mark[data-wl-word] {
  background: transparent !important;
  color: var(--wl-c) !important;
  border-bottom: 1.5px solid var(--wl-c);
  padding-bottom: 1px;
  border-radius: 1px;
  animation: wl-pulse ${PULSE_DUR_S}s ease-in-out infinite;
}`;
    document.head.appendChild(s);
  })();

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
    const re   = new RegExp(`\\b(${words.map(escRe).join('|')})\\b`, 'gi');
    const frag = document.createDocumentFragment();
    let last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const w    = m[1].toLowerCase();
      const c    = getColor(w);
      const mark = document.createElement('mark');
      mark.dataset.wlWord = w;
      // Set --wl-c custom property so the @keyframes animation can reference it reliably
      mark.style.setProperty('--wl-c', c);
      mark.style.color = c;
      mark.textContent = m[1];
      frag.appendChild(mark);
      last = re.lastIndex;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    el.textContent = '';
    el.appendChild(frag);
  }

  // ── update cycle ──────────────────────────────────────────────────────────
  function update(container) {
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
    if (!shared.length) return;

    const elWords = new Map();
    shared.forEach(word => {
      wordMap[word].forEach(e => {
        if (!elWords.has(e)) elWords.set(e, []);
        elWords.get(e).push(word);
      });
    });

    elWords.forEach((words, e) => highlightEl(e, words));
  }

  // ── init ──────────────────────────────────────────────────────────────────
  function init() {
    const container = document.getElementById('messages');
    if (!container) { setTimeout(init, 200); return; }

    let updateTimer = null;
    let updating    = false;

    function schedule() {
      if (updating) return;
      clearTimeout(updateTimer);
      updateTimer = setTimeout(() => {
        updating = true;
        try { update(container); } catch (e) { console.error('[word-links]', e); }
        updating = false;
      }, 60);
    }

    new MutationObserver(schedule).observe(container, {
      childList: true, subtree: true,
      attributes: true, attributeFilter: ['class'],
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
