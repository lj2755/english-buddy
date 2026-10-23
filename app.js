// English Buddy — todo corre en el celular: la IA en la GPU con WebLLM (WebGPU),
// la voz en Chrome y la memoria en el almacenamiento del navegador.
import * as webllm from 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm';

(() => {
  const $ = id => document.getElementById(id);
  const chatEl = $('chat');
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  // Modelos que caben en un celular. El sufijo depende de si la GPU soporta números de 16 bits.
  const MODELS = [
    { base: 'Qwen3-4B', label: 'Qwen3 4B · mejor calidad' },
    { base: 'Qwen3.5-4B', label: 'Qwen3.5 4B · más nuevo' },
    { base: 'Qwen3.5-2B', label: 'Qwen3.5 2B · más rápido' },
    { base: 'Qwen3-1.7B', label: 'Qwen3 1.7B · el más liviano' },
  ];
  const DEFAULT_MODEL = 'Qwen3-4B';

  const SCENARIO_NAMES = {
    free: 'Charla libre', coffee: 'Cafetería', restaurant: 'Restaurante', shopping: 'Tienda de ropa',
    doctor: 'Médico', airport: 'Aeropuerto', directions: 'Direcciones', smalltalk: 'Small talk', interview: 'Entrevista',
  };
  const KIND_NAMES = { grammar: 'gramática', vocabulary: 'vocabulario', naturalness: 'natural', pronunciation: 'pronunciación' };
  const SCENARIOS = {
    free: 'Free everyday conversation about the learner\'s day, life, plans and interests.',
    coffee: 'Role-play: you are a barista at a busy coffee shop and the learner is ordering.',
    restaurant: 'Role-play: you are a waiter at a restaurant; the learner is a customer.',
    doctor: 'Role-play: you are a doctor; the learner is a patient describing symptoms.',
    interview: 'Role-play: you are a friendly hiring manager running a job interview.',
    airport: 'Role-play: you are an airline check-in agent at the airport.',
    shopping: 'Role-play: you are a shop assistant in a clothing store.',
    smalltalk: 'Role-play: you are a new neighbor making small talk.',
    directions: 'Role-play: you are a local; the learner is a tourist asking for directions.',
  };

  // ---------- Almacenamiento ----------
  const load = (k, d) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } };
  const store = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

  const prefs = Object.assign(
    { scenario: 'free', voice: '', rate: 0.95, explainLang: 'es', listenMode: false, handsFree: false, model: DEFAULT_MODEL },
    load('ebm-prefs', {})
  );
  const savePrefs = () => store('ebm-prefs', prefs);

  const emptyMemory = () => ({ facts: [], mistakes: [], summaries: [], messages: [], level: '', stats: { turns: 0, perfect: 0 } });
  let memory = Object.assign(emptyMemory(), load('ebm-memory', {}));
  const saveMemory = () => store('ebm-memory', memory);

  // ---------- Estado visual ----------
  // ---------- Diagnóstico (Ajustes → Diagnóstico) ----------
  const diag = [];
  function log(msg) {
    const line = new Date().toLocaleTimeString('es', { hour12: false }) + '  ' + msg;
    diag.push(line);
    if (diag.length > 60) diag.shift();
    console.log('[EB]', msg);
    const pre = document.getElementById('diagLog');
    if (pre) pre.textContent = diag.join('\n');
  }
  addEventListener('error', e => log('Error: ' + (e.message || e)));
  addEventListener('unhandledrejection', e => log('Error: ' + (e.reason?.message || e.reason)));

  const STATUS = { idle: 'en línea', listening: 'escuchando…', thinking: 'pensando…', speaking: 'hablando' };
  let state = 'idle', thinkTimer = 0;
  function setState(s) {
    state = s;
    document.body.dataset.state = s;
    $('statusText').textContent = STATUS[s];
    clearInterval(thinkTimer);
    if (s === 'thinking') {
      const t0 = Date.now();
      thinkTimer = setInterval(() => { $('statusText').textContent = `pensando… ${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
    }
  }

  // ---------- Ollama ----------
  const RESPONSE_SCHEMA = {
    type: 'object',
    properties: {
      reply: { type: 'string' },
      corrections: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            original: { type: 'string' },
            corrected: { type: 'string' },
            explanation: { type: 'string' },
            type: { type: 'string', enum: ['grammar', 'vocabulary', 'naturalness', 'pronunciation'] },
          },
          required: ['original', 'corrected', 'explanation', 'type'],
        },
      },
      corrected_sentence: { type: 'string' },
      new_memories: { type: 'array', items: { type: 'string' } },
    },
    required: ['reply', 'corrections', 'corrected_sentence', 'new_memories'],
  };

  // ---------- Motor de IA (WebLLM en un Web Worker, para que el video no se trabe) ----------
  let engine = null, loadedId = null, f16 = false, loading = null;
  let queue = Promise.resolve(); // el motor atiende una petición a la vez
  const modelId = base => `${base}-${f16 ? 'q4f16_1' : 'q4f32_1'}-MLC`;

  function showProgress(p, text) {
    $('progress').hidden = false;
    $('progressBar').style.width = Math.round(p * 100) + '%';
    $('progressText').textContent = text;
  }

  async function ensureEngine() {
    const id = modelId(prefs.model);
    if (engine && loadedId === id) return engine;
    if (loading) return loading;
    loading = (async () => {
      const cached = await webllm.hasModelInCache(id).catch(() => false);
      const t0 = Date.now();
      log(`Cargando modelo ${id} (f16=${f16}, ${cached ? 'ya descargado' : 'descargando'})`);
      const onProgress = r => showProgress(r.progress || 0,
        cached ? `Cargando a Mia en la memoria… ${Math.round((r.progress || 0) * 100)}%`
               : `Descargando el modelo (solo la primera vez, usa Wi-Fi)… ${Math.round((r.progress || 0) * 100)}%`);
      if (!engine) {
        engine = await webllm.CreateWebWorkerMLCEngine(
          new Worker(new URL('worker.js', import.meta.url), { type: 'module' }), id, { initProgressCallback: onProgress });
      } else {
        engine.setInitProgressCallback(onProgress);
        await engine.reload(id);
      }
      loadedId = id;
      log(`Modelo listo en ${Math.round((Date.now() - t0) / 1000)} s`);
      $('progress').hidden = true;
      return engine;
    })();
    try { return await loading; }
    catch (e) { log('Falló la carga del modelo: ' + e.message); throw e; }
    finally { loading = null; }
  }

  function llmChat(messages, { format, onChunk } = {}) {
    const run = async () => {
      const eng = await ensureEngine();
      const req = { messages, stream: true, temperature: 0.5, max_tokens: 600, extra_body: { enable_thinking: false } };
      if (format) req.response_format = { type: 'json_object', schema: JSON.stringify(format) };
      let text = '', first = 0, tokens = 0;
      const t0 = Date.now();
      log(`Pregunta enviada a la IA (${messages.length} mensajes)`);
      // Si en 2 minutos no llega nada, se corta para no quedarse esperando para siempre.
      let timedOut = false;
      const watchdog = setTimeout(() => { timedOut = true; eng.interruptGenerate(); }, 120000);
      try {
        for await (const chunk of await eng.chat.completions.create(req)) {
          const d = chunk.choices[0]?.delta?.content;
          if (d) {
            if (!first) { first = Date.now() - t0; clearTimeout(watchdog); log(`Primera palabra a los ${(first / 1000).toFixed(1)} s`); }
            tokens++;
            text += d; onChunk?.(text);
          }
        }
      } finally { clearTimeout(watchdog); }
      if (timedOut && !text) throw new Error('La IA tardó demasiado en responder. Prueba un modelo más liviano en Ajustes.');
      const secs = (Date.now() - t0 - first) / 1000;
      log(`Respuesta completa: ${tokens} piezas en ${((Date.now() - t0) / 1000).toFixed(1)} s (${secs > 0 ? (tokens / secs).toFixed(1) : '?'} por segundo)`);
      return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    };
    const p = queue.then(run, run);
    queue = p.catch(() => {});
    return p;
  }

  function systemPrompt() {
    const facts = memory.facts.length ? memory.facts.map(f => `- ${f}`).join('\n') : '(nothing yet)';
    const summaries = memory.summaries.slice(-5).map(s => `- ${s.text}`).join('\n') || '(none)';
    const mistakes = [...memory.mistakes].sort((a, b) => b.count - a.count).slice(0, 5)
      .map(m => `- "${m.original}" -> "${m.corrected}"`).join('\n') || '(none)';
    const lang = prefs.explainLang === 'en' ? 'simple English' : 'Spanish';
    return `You are Mia, a friendly English conversation partner for a Spanish speaker who practices everyday spoken English.
Scenario: ${SCENARIOS[prefs.scenario] || SCENARIOS.free}

Reply with a JSON object:
- "reply": what you say next, in natural spoken English. 1 or 2 short sentences. End with a question. No emojis, no lists.
- "corrections": the real mistakes in the learner's LAST message (grammar, wrong word, unnatural phrase). Use an empty list [] if the message is correct. Ignore punctuation and capital letters. Each item: "original" (the wrong words), "corrected", "explanation" (one short sentence written in ${lang}${prefs.explainLang === 'en' ? '' : ' / en español'}), "type" (grammar, vocabulary, naturalness or pronunciation).
- "corrected_sentence": the learner's whole last message said correctly.
- "new_memories": facts the learner clearly said about themself in their LAST message, as short sentences like "His name is Luis." or "He works as an engineer." Never guess, never use placeholders. Empty list if none.

What you remember about the learner:
${facts}
Previous conversations:
${summaries}
Mistakes they repeat:
${mistakes}`;
  }

  function contextMessages() {
    const out = [];
    for (const m of memory.messages.slice(-10)) {
      const last = out[out.length - 1];
      if (last && last.role === m.role) last.content += '\n' + m.text;
      else out.push({ role: m.role, content: m.text });
    }
    if (out.length && out[0].role !== 'user') out.unshift({ role: 'user', content: '(start)' });
    return out;
  }

  // Saca el texto de "reply" en cuanto el modelo termina de escribirlo, para que Mia hable antes.
  function earlyReply(partial) {
    const m = /"reply"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(partial);
    if (!m) return null;
    try { return JSON.parse('"' + m[1] + '"'); } catch { return null; }
  }

  function parseResult(text) {
    let r;
    try { r = JSON.parse(text); } catch { r = { reply: earlyReply(text) || text, corrections: [], corrected_sentence: '', new_memories: [] }; }
    r.corrections = (Array.isArray(r.corrections) ? r.corrections : [])
      .filter(c => c && c.original && c.corrected && c.original.trim().toLowerCase() !== c.corrected.trim().toLowerCase());
    r.new_memories = Array.isArray(r.new_memories) ? r.new_memories : [];
    return r;
  }

  function recordMistakes(corrections) {
    for (const c of corrections) {
      const key = c.original.trim().toLowerCase();
      const ex = memory.mistakes.find(m => m.original.trim().toLowerCase() === key);
      if (ex) { ex.count++; ex.corrected = c.corrected; ex.lastSeen = new Date().toISOString(); }
      else memory.mistakes.push({ ...c, count: 1, lastSeen: new Date().toISOString() });
    }
    if (memory.mistakes.length > 150) {
      memory.mistakes.sort((a, b) => b.count - a.count);
      memory.mistakes.length = 150;
    }
  }
  function recordFacts(list) {
    for (const f of list) {
      const t = String(f).trim();
      if (/[()]|to be filled|unknown|not (yet )?(provided|mentioned)/i.test(t)) continue;
      if (t && !memory.facts.some(x => x.toLowerCase() === t.toLowerCase())) memory.facts.push(t);
    }
    if (memory.facts.length > 60) memory.facts = memory.facts.slice(-60);
  }

  let summarizing = false;
  async function summarizeOld() {
    if (summarizing || memory.messages.length <= 30) return;
    summarizing = true;
    const old = memory.messages.slice(0, 16);
    const transcript = old.map(m => `${m.role === 'user' ? 'Learner' : 'Mia'}: ${m.text}`).join('\n');
    try {
      const text = await llmChat([{ role: 'user', content: `Summarize this conversation in 2 sentences: the topics and anything personal the learner shared.\n\n${transcript}` }]);
      memory.summaries.push({ date: (old[0].at || new Date().toISOString()).slice(0, 10), text: text.trim() });
      if (memory.summaries.length > 40) memory.summaries = memory.summaries.slice(-40);
      memory.messages = memory.messages.slice(16);
      saveMemory();
    } catch { /* se intenta otra vez más tarde */ }
    summarizing = false;
  }

  // ---------- Render ----------
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  const scrollDown = () => requestAnimationFrame(() => { chatEl.scrollTop = chatEl.scrollHeight; });

  function addMia(text) {
    const wrap = el('div', 'msg mia');
    const b = el('div', 'bubble', text);
    if (prefs.listenMode) {
      b.classList.add('hidden-text');
      b.addEventListener('click', () => b.classList.remove('hidden-text'), { once: true });
    }
    wrap.append(b);
    chatEl.append(wrap);
    scrollDown();
    return wrap;
  }
  function addMe(text, corrections, corrected) {
    const wrap = el('div', 'msg me');
    wrap.append(el('div', 'bubble', text));
    if (corrections) wrap.append(renderCorrections(corrections, corrected, text));
    chatEl.append(wrap);
    scrollDown();
    return wrap;
  }
  // Compara frases sin mayúsculas ni puntuación (el micrófono no las transcribe bien).
  const norm = t => String(t || '').toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const differs = (a, b) => b && norm(a) !== norm(b);

  function renderCorrections(list, corrected, original) {
    if (!list.length && !differs(original, corrected)) return el('div', 'ok', '✓ ¡Perfecto! Lo dijiste bien.');
    const box = el('div', 'fixes');
    box.append(el('div', 'label', list.length === 0 ? 'Se puede decir mejor' : list.length === 1 ? '1 corrección' : `${list.length} correcciones`));
    for (const c of list) {
      const f = el('div', 'fix');
      const pair = el('div', 'pair');
      pair.append(el('del', '', c.original), document.createTextNode(' → '), el('ins', '', c.corrected));
      if (KIND_NAMES[c.type]) pair.append(el('span', 'kind', KIND_NAMES[c.type]));
      f.append(pair);
      if (c.explanation) f.append(el('div', 'why', c.explanation));
      box.append(f);
    }
    if (differs(original, corrected)) {
      const better = el('div', 'better');
      const p = el('p');
      p.append(el('small', '', 'Mejor dicho'), document.createTextNode(corrected));
      const btn = el('button', 'say', 'Escuchar');
      btn.type = 'button';
      btn.addEventListener('click', () => speak(corrected, { slow: true }));
      better.append(p, btn);
      box.append(better);
    }
    return box;
  }
  function addTyping() {
    const wrap = el('div', 'msg mia');
    const b = el('div', 'bubble typing');
    b.innerHTML = '<i></i><i></i><i></i>';
    wrap.append(b);
    chatEl.append(wrap);
    scrollDown();
    return wrap;
  }
  function addError(msg) { chatEl.append(el('div', 'error', msg)); scrollDown(); }

  // ---------- Voz de Mia ----------
  let voices = [];
  function loadVoices() {
    voices = speechSynthesis.getVoices().filter(v => /^en[-_]/i.test(v.lang));
    const sel = $('voiceSel');
    sel.innerHTML = '';
    const auto = el('option', '', 'Automática');
    auto.value = '';
    sel.append(auto);
    for (const v of voices) {
      const o = el('option', '', `${v.name} (${v.lang})`);
      o.value = v.name;
      sel.append(o);
    }
    sel.value = prefs.voice;
  }
  function pickVoice() {
    if (prefs.voice) { const v = voices.find(v => v.name === prefs.voice); if (v) return v; }
    return voices.find(v => /en[-_]US/i.test(v.lang) && /female|samantha|aria|jenny/i.test(v.name))
      || voices.find(v => /en[-_]US/i.test(v.lang)) || voices[0] || null;
  }
  if ('speechSynthesis' in window) { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }

  let lastMia = '';
  function speak(text, { slow = false, after } = {}) {
    if (!('speechSynthesis' in window) || !text) { after?.(); return; }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const v = pickVoice();
    if (v) { u.voice = v; u.lang = v.lang; } else u.lang = 'en-US';
    u.rate = slow ? Math.max(0.6, prefs.rate - 0.2) : prefs.rate;
    u.onstart = () => { setState('speaking'); mouth.start(); };
    const done = () => { mouth.stop(); if (state === 'speaking') setState('idle'); after?.(); };
    u.onend = done;
    u.onerror = done;
    speechSynthesis.speak(u);
  }

  // ---------- Mia: video en reposo / hablando ----------
  const mouth = (() => {
    const stage = $('stage'), idle = $('idleVid'), talk = $('talkVid');
    let ready = 0;
    const onReady = () => { if (++ready === 2) stage.classList.add('has-video'); };
    [idle, talk].forEach(v => {
      // La librería de IA tarda en cargar: el video puede estar listo antes que este código.
      if (v.readyState >= 3) onReady();
      else v.addEventListener('canplay', onReady, { once: true });
      v.addEventListener('error', () => stage.classList.remove('has-video'));
    });
    const play = v => v.play().catch(() => {});
    return {
      unlock() { play(idle); play(talk); },
      start() { play(talk); stage.classList.add('talking'); stage.style.setProperty('--talk', '0.6'); },
      stop() { stage.classList.remove('talking'); stage.style.setProperty('--talk', '0'); },
    };
  })();

  // ---------- Reconocimiento de voz ----------
  let rec = null, finalText = '', listening = false;
  function startListening() {
    if (!SR) { addError('Este navegador no reconoce voz. Usa Google Chrome, o escribe con el teclado.'); return; }
    if (listening || state === 'thinking' || busy) return;
    speechSynthesis?.cancel();
    rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = true;
    rec.continuous = false;
    finalText = '';
    rec.onstart = () => { listening = true; setState('listening'); $('interim').hidden = false; $('interim').textContent = ''; };
    let silence = 0;
    rec.onresult = e => {
      clearTimeout(silence);
      let fin = '', interim = '';
      for (let i = 0; i < e.results.length; i++) {
        if (e.results[i].isFinal) fin += e.results[i][0].transcript;
        else interim += e.results[i][0].transcript;
      }
      finalText = fin;
      $('interim').textContent = (fin + ' ' + interim).trim();
      if (fin && !interim) { try { rec.stop(); } catch {} }
      else silence = setTimeout(() => { try { rec.stop(); } catch {} }, 1800);
    };
    rec.onerror = e => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') addError('Permite el micrófono para esta página en Chrome.');
      else if (e.error === 'network') addError('El reconocimiento de voz de Chrome necesita internet. Mientras tanto puedes escribir.');
      else if (e.error !== 'no-speech' && e.error !== 'aborted') addError('Error de micrófono: ' + e.error);
      log('Micrófono: ' + e.error);
    };
    rec.onend = () => {
      clearTimeout(silence);
      listening = false;
      log('Micrófono cerrado: "' + (finalText || $('interim').textContent).trim().slice(0, 60) + '"');
      $('interim').hidden = true;
      const text = (finalText || $('interim').textContent).trim();
      if (state === 'listening') setState('idle');
      if (text) send(text);
    };
    try { rec.start(); } catch {}
  }
  function stopListening() { try { rec?.stop(); } catch {} }

  // ---------- Conversación ----------
  let busy = false;
  async function send(text, { greet = false } = {}) {
    if (busy) {
      log('Mensaje ignorado: la IA sigue ocupada');
      addError('Mia todavía está pensando la respuesta anterior. Espera un momento.');
      return;
    }
    busy = true;
    document.body.dataset.busy = '1';
    setState('thinking');
    const mine = greet ? null : addMe(text);
    const typing = addTyping();
    const messages = [{ role: 'system', content: systemPrompt() }, ...contextMessages()];
    messages.push({
      role: 'user',
      content: greet
        ? `(The learner just opened the app. Greet them in one short sentence${memory.facts.length ? ' that mentions something you remember about them' : ' and ask their name'}, then start the scenario. corrections must be [] and corrected_sentence "".)`
        : text,
    });

    let spokenEarly = null, miaBubble = null;
    try {
      const raw = await llmChat(messages, {
        format: RESPONSE_SCHEMA,
        onChunk: partial => {
          if (spokenEarly) return;
          const reply = earlyReply(partial);
          if (!reply) return;
          spokenEarly = reply;
          typing.remove();
          miaBubble = addMia(reply);
          setState('idle');
          speak(reply, { after: () => { if (prefs.handsFree && state === 'idle') setTimeout(startListening, 300); } });
        },
      });
      const r = parseResult(raw);
      if (!spokenEarly) {
        typing.remove();
        setState('idle');
        addMia(r.reply);
        speak(r.reply, { after: () => { if (prefs.handsFree) setTimeout(startListening, 300); } });
      }
      lastMia = r.reply;
      if (mine) mine.append(renderCorrections(r.corrections, r.corrected_sentence, text));
      scrollDown();

      const now = new Date().toISOString();
      if (!greet) {
        memory.messages.push({ role: 'user', text, at: now, corrections: r.corrections, corrected: r.corrected_sentence });
        memory.stats.turns++;
        if (!r.corrections.length && !differs(text, r.corrected_sentence)) memory.stats.perfect++;
        recordMistakes(r.corrections);
      }
      memory.messages.push({ role: 'assistant', text: r.reply, at: now });
      recordFacts(r.new_memories);
      saveMemory();
      renderMemory();
      setTimeout(summarizeOld, 4000);
    } catch (e) {
      typing.remove();
      setState('idle');
      log('Error al responder: ' + e.message);
      addError(e.message);
    } finally {
      busy = false;
      delete document.body.dataset.busy;
    }
  }

  // ---------- Panel de memoria ----------
  function renderMemory() {
    const s = memory.stats;
    const pct = s.turns ? Math.round((s.perfect / s.turns) * 100) : 0;
    $('statsLine').textContent = `${s.turns} frases dichas · ${pct}% sin errores`;
    const fl = $('factsList'); fl.innerHTML = '';
    if (!memory.facts.length) fl.append(el('li', 'empty', 'Todavía nada. Cuéntale a Mia sobre ti.'));
    for (const f of memory.facts) {
      const li = el('li'); li.append(el('span', '', f));
      const b = el('button', '', 'Olvidar'); b.type = 'button';
      b.addEventListener('click', () => { memory.facts = memory.facts.filter(x => x !== f); saveMemory(); renderMemory(); });
      li.append(b); fl.append(li);
    }
    const ml = $('mistakesList'); ml.innerHTML = '';
    if (!memory.mistakes.length) ml.append(el('li', 'empty', 'Sin errores registrados aún.'));
    for (const m of [...memory.mistakes].sort((a, b) => b.count - a.count).slice(0, 15)) {
      const li = el('li');
      li.append(el('del', '', m.original), document.createTextNode(' → '), el('ins', '', m.corrected), el('span', 'count', `  ×${m.count}`));
      ml.append(li);
    }
    const sl = $('summariesList'); sl.innerHTML = '';
    if (!memory.summaries.length) sl.append(el('li', 'empty', 'Aparecerán aquí cuando lleves un rato conversando.'));
    for (const x of [...memory.summaries].reverse()) {
      const li = el('li'); li.append(el('b', '', x.date + ' · '), document.createTextNode(x.text)); sl.append(li);
    }
  }

  // ---------- Modelos ----------
  async function checkGPU() {
    const line = $('modelLine');
    const sel = $('modelSel');
    sel.innerHTML = '';
    for (const m of MODELS) { const o = el('option', '', m.label); o.value = m.base; sel.append(o); }
    if (!MODELS.some(m => m.base === prefs.model)) prefs.model = DEFAULT_MODEL;
    sel.value = prefs.model;
    const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
    if (!adapter) {
      line.textContent = 'Este navegador no tiene WebGPU. Usa Google Chrome actualizado en Android.';
      $('startBtn').disabled = true;
      return false;
    }
    f16 = adapter.features.has('shader-f16');
    log(`GPU: ${adapter.info?.vendor || '?'} ${adapter.info?.architecture || ''} · f16=${f16} · ${navigator.userAgent.match(/Chrome\/[\d.]+/)?.[0] || ''}`);
    navigator.storage?.persist?.().catch(() => {}); // que Chrome no borre el modelo descargado
    const cached = await webllm.hasModelInCache(modelId(prefs.model)).catch(() => false);
    line.textContent = cached
      ? 'Mia ya está en tu celular. Funciona sin conexión.'
      : 'La primera vez se descarga el modelo de IA (1 a 2.5 GB). Conéctate a Wi-Fi.';
    line.classList.toggle('ok', cached);
    return true;
  }

  // ---------- Controles ----------
  $('micBtn').addEventListener('click', () => {
    if (listening) stopListening();
    else if (state === 'speaking') { speechSynthesis.cancel(); startListening(); }
    else startListening();
  });
  $('replayBtn').addEventListener('click', () => lastMia && speak(lastMia));
  $('keyboardBtn').addEventListener('click', () => {
    const tb = $('typebar'); tb.hidden = !tb.hidden;
    if (!tb.hidden) $('textInput').focus();
  });
  $('typebar').addEventListener('submit', e => {
    e.preventDefault();
    const v = $('textInput').value.trim();
    if (!v || busy) return;
    $('textInput').value = '';
    send(v);
  });

  const dlg = $('settings');
  function openSettings() { renderMemory(); dlg.showModal(); }
  $('settingsBtn').addEventListener('click', openSettings);
  $('scenarioChip').addEventListener('click', openSettings);
  $('closeSettings').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });

  function applyPrefsToUI() {
    $('scenarioSel').value = prefs.scenario;
    $('scenarioChip').textContent = SCENARIO_NAMES[prefs.scenario] || 'Charla libre';
    $('rateRange').value = prefs.rate;
    $('rateVal').textContent = Number(prefs.rate).toFixed(2) + '×';
    $('explainSel').value = prefs.explainLang;
    $('listenMode').checked = prefs.listenMode;
    $('handsFree').checked = prefs.handsFree;
  }
  $('scenarioSel').addEventListener('change', e => {
    prefs.scenario = e.target.value; savePrefs(); applyPrefsToUI();
    dlg.close();
    send('', { greet: true });
  });
  $('modelSel').addEventListener('change', async e => {
    prefs.model = e.target.value; savePrefs();
    dlg.close();
    setState('thinking');
    try { await ensureEngine(); addMia('Model changed. Shall we keep talking?'); }
    catch (err) { addError('No se pudo cargar ese modelo: ' + err.message); }
    setState('idle');
  });
  $('voiceSel').addEventListener('change', e => { prefs.voice = e.target.value; savePrefs(); speak('Hi! This is my new voice.'); });
  $('rateRange').addEventListener('input', e => { prefs.rate = Number(e.target.value); savePrefs(); applyPrefsToUI(); });
  $('rateRange').addEventListener('change', () => speak('How does this speed sound to you?'));
  $('explainSel').addEventListener('change', e => { prefs.explainLang = e.target.value; savePrefs(); });
  $('listenMode').addEventListener('change', e => { prefs.listenMode = e.target.checked; savePrefs(); });
  $('handsFree').addEventListener('change', e => { prefs.handsFree = e.target.checked; savePrefs(); });

  // Copia de seguridad (el mismo formato que data/memory.json de la versión de PC)
  function backupMsg(t) { const m = $('backupMsg'); m.hidden = false; m.textContent = t; }
  $('exportBtn').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(memory, null, 2)], { type: 'application/json' });
    const a = el('a');
    a.href = URL.createObjectURL(blob);
    a.download = `mia-memoria-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    backupMsg('Copia guardada en Descargas.');
  });
  $('importInput').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (!Array.isArray(data.messages) || !Array.isArray(data.facts)) throw new Error();
      memory = Object.assign(emptyMemory(), data);
      saveMemory();
      chatEl.innerHTML = '';
      renderHistory();
      renderMemory();
      backupMsg(`Memoria cargada: ${memory.facts.length} datos y ${memory.messages.length} mensajes.`);
    } catch {
      backupMsg('Ese archivo no es una copia de la memoria de Mia.');
    }
    e.target.value = '';
  });

  $('copyDiag').addEventListener('click', async () => {
    const text = diag.join('\n');
    try { await navigator.clipboard.writeText(text); $('copyDiag').textContent = 'Copiado'; }
    catch { const r = document.createRange(); r.selectNodeContents($('diagLog')); getSelection().removeAllRanges(); getSelection().addRange(r); }
  });

  $('wipeBtn').addEventListener('click', () => { $('wipeConfirm').hidden = false; });
  $('wipeConfirm').addEventListener('click', () => {
    memory = emptyMemory(); saveMemory();
    $('wipeConfirm').hidden = true;
    chatEl.innerHTML = '';
    renderMemory();
  });

  // ---------- Inicio ----------
  function renderHistory() {
    for (const m of memory.messages.slice(-30)) {
      if (m.role === 'user') addMe(m.text, m.corrections || [], m.corrected);
      else { addMia(m.text); lastMia = m.text; }
    }
  }
  $('startBtn').addEventListener('click', async () => {
    if ('speechSynthesis' in window) speechSynthesis.speak(new SpeechSynthesisUtterance(''));
    mouth.unlock();
    $('startBtn').disabled = true;
    try {
      await ensureEngine();
    } catch (err) {
      $('progress').hidden = true;
      $('startBtn').disabled = false;
      const w = $('splashWarn'); w.hidden = false;
      w.textContent = /memory|device lost|out of/i.test(err.message)
        ? 'El celular no tiene memoria suficiente para este modelo. Elige uno más liviano en Ajustes.'
        : 'No se pudo cargar la IA: ' + err.message;
      return;
    }
    $('splash').hidden = true;
    log('App lista');
    const last = memory.messages[memory.messages.length - 1];
    const stale = !last || Date.now() - new Date(last.at).getTime() > 3 * 3600e3;
    if (stale) send('', { greet: true });
  });

  applyPrefsToUI();
  renderHistory();
  renderMemory();
  checkGPU();
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
})();
