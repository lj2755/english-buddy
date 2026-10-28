// English Buddy — todo corre en el celular: la IA en la GPU con WebLLM (WebGPU),
// la voz en Chrome y la memoria en el almacenamiento del navegador.
import * as webllm from 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/+esm';

(() => {
  const $ = id => document.getElementById(id);
  const chatEl = $('chat');
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  // Modelos que caben en un celular. El sufijo depende de si la GPU soporta números de 16 bits.
  const MODELS = [
    { base: 'gemma-4-E2B-it', label: 'Gemma 4 E2B · recomendado' },
    { base: 'Qwen3-4B', label: 'Qwen3 4B · más rápido' },
    { base: 'Qwen3.5-4B', label: 'Qwen3.5 4B · más nuevo' },
    { base: 'Qwen3.5-2B', label: 'Qwen3.5 2B · más rápido' },
    { base: 'Qwen3-1.7B', label: 'Qwen3 1.7B · el más liviano' },
  ];
  // Gemma 4 E2B fue el que mejor conversó en las pruebas.
  const DEFAULT_MODEL = 'gemma-4-E2B-it';

  // ---------- Almacenamiento ----------
  const load = (k, d) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } };
  const store = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

  const prefs = Object.assign(
    { voice: '', rate: 0.95, handsFree: false, model: DEFAULT_MODEL },
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

  // ---------- Formato de respuesta ----------
  const RESPONSE_SCHEMA = {
    type: 'object',
    properties: {
      reply: { type: 'string' },
      new_memories: { type: 'array', items: { type: 'string' } },
    },
    required: ['reply', 'new_memories'],
  };

  // ---------- Motor de IA (WebLLM en un Web Worker, para que el video no se trabe) ----------
  let engine = null, loadedId = null, f16 = false, loading = null;
  // El doble de la memoria de trabajo por defecto (4096): la conversación dura ~35 turnos sin reiniciarse.
  const CHAT_OPTS = { context_window_size: 8192 };
  // Respaldos de esta sesión (no se guardan: al abrir la app se vuelve a la configuración normal).
  const session = { noGrammar: false, noWorker: false };
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
      if (!engine && session.noWorker) {
        engine = await webllm.CreateMLCEngine(id, { initProgressCallback: onProgress }, CHAT_OPTS);
      } else if (!engine) {
        engine = await webllm.CreateWebWorkerMLCEngine(
          new Worker(new URL('worker.js', import.meta.url), { type: 'module' }), id, { initProgressCallback: onProgress }, CHAT_OPTS);
      } else {
        engine.setInitProgressCallback(onProgress);
        await engine.reload(id, CHAT_OPTS);
      }
      loadedId = id;
      log(`Modelo listo en ${Math.round((Date.now() - t0) / 1000)} s`);
      $('progress').hidden = true;
      return engine;
    })();
    try { return await loading; }
    catch (e) { log('Falló la carga del modelo: ' + errText(e)); throw e; }
    finally { loading = null; }
  }

  // Describe cualquier error, aunque no traiga mensaje (los del worker a veces llegan vacíos).
  function errText(e) {
    if (!e) return 'error desconocido';
    if (typeof e === 'string') return e;
    const parts = [e.name, e.message, e.code].filter(Boolean);
    if (parts.length) return parts.join(': ');
    try { return JSON.stringify(e) !== '{}' ? JSON.stringify(e) : String(e); } catch { return String(e); }
  }

  // ¿Parece una respuesta de verdad? (con números f16 rotos el modelo escribe basura)
  function looksValid(text, format) {
    let reply = text;
    if (format) {
      try { reply = JSON.parse(text.match(/\{[\s\S]*\}/)[0]).reply; } catch { reply = earlyReply(text); }
    }
    return typeof reply === 'string' && /[a-z]{2,}.*[a-z]{2,}/i.test(reply) && new Set(reply.toLowerCase()).size > 6;
  }

  let lastRaw = ''; // texto exacto generado en la última respuesta (para reutilizar la conversación)

  // Una sola generación con la configuración actual.
  async function generate(messages, { format, onChunk }) {
    const eng = await ensureEngine();
    const grammar = format && !session.noGrammar;
    const msgs = messages.map(m => ({ ...m }));
    if (format && !grammar) msgs[0].content += '\nAnswer ONLY with that JSON object, nothing before or after it.';
    const req = { messages: msgs, stream: true, stream_options: { include_usage: true }, temperature: 0.5, max_tokens: 400 };
    // Solo los modelos Qwen tienen "modo pensar"; apagarlo los hace responder mucho más rápido.
    if (/^Qwen/i.test(prefs.model)) req.extra_body = { enable_thinking: false };
    if (grammar) req.response_format = { type: 'json_object', schema: JSON.stringify(format) };
    let text = '', first = 0, tokens = 0, finish = '', usage = null;
    const t0 = Date.now();
    log(`Pregunta enviada (${msgs.length} mensajes, ${grammar ? 'con' : 'sin'} formato fijo, f16=${f16}, ${session.noWorker ? 'en la página' : 'en worker'})`);
    // Si en 2 minutos no llega nada, se corta para no quedarse esperando para siempre.
    let timedOut = false;
    const watchdog = setTimeout(() => { timedOut = true; eng.interruptGenerate(); }, 120000);
    try {
      for await (const chunk of await eng.chat.completions.create(req)) {
        if (chunk.usage) usage = chunk.usage;
        if (chunk.choices[0]?.finish_reason) finish = chunk.choices[0].finish_reason;
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
    const gen = usage?.completion_tokens ?? tokens;
    log(`Respuesta completa: ${gen} palabras (${tokens} piezas) en ${((Date.now() - t0) / 1000).toFixed(1)} s · ${secs > 0 ? (gen / secs).toFixed(1) : '?'} por segundo · fin: ${finish || '?'} · entrada: ${usage?.prompt_tokens ?? '?'}`);
    lastRaw = text;
    const clean = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    if (!looksValid(clean, format)) {
      log('Respuesta sin sentido. Texto crudo: ' + JSON.stringify(text.slice(0, 160)));
      throw new Error('respuesta sin sentido');
    }
    return clean;
  }

  // Si falla, prueba configuraciones más compatibles y recuerda la que funcionó:
  // 1) sin formato fijo (la gramática JSON falla en algunas GPU de Android), 2) sin f16.
  function llmChat(messages, opts = {}) {
    const run = async () => {
      for (let attempt = 0; ; attempt++) {
        try {
          return await generate(messages, opts);
        } catch (e) {
          if (attempt >= 4) throw new Error('La IA del celular no logró responder. Prueba otro modelo en Ajustes → Modelo de IA (por ejemplo Qwen3.5 2B).');
          log('Falló la generación: ' + errText(e) + (e?.stack ? ' | ' + String(e.stack).split('\n')[1]?.trim() : ''));
          if (/tardó demasiado/.test(e?.message || '')) throw e;
          const msg = errText(e);
          // Conversación demasiado larga para el modelo: se recortan los mensajes viejos.
          if (/context window|prompt tokens/i.test(msg) && messages.length > 3) {
            messages = [messages[0], ...messages.slice(-3)];
            if (messages[1].role !== 'user') messages.splice(1, 1);
            log('Conversación muy larga: uso solo los últimos mensajes');
            continue;
          }
          loadedId = null; // el motor puede quedar dañado tras un error: se recarga
          if (f16 && /sin sentido|Grammar matcher|NaN/i.test(msg)) {
            prefs.forceF32 = true; f16 = false; savePrefs();
            log('Los números f16 de esta GPU fallan: cambio al modelo sin f16');
            continue;
          }
          if (/memory|out of|device (was )?lost|allocation/i.test(msg) && prefs.model !== 'Qwen3.5-2B') {
            prefs.model = 'Qwen3.5-2B'; savePrefs();
            $('modelSel').value = prefs.model;
            log('Falta memoria: cambio a Qwen3.5 2B');
            continue;
          }
          if (opts.format && !session.noGrammar) {
            session.noGrammar = true;
            log('Reintentando sin formato fijo');
            continue;
          }
          if (!session.noWorker) {
            session.noWorker = true;
            log('Reintentando con la IA en la página (sin proceso aparte)');
            try { await engine?.unload(); } catch {}
            engine = null;
            continue;
          }
          if (f16 && !prefs.forceF32) {
            prefs.forceF32 = true; f16 = false; savePrefs();
            session.noGrammar = false; // vuelve a probar el formato con el modelo f32
            log('Reintentando con el modelo sin f16 (puede descargar el modelo otra vez)');
            continue;
          }
          throw new Error('La IA del celular no logró responder (' + msg + '). Prueba otro modelo en Ajustes → Modelo de IA.');
        }
      }
    };
    const p = queue.then(run, run);
    queue = p.catch(() => {});
    return p;
  }

  function systemPrompt() {
    const facts = memory.facts.length ? memory.facts.slice(-25).map(f => `- ${f}`).join('\n') : '(nothing yet)';
    return `You are Mia, your friend's good friend. Your name is Mia. Never say you are an AI, a model, Gemma or Qwen.
About you: warm, curious and positive. You like music, movies, cooking and walks in the park. You share your opinions and small things about your day, like a friend does.

How you talk:
- Like a good friend: warm, relaxed and natural. Everyday English. No emojis.
- Plain text only: never use asterisks, markdown or any formatting symbols.
- Usually one to three sentences. Ask a question when it feels natural, so the conversation keeps going.
- Never correct your friend's English or comment on their grammar or pronunciation. Just chat.
- Never invent details about your friend or assume things they did not say.
- Sometimes your friend's message ends with a [Note for Mia ...]. It contains real information (date, weather, Wikipedia) or a message from the app. Use it in simple words. It was not written by your friend. If there is no note, do not claim to know today's date, the weather or recent news.

Your friend's messages come from a microphone, so they may be cut off or wrong. If a message is unclear or makes no sense, kindly ask them to say it again.

Reply with a JSON object:
- "reply": what you say next.
- "new_memories": facts your friend clearly said about themself in their LAST message, as short sentences like "His name is Luis." Never guess. [] if none.

What you remember about your friend:
${facts}`;
  }

  // La IA solo reutiliza lo que ya leyó si cada pregunta repite exactamente la anterior más el
  // mensaje nuevo. Por eso, durante la sesión, las instrucciones quedan fijas y el historial guarda
  // el texto completo que generó la IA. Así no tiene que releer todo en cada turno.
  let chatSession = null;
  function resetSession() { chatSession = null; }
  function sessionMessages() {
    if (!chatSession) chatSession = { system: systemPrompt(), msgs: contextMessages() };
    return [{ role: 'system', content: chatSession.system }, ...chatSession.msgs];
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

  // El modelo a veces se presenta con su nombre de fábrica (Mia siempre es Mia) o copia en su
  // respuesta las notas internas de la app: esas notas nunca se muestran ni se leen.
  const asMia = t => String(t || '')
    .replace(/\[Note for Mia[^\]]*\]?/gi, '')
    .replace(/\b(?:Gemma|Qwen)(?:[\s-]*[\d.]+[a-z]?)*(?:[\s-]*(?:E2B|it))?\b/gi, 'Mia')
    .replace(/\s+([,.!?])/g, '$1').replace(/,\s*,/g, ',').replace(/\s{2,}/g, ' ').trim();

  function parseResult(text) {
    let r;
    try { r = JSON.parse(text); }
    catch {
      try { r = JSON.parse(text.match(/\{[\s\S]*\}/)[0]); }
      catch { r = { reply: earlyReply(text) || text, new_memories: [] }; }
    }
    r.new_memories = Array.isArray(r.new_memories) ? r.new_memories : [];
    r.reply = asMia(r.reply);
    return r;
  }

  function recordFacts(list) {
    for (const f of list) {
      const t = String(f).trim();
      if (/[()]|to be filled|unknown|not (yet )?(provided|mentioned)/i.test(t)) continue;
      if (t && !memory.facts.some(x => x.toLowerCase() === t.toLowerCase())) memory.facts.push(t);
    }
    if (memory.facts.length > 60) memory.facts = memory.facts.slice(-60);
  }

  // Los mensajes viejos se recortan sin pedirle un resumen a la IA: ese resumen borraba lo que la
  // IA ya había leído y el turno siguiente tardaba ~25 s. Lo que Mia sabe de ti se conserva.
  function summarizeOld() {
    if (memory.messages.length > 60) { memory.messages = memory.messages.slice(-40); saveMemory(); }
  }

  // ---------- Render ----------
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  const scrollDown = () => requestAnimationFrame(() => { chatEl.scrollTop = chatEl.scrollHeight; });

  // Quita las marcas de formato (*palabra*, **palabra**, _palabra_, `palabra`, # títulos)
  // para que no se muestren ni se lean en voz alta.
  const plain = t => String(t || '')
    .replace(/[*_`~]+/g, '')
    .replace(/^\s*#+\s*/gm, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

  function addMia(text) {
    text = plain(text);
    const wrap = el('div', 'msg mia');
    wrap.append(el('div', 'bubble', text));
    chatEl.append(wrap);
    scrollDown();
    return wrap;
  }
  function addMe(text) {
    const wrap = el('div', 'msg me');
    wrap.append(el('div', 'bubble', text));
    chatEl.append(wrap);
    scrollDown();
    return wrap;
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
  const timing = { lastWord: 0, micClosed: 0, sent: 0 }; // para medir dónde se va el tiempo
  function speak(text, { slow = false, after } = {}) {
    text = plain(text);
    if (!('speechSynthesis' in window) || !text) { after?.(); return; }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const v = pickVoice();
    if (v) { u.voice = v; u.lang = v.lang; } else u.lang = 'en-US';
    u.rate = slow ? Math.max(0.6, prefs.rate - 0.2) : prefs.rate;
    u.onstart = () => {
      setState('speaking'); mouth.start();
      if (timing.sent) {
        const now = performance.now();
        log(`Mia empezó a hablar: ${((now - timing.sent) / 1000).toFixed(1)} s después de enviar tu frase` +
          (timing.lastWord ? ` (${((now - timing.lastWord) / 1000).toFixed(1)} s desde tu última palabra)` : '') +
          ` · voz: ${u.voice?.name || 'automática'}${u.voice && !u.voice.localService ? ' (por internet)' : ''}`);
        timing.sent = 0;
      }
    };
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
      else silence = setTimeout(() => { try { rec.stop(); } catch {} }, 1200);
      timing.lastWord = performance.now();
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
      timing.micClosed = performance.now();
      log('Micrófono cerrado: "' + (finalText || $('interim').textContent).trim().slice(0, 60) + '"' +
        (timing.lastWord ? ` (${((timing.micClosed - timing.lastWord) / 1000).toFixed(1)} s después de tu última palabra)` : ''));
      $('interim').hidden = true;
      const text = (finalText || $('interim').textContent).trim();
      if (state === 'listening') setState('idle');
      if (text) send(text);
    };
    try { rec.start(); } catch {}
  }
  function stopListening() { try { rec?.stop(); } catch {} }

  // ---------- Información de internet (fecha y hora, clima, Wikipedia) ----------
  // La app detecta estas preguntas con reglas simples y le pasa el dato a Mia junto con la frase.
  // Un modelo pequeño no decide bien cuándo buscar, por eso no se le deja esa decisión.
  const WEATHER_WORDS = {
    0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'cloudy', 45: 'foggy', 48: 'foggy',
    51: 'light drizzle', 53: 'drizzle', 55: 'heavy drizzle', 61: 'light rain', 63: 'rain', 65: 'heavy rain',
    66: 'freezing rain', 67: 'freezing rain', 71: 'light snow', 73: 'snow', 75: 'heavy snow', 77: 'snow',
    80: 'rain showers', 81: 'rain showers', 82: 'heavy rain showers', 85: 'snow showers', 86: 'snow showers',
    95: 'thunderstorms', 96: 'thunderstorms with hail', 99: 'thunderstorms with hail',
  };

  async function getJSON(url) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 6000);
    try {
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return await res.json();
    } finally { clearTimeout(t); }
  }

  function dateTimeInfo() {
    const now = new Date();
    const date = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    const time = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    return { note: `Today is ${date}. The time is ${time}.`, source: 'tu dispositivo' };
  }

  async function weatherInfo(text) {
    const m = /\b(?:in|at|for)\s+([a-zà-ÿ][a-zà-ÿ .'-]*?)(?:\s+(?:today|tomorrow|now|right now|this week|please))?\s*[?.!]*$/i.exec(text);
    const city = m?.[1]?.trim() || prefs.lastCity;
    if (!city) return { note: 'The learner did not say which city. Ask politely which city they mean.', source: null };
    const geo = await getJSON(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en`);
    const place = geo.results?.[0];
    if (!place) return { note: `No city called "${city}" was found. Ask the learner to repeat the city name.`, source: null };
    prefs.lastCity = place.name; savePrefs();
    const w = await getJSON(`https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}` +
      '&current=temperature_2m,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,weather_code&timezone=auto&forecast_days=2');
    const c = w.current, d = w.daily;
    return {
      note: `Weather in ${place.name}, ${place.country}: now ${Math.round(c.temperature_2m)} °C and ${WEATHER_WORDS[c.weather_code] || 'variable'}, wind ${Math.round(c.wind_speed_10m)} km/h. ` +
        `Today between ${Math.round(d.temperature_2m_min[0])} and ${Math.round(d.temperature_2m_max[0])} °C. ` +
        `Tomorrow ${WEATHER_WORDS[d.weather_code[1]] || 'variable'}, between ${Math.round(d.temperature_2m_min[1])} and ${Math.round(d.temperature_2m_max[1])} °C.`,
      source: 'Open-Meteo',
    };
  }

  async function wikiInfo(subject) {
    const q = await getJSON(`https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(subject)}&srlimit=1&format=json&origin=*`);
    const title = q.query?.search?.[0]?.title;
    if (!title) return { note: `Nothing was found about "${subject}". Say politely that you could not find information about it.`, source: null };
    const sum = await getJSON(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`);
    const extract = (sum.extract || '').split(/(?<=\.)\s+/).slice(0, 3).join(' ').slice(0, 500);
    return { note: `From Wikipedia, "${title}": ${extract}`, source: 'Wikipedia' };
  }

  async function lookupOnline(text) {
    const t = text.toLowerCase().trim();
    // El modelo pequeño a veces confunde su nombre con el tuyo: se lo recordamos.
    if (/\b(what('s| is) your name|who are you|your name)\b/.test(t)) {
      return { note: 'Your name is Mia. You are their friend. Say your name is Mia.', source: null };
    }
    if (/\b(what|which)\b.*\b(day|date|time|year|month)\b|\btoday'?s date\b|\bwhat time\b/.test(t) && !/\bweather\b/.test(t)) {
      return dateTimeInfo();
    }
    const asks = /\b(what|what's|how|how's|is it|will it|is going to|tell me|do you know|check)\b/.test(t);
    if (asks && (/\b(weather|temperature|forecast)\b/.test(t) || /\b(is it|will it|is going to)\b.*\b(rain|raining|sunny|cold|hot|snow|snowing)\b/.test(t))) {
      return weatherInfo(text);
    }
    const m = /^(?:hey mia,?\s*|mia,?\s*)?(?:who (?:is|was|were|are)|what (?:is|was|are|were)(?: an?| the)?|tell me (?:something )?about|do you know (?:who|what)? ?(?:is|was)?)\s+(.+?)[?.!]*$/i.exec(text.trim());
    const subject = m?.[1]?.trim();
    if (subject && subject.length > 2 && !/^(you|your|my|me|i|it|this|that|there|here|today|happening|up|wrong|new|going on|the (time|date|day|weather))\b/i.test(subject)) {
      return wikiInfo(subject);
    }
    return null;
  }

  // ---------- Editar lo que dijiste ----------
  function addEditButton(wrap) {
    chatEl.querySelectorAll('.edit-btn').forEach(b => b.remove()); // solo la última frase se puede editar
    const btn = el('button', 'edit-btn', '✎ No dije eso');
    btn.type = 'button';
    btn.title = 'Corregir lo que Mia entendió';
    btn.addEventListener('click', () => startEdit(wrap));
    wrap.querySelector('.bubble').after(btn);
  }

  function startEdit(wrap) {
    if (busy || !lastTurn || lastTurn.wrap !== wrap) return;
    speechSynthesis?.cancel();
    const form = el('form', 'edit-form');
    const input = el('input');
    input.type = 'text'; input.id = 'editInput'; input.value = lastTurn.text; input.autocomplete = 'off';
    input.setAttribute('aria-label', 'Lo que querías decir');
    const ok = el('button', 'send', 'Enviar'); ok.type = 'submit';
    const cancel = el('button', 'say', 'Cancelar'); cancel.type = 'button';
    form.append(input, ok, cancel);
    const bubble = wrap.querySelector('.bubble'), btn = wrap.querySelector('.edit-btn');
    bubble.hidden = true; if (btn) btn.hidden = true;
    bubble.after(form);
    input.focus(); input.select();
    cancel.addEventListener('click', () => { form.remove(); bubble.hidden = false; if (btn) btn.hidden = false; });
    form.addEventListener('submit', e => {
      e.preventDefault();
      const meant = input.value.trim();
      if (!meant || meant === lastTurn.text) { cancel.click(); return; }
      redoTurn(meant);
    });
  }

  // Deshace el último turno (tu frase, la respuesta de Mia y lo que se guardó) y lo repite con la frase corregida.
  function redoTurn(meant) {
    const t = lastTurn;
    lastTurn = null;
    speechSynthesis?.cancel(); mouth.stop();
    memory = Object.assign(emptyMemory(), JSON.parse(t.before.memory));
    saveMemory(); renderMemory();
    // La conversación de la IA no se recorta (eso la obliga a releer todo, ~40 s):
    // se le avisa que el micrófono entendió mal y responde a la frase corregida.
    while (t.wrap.nextSibling) t.wrap.nextSibling.remove();
    t.wrap.remove();
    lastMia = [...chatEl.querySelectorAll('.msg.mia .bubble')].pop()?.textContent || '';
    log(`Frase editada: "${t.text.slice(0, 50)}" → "${meant.slice(0, 50)}"`);
    timing.lastWord = 0;
    send(meant, { redo: true });
  }

  // ---------- Conversación ----------
  let busy = false;
  let lastTurn = null; // tu último mensaje, para poder editarlo
  async function send(text, { greet = false, redo = false } = {}) {
    if (busy) {
      log('Mensaje ignorado: la IA sigue ocupada');
      addError('Mia todavía está pensando la respuesta anterior. Espera un momento.');
      return;
    }
    busy = true;
    document.body.dataset.busy = '1';
    setState('thinking');
    const mine = greet ? null : addMe(text);
    // Foto del estado antes de este turno: si editas la frase, se vuelve a este punto.
    const before = greet ? null : { memory: JSON.stringify(memory), sessionLen: chatSession ? chatSession.msgs.length : null };
    const typing = addTyping();
    timing.sent = performance.now();
    if (greet) timing.lastWord = 0;
    const messages = sessionMessages();
    const userMsg = {
      role: 'user',
      content: greet
        ? `(Your friend just opened the app. Greet them like a friend who is happy to see them, in one short sentence${memory.facts.length ? ' that mentions something you remember about them' : ' and ask for their name'}, then ask one simple question to start chatting.)`
        : text,
    };
    // ¿Necesita información de internet? Se agrega como nota para Mia.
    let online = null;
    if (!greet) {
      try {
        $('statusText').textContent = 'buscando…';
        online = await lookupOnline(text);
        if (online) log(`Información ${online.source ? 'de ' + online.source : ''}: ${online.note.slice(0, 120)}`);
      } catch (e) {
        log('No se pudo consultar internet: ' + errText(e));
        online = { note: 'You tried to check the internet but it did not work. Say politely that you could not check it right now.', source: null };
      }
      if (online) userMsg.content += `\n\n[Note for Mia: ${online.note}]`;
      if (redo) userMsg.content += '\n\n[Note for Mia: the microphone misheard the previous message. This is what your friend really said. Forget the previous message and reply to this one.]';
    }
    messages.push(userMsg);

    let spokenEarly = null, miaBubble = null;
    try {
      const raw = await llmChat(messages, {
        format: RESPONSE_SCHEMA,
        onChunk: partial => {
          if (spokenEarly) return;
          const reply = asMia(earlyReply(partial));
          if (!reply) return;
          spokenEarly = reply;
          typing.remove();
          miaBubble = addMia(reply);
          setState('idle');
          speak(reply, { after: () => { if (prefs.handsFree && state === 'idle') setTimeout(startListening, 300); } });
        },
      });
      const r = parseResult(raw);
      if (chatSession) {
        chatSession.msgs.push(userMsg, { role: 'assistant', content: lastRaw });
        if (chatSession.msgs.length > 70) resetSession(); // conversación muy larga: se empieza de nuevo
      }
      if (!spokenEarly) {
        typing.remove();
        setState('idle');
        addMia(r.reply);
        speak(r.reply, { after: () => { if (prefs.handsFree) setTimeout(startListening, 300); } });
      }
      lastMia = r.reply;
      if (online?.source) {
        const lastMiaMsg = [...chatEl.querySelectorAll('.msg.mia')].pop();
        lastMiaMsg?.append(el('div', 'source', 'Fuente: ' + online.source));
      }
      scrollDown();

      const now = new Date().toISOString();
      if (!greet) memory.messages.push({ role: 'user', text, at: now });
      memory.messages.push({ role: 'assistant', text: r.reply, at: now });
      recordFacts(r.new_memories);
      saveMemory();
      renderMemory();
      if (mine && before) {
        lastTurn = { text, wrap: mine, before };
        addEditButton(mine);
      }
      setTimeout(summarizeOld, 4000);
    } catch (e) {
      typing.remove();
      setState('idle');
      log('Error al responder: ' + errText(e));
      addError(e?.message || errText(e));
    } finally {
      busy = false;
      delete document.body.dataset.busy;
    }
  }

  // ---------- Panel de memoria ----------
  function renderMemory() {
    const fl = $('factsList'); fl.innerHTML = '';
    if (!memory.facts.length) fl.append(el('li', 'empty', 'Todavía nada. Cuéntale a Mia sobre ti.'));
    for (const f of memory.facts) {
      const li = el('li'); li.append(el('span', '', f));
      const b = el('button', '', 'Olvidar'); b.type = 'button';
      b.addEventListener('click', () => { memory.facts = memory.facts.filter(x => x !== f); saveMemory(); renderMemory(); });
      li.append(b); fl.append(li);
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
      const desktop = !/Android|iPhone|iPad/i.test(navigator.userAgent);
      line.textContent = desktop
        ? 'Chrome no tiene WebGPU activado en esta PC. Abre chrome://flags, activa "Unsafe WebGPU Support" y "Override software rendering list" (no actives Vulkan), y toca "Relaunch".'
        : 'Este navegador no tiene WebGPU. Usa Google Chrome actualizado en Android.';
      log('Sin WebGPU: ' + navigator.userAgent);
      $('startBtn').disabled = true;
      return false;
    }
    // En GPU Qualcomm Adreno el modo f16 de WebLLM genera texto sin sentido: se usa f32.
    const adreno = /qualcomm|adreno/i.test(`${adapter.info?.vendor} ${adapter.info?.architecture}`);
    if (prefs.fixVersion !== 4) {
      delete prefs.noGrammar; delete prefs.noWorker; // versiones anteriores los guardaban para siempre
      if (!prefs.fixVersion || prefs.fixVersion < 2) delete prefs.forceF32;
      if (prefs.model === 'Qwen3-4B') prefs.model = DEFAULT_MODEL; // el antiguo modelo por defecto
      prefs.fixVersion = 4; savePrefs();
      $('modelSel').value = prefs.model;
    }
    f16 = adapter.features.has('shader-f16') && !prefs.forceF32 && !adreno;
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
    timing.lastWord = 0; // escrito, no hablado
    send(v);
  });

  // ---------- Modelos descargados (Ajustes) ----------
  const gb = bytes => (bytes / 1e9).toFixed(1).replace('.', ',') + ' GB';
  async function renderCachedModels() {
    const list = $('cachedList');
    const est = await navigator.storage?.estimate?.().catch(() => null);
    $('storageLine').textContent = est
      ? `Esta app ocupa ${gb(est.usage || 0)} en tu celular.`
      : 'Estos son los modelos guardados en tu celular.';
    const found = [];
    for (const m of MODELS) {
      for (const q of ['q4f16_1', 'q4f32_1']) {
        const id = `${m.base}-${q}-MLC`;
        if (await webllm.hasModelInCache(id).catch(() => false)) found.push({ id, m, q });
      }
    }
    list.innerHTML = '';
    if (!found.length) { list.append(el('li', 'empty', 'No hay modelos descargados.')); return; }
    const current = modelId(prefs.model);
    for (const { id, m, q } of found) {
      const li = el('li');
      const name = el('span', '', m.label);
      name.append(el('small', '', q === 'q4f16_1' ? 'versión f16' : 'versión f32'));
      li.append(name);
      if (id === current || id === loadedId) {
        li.append(el('span', 'in-use', 'En uso'));
      } else {
        const btn = el('button', '', 'Borrar');
        btn.type = 'button';
        btn.addEventListener('click', async () => {
          if (!btn.classList.contains('sure')) { btn.classList.add('sure'); btn.textContent = '¿Seguro? Borrar'; return; }
          btn.disabled = true; btn.textContent = 'Borrando…';
          try { await webllm.deleteModelAllInfoInCache(id); log('Modelo borrado: ' + id); }
          catch (e) { log('No se pudo borrar ' + id + ': ' + errText(e)); }
          renderCachedModels();
        });
        li.append(btn);
      }
      list.append(li);
    }
  }

  const dlg = $('settings');
  function openSettings() { renderMemory(); renderCachedModels(); dlg.showModal(); }
  $('settingsBtn').addEventListener('click', openSettings);
  $('closeSettings').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });

  function applyPrefsToUI() {
    $('rateRange').value = prefs.rate;
    $('rateVal').textContent = Number(prefs.rate).toFixed(2) + '×';
    $('handsFree').checked = prefs.handsFree;
  }
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
      resetSession();
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
    memory = emptyMemory(); saveMemory(); resetSession();
    $('wipeConfirm').hidden = true;
    chatEl.innerHTML = '';
    renderMemory();
  });

  // ---------- Inicio ----------
  function renderHistory() {
    for (const m of memory.messages.slice(-30)) {
      if (m.role === 'user') addMe(m.text);
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
