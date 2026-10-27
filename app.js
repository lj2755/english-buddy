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
    { base: 'Qwen3-4B', label: 'Qwen3 4B · más rápido, corrige peor' },
    { base: 'Qwen3.5-4B', label: 'Qwen3.5 4B · más nuevo' },
    { base: 'Qwen3.5-2B', label: 'Qwen3.5 2B · más rápido' },
    { base: 'Qwen3-1.7B', label: 'Qwen3 1.7B · el más liviano' },
  ];
  // Gemma 4 E2B fue el mejor en las pruebas: entiende cuándo pedir que repitas y explica bien en español.
  const DEFAULT_MODEL = 'gemma-4-E2B-it';

  const SCENARIO_NAMES = {
    free: 'Charla libre', coffee: 'Cafetería', restaurant: 'Restaurante', shopping: 'Tienda de ropa',
    doctor: 'Médico', airport: 'Aeropuerto', directions: 'Direcciones', smalltalk: 'Small talk', interview: 'Entrevista',
  };
  const KIND_NAMES = { grammar: 'gramática', vocabulary: 'vocabulario', naturalness: 'natural', pronunciation: 'pronunciación' };
  const SCENARIOS = {
    free: 'A relaxed chat between friends about the learner\'s day, life, plans and interests.',
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
      understood: { type: 'boolean' },
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
    required: ['understood', 'reply', 'corrections', 'corrected_sentence', 'new_memories'],
  };

  // ---------- Motor de IA (WebLLM en un Web Worker, para que el video no se trabe) ----------
  let engine = null, loadedId = null, f16 = false, loading = null;
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
        engine = await webllm.CreateMLCEngine(id, { initProgressCallback: onProgress });
      } else if (!engine) {
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
    const summaries = memory.summaries.slice(-3).map(s => `- ${s.text}`).join('\n') || '(none)';
    const mistakes = [...memory.mistakes].sort((a, b) => b.count - a.count).slice(0, 5)
      .map(m => `- "${m.original}" -> "${m.corrected}"`).join('\n') || '(none)';
    const lang = prefs.explainLang === 'en' ? 'simple English' : 'Spanish';
    return `You are Mia, the learner's friend. Your name is Mia. Never say you are an AI, a model, Gemma or Qwen.
The learner is a Spanish speaker who loves chatting with you in English, partly because you kindly help when they use a word wrong.
Scenario: ${SCENARIOS[prefs.scenario] || SCENARIOS.free}

About you: warm, curious and positive. You like music, movies, cooking and walks in the park. You may share a short opinion, like a friend, but keep the focus on the learner.

How you talk:
- Like a good friend: warm, relaxed and kind. Simple everyday English, short sentences (beginner to intermediate level). No emojis.
- Plain text only: never use asterisks, markdown or any formatting symbols.
- One or two sentences, then one question about the learner's life, so the conversation keeps going.
- React to what the learner says with real interest ("Oh, that sounds great!", "Really? Tell me more.").
- Your friend counts on you to help with English: always check their LAST message carefully and put every real mistake in "corrections". Never let a mistake pass, even when you are just chatting.
- Talk only about what the learner actually said. Never invent details about the learner or assume things they did not say.
- Sometimes the learner's message ends with a [Note for Mia ...]. It contains real information from the internet (date, weather, Wikipedia). Use it to answer in simple words. It was not written by the learner: never correct it. If there is no note, do not claim to know today's date, the weather or recent news.

The learner's text comes from a microphone, so it may be cut off or wrong.
If the LAST message is unclear, incomplete, makes no sense (for example "car the sing yesterday green"), or you are not sure what it means:
set "understood" to false, ask kindly to hear it again (for example: "Sorry, I didn't catch that. Could you say it again?"), and leave corrections and new_memories empty.

Reply with a JSON object:
- "understood": true if you clearly understood the learner's LAST message, false otherwise.
- "reply": what you say next.
- "corrections": the real mistakes in the learner's LAST message (grammar, wrong word, unnatural phrase). Use [] if it is correct. Ignore punctuation and capital letters. Each item: "original" (the wrong words), "corrected", "explanation" (one short, kind sentence written in ${lang}${prefs.explainLang === 'en' ? '' : ' / en español'}, like a friend explaining), "type" (grammar, vocabulary, naturalness or pronunciation).
- "corrected_sentence": the learner's whole last message said correctly ("" if not understood).
- "new_memories": facts the learner clearly said about themself in their LAST message, as short sentences like "His name is Luis." Never guess. [] if none.

What you remember about the learner:
${facts}
Previous conversations:
${summaries}
Mistakes they repeat:
${mistakes}`;
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

  // El modelo a veces se presenta con su nombre de fábrica: Mia siempre es Mia.
  const asMia = t => String(t || '').replace(/\b(?:Gemma|Qwen)(?:[\s-]*[\d.]+[a-z]?)*(?:[\s-]*(?:E2B|it))?\b/gi, 'Mia');

  // Parecido entre dos frases según las palabras que comparten (0 a 1).
  function overlap(a, b) {
    const A = new Set(norm(a).split(' ').filter(Boolean)), B = new Set(norm(b).split(' ').filter(Boolean));
    if (!A.size || !B.size) return 0;
    let same = 0;
    for (const w of A) if (B.has(w)) same++;
    return same / Math.max(A.size, B.size);
  }

  // Errores típicos de quien habla español. Se revisan con reglas, sin la IA: es instantáneo y
  // no falla con estos casos (el modelo pequeño a veces los deja pasar).
  const COMMON_MISTAKES = [
    { re: /\b(i|you|he|she|we|they|my \w+) (have|has) (\d+) years( old)?\b/i,
      fix: m => `${m[1]} ${({ i: 'am', you: 'are', he: 'is', she: 'is', we: 'are', they: 'are' })[m[1].toLowerCase()] || 'is'} ${m[3]} years old`,
      es: 'Para la edad en inglés se usa "to be": I am 45 years old (no "I have").', en: 'In English we use "to be" for age: I am 45 years old.' },
    { re: /\b(i|we|they|you) (am|are) agree\b/i, fix: m => `${m[1]} agree`,
      es: '"Agree" ya es un verbo: se dice "I agree", sin "am".', en: '"Agree" is already a verb: say "I agree".' },
    { re: /\b(he|she|it) (don't|dont)\b/i, fix: m => `${m[1]} doesn't`,
      es: 'Con he, she o it se usa "doesn\'t".', en: 'With he, she or it we use "doesn\'t".' },
    { re: /\bsince (\w+) (years|months|weeks|days)\b/i, fix: m => `for ${m[1]} ${m[2]}`,
      es: 'Para una cantidad de tiempo se usa "for" (for five years); "since" va con un momento (since 2020).', en: 'Use "for" with a length of time and "since" with a point in time.' },
    { re: /\bexplain me\b/i, fix: () => 'explain to me',
      es: 'Se dice "explain to me" o "explain it to me".', en: 'We say "explain to me".' },
    { re: /\bpeople is\b/i, fix: () => 'people are',
      es: '"People" es plural en inglés: "people are".', en: '"People" is plural: "people are".' },
    { re: /\bmake (a|the) question\b/i, fix: m => `ask ${m[1]} question`,
      es: 'Las preguntas se "ask", no se "make": "ask a question".', en: 'We "ask" a question.' },
    { re: /\b(an? |many |the )advices\b|\ban? advice\b/i, fix: () => 'some advice',
      es: '"Advice" no se cuenta en inglés: "some advice" o "a piece of advice".', en: '"Advice" is uncountable: "some advice".' },
    { re: /\b(i|we|they) (is)\b/i, fix: m => `${m[1]} ${m[1].toLowerCase() === 'i' ? 'am' : 'are'}`,
      es: 'Con I se usa "am"; con we y they, "are".', en: 'Use "am" with I and "are" with we/they.' },
  ];

  function ruleCorrections(text) {
    const out = [];
    for (const r of COMMON_MISTAKES) {
      const m = r.re.exec(text);
      if (m) out.push({ original: m[0], corrected: r.fix(m), explanation: prefs.explainLang === 'en' ? r.en : r.es, type: 'grammar' });
    }
    return out;
  }

  // El modelo pequeño a veces inventa correcciones que no tienen que ver con lo que dijiste.
  // Solo se aceptan si corrigen palabras que de verdad dijiste y se parecen a tu frase.
  function sanitizeCorrections(r, learnerText) {
    const said = norm(learnerText);
    const before = r.corrections.length;
    r.corrections = r.corrections.filter(c => said.includes(norm(c.original)) && overlap(c.original, c.corrected) >= 0.3 &&
      !(/\bmia\b/i.test(c.original) && !/\bmia\b/i.test(c.corrected))); // "Hi Mia" está bien dicho
    const words = t => norm(t).split(' ').filter(Boolean).length;
    if (r.corrected_sentence && (overlap(learnerText, r.corrected_sentence) < 0.5 ||
        words(r.corrected_sentence) > words(learnerText) + 4 ||                    // trae la pregunta de Mia pegada
        (/\bmia\b/i.test(learnerText) && !/\bmia\b/i.test(r.corrected_sentence)))) r.corrected_sentence = '';
    // Sumar los errores típicos que la IA no marcó.
    for (const c of ruleCorrections(learnerText)) {
      const same = r.corrections.find(x => norm(x.corrected).includes(norm(c.corrected)));
      if (same) { same.explanation = c.explanation; same.type = 'grammar'; } // la explicación de la regla es fiable
      else {
        r.corrections.push(c);
        let fixed = r.corrected_sentence || learnerText;
        const at = fixed.toLowerCase().indexOf(c.original.toLowerCase());
        if (at >= 0) fixed = fixed.slice(0, at) + c.corrected + fixed.slice(at + c.original.length);
        r.corrected_sentence = fixed;
      }
    }
    log(`Correcciones: ${before} propuestas, ${r.corrections.length} aceptadas · entendió: ${r.understood !== false}` +
      (before > r.corrections.length ? ' · descartadas por no corresponder a tu frase' : ''));
    return r;
  }

  function parseResult(text) {
    let r;
    try { r = JSON.parse(text); }
    catch {
      try { r = JSON.parse(text.match(/\{[\s\S]*\}/)[0]); }
      catch { r = { reply: earlyReply(text) || text, corrections: [], corrected_sentence: '', new_memories: [] }; }
    }
    r.corrections = (Array.isArray(r.corrections) ? r.corrections : [])
      .filter(c => c && c.original && c.corrected && norm(c.original) !== norm(c.corrected));
    r.new_memories = Array.isArray(r.new_memories) ? r.new_memories : [];
    r.reply = asMia(r.reply);
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
  const norm = t => String(t || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const differs = (a, b) => b && norm(a) !== norm(b);

  function renderCorrections(list, corrected, original) {
    if (!list.length && !differs(original, corrected)) return el('div', 'ok', '✓ ¡Perfecto! Lo dijiste bien.');
    const box = el('div', 'fixes');
    box.append(el('div', 'label', list.length === 0 ? 'Mia te sugiere decirlo así' : list.length === 1 ? 'Un consejo de Mia' : `${list.length} consejos de Mia`));
    for (const c of list) {
      const f = el('div', 'fix');
      const pair = el('div', 'pair');
      pair.append(el('del', '', plain(c.original)), document.createTextNode(' → '), el('ins', '', plain(c.corrected)));
      if (KIND_NAMES[c.type]) pair.append(el('span', 'kind', KIND_NAMES[c.type]));
      f.append(pair);
      if (c.explanation) f.append(el('div', 'why', plain(c.explanation)));
      box.append(f);
    }
    if (differs(original, corrected)) {
      const better = el('div', 'better');
      const p = el('p');
      p.append(el('small', '', 'Mejor dicho'), document.createTextNode(plain(corrected)));
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
      if (text) send(text, { voice: true });
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
      return { note: 'Your name is Mia. You are the learner\'s friend. Say your name is Mia.', source: null };
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
  // Palabras que cambiaron entre lo que Chrome entendió y lo que querías decir.
  function pronunciationPairs(heardText, meantText) {
    const a = heardText.trim().split(/\s+/), b = meantText.trim().split(/\s+/);
    const k = w => norm(w);
    const L = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
    for (let i = a.length - 1; i >= 0; i--)
      for (let j = b.length - 1; j >= 0; j--)
        L[i][j] = k(a[i]) === k(b[j]) ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const pairs = [];
    let i = 0, j = 0, da = [], db = [];
    const flush = () => {
      if (da.length && db.length) pairs.push({ heard: da.join(' '), meant: db.join(' ') });
      da = []; db = [];
    };
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && k(a[i]) === k(b[j])) { flush(); i++; j++; }
      else if (j < b.length && (i >= a.length || L[i][j + 1] >= L[i + 1][j])) db.push(b[j++]);
      else da.push(a[i++]);
    }
    flush();
    return pairs.slice(0, 3);
  }

  function renderPronunciation(pairs) {
    const box = el('div', 'fixes pron');
    box.append(el('div', 'label', 'Pronunciación'));
    for (const p of pairs) {
      const f = el('div', 'fix');
      const pair = el('div', 'pair');
      pair.append(document.createTextNode('Se entendió '), el('del', '', p.heard), document.createTextNode(' → querías decir '), el('ins', '', p.meant));
      const btn = el('button', 'say', 'Escuchar');
      btn.type = 'button';
      btn.addEventListener('click', () => speak(p.meant, { slow: true }));
      f.append(pair, el('div', 'why', `Escucha "${p.meant}" y repítelo con calma.`), btn);
      box.append(f);
    }
    return box;
  }

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
    send(meant, { heard: t.heard, redo: true });
  }

  // ---------- Conversación ----------
  let busy = false;
  let lastTurn = null; // tu último mensaje, para poder editarlo
  async function send(text, { greet = false, voice = false, heard = null, redo = false } = {}) {
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
        ? `(The learner just opened the app. Greet them like a friend who is happy to see them, in one short sentence${memory.facts.length ? ' that mentions something you remember about them' : ' and ask for their name'}, then ask one simple question to start chatting. understood must be true, corrections [] and corrected_sentence "".)`
        : text,
    };
    // ¿Necesita información de internet? Se agrega como nota, marcada para que no se corrija.
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
      if (online) userMsg.content += `\n\n[Note for Mia, not written by the learner: ${online.note}]`;
      if (redo) userMsg.content += '\n\n[Note for Mia, not written by the learner: the microphone misheard the previous message. This is what the learner really said. Forget the previous message and reply to this one.]';
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
      const r = greet ? parseResult(raw) : sanitizeCorrections(parseResult(raw), text);
      if (chatSession) {
        chatSession.msgs.push(userMsg, { role: 'assistant', content: lastRaw });
        if (chatSession.msgs.length > 30) resetSession(); // conversación muy larga: se empieza de nuevo
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
      // Si Mia no entendió, no corrige ni recuerda nada: solo pide que lo repitas.
      const unclear = !greet && r.understood === false;
      if (unclear) log('Mia no entendió: "' + text.slice(0, 60) + '"');
      if (mine) mine.append(unclear ? el('div', 'unclear', 'Mia no te entendió bien. Díselo otra vez.') : renderCorrections(r.corrections, r.corrected_sentence, text));
      // Si editaste una frase dicha por micrófono, lo que cambió son palabras que Chrome entendió mal.
      const pron = heard ? pronunciationPairs(heard, text) : [];
      if (mine && pron.length) mine.append(renderPronunciation(pron));
      scrollDown();

      const now = new Date().toISOString();
      if (!greet) {
        memory.messages.push({ role: 'user', text, at: now, corrections: unclear ? [] : r.corrections, corrected: unclear ? '' : r.corrected_sentence, unclear });
        if (!unclear) {
          memory.stats.turns++;
          if (!r.corrections.length && !differs(text, r.corrected_sentence)) memory.stats.perfect++;
          recordMistakes(r.corrections);
        }
        recordMistakes(pron.map(p => ({ original: p.heard, corrected: p.meant, type: 'pronunciation',
          explanation: `Se entendió "${p.heard}" cuando querías decir "${p.meant}".` })));
      }
      memory.messages.push({ role: 'assistant', text: r.reply, at: now });
      if (!unclear) recordFacts(r.new_memories);
      saveMemory();
      renderMemory();
      if (mine && before) {
        lastTurn = { text, wrap: mine, before, heard: heard || (voice ? text : null) };
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
    prefs.scenario = e.target.value; savePrefs(); applyPrefsToUI(); resetSession();
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
  $('explainSel').addEventListener('change', e => { prefs.explainLang = e.target.value; savePrefs(); resetSession(); });
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
      if (m.role === 'user' && m.unclear) { const w = addMe(m.text); w.append(el('div', 'unclear', 'Mia no te entendió bien. Díselo otra vez.')); }
      else if (m.role === 'user') addMe(m.text, m.corrections || [], m.corrected);
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
