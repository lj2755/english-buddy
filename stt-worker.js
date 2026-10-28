// Reconocimiento de voz con Whisper, dentro del dispositivo. Detecta si hablas en inglés o en español.
// Corre en segundo plano para que la página no se trabe.
import { AutoProcessor, AutoTokenizer, WhisperForConditionalGeneration, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

let MODEL_ID = 'onnx-community/whisper-base';
let ENCODER_DTYPE = 'fp32';
let model, processor, tokenizer, loading;

async function load(device) {
  const progress_callback = p => {
    if (p.status === 'progress' && p.total) postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
  };
  [tokenizer, processor, model] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback }),
    AutoProcessor.from_pretrained(MODEL_ID, { progress_callback }),
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: { encoder_model: ENCODER_DTYPE, decoder_model_merged: device === 'webgpu' ? 'q4' : 'q8' },
      device,
      progress_callback,
    }),
  ]);
  // Primera pasada en vacío para compilar los programas de la GPU.
  await model.generate({ inputs: new Tensor('float32', new Float32Array(80 * 3000), [1, 80, 3000]), max_new_tokens: 1 });
}

// Whisper, sin idioma, asume inglés. Por eso primero se le pregunta cuál de los dos idiomas oye:
// el primer token que predice después de <|startoftranscript|> es el del idioma.
async function detectLanguage(input_features) {
  const cfg = model.generation_config;
  const sot = new Tensor('int64', new BigInt64Array([BigInt(cfg.decoder_start_token_id)]), [1, 1]);
  const { logits } = await model({ input_features, decoder_input_ids: sot });
  const vocab = logits.dims.at(-1), last = (logits.dims.at(-2) - 1) * vocab;
  const score = tag => logits.data[last + cfg.lang_to_id[tag]];
  return score('<|es|>') > score('<|en|>') ? 'es' : 'en';
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      if (data.model) MODEL_ID = data.model;
      if (data.encoder) ENCODER_DTYPE = data.encoder;
      loading ??= load(data.device || 'webgpu');
      await loading;
      postMessage({ type: 'ready' });
    } else if (data.type === 'transcribe') {
      await loading;
      const t0 = performance.now();
      const { input_features } = await processor(data.audio);
      const lang = await detectLanguage(input_features);
      const ids = await model.generate({ inputs: input_features, language: lang, task: 'transcribe', max_new_tokens: 128 });
      const text = tokenizer.batch_decode(ids, { skip_special_tokens: true })[0].trim();
      postMessage({ type: 'result', id: data.id, text, lang, ms: Math.round(performance.now() - t0) });
    }
  } catch (e) {
    postMessage({ type: 'error', id: data.id, message: e?.message || String(e) });
  }
};
