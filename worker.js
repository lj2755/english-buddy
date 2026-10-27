// La IA corre aquí, fuera de la página, para que la interfaz siga fluida.
import { WebWorkerMLCEngineHandler } from 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/+esm';
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = msg => handler.onmessage(msg);
