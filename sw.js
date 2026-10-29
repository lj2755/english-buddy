// Guarda la app en el celular para que abra rápido y sin internet.
// Siempre pide primero la versión nueva (sin usar la copia temporal del navegador) y solo usa la
// guardada si no hay conexión.
const CACHE = 'eb-web-v32';
const SHELL = ['./', 'index.html', 'style.css', 'app.js', 'worker.js', 'stt-worker.js', 'icon.png', 'mia.jpg', 'mia-idle.mp4', 'mia-talk.mp4', 'manifest.webmanifest'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL.map(u => new Request(u, { cache: 'reload' })))));
  self.skipWaiting();
});
self.addEventListener('activate', e => e.waitUntil(
  caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith('eb-web-') && k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim())
));
// GitHub Pages no permite configurar cabeceras. Estas dos activan el "aislamiento entre orígenes",
// que deja a la IA en el procesador usar todos los núcleos (si no, usa uno solo y es mucho más lenta).
function isolate(r) {
  if (!r || r.status === 0) return r;
  const h = new Headers(r.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'credentialless');
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
}
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || e.request.headers.has('range')) return;
  e.respondWith(fetch(e.request, { cache: 'no-cache' })
    .then(r => { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); return r; })
    .catch(() => caches.match(e.request))
    .then(isolate));
});
