// Guarda la app en el celular para que abra rápido y sin internet.
const CACHE = 'eb-web-v2';
const SHELL = ['./', 'index.html', 'style.css', 'app.js', 'worker.js', 'icon.png', 'mia.jpg', 'mia-idle.mp4', 'mia-talk.mp4', 'manifest.webmanifest'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || e.request.headers.has('range')) return;
  e.respondWith(fetch(e.request).then(r => { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); return r; }).catch(() => caches.match(e.request)));
});
