# English Buddy

Conversa en inglés con **Mia**, una amiga virtual. Te escucha, te responde en voz alta y recuerda lo que le cuentas.

**La IA corre dentro de tu celular o tu PC** con [WebLLM](https://github.com/mlc-ai/web-llm) (WebGPU) y el modelo pequeño **Gemma 4 E2B**: no hay servidor, ni cuentas, ni costos por uso. La memoria se guarda en tu dispositivo.

Mia es cálida y curiosa, y conversa como una amiga. Si no entiende lo que dices, te pide que lo repitas.

## Abrirla

**https://lj2755.github.io/english-buddy/**

Escanea el código con la cámara del celular:

<img src="qr.png" alt="Código QR para abrir English Buddy" width="200">

## Usarla

1. Abre la página en **Google Chrome** (Android) con Wi-Fi.
2. Toca **Empezar a hablar**. La primera vez descarga el modelo de IA (1 a 2.5 GB); después abre en segundos y la IA funciona sin conexión.
3. Menú ⋮ → **Instalar y agregar acceso directo** → **Instalar** para tenerla como app.

## Requisitos

- **En la PC (Linux con NVIDIA):** Chrome trae WebGPU apagado. Abre `chrome://flags`, activa **Unsafe WebGPU Support** y **Override software rendering list**, y toca **Relaunch**. No actives **Vulkan**: en Wayland deja la ventana de Chrome negra. También puedes usar `./abrir-en-pc.sh`, que abre la app en un Chrome aparte con esas opciones.

- Un celular con WebGPU en Chrome (Android 12+ con GPU reciente; probado para Samsung S23 Ultra).
- Unos 4 GB de RAM libres para el modelo grande. Si el celular se queda corto, elige uno más liviano en **Ajustes → Modelo de IA**.
- El reconocimiento de voz de Chrome usa internet. La IA y la voz de Mia no.

## Funciones

- Conversación por voz: micrófono, voz de Mia con velocidad ajustable y modo manos libres.
- **✎ No dije eso:** si el micrófono entendió mal tu última frase, la corriges y Mia responde de nuevo.
- Memoria: lo que le cuentas sobre ti. Puedes verlo, borrarlo y guardar una copia en un archivo.
- Información actual: fecha y hora (de tu dispositivo), clima ([Open-Meteo](https://open-meteo.com)) y datos generales ([Wikipedia](https://en.wikipedia.org)). Pregunta, por ejemplo, *"What's the weather in Bogotá?"* o *"Who was Gabriel García Márquez?"*. Debajo de la respuesta aparece la fuente.

## Archivos

Es una página estática: `index.html`, `app.js` (lógica), `worker.js` (la IA, en segundo plano), `style.css`, `sw.js` (funciona sin conexión) y los videos de Mia.
Para probarla en la PC: `python3 -m http.server 8090` y abre http://localhost:8090 en Chrome.
