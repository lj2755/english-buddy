# English Buddy

Conversa en inglés o en español con **Mia**, una amiga virtual. Te escucha, te responde en voz alta y recuerda lo que le cuentas.

**La IA corre dentro de tu celular o tu PC**: no hay servidor, ni cuentas, ni costos por uso. La memoria se guarda en tu dispositivo.

- **Con GPU** ([WebLLM](https://github.com/mlc-ai/web-llm), WebGPU): Gemma 4 E2B, y si no cabe, modelos más livianos (Qwen3.5 2B, Llama 3.2 1B).
- **En el procesador** ([wllama](https://github.com/ngxson/wllama), llama.cpp): Qwen2.5 1.5B. Se usa sola cuando el dispositivo no tiene WebGPU o su GPU calcula mal (pasa con algunas GPU Adreno y Chrome 156). Es más lenta, pero funciona en casi cualquier celular. Se puede elegir en **Ajustes → Motor de IA**.

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
- Todo funciona sin internet después de la primera descarga, salvo la información actual (clima, Wikipedia) y el reconocimiento de voz de Chrome si lo eliges en lugar de Whisper.

## Funciones

- Conversación por voz: micrófono, voz de Mia con velocidad ajustable y modo manos libres.
- **Inglés o español:** Mia responde en el idioma en que le hablas, con una voz para cada idioma.
- **Reconocimiento de voz con [Whisper](https://huggingface.co/onnx-community/whisper-small)** dentro de tu dispositivo: detecta solo si hablas en inglés o en español y no necesita internet. La primera vez descarga unos 300 MB. En **Ajustes → Reconocimiento de voz** puedes cambiar al de Chrome (usa internet y el botón **EN/ES**).
- Memoria: lo que le cuentas sobre ti. Puedes verlo, borrarlo y guardar una copia en un archivo.
- Información actual: fecha y hora (de tu dispositivo), clima ([Open-Meteo](https://open-meteo.com)) y datos generales ([Wikipedia](https://en.wikipedia.org)). Pregunta, por ejemplo, *"What's the weather in Bogotá?"* o *"¿Quién fue Gabriel García Márquez?"*. Debajo de la respuesta aparece la fuente.

## Archivos

Es una página estática: `index.html`, `app.js` (lógica), `worker.js` (la IA, en segundo plano), `stt-worker.js` (Whisper, en segundo plano), `style.css`, `sw.js` (funciona sin conexión) y los videos de Mia.
Para probarla en la PC: `python3 -m http.server 8090` y abre http://localhost:8090 en Chrome.
