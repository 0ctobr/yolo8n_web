// Конфигурация
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5;

let session = null;
let classes = [];

// DOM
const elements = {
    imageUpload: document.getElementById('imageUpload'),
    preview: document.getElementById('preview'),
    canvas: document.getElementById('canvas'),
    results: document.getElementById('results'),
    loading: document.getElementById('loading')
};

// Инициализация
async function init() {
    try {
        elements.loading.style.display = 'block';
        elements.loading.textContent = 'Initializing WASM runtime...';

        await ort.env.wasm.wasmReady;
        elements.loading.textContent = 'WASM ready, loading classes...';

        const classesResponse = await fetch('coco_classes.json');
        if (!classesResponse.ok) throw new Error('Failed to load COCO classes');
        classes = await classesResponse.json();

        ort.env.wasm.numThreads = 1; // отключаем многопоточность

        elements.loading.textContent = 'Loading YOLOv8 model...';
        session = await ort.InferenceSession.create('model/yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        elements.loading.style.display = 'none';
        console.log('Model loaded');
    } catch (err) {
        console.error('WASM init error:', err);
        elements.loading.innerHTML = `
            <div class="error">WASM Error: ${err.message}</div>
            <ul>
                <li>Enable WebAssembly in browser</li>
                <li>Try Firefox if issues in Chrome</li>
                <li>Ensure model/yolov8n.onnx is valid</li>
            </ul>
        `;
    }
}

elements.imageUpload.addEventListener('change', handleImageUpload);

async function handleImageUpload(event) {
    if (!session) {
        alert('Model not ready yet!');
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    const image = new Image();
    image.src = URL.createObjectURL(file);

    image.onload = async () => {
        elements.preview.src = image.src;
        const ctx = elements.canvas.getContext('2d');
        ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        elements.results.innerHTML = '';

        elements.loading.textContent = 'Processing image...';
        elements.loading.style.display = 'block';

        try {
            const tensor = preprocessImage(image);
            const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

            const outputs = await session.run({ images: input });

            const output = outputs[Object.keys(outputs)[0]];
            const predictions = output.data;

            const detections = processOutput(predictions, image.width, image.height);
            renderDetections(detections);
            displayResults(detections);
        } catch (err) {
            console.error('Processing error:', err);
            elements.results.innerHTML = `<div class="error">Error: ${err.message}</div>`;
        } finally {
            elements.loading.style.display = 'none';
        }
    };
}

function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;

    const ratio = Math.min(INPUT_SIZE / image.width, INPUT_SIZE / image.height);
    const newWidth = Math.round(image.width * ratio);
    const newHeight = Math.round(image.height * ratio);
    const xOffset = (INPUT_SIZE - newWidth) / 2;
    const yOffset = (INPUT_SIZE - newHeight) / 2;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(image, xOffset, yOffset, newWidth, newHeight);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const data = imageData.data;
    const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        tensor[idx] = data[i] / 255;
        tensor[idx + INPUT_SIZE * INPUT_SIZE] = data[i + 1] / 255;
        tensor[idx + 2 * INPUT_SIZE * INPUT_SIZE] = data[i + 2] / 255;
    }

    return tensor;
}

function processOutput(predictions, originalWidth, originalHeight) {
    const detections = [];
    for (let i = 0; i < predictions.length; i += 6) {
        const x1 = predictions[i];
        const y1 = predictions[i + 1];
        const x2 = predictions[i + 2];
        const y2 = predictions[i + 3];
        const confidence = predictions[i + 4];
        const classId = predictions[i + 5];

        if (confidence < CONFIDENCE_THRESHOLD) continue;

        detections.push({
            classId,
            className: classes[classId] ?? `Class ${classId}`,
            confidence,
            bbox: [
                Math.max(0, x1),
                Math.max(0, y1),
                Math.min(originalWidth, x2) - Math.max(0, x1),
                Math.min(originalHeight, y2) - Math.max(0, y1)
            ]
        });
    }
    return detections;
}

function renderDetections(detections) {
    const ctx = elements.canvas.getContext('2d');
    const preview = elements.preview;
    elements.canvas.width = preview.width;
    elements.canvas.height = preview.height;

    ctx.clearRect(0, 0, preview.width, preview.height);

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const color = getColorForClass(det.classId);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        const label = `${det.className} ${(det.confidence * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(x, y - 20, textWidth + 8, 20);

        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(label, x + 4, y - 5);
    });
}

function displayResults(detections) {
    elements.results.innerHTML = detections.length === 0
        ? '<div class="detection-item">No objects detected</div>'
        : detections.map(det => {
            const [x, y, w, h] = det.bbox;
            const color = getColorForClass(det.classId);

            return `
                <div class="detection-item">
                    <div class="detection-color" style="background-color: ${color}"></div>
                    <div class="detection-info">
                        <strong>${det.className}</strong> (${(det.confidence * 100).toFixed(1)}%)
                        <div>Position: (${x.toFixed(0)}, ${y.toFixed(0)}), Size: ${w.toFixed(0)}×${h.toFixed(0)}</div>
                    </div>
                </div>
            `;
        }).join('');
}

function getColorForClass(classId) {
    const hue = (classId * 137.508) % 360;
    return `hsl(${hue}, 90%, 50%)`;
}

document.addEventListener('DOMContentLoaded', init);