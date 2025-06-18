// ─────────────────────────────────────────────────────────────
// 0. Константы конфигурации
// ─────────────────────────────────────────────────────────────
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const CLASS_SCORE_THRESHOLD = 0.4;
const IOU_THRESHOLD = 0.45;

// ─────────────────────────────────────────────────────────────
// 1. Глобальные переменные и DOM элементы
// ─────────────────────────────────────────────────────────────
let session = null;
let classes = [];

const elements = {
    imageUpload: document.getElementById('imageUpload'),
    preview: document.getElementById('preview'),
    canvas: document.getElementById('canvas'),
    results: document.getElementById('results'),
    loading: document.getElementById('loading')
};

// ─────────────────────────────────────────────────────────────
// 2. Инициализация модели и WASM окружения
// ─────────────────────────────────────────────────────────────
async function init() {
    try {
        elements.loading.style.display = 'block';
        elements.loading.textContent = 'Initializing WASM runtime...';

        await ort.env.wasm.wasmReady;
        ort.env.wasm.numThreads = 1;

        elements.loading.textContent = 'Loading COCO classes...';
        const classesResponse = await fetch('coco_classes.json');
        if (!classesResponse.ok) throw new Error('Failed to load COCO classes');
        classes = await classesResponse.json();

        elements.loading.textContent = 'Loading YOLOv8 model...';
        session = await ort.InferenceSession.create('model/yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        elements.loading.style.display = 'none';
        console.log('✅ Model loaded successfully');
    } catch (error) {
        console.error('❌ WASM initialization error:', error);
        elements.loading.innerHTML = `
            <div class="error">WASM Error: ${error.message}</div>
            <ul>
                <li>Check if WebAssembly is enabled</li>
                <li>Ensure model file exists at "model/yolov8n.onnx"</li>
            </ul>
        `;
    }
}

// ─────────────────────────────────────────────────────────────
// 3. Обработка загрузки изображения и запуск инференса
// ─────────────────────────────────────────────────────────────
elements.imageUpload.addEventListener('change', handleImageUpload);

async function handleImageUpload(event) {
    if (!session) {
        alert('Model is still loading...');
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    const image = new Image();
    image.src = URL.createObjectURL(file);

    image.onload = async () => {
        elements.preview.src = image.src;
        elements.results.innerHTML = '';
        const ctx = elements.canvas.getContext('2d');
        ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

        elements.loading.textContent = 'Processing image...';
        elements.loading.style.display = 'block';

        try {
            const tensor = preprocessImage(image);
            const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
            const outputs = await session.run({ images: input });

            const output = outputs[Object.keys(outputs)[0]];
            const predictions = output.data;
            console.log('📤 Output length:', predictions.length);

            const detections = processOutput(predictions, image.width, image.height);
            renderDetections(detections);
            displayResults(detections);
        } catch (error) {
            console.error('❌ Processing error:', error);
            elements.results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        } finally {
            elements.loading.style.display = 'none';
        }
    };
}

// ─────────────────────────────────────────────────────────────
// 4. Преобразование изображения в формат тензора
// ─────────────────────────────────────────────────────────────
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
        tensor[idx] = data[i] / 255;                       // R
        tensor[idx + INPUT_SIZE * INPUT_SIZE] = data[i + 1] / 255; // G
        tensor[idx + 2 * INPUT_SIZE * INPUT_SIZE] = data[i + 2] / 255; // B
    }

    return tensor;
}

// ─────────────────────────────────────────────────────────────
// 5. Обработка выхода модели и фильтрация детекций
// ─────────────────────────────────────────────────────────────
function processOutput(predictions, originalWidth, originalHeight) {
    const detections = [];
    const NUM_CLASSES = 80;
    const ATTR_COUNT = 4 + 1 + NUM_CLASSES;
    const numBoxes = predictions.length / ATTR_COUNT;

    const ratio = Math.min(INPUT_SIZE / originalWidth, INPUT_SIZE / originalHeight);
    const newWidth = originalWidth * ratio;
    const newHeight = originalHeight * ratio;
    const xOffset = (INPUT_SIZE - newWidth) / 2;
    const yOffset = (INPUT_SIZE - newHeight) / 2;

    for (let i = 0; i < numBoxes; i++) {
        const base = i * ATTR_COUNT;
        const objConf = predictions[base + 4];
        if (isNaN(objConf) || objConf < CONFIDENCE_THRESHOLD || objConf > 1.0) continue;

        let maxScore = 0;
        let classId = -1;

        for (let c = 0; c < NUM_CLASSES; c++) {
            const clsConf = predictions[base + 5 + c];
            const score = clsConf * objConf;
            if (!isNaN(score) && score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore < CLASS_SCORE_THRESHOLD || classId === -1) continue;

        const cx = predictions[base];
        const cy = predictions[base + 1];
        const w = predictions[base + 2];
        const h = predictions[base + 3];

        const x1 = (cx - w / 2 - xOffset) / ratio;
        const y1 = (cy - h / 2 - yOffset) / ratio;
        const x2 = (cx + w / 2 - xOffset) / ratio;
        const y2 = (cy + h / 2 - yOffset) / ratio;

        const bbox = [
            Math.max(0, x1),
            Math.max(0, y1),
            Math.min(originalWidth, x2) - Math.max(0, x1),
            Math.min(originalHeight, y2) - Math.max(0, y1)
        ];

        detections.push({
            classId,
            className: classes[classId] ?? `Class ${classId}`,
            confidence: maxScore,
            bbox
        });
    }

    return nms(detections);
}

// ─────────────────────────────────────────────────────────────
// 6. NMS (Non-Maximum Suppression)
// ─────────────────────────────────────────────────────────────
function nms(detections) {
    detections.sort((a, b) => b.confidence - a.confidence);
    const result = [];

    while (detections.length > 0) {
        const current = detections.shift();
        result.push(current);

        detections = detections.filter(d => {
            if (d.classId !== current.classId) return true;
            const iou = calculateIoU(current.bbox, d.bbox);
            return iou < IOU_THRESHOLD;
        });
    }

    return result;
}

// ─────────────────────────────────────────────────────────────
// 7. Расчёт IoU между двумя боксами
// ─────────────────────────────────────────────────────────────
function calculateIoU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    const xi1 = Math.max(x1, x2);
    const yi1 = Math.max(y1, y2);
    const xi2 = Math.min(x1 + w1, x2 + w2);
    const yi2 = Math.min(y1 + h1, y2 + h2);

    const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
    const unionArea = w1 * h1 + w2 * h2 - interArea;

    return interArea / unionArea;
}

// ─────────────────────────────────────────────────────────────
// 8. Отрисовка предсказаний на canvas
// ─────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────
// 9. Вывод предсказаний в текстовый блок
// ─────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────
// 10. Генерация цвета для класса
// ─────────────────────────────────────────────────────────────
function getColorForClass(classId) {
    const hue = (classId * 137.508) % 360;
    return `hsl(${hue}, 90%, 50%)`;
}

// ─────────────────────────────────────────────────────────────
// 11. Старт и инициализация при загрузке страницы
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);