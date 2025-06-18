// (0) Константы конфигурации
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const CLASS_SCORE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const NUM_CLASSES = 80;
const NUM_ATTRIBUTES = 84; // 4 bbox + 1 obj_conf + 80 class scores

// (1) Глобальные переменные и DOM элементы
let session = null;
let classes = [];

const elements = {
    imageUpload: document.getElementById('imageUpload'),
    preview: document.getElementById('preview'),
    canvas: document.getElementById('canvas'),
    results: document.getElementById('results'),
    loading: document.getElementById('loading')
};

// (2) Инициализация модели и загрузка классов
async function init() {
    try {
        elements.loading.style.display = 'block';
        elements.loading.textContent = 'Initializing WASM runtime...';

        await ort.env.wasm.wasmReady;
        elements.loading.textContent = 'WASM runtime ready, loading classes...';

        const classesResponse = await fetch('coco_classes.json');
        if (!classesResponse.ok) throw new Error('Failed to load COCO classes');
        classes = await classesResponse.json();

        ort.env.wasm.numThreads = 1;

        elements.loading.textContent = 'Loading YOLOv8 model...';

        session = await ort.InferenceSession.create('model/yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        elements.loading.style.display = 'none';
        console.log('Model loaded with WASM backend');
    } catch (error) {
        console.error('WASM initialization error:', error);
        elements.loading.innerHTML = `
            <div class="error">WASM Error: ${error.message}</div>
            <div>Tips:
                <ul>
                    <li>Enable WebAssembly in browser</li>
                    <li>Try disabling multithreading (done automatically)</li>
                    <li>Model file should be present at 'model/yolov8n.onnx'</li>
                </ul>
            </div>
        `;
    }
}

// (3) Обработчик загрузки изображения
elements.imageUpload.addEventListener('change', async (event) => {
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
            const inputTensor = preprocessImage(image);
            const input = new ort.Tensor('float32', inputTensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

            const outputs = await session.run({ images: input });
            // Выход: [1, 84, 8400] - нужно транспонировать к [8400, 84]
            const rawOutput = outputs[Object.keys(outputs)[0]].data;

            const transposedOutput = transposeOutput(rawOutput, 84, 8400);

            const detections = processOutput(transposedOutput, image.width, image.height);
            renderDetections(detections);
            displayResults(detections);
        } catch (error) {
            console.error('WASM processing error:', error);
            elements.results.innerHTML = `<div class="error">Processing Error: ${error.message}</div>`;
        } finally {
            elements.loading.style.display = 'none';
        }
    };
});

// (4) Предобработка изображения (resize + normalize + CHW)
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
        tensor[idx] = data[i] / 255; // R
        tensor[idx + INPUT_SIZE * INPUT_SIZE] = data[i + 1] / 255; // G
        tensor[idx + 2 * INPUT_SIZE * INPUT_SIZE] = data[i + 2] / 255; // B
    }
    return tensor;
}

// (5) Транспонирование выхода модели [84, 8400] -> [8400, 84]
function transposeOutput(data, dim0, dim1) {
    const transposed = new Float32Array(dim0 * dim1);
    for (let i = 0; i < dim0; i++) {
        for (let j = 0; j < dim1; j++) {
            transposed[j * dim0 + i] = data[i * dim1 + j];
        }
    }
    return transposed;
}

// (6) Обработка выхода модели и фильтрация по confidence, nms
function processOutput(predictions, originalWidth, originalHeight) {
    const detections = [];

    // Константы для масштабирования
    const ratio = Math.min(INPUT_SIZE / originalWidth, INPUT_SIZE / originalHeight);
    const newWidth = originalWidth * ratio;
    const newHeight = originalHeight * ratio;
    const xOffset = (INPUT_SIZE - newWidth) / 2;
    const yOffset = (INPUT_SIZE - newHeight) / 2;

    const numDetections = predictions.length / NUM_ATTRIBUTES;

    for (let i = 0; i < numDetections; i++) {
        const base = i * NUM_ATTRIBUTES;

        const cx = predictions[base];
        const cy = predictions[base + 1];
        const w = predictions[base + 2];
        const h = predictions[base + 3];
        const objConf = predictions[base + 4];

        if (objConf < CONFIDENCE_THRESHOLD || isNaN(objConf)) continue;

        // Поиск класса с максимальной оценкой
        let maxClassScore = 0;
        let classId = -1;

        for (let c = 0; c < NUM_CLASSES; c++) {
            const classScore = predictions[base + 5 + c];
            if (classScore > maxClassScore) {
                maxClassScore = classScore;
                classId = c;
            }
        }

        if (classId === -1) continue;

        // Итоговая confidence = objConf * maxClassScore
        const confidence = objConf * maxClassScore;
        if (confidence < CLASS_SCORE_THRESHOLD || isNaN(confidence) || confidence > 1.0) continue;

        // Координаты bbox с учетом масштабирования и смещения
        const x1 = (cx - w / 2 - xOffset) / ratio;
        const y1 = (cy - h / 2 - yOffset) / ratio;
        const x2 = (cx + w / 2 - xOffset) / ratio;
        const y2 = (cy + h / 2 - yOffset) / ratio;

        const boxX = Math.max(0, x1);
        const boxY = Math.max(0, y1);
        const boxW = Math.min(originalWidth, x2) - boxX;
        const boxH = Math.min(originalHeight, y2) - boxY;

        if (boxW <= 0 || boxH <= 0) continue;

        detections.push({
            classId,
            className: classes[classId] ?? `Class ${classId}`,
            confidence,
            bbox: [boxX, boxY, boxW, boxH]
        });
    }

    return nms(detections);
}

// (7) Non-Maximum Suppression (NMS) для устранения пересекающихся боксов
function nms(detections) {
    detections.sort((a, b) => b.confidence - a.confidence);
    const filtered = [];

    while (detections.length) {
        const current = detections.shift();
        filtered.push(current);

        detections = detections.filter(det => {
            if (det.classId !== current.classId) return true;
            const iou = calculateIoU(current.bbox, det.bbox);
            return iou < IOU_THRESHOLD;
        });
    }

    return filtered;
}

// (8) Вычисление IoU между двумя боксами
function calculateIoU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    const xi1 = Math.max(x1, x2);
    const yi1 = Math.max(y1, y2);
    const xi2 = Math.min(x1 + w1, x2 + w2);
    const yi2 = Math.min(y1 + h1, y2 + h2);

    const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;

    return interArea / (box1Area + box2Area - interArea);
}

// (9) Отрисовка боксов и меток на canvas поверх изображения
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

// (10) Отображение результатов в текстовом блоке
function displayResults(detections) {
    if (detections.length === 0) {
        elements.results.innerHTML = '<div class="detection-item">No objects detected</div>';
        return;
    }

    elements.results.innerHTML = detections.map(det => {
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

// (11) Генерация цвета для класса (для разных bbox разный цвет)
function getColorForClass(classId) {
    const hue = (classId * 137.508) % 360; // золотое сечение для хорошего распределения цветов
    return `hsl(${hue}, 90%, 50%)`;
}

// (12) Запуск и инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', init);