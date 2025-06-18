// ===== 1. КОНФИГУРАЦИЯ =====
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45; // Порог для Non-Maximum Suppression

let session = null;
let classes = [];

// DOM-элементы
const elements = {
    imageUpload: document.getElementById('imageUpload'),
    preview: document.getElementById('preview'),
    canvas: document.getElementById('canvas'),
    results: document.getElementById('results'),
    loading: document.getElementById('loading')
};

// ===== 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ =====
async function init() {
    try {
        elements.loading.style.display = 'block';
        elements.loading.textContent = 'Инициализация WASM...';
        
        await ort.env.wasm.wasmReady;
        ort.env.wasm.numThreads = 1;

        // Загрузка классов COCO
        const classesResponse = await fetch('coco_classes.json');
        if (!classesResponse.ok) throw new Error('Ошибка загрузки классов');
        classes = await classesResponse.json();

        // Загрузка модели YOLOv8
        elements.loading.textContent = 'Загрузка модели YOLOv8...';
        session = await ort.InferenceSession.create('model/yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        elements.loading.style.display = 'none';
    } catch (err) {
        console.error('Ошибка инициализации:', err);
        elements.loading.innerHTML = `<div class="error">${err.message}</div>`;
    }
}

// ===== 3. ОБРАБОТКА ИЗОБРАЖЕНИЙ =====
elements.imageUpload.addEventListener('change', handleImageUpload);

async function handleImageUpload(event) {
    if (!session) {
        alert('Модель не загружена!');
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    const image = new Image();
    image.src = URL.createObjectURL(file);

    image.onload = async () => {
        elements.preview.src = image.src;
        elements.canvas.width = image.width;
        elements.canvas.height = image.height;
        elements.results.innerHTML = '';
        
        elements.loading.style.display = 'block';
        elements.loading.textContent = 'Анализ изображения...';

        try {
            // 3.1 Препроцессинг изображения
            const { tensor, scale, pad } = preprocessImage(image);
            const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

            // 3.2 Выполнение предсказания
            const outputs = await session.run({ images: input });
            const outputTensor = outputs.output0; // Важно: имя выхода 'output0' для YOLOv8
            
            // 3.3 Постобработка результатов
            const detections = processOutput(
                outputTensor.data, 
                image.width, 
                image.height, 
                scale, 
                pad
            );
            
            renderDetections(detections);
            displayResults(detections);
        } catch (err) {
            console.error('Ошибка обработки:', err);
            elements.results.innerHTML = `<div class="error">${err.message}</div>`;
        } finally {
            elements.loading.style.display = 'none';
        }
    };
}

// 3.1 Препроцессинг изображения
function preprocessImage(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;

    // Масштабирование с сохранением пропорций
    const scale = Math.min(INPUT_SIZE / img.width, INPUT_SIZE / img.height);
    const newWidth = img.width * scale;
    const newHeight = img.height * scale;
    const xOffset = (INPUT_SIZE - newWidth) / 2;
    const yOffset = (INPUT_SIZE - newHeight) / 2;

    // Рисуем изображение с паддингом
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(img, xOffset, yOffset, newWidth, newHeight);

    // Нормализация данных [0-255] -> [0-1]
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const data = imageData.data;
    const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        tensor[idx] = data[i] / 255.0;         // R
        tensor[idx + INPUT_SIZE * INPUT_SIZE] = data[i + 1] / 255.0; // G
        tensor[idx + 2 * INPUT_SIZE * INPUT_SIZE] = data[i + 2] / 255.0; // B
    }

    return {
        tensor: tensor,
        scale: scale,
        pad: { x: xOffset, y: yOffset }
    };
}

// ===== 4. ОБРАБОТКА ВЫХОДА МОДЕЛИ =====
function processOutput(predictions, origWidth, origHeight, scale, pad) {
    const detections = [];
    const numClasses = classes.length;
    const gridSize = 8400; // 80*80 + 40*40 + 20*20 = 8400 для YOLOv8
    
    // 4.1 Парсинг выходного тензора формата [1, 84, 8400]
    for (let i = 0; i < gridSize; i++) {
        const startIdx = i * (4 + numClasses);
        const cx = predictions[startIdx] * INPUT_SIZE;     // center x
        const cy = predictions[startIdx + 1] * INPUT_SIZE; // center y
        const w = predictions[startIdx + 2] * INPUT_SIZE;  // width
        const h = predictions[startIdx + 3] * INPUT_SIZE;   // height
        
        // 4.2 Поиск класса с максимальной уверенностью
        let maxConfidence = 0;
        let classId = 0;
        for (let c = 0; c < numClasses; c++) {
            const confidence = predictions[startIdx + 4 + c];
            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                classId = c;
            }
        }
        
        if (maxConfidence < CONFIDENCE_THRESHOLD) continue;
        
        // 4.3 Преобразование координат (центр -> углы)
        const x1 = (cx - w / 2 - pad.x) / scale;
        const y1 = (cy - h / 2 - pad.y) / scale;
        const x2 = (cx + w / 2 - pad.x) / scale;
        const y2 = (cy + h / 2 - pad.y) / scale;
        
        // 4.4 Фильтрация выходящих за границы координат
        const bbox = [
            Math.max(0, x1),
            Math.max(0, y1),
            Math.min(origWidth, x2) - Math.max(0, x1),
            Math.min(origHeight, y2) - Math.max(0, y1)
        ];
        
        detections.push({
            classId,
            className: classes[classId],
            confidence: maxConfidence,
            bbox: bbox
        });
    }
    
    // 4.5 Применение Non-Maximum Suppression
    return nms(detections);
}

// 4.5 Алгоритм Non-Maximum Suppression
function nms(detections) {
    const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);
    const selected = [];
    
    while (sortedDetections.length > 0) {
        const current = sortedDetections.shift();
        selected.push(current);
        
        for (let i = sortedDetections.length - 1; i >= 0; i--) {
            const iou = calculateIOU(current.bbox, sortedDetections[i].bbox);
            if (iou > IOU_THRESHOLD) {
                sortedDetections.splice(i, 1);
            }
        }
    }
    
    return selected;
}

// Расчет Intersection over Union
function calculateIOU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    
    const interX1 = Math.max(x1, x2);
    const interY1 = Math.max(y1, y2);
    const interX2 = Math.min(x1 + w1, x2 + w2);
    const interY2 = Math.min(y1 + h1, y2 + h2);
    
    const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    
    return interArea / (box1Area + box2Area - interArea);
}

// ===== 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ =====
function renderDetections(detections) {
    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    
    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const color = getColorForClass(det.classId);
        
        // Рисуем bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        
        // Рисуем фон для текста
        const label = `${det.className} ${(det.confidence * 100).toFixed(1)}%`;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 18, ctx.measureText(label).width + 10, 18);
        
        // Рисуем текст
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(label, x + 5, y - 3);
    });
}

function displayResults(detections) {
    elements.results.innerHTML = detections.length === 0
        ? '<div class="detection-item">Объекты не обнаружены</div>'
        : detections.map(det => {
            const [x, y, w, h] = det.bbox;
            const color = getColorForClass(det.classId);
            
            return `
            <div class="detection-item">
                <div class="detection-color" style="background: ${color}"></div>
                <div class="detection-info">
                    <strong>${det.className}</strong> (${(det.confidence * 100).toFixed(1)}%)
                    <div>X: ${x.toFixed(0)}, Y: ${y.toFixed(0)}, Ш: ${w.toFixed(0)}, В: ${h.toFixed(0)}</div>
                </div>
            </div>`;
        }).join('');
}

function getColorForClass(classId) {
    const hue = (classId * 137.508) % 360; // Золотой угол для разнообразия цветов
    return `hsl(${hue}, 90%, 50%)`;
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', init);