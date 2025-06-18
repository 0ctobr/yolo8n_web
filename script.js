// Конфигурация модели
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const CLASS_SCORE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

// Глобальные переменные
let session = null;
let classes = [];

// Элементы DOM
const elements = {
    imageUpload: document.getElementById('imageUpload'),
    preview: document.getElementById('preview'),
    canvas: document.getElementById('canvas'),
    results: document.getElementById('results'),
    loading: document.getElementById('loading')
};

// Инициализация приложения
async function init() {
    try {
        // Показать индикатор загрузки
        elements.loading.style.display = 'block';
        
        // Загрузка классов COCO
        const response = await fetch('coco_classes.json');
        classes = await response.json();
        
        // Загрузка модели ONNX
        session = await ort.InferenceSession.create('./model/yolov8n.onnx', {
            executionProviders: ['webgl'] // Используем WebGL для ускорения
        });
        
        // Скрыть индикатор загрузки
        elements.loading.style.display = 'none';
        
        console.log('Model and classes loaded successfully');
    } catch (error) {
        console.error('Initialization error:', error);
        elements.loading.textContent = 'Error loading model. See console for details.';
    }
}

// Обработчик загрузки изображения
elements.imageUpload.addEventListener('change', handleImageUpload);

async function handleImageUpload(event) {
    if (!session) {
        alert('Model is still loading. Please wait...');
        return;
    }
    
    const file = event.target.files[0];
    if (!file) return;
    
    const image = new Image();
    image.src = URL.createObjectURL(file);
    
    image.onload = async () => {
        // Отобразить изображение
        elements.preview.src = image.src;
        
        // Очистить предыдущие результаты
        elements.results.innerHTML = '';
        
        // Показать индикатор обработки
        elements.loading.textContent = 'Processing image...';
        elements.loading.style.display = 'block';
        
        try {
            // Препроцессинг и запуск модели
            const tensor = preprocessImage(image);
            const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
            const outputs = await session.run({ images: input });
            const predictions = outputs.output0.data;
            
            // Постобработка результатов
            const detections = processOutput(predictions, image.width, image.height);
            
            // Отрисовка результатов
            renderDetections(detections);
            displayResults(detections);
        } catch (error) {
            console.error('Detection error:', error);
            elements.results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        } finally {
            // Скрыть индикатор обработки
            elements.loading.style.display = 'none';
        }
    };
}

// Препроцессинг изображения
function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    
    // Рассчитать соотношение сторон
    const ratio = Math.min(INPUT_SIZE / image.width, INPUT_SIZE / image.height);
    const newWidth = Math.round(image.width * ratio);
    const newHeight = Math.round(image.height * ratio);
    
    // Отрисовать изображение с центрированием
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    const xOffset = (INPUT_SIZE - newWidth) / 2;
    const yOffset = (INPUT_SIZE - newHeight) / 2;
    ctx.drawImage(image, xOffset, yOffset, newWidth, newHeight);
    
    // Получить данные изображения
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const data = imageData.data;
    const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // Нормализация и преобразование в CHW-формат
    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        tensor[idx] = data[i] / 255;         // R
        tensor[idx + INPUT_SIZE * INPUT_SIZE] = data[i + 1] / 255; // G
        tensor[idx + 2 * INPUT_SIZE * INPUT_SIZE] = data[i + 2] / 255; // B
    }
    
    return tensor;
}

// Обработка выходных данных модели
function processOutput(predictions, originalWidth, originalHeight) {
    const detections = [];
    
    // Рассчитать масштаб и смещение
    const ratio = Math.min(INPUT_SIZE / originalWidth, INPUT_SIZE / originalHeight);
    const newWidth = originalWidth * ratio;
    const newHeight = originalHeight * ratio;
    const xOffset = (INPUT_SIZE - newWidth) / 2;
    const yOffset = (INPUT_SIZE - newHeight) / 2;
    
    // Обработка 8400 возможных детекций
    for (let i = 0; i < 8400; i++) {
        const baseIndex = i * 84;
        const confidence = predictions[baseIndex + 4];
        
        // Фильтрация по порогу уверенности
        if (confidence < CONFIDENCE_THRESHOLD) continue;
        
        // Поиск класса с максимальной вероятностью
        let maxScore = 0;
        let classId = 0;
        for (let c = 0; c < 80; c++) {
            const score = predictions[baseIndex + 5 + c] * confidence;
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }
        
        // Фильтрация по порогу класса
        if (maxScore < CLASS_SCORE_THRESHOLD) continue;
        
        // Извлечение координат bounding box
        const cx = predictions[baseIndex];
        const cy = predictions[baseIndex + 1];
        const w = predictions[baseIndex + 2];
        const h = predictions[baseIndex + 3];
        
        // Преобразование координат в оригинальный размер
        const x1 = (cx - w / 2 - xOffset) / ratio;
        const y1 = (cy - h / 2 - yOffset) / ratio;
        const x2 = (cx + w / 2 - xOffset) / ratio;
        const y2 = (cy + h / 2 - yOffset) / ratio;
        
        // Сохранение детекции
        detections.push({
            classId,
            className: classes[classId],
            confidence: maxScore,
            bbox: [
                Math.max(0, x1),
                Math.max(0, y1),
                Math.min(originalWidth, x2) - Math.max(0, x1),
                Math.min(originalHeight, y2) - Math.max(0, y1)
            ]
        });
    }
    
    // Применение Non-Maximum Suppression
    return nms(detections);
}

// Non-Maximum Suppression
function nms(detections) {
    detections.sort((a, b) => b.confidence - a.confidence);
    const filtered = [];
    
    while (detections.length > 0) {
        const current = detections[0];
        filtered.push(current);
        
        detections = detections.filter(det => {
            if (det.classId !== current.classId) return true;
            const iou = calculateIoU(current.bbox, det.bbox);
            return iou < IOU_THRESHOLD;
        });
    }
    
    return filtered;
}

// Расчет Intersection over Union
function calculateIoU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    
    const xLeft = Math.max(x1, x2);
    const yTop = Math.max(y1, y2);
    const xRight = Math.min(x1 + w1, x2 + w2);
    const yBottom = Math.min(y1 + h1, y2 + h2);
    
    if (xRight < xLeft || yBottom < yTop) return 0;
    
    const intersection = (xRight - xLeft) * (yBottom - yTop);
    const area1 = w1 * h1;
    const area2 = w2 * h2;
    
    return intersection / (area1 + area2 - intersection);
}

// Отрисовка bounding boxes
function renderDetections(detections) {
    const ctx = elements.canvas.getContext('2d');
    const preview = elements.preview;
    
    // Установка размеров canvas
    elements.canvas.width = preview.width;
    elements.canvas.height = preview.height;
    
    // Очистка canvas
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    
    // Отрисовка каждого обнаружения
    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const color = getColorForClass(det.classId);
        
        // Рисование bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        
        // Рисование фона для текста
        ctx.fillStyle = color;
        const text = `${det.className} ${(det.confidence * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(x - 1, y - 20, textWidth + 10, 20);
        
        // Рисование текста
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(text, x + 4, y - 4);
    });
}

// Отображение результатов
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

// Генерация цвета для класса
function getColorForClass(classId) {
    const hue = (classId * 137.508) % 360; // Использование золотого угла
    return `hsl(${hue}, 90%, 50%)`;
}

// Инициализация приложения при загрузке страницы
document.addEventListener('DOMContentLoaded', init);