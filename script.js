document.addEventListener('DOMContentLoaded', async () => {
    // DOM элементы
    const imageUpload = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const loading = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    
    let session = null;
    let classes = null;
    let isModelReady = false;
    
    // Инициализация модели и классов
    async function init() {
        showLoading(true, "Initializing WASM runtime...");
        try {
            // Инициализация ONNX Runtime
            await ort.init();
            
            // Загрузка классов COCO
            const response = await fetch('coco_classes.json');
            classes = await response.json();
            
            // Загрузка модели YOLOv8n
            showLoading(true, "Loading YOLOv8n model...");
            session = await ort.InferenceSession.create(
                './model/yolov8n.onnx', 
                { executionProviders: ['wasm'] }
            );
            
            isModelReady = true;
            showLoading(false);
            console.log("Model and WASM runtime initialized");
        } catch (error) {
            console.error("Initialization failed:", error);
            showLoading(false, `Error: ${error.message}`);
        }
    }

    // Показать/скрыть загрузку
    function showLoading(show, message = "Processing...") {
        loading.textContent = message;
        loading.style.display = show ? 'block' : 'none';
    }

    // Обработка загрузки изображения
    imageUpload.addEventListener('change', async (e) => {
        if (!e.target.files || !e.target.files[0]) return;
        
        const file = e.target.files[0];
        const reader = new FileReader();
        
        reader.onload = async (event) => {
            // Отображаем превью
            preview.src = event.target.result;
            
            preview.onload = async () => {
                if (!isModelReady) {
                    showLoading(true, "Model not ready, initializing...");
                    await init();
                }
                
                try {
                    showLoading(true, "Detecting objects...");
                    
                    // Обработка изображения
                    const { inputTensor, xRatio, yRatio } = preprocessImage(preview);
                    
                    // Выполнение вывода
                    const output = await runInference(inputTensor);
                    
                    // Обработка результатов
                    const detections = processOutput(output, preview, xRatio, yRatio);
                    
                    // Отрисовка результатов
                    drawDetections(detections);
                    displayResults(detections);
                    
                    showLoading(false);
                } catch (error) {
                    console.error("Detection error:", error);
                    showLoading(false, `Detection failed: ${error.message}`);
                }
            };
        };
        
        reader.readAsDataURL(file);
    });

    // Предварительная обработка изображения
    function preprocessImage(img) {
        const modelSize = 640;
        const channels = 3;
        
        // Рассчитываем соотношения сторон
        const xRatio = img.width / modelSize;
        const yRatio = img.height / modelSize;
        
        // Создаем временный canvas для ресайза
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = modelSize;
        tempCanvas.height = modelSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Рисуем изображение с сохранением пропорций
        tempCtx.fillStyle = '#121212'; // Цвет фона
        tempCtx.fillRect(0, 0, modelSize, modelSize);
        
        const scale = Math.min(modelSize / img.width, modelSize / img.height);
        const newWidth = img.width * scale;
        const newHeight = img.height * scale;
        const xOffset = (modelSize - newWidth) / 2;
        const yOffset = (modelSize - newHeight) / 2;
        
        tempCtx.drawImage(
            img, 
            0, 0, img.width, img.height,
            xOffset, yOffset, newWidth, newHeight
        );
        
        // Получаем данные изображения
        const imageData = tempCtx.getImageData(0, 0, modelSize, modelSize);
        
        // Преобразуем в тензор (нормализация 0-1, формат CHW)
        const inputData = new Float32Array(channels * modelSize * modelSize);
        for (let i = 0; i < imageData.data.length; i += 4) {
            const pixelIndex = i / 4;
            inputData[pixelIndex] = imageData.data[i] / 255;       // R
            inputData[pixelIndex + modelSize * modelSize] = imageData.data[i + 1] / 255; // G
            inputData[pixelIndex + 2 * modelSize * modelSize] = imageData.data[i + 2] / 255; // B
        }
        
        return {
            inputTensor: new ort.Tensor('float32', inputData, [1, channels, modelSize, modelSize]),
            xRatio,
            yRatio
        };
    }

    // Выполнение вывода модели
    async function runInference(inputTensor) {
        const feeds = { 
            images: inputTensor 
        };
        
        const results = await session.run(feeds);
        return results.output0.data;
    }

    // Обработка выходных данных модели
    function processOutput(output, img, xRatio, yRatio) {
        const detections = [];
        const outputSize = 84; // 4 bbox + 80 classes
        const numDetections = output.length / outputSize;
        
        for (let i = 0; i < numDetections; i++) {
            const offset = i * outputSize;
            
            // Извлекаем параметры bbox
            const xc = output[offset];
            const yc = output[offset + 1];
            const w = output[offset + 2];
            const h = output[offset + 3];
            
            // Рассчитываем координаты
            const x1 = (xc - w / 2) * xRatio;
            const y1 = (yc - h / 2) * yRatio;
            const x2 = (xc + w / 2) * xRatio;
            const y2 = (yc + h / 2) * yRatio;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Находим класс с максимальной уверенностью
            let maxProb = -1;
            let classId = -1;
            for (let j = 4; j < 84; j++) {
                const prob = output[offset + j];
                if (prob > maxProb) {
                    maxProb = prob;
                    classId = j - 4;
                }
            }
            
            // Применяем сигмоиду для получения вероятности
            const confidence = 1 / (1 + Math.exp(-maxProb));
            
            if (confidence > 0.5 && classId in classes) {
                detections.push({
                    bbox: [x1, y1, width, height],
                    classId,
                    className: classes[classId],
                    confidence,
                    color: `hsl(${(classId * 137) % 360}, 80%, 60%)`
                });
            }
        }
        
        // Применяем Non-Maximum Suppression (NMS)
        return applyNMS(detections, 0.45);
    }

    // Non-Maximum Suppression
    function applyNMS(detections, iouThreshold) {
        detections.sort((a, b) => b.confidence - a.confidence);
        
        const finalDetections = [];
        while (detections.length > 0) {
            const current = detections.shift();
            finalDetections.push(current);
            
            for (let i = detections.length - 1; i >= 0; i--) {
                if (current.classId !== detections[i].classId) continue;
                
                const iou = calculateIOU(current.bbox, detections[i].bbox);
                if (iou > iouThreshold) {
                    detections.splice(i, 1);
                }
            }
        }
        
        return finalDetections;
    }

    // Расчет Intersection over Union (IoU)
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

    // Отрисовка обнаруженных объектов
    function drawDetections(detections) {
        // Настройка canvas
        canvas.width = preview.width;
        canvas.height = preview.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Рисуем каждый обнаруженный объект
        detections.forEach(det => {
            const [x, y, width, height] = det.bbox;
            
            // Рисуем bounding box
            ctx.strokeStyle = det.color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);
            
            // Рисуем фон для текста
            ctx.fillStyle = det.color;
            const text = `${det.className} ${(det.confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(text).width;
            
            ctx.fillRect(
                x - 1,
                y - 20,
                textWidth + 10,
                20
            );
            
            // Рисуем текст
            ctx.fillStyle = 'white';
            ctx.font = '14px Arial';
            ctx.textBaseline = 'top';
            ctx.fillText(text, x + 4, y - 18);
        });
    }

    // Отображение результатов в виде списка
    function displayResults(detections) {
        resultsDiv.innerHTML = '';
        
        if (detections.length === 0) {
            resultsDiv.innerHTML = '<div class="detection-item">No objects detected</div>';
            return;
        }
        
        detections.forEach(det => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            
            item.innerHTML = `
                <div class="detection-color" style="background-color: ${det.color}"></div>
                <div class="detection-info">
                    ${det.className} - ${(det.confidence * 100).toFixed(1)}%
                </div>
            `;
            
            resultsDiv.appendChild(item);
        });
    }

    // Инициализация при загрузке страницы
    init();
});