document.addEventListener('DOMContentLoaded', init);

// Глобальные переменные
let session = null;
let classNames = [];
let isWasmLoaded = false;
const inputSize = 640;

// Цвета для разных классов
const colors = [
    '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A',
    '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF',
    '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF',
    '#FF95C8', '#FF37C7'
];

// Инициализация приложения
async function init() {
    try {
        showLoading(true, "Loading ONNX Runtime and model...");
        await ort.env.wasm.init();
        isWasmLoaded = true;
        
        showLoading(true, "Loading YOLOv8n model...");
        session = await ort.InferenceSession.create(
            './model/yolov8n.onnx', 
            { executionProviders: ['wasm'] }
        );
        
        const response = await fetch('coco_classes.json');
        classNames = await response.json();
        
        showLoading(false);
        document.getElementById('imageUpload').disabled = false;
        console.log("Model and runtime initialized successfully");
    } catch (error) {
        console.error("Initialization failed:", error);
        showLoading(false);
        alert(`Initialization failed: ${error.message}`);
    }
}

function showLoading(show, message = "Processing...") {
    const loading = document.getElementById('loading');
    if (show) {
        loading.textContent = message;
        loading.style.display = 'block';
    } else {
        loading.style.display = 'none';
    }
}

// Обработка загрузки изображения
document.getElementById('imageUpload').addEventListener('change', async function(e) {
    if (!session) return alert("Model not loaded yet");
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    const image = new Image();
    const reader = new FileReader();
    
    reader.onload = async function(event) {
        try {
            showLoading(true, "Processing image...");
            image.src = event.target.result;
            
            image.onload = async function() {
                const preview = document.getElementById('preview');
                preview.src = image.src;
                await new Promise(resolve => setTimeout(resolve, 50));
                await detectObjects(image);
            };
        } catch (error) {
            console.error("Image processing error:", error);
            alert(`Error: ${error.message}`);
            showLoading(false);
        }
    };
    reader.readAsDataURL(file);
});

async function detectObjects(image) {
    try {
        const { tensor, padding } = preprocessImage(image);
        
        showLoading(true, "Running object detection...");
        const startTime = performance.now();
        const outputs = await session.run({ images: tensor });
        const endTime = performance.now();
        console.log(`Inference time: ${(endTime - startTime).toFixed(1)}ms`);
        
        // Передаем image для преобразования координат
        const detections = processOutput(outputs, padding, image);
        
        drawResults(detections, image);
        showResults(detections, endTime - startTime);
    } catch (error) {
        console.error("Detection error:", error);
        alert(`Detection failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = inputSize;
    canvas.height = inputSize;
    
    const scale = Math.min(
        inputSize / image.naturalWidth,
        inputSize / image.naturalHeight
    );
    
    const newWidth = image.naturalWidth * scale;
    const newHeight = image.naturalHeight * scale;
    const padX = (inputSize - newWidth) / 2;
    const padY = (inputSize - newHeight) / 2;
    
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(image, padX, padY, newWidth, newHeight);
    
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
    const tensorData = new Float32Array(3 * inputSize * inputSize);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
        const idx = i / 4;
        tensorData[idx] = imageData.data[i] / 255;
        tensorData[idx + inputSize * inputSize] = imageData.data[i + 1] / 255;
        tensorData[idx + 2 * inputSize * inputSize] = imageData.data[i + 2] / 255;
    }
    
    return {
        tensor: new ort.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]),
        padding: { x: padX, y: padY, scale }
    };
}

// Обработка вывода модели для nms=True
function processOutput(outputs, padding, image) {
    const output = outputs.output0.data;
    const dims = outputs.output0.dims;
    const detections = [];
    
    // Проверяем формат вывода (для nms=True)
    if (dims.length === 3 && dims[2] === 6) {
        // Формат [1, num_detections, 6]
        const numDetections = dims[1];
        
        for (let i = 0; i < numDetections; i++) {
            const offset = i * 6;
            const confidence = output[offset + 4];
            const classId = output[offset + 5];
            
            if (confidence > 0.5) {
                const x1 = output[offset];
                const y1 = output[offset + 1];
                const x2 = output[offset + 2];
                const y2 = output[offset + 3];
                
                // Преобразуем координаты
                const scale = padding.scale;
                const padX = padding.x;
                const padY = padding.y;
                
                let x1_orig = (x1 - padX) / scale;
                let y1_orig = (y1 - padY) / scale;
                let x2_orig = (x2 - padX) / scale;
                let y2_orig = (y2 - padY) / scale;
                
                // Обрезаем по границам изображения
                x1_orig = Math.max(0, x1_orig);
                y1_orig = Math.max(0, y1_orig);
                x2_orig = Math.min(image.naturalWidth, x2_orig);
                y2_orig = Math.min(image.naturalHeight, y2_orig);
                
                const width = x2_orig - x1_orig;
                const height = y2_orig - y1_orig;
                
                if (width > 0 && height > 0) {
                    detections.push({
                        x: x1_orig,
                        y: y1_orig,
                        width: width,
                        height: height,
                        confidence: confidence,
                        classId: classId
                    });
                }
            }
        }
    } else {
        console.error("Unexpected output format:", dims);
    }
    
    return detections;
}

function drawResults(detections, originalImage) {
    const preview = document.getElementById('preview');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = preview.offsetWidth;
    canvas.height = preview.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / originalImage.naturalWidth;
    const scaleY = canvas.height / originalImage.naturalHeight;
    
    detections.forEach(det => {
        const x = det.x * scaleX;
        const y = det.y * scaleY;
        const width = det.width * scaleX;
        const height = det.height * scaleY;
        
        const color = colors[det.classId % colors.length];
        
        // Рисуем bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        
        // Рисуем фон для текста
        const className = classNames[det.classId] || `Class ${det.classId}`;
        const label = `${className} ${(det.confidence * 100).toFixed(1)}%`;
        
        ctx.fillStyle = color;
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(label).width;
        const textHeight = 20;
        
        // Проверяем, чтобы текст не выходил за верхнюю границу
        const textY = y - textHeight < 0 ? y + height : y - 5;
        
        ctx.fillRect(x, textY - textHeight + 5, textWidth + 10, textHeight);
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 5, textY);
    });
}

function showResults(detections, inferenceTime) {
    const resultsContainer = document.getElementById('results');
    
    if (detections.length === 0) {
        resultsContainer.innerHTML = '<div class="detection-item">No objects detected</div>';
        return;
    }
    
    let html = `
        <div class="detection-item">
            <div class="detection-info" style="font-weight:bold">
                Detected ${detections.length} objects in ${inferenceTime.toFixed(1)}ms
            </div>
        </div>
    `;
    
    detections.forEach(det => {
        const className = classNames[det.classId] || `Class ${det.classId}`;
        const color = colors[det.classId % colors.length];
        
        html += `
        <div class="detection-item">
            <div class="detection-color" style="background-color: ${color}"></div>
            <div class="detection-info">
                ${className} - ${(det.confidence * 100).toFixed(1)}%
            </div>
        </div>
        `;
    });
    
    resultsContainer.innerHTML = html;
}