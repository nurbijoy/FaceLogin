// Training UI JavaScript
let video, canvas, ctx;
let realImages = [];
let fakeImages = [];
let stream = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    // Event listeners
    document.getElementById('startCamera').addEventListener('click', startCamera);
    document.getElementById('captureRealBtn').addEventListener('click', () => captureImage('real'));
    document.getElementById('captureFakeBtn').addEventListener('click', () => captureImage('fake'));
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('retrainBtn').addEventListener('click', retrainModel);
    document.getElementById('clearBtn').addEventListener('click', clearData);
    document.getElementById('trainAgainBtn').addEventListener('click', () => {
        document.getElementById('resultsPanel').classList.add('hidden');
        clearData();
    });
    
    document.getElementById('viewPerformanceBtn').addEventListener('click', viewModelPerformance);
    document.getElementById('closePerformanceBtn').addEventListener('click', () => {
        document.getElementById('currentPerformancePanel').classList.add('hidden');
    });

    // Settings modal handlers
    document.getElementById('settingsBtn').addEventListener('click', () => {
        document.getElementById('settingsModal').classList.remove('hidden');
    });
    document.getElementById('closeSettingsBtn').addEventListener('click', () => {
        document.getElementById('settingsModal').classList.add('hidden');
    });
    document.getElementById('saveSettingsBtn').addEventListener('click', () => {
        document.getElementById('settingsModal').classList.add('hidden');
        showMessage('Settings saved successfully', 'success');
    });
    document.getElementById('resetSettingsBtn').addEventListener('click', () => {
        document.getElementById('epochs').value = 20;
        document.getElementById('batchSize').value = 16;
        document.getElementById('confidenceThreshold').value = 90;
        showMessage('Settings reset to defaults', 'info');
    });



    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' || e.key === 'R') {
            captureImage('real');
        } else if (e.key === 'f' || e.key === 'F') {
            captureImage('fake');
        } else if (e.key === 't' || e.key === 'T') {
            if (!document.getElementById('trainBtn').disabled) {
                trainModel();
            }
        }
    });

    // Load existing training stats
    loadTrainingStats();
});

function updateTrainButtonState() {
    const minImages = 20; // CNN requires minimum 20 images
    const hasEnoughData = realImages.length >= minImages && fakeImages.length >= minImages;
    document.getElementById('trainBtn').disabled = !hasEnoughData;
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        
        document.getElementById('overlay').classList.add('hidden');
        document.getElementById('captureRealBtn').disabled = false;
        document.getElementById('captureFakeBtn').disabled = false;
        
        updateInstructions('real');
        updateStepIndicator(1);
        
        showMessage('Camera started! Press R for Real, F for Fake', 'success');
    } catch (err) {
        console.error('Camera error:', err);
        showMessage('Cannot access camera. Please check permissions.', 'error');
    }
}

function captureImage(type) {
    // Flash effect
    const flash = document.getElementById('flash');
    flash.classList.remove('hidden');
    setTimeout(() => flash.classList.add('hidden'), 100);

    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to blob
    canvas.toBlob((blob) => {
        const imageData = {
            blob: blob,
            url: URL.createObjectURL(blob),
            timestamp: Date.now()
        };

        if (type === 'real') {
            realImages.push(imageData);
            updateCount('real');
            addToGallery(imageData.url, 'real', realImages.length - 1);
            showMessage(`Real image ${realImages.length} captured!`, 'success');
            
            if (realImages.length >= 10 && fakeImages.length === 0) {
                updateInstructions('fake');
                updateStepIndicator(2);
            }
        } else {
            fakeImages.push(imageData);
            updateCount('fake');
            addToGallery(imageData.url, 'fake', fakeImages.length - 1);
            showMessage(`Fake image ${fakeImages.length} captured!`, 'success');
            
            if (fakeImages.length >= 10) {
                updateInstructions('train');
                updateStepIndicator(3);
            }
        }

        // Enable train button if we have enough data
        updateTrainButtonState();

        // Save to localStorage
        saveData();
    }, 'image/jpeg', 0.95);
}

function updateCount(type) {
    const count = type === 'real' ? realImages.length : fakeImages.length;
    document.getElementById(`${type}Count`).textContent = count;
}

function addToGallery(url, type, index) {
    const gallery = document.getElementById(`${type}Gallery`);
    
    // Remove "no images" message
    if (gallery.querySelector('p')) {
        gallery.innerHTML = '';
    }

    const imgContainer = document.createElement('div');
    imgContainer.className = 'relative group';
    imgContainer.innerHTML = `
        <img src="${url}" class="w-full h-20 object-cover rounded border-2 ${type === 'real' ? 'border-green-300' : 'border-red-300'}">
        <button onclick="removeImage('${type}', ${index})" class="absolute top-1 right-1 bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center opacity-0 group-hover:opacity-100 transition">
            <i class="fas fa-times text-xs"></i>
        </button>
    `;
    gallery.appendChild(imgContainer);
}

function removeImage(type, index) {
    if (type === 'real') {
        URL.revokeObjectURL(realImages[index].url);
        realImages.splice(index, 1);
    } else {
        URL.revokeObjectURL(fakeImages[index].url);
        fakeImages.splice(index, 1);
    }
    
    updateCount(type);
    rebuildGallery(type);
    saveData();
    
    // Update train button state based on model type
    updateTrainButtonState();
}

function rebuildGallery(type) {
    const gallery = document.getElementById(`${type}Gallery`);
    gallery.innerHTML = '';
    
    const images = type === 'real' ? realImages : fakeImages;
    
    if (images.length === 0) {
        gallery.innerHTML = '<p class="col-span-3 text-gray-400 text-sm text-center py-8">No images yet</p>';
    } else {
        images.forEach((img, index) => {
            addToGallery(img.url, type, index);
        });
    }
}

async function trainModel() {
    const minImages = 20; // CNN requires minimum 20 images
    
    if (realImages.length < minImages || fakeImages.length < minImages) {
        showMessage(`Need at least ${minImages} real and ${minImages} fake images for CNN training!`, 'error');
        return;
    }

    // Show training panel
    document.getElementById('trainingPanel').classList.remove('hidden');
    document.getElementById('trainBtn').disabled = true;
    
    const startTime = Date.now();

    try {
        // Prepare form data
        const formData = new FormData();
        
        // Add CNN configuration
        const epochs = document.getElementById('epochs').value;
        const batchSize = document.getElementById('batchSize').value;
        const confidenceThreshold = document.getElementById('confidenceThreshold').value;
        
        formData.append('epochs', epochs);
        formData.append('batch_size', batchSize);
        formData.append('confidence_threshold', confidenceThreshold);
        
        updateProgress(10, `Training CNN with ${epochs} epochs...`);
        
        // Add real images
        for (let i = 0; i < realImages.length; i++) {
            formData.append('real_images', realImages[i].blob, `real_${i}.jpg`);
        }
        
        // Add fake images
        for (let i = 0; i < fakeImages.length; i++) {
            formData.append('fake_images', fakeImages[i].blob, `fake_${i}.jpg`);
        }

        // Update progress
        updateProgress(20, 'Uploading images...');

        // Send to server
        const response = await fetch('http://localhost:8000/api/train', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Training failed');
        }

        const startResult = await response.json();
        
        updateProgress(30, 'Building CNN architecture...');
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Poll for real training progress
        const totalEpochs = parseInt(document.getElementById('epochs').value);
        updateProgress(40, `Training CNN: Starting...`);
        
        let result = null;
        
        // Start polling for real-time progress
        const progressInterval = setInterval(async () => {
            try {
                const progressRes = await fetch('http://localhost:8000/api/training-progress');
                const progress = await progressRes.json();
                
                if (progress.epoch > 0 && progress.status !== 'idle') {
                    const percentComplete = 40 + (progress.epoch / progress.total_epochs) * 55;
                    
                    // Only show metrics if they exist (after epoch completes)
                    let statusText = `Epoch ${progress.epoch}/${progress.total_epochs}`;
                    if (progress.loss > 0 || progress.accuracy > 0) {
                        statusText += ` - Loss: ${progress.loss.toFixed(4)}, ` +
                                     `Acc: ${(progress.accuracy * 100).toFixed(1)}%, ` +
                                     `Val Loss: ${progress.val_loss.toFixed(4)}, ` +
                                     `Val Acc: ${(progress.val_accuracy * 100).toFixed(1)}%`;
                    } else {
                        statusText += ` - Training...`;
                    }
                    updateProgress(percentComplete, statusText);
                }
                
                // Check if training is complete
                const resultRes = await fetch('http://localhost:8000/api/training-result');
                const resultData = await resultRes.json();
                
                if (resultData.complete) {
                    clearInterval(progressInterval);
                    
                    if (resultData.status === 'error') {
                        throw new Error(resultData.error);
                    }
                    
                    if (resultData.status === 'complete') {
                        result = resultData.result;
                        updateProgress(100, 'Training complete!');
                        
                        // Calculate training time
                        const trainingTime = ((Date.now() - startTime) / 1000).toFixed(1);

                        // Show detailed results
                        setTimeout(() => {
                            document.getElementById('trainingPanel').classList.add('hidden');
                            showDetailedResults(result, trainingTime);
                            showMessage('CNN model trained successfully!', 'success');
                        }, 1000);
                    }
                }
            } catch (e) {
                clearInterval(progressInterval);
                console.error('Progress polling error:', e);
                document.getElementById('trainingPanel').classList.add('hidden');
                document.getElementById('trainBtn').disabled = false;
                showMessage(`Training failed: ${e.message}`, 'error');
            }
        }, 1000); // Poll every second


    } catch (error) {
        console.error('Training error:', error);
        document.getElementById('trainingPanel').classList.add('hidden');
        document.getElementById('trainBtn').disabled = false;
        showMessage(`Training failed: ${error.message}`, 'error');
    }
}

function updateProgress(percent, status) {
    document.getElementById('progressBar').style.width = `${percent}%`;
    document.getElementById('trainingStatus').textContent = status;
}

async function clearData() {
    const confirmed = await showConfirmModal(
        'Clear All Data',
        'Are you sure you want to clear all captured images? This action cannot be undone.',
        'fa-trash',
        'bg-red-100 text-red-600'
    );
    
    if (!confirmed) {
        return;
    }

    // Revoke URLs
    realImages.forEach(img => URL.revokeObjectURL(img.url));
    fakeImages.forEach(img => URL.revokeObjectURL(img.url));

    realImages = [];
    fakeImages = [];

    updateCount('real');
    updateCount('fake');

    rebuildGallery('real');
    rebuildGallery('fake');

    document.getElementById('trainBtn').disabled = true;
    
    localStorage.removeItem('trainingData');
    
    updateInstructions('real');
    updateStepIndicator(1);
    
    showMessage('All data cleared', 'info');
}

function saveData() {
    // Save image data URLs to localStorage (for persistence)
    const data = {
        realCount: realImages.length,
        fakeCount: fakeImages.length,
        timestamp: Date.now()
    };
    localStorage.setItem('trainingData', JSON.stringify(data));
}

async function loadTrainingStats() {
    try {
        const response = await fetch('http://localhost:8000/api/training-stats');
        const stats = await response.json();
        
        if (stats.total_images > 0) {
            showMessage(`Existing training data: ${stats.total_real_images} real, ${stats.total_fake_images} fake images saved`, 'info');
            
            // Enable retrain button if we have enough data for CNN (minimum 20 each)
            if (stats.total_real_images >= 20 && stats.total_fake_images >= 20) {
                document.getElementById('retrainBtn').disabled = false;
            }
        }
    } catch (error) {
        console.error('Failed to load training stats:', error);
    }
}

async function retrainModel() {
    const confirmed = await showConfirmModal(
        'Retrain CNN Model',
        'Retrain CNN model using all saved training data? This will update the current model.',
        'fa-sync-alt',
        'bg-blue-100 text-blue-600'
    );
    
    if (!confirmed) {
        return;
    }

    // Show training panel
    document.getElementById('trainingPanel').classList.remove('hidden');
    document.getElementById('retrainBtn').disabled = true;
    
    const startTime = Date.now();

    try {
        updateProgress(20, 'Loading saved training data...');

        // Prepare form data with CNN configuration
        const formData = new FormData();
        const epochs = document.getElementById('epochs').value;
        const batchSize = document.getElementById('batchSize').value;
        const confidenceThreshold = document.getElementById('confidenceThreshold').value;
        
        formData.append('epochs', epochs);
        formData.append('batch_size', batchSize);
        formData.append('confidence_threshold', confidenceThreshold);
        
        updateProgress(30, `Retraining CNN with ${epochs} epochs...`);

        // Send retrain request
        const response = await fetch('http://localhost:8000/api/retrain', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Retraining failed');
        }

        const startResult = await response.json();

        // Poll for real training progress
        const totalEpochsRetrain = parseInt(document.getElementById('epochs').value);
        updateProgress(40, `Training CNN: Starting...`);
        
        let result = null;
        
        // Start polling for real-time progress
        const progressIntervalRetrain = setInterval(async () => {
            try {
                const progressRes = await fetch('http://localhost:8000/api/training-progress');
                const progress = await progressRes.json();
                
                if (progress.epoch > 0 && progress.status !== 'idle') {
                    const percentComplete = 40 + (progress.epoch / progress.total_epochs) * 55;
                    
                    // Only show metrics if they exist (after epoch completes)
                    let statusText = `Epoch ${progress.epoch}/${progress.total_epochs}`;
                    if (progress.loss > 0 || progress.accuracy > 0) {
                        statusText += ` - Loss: ${progress.loss.toFixed(4)}, ` +
                                     `Acc: ${(progress.accuracy * 100).toFixed(1)}%, ` +
                                     `Val Loss: ${progress.val_loss.toFixed(4)}, ` +
                                     `Val Acc: ${(progress.val_accuracy * 100).toFixed(1)}%`;
                    } else {
                        statusText += ` - Training...`;
                    }
                    updateProgress(percentComplete, statusText);
                }
                
                // Check if training is complete
                const resultRes = await fetch('http://localhost:8000/api/training-result');
                const resultData = await resultRes.json();
                
                if (resultData.complete) {
                    clearInterval(progressIntervalRetrain);
                    
                    if (resultData.status === 'error') {
                        throw new Error(resultData.error);
                    }
                    
                    if (resultData.status === 'complete') {
                        result = resultData.result;
                        updateProgress(100, 'Retraining complete!');
                        
                        // Calculate training time
                        const trainingTime = ((Date.now() - startTime) / 1000).toFixed(1);

                        // Show detailed results
                        setTimeout(() => {
                            document.getElementById('trainingPanel').classList.add('hidden');
                            showDetailedResults(result, trainingTime);
                            showMessage('CNN model retrained successfully!', 'success');
                            document.getElementById('retrainBtn').disabled = false;
                        }, 1000);
                    }
                }
            } catch (e) {
                clearInterval(progressIntervalRetrain);
                console.error('Progress polling error:', e);
                document.getElementById('trainingPanel').classList.add('hidden');
                document.getElementById('retrainBtn').disabled = false;
                showMessage(`Retraining failed: ${e.message}`, 'error');
            }
        }, 1000); // Poll every second

    } catch (error) {
        console.error('Retraining error:', error);
        document.getElementById('trainingPanel').classList.add('hidden');
        document.getElementById('retrainBtn').disabled = false;
        showMessage(`Retraining failed: ${error.message}`, 'error');
    }
}

async function viewModelPerformance() {
    const panel = document.getElementById('currentPerformancePanel');
    const content = document.getElementById('performanceContent');
    
    panel.classList.remove('hidden');
    content.innerHTML = '<p class="text-gray-500 text-center py-8"><i class="fas fa-spinner fa-spin mr-2"></i>Loading performance data...</p>';
    
    try {
        const response = await fetch('http://localhost:8000/api/model-performance');
        const data = await response.json();
        
        if (!data.is_trained) {
            content.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-exclamation-circle text-yellow-500 text-5xl mb-4"></i>
                    <p class="text-gray-600 text-lg">No trained model found</p>
                    <p class="text-gray-500 text-sm mt-2">Train a model first to see performance metrics</p>
                </div>
            `;
            return;
        }
        
        if (!data.metrics) {
            content.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-info-circle text-blue-500 text-5xl mb-4"></i>
                    <p class="text-gray-600 text-lg">${data.message}</p>
                    <p class="text-gray-500 text-sm mt-2">Total: ${data.total_real} real, ${data.total_fake} fake images</p>
                </div>
            `;
            return;
        }
        
        const metrics = data.metrics;
        
        content.innerHTML = `
            <!-- Main Metrics -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-green-50 p-4 rounded-lg text-center">
                    <p class="text-green-600 text-sm font-semibold">Accuracy</p>
                    <p class="text-3xl font-bold text-green-700">${(metrics.accuracy * 100).toFixed(1)}%</p>
                </div>
                <div class="bg-blue-50 p-4 rounded-lg text-center">
                    <p class="text-blue-600 text-sm font-semibold">Precision</p>
                    <p class="text-3xl font-bold text-blue-700">${(metrics.precision * 100).toFixed(1)}%</p>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg text-center">
                    <p class="text-purple-600 text-sm font-semibold">Recall</p>
                    <p class="text-3xl font-bold text-purple-700">${(metrics.recall * 100).toFixed(1)}%</p>
                </div>
                <div class="bg-orange-50 p-4 rounded-lg text-center">
                    <p class="text-orange-600 text-sm font-semibold">F1 Score</p>
                    <p class="text-3xl font-bold text-orange-700">${(metrics.f1_score * 100).toFixed(1)}%</p>
                </div>
            </div>

            <!-- Dataset Info -->
            <div class="grid grid-cols-3 gap-4 mb-6">
                <div class="bg-gray-50 p-3 rounded-lg text-center">
                    <p class="text-gray-600 text-xs font-semibold">Real Images</p>
                    <p class="text-xl font-bold text-gray-700">${metrics.total_real}</p>
                </div>
                <div class="bg-gray-50 p-3 rounded-lg text-center">
                    <p class="text-gray-600 text-xs font-semibold">Fake Images</p>
                    <p class="text-xl font-bold text-gray-700">${metrics.total_fake}</p>
                </div>
                <div class="bg-gray-50 p-3 rounded-lg text-center">
                    <p class="text-gray-600 text-xs font-semibold">Total Samples</p>
                    <p class="text-xl font-bold text-gray-700">${metrics.total_samples}</p>
                </div>
            </div>

            <!-- Confusion Matrix -->
            <div class="bg-gray-50 p-4 rounded-lg mb-6">
                <h3 class="font-semibold text-gray-800 mb-3 text-center">Confusion Matrix</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-center">
                        <thead>
                            <tr>
                                <th class="p-2"></th>
                                <th class="p-2 text-sm font-semibold text-gray-600" colspan="2">Predicted</th>
                            </tr>
                            <tr>
                                <th class="p-2"></th>
                                <th class="p-2 text-sm font-semibold text-red-600">Fake</th>
                                <th class="p-2 text-sm font-semibold text-green-600">Real</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="p-2 text-sm font-semibold text-red-600">Actual Fake</td>
                                <td class="p-2 bg-green-100 font-bold text-lg">${metrics.confusion_matrix[0][0]}</td>
                                <td class="p-2 bg-red-100 font-bold text-lg">${metrics.confusion_matrix[0][1]}</td>
                            </tr>
                            <tr>
                                <td class="p-2 text-sm font-semibold text-green-600">Actual Real</td>
                                <td class="p-2 bg-red-100 font-bold text-lg">${metrics.confusion_matrix[1][0]}</td>
                                <td class="p-2 bg-green-100 font-bold text-lg">${metrics.confusion_matrix[1][1]}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <p class="text-xs text-gray-600 mt-2 text-center">Green = Correct, Red = Incorrect</p>
            </div>

            <!-- Metrics Explanation -->
            <div class="bg-blue-50 p-4 rounded-lg">
                <h3 class="font-semibold text-blue-900 mb-2">
                    <i class="fas fa-info-circle mr-2"></i>Performance Metrics
                </h3>
                <ul class="text-blue-800 text-sm space-y-1">
                    <li><strong>Accuracy:</strong> ${(metrics.accuracy * 100).toFixed(1)}% - Overall correctness</li>
                    <li><strong>Precision:</strong> ${(metrics.precision * 100).toFixed(1)}% - When predicting "real", how often correct</li>
                    <li><strong>Recall:</strong> ${(metrics.recall * 100).toFixed(1)}% - Of all real faces, how many detected</li>
                    <li><strong>F1 Score:</strong> ${(metrics.f1_score * 100).toFixed(1)}% - Balance of precision and recall</li>
                </ul>
            </div>

            <!-- Recommendation -->
            ${metrics.accuracy < 0.85 ? `
                <div class="bg-yellow-50 p-4 rounded-lg mt-4">
                    <h3 class="font-semibold text-yellow-900 mb-2">
                        <i class="fas fa-exclamation-triangle mr-2"></i>Recommendation
                    </h3>
                    <p class="text-yellow-800 text-sm">
                        Accuracy is below 85%. Consider adding more diverse training images (different angles, lighting, fake types) to improve performance.
                    </p>
                </div>
            ` : `
                <div class="bg-green-50 p-4 rounded-lg mt-4">
                    <h3 class="font-semibold text-green-900 mb-2">
                        <i class="fas fa-check-circle mr-2"></i>Good Performance
                    </h3>
                    <p class="text-green-800 text-sm">
                        Model is performing well! You can still improve by adding more diverse training data.
                    </p>
                </div>
            `}
        `;
        
    } catch (error) {
        console.error('Failed to load performance:', error);
        content.innerHTML = `
            <div class="text-center py-8">
                <i class="fas fa-exclamation-triangle text-red-500 text-5xl mb-4"></i>
                <p class="text-gray-600 text-lg">Failed to load performance data</p>
                <p class="text-gray-500 text-sm mt-2">${error.message}</p>
            </div>
        `;
    }
}

function updateInstructions(phase) {
    const instructions = {
        real: 'Capture 20-30 images of your REAL face. Try different angles and expressions. Press R or click "Capture Real Image".',
        fake: 'Now capture 20-30 FAKE images. Use printed photos, phone screens, etc. Press F or click "Capture Fake Image".',
        train: 'Great! You have enough data. Click "Train Model" or press T to start training.'
    };
    
    document.getElementById('instructionText').textContent = instructions[phase];
}

function updateStepIndicator(step) {
    // Reset all steps
    for (let i = 1; i <= 3; i++) {
        const stepEl = document.getElementById(`step${i}`);
        const circle = stepEl.querySelector('div');
        
        if (i < step) {
            circle.className = 'w-12 h-12 bg-green-600 text-white rounded-full flex items-center justify-center mx-auto mb-2';
            circle.innerHTML = '<i class="fas fa-check"></i>';
        } else if (i === step) {
            circle.className = 'w-12 h-12 bg-purple-600 text-white rounded-full flex items-center justify-center mx-auto mb-2';
        } else {
            circle.className = 'w-12 h-12 bg-gray-300 text-white rounded-full flex items-center justify-center mx-auto mb-2';
        }
    }
}

function showDetailedResults(result, trainingTime) {
    const metrics = result.metrics;
    const resultsPanel = document.getElementById('resultsPanel');
    const config = result.config || {};
    
    let configInfo = '';
    if (config.epochs) {
        configInfo = `
            <div class="bg-purple-50 p-4 rounded-lg mb-6">
                <h3 class="font-semibold text-purple-900 mb-2">
                    <i class="fas fa-cog mr-2"></i>CNN Configuration
                </h3>
                <div class="grid grid-cols-3 gap-3 text-sm">
                    <div>
                        <p class="text-purple-600 font-semibold">Epochs</p>
                        <p class="text-purple-900 font-bold">${config.epochs}</p>
                    </div>
                    <div>
                        <p class="text-purple-600 font-semibold">Batch Size</p>
                        <p class="text-purple-900 font-bold">${config.batch_size}</p>
                    </div>
                    <div>
                        <p class="text-purple-600 font-semibold">Threshold</p>
                        <p class="text-purple-900 font-bold">${config.confidence_threshold}%</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    resultsPanel.innerHTML = `
        <h2 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
            <i class="fas fa-check-circle text-green-600 mr-3"></i>Training Complete!
        </h2>
        
        <!-- Model Type Badge -->
        <div class="mb-6">
            <span class="inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold bg-purple-100 text-purple-800">
                <i class="fas fa-brain mr-2"></i>CNN Deep Learning Model
            </span>
            <span class="ml-2 text-gray-600 text-sm">
                <i class="fas fa-clock mr-1"></i>Trained in ${trainingTime}s
            </span>
        </div>
        
        ${configInfo}
        
        <!-- Main Metrics -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-green-50 p-4 rounded-lg text-center">
                <p class="text-green-600 text-sm font-semibold">Accuracy</p>
                <p class="text-3xl font-bold text-green-700">${(metrics.accuracy * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-blue-50 p-4 rounded-lg text-center">
                <p class="text-blue-600 text-sm font-semibold">Precision</p>
                <p class="text-3xl font-bold text-blue-700">${(metrics.precision * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-purple-50 p-4 rounded-lg text-center">
                <p class="text-purple-600 text-sm font-semibold">Recall</p>
                <p class="text-3xl font-bold text-purple-700">${(metrics.recall * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-orange-50 p-4 rounded-lg text-center">
                <p class="text-orange-600 text-sm font-semibold">F1 Score</p>
                <p class="text-3xl font-bold text-orange-700">${(metrics.f1_score * 100).toFixed(1)}%</p>
            </div>
        </div>

        <!-- Training Info -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-gray-50 p-3 rounded-lg text-center">
                <p class="text-gray-600 text-xs font-semibold">Total Real</p>
                <p class="text-xl font-bold text-gray-700">${metrics.total_real}</p>
            </div>
            <div class="bg-gray-50 p-3 rounded-lg text-center">
                <p class="text-gray-600 text-xs font-semibold">Total Fake</p>
                <p class="text-xl font-bold text-gray-700">${metrics.total_fake}</p>
            </div>
            <div class="bg-gray-50 p-3 rounded-lg text-center">
                <p class="text-gray-600 text-xs font-semibold">Training Set</p>
                <p class="text-xl font-bold text-gray-700">${metrics.train_size}</p>
            </div>
            <div class="bg-gray-50 p-3 rounded-lg text-center">
                <p class="text-gray-600 text-xs font-semibold">Test Set</p>
                <p class="text-xl font-bold text-gray-700">${metrics.test_size}</p>
            </div>
        </div>

        <!-- Confusion Matrix -->
        <div class="bg-gray-50 p-4 rounded-lg mb-6">
            <h3 class="font-semibold text-gray-800 mb-3">Confusion Matrix</h3>
            <div class="overflow-x-auto">
                <table class="w-full text-center">
                    <thead>
                        <tr>
                            <th class="p-2"></th>
                            <th class="p-2 text-sm font-semibold text-gray-600" colspan="2">Predicted</th>
                        </tr>
                        <tr>
                            <th class="p-2"></th>
                            <th class="p-2 text-sm font-semibold text-red-600">Fake</th>
                            <th class="p-2 text-sm font-semibold text-green-600">Real</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="p-2 text-sm font-semibold text-red-600">Actual Fake</td>
                            <td class="p-2 bg-green-100 font-bold">${metrics.confusion_matrix[0][0]}</td>
                            <td class="p-2 bg-red-100 font-bold">${metrics.confusion_matrix[0][1]}</td>
                        </tr>
                        <tr>
                            <td class="p-2 text-sm font-semibold text-green-600">Actual Real</td>
                            <td class="p-2 bg-red-100 font-bold">${metrics.confusion_matrix[1][0]}</td>
                            <td class="p-2 bg-green-100 font-bold">${metrics.confusion_matrix[1][1]}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <p class="text-xs text-gray-600 mt-2 text-center">Green = Correct predictions, Red = Incorrect predictions</p>
        </div>

        <!-- Performance Info -->
        <div class="bg-blue-50 p-4 rounded-lg mb-6">
            <h3 class="font-semibold text-blue-900 mb-2">
                <i class="fas fa-info-circle mr-2"></i>What do these metrics mean?
            </h3>
            <ul class="text-blue-800 text-sm space-y-1">
                <li><strong>Accuracy:</strong> Overall correctness (${(metrics.accuracy * 100).toFixed(1)}%)</li>
                <li><strong>Precision:</strong> When model says "real", how often is it correct (${(metrics.precision * 100).toFixed(1)}%)</li>
                <li><strong>Recall:</strong> Of all real faces, how many did we catch (${(metrics.recall * 100).toFixed(1)}%)</li>
                <li><strong>F1 Score:</strong> Balance between precision and recall (${(metrics.f1_score * 100).toFixed(1)}%)</li>
            </ul>
        </div>

        <!-- Training Time -->
        <div class="bg-yellow-50 p-4 rounded-lg mb-6 text-center">
            <p class="text-yellow-800 text-sm">Training completed in <strong>${trainingTime}s</strong></p>
            <p class="text-yellow-700 text-xs mt-1">New images: ${result.new_real_images} real + ${result.new_fake_images} fake</p>
        </div>

        <!-- Action Buttons -->
        <div class="flex gap-3">
            <a href="/" class="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-semibold transition text-center">
                <i class="fas fa-sign-in-alt mr-2"></i>Test Login
            </a>
            <button id="trainMoreBtn" class="flex-1 bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-semibold transition">
                <i class="fas fa-plus mr-2"></i>Add More Data
            </button>
        </div>
    `;
    
    resultsPanel.classList.remove('hidden');
    
    // Add event listener for train more button
    document.getElementById('trainMoreBtn').addEventListener('click', () => {
        resultsPanel.classList.add('hidden');
        showMessage('Add more images to improve accuracy', 'info');
    });
}

function showMessage(text, type) {
    // Create toast notification
    const toast = document.createElement('div');
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        info: 'bg-blue-600'
    };
    
    toast.className = `fixed top-4 right-4 ${colors[type]} text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-in`;
    toast.textContent = text;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Confirmation Modal
function showConfirmModal(title, message, icon = 'fa-question-circle', iconColor = 'bg-blue-100 text-blue-600') {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalMessage = document.getElementById('modalMessage');
        const modalIcon = document.getElementById('modalIcon');
        const modalIconContainer = document.getElementById('modalIconContainer');
        const confirmBtn = document.getElementById('modalConfirmBtn');
        const cancelBtn = document.getElementById('modalCancelBtn');

        // Set content
        modalTitle.textContent = title;
        modalMessage.textContent = message;
        modalIcon.className = `fas ${icon} text-3xl`;
        modalIconContainer.className = `flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full ${iconColor}`;

        // Show modal
        modal.classList.remove('hidden');

        // Handle confirm
        const handleConfirm = () => {
            modal.classList.add('hidden');
            confirmBtn.removeEventListener('click', handleConfirm);
            cancelBtn.removeEventListener('click', handleCancel);
            resolve(true);
        };

        // Handle cancel
        const handleCancel = () => {
            modal.classList.add('hidden');
            confirmBtn.removeEventListener('click', handleConfirm);
            cancelBtn.removeEventListener('click', handleCancel);
            resolve(false);
        };

        confirmBtn.addEventListener('click', handleConfirm);
        cancelBtn.addEventListener('click', handleCancel);

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                handleCancel();
            }
        });
    });
}

// Add CSS for animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slide-in {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    .animate-slide-in {
        animation: slide-in 0.3s ease-out;
    }
`;
document.head.appendChild(style);
