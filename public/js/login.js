const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('captureBtn');
const overlay = document.getElementById('overlay');
const message = document.getElementById('message');

let stream = null;
let livenessDetector = null;

startCameraBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        video.srcObject = stream;
        overlay.classList.add('hidden');
        captureBtn.disabled = false;
        
        livenessDetector = new LivenessDetector(video, canvas);
        
        // Start motion tracking after video loads
        video.addEventListener('loadeddata', () => {
            setTimeout(() => {
                livenessDetector.startMotionTracking();
                showMessage('👁️ Please look at the camera naturally', 'info');
            }, 500);
        });
    } catch (err) {
        showMessage('Camera access denied', 'error');
    }
});

captureBtn.addEventListener('click', async () => {
    captureBtn.disabled = true;
    captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    try {
        // Wait a moment to collect motion data (reduced requirement)
        if (livenessDetector.frameCount < 3) {
            showMessage('⏳ Please wait, initializing...', 'info');
            await new Promise(resolve => setTimeout(resolve, 600));
        }
        
        showMessage('🔍 Performing anti-spoofing checks...', 'info');
        
        // Capture frame
        const blob = await livenessDetector.captureFrame();
        
        // Verify liveness with backend (multi-layer detection)
        const livenessResult = await livenessDetector.verifyWithBackend(blob);
        
        if (!livenessResult.is_live) {
            showMessage(`❌ ${livenessResult.message}`, 'error');
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture & Login';
            return;
        }
        
        showMessage('✅ Liveness verified! Authenticating...', 'success');
        
        // Proceed with login
        const formData = new FormData();
        formData.append('image', blob, 'face.jpg');
        
        const response = await fetch('http://localhost:8000/api/login', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            livenessDetector.stopMotionTracking();
            showMessage('🎉 Login successful! Redirecting...', 'success');
            localStorage.setItem('user', JSON.stringify(data.user));
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1000);
        } else {
            showMessage(data.detail || 'Login failed', 'error');
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture & Login';
        }
    } catch (err) {
        showMessage('Server error. Please try again.', 'error');
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture & Login';
    }
});

function showMessage(text, type) {
    message.textContent = text;
    const colors = {
        'success': 'bg-green-100 text-green-700',
        'error': 'bg-red-100 text-red-700',
        'info': 'bg-blue-100 text-blue-700'
    };
    message.className = `p-3 rounded-lg text-center font-medium text-sm ${colors[type] || colors.info}`;
    message.classList.remove('hidden');
}
