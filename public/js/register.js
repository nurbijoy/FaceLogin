const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('captureBtn');
const registerBtn = document.getElementById('registerBtn');
const overlay = document.getElementById('overlay');
const message = document.getElementById('message');
const form = document.getElementById('registerForm');

let stream = null;
let capturedBlob = null;

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
            captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture Photo';
            return;
        }
        
        capturedBlob = blob;
        showMessage('✅ Liveness verified! Photo captured. Now click Register.', 'success');
        registerBtn.disabled = false;
        captureBtn.innerHTML = '<i class="fas fa-check"></i> Photo Captured';
        captureBtn.classList.add('bg-green-600', 'hover:bg-green-700');
        captureBtn.classList.remove('bg-purple-600', 'hover:bg-purple-700');
    } catch (err) {
        showMessage('Verification failed. Please try again.', 'error');
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture Photo';
    }
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!capturedBlob) {
        showMessage('Please capture your photo first', 'error');
        return;
    }
    
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    
    const formData = new FormData();
    formData.append('username', username);
    formData.append('email', email);
    formData.append('image', capturedBlob, 'face.jpg');
    
    registerBtn.disabled = true;
    registerBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Registering...';
    
    try {
        const response = await fetch('http://localhost:8000/api/register', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('Registration successful! Redirecting to login...', 'success');
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
        } else {
            showMessage(data.detail || 'Registration failed', 'error');
            registerBtn.disabled = false;
            registerBtn.innerHTML = '<i class="fas fa-check-circle"></i> Register';
        }
    } catch (err) {
        showMessage('Server error. Please try again.', 'error');
        registerBtn.disabled = false;
        registerBtn.innerHTML = '<i class="fas fa-check-circle"></i> Register';
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
