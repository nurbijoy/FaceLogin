// Enhanced Liveness Detection with Motion Analysis
class LivenessDetector {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.previousFrame = null;
        this.motionHistory = [];
        this.frameCount = 0;
    }

    async detectMotion() {
        // Capture current frame
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.ctx.drawImage(this.video, 0, 0);
        
        const currentFrame = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        
        if (this.previousFrame) {
            // Calculate frame difference
            let diff = 0;
            const pixels = currentFrame.data.length;
            
            for (let i = 0; i < pixels; i += 4) {
                const r_diff = Math.abs(currentFrame.data[i] - this.previousFrame.data[i]);
                const g_diff = Math.abs(currentFrame.data[i + 1] - this.previousFrame.data[i + 1]);
                const b_diff = Math.abs(currentFrame.data[i + 2] - this.previousFrame.data[i + 2]);
                diff += (r_diff + g_diff + b_diff) / 3;
            }
            
            const avgDiff = diff / (pixels / 4);
            this.motionHistory.push(avgDiff);
            
            // Keep only last 10 frames
            if (this.motionHistory.length > 10) {
                this.motionHistory.shift();
            }
        }
        
        this.previousFrame = currentFrame;
        this.frameCount++;
    }

    isMotionNatural() {
        if (this.motionHistory.length < 3) return true;  // More lenient
        
        // Calculate motion variance
        const avg = this.motionHistory.reduce((a, b) => a + b, 0) / this.motionHistory.length;
        const variance = this.motionHistory.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / this.motionHistory.length;
        
        // More lenient thresholds - accept wider range of motion
        // Static images have very low variance (<1), videos have some variance
        return variance > 0.5 && variance < 1000 && avg > 0.5;
    }

    async verifyWithBackend(imageBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob, 'liveness.jpg');
        
        try {
            const response = await fetch('http://localhost:8000/api/verify-liveness', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Add client-side motion check (more lenient)
            const hasNaturalMotion = this.isMotionNatural();
            
            // Only reject if we have enough frames AND motion is clearly unnatural
            if (data.is_live && !hasNaturalMotion && this.frameCount > 15) {
                return { 
                    is_live: false, 
                    message: 'Static image or video replay detected' 
                };
            }
            
            return data;
        } catch (err) {
            console.error('Liveness verification error:', err);
            return { is_live: false, message: 'Verification failed' };
        }
    }

    captureFrame() {
        return new Promise((resolve) => {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
            
            this.canvas.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.95);
        });
    }

    startMotionTracking() {
        // Track motion every 200ms
        this.motionInterval = setInterval(() => {
            this.detectMotion();
        }, 200);
    }

    stopMotionTracking() {
        if (this.motionInterval) {
            clearInterval(this.motionInterval);
        }
    }
}

window.LivenessDetector = LivenessDetector;
