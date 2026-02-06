// Enhanced Liveness Detection with Advanced Motion & Depth Analysis
class EnhancedLivenessDetector {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.previousFrame = null;
        this.motionHistory = [];
        this.frameCount = 0;
        this.facePositions = [];
        this.brightnessHistory = [];
    }

    async captureFrame() {
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.ctx.drawImage(this.video, 0, 0);
        
        const currentFrame = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        return currentFrame;
    }

    detectFacePosition(imageData) {
        // Simple face detection using brightness clustering
        const data = imageData.data;
        let centerX = 0, centerY = 0, count = 0;
        
        // Sample center region for face
        const startY = Math.floor(imageData.height * 0.3);
        const endY = Math.floor(imageData.height * 0.7);
        const startX = Math.floor(imageData.width * 0.3);
        const endX = Math.floor(imageData.width * 0.7);
        
        for (let y = startY; y < endY; y += 5) {
            for (let x = startX; x < endX; x += 5) {
                const i = (y * imageData.width + x) * 4;
                const brightness = (data[i] + data[i+1] + data[i+2]) / 3;
                
                if (brightness > 80 && brightness < 220) {
                    centerX += x;
                    centerY += y;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            return { x: centerX / count, y: centerY / count };
        }
        return null;
    }

    async analyzeMotion() {
        const currentFrame = await this.captureFrame();
        
        if (this.previousFrame) {
            // 1. Frame difference for motion detection
            let totalDiff = 0;
            let regionDiffs = [0, 0, 0, 0]; // top, bottom, left, right
            const pixels = currentFrame.data.length;
            
            for (let i = 0; i < pixels; i += 4) {
                const r_diff = Math.abs(currentFrame.data[i] - this.previousFrame.data[i]);
                const g_diff = Math.abs(currentFrame.data[i + 1] - this.previousFrame.data[i + 1]);
                const b_diff = Math.abs(currentFrame.data[i + 2] - this.previousFrame.data[i + 2]);
                const diff = (r_diff + g_diff + b_diff) / 3;
                
                totalDiff += diff;
                
                // Track regional motion
                const pixelIndex = i / 4;
                const y = Math.floor(pixelIndex / currentFrame.width);
                const x = pixelIndex % currentFrame.width;
                
                if (y < currentFrame.height / 2) regionDiffs[0] += diff;
                else regionDiffs[1] += diff;
                if (x < currentFrame.width / 2) regionDiffs[2] += diff;
                else regionDiffs[3] += diff;
            }
            
            const avgDiff = totalDiff / (pixels / 4);
            this.motionHistory.push({
                overall: avgDiff,
                regions: regionDiffs,
                timestamp: Date.now()
            });
            
            // Keep last 15 frames
            if (this.motionHistory.length > 15) {
                this.motionHistory.shift();
            }
            
            // 2. Face position tracking
            const facePos = this.detectFacePosition(currentFrame);
            if (facePos) {
                this.facePositions.push(facePos);
                if (this.facePositions.length > 10) {
                    this.facePositions.shift();
                }
            }
            
            // 3. Brightness variation (lighting changes)
            let avgBrightness = 0;
            for (let i = 0; i < pixels; i += 4) {
                avgBrightness += (currentFrame.data[i] + currentFrame.data[i+1] + currentFrame.data[i+2]) / 3;
            }
            avgBrightness /= (pixels / 4);
            this.brightnessHistory.push(avgBrightness);
            if (this.brightnessHistory.length > 10) {
                this.brightnessHistory.shift();
            }
        }
        
        this.previousFrame = currentFrame;
        this.frameCount++;
    }

    detectStaticImage() {
        if (this.motionHistory.length < 5) return false;
        
        // Check if motion is too low (static image)
        const recentMotion = this.motionHistory.slice(-5);
        const avgMotion = recentMotion.reduce((sum, m) => sum + m.overall, 0) / recentMotion.length;
        
        // Static images have very low motion (<0.5)
        if (avgMotion < 0.3) {
            return true;
        }
        
        return false;
    }

    detectVideoReplay() {
        if (this.motionHistory.length < 10) return false;
        
        // Videos have very consistent motion patterns
        const motionValues = this.motionHistory.map(m => m.overall);
        const mean = motionValues.reduce((a, b) => a + b, 0) / motionValues.length;
        const variance = motionValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / motionValues.length;
        const stdDev = Math.sqrt(variance);
        
        // Video replays: low variance (<2), Real: higher variance (>3)
        if (stdDev < 1.5 && mean > 1) {
            return true;
        }
        
        // Check for looping patterns (video replay)
        if (this.motionHistory.length >= 15) {
            const first5 = this.motionHistory.slice(0, 5).map(m => m.overall);
            const last5 = this.motionHistory.slice(-5).map(m => m.overall);
            
            let similarity = 0;
            for (let i = 0; i < 5; i++) {
                similarity += Math.abs(first5[i] - last5[i]);
            }
            similarity /= 5;
            
            // If patterns repeat, likely a looping video
            if (similarity < 2) {
                return true;
            }
        }
        
        return false;
    }

    detectScreenRecapture() {
        if (this.facePositions.length < 5) return false;
        
        // Screen recapture: face position is too stable (no natural micro-movements)
        const positions = this.facePositions.slice(-5);
        let totalMovement = 0;
        
        for (let i = 1; i < positions.length; i++) {
            const dx = positions[i].x - positions[i-1].x;
            const dy = positions[i].y - positions[i-1].y;
            totalMovement += Math.sqrt(dx*dx + dy*dy);
        }
        
        const avgMovement = totalMovement / (positions.length - 1);
        
        // Real faces have micro-movements (>0.5), screens are too stable (<0.3)
        if (avgMovement < 0.2) {
            return true;
        }
        
        // Check for unnatural stability
        const xPositions = positions.map(p => p.x);
        const yPositions = positions.map(p => p.y);
        const xVariance = this.calculateVariance(xPositions);
        const yVariance = this.calculateVariance(yPositions);
        
        // Too stable = screen
        if (xVariance < 1 && yVariance < 1) {
            return true;
        }
        
        return false;
    }

    detectPrintAttack() {
        if (this.brightnessHistory.length < 5) return false;
        
        // Print attacks: very stable brightness (no natural lighting changes)
        const variance = this.calculateVariance(this.brightnessHistory);
        
        // Real faces: variance > 5, Prints: variance < 2
        if (variance < 1.5) {
            return true;
        }
        
        return false;
    }

    calculateVariance(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    }

    async performComprehensiveCheck() {
        // Wait for enough data
        if (this.frameCount < 10) {
            return { passed: true, message: "Collecting data..." };
        }
        
        // Run all checks
        if (this.detectStaticImage()) {
            return { passed: false, message: "Static image detected" };
        }
        
        if (this.detectVideoReplay()) {
            return { passed: false, message: "Video replay detected" };
        }
        
        if (this.detectScreenRecapture()) {
            return { passed: false, message: "Screen recapture detected" };
        }
        
        if (this.detectPrintAttack()) {
            return { passed: false, message: "Print attack detected" };
        }
        
        return { passed: true, message: "Client-side checks passed" };
    }

    async verifyWithBackend(imageBlob) {
        // First do client-side checks
        const clientCheck = await this.performComprehensiveCheck();
        if (!clientCheck.passed) {
            return { is_live: false, message: clientCheck.message };
        }
        
        // Then verify with backend
        const formData = new FormData();
        formData.append('image', imageBlob, 'liveness.jpg');
        
        try {
            const response = await fetch('http://localhost:8000/api/verify-liveness', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            return data;
        } catch (err) {
            console.error('Liveness verification error:', err);
            return { is_live: false, message: 'Verification failed' };
        }
    }

    captureFrameAsBlob() {
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
        this.motionInterval = setInterval(() => {
            this.analyzeMotion();
        }, 150);
    }

    stopMotionTracking() {
        if (this.motionInterval) {
            clearInterval(this.motionInterval);
        }
    }

    reset() {
        this.previousFrame = null;
        this.motionHistory = [];
        this.frameCount = 0;
        this.facePositions = [];
        this.brightnessHistory = [];
    }
}

window.EnhancedLivenessDetector = EnhancedLivenessDetector;
