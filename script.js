/**
 * AI Speed Detection - Real Object Tracking & Physics
 * Uses OpenCV.js (frame diffs/motion) + TensorFlow.js (object classification)
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- UI Elements ---
    const video = document.getElementById('videoElement');
    const procCanvas = document.getElementById('processingCanvas');
    const procCtx = procCanvas.getContext('2d', { willReadFrequently: true });
    const trackCanvas = document.getElementById('trackingCanvas');
    const trackCtx = trackCanvas.getContext('2d');
    
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const captureBtn = document.getElementById('capture-btn');
    const loadingOverlay = document.getElementById('camera-loading');
    const loadingText = document.getElementById('loading-text');
    const navStatus = document.getElementById('nav-camera-status');
    const sysStatusText = document.getElementById('sys-status-text');
    const recIndicator = document.getElementById('rec-indicator');
    
    const speedDisplay = document.getElementById('current-speed');
    const avgSpeedDisplay = document.getElementById('avg-speed');
    const topSpeedDisplay = document.getElementById('top-speed');
    const speedBar = document.getElementById('speed-bar');
    const fpsCounter = document.getElementById('fps-counter');
    const confMeter = document.getElementById('confidence-meter');
    const cvStatus = document.getElementById('cv-status');
    const detectedType = document.getElementById('detected-type');
    
    const historyList = document.getElementById('speed-history');
    const clearLogBtn = document.getElementById('clear-log');
    const exportLogBtn = document.getElementById('export-log');
    const fovInput = document.getElementById('fov-width');
    
    const modeBtns = document.querySelectorAll('.mode-btn');
    const unitBtns = document.querySelectorAll('.unit-btn');
    const beepSound = document.getElementById('beep-sound');

    // --- State ---
    let stream = null;
    let isTracking = false;
    let animationId = null;
    let currentMode = 'generic'; // generic, cricket, vehicle
    let currentUnit = 'kmh'; // kmh, mph, ms
    
    // Physics & Calibration
    let pxToMeter = 1; 
    let topSpeed = 0;
    let speedSum = 0;
    let speedCount = 0;
    
    // AI Models State
    let cocoModel = null;
    let isCvReady = false;
    let isCocoReady = false;
    
    // OpenCV Matrices
    let matPrev = null;
    let matGray = null;
    let matDiff = null;
    let matThresh = null;
    
    // Tracking Data
    let lastFrameTime = 0;
    let trackedObjects = {}; // OpenCV tracked objects
    let nextObjId = 0;
    let currentAiClasses = []; // Latest classes from TFJS

    // --- Initialization ---
    initParticles();
    checkDependencies();
    
    function checkDependencies() {
        // Check TFJS COCO-SSD
        if (typeof cocoSsd !== 'undefined') {
            cocoSsd.load().then(model => {
                cocoModel = model;
                isCocoReady = true;
                checkReadyState();
            }).catch(e => console.error("TFJS Load Error:", e));
        }
        
        // Check OpenCV.js (It loads async, we must wait for cv.onRuntimeInitialized)
        let cvCheckInterval = setInterval(() => {
            if (typeof cv !== 'undefined' && cv.Mat) {
                isCvReady = true;
                clearInterval(cvCheckInterval);
                checkReadyState();
            }
        }, 500);
    }
    
    function checkReadyState() {
        if (isCocoReady && isCvReady) {
            loadingOverlay.classList.add('hidden');
            startBtn.disabled = false;
            startBtn.innerHTML = '<span class="btn-icon">▶</span> START OPTICS';
            sysStatusText.innerText = "SYSTEM_READY";
        }
    }

    // --- Event Listeners ---
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    captureBtn.addEventListener('click', captureScreenshot);
    clearLogBtn.addEventListener('click', clearHistory);
    exportLogBtn.addEventListener('click', exportHistory);

    modeBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            modeBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentMode = e.target.dataset.mode;
        });
    });

    unitBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            unitBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentUnit = e.target.dataset.unit;
            // Update UI displays immediately
            updateSpeedDisplay(speedDisplay.dataset.val || 0, speedDisplay);
            updateSpeedDisplay(topSpeed, topSpeedDisplay);
            updateSpeedDisplay(speedCount > 0 ? speedSum/speedCount : 0, avgSpeedDisplay);
        });
    });

    fovInput.addEventListener('change', () => {
        if(fovInput.value <= 0) fovInput.value = 1;
        updateCalibration();
    });

    // --- Camera & Setup ---
    async function startCamera() {
        try {
            loadingOverlay.classList.remove('hidden');
            loadingText.innerText = "ACCESSING OPTICS...";
            loadingOverlay.style.opacity = '1';
            
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
            });
            
            video.srcObject = stream;
            
            video.onloadedmetadata = () => {
                // Setup Canvases
                procCanvas.width = video.videoWidth;
                procCanvas.height = video.videoHeight;
                trackCanvas.width = video.videoWidth;
                trackCanvas.height = video.videoHeight;
                
                updateCalibration();
                
                // Initialize OpenCV Matrices
                matPrev = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);
                matGray = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);
                matDiff = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);
                matThresh = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);
                
                loadingOverlay.style.opacity = '0';
                setTimeout(() => loadingOverlay.classList.add('hidden'), 500);
                
                startBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
                captureBtn.disabled = false;
                
                navStatus.classList.remove('offline');
                navStatus.classList.add('online');
                recIndicator.classList.remove('hidden');
                sysStatusText.innerText = "TRACKING_ACTIVE";
                cvStatus.innerText = "ACTIVE";
                cvStatus.classList.add("neon-text-green");
                
                isTracking = true;
                trackedObjects = {};
                
                // Start tracking loops
                trackLoop();
                aiClassificationLoop();
            };
        } catch (err) {
            console.error("Camera Error:", err);
            alert("Camera access denied or unavailable.");
            loadingOverlay.classList.add('hidden');
        }
    }

    function stopCamera() {
        isTracking = false;
        if (stream) stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        
        cancelAnimationFrame(animationId);
        trackCtx.clearRect(0, 0, trackCanvas.width, trackCanvas.height);
        
        // Free OpenCV memory
        if(matPrev) { matPrev.delete(); matGray.delete(); matDiff.delete(); matThresh.delete(); }
        
        startBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        captureBtn.disabled = true;
        
        navStatus.classList.remove('online');
        navStatus.classList.add('offline');
        recIndicator.classList.add('hidden');
        sysStatusText.innerText = "SYSTEM_READY";
        cvStatus.innerText = "OFFLINE";
        cvStatus.classList.remove("neon-text-green");
        
        updateSpeedDisplay(0, speedDisplay);
    }

    function updateCalibration() {
        if(!video.videoWidth) return;
        let fovMeters = parseFloat(fovInput.value) || 10;
        pxToMeter = fovMeters / video.videoWidth;
    }

    // --- AI Classification Loop (TensorFlow.js) ---
    // Runs slower to not block the main thread, updates labels of tracked objects
    async function aiClassificationLoop() {
        if (!isTracking || !cocoModel || video.readyState < 2) {
            if(isTracking) setTimeout(aiClassificationLoop, 500);
            return;
        }
        
        try {
            let predictions = await cocoModel.detect(video);
            
            // Filter predictions based on mode
            currentAiClasses = predictions.filter(p => {
                if (p.score < 0.3) return false;
                if (currentMode === 'vehicle' && !['car', 'truck', 'bus', 'motorcycle', 'bicycle'].includes(p.class)) return false;
                if (currentMode === 'cricket' && !['sports ball', 'person'].includes(p.class)) return false;
                return true;
            });
            
            if(currentAiClasses.length > 0) {
                confMeter.innerText = Math.round(currentAiClasses[0].score * 100) + "%";
            } else {
                confMeter.innerText = "0%";
            }
            
        } catch(e) { console.error("TFJS error", e); }
        
        if (isTracking) setTimeout(aiClassificationLoop, 200); // 5 FPS for AI classification
    }

    // --- OpenCV Motion Tracking Loop (High Speed) ---
    function trackLoop(timestamp) {
        if (!isTracking) return;

        let now = Date.now();
        if (lastFrameTime) {
            let fps = Math.round(1000 / (now - lastFrameTime));
            if (timestamp % 5 === 0) fpsCounter.innerText = fps > 60 ? 60 : fps;
        }
        
        if (video.readyState >= 2) {
            processOpenCVFrame(now);
        }

        lastFrameTime = now;
        animationId = requestAnimationFrame(trackLoop);
    }

    function processOpenCVFrame(now) {
        // Draw video to processing canvas
        procCtx.drawImage(video, 0, 0, procCanvas.width, procCanvas.height);
        
        // Read image to OpenCV
        let src = cv.imread(procCanvas);
        cv.cvtColor(src, matGray, cv.COLOR_RGBA2GRAY);
        cv.GaussianBlur(matGray, matGray, new cv.Size(21, 21), 0);
        
        // If first frame, copy and return
        if (cv.countNonZero(matPrev) === 0) {
            matGray.copyTo(matPrev);
            src.delete();
            return;
        }
        
        // Compute absolute difference between current and previous frame
        cv.absdiff(matPrev, matGray, matDiff);
        cv.threshold(matDiff, matThresh, 30, 255, cv.THRESH_BINARY); // slightly higher threshold to reduce light noise
        cv.dilate(matThresh, matThresh, new cv.Mat(), new cv.Point(-1, -1), 4); // dilate heavily to merge fragments of the same object
        
        // Find contours
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(matThresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        
        let currentRects = [];
        let minArea = currentMode === 'cricket' ? 100 : (currentMode === 'vehicle' ? 1500 : 500);
        let totalArea = 0;
        
        // Collect valid rects and calculate total area for camera shake detection
        for (let i = 0; i < contours.size(); ++i) {
            let contour = contours.get(i);
            let area = cv.contourArea(contour);
            totalArea += area;
            if (area > minArea) {
                let rect = cv.boundingRect(contour);
                currentRects.push({rect: rect, area: area});
            }
            contour.delete();
        }
        
        contours.delete();
        hierarchy.delete();
        src.delete();
        
        // Update previous frame for next loop
        matGray.copyTo(matPrev);
        
        // Filter out massive camera shake (if > 25% of screen is moving, it's the camera moving, not an object)
        let screenArea = procCanvas.width * procCanvas.height;
        if (totalArea > screenArea * 0.25) {
            return; // Skip tracking update this frame
        }

        // Focus Mode: Sort by area and only track the largest 2 objects to avoid background noise distractions
        currentRects.sort((a, b) => b.area - a.area);
        let focusedRects = currentRects.slice(0, 2).map(item => item.rect);
        
        // Match contours to tracked objects
        updateTrackedObjects(focusedRects, now);
        
        // Draw HUD
        drawTrackingHUD();
    }

    function updateTrackedObjects(rects, now) {
        let newTrackedObjects = {};
        let activeSpeeds = [];
        
        rects.forEach(rect => {
            let centerX = rect.x + rect.width / 2;
            let centerY = rect.y + rect.height / 2;
            
            let bestMatchId = null;
            let bestDist = Infinity;
            
            // Match with existing objects
            for (let id in trackedObjects) {
                let obj = trackedObjects[id];
                let dist = Math.hypot(obj.centerX - centerX, obj.centerY - centerY);
                let maxDist = currentMode === 'cricket' ? 400 : 250; // Fast moving objects need large search radius
                
                if (dist < maxDist && dist < bestDist) {
                    bestDist = dist;
                    bestMatchId = id;
                }
            }
            
            let aiClass = matchAiClass(rect);
            
            if (bestMatchId) {
                // Update existing object
                let prevObj = trackedObjects[bestMatchId];
                let dt = Math.max(1, now - prevObj.lastSeen); // ms
                
                // Real physics calculation:
                // distance in meters = pixel distance * pxToMeter
                let distMeters = bestDist * pxToMeter;
                // speed in m/s = distance(m) / time(s)
                let speedMs = distMeters / (dt / 1000);
                let speedKmh = speedMs * 3.6;
                
                // Smooth speed (Exponential Moving Average)
                let smoothedSpeed = prevObj.speed === 0 ? speedKmh : (prevObj.speed * 0.7 + speedKmh * 0.3);
                
                // Ignore micro-jitter speed
                if (smoothedSpeed < 2) smoothedSpeed = 0;
                
                newTrackedObjects[bestMatchId] = {
                    id: bestMatchId,
                    rect: rect,
                    centerX, centerY,
                    vx: centerX - prevObj.centerX,
                    vy: centerY - prevObj.centerY,
                    speed: smoothedSpeed,
                    label: aiClass || prevObj.label || 'UNKNOWN',
                    life: prevObj.life + 1,
                    lastSeen: now
                };
                
                if(smoothedSpeed > 0) activeSpeeds.push(smoothedSpeed);
                delete trackedObjects[bestMatchId];
                
                // Beep if new fast object
                if (smoothedSpeed > 10 && prevObj.life === 2) {
                    beepSound.currentTime = 0;
                    beepSound.play().catch(e=>{});
                }
                
            } else {
                // New object
                let id = nextObjId++;
                newTrackedObjects[id] = {
                    id, rect, centerX, centerY, vx: 0, vy: 0, speed: 0,
                    label: aiClass || 'UNKNOWN', life: 0, lastSeen: now
                };
            }
        });
        
        // Handle disappeared objects (log their final speed)
        for (let id in trackedObjects) {
            let obj = trackedObjects[id];
            if (obj.life > 3 && obj.speed > 5) {
                logHistory(obj.speed, obj.label);
                
                // Update global stats
                if (obj.speed > topSpeed) topSpeed = obj.speed;
                speedSum += obj.speed;
                speedCount++;
                
                updateSpeedDisplay(topSpeed, topSpeedDisplay);
                updateSpeedDisplay(speedSum / speedCount, avgSpeedDisplay);
            }
        }
        
        trackedObjects = newTrackedObjects;
        
        // Update Dashboard Display
        if (activeSpeeds.length > 0) {
            let maxSpeed = Math.max(...activeSpeeds);
            updateSpeedDisplay(maxSpeed, speedDisplay);
            
            // Find dominant label
            let activeLabels = Object.values(trackedObjects).filter(o=>o.speed > 0).map(o=>o.label);
            if(activeLabels.length > 0 && activeLabels[0] !== 'UNKNOWN') {
                detectedType.innerText = activeLabels[0].toUpperCase() + " DETECTED";
            } else {
                detectedType.innerText = "MOTION DETECTED";
            }
        } else {
            detectedType.innerText = "SCANNING...";
            setTimeout(() => { 
                if (Object.keys(trackedObjects).length === 0) updateSpeedDisplay(0, speedDisplay); 
            }, 300);
        }
    }

    function matchAiClass(rect) {
        // Find if any TFJS bounding box overlaps significantly with OpenCV rect
        for(let ai of currentAiClasses) {
            let [ax, ay, aw, ah] = ai.bbox;
            // Simple center inside check
            let centerX = rect.x + rect.width/2;
            let centerY = rect.y + rect.height/2;
            if (centerX >= ax && centerX <= ax+aw && centerY >= ay && centerY <= ay+ah) {
                return ai.class;
            }
        }
        return null;
    }

    function drawTrackingHUD() {
        trackCtx.clearRect(0, 0, trackCanvas.width, trackCanvas.height);
        
        for (let id in trackedObjects) {
            let obj = trackedObjects[id];
            if (obj.life < 1) continue; // Ignore 1-frame noise
            
            let rect = obj.rect;
            
            // Box properties
            trackCtx.strokeStyle = obj.speed > 50 ? "rgba(255, 0, 60, 0.8)" : "rgba(0, 243, 255, 0.8)";
            trackCtx.lineWidth = 2;
            
            // Draw Futuristic Corners
            let cLen = Math.min(rect.width, rect.height) * 0.2; 
            trackCtx.beginPath();
            trackCtx.moveTo(rect.x, rect.y + cLen); trackCtx.lineTo(rect.x, rect.y); trackCtx.lineTo(rect.x + cLen, rect.y);
            trackCtx.moveTo(rect.x + rect.width - cLen, rect.y); trackCtx.lineTo(rect.x + rect.width, rect.y); trackCtx.lineTo(rect.x + rect.width, rect.y + cLen);
            trackCtx.moveTo(rect.x + rect.width, rect.y + rect.height - cLen); trackCtx.lineTo(rect.x + rect.width, rect.y + rect.height); trackCtx.lineTo(rect.x + rect.width - cLen, rect.y + rect.height);
            trackCtx.moveTo(rect.x + cLen, rect.y + rect.height); trackCtx.lineTo(rect.x, rect.y + rect.height); trackCtx.lineTo(rect.x, rect.y + rect.height - cLen);
            trackCtx.stroke();
            
            // Trajectory Vector
            if (obj.speed > 0) {
                trackCtx.beginPath();
                trackCtx.strokeStyle = "rgba(188, 19, 254, 0.6)";
                trackCtx.lineWidth = 2;
                trackCtx.moveTo(obj.centerX, obj.centerY);
                trackCtx.lineTo(obj.centerX - (obj.vx * 3), obj.centerY - (obj.vy * 3));
                trackCtx.stroke();
            }
            
            // Label
            trackCtx.fillStyle = "rgba(0, 243, 255, 0.9)";
            trackCtx.font = "14px Orbitron";
            
            let displaySpd = formatSpeed(obj.speed);
            trackCtx.fillText(`${obj.label.toUpperCase()} [${displaySpd}]`, rect.x, rect.y - 8);
        }
    }

    // --- Format & Utilities ---
    function formatSpeed(speedKmh) {
        if (currentUnit === 'mph') return Math.round(speedKmh * 0.621371) + ' MPH';
        if (currentUnit === 'ms') return (speedKmh * 0.277778).toFixed(1) + ' M/S';
        return Math.round(speedKmh) + ' KM/H';
    }

    function updateSpeedDisplay(speedKmh, el) {
        if (!el) return;
        el.dataset.val = speedKmh; // store raw kmh
        
        if(speedKmh === 0) {
            el.innerText = "000";
            if(el === speedDisplay) speedBar.style.width = "0%";
            return;
        }

        let displayVal;
        if (currentUnit === 'mph') displayVal = Math.round(speedKmh * 0.621371);
        else if (currentUnit === 'ms') displayVal = (speedKmh * 0.277778).toFixed(1);
        else displayVal = Math.round(speedKmh);

        el.innerText = displayVal.toString().padStart(3, '0');
        
        // Update bar for main display
        if (el === speedDisplay) {
            let percent = Math.min(100, (speedKmh / 150) * 100);
            speedBar.style.width = percent + "%";
            
            if(percent > 80) el.style.color = 'var(--danger)';
            else if (percent > 40) el.style.color = 'var(--neon-purple)';
            else el.style.color = 'var(--neon-cyan)';
        }
    }

    function logHistory(speedKmh, label) {
        const emptyMsg = document.querySelector('.history-empty');
        if(emptyMsg) emptyMsg.remove();

        const li = document.createElement('li');
        li.className = 'history-item';
        
        let displayStr = formatSpeed(speedKmh);
        const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false });

        li.innerHTML = `<span class="time">[${timeStr}] ${label.toUpperCase()}</span> <span class="val">${displayStr}</span>`;
        historyList.prepend(li);
        
        if(historyList.children.length > 20) historyList.lastChild.remove();
    }

    function clearHistory() {
        historyList.innerHTML = '<li class="history-empty">No data available</li>';
        topSpeed = 0; speedSum = 0; speedCount = 0;
        updateSpeedDisplay(0, topSpeedDisplay);
        updateSpeedDisplay(0, avgSpeedDisplay);
    }

    function exportHistory() {
        let items = document.querySelectorAll('.history-item');
        if(items.length === 0) return alert("No history to export.");
        
        let csv = "Time,Object,Speed\n";
        items.forEach(item => {
            let timeObj = item.querySelector('.time').innerText.match(/\[(.*?)\] (.*)/);
            let speed = item.querySelector('.val').innerText;
            if(timeObj) csv += `${timeObj[1]},${timeObj[2]},${speed}\n`;
        });
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `SpeedAI_Log_${Date.now()}.csv`;
        a.click();
    }

    function captureScreenshot() {
        if(!isTracking) return;
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = trackCanvas.width;
        tempCanvas.height = trackCanvas.height;
        const tCtx = tempCanvas.getContext('2d');
        
        tCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        tCtx.drawImage(trackCanvas, 0, 0);
        
        tCtx.fillStyle = "rgba(0,0,0,0.6)";
        tCtx.fillRect(10, tempCanvas.height - 45, 350, 35);
        tCtx.fillStyle = "#00f3ff";
        tCtx.font = "16px Orbitron";
        tCtx.fillText(`SPEEDAI SYSTEM LOG | ${new Date().toLocaleTimeString()}`, 20, tempCanvas.height - 22);

        const link = document.createElement('a');
        link.download = `SpeedAI_Capture_${Date.now()}.png`;
        link.href = tempCanvas.toDataURL('image/png');
        link.click();
    }

    // --- Background Particles ---
    function initParticles() {
        const bgCanvas = document.getElementById('particles-bg');
        const bgCtx = bgCanvas.getContext('2d');
        let width, height;
        let particles = [];

        function resize() { width = bgCanvas.width = window.innerWidth; height = bgCanvas.height = window.innerHeight; }
        window.addEventListener('resize', resize); resize();

        for(let i=0; i<50; i++) particles.push({ x: Math.random() * width, y: Math.random() * height, vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5, size: Math.random() * 2 });

        function animateBg() {
            bgCtx.clearRect(0, 0, width, height);
            bgCtx.fillStyle = "rgba(0, 243, 255, 0.5)";
            
            particles.forEach(p => {
                p.x += p.vx; p.y += p.vy;
                if(p.x < 0) p.x = width; if(p.x > width) p.x = 0;
                if(p.y < 0) p.y = height; if(p.y > height) p.y = 0;
                bgCtx.beginPath(); bgCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2); bgCtx.fill();
            });
            
            bgCtx.strokeStyle = "rgba(0, 243, 255, 0.05)";
            for(let i=0; i<particles.length; i++) {
                for(let j=i+1; j<particles.length; j++) {
                    let dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
                    if(Math.sqrt(dx*dx + dy*dy) < 150) { bgCtx.beginPath(); bgCtx.moveTo(particles[i].x, particles[i].y); bgCtx.lineTo(particles[j].x, particles[j].y); bgCtx.stroke(); }
                }
            }
            requestAnimationFrame(animateBg);
        }
        animateBg();
    }
});
