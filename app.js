const micBtn = document.getElementById('micBtn');
const uploadBtn = document.getElementById('uploadBtn');
const audioInput = document.getElementById('audioInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileRow = document.getElementById('fileRow');
const fileName = document.getElementById('fileName');
const fileDuration = document.getElementById('fileDuration');
const audioPreview = document.getElementById('audioPreview');
const clearFile = document.getElementById('clearFile');
const errorMessage = document.getElementById('errorMessage');
const assistantPanel = document.querySelector('.assistant-panel');
const liveState = document.getElementById('liveState');
const assistantKicker = document.getElementById('assistantKicker');
const assistantTitle = document.getElementById('assistantTitle');
const assistantText = document.getElementById('assistantText');
const micHint = document.getElementById('micHint');
const waveform = document.getElementById('waveform');
const resultCard = document.getElementById('resultCard');
const recordingIndicator = document.getElementById('recordingIndicator');
const recordingTimer = document.getElementById('recordingTimer');
const recordingGuide = document.getElementById('recordingGuide');

let recorder = null;
let stream = null;
let audioChunks = [];
let selectedAudio = null;
let selectedAudioUrl = null;
let recordStartTime = 0;
let timerInterval = null;
let audioContext = null;
let analyserNode = null;
let visualizerAnimationId = null;

const MIN_RECORD_SECONDS = 3.0;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = '';
}

function formatDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function setAudio(file, name, durationSec = null) {
  selectedAudio = file;
  fileName.textContent = name || file.name || 'Voice recording';

  if (fileDuration) {
    if (durationSec && durationSec > 0) {
      fileDuration.textContent = `Duration: ${formatDuration(durationSec)}`;
      fileDuration.hidden = false;
    } else {
      fileDuration.textContent = '';
      fileDuration.hidden = true;
    }
  }

  // Audio preview playback
  if (audioPreview) {
    if (selectedAudioUrl) {
      URL.revokeObjectURL(selectedAudioUrl);
    }
    selectedAudioUrl = URL.createObjectURL(file);
    audioPreview.src = selectedAudioUrl;
    audioPreview.hidden = false;
  }

  fileRow.hidden = false;
  analyzeBtn.disabled = false;
  resultCard.hidden = true;
  clearError();

  assistantKicker.textContent = 'VOICE CAPTURED';
  assistantTitle.textContent = 'Ready for emotion analysis.';
  assistantText.textContent = 'Your recording is prepared. Tap "Analyze emotion" below to evaluate your speech.';
  liveState.textContent = 'AUDIO READY';
  micHint.textContent = 'READY TO ANALYZE';
}

function resetAudio() {
  stopLiveVisualizer();
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  if (recorder && recorder.state === 'recording') {
    try { recorder.stop(); } catch (e) {}
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  selectedAudio = null;
  if (selectedAudioUrl) {
    URL.revokeObjectURL(selectedAudioUrl);
    selectedAudioUrl = null;
  }
  if (audioPreview) {
    audioPreview.removeAttribute('src');
    audioPreview.hidden = true;
  }

  audioInput.value = '';
  fileRow.hidden = true;
  analyzeBtn.disabled = true;
  resultCard.hidden = true;
  clearError();

  assistantPanel.classList.remove('is-listening', 'is-analyzing');
  micBtn.classList.remove('is-recording');
  micBtn.setAttribute('aria-label', 'Start recording');
  if (recordingIndicator) recordingIndicator.hidden = true;
  waveform.classList.remove('active');

  // Reset waveform bars transform
  const bars = waveform.querySelectorAll('span');
  bars.forEach(b => b.style.transform = '');

  assistantKicker.textContent = 'YOUR VOICE, IN A NEW LIGHT';
  assistantTitle.textContent = 'Ready when you are.';
  assistantText.textContent = 'Press the microphone and say a few words. I’ll listen, then analyze the emotional tone of your speech.';
  liveState.textContent = 'STANDBY';
  micHint.textContent = 'TAP TO START RECORDING';
}

// Live real-time audio waveform visualizer using Web Audio API
function startLiveVisualizer(mediaStream) {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    audioContext = new AudioCtx();
    const sourceNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 64;
    analyserNode.smoothingTimeConstant = 0.8;
    sourceNode.connect(analyserNode);

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const bars = waveform.querySelectorAll('span');

    function draw() {
      if (!analyserNode) return;
      visualizerAnimationId = requestAnimationFrame(draw);
      analyserNode.getByteFrequencyData(dataArray);

      for (let i = 0; i < bars.length; i++) {
        const binIndex = Math.min(i % bufferLength, bufferLength - 1);
        const val = dataArray[binIndex] / 255;
        const scale = Math.max(0.2, val * 2.2);
        bars[i].style.transform = `scaleY(${scale.toFixed(2)})`;
      }
    }
    draw();
  } catch (err) {
    console.warn('Web Audio Visualizer could not start:', err);
  }
}

function stopLiveVisualizer() {
  if (visualizerAnimationId) {
    cancelAnimationFrame(visualizerAnimationId);
    visualizerAnimationId = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }
  analyserNode = null;
  const bars = waveform.querySelectorAll('span');
  bars.forEach(b => b.style.transform = '');
}

// Upload local audio file
uploadBtn.addEventListener('click', () => audioInput.click());
audioInput.addEventListener('change', () => {
  const file = audioInput.files && audioInput.files[0];
  if (file) {
    setAudio(file, file.name);
  }
});

clearFile.addEventListener('click', resetAudio);

// Microphone recording control
micBtn.addEventListener('click', async () => {
  clearError();

  // If already recording -> STOP RECORDING
  if (recorder && recorder.state === 'recording') {
    const elapsed = (performance.now() - recordStartTime) / 1000;
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }

    if (elapsed < MIN_RECORD_SECONDS) {
      // User stopped before 3 seconds
      showError(`Recording was too short (${elapsed.toFixed(1)}s). Please speak continuously for at least 3 to 5 seconds so speech emotion can be accurately detected.`);
      try { recorder.stop(); } catch (e) {}
      if (stream) stream.getTracks().forEach(t => t.stop());
      stream = null;
      resetAudio();
      return;
    }

    // Valid duration -> stop and let onstop handle
    micHint.textContent = 'FINALIZING RECORDING...';
    try {
      recorder.stop();
    } catch (err) {
      console.error('Error stopping recorder:', err);
    }
    return;
  }

  // Check browser capability
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    showError('Microphone recording is not supported in this browser. Please use Google Chrome, Edge, or Firefox on localhost / HTTPS.');
    return;
  }

  // START RECORDING
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    audioChunks = [];

    // Detect supported MIME type
    let mimeType = 'audio/webm;codecs=opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'audio/webm';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/ogg;codecs=opus';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'audio/mp4';
          if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = '';
          }
        }
      }
    }

    const options = mimeType ? { mimeType } : {};
    recorder = new MediaRecorder(stream, options);

    recorder.ondataavailable = event => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    recorder.onstop = () => {
      stopLiveVisualizer();
      if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
      }

      const totalElapsed = (performance.now() - recordStartTime) / 1000;

      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }

      assistantPanel.classList.remove('is-listening');
      micBtn.classList.remove('is-recording');
      micBtn.setAttribute('aria-label', 'Start recording');
      if (recordingIndicator) recordingIndicator.hidden = true;
      waveform.classList.remove('active');

      if (totalElapsed < MIN_RECORD_SECONDS) {
        showError(`Recording was too short (${totalElapsed.toFixed(1)}s). Please record for at least 3 to 5 seconds.`);
        return;
      }

      if (audioChunks.length === 0) {
        showError('No audio data was captured by your microphone. Please check your mic connection.');
        return;
      }

      const actualMime = recorder.mimeType || mimeType || 'audio/webm';
      const ext = actualMime.includes('ogg') ? 'ogg' : (actualMime.includes('mp4') ? 'm4a' : 'webm');
      const audioBlob = new Blob(audioChunks, { type: actualMime });

      if (audioBlob.size < 1000) {
        showError('Recording was empty or silent. Please check your microphone input level and speak clearly.');
        return;
      }

      const recordedFile = new File([audioBlob], `voxaura-recording.${ext}`, { type: actualMime });
      setAudio(recordedFile, `Your voice recording (${totalElapsed.toFixed(1)}s)`, totalElapsed);
    };

    recorder.start(250); // Emit audio chunk every 250ms
    recordStartTime = performance.now();

    // UI Updates for Active Recording
    assistantPanel.classList.add('is-listening');
    micBtn.classList.add('is-recording');
    micBtn.setAttribute('aria-label', 'Stop recording');
    waveform.classList.add('active');
    startLiveVisualizer(stream);

    liveState.textContent = 'LISTENING';
    assistantKicker.textContent = 'I’M LISTENING TO YOUR VOICE';
    assistantTitle.textContent = 'Speak with emotion now.';
    assistantText.textContent = 'Say a sentence or two naturally (e.g., happy, calm, or expressive). Keep speaking for at least 3 seconds, then tap Stop.';

    if (recordingIndicator) {
      recordingIndicator.hidden = false;
      recordingIndicator.classList.remove('ready');
      recordingTimer.textContent = '0.0s';
      recordingGuide.textContent = '(keep speaking: 3.0s minimum)';
    }
    micHint.textContent = 'RECORDING (0.0s) — TAP TO STOP';

    // Live Timer
    timerInterval = setInterval(() => {
      const elapsed = (performance.now() - recordStartTime) / 1000;
      if (recordingTimer) recordingTimer.textContent = `${elapsed.toFixed(1)}s`;

      if (elapsed < MIN_RECORD_SECONDS) {
        const remaining = (MIN_RECORD_SECONDS - elapsed).toFixed(1);
        if (recordingGuide) recordingGuide.textContent = `(keep speaking: ${remaining}s left)`;
        micHint.textContent = `RECORDING (${elapsed.toFixed(1)}s) — MIN 3s`;
        if (recordingIndicator) recordingIndicator.classList.remove('ready');
      } else {
        if (recordingGuide) recordingGuide.textContent = '(Ready! Tap Stop whenever you are done)';
        micHint.textContent = `READY TO STOP (${elapsed.toFixed(1)}s)`;
        if (recordingIndicator) recordingIndicator.classList.add('ready');
      }
    }, 100);

  } catch (error) {
    console.error('Microphone access failed:', error);
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      showError('Microphone permission was denied. Please allow microphone access in your browser settings (click the camera/mic icon in the address bar) and try again.');
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      showError('No microphone detected on your computer. Please connect a microphone or headset and try again.');
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      showError('Microphone is in use by another program. Please close other audio applications and try again.');
    } else {
      showError(`Could not access microphone: ${error.message || error}`);
    }
    resetAudio();
  }
});

// Analyze Audio
analyzeBtn.addEventListener('click', async () => {
  if (!selectedAudio) return;
  clearError();

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = 'Analyzing voice… <span>✦</span>';
  assistantPanel.classList.remove('is-listening');
  assistantPanel.classList.add('is-analyzing');
  waveform.classList.add('active');

  liveState.textContent = 'ANALYZING';
  assistantKicker.textContent = 'EXTRACTING ACOUSTIC PATTERNS';
  assistantTitle.textContent = 'Decoding your voice emotion...';
  assistantText.textContent = 'Calculating 236 acoustic features (MFCCs, spectral centroid, zero-crossing rate, chroma, and mel spectrograms) and evaluating them against your trained model.';

  try {
    const formData = new FormData();
    formData.append('audio', selectedAudio, selectedAudio.name || 'recording.webm');

    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Emotion analysis failed. Please try speaking clearly again.');
    }

    // Display prediction and confidence
    const emotionEl = document.getElementById('resultEmotion');
    const confidenceEl = document.getElementById('resultConfidence');

    emotionEl.textContent = data.emotion;
    confidenceEl.textContent = `${Number(data.confidence).toFixed(1)}%`;

    // Populate probability breakdown
    const list = document.getElementById('probabilityList');
    list.innerHTML = '';

    const sortedProbs = Object.entries(data.probabilities || {}).sort((a, b) => b[1] - a[1]);
    sortedProbs.forEach(([emotion, probability]) => {
      const item = document.createElement('div');
      item.className = 'probability-item';

      const label = document.createElement('span');
      label.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);

      const track = document.createElement('div');
      track.className = 'probability-track';

      const fill = document.createElement('div');
      fill.className = 'probability-fill';
      fill.style.width = `${Math.max(0, Math.min(100, probability))}%`;
      track.appendChild(fill);

      const value = document.createElement('span');
      value.className = 'probability-value';
      value.textContent = `${Number(probability).toFixed(1)}%`;

      item.append(label, track, value);
      list.appendChild(item);
    });

    resultCard.hidden = false;
    assistantKicker.textContent = 'ANALYSIS COMPLETE';
    assistantTitle.textContent = `I detected ${data.emotion.toLowerCase()} in your voice.`;
    assistantText.textContent = `Your voice expressed a primary emotional tone of ${data.emotion} with ${Number(data.confidence).toFixed(1)}% model confidence. Check the distribution below for details.`;
    liveState.textContent = 'COMPLETE';
    micHint.textContent = 'ANALYSIS COMPLETE — READY FOR NEXT TEST';

  } catch (error) {
    console.error('Analysis error:', error);
    showError(error.message || 'Something went wrong during emotion analysis.');
    assistantTitle.textContent = 'Let’s try that again.';
    assistantText.textContent = 'Ensure your microphone captured clear speech and that the recording is at least 3 seconds long.';
    liveState.textContent = 'NEEDS ATTENTION';
  } finally {
    assistantPanel.classList.remove('is-analyzing');
    waveform.classList.remove('active');
    analyzeBtn.disabled = !selectedAudio;
    analyzeBtn.innerHTML = 'Analyze emotion <span>↗</span>';
  }
});

// About modal controls
const aboutModal = document.getElementById('aboutModal');
if (aboutModal) {
  const aboutBtn = document.getElementById('aboutBtn');
  const closeModal = document.getElementById('closeModal');
  const modalOk = document.getElementById('modalOk');

  if (aboutBtn) aboutBtn.addEventListener('click', () => aboutModal.hidden = false);
  if (closeModal) closeModal.addEventListener('click', () => aboutModal.hidden = true);
  if (modalOk) modalOk.addEventListener('click', () => aboutModal.hidden = true);
  aboutModal.addEventListener('click', e => { if (e.target === aboutModal) aboutModal.hidden = true; });
}

// Initial status polling
fetch('/api/status')
  .then(r => r.json())
  .then(data => {
    const status = document.getElementById('modelStatus');
    if (status) {
      status.textContent = data.model_ready ? 'ML MODEL CONNECTED' : 'MODEL NOT CONNECTED';
    }
  })
  .catch(() => {});
