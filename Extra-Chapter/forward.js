// ========== 前向传播页面 - 逐层数据流可视化 ==========
const fwdNet = [3, 4, 4, 1];
let fwdAnimId = null;
let fwdIsPlaying = false;
let fwdCurrentLayer = 0;
let fwdProgress = 0;
let fwdFnKey = 'relu';
let fwdData = null;
let fwdCtx = null;
const FWD_TOTAL_LAYERS = fwdNet.length - 1;

const fwdActivations = {
    sigmoid: { name: 'Sigmoid', fn: x => 1 / (1 + Math.exp(-x)), color: '#e63946' },
    relu: { name: 'ReLU', fn: x => Math.max(0, x), color: '#2a9d8f' },
    tanh: { name: 'Tanh', fn: x => Math.tanh(x), color: '#457b9d' }
};

// Generate deterministic weights and biases
function generateFwdParams() {
    let seed = 123;
    function seededRandom() {
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        return (seed / 4294967296) - 0.5;
    }
    const weights = []; // weights[l][j][i] = weight from layer l node i to layer l+1 node j
    const biases = [];  // biases[l][j] = bias of layer l+1 node j
    for (let l = 0; l < fwdNet.length - 1; l++) {
        const wl = [];
        const bl = [];
        for (let j = 0; j < fwdNet[l + 1]; j++) {
            const wj = [];
            for (let i = 0; i < fwdNet[l]; i++) {
                wj.push(Math.round(seededRandom() * 2 * 100) / 100); // round to 2 decimals
            }
            wl.push(wj);
            bl.push(Math.round(seededRandom() * 0.5 * 100) / 100);
        }
        weights.push(wl);
        biases.push(bl);
    }
    return { weights, biases };
}

// Forward pass computation
function computeForward(fnKey) {
    const actFn = fwdActivations[fnKey].fn;
    const inputVals = [0.8, -1.2, 1.5];
    const params = generateFwdParams();
    const layers = [inputVals.slice()]; // a values per layer
    const zLayers = [null]; // z values (null for input layer)

    for (let l = 0; l < fwdNet.length - 1; l++) {
        const zVals = [];
        const aVals = [];
        for (let j = 0; j < fwdNet[l + 1]; j++) {
            let z = params.biases[l][j];
            for (let i = 0; i < fwdNet[l]; i++) {
                z += layers[l][i] * params.weights[l][j][i];
            }
            z = Math.round(z * 1000) / 1000;
            zVals.push(z);
            aVals.push(actFn(z));
        }
        zLayers.push(zVals);
        layers.push(aVals);
    }
    return { layers, zLayers, params };
}

// Main drawing function
function drawForwardPage(ctx, W, H, fnKey, data, animLayer, animProgress) {
    ctx.clearRect(0, 0, W, H);
    const fn = fwdActivations[fnKey];

    // Title
    ctx.fillStyle = '#333'; ctx.font = 'bold 15px Arial'; ctx.textAlign = 'center';
    ctx.fillText('前向传播: 数据逐层流动', W / 2, 22);

    // Layout
    const marginX = 90, marginTop = 45, marginBottom = 70;
    const netW = W - marginX * 2;
    const netH = H - marginTop - marginBottom;
    const layerSpacing = netW / (fwdNet.length - 1);
    const nodeR = 30;

    // Node positions
    const positions = [];
    for (let l = 0; l < fwdNet.length; l++) {
        const lx = marginX + l * layerSpacing;
        const count = fwdNet[l];
        const totalH = count > 1 ? Math.min(netH - 30, (count - 1) * 110) : 0;
        const startY = marginTop + (netH - totalH) / 2;
        const layerPos = [];
        for (let i = 0; i < count; i++) {
            const y = count > 1 ? startY + i * (totalH / (count - 1)) : marginTop + netH / 2;
            layerPos.push({ x: lx, y: y });
        }
        positions.push(layerPos);
    }

    // Draw connections with weight values
    for (let l = 0; l < positions.length - 1; l++) {
        const isDone = l < animLayer;
        const isAnimating = (l === animLayer - 1) && animProgress < 1;
        for (let i = 0; i < positions[l].length; i++) {
            for (let j = 0; j < positions[l + 1].length; j++) {
                const from = positions[l][i], to = positions[l + 1][j];
                const w = data.params.weights[l][j][i];
                ctx.beginPath();
                ctx.moveTo(from.x + nodeR, from.y);
                ctx.lineTo(to.x - nodeR, to.y);

                if (isDone) {
                    ctx.strokeStyle = w >= 0 ? '#4a90d9' : '#e63946';
                    ctx.lineWidth = Math.min(3, 0.8 + Math.abs(w) * 1.5);
                    ctx.globalAlpha = 0.7;
                } else if (isAnimating) {
                    ctx.strokeStyle = w >= 0 ? '#4a90d9' : '#e63946';
                    ctx.lineWidth = Math.min(3, 0.8 + Math.abs(w) * 1.5);
                    ctx.globalAlpha = 0.3 + 0.5 * animProgress;
                } else {
                    ctx.strokeStyle = '#e0e0e0';
                    ctx.lineWidth = 0.8;
                    ctx.globalAlpha = 1;
                }
                ctx.stroke();
                ctx.globalAlpha = 1;

                // Weight label on connection
                if (isDone || isAnimating) {
                    const t = 0.35;
                    const mx = from.x + nodeR + (to.x - nodeR - from.x - nodeR) * t;
                    const my = from.y + (to.y - from.y) * t;
                    const angle = Math.atan2(to.y - from.y, to.x - from.x);
                    const perpX = -Math.sin(angle) * 8;
                    const perpY = Math.cos(angle) * 8;
                    ctx.font = '8px Arial';
                    ctx.fillStyle = w >= 0 ? '#2a6cb8' : '#c0392b';
                    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    if (isAnimating) ctx.globalAlpha = animProgress;
                    ctx.fillText(w.toFixed(2), mx + perpX, my + perpY);
                    ctx.globalAlpha = 1;
                }
            }
        }
    }

    // Draw data flow particles (animated dots along connections)
    if (animLayer >= 1 && animProgress < 1) {
        const l = animLayer - 1;
        for (let i = 0; i < positions[l].length; i++) {
            for (let j = 0; j < positions[l + 1].length; j++) {
                const from = positions[l][i], to = positions[l + 1][j];
                const px = (from.x + nodeR) + ((to.x - nodeR) - (from.x + nodeR)) * animProgress;
                const py = from.y + (to.y - from.y) * animProgress;
                ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fillStyle = fn.color; ctx.globalAlpha = 0.8; ctx.fill();
                ctx.globalAlpha = 1;
            }
        }
    }

    // Draw nodes
    for (let l = 0; l < positions.length; l++) {
        for (let i = 0; i < positions[l].length; i++) {
            const pos = positions[l][i];
            const isInput = (l === 0);
            const isDone = (l >= 1 && l < animLayer);
            const isCurrentAnim = (l === animLayer && animProgress >= 0.8);
            const isJustDone = (l === animLayer && animProgress >= 1);

            ctx.beginPath(); ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);

            if (isInput) {
                ctx.fillStyle = '#fffde6'; ctx.fill();
                ctx.strokeStyle = '#d4a017'; ctx.lineWidth = 2; ctx.stroke();
                // Show input value
                ctx.fillStyle = '#333'; ctx.font = 'bold 11px Arial';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText('a=' + data.layers[0][i].toFixed(1), pos.x, pos.y);
            } else if (isDone || isJustDone) {
                ctx.fillStyle = '#f0f8ff'; ctx.fill();
                ctx.strokeStyle = fn.color; ctx.lineWidth = 2; ctx.stroke();
                drawFwdNodeContent(ctx, pos, data.zLayers[l][i], data.layers[l][i], fn, nodeR);
            } else if (isCurrentAnim) {
                // Node just received data, show computation result
                const pulse = 1 + 0.05 * Math.sin((animProgress - 0.8) * 5 * Math.PI);
                ctx.beginPath(); ctx.arc(pos.x, pos.y, nodeR * pulse, 0, Math.PI * 2);
                ctx.fillStyle = '#f8f8ff'; ctx.fill();
                ctx.strokeStyle = fn.color; ctx.lineWidth = 2.5; ctx.stroke();
                const showProgress = (animProgress - 0.8) / 0.2;
                drawFwdNodeContent(ctx, pos, data.zLayers[l][i], data.layers[l][i], fn, nodeR, showProgress);
            } else {
                ctx.fillStyle = '#fafafa'; ctx.fill();
                ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1.5; ctx.stroke();
                ctx.fillStyle = '#bbb'; ctx.font = '11px Arial';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                const label = l === positions.length - 1 ? 'y' + (i + 1) : 'h' + l + '.' + (i + 1);
                ctx.fillText(label, pos.x, pos.y);
            }
        }
    }

    // Layer labels
    const layerNames = ['输入层', '隐藏层1', '隐藏层2', '输出层'];
    ctx.fillStyle = '#555'; ctx.font = '12px Arial'; ctx.textAlign = 'center';
    for (let l = 0; l < positions.length; l++) {
        ctx.fillText(layerNames[l], positions[l][0].x, H - marginBottom + 20);
    }

    // Current computation indicator
    if (animLayer >= 1 && animLayer < fwdNet.length && animProgress < 1) {
        const lx = positions[animLayer][0].x;
        ctx.font = 'bold 12px Arial'; ctx.fillStyle = fn.color; ctx.textAlign = 'center';
        ctx.fillText('→ 计算 z = Σ(w·a) + b', lx, marginTop - 10);
    }

    // Bottom legend
    const legendY = H- 18;
    ctx.font = '11px Arial'; ctx.textAlign = 'left';
    ctx.fillStyle = '#4a90d9'; ctx.fillRect(50, legendY - 4, 20, 3);
    ctx.fillStyle = '#555'; ctx.fillText('正权重(w>0)', 75, legendY + 1);
    ctx.fillStyle = '#e63946'; ctx.fillRect(200, legendY - 4, 20, 3);
    ctx.fillStyle = '#555'; ctx.fillText('负权重(w<0)', 225, legendY + 1);
    ctx.fillStyle = fn.color;
    ctx.beginPath(); ctx.arc(365, legendY - 2, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#555'; ctx.fillText('数据流', 373, legendY + 1);
}

// Draw node content: z and a values
function drawFwdNodeContent(ctx, pos, z, a, fn, nodeR, progress) {
    if (progress === undefined) progress = 1;

    ctx.font = '9px Arial'; ctx.textAlign = 'center';

    // z value (top half)
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = '#666';
    ctx.globalAlpha = progress;
    ctx.fillText('z=' + z.toFixed(2), pos.x, pos.y - 2);

    // a value (bottom half)
    ctx.textBaseline = 'top';
    ctx.fillStyle = fn.color;
    ctx.fillText('a=' + a.toFixed(2), pos.x, pos.y + 4);
    ctx.globalAlpha = 1;
}

// Update info panel
function updateFwdInfoPanel(fnKey, data, animLayer) {
    const panel = document.getElementById('forwardInfo');
 const fn = fwdActivations[fnKey];
    let html = '<h3>前向传播计算过程</h3>';
    html += '<div class="section"><div class="section-title">网络结构</div>';
    html += '<div style="font-size:12px;">' + fwdNet.join(' → ') + ' (激活: ' + fn.name + ')</div></div>';
    html += '<div class="section"><div class="section-title">输入值</div>';
    html += '<div class="formula">a⁰ = [' + data.layers[0].map(v => v.toFixed(1)).join(', ') + ']</div></div>';

    for (let l = 1; l < fwdNet.length && l <= animLayer; l++) {
        const lName = l === fwdNet.length - 1 ? '输出层' : '隐藏层' + l;
        html += '<div class="section"><div class="section-title">' + lName + ' 计算</div>';
        html += '<div style="font-size:11px;margin:4px 0;">';
        for (let j = 0; j < fwdNet[l]; j++) {
    html += '<div style="margin:4px 0;padding:6px;background:#f8f9fa;border-radius:4px;border-left:3px solid ' + fn.color + ';">';
            html += '<b>节点' + (j + 1) + ':</b><br>';
            // Show z computation
            let terms = [];
            for (let i = 0; i < fwdNet[l - 1]; i++) {
                const w = data.params.weights[l - 1][j][i];
                const aVal = data.layers[l - 1][i];
                terms.push(w.toFixed(2) + '×' + aVal.toFixed(2));
            }
            html += '<span style="color:#666;font-size:10px;">z = ' + terms.join(' + ') + ' + ' + data.params.biases[l - 1][j].toFixed(2) + '</span><br>';
            html += '<span style="color:#333;">z = <b>' + data.zLayers[l][j].toFixed(3) + '</b></span><br>';
            html += '<span style="color:' + fn.color + ';">a = ' + fn.name + '(' + data.zLayers[l][j].toFixed(3) + ') = <b>' + data.layers[l][j].toFixed(4) + '</b></span>';
            html += '</div>';
        }
        html += '</div></div>';
    }

    if (animLayer >= fwdNet.length - 1) {
        html += '<div class="section" style="background:#e8f5e9;padding:10px;border-radius:6px;">';
        html += '<div class="section-title" style="color:#2e7d32;">最终输出</div>';
        html += '<div class="formula">ŷ = ' + data.layers[fwdNet.length - 1][0].toFixed(4) + '</div>';
        html += '<div style="font-size:11px;color:#666;">数据从输入层经过' + (fwdNet.length - 2) + '个隐藏层，最终得到预测输出。</div>';
        html += '</div>';
    }

    panel.innerHTML = html;
}

function fwdInitCanvas() {
    fwdFnKey = document.getElementById('forwardActSelect').value;
    fwdData = computeForward(fwdFnKey);
    var canvas = document.getElementById('forwardCanvas');
    canvas.width = 850; canvas.height = 560;
    fwdCtx = canvas.getContext('2d');
}

function fwdRender() {
    if (!fwdCtx || !fwdData) fwdInitCanvas();
    drawForwardPage(fwdCtx, 850, 560, fwdFnKey, fwdData, fwdCurrentLayer, fwdProgress >= 1 ? 1 : fwdProgress);
    updateFwdInfoPanel(fwdFnKey, fwdData, fwdCurrentLayer);
}

// Render static page
function renderForwardPage() {
    stopForwardAnim();
    fwdCurrentLayer = 0;
    fwdProgress = 1;
    fwdInitCanvas();
    fwdRender();
}

function fwdAnimStep() {
    if (!fwdIsPlaying) return;
    fwdProgress += 0.012;
    if (fwdProgress >= 1) {
        fwdProgress = 1;
        fwdRender();
        fwdCurrentLayer++;
        if (fwdCurrentLayer >= fwdNet.length) {
            fwdIsPlaying = false;
            var btn = document.getElementById('fwdPlayBtn');
            if (btn) btn.textContent = '播放';
            fwdAnimId = null;
            return;
        }
        fwdProgress = 0;
        fwdAnimId = setTimeout(function() { fwdAnimId = requestAnimationFrame(fwdAnimStep); }, 600);
        return;
    }
    fwdRender();
    fwdAnimId = requestAnimationFrame(fwdAnimStep);
}

// Start animation
function startForwardAnim() {
    stopForwardAnim();
    fwdCurrentLayer = 1;
    fwdProgress = 0;
    fwdIsPlaying = true;
    fwdInitCanvas();
    fwdRender();
    var btn = document.getElementById('fwdPlayBtn');
    if (btn) btn.textContent = '暂停';
    setTimeout(function() { fwdAnimId = requestAnimationFrame(fwdAnimStep); }, 500);
}

function stopForwardAnim() {
    fwdIsPlaying = false;
    if (fwdAnimId) {
        cancelAnimationFrame(fwdAnimId);
        clearTimeout(fwdAnimId);
        fwdAnimId = null;
    }
}

function toggleForwardAnim() {
    var btn = document.getElementById('fwdPlayBtn');
    if (fwdIsPlaying) {
        stopForwardAnim();
        if (btn) btn.textContent = '播放';
    } else {
        if (fwdCurrentLayer === 0 || fwdCurrentLayer >= fwdNet.length) {
            startForwardAnim();
        } else {
            fwdIsPlaying = true;
            if (btn) btn.textContent = '暂停';
            fwdAnimId = requestAnimationFrame(fwdAnimStep);
        }
    }
}

function nextForwardStep() {
    stopForwardAnim();
    var btn = document.getElementById('fwdPlayBtn');
    if (btn) btn.textContent = '播放';
    if (fwdCurrentLayer === 0) {
        fwdInitCanvas();
        fwdCurrentLayer = 1;
    } else {
        fwdCurrentLayer++;
    }
    if (fwdCurrentLayer >= fwdNet.length) {
        fwdCurrentLayer = fwdNet.length - 1;
    }
    fwdProgress = 1;
    fwdRender();
}

// Reset
function resetForwardAnim() {
    stopForwardAnim();
    fwdCurrentLayer = 0;
    fwdProgress = 1;
    fwdInitCanvas();
    fwdRender();
    var btn = document.getElementById('fwdPlayBtn');
    if (btn) btn.textContent = '播放';
}