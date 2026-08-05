// ========== 反向传播页面 - 梯度逐层回传可视化 ==========
const bwdNet = [3, 4, 4, 1];
let bwdAnimId = null;
let bwdIsPlaying = false;
let bwdCurrentStep = 0; // 0=not started, 1=output done, 2=hidden2 done, 3=hidden1 done
let bwdProgress = 0;
let bwdFnKey = 'sigmoid';
let bwdData = null;
let bwdCtx = null;
const BWD_TOTAL_STEPS = bwdNet.length - 1; // 3 steps

const bwdActivations = {
    sigmoid: { name: 'Sigmoid', fn: x => 1 / (1 + Math.exp(-x)), dfn: x => { const s = 1/(1+Math.exp(-x)); return s*(1-s); }, color: '#e63946' },
    relu: { name: 'ReLU', fn: x => Math.max(0, x), dfn: x => x > 0 ? 1 : 0, color: '#2a9d8f' },
    tanh: { name: 'Tanh', fn: x => Math.tanh(x), dfn: x => 1 - Math.tanh(x) * Math.tanh(x), color: '#457b9d' }
};

// Generate same params as forward page for consistency
function generateBwdParams() {
    let seed = 123;
    function seededRandom() {
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        return (seed / 4294967296) - 0.5;
    }
    const weights = [];
    const biases = [];
    for (let l = 0; l < bwdNet.length - 1; l++) {
        const wl = [], bl = [];
        for (let j = 0; j < bwdNet[l + 1]; j++) {
            const wj = [];
            for (let i = 0; i < bwdNet[l]; i++) {
                wj.push(Math.round(seededRandom() * 2 * 100) / 100);
            }
            wl.push(wj);
            bl.push(Math.round(seededRandom() * 0.5 * 100) / 100);
        }
        weights.push(wl);
        biases.push(bl);
    }
    return { weights, biases };
}

// Full forward + backward computation
function computeBackward(fnKey) {
    const act = bwdActivations[fnKey];
    const inputVals = [0.8, -1.2, 1.5];
    const params = generateBwdParams();
    const target = 1.0; // target output

    // Forward pass
    const aLayers = [inputVals.slice()];
    const zLayers = [null];
    for (let l = 0; l < bwdNet.length - 1; l++) {
        const zVals = [], aVals = [];
        for (let j = 0; j < bwdNet[l + 1]; j++) {
            let z = params.biases[l][j];
            for (let i = 0; i < bwdNet[l]; i++) {
                z += aLayers[l][i] * params.weights[l][j][i];
            }
            zVals.push(z);
            aVals.push(act.fn(z));
        }
        zLayers.push(zVals);
        aLayers.push(aVals);
    }

    // Loss: MSE = 0.5 * (y - t)^2
    const output = aLayers[bwdNet.length - 1][0];
    const loss = 0.5 * Math.pow(output - target, 2);

    // Backward pass: compute deltas (δ = ∂L/∂z for each node)
    const deltas = new Array(bwdNet.length).fill(null);

    // Output layer delta: dL/da * da/dz = (a - t) * f'(z)
    const L = bwdNet.length - 1;
    deltas[L] = [(output - target) * act.dfn(zLayers[L][0])];

    // Hidden layers: delta[l][i] = (Σ_j w[l][j][i] * delta[l+1][j]) * f'(z[l][i])
    for (let l = L - 1; l >= 1; l--) {
        const dLayer = [];
        for (let i = 0; i < bwdNet[l]; i++) {
            let sum = 0;
            for (let j = 0; j < bwdNet[l + 1]; j++) {
                sum += params.weights[l][j][i] * deltas[l + 1][j];
            }
            dLayer.push(sum * act.dfn(zLayers[l][i]));
        }
        deltas[l] = dLayer;
    }

    // Weight gradients: dL/dw[l][j][i] = delta[l+1][j] * a[l][i]
    const wGrads = [];
    const bGrads = [];
    for (let l = 0; l < bwdNet.length - 1; l++) {
        const wgl = [], bgl = [];
        for (let j = 0; j < bwdNet[l + 1]; j++) {
            const wgj = [];
            for (let i = 0; i < bwdNet[l]; i++) {
                wgj.push(deltas[l + 1][j] * aLayers[l][i]);
            }
            wgl.push(wgj);
            bgl.push(deltas[l + 1][j]);
        }
        wGrads.push(wgl);
        bGrads.push(bgl);
    }

    return { aLayers, zLayers, params, deltas, wGrads, bGrads, loss, target, output };
}

// Main drawing
function drawBackwardPage(ctx, W, H, fnKey, data, animLayer, animProgress) {
    ctx.clearRect(0, 0, W, H);
    const fn = bwdActivations[fnKey];

    // Title
    ctx.fillStyle = '#333'; ctx.font = 'bold 15px Arial'; ctx.textAlign = 'center';
    ctx.fillText('反向传播: 梯度逐层回传', W / 2, 22);

    // Layout
    const marginX = 90, marginTop = 45, marginBottom = 70;
    const netW = W - marginX * 2;
    const netH = H - marginTop - marginBottom;
    const layerSpacing = netW / (bwdNet.length - 1);
    const nodeR = 30;

    // Node positions
    const positions = [];
    for (let l = 0; l < bwdNet.length; l++) {
        const lx = marginX + l * layerSpacing;
        const count = bwdNet[l];
        const totalH = count > 1 ? Math.min(netH - 30, (count - 1) * 110) : 0;
        const startY = marginTop + (netH - totalH) / 2;
        const layerPos = [];
        for (let i = 0; i < count; i++) {
            const y = count > 1 ? startY + i * (totalH / (count - 1)) : marginTop + netH / 2;
            layerPos.push({ x: lx, y: y });
        }
        positions.push(layerPos);
    }

    const L = bwdNet.length - 1;
    // animLayer counts from output backward: 0=not started, 1=output done, 2=hidden2 done...
    // When animLayer=0, no layer has gradients yet, so doneUpTo should be > L
    const doneUpTo = animLayer === 0 ? L + 1 : L - animLayer + 1;

    // Find max gradient for normalization
    let maxGrad = 0;
    for (let l = 0; l < data.wGrads.length; l++) {
        for (let j = 0; j < data.wGrads[l].length; j++) {
            for (let i = 0; i < data.wGrads[l][j].length; i++) {
                maxGrad = Math.max(maxGrad, Math.abs(data.wGrads[l][j][i]));
            }
        }
    }
    if (maxGrad === 0) maxGrad = 1;

    // Draw connections with gradient intensity
    for (let l = 0; l < positions.length - 1; l++) {
        const layerGradDone = (l + 1) >= doneUpTo && (l + 1) <= L;
        const isAnimating = (l + 1 === L - animLayer + 1) && animProgress < 1;
        for (let i = 0; i < positions[l].length; i++) {
            for (let j = 0; j < positions[l + 1].length; j++) {
                const from = positions[l][i], to = positions[l + 1][j];
                ctx.beginPath();
                ctx.moveTo(from.x + nodeR, from.y);
                ctx.lineTo(to.x - nodeR, to.y);

                if (layerGradDone && !isAnimating) {
                    const grad = Math.abs(data.wGrads[l][j][i]);
                    const intensity = Math.min(1, grad / maxGrad);
                    ctx.strokeStyle = 'rgba(231, 76, 60, ' + (0.15 + intensity * 0.85) + ')';
                    ctx.lineWidth = 0.8 + intensity * 3;
                } else if (isAnimating) {
                    ctx.strokeStyle = 'rgba(231, 76, 60, ' + (0.2 + 0.5 * animProgress) + ')';
                    ctx.lineWidth = 1.5;
                } else {
                    ctx.strokeStyle = '#e0e0e0';
                    ctx.lineWidth = 0.8;
                }
                ctx.stroke();

                // Show gradient value on connection
                if (layerGradDone && !isAnimating) {
                    const grad = data.wGrads[l][j][i];
                    const t = 0.65;
                    const mx = from.x + nodeR + (to.x - nodeR - from.x - nodeR) * t;
                    const my = from.y + (to.y - from.y) * t;
                    const angle = Math.atan2(to.y - from.y, to.x - from.x);
                    const perpX = -Math.sin(angle) * 8;
                    const perpY = Math.cos(angle) * 8;
                    ctx.font = '7px Arial';
                    ctx.fillStyle = Math.abs(grad) > 0.1 ? '#c0392b' : '#999';
                    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    ctx.fillText('∇' + grad.toFixed(3), mx + perpX, my + perpY);
                }
            }
        }
    }

    // Draw gradient flow particles (backward direction)
    if (animLayer >= 1 && animProgress < 1) {
        const targetL = L - animLayer + 1; // layer receiving gradient
        if (targetL >= 1 && targetL < bwdNet.length) {
            for (let i = 0; i < positions[targetL].length; i++) {
                for (let j = 0; j < positions[targetL - 1].length; j++) {
                    const from = positions[targetL][i], to = positions[targetL - 1][j];
                    const px = from.x - nodeR + ((to.x + nodeR) - (from.x - nodeR)) * animProgress;
                    const py = from.y + (to.y - from.y) * animProgress;
                    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2);
                    ctx.fillStyle = '#e74c3c'; ctx.globalAlpha = 0.7; ctx.fill();
                    ctx.globalAlpha = 1;
                }
            }
        }
    }

    // Draw nodes
    for (let l = 0; l < positions.length; l++) {
        for (let i = 0; i < positions[l].length; i++) {
            const pos = positions[l][i];
            const hasGrad = l >= doneUpTo && l >= 1;
            const isAnimNode = (l === L - animLayer + 1) && animProgress >= 0.7 && animProgress < 1;

            ctx.beginPath(); ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);

            if (l === 0) {
                ctx.fillStyle = '#fffde6'; ctx.fill();
                ctx.strokeStyle = '#d4a017'; ctx.lineWidth = 2; ctx.stroke();
                ctx.fillStyle = '#333'; ctx.font = 'bold 11px Arial';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText('a=' + data.aLayers[0][i].toFixed(1), pos.x, pos.y);
            } else if (hasGrad && !isAnimNode) {
                ctx.fillStyle = '#fff5f5'; ctx.fill();
                ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 2; ctx.stroke();
                drawBwdNodeContent(ctx, pos, data.deltas[l][i], data.aLayers[l][i], nodeR);
            } else if (isAnimNode) {
                const pulse = 1 + 0.05 * Math.sin((animProgress - 0.7) * 6 * Math.PI);
                ctx.beginPath(); ctx.arc(pos.x, pos.y, nodeR * pulse, 0, Math.PI * 2);
                ctx.fillStyle = '#fff8f8'; ctx.fill();
                ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 2.5; ctx.stroke();
                const showP = (animProgress - 0.7) / 0.3;
                drawBwdNodeContent(ctx, pos, data.deltas[l][i], data.aLayers[l][i], nodeR, showP);
            } else {
                ctx.fillStyle = '#fafafa'; ctx.fill();
                ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1.5; ctx.stroke();
                ctx.fillStyle = '#999'; ctx.font = '10px Arial';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText('a=' + data.aLayers[l][i].toFixed(2), pos.x, pos.y);
            }
        }
    }

    // Layer labels
    const layerNames = ['输入层', '隐藏层1', '隐藏层2', '输出层'];
    ctx.fillStyle = '#555'; ctx.font = '12px Arial'; ctx.textAlign = 'center';
    for (let l = 0; l < positions.length; l++) {
        ctx.fillText(layerNames[l], positions[l][0].x, H - marginBottom + 20);
    }

    // Current backward direction indicator
    if (animLayer >= 1 && animLayer <= L && animProgress < 1) {
        const targetL = L - animLayer + 1;
        if (targetL >= 1) {
            const lx = positions[targetL][0].x;
            ctx.font = 'bold 12px Arial'; ctx.fillStyle = '#e74c3c'; ctx.textAlign = 'center';
            ctx.fillText('← 回传 δ = Σ(w·δ)·f\'(z)', lx, marginTop - 10);
        }
    }

    // Loss display
    ctx.font = '12px Arial'; ctx.fillStyle = '#333'; ctx.textAlign = 'right';
    ctx.fillText('Loss = ' + data.loss.toFixed(4) + '  (target=' + data.target + ')', W - 30, 22);

    // Bottom legend
    const legendY = H - 18;
    ctx.font = '11px Arial'; ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(231, 76, 60, 0.9)'; ctx.fillRect(50, legendY - 4, 20, 3);
    ctx.fillStyle = '#555'; ctx.fillText('梯度大(更新多)', 75, legendY + 1);
    ctx.fillStyle = 'rgba(231, 76, 60, 0.3)'; ctx.fillRect(220, legendY - 4, 20, 3);
    ctx.fillStyle = '#555'; ctx.fillText('梯度小(更新少)', 245, legendY + 1);
    ctx.fillStyle = '#e74c3c';
    ctx.beginPath(); ctx.arc(415, legendY - 2, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#555'; ctx.fillText('梯度流(反向)', 423, legendY + 1);
}

// Node content: delta and a
function drawBwdNodeContent(ctx, pos, delta, a, nodeR, progress) {
    if (progress === undefined) progress = 1;
    ctx.font = '9px Arial'; ctx.textAlign = 'center';
    ctx.globalAlpha = progress;

    // delta (top)
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = '#e74c3c';
    ctx.fillText('δ=' + delta.toFixed(3), pos.x, pos.y - 2);

    // a value (bottom)
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#666';
    ctx.fillText('a=' + a.toFixed(2), pos.x, pos.y + 4);
    ctx.globalAlpha = 1;
}

// Info panel
function updateBwdInfoPanel(fnKey, data, animLayer) {
    const panel = document.getElementById('backwardInfo');
    const fn = bwdActivations[fnKey];
    const lr = parseFloat(document.getElementById('lrInput').value) || 0.1;
    const L = bwdNet.length - 1;

    let html = '<h3>反向传播计算过程</h3>';
    html += '<div class="section"><div class="section-title">损失函数</div>';
    html += '<div class="formula">L = ½(ŷ - t)² = ½(' + data.output.toFixed(4) + ' - ' + data.target + ')²</div>';
    html += '<div class="formula">L = <span class="highlight">' + data.loss.toFixed(4) + '</span></div></div>';

    // Output layer
    if (animLayer >= 1) {
        html += '<div class="section"><div class="section-title">输出层 δ</div>';
        html += '<div style="font-size:11px;padding:6px;background:#fef5f5;border-radius:4px;border-left:3px solid #e74c3c;">';
        html += 'δ = (ŷ - t) · f\'(z)<br>';
        html += 'δ = (' + data.output.toFixed(4) + ' - ' + data.target + ') × f\'(' + data.zLayers[L][0].toFixed(3) + ')<br>';
        html += '<b>δ = ' + data.deltas[L][0].toFixed(4) + '</b>';
        html += '</div></div>';
    }

    // Hidden layers
    for (let step = 2; step <= animLayer; step++) {
        const l = L - step + 1;
        if (l < 1) break;
        const lName = '隐藏层' + l;
        html += '<div class="section"><div class="section-title">' + lName + ' δ</div>';
        html += '<div style="font-size:11px;">';
        for (let i = 0; i < bwdNet[l]; i++) {
            html += '<div style="margin:3px 0;padding:5px;background:#fef5f5;border-radius:4px;border-left:3px solid #e74c3c;">';
            html += '<b>节点' + (i + 1) + ':</b> ';
            let terms = [];
            for (let j = 0; j < bwdNet[l + 1]; j++) {
                terms.push('w·δ=' + (data.params.weights[l][j][i] * data.deltas[l + 1][j]).toFixed(4));
            }
            html += '<span style="color:#666;font-size:10px;">Σ(w·δ)=' + terms.join('+') + '</span><br>';
            html += 'δ = Σ · f\'(z) = <b>' + data.deltas[l][i].toFixed(4) + '</b>';
            html += '</div>';
        }
        html += '</div></div>';
    }

    // Weight update preview
    if (animLayer >= L) {
        html += '<div class="section" style="background:#e8f5e9;padding:10px;border-radius:6px;">';
        html += '<div class="section-title" style="color:#2e7d32;">权重更新 (lr=' + lr + ')</div>';
        html += '<div style="font-size:10px;max-height:120px;overflow-y:auto;">';
        for (let l = 0; l < bwdNet.length - 1; l++) {
            for (let j = 0; j < bwdNet[l + 1]; j++) {
                for (let i = 0; i < bwdNet[l]; i++) {
                    const grad = data.wGrads[l][j][i];
                    const oldW = data.params.weights[l][j][i];
                    const newW = oldW - lr * grad;
                    if (Math.abs(grad) > 0.05) {
                        html += 'w[' + l + '][' + j + '][' + i + ']: ' + oldW.toFixed(3) + ' → <b>' + newW.toFixed(3) + '</b> (Δ=' + (-lr * grad).toFixed(4) + ')<br>';
                    }
                }
            }
        }
        html += '</div></div>';
    }

    panel.innerHTML = html;
}

function bwdInitCanvas() {
    bwdFnKey = document.getElementById('backwardActSelect').value;
    bwdData = computeBackward(bwdFnKey);
    var canvas = document.getElementById('backwardCanvas');
    canvas.width = 850; canvas.height = 560;
    bwdCtx = canvas.getContext('2d');
}

function bwdRender() {
    if (!bwdCtx || !bwdData) bwdInitCanvas();
    drawBackwardPage(bwdCtx, 850, 560, bwdFnKey, bwdData, bwdCurrentStep, bwdProgress >= 1 ? 1 : bwdProgress);
    updateBwdInfoPanel(bwdFnKey, bwdData, bwdCurrentStep);
}

// Render static
function renderBackwardPage() {
    stopBackwardAnim();
    bwdCurrentStep = 0;
    bwdProgress = 1;
    bwdInitCanvas();
    bwdRender();
}

function bwdAnimStep() {
    if (!bwdIsPlaying) return;
    bwdProgress += 0.012;
    var L = bwdNet.length - 1;
    if (bwdProgress >= 1) {
        bwdProgress = 1;
        bwdRender();
        bwdCurrentStep++;
        if (bwdCurrentStep > L) {
            bwdIsPlaying = false;
            var btn = document.getElementById('bwdPlayBtn');
            if (btn) btn.textContent = '播放';
            bwdAnimId = null;
            return;
        }
        bwdProgress = 0;
        bwdAnimId = setTimeout(function() { bwdAnimId = requestAnimationFrame(bwdAnimStep); }, 600);
        return;
    }
    bwdRender();
    bwdAnimId = requestAnimationFrame(bwdAnimStep);
}

// Animate
function startBackwardAnim() {
    stopBackwardAnim();
    bwdCurrentStep = 1;
    bwdProgress = 0;
    bwdIsPlaying = true;
    bwdInitCanvas();
    bwdRender();
    var btn = document.getElementById('bwdPlayBtn');
    if (btn) btn.textContent = '暂停';
setTimeout(function() { bwdAnimId = requestAnimationFrame(bwdAnimStep); }, 500);
}

function stopBackwardAnim() {
    bwdIsPlaying = false;
    if (bwdAnimId) {
        cancelAnimationFrame(bwdAnimId);
        clearTimeout(bwdAnimId);
        bwdAnimId = null;
    }
}

function toggleBackwardAnim() {
    var btn = document.getElementById('bwdPlayBtn');
    var L = bwdNet.length - 1;
    if (bwdIsPlaying) {
        stopBackwardAnim();
        if (btn) btn.textContent = '播放';
    } else {
        if (bwdCurrentStep === 0 || bwdCurrentStep > L) {
            startBackwardAnim();
        } else {
            bwdIsPlaying = true;
            if (btn) btn.textContent = '暂停';
            bwdAnimId = requestAnimationFrame(bwdAnimStep);
        }
    }
}

function nextBackwardStep() {
    stopBackwardAnim();
    var btn = document.getElementById('bwdPlayBtn');
    if (btn) btn.textContent = '播放';
    var L = bwdNet.length - 1;
    if (bwdCurrentStep === 0) {
        bwdInitCanvas();
        bwdCurrentStep = 1;
    } else {
        bwdCurrentStep++;
    }
    if (bwdCurrentStep > L) {
        bwdCurrentStep = L;
    }
    bwdProgress = 1;
    bwdRender();
}

// Reset
function resetBackwardAnim() {
    stopBackwardAnim();
    bwdCurrentStep = 0;
    bwdProgress = 1;
    bwdInitCanvas();
    bwdRender();
    var btn = document.getElementById('bwdPlayBtn');
    if (btn) btn.textContent = '播放';
}