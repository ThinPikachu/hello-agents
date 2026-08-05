// ========== 激活函数页面 - z→a 对比可视化 ==========
const activationFns = {
    sigmoid: { name:'Sigmoid', fn: x => 1/(1+Math.exp(-x)), formula:'σ(x) = 1/(1+e⁻ˣ)', range:'(0, 1)', color:'#e63946', pros:'输出范围(0,1)，适合概率输出', cons:'梯度消失问题，输出非零中心' },
    tanh: { name:'Tanh', fn: x => Math.tanh(x), formula:'tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)', range:'(-1, 1)', color:'#457b9d', pros:'零中心输出，收敛更快', cons:'梯度消失问题仍存在' },
    relu: { name:'ReLU', fn: x => Math.max(0,x), formula:'ReLU(x) = max(0, x)', range:'[0, +∞)', color:'#2a9d8f', pros:'计算简单，缓解梯度消失', cons:'Dead ReLU问题（负区间梯度为0）' },
    leakyrelu: { name:'LeakyReLU', fn: x => x>0?x:0.1*x, formula:'LeakyReLU(x) = max(0.1x, x)', range:'(-∞, +∞)', color:'#e9c46a', pros:'解决Dead ReLU问题', cons:'超参数α需要调节' }
};

const actNet = [3, 4, 4, 1];
let actAnimId = null;
let actIsPlaying = false;
let actCurrentLayer = 0; // 0=idle, 1..3=layers done
let actProgress = 0;
let actFnKey = 'sigmoid';
let actData = null;
let actCtx = null;
const ACT_TOTAL_LAYERS = actNet.length - 1; // 3 layers to animate

function initActValues() {
    return [0.8, -1.2, 1.5];
}


function computeActValues(fnKey) {
    const fn = activationFns[fnKey].fn;
    const inputVals = initActValues();
    const raw = [inputVals.slice()];
    const activated = [inputVals.slice()];
    const weights = []; // weights[l][j][i]
    const biases = [];  // biases[l][j]
    let seed = 42;
    function seededRandom() { seed = (seed * 1664525 + 1013904223) % 4294967296; return (seed / 4294967296) - 0.5; }
    for (let l = 1; l < actNet.length; l++) {
        const zVals = [], aVals = [];
        const wl = [], bl = [];
        for (let j = 0; j < actNet[l]; j++) {
            const wj = [];
            let z = 0;
            for (let i = 0; i < actNet[l-1]; i++) {
                const w = seededRandom() * 2;
                wj.push(w);
                z += activated[l-1][i] * w;
            }
            const b = seededRandom() * 0.5;
            z += b;
            wl.push(wj);
            bl.push(b);
            zVals.push(z);
            aVals.push(fn(z));
        }
        weights.push(wl);
        biases.push(bl);
        raw.push(zVals);
        activated.push(aVals);
    }
    return { raw, activated, weights, biases };
}

function drawActivationPage(ctx2, W, H, fnKey, raw, activated, animLayer, animProgress) {
    ctx2.clearRect(0, 0, W, H);
    const fn = activationFns[fnKey];

    // === Title ===
    ctx2.fillStyle = fn.color; ctx2.font = 'bold 15px Arial'; ctx2.textAlign = 'center';
    ctx2.fillText(fn.name + ' 激活效果演示', W / 2, 22);

    // === Network layout: centered, full canvas ===
    const netMarginX = 80, netMarginTop = 40, netMarginBottom = 90;
    const netW = W - netMarginX * 2;
    const netH = H - netMarginTop - netMarginBottom;
    const layerSpacing = netW / (actNet.length - 1);
    const nodeR = 30;

    // Compute node positions
    const positions = [];
    for (let l = 0; l < actNet.length; l++) {
        const lx = netMarginX + l * layerSpacing;
        const count = actNet[l];
        const totalH = count > 1 ? Math.min(netH - 30, (count - 1) * 110) : 0;
        const startY = netMarginTop + (netH - totalH) / 2;
        const layerPos = [];
        for (let i = 0; i < count; i++) {
            const y = count > 1 ? startY + i * (totalH / (count - 1)) : netMarginTop + netH / 2;
            layerPos.push({ x: lx, y: y });
        }
        positions.push(layerPos);
    }

    // Draw connections
    for (let l = 0; l < positions.length - 1; l++) {
        const isDone = l < animLayer;
        const isAnimConn = (l === animLayer - 1) && animProgress < 1;
        for (let i = 0; i < positions[l].length; i++) {
            for (let j = 0; j < positions[l + 1].length; j++) {
                const from = positions[l][i], to = positions[l + 1][j];
                ctx2.beginPath();
                ctx2.moveTo(from.x + nodeR, from.y);
                ctx2.lineTo(to.x - nodeR, to.y);
                if (isDone) {
                    ctx2.strokeStyle = '#aaa';
                    ctx2.lineWidth = 1.5;
                } else if (isAnimConn) {
                    ctx2.strokeStyle = fn.color;
                    ctx2.lineWidth = 2;
                    ctx2.globalAlpha = 0.3 + 0.5 * animProgress;
                } else {
                    ctx2.strokeStyle = '#e0e0e0';
                    ctx2.lineWidth = 0.8;
                }
                ctx2.stroke();
                ctx2.globalAlpha = 1;
            }
        }
    }

    // Data flow: single particle per connection (same as forward page)
    if (animLayer >= 1 && animProgress < 1) {
        const l = animLayer - 1;
        for (let i = 0; i < positions[l].length; i++) {
            for (let j = 0; j < positions[l + 1].length; j++) {
                const from = positions[l][i], to = positions[l + 1][j];
                const px = (from.x + nodeR) + ((to.x - nodeR) - (from.x + nodeR)) * animProgress;
                const py = from.y + (to.y - from.y) * animProgress;
                ctx2.beginPath(); ctx2.arc(px, py, 3, 0, Math.PI * 2);
                ctx2.fillStyle = fn.color; ctx2.globalAlpha = 0.8; ctx2.fill();
                ctx2.globalAlpha = 1;
            }
        }
    }



    // Draw nodes
    for (let l = 0; l < positions.length; l++) {
        for (let i = 0; i < positions[l].length; i++) {
            const pos = positions[l][i];
            const isInput = (l === 0);
            const isFullyDone = (l >= 1 && l < animLayer);
            const isAnimating = (l === animLayer && l >= 1 && animProgress < 1);
            const isJustDone = (l === animLayer && l >= 1 && animProgress >= 1);

            ctx2.beginPath(); ctx2.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
            if (isInput) {
                ctx2.fillStyle = '#fffde6'; ctx2.fill();
                ctx2.strokeStyle = '#d4a017'; ctx2.lineWidth = 2; ctx2.stroke();
                ctx2.fillStyle = '#333'; ctx2.font = 'bold 12px Arial';
                ctx2.textAlign = 'center'; ctx2.textBaseline = 'middle';
                ctx2.fillText('x=' + raw[0][i].toFixed(2), pos.x, pos.y);
            } else if (isFullyDone || isJustDone) {
                ctx2.fillStyle = '#f0f8ff'; ctx2.fill();
                ctx2.strokeStyle = fn.color; ctx2.lineWidth = 2; ctx2.stroke();
                drawNodeBars(ctx2, pos, raw[l][i], activated[l][i], fn, fnKey, nodeR, 1);
            } else if (isAnimating) {
                const pulse = 1 + 0.06 * Math.sin(animProgress * Math.PI);
                const r = nodeR * pulse;
                ctx2.beginPath(); ctx2.arc(pos.x, pos.y, r, 0, Math.PI * 2);
                ctx2.fillStyle = '#f8f8ff'; ctx2.fill();
                ctx2.strokeStyle = fn.color; ctx2.lineWidth = 2; ctx2.stroke();
                // Glow ring
               ctx2.beginPath(); ctx2.arc(pos.x, pos.y, r + 4, 0, Math.PI * 2);
                ctx2.strokeStyle = fn.color; ctx2.lineWidth = 1.2;
                ctx2.globalAlpha = 0.2 * Math.sin(animProgress * Math.PI); ctx2.stroke(); ctx2.globalAlpha = 1;
                drawNodeBars(ctx2, pos, raw[l][i], activated[l][i], fn, fnKey, nodeR, animProgress);
            } else {
                ctx2.fillStyle = '#fafafa'; ctx2.fill();
                ctx2.strokeStyle = '#ddd'; ctx2.lineWidth = 1.5; ctx2.stroke();
                ctx2.fillStyle = '#bbb'; ctx2.font = '11px Arial';
                ctx2.textAlign = 'center'; ctx2.textBaseline = 'middle';
                const label = l === positions.length - 1 ? 'y' + (i + 1) : 'h' + l + '.' + (i + 1);
                ctx2.fillText(label, pos.x, pos.y);
            }
        }
    }

    // Layer labels
    const layerNames = ['输入层', '隐藏层1', '隐藏层2', '输出层'];
    ctx2.fillStyle = '#555'; ctx2.font = '12px Arial'; ctx2.textAlign = 'center';
    for (let l = 0; l < positions.length; l++) {
        ctx2.fillText(layerNames[l], positions[l][0].x, H - netMarginBottom + 20);
    }

    // Activation label on animating layer
    if (animLayer >= 1 && animLayer < actNet.length && animProgress < 1) {
        const lx = positions[animLayer][0].x;
        ctx2.font = 'bold 12px Arial'; ctx2.fillStyle = fn.color; ctx2.textAlign = 'center';
        ctx2.fillText('⚡ ' + fn.name + '(z) → a', lx, netMarginTop - 8);
    }

    // === Bottom legend ===
    const legendY = H - 18;
    ctx2.fillStyle = '#666'; ctx2.font = '11px Arial'; ctx2.textAlign = 'left';
    // z bar sample
    const lx = 50;
    ctx2.fillStyle = '#999'; ctx2.fillRect(lx, legendY - 4, 20, 7);
    ctx2.fillStyle = '#555'; ctx2.font = '11px Arial';
    ctx2.fillText('z值(激活前)', lx + 25, legendY + 2);
    // a bar sample
    const lx2 = 200;
    ctx2.fillStyle = fn.color; ctx2.fillRect(lx2, legendY - 4, 20, 7);
    ctx2.fillStyle = '#555';
    ctx2.fillText('a值(激活后)', lx2 + 25, legendY + 2);
    // Key insight
    ctx2.fillStyle = '#e63946'; ctx2.font = 'bold 11px Arial';
    const lx3 = 360;
    if (fnKey === 'relu') {
        ctx2.fillText('⚠ 负z → a=0 (神经元死亡)', lx3, legendY + 2);
    } else if (fnKey === 'sigmoid') {
        ctx2.fillText('⚠ 所有值压缩到(0,1)范围', lx3, legendY + 2);
    } else if (fnKey === 'tanh') {
        ctx2.fillText('⚠ 大|z|被压缩到±1(饱和区)', lx3, legendY + 2);
    } else {
        ctx2.fillText('⚠ 负值缩为0.1倍(微弱保留)', lx3, legendY + 2);
    }
}

function drawNodeBars(ctx2, pos, z, a, fn, fnKey, nodeR, progress) {
    const barW = 36, barH = 7;
    const barX = pos.x - barW / 2;

    // Unified normalization: use same scale for z and a
    // Scale based on max possible display value (use 2 as reference for good visual range)
    const scale = 2;
    const zNorm = Math.min(1, Math.abs(z) / scale);
    let aNorm = Math.min(1, Math.abs(a) / scale);

    // z value label (above z bar, inside circle)
    ctx2.font = '9px Arial'; ctx2.textAlign = 'center'; ctx2.textBaseline = 'bottom';
    ctx2.fillStyle = '#666';
    ctx2.fillText('z=' + z.toFixed(2), pos.x, pos.y - 14);

    // z bar (upper-middle area)
    const zBarY = pos.y - 7;
    ctx2.fillStyle = '#eaeaea'; ctx2.fillRect(barX, zBarY, barW, barH);
    ctx2.fillStyle = z >= 0 ? '#999' : '#c07070';
    ctx2.fillRect(barX, zBarY, barW * zNorm, barH);

    // a bar (lower-middle area) - animated
    const aBarY = pos.y + 2;
    const displayNorm = aNorm * progress;

    ctx2.fillStyle = '#eaeaea'; ctx2.fillRect(barX, aBarY, barW, barH);
    if (a === 0 && fnKey === 'relu') {
        ctx2.fillStyle = '#f5d0d0'; ctx2.fillRect(barX, aBarY, barW, barH);
    } else {
        ctx2.fillStyle = fn.color;
        ctx2.globalAlpha = 0.5 + 0.5 * progress;
        ctx2.fillRect(barX, aBarY, barW * displayNorm, barH);
        ctx2.globalAlpha = 1;
    }

    // a value label (below a bar, inside circle)
    if (progress > 0.4) {
        ctx2.font = '9px Arial'; ctx2.textAlign = 'center'; ctx2.textBaseline = 'top';
        ctx2.fillStyle = fn.color;
        ctx2.fillText('a=' + a.toFixed(2), pos.x, pos.y + 11);
    }

    // Special markers (below circle)
    if (fnKey === 'relu' && a === 0 && progress > 0.5) {
        ctx2.font = 'bold 9px Arial'; ctx2.fillStyle = '#e63946';
        ctx2.textAlign = 'center'; ctx2.textBaseline = 'top';
        ctx2.fillText('DEAD', pos.x, pos.y + nodeR + 4);
    } else if (fnKey === 'leakyrelu' && z < 0 && progress > 0.5) {
        ctx2.font = 'bold 9px Arial'; ctx2.fillStyle = '#e9c46a';
        ctx2.textAlign = 'center'; ctx2.textBaseline = 'top';
        ctx2.fillText('×0.1', pos.x, pos.y + nodeR + 4);
    }
}

function updateActInfoPanel(fnKey, raw, activated, animLayer, weights, biases) {
    const panel = document.getElementById('activationInfo');
    const fn = activationFns[fnKey];
    let html = '<h3>' + fn.name + '</h3>';
    html += '<div class="section"><div class="formula">' + fn.formula + '</div>';
    html += '<div style="font-size:12px;color:#666;">输出范围: ' + fn.range + '</div></div>';
    html += '<div class="section"><div class="section-title">优缺点</div>';
    html += '<div style="font-size:11px;"><b>优:</b> ' + fn.pros + '<br><b>缺:</b> ' + fn.cons + '</div></div>';
    html += '<div class="section"><div class="section-title">逐层计算过程</div>';
    for (let l = 1; l < actNet.length && l <= animLayer; l++) {
        const lName = l === actNet.length - 1 ? '输出层' : '隐藏层' + l;
        html += '<div style="margin:6px 0;padding:8px;background:#f8f9fa;border-radius:6px;border-left:3px solid ' + fn.color + ';font-size:11px;">';
        html += '<b>' + lName + '</b>';
        for (let j = 0; j < actNet[l]; j++) {
            html += '<div style="margin:4px 0 2px;padding:4px;background:#fff;border-radius:3px;">';
            html += '<b>节点' + (j + 1) + ':</b> ';
            var terms = [];
            for (var i = 0; i < actNet[l - 1]; i++) {
                var w = weights[l - 1][j][i];
                var a = activated[l - 1][i];
                terms.push(w.toFixed(2) + '×' + a.toFixed(2));
            }
            html += '<span style="color:#666;font-size:10px;">' + terms.join(' + ') + ' + b(' + biases[l - 1][j].toFixed(2) + ')</span><br>';
            html += 'z = <b>' + raw[l][j].toFixed(3) + '</b>';
            html += ' → <span style="color:' + fn.color + ';">' + fn.name + '(z)</span> = <b style="color:' + fn.color + ';">' + activated[l][j].toFixed(3) + '</b>';
            if (fnKey === 'relu' && raw[l][j] < 0) {
                html += ' <span style="color:#e63946;font-size:9px;">⚠DEAD</span>';
            }
            html += '</div>';
        }
        html += '</div>';
    }
    html += '</div>';
    panel.innerHTML = html;
}

function actInitCanvas() {
    actFnKey = document.getElementById('activationSelect').value;
    actData = computeActValues(actFnKey);
    var canvas2 = document.getElementById('activationCanvas');
    canvas2.width = 850; canvas2.height = 560;
    actCtx = canvas2.getContext('2d');
}

function actRender() {
    if (!actCtx || !actData) actInitCanvas();
    drawActivationPage(actCtx, 850, 560, actFnKey, actData.raw, actData.activated, actCurrentLayer, actProgress >= 1 ? 1 : actProgress);
    updateActInfoPanel(actFnKey, actData.raw, actData.activated, actCurrentLayer, actData.weights, actData.biases);
}

function renderActivationPage() {
    stopActivationAnim();
    actCurrentLayer = 0;
   actProgress = 1;
    actInitCanvas();
    actRender();
}

function actAnimStep() {
    if (!actIsPlaying) return;
    actProgress += 0.016;
    if (actProgress >= 1) {
        actProgress = 1;
        actRender();
        actCurrentLayer++;
        if (actCurrentLayer >= actNet.length) {
            actIsPlaying = false;
            var btn = document.getElementById('actPlayBtn');
            if (btn) btn.textContent = '播放';
            actAnimId = null;
            return;
        }
        actProgress = 0;
        actAnimId = setTimeout(function() { actAnimId = requestAnimationFrame(actAnimStep);}, 500);
        return;
    }
    actRender();
    actAnimId = requestAnimationFrame(actAnimStep);
}

function startActivationAnim() {
    stopActivationAnim();
    actCurrentLayer = 1;
    actProgress = 0;
    actIsPlaying = true;
    actInitCanvas();
    actRender();
 var btn = document.getElementById('actPlayBtn');
    if (btn) btn.textContent = '暂停';
    setTimeout(function() { actAnimId = requestAnimationFrame(actAnimStep); }, 500);
}

function stopActivationAnim() {
    actIsPlaying = false;
    if (actAnimId) {
        cancelAnimationFrame(actAnimId);
        clearTimeout(actAnimId);
        actAnimId = null;
    }
}

function toggleActivationAnim() {
    var btn = document.getElementById('actPlayBtn');
    if (actIsPlaying) {
        stopActivationAnim();
        if (btn) btn.textContent = '播放';
    } else {
        if (actCurrentLayer === 0 || actCurrentLayer >= actNet.length) {
            // start from beginning
            startActivationAnim();
        } else {
            // resume
            actIsPlaying = true;
            if (btn) btn.textContent = '暂停';
            actAnimId = requestAnimationFrame(actAnimStep);
        }
    }
}

function nextActivationStep() {
    stopActivationAnim();
    var btn = document.getElementById('actPlayBtn');
    if (btn) btn.textContent = '播放';
    if (actCurrentLayer === 0) {
        actInitCanvas();
        actCurrentLayer = 1;
    } else {
        actCurrentLayer++;
   }
    if (actCurrentLayer >= actNet.length) {
        actCurrentLayer = actNet.length - 1;
    }
    actProgress = 1;
    actRender();
}

function resetActivationAnim() {
    stopActivationAnim();
    actCurrentLayer = 0;
    actProgress = 1;
    actInitCanvas();
    actRender();
    var btn = document.getElementById('actPlayBtn');
    if (btn) btn.textContent = '播放';
    document.getElementById('activationInfo').innerHTML = '<h3>激活函数可视化</h3><div class="section"><div style="color:#666;font-size:12px;">选择一个激活函数，点击<b>"播放"</b>观看逐层激活过程。<br><br><b>核心观察点：</b><br>• 节点内<b>灰色条</b> = z值(加权和，激活前)<br>• 节点内<b>彩色条</b> = a值(激活后)<br>• 对比两条的<b>长度差异</b> = 激活函数变换效果<br>• ReLU: 负值归零(DEAD标记)<br>• Sigmoid: 全部压缩到0~1</div></div>';
}

function renderActivation() {
    renderActivationPage();
}
