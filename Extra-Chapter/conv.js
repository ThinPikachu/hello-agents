// ========== CNN卷积运算可视化页面（Conv → ReLU → Pooling 一条龙演示） ==========

let convAnimId = null;
let convStep = -1;
let convIsPlaying = false;

// === 阶段定义 ===
// Phase 1: Convolution (16 steps: fill 4x4 output one by one)
// Phase 2: ReLU activation (1 step: show transformation)
// Phase 3: Max Pooling (4 steps: fill 2x2 output one by one)
const CONV_STEPS = 16;
const RELU_STEPS = 1;
const POOL_STEPS = 4;
const TOTAL_STEPS = CONV_STEPS + RELU_STEPS + POOL_STEPS; // 21

function getPhase(step) {
    if (step < 0) return 'idle';
    if (step < CONV_STEPS) return 'conv';
    if (step < CONV_STEPS + RELU_STEPS) return 'relu';
    return 'pool';
}

function renderConvPage() {
    const canvas = document.getElementById('convCanvas');
    const dpr = window.devicePixelRatio || 1;
    const logicalW = 1050;
    const logicalH = 560;
    canvas.width = logicalW * dpr;
    canvas.height = logicalH * dpr;
    canvas.style.width = logicalW + 'px';
    canvas.style.height = logicalH + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    convStep = -1;
    convIsPlaying = false;
    drawPipeline(ctx, logicalW, logicalH, convStep);
    updateConvInfoPanel();
}

// Input matrix (6x6)
const convInput = [
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1]
];

// Kernel (3x3) - vertical edge detector
const convKernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
];

// Compute convolution output (4x4)
function computeConvOutput() {
    const output = [];
    for (let i = 0; i <= 3; i++) {
        output[i] = [];
        for (let j = 0; j <= 3; j++) {
            let sum = 0;
            for (let ki = 0; ki < 3; ki++) {
                for (let kj = 0; kj < 3; kj++) {
                    sum += convInput[i + ki][j + kj] * convKernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}
const convOutput = computeConvOutput();

// ReLU output
function computeReluOutput() {
    const relu = [];
    for (let i = 0; i < 4; i++) {
        relu[i] = [];
        for (let j = 0; j < 4; j++) {
            relu[i][j] = Math.max(0, convOutput[i][j]);
        }
    }
    return relu;
}
const reluOutput = computeReluOutput();

// MaxPooling output (2x2 window, stride 2, on 4x4 → 2x2)
function computePoolOutput() {
    const pool = [];
    for (let i = 0; i < 2; i++) {
        pool[i] = [];
        for (let j = 0; j < 2; j++) {
            const vals = [
                reluOutput[i * 2][j * 2], reluOutput[i * 2][j * 2 + 1],
                reluOutput[i * 2 + 1][j * 2], reluOutput[i * 2 + 1][j * 2 + 1]
            ];
            pool[i][j] = Math.max(...vals);
        }
    }
    return pool;
}
const poolOutput = computePoolOutput();

// ========== Main draw function: all stages on one canvas ==========
function drawPipeline(ctx, W, H, step) {
    ctx.clearRect(0, 0, W, H);
    const phase = getPhase(step);

    // Title
    ctx.fillStyle = '#333'; ctx.font = 'bold 15px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Conv → ReLU → MaxPooling 一条龙演示', W / 2, 22);

    // Phase progress bar
    drawPhaseBar(ctx, W, phase, step);

    // Layout: all matrices in one row
    // Input(6x6) → [Kernel(3x3)] → ConvOut(4x4) → ReluOut(4x4) → PoolOut(2x2)
    const cs = 28; // cell size for 6x6 and 4x4
    const topY = 80;

    // --- Input 6x6 ---
    const inX = 15;
    drawLabel(ctx, 'Input 6×6', inX + 3 * cs, topY - 10);
    const convRow = (step >= 0 && step < CONV_STEPS) ? Math.floor(step / 4) : -1;
    const convCol = (step >= 0 && step < CONV_STEPS) ? step % 4 : -1;

    for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
            const x = inX + j * cs, y = topY + i * cs;
            const inKernel = convRow >= 0 && i >= convRow && i < convRow + 3 && j >= convCol && j < convCol + 3;
            ctx.fillStyle = inKernel ? '#fff3cd' : '#f8f9fa';
            ctx.fillRect(x, y, cs, cs);
            ctx.strokeStyle = '#bbb'; ctx.lineWidth = 0.8;
            ctx.strokeRect(x, y, cs, cs);
         ctx.fillStyle = '#333'; ctx.font = '11px Arial';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(convInput[i][j], x + cs / 2, y + cs / 2);
        }
    }
    if (convRow >= 0) {
        ctx.strokeStyle = '#e63946'; ctx.lineWidth = 2.5;
        ctx.strokeRect(inX + convCol * cs, topY + convRow * cs, 3 * cs, 3 * cs);
    }

    // --- Kernel 3x3 (small, below input) ---
    const kX = inX + 20, kY = topY + 6 * cs + 15;
    const kcs = 24;
    drawLabel(ctx, 'Kernel 3×3', kX + 1.5 * kcs, kY - 8);
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const x = kX + j * kcs, y = kY + i * kcs;
            ctx.fillStyle = '#d4edda';
            ctx.fillRect(x, y, kcs, kcs);
            ctx.strokeStyle = '#28a745'; ctx.lineWidth = 1;
            ctx.strokeRect(x, y, kcs, kcs);
            ctx.fillStyle = '#155724'; ctx.font = 'bold 11px Arial';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(convKernel[i][j], x + kcs / 2, y + kcs / 2);
        }
    }

    // --- Arrow 1: Input → Conv Output ---
    const arrow1X = inX + 6 * cs + 8;
    drawArrow(ctx, arrow1X, topY + 3 * cs, arrow1X + 30, topY + 3 * cs, '#666');
    ctx.fillStyle = '#666'; ctx.font = '16px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('*', arrow1X + 15, topY + 3 * cs - 14);

    // --- Conv Output 4x4 ---
    const coX = arrow1X + 35;
    drawLabel(ctx, 'Conv 4×4', coX + 2 * cs, topY - 10);
    const convDone = step >= CONV_STEPS; // all conv steps done

    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const x = coX + j * cs, y = topY + i * cs + cs; // offset slightly
            const idx = i * 4 + j;
            const filled = convDone || (step >= 0 && idx <= step && step < CONV_STEPS);
            const allFilled = convDone || (step >= CONV_STEPS);
            let bg = '#f8f9fa';
            if (allFilled) {
                bg = convOutput[i][j] < 0 ? '#ffebee' : '#e3f2fd';
            } else if (step >= 0 && step < CONV_STEPS) {
                if (idx < step) bg = '#cce5ff';
                else if (idx === step) bg = '#ffc107';
            }
            ctx.fillStyle = bg;
            ctx.fillRect(x, y, cs, cs);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 0.8;
            ctx.strokeRect(x, y, cs, cs);
            if (allFilled || (step >= 0 && idx <= step && step < CONV_STEPS)) {
                ctx.fillStyle = convOutput[i][j] < 0 ? '#c62828' : '#333';
                ctx.font = 'bold 11px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(convOutput[i][j], x + cs / 2, y + cs / 2);
            }
        }
    }

    // --- Arrow 2: Conv → ReLU ---
    const arrow2X = coX + 4 * cs + 8;
    drawArrow(ctx, arrow2X, topY + 3 * cs, arrow2X + 30, topY + 3 * cs, '#1976d2');
    ctx.fillStyle = '#1976d2'; ctx.font = 'bold 10px Arial'; ctx.textAlign = 'center';
    ctx.fillText('ReLU', arrow2X + 15, topY + 3 * cs - 12);

    // --- ReLU Output 4x4 ---
    const rlX = arrow2X + 35;
    drawLabel(ctx, 'ReLU 4×4', rlX + 2 * cs, topY - 10);
    const reluDone = step >= CONV_STEPS + RELU_STEPS;

    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const x = rlX + j * cs, y = topY + i * cs + cs;
            const showRelu = reluDone || (phase === 'relu');
            let bg = '#f8f9fa';
            if (showRelu) {
                const changed = convOutput[i][j] < 0;
                bg = changed ? '#c8e6c9' : '#f8f9fa';
            }
            ctx.fillStyle = bg;
            ctx.fillRect(x, y, cs, cs);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 0.8;
            ctx.strokeRect(x, y, cs, cs);
            if (showRelu) {
                const changed = convOutput[i][j] < 0;
                ctx.fillStyle = changed ? '#2e7d32' : '#333';
                ctx.font = 'bold 11px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(reluOutput[i][j], x + cs / 2, y + cs / 2);
            }
        }
    }

    // --- Arrow 3: ReLU → Pool ---
    const arrow3X = rlX + 4 * cs + 8;
    drawArrow(ctx, arrow3X, topY + 3 * cs, arrow3X + 30, topY + 3 * cs, '#ff6f00');
    ctx.fillStyle = '#ff6f00'; ctx.font = 'bold 10px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Pool', arrow3X + 15, topY + 3 * cs - 12);

    // --- Pool Output 2x2 ---
    const plX = arrow3X + 35;
    const pcs = 42; // bigger cells for pool output
    drawLabel(ctx, 'Pool 2×2', plX + pcs, topY - 10);
    const poolLocalStep = step - CONV_STEPS - RELU_STEPS;

    // Highlight pooling window on ReLU output
    if (phase === 'pool') {
        const pRow = Math.floor(poolLocalStep / 2);
        const pCol = poolLocalStep % 2;
        ctx.strokeStyle = '#ff6f00'; ctx.lineWidth = 2.5;
        ctx.strokeRect(rlX + pCol * 2 * cs, topY + cs + pRow * 2 * cs, 2 * cs, 2 * cs);
    }

    for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
            const x = plX + j * pcs, y = topY + cs + i * pcs;
            const idx = i * 2 + j;
            let bg = '#f8f9fa';
            if (phase === 'pool' && idx < poolLocalStep) bg = '#c8e6c9';
            else if (phase === 'pool' && idx === poolLocalStep) bg = '#fff176';
            ctx.fillStyle = bg;
            ctx.fillRect(x, y, pcs, pcs);
            ctx.strokeStyle = '#6c757d'; ctx.lineWidth = 1;
            ctx.strokeRect(x, y, pcs, pcs);
            if (phase === 'pool' && idx <= poolLocalStep) {
                ctx.fillStyle = '#333'; ctx.font = 'bold 14px Arial';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(poolOutput[i][j], x + pcs / 2, y + pcs / 2);
            }
        }
    }

    // --- Bottom detail panel ---
    const detY = topY + 6 * cs + 15;
    ctx.fillStyle = '#333'; ctx.font = '12px Arial'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';

    if (phase === 'idle') {
        ctx.fillStyle = '#999';
        ctx.fillText('点击"播放动画"或"下一步"开始演示完整流程', coX, detY);
    } else if (phase === 'conv') {
        const row = Math.floor(step / 4), col = step % 4;
        ctx.fillText(`卷积 Step ${step + 1}/16: 窗口位置(${row},${col})`, coX, detY);
        let parts = [];
        let sum = 0;
        for (let ki = 0; ki < 3; ki++) {
            for (let kj = 0; kj < 3; kj++) {
                const v = convInput[row + ki][col + kj] * convKernel[ki][kj];
                sum += v;
                parts.push(`${convInput[row + ki][col + kj]}×(${convKernel[ki][kj]})`);
            }
        }
        ctx.font = '10px Courier New';
        const line = parts.join(' + ');
        if (line.length > 60) {
            ctx.fillText(parts.slice(0, 5).join(' + ') + ' +', coX, detY + 18);
            ctx.fillText(parts.slice(5).join(' + '), coX, detY + 32);
        } else {
            ctx.fillText(line, coX, detY + 18);
        }
        ctx.fillStyle = '#e63946'; ctx.font = 'bold 12px Arial';
        ctx.fillText(`= ${sum}`, coX, detY + 48);
    } else if (phase === 'relu') {
        ctx.fillText('ReLU 激活: f(x) = max(0, x)  — 负值→0，正值不变', coX, detY);
        ctx.fillStyle = '#e53935'; ctx.font = '11px Arial';
        ctx.fillText('红底=负值被清零    绿底=ReLU修改后的0', coX, detY + 20);
    } else if (phase === 'pool') {
        const pRow = Math.floor(poolLocalStep / 2), pCol = poolLocalStep % 2;
        ctx.fillText(`MaxPooling Step ${poolLocalStep + 1}/4: 2×2窗口位置(${pRow * 2},${pCol * 2})`, coX, detY);
        const wv = [
            reluOutput[pRow * 2][pCol * 2], reluOutput[pRow * 2][pCol * 2 + 1],
            reluOutput[pRow * 2 + 1][pCol * 2], reluOutput[pRow * 2 + 1][pCol * 2 + 1]
        ];
        ctx.font = '11px Courier New';
        ctx.fillText(`max(${wv.join(', ')}) = ${poolOutput[pRow][pCol]}`, coX, detY + 20);
    }

    // Bottom summary
    ctx.fillStyle = '#888'; ctx.font = '11px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('6×6 Input → 3×3 Conv(stride1) → 4×4 → ReLU → 4×4 → MaxPool(2×2,stride2) → 2×2', W / 2, H - 8);
}

// ========== Helpers ==========
function drawLabel(ctx, text, x, y) {
    ctx.fillStyle = '#333'; ctx.font = 'bold 11px Arial';
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(text, x, y);
}

function drawArrow(ctx, x1, y1, x2, y2, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 7 * Math.cos(angle - 0.4), y2 - 7 * Math.sin(angle - 0.4));
    ctx.lineTo(x2 - 7 * Math.cos(angle + 0.4), y2 - 7 * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
}

function drawPhaseBar(ctx, W, phase, step) {
    const barY = 40;
    const barW = 380;
    const barX = (W - barW) / 2;
    const phases = ['conv', 'relu', 'pool'];
    const labels = ['① Convolution (16步)', '② ReLU (1步)', '③ MaxPool (4步)'];
    const widths = [170, 100, 110];
    let x = barX;

    for (let i = 0; i < 3; i++) {
        const w = widths[i];
        const isActive = phases[i] === phase;
        const isDone = (phase === 'relu' && i === 0) || (phase === 'pool' && i <= 1);

        ctx.fillStyle = isDone ? '#c8e6c9' : (isActive ? '#bbdefb' : '#f0f0f0');
        ctx.fillRect(x, barY, w, 20);
        ctx.strokeStyle = isDone ? '#4caf50' : (isActive ? '#1976d2' : '#ccc');
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.strokeRect(x, barY, w, 20);

        ctx.fillStyle = isDone ? '#2e7d32' : (isActive ? '#1565c0' : '#888');
        ctx.font = isActive ? 'bold 10px Arial' : '10px Arial';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(labels[i], x + w / 2, barY + 10);

        if (i < 2) {
            ctx.fillStyle = '#aaa'; ctx.font = '12px Arial';
            ctx.fillText('→', x + w + 3, barY + 10);
        }
        x += w + 12;
    }
}

// ========== Animation Controls ==========
function toggleConvAnim() {
    if (convIsPlaying) {
        stopConvAnim();
        document.getElementById('convPlayBtn').textContent = '播放';
    } else {
        startConvAnim();
        document.getElementById('convPlayBtn').textContent = '暂停';
    }
}

function startConvAnim() {
    if (convIsPlaying) return;
    convIsPlaying = true;
    if (convStep < 0) convStep = 0;
    convAnimStep();
}

function convAnimStep() {
    if (!convIsPlaying) return;
    const canvas = document.getElementById('convCanvas');
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawPipeline(ctx, 1050, 560, convStep);
    convStep++;
    if (convStep >= TOTAL_STEPS) {
        convStep = 0;
    }
    // Pause longer between phases
    const curPhase = getPhase(convStep);
    const prevPhase = getPhase(convStep - 1);
    const delay = (curPhase !== prevPhase && convStep > 0) ? 1500 : 700;
    convAnimId = setTimeout(convAnimStep, delay);
}

function stopConvAnim() {
    convIsPlaying = false;
    if (convAnimId) { clearTimeout(convAnimId); convAnimId = null; }
}

function resetConvAnim() {
    stopConvAnim();
    convStep = -1;
    document.getElementById('convPlayBtn').textContent = '播放';
    const canvas = document.getElementById('convCanvas');
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawPipeline(ctx, 1050, 560, -1);
}

function nextConvStep() {
    stopConvAnim();
    document.getElementById('convPlayBtn').textContent = '播放';
    convStep++;
    if (convStep >= TOTAL_STEPS) convStep = 0;
    const canvas = document.getElementById('convCanvas');
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawPipeline(ctx, 1050, 560, convStep);
}

function updateConvInfoPanel() {
    const panel = document.getElementById('convInfo');
    let html = '<h3>Conv → ReLU → Pooling</h3>';

    html += '<div class="section"><div class="section-title">完整流程</div>';
    html += '<div class="formula">Input → Conv → ReLU → Pool</div>';
    html += '<div style="font-size:11px;">6×6 → 4×4 → 4×4 → 2×2</div></div>';

    html += '<div class="section"><div class="section-title">① 卷积 Convolution</div>';
    html += '<div style="font-size:11px;">';
    html += '3×3核滑动，逐位相乘求和<br>';
    html += 'Stride=1, Pad=0<br>';
    html += 'Output = (6-3)/1+1 = 4×4</div></div>';

    html += '<div class="section"><div class="section-title">② 激活 ReLU</div>';
    html += '<div style="font-size:11px;">';
    html += '<div class="formula">f(x) = max(0, x)</div>';
    html += '引入非线性，负值清零</div></div>';

    html += '<div class="section"><div class="section-title">③ 池化 MaxPooling</div>';
    html += '<div style="font-size:11px;">';
    html += '2×2窗口取max, stride=2<br>';
    html += 'Output = 4/2 = 2×2<br>';
    html += '降维 + 平移不变性</div></div>';

    html += '<div class="section"><div class="section-title">各阶段作用</div>';
    html += '<div style="font-size:11px;">';
    html += '• <b>Conv：</b>提取特征（边缘等）<br>';
    html += '• <b>ReLU：</b>非线性，增强表达<br>';
    html += '• <b>Pool：</b>降维，保留最强响应<br>';
    html += '• <b>组合：</b>逐层抽象更高级特征</div></div>';

    html += '<div class="section" style="background:#e8f5e9;padding:10px;border-radius:6px;">';
    html += '<div class="section-title" style="color:#2e7d32;">核心要点</div>';
    html += '<div style="font-size:11px;">';
    html += '• 卷积核=垂直边缘检测器<br>';
    html += '• ReLU只保留正响应<br>';
    html += '• Pool保留最大激活值<br>';
    html += '• 一条龙完成特征提取</div></div>';

    panel.innerHTML = html;
}