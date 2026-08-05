// ========== CNN基础结构页面 - 卷积神经网络架构可视化 ==========

function renderCNNPage() {
    const canvas = document.getElementById('cnnCanvas');
    const dpr = window.devicePixelRatio || 1;
    const logicalW = 1050;
    const logicalH = 520;
    canvas.width = logicalW * dpr;
    canvas.height = logicalH * dpr;
    canvas.style.width = logicalW + 'px';
    canvas.style.height = logicalH + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    drawCNNArchitecture(ctx, logicalW, logicalH);
    updateCNNInfoPanel();
}

function drawCNNArchitecture(ctx, W, H) {
    ctx.clearRect(0, 0, W, H);

    // Title
    ctx.fillStyle = '#333'; ctx.font = 'bold 16px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Convolution Neural Network (CNN)', W / 2, 28);

    // Layout parameters
    const baseY = H / 2 - 10;
    const stageGap = 18;

    // === Stage positions (x centers) ===
    const stages = [
        { x: 55, label: 'Input' },            // 0: input image
        { x: 160, label: 'Conv+ReLU' },        // 1: conv1
        { x: 245, label: 'Conv+ReLU' },        // 2: conv2
        { x: 340, label: 'Conv+ReLU' },        // 3: conv3
        { x: 440, label: 'Flatten' },          // 4: flatten (1D array)
        { x: 590, label: 'FC Layer' },         // 5: fully connected (includes input col)
        { x: 790, label: 'Output' }            // 6: output + softmax
    ];

    // === 1. Input Image ===
    drawInputImage(ctx, stages[0].x, baseY);

    // === 2. Kernel indicator ===
    ctx.strokeStyle = '#d4a017'; ctx.lineWidth = 1.5;
    ctx.strokeRect(stages[0].x - 8, baseY + 10, 14, 14);
    ctx.beginPath();
    ctx.moveTo(stages[0].x - 1, baseY + 24);
    ctx.lineTo(stages[0].x - 1, baseY + 85);
    ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Kernel', stages[0].x - 1, baseY + 100);

    // Unified label Y position for all stage labels
    const labelY1 = baseY + 100;
    const labelY2 = labelY1 + 13;

    // === 3. Conv + Poolingblocks ===
    // Block 1: larger feature maps
    drawFeatureMapStack(ctx, stages[1].x, baseY, 70, 70, 4, '#c9b896');
    drawPoolingLabel(ctx, stages[1].x + 50, baseY - 60, 'Pooling');
    drawConvLabelAt(ctx, stages[1].x, labelY1, labelY2);

    // Block 2: medium feature maps
    drawFeatureMapStack(ctx, stages[2].x, baseY, 55, 55, 6, '#b8a886');
    drawPoolingLabel(ctx, stages[2].x + 45, baseY - 50, 'Pooling');
    drawConvLabelAt(ctx, stages[2].x, labelY1, labelY2);

    // Block 3: smaller feature maps
    drawFeatureMapStack(ctx, stages[3].x, baseY, 42, 42, 10, '#a89876');
    drawConvLabelAt(ctx, stages[3].x, labelY1, labelY2);

    // Kernel markers on feature maps
    ctx.strokeStyle = '#d4a017'; ctx.lineWidth = 1.2;
    ctx.strokeRect(stages[1].x + 10, baseY + 5, 10, 10);
    ctx.strokeRect(stages[2].x + 5, baseY + 5, 8, 8);
    ctx.strokeRect(stages[3].x + 3, baseY + 3, 7, 7);

    // === 4. Flatten Layer ===
    drawFlattenLayer(ctx, stages[4].x, baseY);

    // === 5. Fully Connected Layer (includes input column) ===
    drawFCLayer(ctx, stages[5].x, baseY);

    // === 5.5 One-to-one connections from Flatten to FC input column ===
    drawFlattenToFCConnections(ctx, stages[4].x, stages[5].x, baseY);

    // === 6. Output ===
    drawOutputLayer(ctx, stages[6].x, baseY);

    // === Connecting arrows between stages ===
    drawArrow(ctx, stages[0].x + 35, baseY, stages[1].x - 40, baseY);
    drawArrow(ctx, stages[1].x + 45, baseY, stages[2].x - 35, baseY);
    drawArrow(ctx, stages[2].x + 40, baseY, stages[3].x - 30, baseY);
    drawArrow(ctx, stages[3].x + 50, baseY, stages[4].x - 15, baseY);
    drawArrow(ctx, stages[5].x + 75, baseY, stages[6].x - 55, baseY);

    // === Bottom annotations ===
    const bottomY = H - 55;
    ctx.font = '12px Arial'; ctx.textAlign = 'center';

    // Feature Extraction bracket
    ctx.strokeStyle = '#666'; ctx.lineWidth = 1;
    const feStart = stages[0].x - 30, feEnd = stages[3].x + 60;
    ctx.beginPath();
    ctx.moveTo(feStart, bottomY - 20); ctx.lineTo(feStart, bottomY - 15);
    ctx.lineTo(feEnd, bottomY - 15); ctx.lineTo(feEnd, bottomY - 20);
    ctx.stroke();
    ctx.beginPath(); ctx.moveTo((feStart + feEnd) / 2, bottomY - 15);
    ctx.lineTo((feStart + feEnd) / 2, bottomY - 8); ctx.stroke();

    ctx.fillStyle = '#2a9d8f'; ctx.font = 'bold 11px Arial';
    ctx.fillText('卷积 + 激活 + 池化', (feStart + feEnd) / 2, bottomY + 2);
    ctx.fillStyle = '#333'; ctx.font = '12px Arial';
    ctx.fillText('Feature Extraction', (feStart + feEnd) / 2, bottomY + 18);
    ctx.fillStyle = '#e63946'; ctx.font = '10px Arial';
    ctx.fillText('特征提取，用作传统神经网络的输入', (feStart + feEnd) / 2, bottomY + 33);

    // Classification bracket
    const clStart = stages[5].x - 50, clEnd = stages[5].x + 80;
    ctx.strokeStyle = '#666'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(clStart, bottomY - 20); ctx.lineTo(clStart, bottomY - 15);
    ctx.lineTo(clEnd, bottomY - 15); ctx.lineTo(clEnd, bottomY - 20);
    ctx.stroke();
    ctx.beginPath(); ctx.moveTo((clStart + clEnd) / 2, bottomY - 15);
    ctx.lineTo((clStart + clEnd) / 2, bottomY - 8); ctx.stroke();

    ctx.fillStyle = '#457b9d'; ctx.font = 'bold 11px Arial';
    ctx.fillText('神经网络', (clStart + clEnd) / 2, bottomY + 2);
    ctx.fillStyle = '#333'; ctx.font = '12px Arial';
    ctx.fillText('Classification', (clStart + clEnd) / 2, bottomY + 18);

    // Probabilistic Distribution label
    ctx.fillStyle = '#333'; ctx.font = '11px Arial';
    ctx.fillText('Probabilistic', stages[6].x, bottomY + 10);
    ctx.fillText('Distribution', stages[6].x, bottomY + 24);

    // Feature Maps label
    ctx.fillStyle = '#666'; ctx.font = '11px Arial';
    const fmMid = (stages[1].x + stages[3].x) / 2;
    ctx.beginPath();
    ctx.moveTo(stages[1].x - 30, bottomY - 35);
    ctx.lineTo(stages[3].x + 50, bottomY - 35);
    ctx.strokeStyle = '#999'; ctx.lineWidth = 0.5; ctx.stroke();
    // arrows on both ends
    ctx.beginPath(); ctx.moveTo(stages[1].x - 30, bottomY - 35);
    ctx.lineTo(stages[1].x - 24, bottomY - 38); ctx.lineTo(stages[1].x - 24, bottomY - 32);
    ctx.closePath(); ctx.fillStyle = '#999'; ctx.fill();
    ctx.beginPath(); ctx.moveTo(stages[3].x + 50, bottomY - 35);
    ctx.lineTo(stages[3].x + 44, bottomY - 38); ctx.lineTo(stages[3].x + 44, bottomY - 32);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#666'; ctx.font = '12px Arial';
    ctx.fillText('Feature Maps', fmMid, bottomY - 40);
}

// Draw input image (simplified zebra-like pattern)
function drawInputImage(ctx, cx, cy) {
    const size = 65;
    const x = cx - size / 2, y = cy - size / 2;

    // Image background
    ctx.fillStyle = '#e8dcc8'; ctx.fillRect(x, y, size, size);
    ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.strokeRect(x, y, size, size);

    // Zebra-like stripes
    ctx.strokeStyle = '#333'; ctx.lineWidth = 2;
    for (let i = 0; i < 6; i++) {
        ctx.beginPath();
        const sx = x + 10 + i * 8;
        ctx.moveTo(sx, y + 15);
        ctx.quadraticCurveTo(sx + 4, cy, sx - 2, y + size - 12);
        ctx.stroke();
    }

    // Label
    ctx.fillStyle = '#333'; ctx.font = '12px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Input', cx, cy - size / 2 - 10);
}

// Draw a stack of feature maps (3D effect)
function drawFeatureMapStack(ctx, cx, cy, w, h, depth, color) {
    const offsetX = 3, offsetY = -3;
    // Draw from back to front
    for (let d = depth - 1; d >= 0; d--) {
        const dx = cx - w / 2 + d * offsetX;
        const dy = cy - h / 2 + d * offsetY;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.6 + 0.4 * (d / depth);
        ctx.fillRect(dx, dy, w, h);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#888'; ctx.lineWidth = 0.8;
        ctx.strokeRect(dx, dy, w, h);
    }
}

// Draw "Convolution + ReLU" label below feature maps
function drawConvLabel(ctx, cx, y) {
    ctx.fillStyle = '#444'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Convolution', cx, y + 10);
    ctx.fillText('+ ReLU', cx, y + 22);
}

// Draw "Convolution + ReLU" label at fixed Y positions (for alignment)
function drawConvLabelAt(ctx, cx, y1, y2) {
    ctx.fillStyle = '#444'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Convolution', cx, y1);
    ctx.fillText('+ ReLU', cx, y2);
}

// Draw "Pooling" label above
function drawPoolingLabel(ctx, x, y, text) {
    ctx.fillStyle = '#555'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
    ctx.fillText(text, x, y);
}

// Draw Flatten layer (1D vector - vertical strip of cells)
function drawFlattenLayer(ctx, cx, cy) {
    const cellCount = 10;
    const cellW = 12, cellH = 14;
    const totalH = cellCount * cellH;
    const startY = cy - totalH / 2;

    // Draw vertical strip of cells (1D array representation)
    for (let i = 0; i < cellCount; i++) {
        const ny = startY + i * cellH;
        // Alternate slightly different shades to show array structure
        const shade = (i % 2 === 0) ? '#a8d8ea' : '#86c5da';
        ctx.fillStyle = shade;
        ctx.fillRect(cx - cellW / 2, ny, cellW, cellH);
        ctx.strokeStyle = '#5dade2'; ctx.lineWidth = 0.6;
        ctx.strokeRect(cx - cellW / 2, ny, cellW, cellH);
    }

    // Outer border for the whole vector
    ctx.strokeStyle = '#2980b9'; ctx.lineWidth = 1.2;
    ctx.strokeRect(cx - cellW / 2, startY, cellW, totalH);

    ctx.fillStyle = '#444'; ctx.font = '11px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Flatten', cx, cy + totalH / 2 + 30);
    ctx.fillText('Layer', cx, cy + totalH / 2 + 43);
}

// Draw FC layer (4 columns including input layer, funnel shape)
function drawFCLayer(ctx, cx, cy) {
    const cols = [
        { x: cx - 55, count: 10, r: 3.5, totalH: 140 },  // input layer
        { x: cx - 15, count: 8, r: 4, totalH: 160 },
        { x: cx + 25, count: 6, r: 4.5, totalH: 120 },
        { x: cx + 60, count: 5, r: 5, totalH: 85 }
    ];

    // Draw connections between FC columns
    for (let c = 0; c < cols.length - 1; c++) {
        const from = cols[c], to = cols[c + 1];
        const fromSpacing = from.totalH / (from.count - 1);
        const toSpacing = to.totalH / (to.count - 1);
        for (let i = 0; i < from.count; i++) {
            for (let j = 0; j < to.count; j++) {
                const fx = from.x, fy = cy - from.totalH / 2 + i * fromSpacing;
                const tx = to.x, ty = cy - to.totalH / 2 + j * toSpacing;
                ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty);
                ctx.strokeStyle = 'rgba(0,0,0,0.1)'; ctx.lineWidth = 0.4; ctx.stroke();
            }
        }
    }

    // Draw nodes
    for (let c = 0; c < cols.length; c++) {
        const col = cols[c];
        const spacing = col.totalH / (col.count - 1);
        for (let i = 0; i < col.count; i++) {
            const ny = cy - col.totalH / 2 + i * spacing;
            ctx.beginPath(); ctx.arc(col.x, ny, col.r, 0, Math.PI * 2);
            ctx.fillStyle = '#5dade2'; ctx.fill();
            ctx.strokeStyle = '#2980b9'; ctx.lineWidth = 0.8; ctx.stroke();
        }
    }

    ctx.fillStyle = '#444'; ctx.font = '11px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Fully Connected', cx, cy + cols[0].totalH / 2 + 30);
    ctx.fillText('Layer', cx, cy + cols[0].totalH / 2 + 43);
}

// Draw one-to-one connections from Flatten to FC first column (input layer)
function drawFlattenToFCConnections(ctx, flattenCx, fcCx, cy) {
    const count = 10;
    const cellH = 14, cellW = 12;
    const flattenTotalH = count * cellH;
    const fcTotalH = 140; // matches FC first column
    const fcSpacing = fcTotalH / (count - 1);

    for (let i = 0; i < count; i++) {
        const fx = flattenCx + cellW / 2;
        const fy = cy - flattenTotalH / 2 + i * cellH + cellH / 2;
        const tx = fcCx - 55 - 3.5; // left edge of first column circle
        const ty = cy - fcTotalH / 2 + i * fcSpacing;
        ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty);
        ctx.strokeStyle = 'rgba(80,80,80,0.3)'; ctx.lineWidth = 0.8; ctx.stroke();
    }
}

// Draw output layer with class probabilities
function drawOutputLayer(ctx, cx, cy) {
    const classes = [
        { label: 'Horse', prob: 0.2 },
        { label: 'Zebra', prob: 0.7 },
        { label: 'Dog', prob: 0.1 }
    ];
    const boxW = 55, boxH = 28, gap = 8;
    const totalH = classes.length * boxH + (classes.length - 1) * gap;
    const startY = cy - totalH / 2;

    ctx.fillStyle = '#333'; ctx.font = '12px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Output', cx, startY - 25);

    for (let i = 0; i < classes.length; i++) {
      const y = startY + i * (boxH + gap);
        // Probability box
        const fillIntensity = classes[i].prob;
        ctx.fillStyle = `rgba(243, 156, 18, ${0.3 + fillIntensity * 0.7})`;
        ctx.fillRect(cx - boxW / 2, y, boxW, boxH);
        ctx.strokeStyle = '#e67e22'; ctx.lineWidth = 1.5;
        ctx.strokeRect(cx - boxW / 2, y, boxW, boxH);

        // Probability text
        ctx.fillStyle = '#333'; ctx.font = 'bold 12px Arial'; ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(classes[i].prob.toFixed(1), cx, y + boxH / 2);

        // Class label
        ctx.fillStyle = '#444'; ctx.font = '11px Arial'; ctx.textAlign = 'left';
        ctx.fillText(classes[i].label, cx + boxW / 2 + 8, y + boxH / 2);
    }

    // SoftMax label
    ctx.fillStyle = '#555'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
    ctx.fillText('SoftMax', cx, startY + totalH + 50);
    ctx.fillText('Activation', cx, startY + totalH + 62);
    ctx.fillText('Function', cx, startY + totalH + 74);
}

// Draw dashed arrow
function drawArrow(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.setLineDash([4, 3]);
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#888'; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.setLineDash([]);

    // Arrowhead
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const aLen = 6;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - aLen * Math.cos(angle - 0.4), y2 - aLen * Math.sin(angle - 0.4));
    ctx.lineTo(x2 - aLen * Math.cos(angle + 0.4), y2 - aLen * Math.sin(angle + 0.4));
    ctx.closePath(); ctx.fillStyle = '#888'; ctx.fill();
}

// Info panel
function updateCNNInfoPanel() {
    const panel = document.getElementById('cnnInfo');
    let html = '<h3>CNN 架构解析</h3>';

    html += '<div class="section"><div class="section-title">整体流程</div>';
    html += '<div style="font-size:12px;">Input → [Conv+ReLU → Pooling] × N → Flatten → FC → Softmax → Output</div></div>';

    html += '<div class="section"><div class="section-title">1. 卷积层 (Convolution)</div>';
    html += '<div style="font-size:11px;">';
    html += '<b>作用：</b>用卷积核(Kernel)在输入上滑动，提取局部特征<br>';
    html += '<b>输出：</b>特征图(Feature Map)，每个核提取一种特征<br>';
    html += '<div class="formula">output[i,j] = Σ Σ input[i+m, j+n] × kernel[m,n] + bias</div>';
    html += '<b>特点：</b>参数共享 + 局部连接 → 大幅减少参数量</div></div>';

    html += '<div class="section"><div class="section-title">2. 激活函数 (ReLU)</div>';
    html += '<div style="font-size:11px;">';
    html += '<b>作用：</b>引入非线性，让网络能学习复杂模式<br>';
    html += '<div class="formula">ReLU(x) = max(0, x)</div>';
    html += '负值归零，正值保持不变</div></div>';

    html += '<div class="section"><div class="section-title">3. 池化层 (Pooling)</div>';
    html += '<div style="font-size:11px;">';
    html += '<b>作用：</b>降低特征图尺寸，减少计算量，增强平移不变性<br>';
    html += '<b>常见方式：</b>Max Pooling（取区域最大值）<br>';
    html += '<div class="formula">2×2 MaxPool: 特征图尺寸减半</div></div>';

    html += '<div class="section"><div class="section-title">4. 展平层 (Flatten)</div>';
    html += '<div style="font-size:11px;">';
    html += '将多维特征图展平为一维向量，作为全连接层的输入<br>';
    html += '<div class="formula">例: 7×7×64 → 3136维向量</div></div>';

    html += '<div class="section"><div class="section-title">5. 全连接层 (FC Layer)</div>';
    html += '<div style="font-size:11px;">';
    html += '<b>作用：</b>综合所有特征进行分类决策<br>';
    html += '与普通神经网络相同：z = Wx + b</div></div>';

    html += '<div class="section"><div class="section-title">6. Softmax 输出</div>';
    html += '<div style="font-size:11px;">';
    html += '<b>作用：</b>将输出转为概率分布(总和=1)<br>';
    html += '<div class="formula">P(class_i) = e^(z_i) / Σ e^(z_j)</div>';
    html += '概率最大的类别即为预测结果</div></div>';

    html += '<div class="section" style="background:#e8f5e9;padding:10px;border-radius:6px;">';
    html += '<div class="section-title" style="color:#2e7d32;">核心优势</div>';
    html += '<div style="font-size:11px;">';
    html += '• <b>局部感知：</b>卷积核只关注局部区域<br>';
    html += '• <b>参数共享：</b>同一卷积核扫描整个图像<br>';
    html += '• <b>层次特征：</b>浅层→边缘，深层→语义<br>';
    html += '• <b>平移不变：</b>目标位置改变不影响识别</div></div>';

    panel.innerHTML = html;
}