// ==================== 去噪页面逻辑 ====================

document.addEventListener('DOMContentLoaded', function() {
    // 元素引用
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImage = document.getElementById('preview-image');
    const changeImageBtn = document.getElementById('change-image');
    const configSection = document.getElementById('config-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const methodResultsGrid = document.getElementById('method-results-grid');
    const metricsTbody = document.getElementById('metrics-tbody');
    const startBtn = document.getElementById('start-denoise');
    const resetBtn = document.getElementById('reset-btn');
    const noiseLevelSlider = document.getElementById('noise-level');
    const noiseLevelValue = document.getElementById('noise-level-value');
    const tradMethodGroup = document.getElementById('trad-method-group');

    // 对比视图元素
    const compareSection = document.getElementById('compare-section');
    const compareLeft = document.getElementById('compare-left');
    const compareRight = document.getElementById('compare-right');
    const compareLeftImg = document.getElementById('compare-left-img');
    const compareRightImg = document.getElementById('compare-right-img');
    const compareLeftLabel = document.getElementById('compare-left-label');
    const compareRightLabel = document.getElementById('compare-right-label');
    const compareLeftMetrics = document.getElementById('compare-left-metrics');
    const compareRightMetrics = document.getElementById('compare-right-metrics');

    let uploadedFilename = null;
    let denoiseResults = []; // 存储去噪结果用于对比

    // ==================== 上传功能 ====================

    // 点击上传
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 文件选择
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // 更换图像
    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', function() {
            resetUpload();
        });
    }

    // 处理文件上传
    async function handleFile(file) {
        // 验证文件类型
        const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!allowedTypes.includes(file.type)) {
            alert('请上传 JPG、PNG、BMP 或 TIFF 格式的图像');
            return;
        }

        // 验证文件大小
        if (file.size > 10 * 1024 * 1024) {
            alert('文件大小不能超过 10MB');
            return;
        }

        // 预览
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            uploadPreview.style.display = 'block';
            configSection.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // 上传到服务器
        try {
            const result = await API.uploadImage(file);
            if (result.success) {
                uploadedFilename = result.filename;
            } else {
                alert('上传失败: ' + (result.error || '未知错误'));
            }
        } catch (error) {
            alert('上传失败: ' + error.message);
        }
    }

    // 重置上传
    function resetUpload() {
        uploadedFilename = null;
        fileInput.value = '';
        uploadArea.style.display = 'block';
        uploadPreview.style.display = 'none';
        configSection.style.display = 'none';
        hideElement('loading-section');
        hideElement('results-section');
        methodResultsGrid.innerHTML = '';
        metricsTbody.innerHTML = '';
        denoiseResults = [];
    }

    // ==================== 配置功能 ====================

    // 噪声强度滑块
    if (noiseLevelSlider) {
        noiseLevelSlider.addEventListener('input', function() {
            noiseLevelValue.textContent = this.value;
        });
    }

    // 传统方法复选框显示/隐藏
    document.querySelectorAll('input[name="method"]').forEach(function(checkbox) {
        checkbox.addEventListener('change', function() {
            const hasTraditional = document.querySelector('input[name="method"][value="traditional"]').checked;
            tradMethodGroup.style.display = hasTraditional ? 'block' : 'none';
        });
    });

    // ==================== 去噪执行 ====================

    if (startBtn) {
        startBtn.addEventListener('click', async function() {
            if (!uploadedFilename) {
                alert('请先上传图像');
                return;
            }

            // 获取选中的方法
            const methods = [];
            document.querySelectorAll('input[name="method"]:checked').forEach(function(cb) {
                methods.push(cb.value);
            });

            if (methods.length === 0) {
                alert('请至少选择一种去噪方法');
                return;
            }

            // 获取配置
            const noiseType = document.querySelector('input[name="noise-type"]:checked').value;
            const noiseLevel = parseInt(noiseLevelSlider.value);
            const traditionalMethod = document.getElementById('traditional-method').value;

            // 显示加载
            hideElement('config-section');
            showElement('loading-section');

            try {
                // 执行去噪
                const result = await API.denoise(
                    uploadedFilename,
                    methods,
                    noiseType,
                    noiseLevel,
                    traditionalMethod
                );

                if (result.success) {
                    displayResults(result);
                } else {
                    alert('去噪失败: ' + (result.error || '未知错误'));
                    showElement('config-section');
                }
            } catch (error) {
                alert('去噪失败: ' + error.message);
                showElement('config-section');
            } finally {
                hideElement('loading-section');
            }
        });
    }

    // ==================== 结果展示 ====================

    function displayResults(result) {
        methodResultsGrid.innerHTML = '';
        metricsTbody.innerHTML = '';
        denoiseResults = [];

        // 隐藏原来的原图对比区域（改为卡片展示）
        const originalCompare = document.getElementById('original-compare');
        if (originalCompare) originalCompare.style.display = 'none';

        // 1. 首先添加带噪原图作为基准卡片
        const baselineCard = document.createElement('div');
        baselineCard.className = 'method-result-card baseline-card';
        baselineCard.innerHTML = `
            <div class="method-result-card__header">
                <span class="method-result-card__dot" style="background: #6c757d"></span>
                <span class="method-result-card__name">带噪原图 (基准)</span>
            </div>
            <div class="method-result-card__image-wrap">
                <img class="method-result-card__image" src="${result.noisy_path}" alt="带噪原图">
            </div>
            <div class="method-result-card__metrics">
                <div class="method-result-card__metric">
                    <span class="method-result-card__metric-label">PSNR</span>
                    <span class="method-result-card__metric-value">--</span>
                </div>
                <div class="method-result-card__metric">
                    <span class="method-result-card__metric-label">SSIM</span>
                    <span class="method-result-card__metric-value">--</span>
                </div>
                <div class="method-result-card__metric">
                    <span class="method-result-card__metric-label">TIME</span>
                    <span class="method-result-card__metric-value">--</span>
                </div>
            </div>
        `;
        methodResultsGrid.appendChild(baselineCard);

        // 2. 收集有效的去噪结果
        result.results.forEach(function(item) {
            if (item.error) {
                console.error(`Method ${item.method} failed:`, item.error);
                return;
            }

            const methodName = getMethodName(item.method);
            const methodType = getMethodType(item.method);

            denoiseResults.push({
                method: item.method,
                name: methodName,
                type: methodType,
                path: item.denoised_path,
                psnr: item.psnr,
                ssim: item.ssim,
                time_ms: item.time_ms
            });

            // 创建方法结果卡片
            const card = createMethodCard(methodName, methodType, item.denoised_path, item.psnr, item.ssim, item.time_ms);
            methodResultsGrid.appendChild(card);

            // 添加表格行
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${methodName}</strong></td>
                <td><span class="method-type-tag">${methodType}</span></td>
                <td>${formatNumber(item.psnr)}</td>
                <td>${formatNumber(item.ssim, 4)}</td>
                <td>${formatNumber(item.time_ms, 1)}</td>
            `;
            metricsTbody.appendChild(row);
        });

        // 设置对比视图
        setupCompareView();

        showElement('results-section');

        // 滚动到结果
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function createMethodCard(name, type, imagePath, psnr, ssim, timeMs) {
        const card = document.createElement('div');
        card.className = 'method-result-card';

        // 找到最佳指标
        const bestPsnr = denoiseResults.every(r => r.psnr === null) ? null :
            Math.max(...denoiseResults.filter(r => r.psnr !== null).map(r => r.psnr), psnr || 0);
        const bestSsim = denoiseResults.every(r => r.ssim === null) ? null :
            Math.max(...denoiseResults.filter(r => r.ssim !== null).map(r => r.ssim), ssim || 0);

        const isBestPsnr = psnr !== null && psnr === bestPsnr && denoiseResults.filter(r => r.psnr !== null).length > 1;
        const isBestSsim = ssim !== null && ssim === bestSsim && denoiseResults.filter(r => r.ssim !== null).length > 1;

        // 根据方法类型选择颜色
        const dotColor = getMethodColor(type);

        card.innerHTML = `
            <div class="method-result-card__header">
                <span class="method-result-card__dot" style="background: ${dotColor}"></span>
                <span class="method-result-card__name">${name} (${type})</span>
            </div>
            <div class="method-result-card__image-wrap">
                <img class="method-result-card__image" src="${imagePath}" alt="${name} 去噪结果">
            </div>
            <div class="method-result-card__metrics">
                <div class="method-result-card__metric ${isBestPsnr ? 'method-result-card__metric--best' : ''}">
                    <span class="method-result-card__metric-label">PSNR</span>
                    <span class="method-result-card__metric-value">${psnr !== null ? formatNumber(psnr) + ' dB' : '--'}</span>
                    ${isBestPsnr ? '<span class="best-badge">Best</span>' : ''}
                </div>
                <div class="method-result-card__metric ${isBestSsim ? 'method-result-card__metric--best' : ''}">
                    <span class="method-result-card__metric-label">SSIM</span>
                    <span class="method-result-card__metric-value">${ssim !== null ? formatNumber(ssim, 3) : '--'}</span>
                    ${isBestSsim ? '<span class="best-badge">Best</span>' : ''}
                </div>
                <div class="method-result-card__metric">
                    <span class="method-result-card__metric-label">TIME</span>
                    <span class="method-result-card__metric-value">${timeMs !== null ? formatNumber(timeMs, 0) + ' ms' : '--'}</span>
                </div>
            </div>
        `;

        return card;
    }

    function getMethodColor(type) {
        const colors = {
            '自监督': '#2563eb',
            '监督学习': '#7c3aed',
            '无需训练': '#16a34a'
        };
        return colors[type] || '#6c757d';
    }

    function setupCompareView() {
        // 清空选择器
        compareLeft.innerHTML = '';
        compareRight.innerHTML = '';

        // 填充选项
        denoiseResults.forEach(function(r, i) {
            const opt1 = document.createElement('option');
            opt1.value = i;
            opt1.textContent = r.name;
            compareLeft.appendChild(opt1);

            const opt2 = document.createElement('option');
            opt2.value = i;
            opt2.textContent = r.name;
            compareRight.appendChild(opt2);
        });

        // 默认选择前两个不同的方法
        if (denoiseResults.length >= 2) {
            compareRight.value = 1;
        }

        // 更新对比视图
        updateCompareView();

        // 监听选择变化
        compareLeft.addEventListener('change', updateCompareView);
        compareRight.addEventListener('change', updateCompareView);

        // 显示对比区域（至少需要2个结果）
        compareSection.style.display = denoiseResults.length >= 2 ? 'block' : 'none';
    }

    function updateCompareView() {
        const leftIdx = parseInt(compareLeft.value);
        const rightIdx = parseInt(compareRight.value);

        if (isNaN(leftIdx) || isNaN(rightIdx) || !denoiseResults[leftIdx] || !denoiseResults[rightIdx]) return;

        const left = denoiseResults[leftIdx];
        const right = denoiseResults[rightIdx];

        compareLeftLabel.textContent = left.name;
        compareRightLabel.textContent = right.name;
        compareLeftImg.src = left.path;
        compareRightImg.src = right.path;

        compareLeftMetrics.innerHTML = buildCompareMetrics(left);
        compareRightMetrics.innerHTML = buildCompareMetrics(right);

        // 高亮更好的指标
        highlightBetterMetric('compare-left', 'compare-right', left, right);
    }

    function buildCompareMetrics(result) {
        let html = '';
        if (result.psnr !== null) {
            html += `<div class="compare-metric"><span>PSNR</span><span>${formatNumber(result.psnr)} dB</span></div>`;
        }
        if (result.ssim !== null) {
            html += `<div class="compare-metric"><span>SSIM</span><span>${formatNumber(result.ssim, 4)}</span></div>`;
        }
        if (result.time_ms !== null) {
            html += `<div class="compare-metric"><span>耗时</span><span>${formatNumber(result.time_ms, 1)} ms</span></div>`;
        }
        return html;
    }

    function highlightBetterMetric(leftPrefix, rightPrefix, left, right) {
        // 重置样式
        const leftMetricsEl = document.getElementById(leftPrefix + '-metrics');
        const rightMetricsEl = document.getElementById(rightPrefix + '-metrics');
        leftMetricsEl.querySelectorAll('.compare-metric').forEach(el => el.classList.remove('compare-metric--better'));
        rightMetricsEl.querySelectorAll('.compare-metric').forEach(el => el.classList.remove('compare-metric--better'));

        if (left.psnr !== null && right.psnr !== null) {
            const betterPsnr = left.psnr > right.psnr ? leftMetricsEl : rightMetricsEl;
            betterPsnr.querySelectorAll('.compare-metric')[0]?.classList.add('compare-metric--better');
        }
        if (left.ssim !== null && right.ssim !== null) {
            const betterSsim = left.ssim > right.ssim ? leftMetricsEl : rightMetricsEl;
            const idx = left.psnr !== null ? 1 : 0;
            betterSsim.querySelectorAll('.compare-metric')[idx]?.classList.add('compare-metric--better');
        }
    }

    function getMethodName(method) {
        const names = {
            'mash': 'MASH',
            'dncnn': 'DnCNN',
            'traditional': '传统方法'
        };
        return names[method] || method;
    }

    function getMethodType(method) {
        const types = {
            'mash': '自监督',
            'dncnn': '监督学习',
            'traditional': '无需训练'
        };
        return types[method] || method;
    }

    // 重置按钮
    if (resetBtn) {
        resetBtn.addEventListener('click', resetUpload);
    }

    // ==================== 快速尝试 Demo 图 ====================

    document.querySelectorAll('.demo-thumb').forEach(function(thumb) {
        thumb.addEventListener('click', function() {
            const demoName = this.getAttribute('data-demo');
            const demoPath = '/static/demo_images/' + demoName;

            // 直接使用已有的 demo 图（无需重新上传）
            uploadedFilename = demoName;

            // 显示预览
            previewImage.src = demoPath;
            uploadArea.style.display = 'none';
            uploadPreview.style.display = 'block';
            configSection.style.display = 'block';
        });
    });
});
