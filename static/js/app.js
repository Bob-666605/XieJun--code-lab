// ==================== 全局工具函数 ====================

// 移动端导航菜单
document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.querySelector('.navbar__toggle');
    const navLinks = document.querySelector('.navbar__links');

    if (toggle && navLinks) {
        toggle.addEventListener('click', function() {
            navLinks.classList.toggle('navbar__links--open');
        });

        // 点击链接后关闭菜单
        navLinks.querySelectorAll('a').forEach(function(link) {
            link.addEventListener('click', function() {
                navLinks.classList.remove('navbar__links--open');
            });
        });
    }
});

// ==================== API 工具函数 ====================

const API = {
    /**
     * 上传图像
     */
    async uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        return await response.json();
    },

    /**
     * 执行去噪
     */
    async denoise(filename, methods, noiseType, noiseLevel, traditionalMethod) {
        const response = await fetch('/api/denoise', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                methods: methods,
                noise_type: noiseType,
                noise_level: noiseLevel,
                traditional_method: traditionalMethod || 'median'
            })
        });

        return await response.json();
    },

    /**
     * 运行评估
     */
    async evaluate(methods, noiseLevel) {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                methods: methods,
                noise_level: noiseLevel
            })
        });

        return await response.json();
    },

    /**
     * 获取可用方法
     */
    async getMethods() {
        const response = await fetch('/api/methods');
        return await response.json();
    }
};

// ==================== 工具函数 ====================

function showElement(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = 'block';
}

function hideElement(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
}

function formatNumber(num, decimals = 2) {
    return num !== null && num !== undefined ? num.toFixed(decimals) : 'N/A';
}
