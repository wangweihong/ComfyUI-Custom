把这两个文件覆盖到你的 nfypnode 节点包里：

1. nf_video_preview.py -> 覆盖原文件
2. js/nf_video_preview.js -> 放到包内 js 文件夹

然后：
- 重启 ComfyUI
- 浏览器 Ctrl+F5 强刷

这版改法：
- 预览 JS 改成直接参考 VHS 的 addDOMWidget 逻辑，不再靠全局扫描 video 元素绑定
- 继续保留 output 同名覆盖逻辑
- 鼠标移入尝试开声，移出强制静音
- format/crf 改动会实时影响预览样式
