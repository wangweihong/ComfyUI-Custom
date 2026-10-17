# 3DGS离线发行材料

只存Wheels-Containerfiles的3dgs-v1 Release原始发行文件，不存源码、模型或宿主机设备驱动。文件由本目录.gitignore排除，不纳入Git。

在本目录执行：

```bash
release=https://github.com/wangweihong/Wheels-Containerfiles/releases/download/3dgs-v1
wget "$release/pycolmap-4.2.1-cp313-cp313-manylinux_2_28_x86_64.whl"
wget "$release/brush-app-x86_64-unknown-linux-gnu.tar.xz"
wget "$release/vulkan-loader-1.4.328.1-1.el10.x86_64.rpm"
sha256sum -c ../../3dgs.sha256
```

也可以从Wheels-Containerfiles执行`make prepare-3dgs verify-3dgs`，将其中三个发行文件复制到这里。版本与摘要须和本组件3dgs.sha256一致。

INSTALL_3DGS=true且OFFLINE_MODE=true时，只读取本地材料；缺件或摘要错误立即失败。在线分支则通过wget从固定Release下载。RPM安装禁用远程仓库，不增加启动时安装逻辑。
