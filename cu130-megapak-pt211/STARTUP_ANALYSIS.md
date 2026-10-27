# ComfyUI 镜像启动耗时分析与修复方案

分析日期：2026-10-27。对象：正在运行的 `comfyui-test-cu130-megapak-pt211`。
初次诊断只读取配置、源码、日志并复现依赖查询行为。后续代码修复与独立容器验证见文末实施记录；原有运行服务保持不变。

## 结论与实测

最新一次容器启动至服务开始监听约 **326 秒（5 分 26 秒）**。
主要问题是节点在导入阶段同步执行 pip，而不是 GPU 算力不足。

| 项目 | 耗时 | 证据 |
| --- | ---: | --- |
| comfyui_LLM_party 导入 | 166.5 秒 | 导入统计；其中约 162.8 秒等待 Discord 依赖处理 |
| z-tipo-extension 导入 | 77.8 秒 | 导入统计；其中约 77.7 秒执行 tipo-kgen 安装 |
| 自定义节点初始化复制 | 7.7 秒 | entrypoint 日志时间差 |
| Manager 前端依赖修复等待 | 约 11 秒 | PyTorch 误报后至前端修复完成的日志时间差 |
| 其他启动过程 | 约 63 秒 | 总时长减去上述项目，含基础模块、GPU 初始化及其他节点等 |

两个节点合计 244.3 秒，占启动总耗时约 **75%**。169 条自定义节点导入记录合计约 267 秒，其他节点约 22.7 秒。
日志缓冲和不同输出流会影响细分时间边界；总耗时及节点导入统计足以确定优先级。

启动时间线（北京时间）：

- 19:50:00.124：容器启动。
- 19:50:08.295：entrypoint 开始启动 Python。
- 19:51:20.725：LLM_party 输出 `llama-cpp-python installed`。
- 19:54:03.507：LLM_party 输出 `Successfully installed py-cord[voice]`。
- 19:54:05.273：TIPO 输出 `Installing tipo-kgen>=0.3.1`。
- 19:55:22.984：TIPO 安装阶段结束。
- 19:55:26.457：输出 Web 界面地址。

Docker 原始日志为 UTC，以上已转换为 Asia/Shanghai。日志中更晚的浏览器访问不计入启动耗时。
检查时首页返回 HTTP 200，服务已经正常监听。

## P0：修正 LLM_party 的包检测错误

运行中的节点版本为 `3790b26a438bb8e77c1ad24d93870f710cc8ae41`。
其 `__init__.py` 在导入阶段调用 `manage_discord_packages()`，
`install.py:77-82` 通过 `importlib.metadata.version(package_name)` 检测发行包，
但 `install.py:298` 传入了 `py-cord[voice]`。

现场复现：

```text
metadata.version('py-cord')        -> 2.8.1
metadata.version('py-cord[voice]') -> PackageNotFoundError
```

`[voice]` 是可选依赖标记，不是发行包名。即使包已经存在，每次启动仍会执行 pip。
成功提示也不代表每次真正下载了文件，只能证明 pip 命令执行成功；依赖解析、网络访问及重试均可能消耗时间。
本次 pip 的标准输出被节点捕获，无法进一步确定 162.8 秒内部各项网络等待的占比。

修复方案：

1. 镜像构建阶段预装并锁定 `py-cord[voice]`。本机已安装版本 `py-cord==2.8.1`、`PyNaCl==1.6.2` 可作为验证起点。
2. 在节点 fork 中修正检测：使用 `Requirement(requirement).name` 取得发行包名，按需验证版本；对 voice 功能另查 `PyNaCl` 等必要依赖。
3. 将自动安装移出模块导入阶段。镜像环境启动时只检测缺失项并提示，在构建或显式维护操作中安装。
4. 固定修复后的节点 commit，并保证在线和离线构建分支均使用该版本。

仅提前安装 py-cord 无法修正错误的检测逻辑，仍会重复调用 pip。
如果不使用这个节点包，可将其目录改为 `.disabled` 后在维护窗口重启；这会使依赖它的工作流缺少节点。

## P0：将 TIPO 依赖安装移至镜像构建阶段

节点版本为 `319be75ed08e237770c31fcfde87cab711f74372`。
`tipo_installer/__init__.py:115-135` 检查 `tipo-kgen` 版本；低于 0.3.1 或缺失时执行 `pip install -U tipo-kgen>=0.3.1`。
该调用发生在扩展加载时，阻塞后续启动。

本次安装后，`tipo-kgen==0.3.1` 位于 `/root/.local/lib/python3.13/site-packages`。
上一份启动日志中该节点仅耗时 0.1 秒，证明依赖存在时通常无需这段等待；它与 LLM_party 每次误判的情况不同。

修复方案：

1. 在 `builder-scripts/pak1.txt` 或专门的启动依赖文件中加入经过验证的固定版本，例如 `tipo-kgen==0.3.1`，沿用 PyTorch constraints。
2. 构建时确认此版本满足当前固定的节点源码要求，并验证 `import kgen`。
3. 确认依赖齐全后设置 `TIPO_NO_AUTO_INSTALL=1`，禁止运行时自动安装。该开关已在当前节点源码中确认存在。
4. 后续更新节点时同步调整依赖锁定版本。

不要在缺失依赖时单独打开禁装开关，否则 TIPO 节点可能无法加载。
Dockerfile 目前先安装依赖、随后更新或复制节点源码，两者可能失配，需要统一锁定。

## P1：消除 PIP_USER 对 Manager 包列表的干扰

`runner-scripts/entrypoint.sh:114` 全局导出 `PIP_USER=true`。
Manager 4.2.2 的 `common/manager_util.py:292-313` 用 `pip list` 收集已安装包，继承了这个变量。
该变量同时影响 pip 的安装与列表命令。

现场只读复现结果：

| 环境 | 列出的包数量 | torch / 前端包 |
| --- | ---: | --- |
| PIP_USER=true | 3 | 均不可见 |
| PIP_USER=false | 527 | 均可见 |

实际已安装 `torch==2.11.0+cu130`、`torchvision==0.26.0+cu130`、`torchaudio==2.11.0+cu130` 和前端包 1.55.18。
因此日志中的 `PyTorch is not installed` 是查询范围导致的误报。
Manager 同时误判前端包缺失，随后触发不必要的 pip 修复。

推荐在保留用户安装持久化行为的前提下，修正 Manager 包查询子进程：

```python
query_env = os.environ.copy()
query_env["PIP_USER"] = "false"
result = subprocess.check_output(
    make_pip_cmd(["list"]),
    universal_newlines=True,
    env=query_env,
)
```

补丁只应用于查询命令；安装命令继续使用原环境，保留 `/root/.local` 的安装位置。
另一种方案是删除全局 PIP_USER、只在安装操作上显式使用 `--user`，但必须同步修改 Manager 等安装入口，否则新增依赖可能落到容器层，重建后丢失。

## P1：避免临时目录丢失依赖、节点和缓存

`Makefile:161` 每次测试创建新的 `mktemp` 目录。
当前容器的 `/root/.local`、`/root/.cache`、`custom_nodes` 等实际挂载到 `/tmp/comfyui-test-U0u832/`。
这种路径无法可靠跨宿主机清理保留；反复运行测试也不会复用旧目录。

本次启动重新复制了 custom_nodes，随后将 tipo-kgen 装入用户目录，说明这次确实走了重新初始化路径。
目录此前丢失的具体原因尚未取得直接证据，不能仅凭日志断言是宿主机重启。

修复方案：

1. 日常使用改为稳定的宿主机目录或 Docker named volume；Makefile 增加可配置、可复用的存储目录。
2. 将临时目录测试明确留作冷启动测试，另加复用卷测试。
3. `entrypoint.sh:35` 当前用示例文件是否存在判断节点是否已初始化，建议改为复制成功后才写入的独立完成标记，并保留中断后重试能力。
4. 对镜像更新制定节点迁移规则。当前复制使用不覆盖策略，重建镜像不会自动覆盖持久化目录里的旧源码，修复后的 fork 需要单独迁移或重新初始化测试卷。

## 次要问题与优先级边界

- 检查了现有 pre-start 脚本，没有额外安装或下载行为，不是本次主要瓶颈。
- 现有 Dockerfile 已删除 Manager 的 `default_cache_update()` 调用，不能将本次主要耗时归因于 `FETCH ComfyRegistry Data`。
- 节点与字节码缓存当前挂在 ext4 上。模型位于 NTFS，但 LoRA-Manager 日志显示 LoRA 缓存初始化约 0.31 秒、checkpoint 缓存约 0.07 秒；没有证据表明模型扫描造成这次几分钟的等待。
- 镜像约 15.95 GB（Docker 本地 Size）。拉取和解压会影响新部署，但本次时间线从容器已经启动算起，不能用镜像体积解释随后 244 秒的 pip 阻塞。
- 日志还存在 LiTo 缺少 `comfy_3d_viewers`、GaussianPack 与 comfy-env 配置不兼容、Sharp 隔离环境缺失、comfy-kitchen 目录无法作为节点加载等问题。需要单独修复，但不是已测出的主要时间来源。
- 当前节点源码与镜像配置存在漂移，例如运行目录仍有 Dockerfile 已声明删除的 InstantID。依赖更新与持久化节点更新应一起验证。

## 实施顺序与验收

1. 修正并固定 LLM_party fork，同时在构建阶段预装 py-cord voice 和 tipo-kgen。
2. 修正 Manager 查询环境，确认能列出全局依赖。
3. 设置 TIPO 禁止自动安装，并改用稳定存储。
4. 增加启动分阶段计时，以及以 HTTP 可用为标准的健康检查。
5. 在独立测试容器验证空卷冷启动、复用卷重启及容器重建，分别记录从容器启动到 HTTP 可用的时间。
6. 核对两组节点注册，并运行依赖它们的最小工作流，确认速度改善没有通过缺失节点实现。

验收要求：启动日志不再执行上述两次 pip 安装；Manager 不再误报缺少 PyTorch；重复启动不重新复制节点；重建后依赖仍存在。
必要依赖与资源已准备的条件下，可限制测试容器外网验证不再发生下载等待，同时避免触碰当前运行服务。

按本次耗时直接扣除两个阻塞节点的全部导入时间，其他阶段约为 82 秒；修复后仍须保留这两个节点正常导入的耗时。
历史暖启动日志从 Manager 记录到监听约 18 秒，但其缓存、网络及主机负载条件不同，不能承诺修复后固定达到该数字。
最终收益需要独立容器实测确认。

## 官方语义参考

- [pip list：--user / PIP_USER 只列出 user-site 包](https://pip.pypa.io/en/stable/cli/pip_list/)
- [PyPA dependency specifiers：extras 是发行包的可选依赖集合](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)

本报告中的耗时和代码结论来自当前容器现场；上面的文档仅用于核对 pip 与依赖标记语义。

## 修复实施与验证记录

已在 ComfyUI-Custom 中落实以下修改：

- `builder-scripts/startup-requirements.txt` 固定 `py-cord[voice]==2.8.1` 和 `tipo-kgen==0.3.1`，由 `pak1.txt` 引入；构建时显式使用 `PIP_USER=false` 安装到镜像系统环境。
- `runner-scripts/patch-startup.py` 修正 Manager 包查询环境，以及 LLM_party 对 extras 和 voice 依赖的检测。补丁先验证源码结构，再写入；重复应用不会重新写文件。无法识别的上游源码会明确失败，避免静默应用错误补丁。
- Dockerfile 在在线/离线节点路径汇合后应用补丁，并验证 `kgen`、`discord`、`nacl`、`davey` 可以导入。
- entrypoint 在启动前也检查持久化节点源码，使旧卷无需整体替换即可应用修复。修复保留上游在依赖确实缺失时的安装能力；正常镜像启动复用构建时安装的依赖。
- 默认 `TIPO_NO_AUTO_INSTALL=1`；设为 `0` 可显式恢复上游自动安装行为。
- 节点复制完成后写独立标记；已有示例标记的旧卷直接迁移，避免恢复用户刻意删除的节点；新复制中断时保留进行中标记，下次重试。
- Makefile 默认复用 `$(HOME)/.local/share/comfyui-custom/tests/<镜像标签>`，并持久化用户目录。可配置 `TEST_STORAGE_ROOT`；`COLD_START=true` 会在该目录下另建独立存储。
- 添加启动分阶段日志和 HTTP 健康检查；用 `exec` 启动主进程。提示词种子库改为复制，避免首次初始化消耗掉镜像内原始文件。

### 验证结果

在现有镜像 `865db9016e67` 上叠加新增依赖和相同启动脚本，构建本地独立验证镜像 `comfyui-custom:startup-fix-validation`。
首次在线构建的 wheel 下载等待较长，改用本机 pip 已缓存的原始 wheel，在断网条件下完成依赖解析、安装和导入检查。
本次未从更新上游源码开始执行整个正式 Dockerfile 的完整重建，也未覆盖正式镜像标签。

验证容器使用独立的节点、缓存、用户依赖及用户配置卷，启用 GPU，禁止外网，未挂载实际模型目录。
不禁用任何整包节点，以服务输出监听地址为计时终点：

| 场景 | 启动耗时 | LLM_party 导入 | TIPO 导入 |
| --- | ---: | ---: | ---: |
| 空卷首次启动 | 67.175 秒 | 3.8 秒 | 0.1 秒 |
| 同容器复用卷重启 | 16.976 秒 | 2.6 秒 | 0.1 秒 |
| 持久化卷恢复旧版 LLM_party 源码、移除新标记，再重建容器 | 43.311 秒 | 2.7 秒 | 0.1 秒 |

第三种场景确认日志出现 `Adopted existing Custom Nodes` 和旧 `install.py` 补丁应用记录，且没有重新复制整套节点。
启动条件及缓存状态不同，因此上述数字不能作为正式部署耗时承诺。

首次启动 67 秒的进一步拆分（节点统计按 0.1 秒取整，阶段余量为近似值）：

| 项目 | 首次空卷 | 同容器重启 |
| --- | ---: | ---: |
| lf-nodes 导入 | 26.4 秒 | 0.4 秒 |
| LLM_party 导入 | 3.8 秒 | 2.6 秒 |
| ZML-Image 导入 | 3.0 秒 | 1.7 秒 |
| 其他节点导入，含 TIPO | 14.4 秒 | 4.5 秒 |
| entrypoint 初始化与复制 | 约 3.3 秒 | 约 0.2 秒 |
| Python / ComfyUI / GPU 基础启动及其余开销 | 约 16.3 秒 | 约 7.6 秒 |

169 条节点导入记录合计首次约 47.6 秒，重启约 9.2 秒。重建容器时 lf-nodes 再次耗时 25.6 秒，是重建耗时 43 秒的主要增量。
日志中约 24 秒的空档位于 Numba 初始化警告与 LF Nodes 初始化完成之间。
源码显示 LF Nodes 会遍历并导入整套模块，其中 CLIPSeg/WD14 模块在顶层导入 Transformers 模型类。
这些证据把剩余主要瓶颈定位在 LF Nodes 的导入链；具体依赖内部的 CPU 时间仍需要导入 profiling 才能精确分配。

另已确认新验证镜像的系统 site-packages 中没有 Numba `.nbc` 缓存，而现有长期运行容器中有 85 个，位于 librosa 等依赖的系统目录。
这些位置不在当前持久化的 `/root/.cache` 内。该现象高度指向首次编译及缓存随容器层丢失的问题，尚未通过调整缓存目录的对照实验完成因果验证。
后续优先验证 LF Nodes 重依赖延迟导入，以及将 `NUMBA_CACHE_DIR` 指向持久化缓存目录；不能把同容器重启的 17 秒当作容器重建的稳定耗时。

其他验证：

- 13 项测试全部通过，包括已有的 3DGS 构建接口测试，以及包检测、依赖缺失、查询环境、重复补丁、未知源码保护、旧卷迁移、复制失败重试和用户文件保留。
- 在实际节点源码上执行检测函数，确认已安装的 `py-cord[voice]` 不会触发安装或卸载。
- 在实际 Manager 源码上执行包查询，确认父环境 `PIP_USER=true` 不变时仍能查到系统 PyTorch 和前端包。
- HTTP `/object_info` 确认 3 个 TIPO 节点和 170 个 LLM_party 节点成功注册；未执行需要下载模型的推理工作流。
- 启动日志不再出现两个目标包的自动安装、Manager 的 PyTorch 缺失误报或前端依赖自动修复。
- shell 语法、默认和 3DGS 两组 Compose 配置、Makefile 的冷/暖存储命令展开及 `git diff --check` 通过。

### 部署生效

在仓库根目录执行 `make build-cu130-megapak-pt211`；在维护窗口使用新镜像重建现有容器，并沿用原来的模型、工作流及存储挂载。
仅重启旧容器不会更新它使用的镜像。旧节点卷会由新的 entrypoint 自动应用兼容补丁，不必删除用户节点目录。
如果使用其他宿主机启动命令，需将临时存储改为稳定目录；已有 Compose 的相对存储目录可以继续使用。
