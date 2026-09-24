# P3Bone：GitHub 发布操作说明

这次分成两个包：

| 文件 | 用途 | 在 GitHub 的位置 |
|---|---|---|
| `P3Bone_Code_Release_v1.zip` | 完整代码、配置、英文 README、许可和文档 | 解压后上传到仓库 |
| `P3Bone_Weights_seed20260814_v1.zip` | 一个固定 seed 的最终网络、匹配门控和校准参数 | 原样上传到 Release 附件 |

源码包中没有实验影像、医生标注、预测结果、结果图片、训练日志，也没有其他对比模型。权重包不包含 SAM 2.1 或历史定位器权重。固定分割网络可以直接对自行准备的影像推理。

## 一、建立仓库

1. 登录 GitHub，打开 <https://github.com/new>。
2. 仓库名称填写 `P3Bone`。
3. Description 可填写：`Code for P3Bone: Learning a Pediatric Wrist Bone Segmentation Network from Pseudo-Label Errors.`
4. 准备正式公开时选择 **Public**。如果想先自己预览，也可以先建 Private，确认后再改公开。
5. 不勾选自动添加 README，不选择自动生成许可证；源码包中已经有 README、LICENSE 和忽略规则。
6. 点击 **Create repository**。

本包使用“非商业科研与教学可用，商业用途需另行授权”的许可。因此不要再给仓库添加 MIT、Apache 等允许商业使用的通用许可证，以免出现相互冲突的条款。

## 二、上传源码

1. 在电脑上解压 `P3Bone_Code_Release_v1.zip`，打开其中的 `P3Bone` 文件夹。
2. 进入 GitHub 新仓库，点击 **uploading an existing file**；已有文件的仓库使用 **Add file → Upload files**。
3. 把 `P3Bone` 文件夹**里面的文件和子文件夹**拖到上传区，保持目录结构。不要只上传源码 ZIP，也不要把外层 `P3Bone` 文件夹多套一层。
4. 确认仓库根目录能直接看到 `README.md`、`LICENSE`、`pyproject.toml`、`p3bone/`、`configs/`、`docs/` 和 `tests/`。
5. Commit message 填 `Initial P3Bone code release`，按页面提示提交到主分支。若选择了新分支，则按提示创建并合并 pull request。

文件选择器可能隐藏 `.gitignore`。可以开启“显示隐藏文件”后一起上传；如果漏了，就用 **Add file → Create new file** 新建 `.gitignore`，复制包内同名文件内容。它用于避免后续把数据、权重和运行结果误加进代码仓库。

GitHub 官方目前允许浏览器单次上传最多 100 个文件、每个文件最大 25 MiB。本源码包在该范围内；若浏览器不支持拖入文件夹，可使用 GitHub Desktop 保持目录结构上传。

## 三、发布固定权重

1. 仓库右侧找到 **Releases**，点击 **Create a new release** 或 **Draft a new release**。
2. 新建标签 `v1.0.0`，目标分支选择当前主分支。
3. Release title 填 `P3Bone v1.0.0`。
4. 把 `P3Bone_Weights_seed20260814_v1.zip` **原样作为附件上传**。不用把它解压后放进源码目录。
5. Release notes 可以复制：

```text
P3Bone code and fixed seed-20260814 weights for the final LOCAL_GATE_B_W1 method.

The weight archive contains the final segmentation network, matched local gate,
FREC/direct foreground head coefficients, the frozen C4 prior, and SHA256 checksums.
The final network performs inference without SAM or the local gate.

Study images, annotations, patient manifests, predictions and comparison methods
are not included. Data-access and commercial-use inquiries: 1310434684@qq.com.

Original P3Bone code and weights are provided for noncommercial research and
teaching under the included license. Third-party terms remain applicable.
```

6. 如果希望正式发布稳定版，不要勾选 **Set as a pre-release**，检查后点击 **Publish release**。

## 四、发布后做一次检查

1. 仓库首页能正常显示英文 README；其中联系邮箱应为 `1310434684@qq.com`。
2. Release 附件可以下载，文件名和 README 中一致。
3. 打开许可证，确认保留了非商业科研与教学用途以及商业联系条款。
4. 仓库文件列表中没有 `data/`、训练结果、患者名单、医生标注或服务器原始压缩包。
5. 把仓库主页链接复制给导师。后续论文代码链接填写仓库 URL，而不是某次 ZIP 的临时下载地址。

想给 README 添加直接下载链接，可在 Release 页面右键复制权重附件链接，编辑 README 的“Fixed weights and inference”段落，替换为实际链接。不要在尚未创建 Release 时填写猜测的地址。

## 五、别人下载后怎样运行

1. 下载源码和固定权重 ZIP。
2. 将权重包内的 `.pt`、`.json` 等文件放到源码目录中的 `weights/seed20260814/`。
3. 按 README 安装依赖、准备自己的影像 CSV。
4. 先运行 `verify-weights`，再运行 `predict`。有参考掩膜时才能做准确率评价。

完整训练需要自行提供标注、划分和上游依赖；公开一个固定 seed 不能替代论文三个 seed 的复现实验。README 已写明标注暂不公开，有需要可以联系作者讨论，不承诺所有申请都必然获得数据。

## 官方帮助

- [新建仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [通过网页上传文件](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [创建和管理 Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
