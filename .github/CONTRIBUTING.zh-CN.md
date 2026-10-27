# 参与开发

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md) · [开发指南](../docs/zh-CN/development.md)

欢迎改进配准算法、CPU、Metal 和 CUDA 算子，以及完善影像 I/O、测试和文档。较大改动建议先说明要解决的问题，以及涉及的后端或配准阶段。后续工作和验证见[开发计划](../docs/zh-CN/roadmap.md)。

## 开发环境

```bash
python -m pip install -e '.[test,build]'
# For Metal development on Apple Silicon:
python -m pip install -e '.[metal]'
# For CUDA development on NVIDIA:
python -m pip install -e '.[cuda]'
```

模块划分见[代码结构](../docs/zh-CN/architecture.md)。共用的图像类型、配置和主机端优化器放在 `core` 或 `pipeline`，数值计算放在对应后端。

## 代码修改

- 保持 LPS 坐标、毫米单位、变换顺序、数组布局和支持的数据类型一致。
- CPU 循环使用 Numba，GPU 算子使用对应后端的自定义 kernel。标量参数准备、数组管理与体素循环分开实现。
- 在安全的前提下复用临时缓冲区，不能让后续迭代覆盖已返回的数组或回调保存的数组。
- CPU 导入不依赖可选的 GPU 运行库；日志输出不增加 GPU 同步或度量计算。
- 为修改的行为补充小型测试，覆盖相关边界情况；新增参数需同步更新中英文文档。

## 测试

在项目根目录运行相关测试组：

```bash
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp64 --pattern test_fields.py
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_gaussian_sections32.py
python -B scripts/maintenance/check_markdown_links.py
npx --yes markdownlint-cli2@0.23.2
```

使用 `--list` 查看选中的测试。Metal 测试需要 Apple GPU，通过 `--suite metal` 选择；CUDA 测试需要 NVIDIA GPU，通过 `--suite cuda` 选择，详见[测试说明](../tests/README.zh-CN.md)。数值计算改动还需按照[测试与质量评估](../docs/zh-CN/validation.md)检查配准结果。修改文档不需要运行配准。

Markdown 检查需要 Node.js，使用 `.markdownlint-cli2.jsonc` 中的规则。配置覆盖中英文文档，不限制行长，并允许 README 使用 `div` 和 `h1` 居中排版。

## 提交修改

说明问题、修改后的行为以及运行过的测试。修复问题时提供可复现示例；优化性能时提供输入尺寸、参数、精度、线程数、预热方法和计时方式，并解释实际采用的实现方法。

影像数据、生成结果、缓存、凭据和本机配置不应放入软件包。保留适用的版权和许可证声明，详见 [NOTICE](../licenses/NOTICE.zh-CN.md)。运行日志和 CLI 消息统一使用英文。
