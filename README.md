# PT Rollout Viewer

一个独立于 `Collab-Overcooked` 的小型浏览器工具，用来查看 PyTorch `.pt` rollout cache 文件。

## 功能

- 扫描目录中的 `.pt` 文件
- 加载指定文件并展示根对象概览
- 以表格形式浏览 rollout 样本
- 查看单条样本的字段摘要
- 对 `torch.Tensor` 展示 `dtype`、`shape`、`numel`、数值预览和基础统计
- 对 `prompt_ids`、`response_ids`、`critic_input_ids` 这类 token id 张量做文本解码
- 读取 `reward_curve.csv` 和 `train_curve.csv` 并自动轮询更新曲线

当前针对你给的这类文件做了优化：

- 根对象是 `list`
- 每个元素通常是 `dict`
- 常见字段包括 `prompt_ids`、`response_ids`、`critic_input_ids`、`reward`、`value` 等
- 默认 tokenizer 路径是 `/home/zhangshuwen/Collab-Overcooked/runs/Chef`
- 默认训练曲线路径是 `/home/zhangshuwen/Collab-Overcooked/runs/rl/train/`

## 运行

优先使用已经装好 `torch + fastapi + uvicorn` 的 Python 环境：

```bash
/home/zhangshuwen/vllm/bin/python3.10 -m app.main
```

或者直接：

```bash
./run_viewer.sh
```

默认会监听：

```text
http://127.0.0.1:8765
```

也可以自定义：

```bash
PT_VIEWER_HOST=0.0.0.0 PT_VIEWER_PORT=8765 /home/zhangshuwen/vllm/bin/python3.10 -m app.main
```

## 说明

- 这个仓库不会修改 `Collab-Overcooked` 代码。
- 前端是原生 HTML/CSS/JS，没有 Node 依赖。
- 后端读取 `.pt` 时只返回摘要，不会把完整大 tensor 全量传给浏览器。
