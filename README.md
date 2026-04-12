# AI Infra 学习记录

本仓库用于整理 **AI 基础设施（AI Infra）** 相关的学习与实验笔记，包括推理优化、分布式、算子与框架使用等主题的代码与备忘。

## 内容概览

| 目录 | 说明 |
|------|------|
| `chunked_prefill/` | Chunked Prefill 等与 LLM 推理相关的实验代码 |
| `test_code/` | 小型语法或 API 验证脚本 |
| `requirements.txt` | Python 依赖（含 PyTorch 等） |

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

建议使用仓库内配置的虚拟环境；若使用 VS Code / Cursor，可选择解释器 `.venv/bin/python`。

---

个人学习记录，随学随更新。
