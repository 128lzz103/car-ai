# 项目运行指南 - 快速解决方案

## 🚨 当前状态诊断

### ✅ 正常运行的部分
- [x] Mock Vehicle API: http://127.0.0.1:8765 ✓ 运行正常
- [x] minicoder 模块导入: ✓ 成功
- [x] 意图分析（规则模式）: ✓ 正常工作
- [x] 配置文件: .env ✓ 已创建

### ⚠️ 需要配置的部分
- [ ] OpenAI API Key: 当前是占位符 `sk-proj-test-key-placeholder`
- [ ] 需要替换为真实的 API Key 才能使用 LLM 功能

---

## 🔧 三种运行方式

### 方式1: 仅演示意图识别（无需API Key）✅

**适用场景**: 快速演示项目架构、意图识别能力

```bash
# 意图分析（完全离线，无需API Key）
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '查询A102的续航']; main()"

# 输出：结构化意图JSON
# intent: range_query
# entities: {"vehicle_id": "A102"}
# plan: 1步计划（get_vehicle_status）
```

**其他演示命令**:
```bash
# 演示槽位补全识别
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '明天去上海，需要充电吗']; main()"

# 演示行程规划
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '明早8点去上海虹桥站，判断是否需要充电']; main()"

# 演示知识查询
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '冬天如何保养电池']; main()"
```

---

### 方式2: 使用本地 Ollama（免费，无需API Key）✅

**适用场景**: 完整功能测试，不想付费使用云端API

#### 步骤1: 安装 Ollama
```bash
# Windows: 下载安装
# https://ollama.com/download

# 启动 Ollama 服务（默认端口 11434）
```

#### 步骤2: 拉取模型
```bash
# 拉取轻量级模型（推荐）
ollama pull qwen2.5:7b

# 或其他模型
ollama pull llama3.1:8b
```

#### 步骤3: 修改配置
编辑 `.env` 文件：
```bash
MINICODER_PROVIDER=openai
OPENAI_API_KEY=ollama  # 随便填，Ollama不验证
OPENAI_BASE_URL=http://localhost:11434/v1
MINICODER_MODEL=qwen2.5:7b  # 或你拉取的模型名
```

#### 步骤4: 运行
```bash
python -c "from minicoder.cli import main; main()" --profile automotive
```

**优点**: 完全免费，数据不离开本地
**缺点**: 需要下载模型（约4-8GB），速度较慢

---

### 方式3: 使用云端API（完整功能，需付费）💰

**适用场景**: 生产环境、需要高质量回答

#### 选项A: OpenAI GPT-4o
```bash
# 编辑 .env
MINICODER_PROVIDER=openai
OPENAI_API_KEY=sk-你的真实key
OPENAI_BASE_URL=https://api.openai.com/v1
MINICODER_MODEL=gpt-4o

# 费用参考
# gpt-4o: $2.50 / 1M input tokens, $10 / 1M output tokens
```

#### 选项B: DeepSeek（性价比高）
```bash
# 编辑 .env
MINICODER_PROVIDER=openai
OPENAI_API_KEY=你的DeepSeek API Key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MINICODER_MODEL=deepseek-chat

# 费用参考
# deepseek-chat: ¥0.001 / 1K tokens (输入)
# 约为 OpenAI 的 1/100 价格
```

#### 选项C: Kimi（国内稳定）
```bash
# 编辑 .env
MINICODER_PROVIDER=openai
OPENAI_API_KEY=你的Kimi API Key
OPENAI_BASE_URL=https://api.moonshot.cn/v1
MINICODER_MODEL=moonshot-v1-8k
```

#### 选项D: Anthropic Claude
```bash
# 编辑 .env
MINICODER_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-你的key
MINICODER_MODEL=claude-sonnet-4-5
```

---

## 🎬 快速演示脚本（无需API Key）

创建一个演示脚本，展示项目所有核心能力：

<function_calls>
<invoke name="Write">
<parameter name="content">#!/bin/bash
# demo.sh - 项目演示脚本（无需API Key）

echo "========================================="
echo "车载AI智能助手 - 项目演示"
echo "========================================="
echo ""

echo "【演示1】简单续航查询"
echo "输入: 查询A102的续航"
echo "---"
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '查询A102的续航']; main()" | python -m json.tool
echo ""
echo ""

echo "【演示2】槽位缺失识别"
echo "输入: 明天去上海，需要充电吗"
echo "---"
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '明天去上海，需要充电吗']; main()" | python -m json.tool
echo ""
echo ""

echo "【演示3】复杂行程规划"
echo "输入: A102明早8点去上海虹桥站，判断是否需要充电"
echo "---"
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', 'A102明早8点去上海虹桥站，判断是否需要充电']; main()" | python -m json.tool
echo ""
echo ""

echo "【演示4】知识查询意图"
echo "输入: 冬天如何保养电池"
echo "---"
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '冬天如何保养电池']; main()" | python -m json.tool
echo ""
echo ""

echo "【演示5】空调控制"
echo "输入: 把A102的空调设为24度"
echo "---"
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '把A102的空调设为24度']; main()" | python -m json.tool
echo ""
echo ""

echo "========================================="
echo "演示完成！"
echo ""
echo "📊 核心能力展示："
echo "  ✓ 意图识别（7种车载意图）"
echo "  ✓ 实体抽取（车辆ID、地点、时间、温度）"
echo "  ✓ 槽位状态管理（缺失字段识别）"
echo "  ✓ 任务规划（自动生成执行步骤）"
echo ""
echo "🔗 Mock Vehicle API: http://127.0.0.1:8765/docs"
echo "========================================="
