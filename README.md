# McDeltaSearch：mcdelta 版本变更检索

用自然语言检索 [Fallen-Breath/mcdelta](https://github.com/Fallen-Breath/mcdelta) 中 Minecraft 19w36a → 26.4-snapshot-5 的全部版本变更分析（448 个版本，6227 条变更）。

数据对应 mcdelta 提交 `12ec70c`（2026-10-27）。

## 功能

- **混合检索**：字符二元组 BM25 关键词检索 + 语义向量检索，用 RRF（倒数排名融合）合并。
- **浏览器内语义模型**：bge-small-zh-v1.5（int8 权重），用手写的纯 JavaScript 推理，不依赖服务器、transformers.js 或 WASM。
- **术语与黑话识别**：基于 [TechMC Glossary](https://github.com/TechMC-Glossary/TechMC-Glossary) 的 339 个术语，以及“术语 → 机制 → 代码类”的依赖图扩展检索。
- **重要性与条目形式**：标出每条变更在本版报告中的排位（mcdelta 按对已有玩法的影响排列条目），以及它是完整分析、简述还是合并简述。
- **变更类别**：按“代码出处”的源码路径自动推断类别（方块与物品、实体与玩家、地形生成、界面与显示等，沿用 mcdelta 提示词的优先级表），以虚线标签显示。抽样核对的档位准确率约 86%，只用于标注和筛选，不参与排序。
- **排序与筛选**：按相关度、重要性、版本时间排序；按正式版范围、条目形式和变更类别筛选。

## 在本地运行

```shell
cd site
python -m http.server 8000
```

打开 <http://localhost:8000>。不能直接双击 `index.html`，浏览器会阻止它读取同目录的数据文件。首次加载约 45 MB，之后由浏览器缓存。页面会尝试从 Google Fonts 加载一个等宽字体，离线时自动改用系统字体。

部署到任何静态托管（GitHub Pages 等）只需上传 `site/` 目录。

## 目录结构

```text
site/                      可直接部署的静态网站
  index.html               页面（已内嵌 encoder.js）
  docs.json                全部变更段落及元数据（格式见下）
  vectors.0.txt            段落向量（int8，base64）
  model.json               分词表与权重清单
  emb.*.txt layers.*.txt   模型权重（base64 分片）
  blobs.json               base64 分片清单
  concepts.json            术语与机制依赖图（网页用精简版）
build/                     构建、重建与评测脚本
data/
  concepts_full.json       完整概念图（含别名在报告中的出现次数、置信度），供审核
  concepts_pilot.json      最初 10 个概念的试点版
  eval_questions.json      51 个普通测试问题
  eval_jargon.json         15 个黑话测试问题
  TechMC-Glossary-LICENSE.txt
```

权重和向量之所以是 base64 文本：claude.ai 的 Artifact 只提供文本和常见网页文件类型。自己托管时可以改回二进制，在 `encoder.js` 的 `fetchBlob` 中改为直接读取 `arrayBuffer()`。

### docs.json 格式

```text
commit    mcdelta 提交哈希（原文链接固定到这个提交）
nwin      向量窗口数
versions  [正式版目录, 文件名, 版本号, 显示名, 本报告的条目数]
docs      [版本序号, 标题, slug, 小节列表[[小节名, 正文]], 代码类名列表, 形式, 本版排位, 类别]
          形式：full = 完整分析，single = 简述，merged = 合并简述
          排位：1 起算，即在报告中的先后
          类别：block_item | entity | worldgen | world_run | network | admin | ui | command | protocol | null
                由 build/path_category.py 按源码路径推断，null 表示路径无法判断
```

## 重要性与条目形式

mcdelta 的写作规则（`prompts/snapshot_analysis_worker.md`、`docs/report_writing_requirements.md`）规定：

- 条目按“对已有玩法的实际影响”排列先后，这一点从最早的 v0 提示词起就有。
- 篇幅取决于解释所需的信息：“代码量少的重要变化可以靠前且简述”，并且要“避免模板的展示顺序变成重要性排序”。

所以这里把两者分开处理：

| 属性 | 来源 | 网页上的表现 |
|---|---|---|
| **重要性** | 条目在本版报告中的排位 | “本版第 r/n 条”；第 1 条标为“本版首要”，前三分之一标为“本版靠前”，结果左侧有色条 |
| **形式** | 报告格式 | “完整分析”（有“差异出现的场景 / 版本变化 / 实际影响”小节）、“合并简述”（无小节、正文以列表开头且至少 2 项）、“简述”（其余） |

统计（6227 条）：完整分析 1328 条、简述 3902 条、合并简述 997 条。完整分析平均位于报告前 20% 处，但简述条目也经常排在前面，印证了“形式不等于重要性”。

“重要性”排序：相关度前 20 条按档位（首要 → 靠前 → 其余）重排，档内保持相关度顺序；其余结果随后，同样处理。这样做是为了避免相关度很弱的“本版首要”条目挤到明显相关的结果前面。

注意：排位只在同一份报告内有比较意义。一份只有 3 条的报告的第 1 条，和一份 60 条的报告的第 1 条，重要程度未必相当。

## 构建流程

依赖：Python 3.11+、`numpy`、`scipy`、`onnx`、`onnxruntime`、`tokenizers`。**脚本中的路径大多写死为 `/root/...`，使用前需要改成本地路径**（见 `HANDOFF.md` 或各脚本开头）。

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1 | `parse.py` | 解析 mcdelta `reports/`，按 `##` 切段，提取类名、源码路径、形式、排位，沿“上一个版本”链排序，输出 `corpus.json`（源码路径 `paths` 只在 `corpus.json` 中，网站数据不含） |
| 2 | `embed_docs.py`（依赖 `emb.py`） | 首次全量编码：每段“标题 + 正文”切成最多 3 个约 440 字的窗口 |
| 3 | `export_model.py`（依赖 `model.py`） | 把 int8 ONNX 权重导出为 `encoder.js` 读取的格式（模型不变就不用重跑） |
| 4 | `export_data.py`（读 `path_category.py`） | 向量量化为 int8；按源码路径推断每段的类别；写出 `docs.json` 和 `vectors.bin` |
| 5 | `glossary_norm.py` → `graph_build.py`（读 `graph_spec.py`） | 整理词汇库、生成概念图 |
| 6 | `assemble_site.py` | 切分 base64、内嵌 `encoder.js`、生成 `site/` |

模型来源：bge-small-zh-v1.5 的量化 ONNX 与 `tokenizer.json`。原构建环境无法访问 Hugging Face，是从 npm 包 `dsh-code-search@0.1.0` 取得的（transformers.js 格式，等同 `Xenova/bge-small-zh-v1.5` 的 `onnx/model_quantized.onnx`）。分词结果和向量已与 Python 参考实现 `model.py` 逐一核对一致，但没有与官方文件逐字节比对，正式使用时建议从 Hugging Face 重新下载。

### mcdelta 更新后重建索引

```shell
cd mcdelta && git pull && cd ../build
cp corpus.json corpus_prev.json && cp vecs.npy vecs_prev.npy && cp owner.npy owner_prev.npy
python reindex.py corpus_prev.json vecs_prev.npy owner_prev.npy
python assemble_site.py <export_data.py 的输出目录> ../data/concepts_full.json ../site
python eval_full.py
```

`reindex.py` 用“版本号#slug + 内容哈希”比对新旧段落，未改动的段落复用旧向量，只编码新增或改动的段落。

| 变动 | 影响 | 需要人工处理 |
|---|---|---|
| 报告移动目录 | 段落标识不变，无需编码 | 否 |
| 新增报告 / 修改正文 | 只编码相关段落 | 否 |
| 修改标题 | slug 改变，视为删除旧段落并新增 | 测试题若引用旧标识，`eval_full.py` 会报错指出 |
| 删除段落 | 从索引移除 | 同上 |

形式和排位是解析时从报告格式直接得出的，重建时会自动更新，不影响向量。

## 评测

| 脚本 | 内容 |
|---|---|
| `eval_full.py` | 主评测：普通 51 题 + 黑话 15 题，对比有无概念图 |
| `search_eval.py` | Python 版检索，与网页逻辑一致 |
| `eval_graph.py` | 试点概念图评测（旧，测试题仍按列表下标，已不再维护） |
| `eval_map.py`、`embed_large*.py` | bge-large 线性映射实验（结论：无效） |

测试题用稳定标识 `版本号#slug` 引用段落。

当前基线（`12ec70c`）：

| 测试集 | 前 10 命中 | MRR |
|---|---|---|
| 普通 51 题 | 0.90 | 0.671 |
| 黑话 15 题（启用概念图） | 0.93 | 0.763 |
| 黑话 15 题（不启用） | 0.60 | 0.489 |

## 已知局限

- 概念图中所有“术语 → 机制”关系都是人工凭经验起草的，没有证据，其中 36 条标为低置信度。
- 测试题由同一作者编写和标注，可能有偏。“地毯复制机坏了”一题沿用刷铁轨机的相关改动，因为两者原理相同。
- 部分机制（如 MSPT、半连接性）在报告覆盖期间几乎没有改动，相关查询的结果是噪音，页面暂不会提示“没有直接相关的改动”。
- int8 量化对 bge-small 的影响未与 fp32 版本对比过。

## 许可

- 代码：MIT，见 `LICENSE`。下列数据与模型权重保持各自的许可。
- 报告数据：Fallen-Breath/mcdelta，WTFPL。
- 术语表：TechMC Glossary，MIT。
- 模型：bge-small-zh-v1.5（BAAI），MIT。
