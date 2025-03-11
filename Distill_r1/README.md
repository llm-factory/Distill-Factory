# Distill-r1

Distill-r1 是 [Distill-Factory](https://github.com/llm-factory/Distill-Factory) 项目下一个用于蒸馏推理数据的工具。

## 安装

```bash
git clone https://github.com/llm-factory/Distill-Factory.git
cd Distill-Factory
cd Distill-r1
pip install -e .
```

## 使用

### 数据集配置

您可以参考 [数据处理](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html) 文档将所需数据集的描述添加到 Distill-r1/data/dataset_info.json中。

示例:

```json
"gaokao_math":{
	"hf_hub_url":"hails/agieval-gaokao-mathcloze",
	"columns": {
		"prompt": "query",
		"response": "answer"
	},
	"split":"test"
}
```

### 运行

完成安装后，您可以设置配置文件，通过运行 `distillr1-cli run example.yaml` 进行数据合成。

```yaml
# example.yaml
model_name_or_path: "deepseek-reasoner"
dataset: gaokao_math
base_url: https://api.deepseek.com
api_key: ""
max_samples: 20
```

### api部署

除了调用外部 api, 您也可以通过 [vllm serve](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai#cli-reference) 指令自行部署。



