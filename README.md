# DatasetGenerator
a tool for gerenate dataset from doc 



## 使用步骤

### 安装

```bash
git clone https://github.com/the-seeds/DatasetGenerator.git
cd DatasetGenerator
pip install -r requirements.txt
```

### 配置文件

配置文件示例:
```yaml
# config.yaml
openai:
  model: ""      # OpenAI 模型名称
  base_url: ""   # API 基础URL
  api_key: ""    # OpenAI API密钥
save_dir: "./example"              # 生成数据集保存目录
file_path: "./example/Olympics.txt" # 文件源地址
main_theme: "巴黎奥运会"           # 文本主题
concurrent_api_requests_num: 4      # api异步请求数
method: "genQA"                    # 数据生成方式
file_folder: "../../LLaMA-Factory-Doc" # 文件夹路径
file_type: rst                         # 文件类型
```

### 参数说明

| 参数名                      | 参数介绍                                   | 默认值                   |
| --------------------------- | ------------------------------------------ | ------------------------ |
| openai.model                | API模型名称                                | \                      |
| openai.base_url             | APIURL地址                                 | \                      |
| openai.api_key              | API密钥                                    | \                      |
| save_dir                    | 生成的数据集保存目录                       | "./example"              |
| file_path                   | 源文件路径，用于单文件处理                 | \ |
|file_folder|文件夹路径，用于批量处理多个文件| \           |
|file_type|要处理的文件类型|txt|
| main_theme                  | 文本主题，用于生成相关问题                 | \                |
| concurrent_api_requests_num | API并发请求数量       | 4                        |
| method                      | 数据生成方式 | "genQA" |


### 使用方式

#### 命令行

#### 单文件处理

在 `config.yaml` 中配置 `file_path` 指向单个文件，运行工具生成数据集。

```yaml
file_path: "./example/Olympics.txt"
main_theme: "巴黎奥运会"
```

------

#### 多文件处理

如果需要批量处理多个文件，可以指定 `file_folder` 和 `file_type`。

```
file_folder: "./example_docs"
file_type: "rst txt md"
```

----

运行命令：

```bash
python main.py config.yaml
```

生成的数据集保存在 `save_dir` 指定的目录中。

### webui

