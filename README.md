# DatasetGenerator
a tool for gerenate dataset from doc 




usage: `python main.py config.yaml`

```yaml
# config.yaml
openai:
  model: ""
  base_url: ""
  api_key: ""
save_dir: "./example" # 生成数据集保存目录
file_path: "./example/Olympics.txt" # 文件源地址
main_theme: "巴黎奥运会" # 文本主题
concurrent_api_requests_num: 4 # api异步请求数
method: "genQA" # 数据生成方式
```