# DatasetGenerator
a tool for gerenate dataset from doc 




usage: `python main.py config.yaml`

```yaml
# config.yaml
openai:
  model: ""
  base_url: ""
  api_key: ""
save_dir: "./example"
file_path: "./example/Olympics.txt"
main_theme: "巴黎奥运会"
batch_size: 4
method: "genQA"
```