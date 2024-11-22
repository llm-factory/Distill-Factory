import os
import sys
import yaml
import gradio as gr
import asyncio
from model.config import Config
from api.api import API
from strategy.getter import StrategyGetter
from tools.tool import save_QA_dataset
from log.logger import Logger

def read_logs():
    # print("Reading logs")
    if os.path.exists("output.log"):
        with open("output.log", "r", encoding='utf-8') as f:
            return f.read()
    return "No logs yet."

class WebUI:
    def __init__(self):
        self.config = None
        self.api = None
        sys.stdout = Logger("output.log")

    def load_config_from_file(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        else:
            return None

    def create_ui(self):
        with gr.Blocks() as demo:
            with gr.Column():
                config_path = gr.Textbox(
                    label="Config Path",
                    info="配置文件路径"
                )
                

                with gr.Row():
                    model = gr.Textbox(label="Model", 
                                       info="使用的api模型名称",
                                       max_lines=1
                                       )
                    base_url = gr.Textbox(label="Base Url", info="api请求地址",
                                       max_lines=1)
                    api_key = gr.Textbox(label="Api Key", info="api密钥",
                                       max_lines=1)
                    save_dir = gr.Textbox(label="Save Dir", info="保存文件路径",
                                       max_lines=1)
                    file_path = gr.Textbox(label="File Path", info="文件源路径",
                                       max_lines=1)
                    file_folder = gr.Textbox(label="File Folder", info="文件源目录",
                                       max_lines=1)
                    main_theme = gr.Textbox(label="Main Theme", info="文本主题",
                                       max_lines=1)
                    concurrent_api_requests_num = gr.Number(
                        label="Concurrent Api Requests Num",
                        value=1,
                        info="api并发请求数",
                    )
                    method = gr.Dropdown(
                        label="Method",
                        info="数据生成方式",
                        choices=["genQA", "genQA_persona", "backtranslation_rewrite"]
                    )

                task_output = gr.Textbox(
                    label="输出",
                    lines=10,
                    interactive=False
                )
                
                generate_button = gr.Button("Run")

                def update_logs_periodically():
                    logs = read_logs()
                    return logs

                timer = gr.Timer(2)
                timer.tick(update_logs_periodically, outputs=task_output)

                async def config_loader_and_run(config_path, model, base_url, api_key, 
                                                save_dir, file_path, file_folder, main_theme, 
                                                concurrent_api_requests_num, method):
                    try:
                        if config_path:
                            config_dict = self.load_config_from_file(config_path)
                            if config_dict is None:
                                return "Error: Could not load config file"
                        else:
                            config_dict = {
                                "openai": {
                                    "model": model,
                                    "base_url": base_url,
                                    "api_key": api_key
                                },
                                "save_dir": save_dir,
                                "file_path": file_path,
                                "file_folder": file_folder,
                                "main_theme": main_theme,
                                "concurrent_api_requests_num": int(concurrent_api_requests_num),
                                "method": method
                            }
                        
                        self.config = Config(config_dict)
                        self.api = API(self.config)
                        Method = StrategyGetter.get_strategy(self.config.method)(self.api)
                        questions, answers = await Method.run(
                            config=self.config,
                            num_question_per_title=5,
                            concurrent_api_requests_num=self.config.concurrent_api_requests_num
                        )
                        save_QA_dataset(questions, answers, self.config.save_dir, "test.json")
                        return f"Successfully generated {len(questions)} questions and answers"
                    
                    except Exception as e:
                        return f"Error: {str(e)}"

                generate_button.click(
                    fn=config_loader_and_run,
                    inputs=[config_path, model, base_url, api_key, save_dir, 
                            file_path, file_folder, main_theme, 
                            concurrent_api_requests_num, method],
                    outputs=None, 
                    queue=True
                )

        return demo

def main():
    ui = WebUI()
    demo = ui.create_ui()
    demo.launch()

if __name__ == '__main__':
    main()
