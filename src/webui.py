import os
import sys
import yaml
import gradio as gr
from model.config import Config
from api.api import API
from strategy.getter import StrategyGetter
from log.logger import setup_logger

class WebUI:
    def __init__(self):
        self.config = None
        self.api = None
        self.logger = setup_logger("output.log")
        
 
    def load_config_from_file(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        else:
            return None

    def read_from_logs(self):
        with open("output.log", "r", encoding="utf-8") as f:
            logs = f.read()
        return logs
    
    def read_from_configs(self, config_path):
        if config_path:
            config_dict = self.load_config_from_file(config_path)
            if config_dict is None:
                return f"Error: Could not load config file:{config_path}"
            model = config_dict.get("openai", {}).get("model")
            base_url = config_dict.get("openai", {}).get("base_url")
            api_key = config_dict.get("openai", {}).get("api_key")
            save_dir = config_dict.get("save_dir", "")
            file_path = config_dict.get("file_path", "")
            file_folder = config_dict.get("file_folder", "")
            file_name = config_dict.get("save_file_name", "dataset.json")
            main_theme = config_dict.get("main_theme", "")
            concurrent_api_requests_num = config_dict.get("concurrent_api_requests_num", 1)
            method = config_dict.get("method", "")
            file_types = config_dict.get("file_type", "txt").split()
            is_structure_data = config_dict.get("is_structure_data", False)
            text_template = config_dict.get("text_template", None)
            return model, base_url, api_key, save_dir, file_path, file_folder, file_name,main_theme, concurrent_api_requests_num, method, file_types, is_structure_data, text_template
        return "Error: Config path is empty"

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
                                       max_lines=1,
                                       type="password"
                                       )
                    save_dir = gr.Textbox(label="Save Dir", info="保存文件目录",
                                       max_lines=1)
                    file_path = gr.Textbox(label="File Path", info="文件源路径",
                                       max_lines=1)
                    file_folder = gr.Textbox(label="File Folder", info="文件源目录",
                                       max_lines=1)
                    
                    file_name = gr.Textbox(label="File Name", 
                                           info="保存文件名",
                                           max_lines=1
                                           )
                
                    main_theme = gr.Textbox(label="Main Theme", info="文本主题",
                                       max_lines=1)
                    concurrent_api_requests_num = gr.Number(
                        label="Concurrent Api Requests Num",
                        value=1,
                        info="api并发请求数",
                        minimum=1,
                        scale = 2
                    )
                    method = gr.Dropdown(
                        label="Method",
                        info="数据生成方式",
                        choices=["genQA", "genQA_persona", "backtranslation_rewrite"]
                    )

                    file_types = gr.Dropdown(
                        label="File Type",
                        info="文件类型",
                        multiselect=True, allow_custom_value=True, scale=2)
                
                    is_structure_data = gr.Checkbox(
                        label="Is Structure Data",
                        show_label=True,
                        info="是否为结构化JSON数据",
                        value=False
                    )
                
                    text_template = gr.Textbox(
                        label="Text Template",
                        info="JSON数据格式化模板",
                        lines=3,
                        scale=3
                    )
                    config_path.submit(self.read_from_configs,inputs=[config_path],outputs=[model, base_url, api_key, save_dir, file_path, file_folder, file_name,main_theme, concurrent_api_requests_num, method,file_types,is_structure_data,text_template])

                task_output = gr.Textbox(
                    label="输出",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    autoscroll = True,
                    show_copy_button = True,
                    
                )
                
                generate_button = gr.Button("Run")

                # def update_logs_periodically():
                #     return self.logger.getLogs()

                timer = gr.Timer(1)
                timer.tick(self.read_from_logs, outputs=task_output)

                async def config_loader_and_run(config_path, model, base_url, api_key, 
                                                save_dir, file_path, file_folder, main_theme, 
                                                concurrent_api_requests_num, method,file_types,file_name,
                                                is_structure_data,text_template
                                                ):
                    try:
                        self.logger = setup_logger("output.log")
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
                            "method": method,
                            "file_type": list(set(file_types)),
                            "save_file_name": file_name,
                            "is_structure_data": is_structure_data,
                            "text_template": text_template
                        }
                        self.config = Config(config_dict=config_dict)
                        self.api = API(self.config)
                        Method = StrategyGetter.get_strategy(self.config.method)(self.api)
                        questions, answers = await Method.run(
                            config=self.config,
                            num_question_per_title=3,
                            concurrent_api_requests_num=self.config.concurrent_api_requests_num
                        )
                        return
                    except Exception as e:
                        self.logger.error(f"Error: {e}")

                generate_button.click(
                    fn=config_loader_and_run,
                    inputs=[config_path, model, base_url, api_key, save_dir, 
                            file_path, file_folder, main_theme, 
                            concurrent_api_requests_num, method,
                            file_types,file_name,is_structure_data,text_template
                            ],
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
