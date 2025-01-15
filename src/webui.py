import os
import sys
import yaml
import gradio as gr
from model.config import Config
from api.api import API
from strategy.getter import StrategyGetter
from log.logger import Logger
from pathlib import Path
import time
import json
import pandas as pd
from multiprocessing import Process, Event

class WebUI:
    def __init__(self):
        self.config = None
        self.api = None
        self.logger = Logger()
        self.loggerName = self.logger.getName()
        # self.P = None
        
    def load_config_from_file(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        else:
            return None

    def read_from_logs(self):
        with open(self.loggerName, "r", encoding="utf-8") as f:
            logs = f.read()
        return logs
    
    def read_from_configs(self, config_path):
        if config_path:
            config_dict = self.load_config_from_file(config_path)
            print(config_dict)
            if config_dict is None:
                return f"Error: Could not load config file:{config_path}"
            
            api_config = config_dict.get("api", {})
            model = api_config.get("model")
            base_url = api_config.get("base_url")
            api_key = api_config.get("api_key")
            
            file_config = config_dict.get("file", {})
            save_dir = file_config.get("save_dir", "./")
            file_path = file_config.get("file_path")
            file_folder = file_config.get("file_folder")
            main_theme = file_config.get("main_theme", "")
            file_types = file_config.get("file_type", ["txt"])
            is_structure_data = file_config.get("is_structure_data", False)
            text_template = file_config.get("text_template")
            
            generation_config = config_dict.get("generation", {})
            method = generation_config.get("method")
            concurrent_api_requests_num = generation_config.get("concurrent_api_requests_num", 1)
            file_name = generation_config.get("save_file_name", "dataset.json")
            question_prompt = generation_config.get("question_prompt")
            answer_prompt = generation_config.get("answer_prompt")
            max_nums = generation_config.get("max_nums", 1e6)
            
            return model, base_url, api_key, save_dir, file_path, file_folder, file_name, main_theme, concurrent_api_requests_num, method, file_types, is_structure_data, text_template, question_prompt, answer_prompt, max_nums
        return "Error: Config path is empty"

    def run(self):
        with gr.Blocks() as demo:

            gr.Markdown("""## 配置设置""")
            config_path = gr.Textbox(
                label="Config Path",
                info="配置文件路径",
                scale=2
            )

            with gr.Column():
                with gr.Group():
                    gr.Markdown("""
                    ### Step 1: API Configuration
                    """)
                    with gr.Row():
                        model = gr.Textbox(
                            label="Model", 
                            info="使用的api模型名称",
                            max_lines=1,
                            scale=2
                        )
                    with gr.Row():
                        base_url = gr.Textbox(
                            label="Base Url", 
                            info="api请求地址",
                            max_lines=1,
                            scale=2
                        )
                        api_key = gr.Textbox(
                            label="Api Key", 
                            info="api密钥",
                            max_lines=1,
                            type="password",
                            scale=2
                        )
                
                with gr.Group():
                    gr.Markdown("### Step 2: File Configuration(upload Files or Folder)")
                    with gr.Tab("upload files"):
                        file_upload = gr.File(
                            label="Upload Files",
                            file_count="multiple",
                            scale=2
                        )
                        print("file_upload is ",file_upload)
                        file_path = gr.Textbox(
                            label="File Path", 
                            info="输入文本路径",
                            max_lines=3,
                            scale=2
                        )
                        
                    with gr.Tab("upload Folder"):
                        folder_upload = gr.File(
                            label="Upload Folder",
                            file_count="directory",
                            scale=2
                        )
                        file_folder = gr.Textbox(
                            label="File Folder", 
                            info="输入文件夹路径",
                            max_lines=1,
                            scale=2
                        )
                        file_types = gr.Dropdown(
                            label="File Type",
                            info="文件类型",
                            multiselect=True, 
                            allow_custom_value=True, 
                            choices=["txt", "json", "md", "rst","pdf","word",],
                            scale=2
                        )
                    
                    with gr.Row():
                        main_theme = gr.Textbox(
                            label="Main Theme", 
                            info="文本主题",
                            max_lines=1,
                            scale=2
                        )
                    with gr.Row():

                        is_structure_data = gr.Checkbox(
                            label="Is Structure Data",
                            show_label=True,
                            info="是否为 JSON 格式数据",
                            value=False,
                            scale=1
                        )
                        text_template = gr.Textbox(
                            label="Text Template",
                            info="JSON数据格式化模板",
                            lines=3,
                            scale=2
                        )
                
                with gr.Group():
                    gr.Markdown("### Step 3: Generation Configuration")
                    with gr.Row():
                        method = gr.Dropdown(
                            label="Method",
                            info="数据生成方式",
                            choices=["genQA", "backtranslation_rewrite"],
                            scale=2
                        )

                    with gr.Row(visible=True) as genqa_config:
                        question_prompt = gr.Textbox(
                            label="Question Prompt[Optional]",
                            info="问题生成提示词要求",
                            lines=3,
                            scale=2
                        )
                        answer_prompt = gr.Textbox(
                            label="Answer Prompt[Optional]",
                            info="答案生成提示词要求",
                            lines=3,
                            scale=2
                        )
                    
                    with gr.Row(visible=False) as backtrans_config:
                        question_prompt = gr.Textbox(
                            label="Question Prompt[Optional]",
                            info="问题生成提示词要求",
                            lines=3,
                            scale=2
                        )
                        answer_prompt = gr.Textbox(
                            label="Answer Prompt[Optional]",
                            info="答案生成提示词要求",
                            lines=3,
                            scale=2
                        )
                        # pass
                    
                    gr.Markdown("#### Quantity Control")
                    with gr.Row():
                        quantity_level = gr.Slider(
                            label="Quantity Level",
                            info = "生成数据数量控制(1-5)",
                            minimum=1,
                            value= 3,
                            step=1,
                            maximum=5
                        )
                            

                        max_nums = gr.Slider(
                            label="Max Nums",
                            info = "最大生成数据数量",
                            minimum=1,
                            step=1,
                            value=10000,
                            maximum=1e5,
                            interactive=True
                        )
                    
                    gr.Markdown("#### Diversity Control")
                    with gr.Row():
                        diversity_mode = gr.Radio(
                            label="Diversity Mode",
                            choices=["basic", "persona"],
                            value="basic",
                            scale=1
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0,
                            maximum=2,
                            value=1,
                            step=0.1,
                        )
                    gr.Markdown("#### Quality Control")
                    with gr.Row():
                        verify_qa = gr.Checkbox(
                            label="Verify QA",
                            value=False,
                            info="是否进行答案验证",
                        )
                    
                    with gr.Row():
                        save_dir = gr.Textbox(
                            label="Save Dir", 
                            info="保存文件目录",
                            max_lines=1,
                            scale=2
                        )
                        file_name = gr.Textbox(
                            label="File Name", 
                            info="保存文件名",
                            max_lines=1,
                            scale=2
                        )
                    with gr.Row():

                        concurrent_api_requests_num = gr.Number(
                            label="Concurrent Api Requests Num",
                            value=1,
                            info="api并发请求数",
                            minimum=1,
                            scale=1
                        )

                with gr.Group():
                    gr.Markdown("### config")
                    display_config = gr.Textbox(
                        label="Config",
                        max_lines=20,
                        interactive=False
                    )

                def update_config(*args):
                    config_path, model, base_url, api_key, save_dir, file_path, file_folder, main_theme, \
                    concurrent_api_requests_num, method, file_types, file_name, is_structure_data, text_template, \
                    question_prompt, answer_prompt, max_nums, \
                    diversity_mode, temperature, verify_qa = args
                    
                    config_dict = {
                        "api": {
                            "model": model,
                            "base_url": base_url,
                            "api_key": api_key
                        },
                        "file": {
                            "save_dir": save_dir,
                            "file_path": file_path,
                            "file_folder": file_folder,
                            "main_theme": main_theme,
                            "file_type": file_types if isinstance(file_types, list) else [file_types] if file_types else [],
                            "is_structure_data": is_structure_data,
                            "text_template": text_template
                        },
                        "generation": {
                            "method": method,
                            "concurrent_api_requests_num": int(concurrent_api_requests_num) if concurrent_api_requests_num else 1,
                            "save_file_name": file_name,
                            "question_prompt": question_prompt,
                            "answer_prompt": answer_prompt,
                            "max_nums": max_nums,
                            "diversity_mode": diversity_mode,
                            "temperature": temperature,
                            "verify_qa": verify_qa
                        }
                    }
                    return yaml.dump(config_dict, allow_unicode=True, sort_keys=False)

                with gr.Group():
                    gr.Markdown("### Output")
                    task_output = gr.Textbox(
                        label="Log Output",
                        lines=15,
                        max_lines=15,
                        interactive=False,
                        autoscroll=True,
                        show_copy_button=True
                    )
                    generate_button = gr.Button("Generate Dataset", variant="primary")
                    # stop_button = gr.Button("Stop", variant="danger")
                
                gr.Markdown("### Dataset preview")
                display_data = gr.Dataframe(
                    headers=["question", "answer"]
                )
                
                def read_from_datas(save_dir,file_name):
                    save_path = Path(save_dir)/ Path(file_name)
                    print("save_path is : ",save_path)
                    if not save_path.exists():
                        return pd.DataFrame(
                            columns=["question", "answer"]
                        )
                    else:
                        if save_path.is_file():
                            with open(save_path,'r',encoding='utf-8') as f:
                                datas = json.load(f)
                            if datas:
                                df = pd.DataFrame(datas)
                                if 'input' in df.columns:
                                    del df['input']
                                df.rename(columns={'instruction':'question','output':'answer'},inplace=True)
                                return df.head(100)                   
                            else:
                                return pd.DataFrame(
                                    columns=["question", "answer"]
                                )

                def get_file_path(uploaded_files):
                    if uploaded_files is None:
                        return ""
                    
                    if isinstance(uploaded_files, list):
                        paths = [file.name for file in uploaded_files]
                        return "\n".join(paths)
                    else:
                        return uploaded_files.name

                def get_folder_path(uploaded_folder):
                    if uploaded_folder is None:
                        return ""
                    
                    if isinstance(uploaded_folder, list):
                        paths = [file.name for file in uploaded_folder]
                        if paths:
                            common_prefix = os.path.commonpath(paths)
                            return common_prefix
                    return ""

                async def config_loader_and_run(
                    config_path, model, base_url, api_key, 
                    save_dir, file_path, file_folder, main_theme, 
                    concurrent_api_requests_num, method, file_types,
                    file_name, is_structure_data, text_template,
                    question_prompt, answer_prompt,
                    diversity_mode, quantity_level,
                    temperature, verify_qa
                ):
                    try:
                        config_dict = {
                            "api": {
                                "model": model,
                                "base_url": base_url,
                                "api_key": api_key
                            },
                            "file": {
                                "save_dir": save_dir,
                                "file_path": file_path,
                                "file_folder": file_folder,
                                "main_theme": main_theme,
                                "file_type": file_types if isinstance(file_types, list) else [file_types],
                                "is_structure_data": is_structure_data,
                                "text_template": text_template
                            },
                            "generation": {
                                "method": method,
                                "concurrent_api_requests_num": int(concurrent_api_requests_num),
                                "save_file_name": file_name,
                                "question_prompt": question_prompt,
                                "answer_prompt": answer_prompt,
                                "diversity_mode": diversity_mode,
                                "quantity_level": quantity_level,
                                "temperature": temperature,
                                "verify_qa": verify_qa
                            }
                        }
                        
                        self.config = Config(config_dict=config_dict)
                        # yield "配置已载入"
                        gr.Info("配置已载入")

                        self.api = API(self.config)
                        Method = StrategyGetter.get_strategy(self.config.method)(self.api,self.config)                        
                        # yield "数据生成中"
                        gr.Info("数据生成中")
                        
                        questions, answers = await Method.run(
                            config=self.config,
                        )                        
                        # yield f"数据生成完成。共生成 {len(questions)} 个问答对"
                        gr.Info(f"数据生成完成。共生成 {len(questions)} 个问答对")
                        gr.Info(f"保存路径：{self.config.save_dir}/{self.config.save_file_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error: {e}")
                        yield f"发生错误: {str(e)}"
                        gr.Error(str(e))

                for component in [
                    config_path, model, base_url, api_key, save_dir, 
                    file_path, file_folder, main_theme, concurrent_api_requests_num,
                    method, file_types, file_name, is_structure_data, text_template,
                    question_prompt, answer_prompt,
                    max_nums, diversity_mode, temperature, verify_qa
                ]:
                    component.change(
                        fn=update_config,
                        inputs=[
                            config_path, model, base_url, api_key, save_dir, 
                            file_path, file_folder, main_theme, concurrent_api_requests_num,
                            method, file_types, file_name, is_structure_data, text_template,
                            question_prompt, answer_prompt,  max_nums, 
                            diversity_mode, temperature, verify_qa
                        ],
                        outputs=display_config
                    )

                timer = gr.Timer(0.5)
                timer.tick(self.read_from_logs, outputs=task_output)
                timer.tick(fn=read_from_datas,inputs=[save_dir,file_name],outputs=display_data)
                
                generate_button.click(
                    fn=config_loader_and_run,
                    inputs=[
                        config_path, model, base_url, api_key, 
                        save_dir, file_path, file_folder, main_theme, 
                        concurrent_api_requests_num, method, file_types,
                        file_name, is_structure_data, text_template,
                        question_prompt, answer_prompt,
                        diversity_mode, quantity_level,
                        temperature, verify_qa
                    ],
                    outputs=[task_output],
                    queue=True
                )
                
                def toggle_visibility(selected_method):
                    if selected_method == "genQA":
                        return gr.update(visible=True), gr.update(visible=False)
                    elif selected_method == "backtranslation_rewrite":
                        return gr.update(visible=False), gr.update(visible=True)
                    else:
                        return gr.update(visible=False), gr.update(visible=False)

                method.change(
                    toggle_visibility,
                    inputs=[method],
                    outputs=[genqa_config, backtrans_config]
                )
                config_path.submit(
                    fn = self.read_from_configs,
                    inputs=[config_path],
                    outputs=[
                        model, base_url, api_key, save_dir, file_path, file_folder, file_name, 
                        main_theme, concurrent_api_requests_num, method, file_types, is_structure_data, 
                        text_template, question_prompt, answer_prompt, max_nums
                        # display_config
                    ]
                )

                method.change(
                    toggle_visibility,
                    inputs=[method],
                    outputs=[genqa_config, backtrans_config]
                )

                file_upload.change(fn=get_file_path, inputs=file_upload, outputs=file_path)
                folder_upload.change(fn=get_folder_path, inputs=folder_upload, outputs=file_folder)
        return demo


ui = WebUI()
demo = ui.run() 

def main():
    demo.launch()

if __name__ == '__main__':
    main()
