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
from webui.css import CSS


class WebUI:
    def __init__(self):
        self.config = None
        self.api = None
        self.logger = Logger()
        self.loggerName = self.logger.getName()
        self.timer_active = False

    def load_config_from_file(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        else:
            return None

    def read_from_logs(self):
        if self.timer_active:
            with open(self.loggerName, "r", encoding="utf-8") as f:
                logs = f.read()
            return logs

    def read_from_configs(self, config_path):
        if config_path:
            config_dict = self.load_config_from_file(config_path)
            if config_dict is None:
                return f"Error: Could not load config file:{config_path}"

            api_config = config_dict.get("api", {})
            model = api_config.get("model")
            base_url = api_config.get("base_url")
            api_key = api_config.get("api_key")

            file_config = config_dict.get("file", {})
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
            save_dir = generation_config.get("save_dir", "./")
            question_prompt = generation_config.get("question_prompt")
            answer_prompt = generation_config.get("answer_prompt")
            max_nums = generation_config.get("max_nums", 1e6)

            rag_conf_dict = config_dict.get("rag", {})
            enable_rag = rag_conf_dict.get("enable_rag", False)
            rag_api_conf_dict = rag_conf_dict.get("api", {})

            rag_model_name = rag_api_conf_dict.get("model")
            rag_base_url = rag_api_conf_dict.get("base_url")
            rag_api_key = rag_api_conf_dict.get("api_key")

            return model, base_url, api_key, save_dir, file_path, file_folder, file_name, main_theme, concurrent_api_requests_num, method, file_types, is_structure_data, text_template, question_prompt, answer_prompt, max_nums, enable_rag, rag_model_name, rag_base_url, rag_api_key
        return "Error: Config path is empty"

    def run(self):
        with gr.Blocks(theme=gr.themes.Default(), css=CSS) as demo:
            gr.HTML("<h1><center>Distill-sft</center></h1>")
            with gr.Row():
                config_path = gr.Textbox(
                    label="Config Path[Optional]",
                    info="配置文件路径[可选]",
                    scale=2
                )
            with gr.Blocks():
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h2>Step 1: API Configuration</h2>")
                        with gr.Row():
                            model = gr.Textbox(
                                label="Model",
                                info="使用的api模型名称",
                                value="gpt-4o-mini",
                                max_lines=1,
                                scale=2
                            )
                            base_url = gr.Textbox(
                                label="Base Url",
                                info="api请求地址",
                                value="https://api.openai.com",
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
                    with gr.Column():
                        gr.HTML("<h2>Step 2: Saving Configuration</h2>")
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
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="gr-column"):
                    gr.HTML("<h2>Step 3: File Configuration(upload Files or Folder)</h2>")
                    with gr.Blocks():
                        with gr.Tab("Files"):
                            file_upload = gr.File(
                                label="Upload Files",
                                file_count="multiple",
                                scale=2
                            )
                            file_path = gr.Textbox(
                                label="File Path",
                                info="输入文本路径",
                                value="../example/dataset/Olympics.txt",
                                max_lines=3,
                                scale=2
                            )
                        with gr.Tab("Folder"):
                            folder_upload = gr.File(
                                label="Upload Folder",
                                file_count="directory",
                                scale=2
                            )
                            with gr.Row():
                                file_folder = gr.Textbox(
                                    label="File Folder",
                                    info="输入文件夹路径",
                                    max_lines=1,
                                    scale=2
                                )
                            with gr.Row():
                                file_types = gr.Dropdown(
                                    label="File Type",
                                    info="文件类型",
                                    multiselect=True,
                                    allow_custom_value=True,
                                    choices=["txt", "json", "md", "rst", "pdf", "word", ],
                                    scale=2
                                )
                        with gr.Row():
                            main_theme = gr.Textbox(
                                label="Main Theme",
                                info="文本主题",
                                max_lines=1,
                                scale=2,
                                visible=False
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
                with gr.Column():
                    gr.HTML("<h2>Step 4: Generation Configuration</h2>")
                    with gr.Blocks():
                        with gr.Row():
                            method = gr.Dropdown(
                                label="Method",
                                info="数据生成方式",
                                value='basic',
                                choices=["basic", "genQA", "backtranslation_rewrite", "VisGen"],
                                scale=2
                            )

                        with gr.Row():
                            question_prompt = gr.Textbox(
                                label="Question Prompt For Additional Requirement[Optional]",
                                info="问题生成提示词额外要求[可选]",
                                lines=3,
                                scale=2
                            )
                            answer_prompt = gr.Textbox(
                                label="Answer Prompt For Additional Requirement[Optional]",
                                info="答案生成提示词额外要求[可选]",
                                lines=3,
                                scale=2
                            )

                        with gr.Accordion("Quantity Control", open=False):
                            with gr.Row():
                                quantity_level = gr.Slider(
                                    label="Quantity Level",
                                    info="生成数据数量控制(1-5)",
                                    minimum=1,
                                    value=3,
                                    step=1,
                                    maximum=5
                                )
                                max_nums = gr.Slider(
                                    label="Max Nums",
                                    info="最大生成数据数量",
                                    minimum=1,
                                    step=1,
                                    value=10000,
                                    maximum=1e5,
                                    interactive=True
                                )
                        with gr.Accordion("Diversity Control", open=False):
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
                        with gr.Accordion("Quality Control", open=False):
                            with gr.Row():
                                verify_qa = gr.Checkbox(
                                    label="Verify QA",
                                    value=False,
                                    info="是否进行答案验证",
                                )
                        with gr.Accordion("RAG", open=False) as rag_accordion:
                            with gr.Row():
                                enable_rag = gr.Checkbox(
                                    label="Enable RAG",
                                    value=False,
                                    info="是否启用RAG",
                                    interactive=True
                                )
                            with gr.Row():
                                rag_model_name = gr.Textbox(
                                    label="Model",
                                    info="使用的api模型名称",
                                    max_lines=1,
                                    scale=2,

                                )
                            with gr.Row():
                                rag_base_url = gr.Textbox(
                                    label="RAG Base Url",
                                    info="api请求地址",
                                    max_lines=1,
                                    scale=2
                                )
                                rag_api_key = gr.Textbox(
                                    label="RAG Api Key",
                                    info="api密钥",
                                    max_lines=1,
                                    type="password",
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

            with gr.Row():
                display_config = gr.Textbox(
                    label="Config",
                    max_lines=20,
                    interactive=False,
                    visible=False
                )

            def update_config(*args):
                config_path, model, base_url, api_key, save_dir, file_path, file_folder, main_theme, \
                    concurrent_api_requests_num, method, file_types, file_name, is_structure_data, text_template, \
                    question_prompt, answer_prompt, max_nums, \
                    diversity_mode, temperature, verify_qa, enable_rag, rag_model_name, rag_base_url, rag_api_key = args

                config_dict = {
                    "api": {
                        "model": model,
                        "base_url": base_url,
                        "api_key": api_key
                    },
                    "file": {
                        "file_path": file_path,
                        "file_folder": file_folder,
                        "main_theme": main_theme,
                        "file_type": file_types if isinstance(file_types, list) else [file_types] if file_types else [],
                        "is_structure_data": is_structure_data,
                        "text_template": text_template
                    },
                    "generation": {
                        "method": method,
                        "concurrent_api_requests_num": int(
                            concurrent_api_requests_num) if concurrent_api_requests_num else 1,
                        "save_dir": save_dir,
                        "save_file_name": file_name,
                        "question_prompt": question_prompt,
                        "answer_prompt": answer_prompt,
                        "max_nums": max_nums,
                        "diversity_mode": diversity_mode,
                        "temperature": temperature,
                        "verify_qa": verify_qa
                    },
                    "rag": {
                        "enable_rag": enable_rag,
                        "api": {
                            "model": rag_model_name,
                            "base_url": rag_base_url,
                            "api_key": rag_api_key
                        }
                    }
                }
                return yaml.dump(config_dict, allow_unicode=True, sort_keys=False)

            with gr.Group():
                task_output = gr.Textbox(
                    label="LOG",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True,
                    show_copy_button=True
                )
                generate_button = gr.Button("Generate Dataset", variant="primary", elem_classes="generate-button")

            gr.HTML("<h2>Dataset Preview</h2>")
            display_data = gr.Dataframe(
                headers=["question", "answer"]
            )

            def read_from_datas(save_dir, file_name):
                if self.timer_active:
                    save_path = Path(save_dir) / Path(file_name)
                    print(f"save_path:{save_path}")
                    if not save_path.exists():
                        return pd.DataFrame(
                            columns=["question", "answer"]
                        )
                    else:
                        if save_path.is_file():
                            with open(save_path, 'r', encoding='utf-8') as f:
                                datas = json.load(f)
                            if datas:
                                df = pd.DataFrame(datas)
                                if 'input' in df.columns:
                                    del df['input']
                                df.rename(columns={'instruction': 'question', 'output': 'answer'}, inplace=True)
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
                    temperature, verify_qa,
                    enable_rag, rag_model_name, rag_base_url, rag_api_key
            ):
                try:
                    config_dict = {
                        "api": {
                            "model": model,
                            "base_url": base_url,
                            "api_key": api_key
                        },
                        "file": {
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
                            "save_dir": save_dir,
                            "save_file_name": file_name,
                            "question_prompt": question_prompt,
                            "answer_prompt": answer_prompt,
                            "diversity_mode": diversity_mode,
                            "quantity_level": quantity_level,
                            "temperature": temperature,
                            "verify_qa": verify_qa
                        },
                        "rag": {
                            "enable_rag": enable_rag,
                            "api": {
                                "model": rag_model_name,
                                "base_url": rag_base_url,
                                "api_key": rag_api_key
                            }
                        }
                    }
                    self.timer_active = True
                    self.config = Config(config_dict=config_dict)
                    gr.Info("配置已载入")

                    self.api = API(self.config)
                    Method = StrategyGetter.get_strategy(self.config.method)(self.api, self.config)
                    gr.Info("数据生成中")

                    questions, answers = await Method.run(
                        config=self.config,
                    )
                    gr.Info(f"数据生成完成。共生成 {len(questions)} 个问答对")
                    gr.Info(f"保存路径：{self.config.save_dir}/{self.config.save_file_name}")
                    self.logger.info(f"Generation Completed")
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    yield f"发生错误: {str(e)}"
                    gr.Error(str(e))

            for component in [
                config_path, model, base_url, api_key, save_dir,
                file_path, file_folder, main_theme, concurrent_api_requests_num,
                method, file_types, file_name, is_structure_data, text_template,
                question_prompt, answer_prompt,
                max_nums, diversity_mode, temperature, verify_qa,
                enable_rag, rag_model_name, rag_base_url, rag_api_key
            ]:
                component.change(
                    fn=update_config,
                    inputs=[
                        config_path, model, base_url, api_key, save_dir,
                        file_path, file_folder, main_theme, concurrent_api_requests_num,
                        method, file_types, file_name, is_structure_data, text_template,
                        question_prompt, answer_prompt, max_nums,
                        diversity_mode, temperature, verify_qa, enable_rag, rag_model_name, rag_base_url, rag_api_key
                    ],
                    outputs=display_config
                )

            timer = gr.Timer(0.5)

            def toggle_rag_fields(enable):
                return {
                    rag_accordion: gr.update(open=enable),
                }

            enable_rag.change(
                fn=toggle_rag_fields,
                inputs=[enable_rag],
                outputs=[rag_accordion]
            )

            timer.tick(self.read_from_logs, outputs=task_output)
            timer.tick(fn=read_from_datas, inputs=[save_dir, file_name], outputs=display_data)

            generate_button.click(
                fn=config_loader_and_run,
                inputs=[
                    config_path, model, base_url, api_key,
                    save_dir, file_path, file_folder, main_theme,
                    concurrent_api_requests_num, method, file_types,
                    file_name, is_structure_data, text_template,
                    question_prompt, answer_prompt,
                    diversity_mode, quantity_level,
                    temperature, verify_qa, enable_rag, rag_model_name, rag_base_url, rag_api_key
                ],
                outputs=[task_output],
                queue=True
            )

            config_path.submit(
                fn=self.read_from_configs,
                inputs=[config_path],
                outputs=[
                    model, base_url, api_key, save_dir, file_path, file_folder, file_name,
                    main_theme, concurrent_api_requests_num, method, file_types, is_structure_data,
                    text_template, question_prompt, answer_prompt, max_nums, enable_rag, rag_model_name, rag_base_url,
                    rag_api_key
                    # display_config
                ]
            )
            file_upload.change(fn=get_file_path, inputs=file_upload, outputs=file_path)
            folder_upload.change(fn=get_folder_path, inputs=folder_upload, outputs=file_folder)

        return demo


ui = WebUI()
demo = ui.run()


def run_webui():
    demo.launch()


if __name__ == '__main__':
    run_webui()
