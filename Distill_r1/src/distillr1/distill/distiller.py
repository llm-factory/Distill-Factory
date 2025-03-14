class Distiller():
    def __init__(self,distill_args,clients,questions,answers):
        self.distill_args = distill_args
        self.clients = clients
        self.questions = questions
        self.answers = answers
        self.reasoning_curation_args = distill_args.reasoning_curation_args
        self.answer_curation_args = distill_args.answer_curation_args
        self.chat_client = clients["chat"]
        if distill_args.enable_reward_model:
            self.reward_client = clients["reward"]
        
    async def distill(self):
        
        for question, answer in zip(self.questions, self.answers):
            question = question[0]['content']
            answer = answer[0]['content']
            llm_response = await self.clients["chat"].create_chat_from_message(self.distill_args.meta_prompt + question,self.distill_args.model_name_or_path) # TODO-> 
            llm_answer = llm_response.message.content
            llm_reason = llm_response.message.reasoning_content
            # TODO
            # for i in range(self.distill_args.max_try):
            #     judge_result = await self.clients["reward"].judge_answer_correctness(question, answer, llm_answer)
            #     logger.info_rank0(f"judge_result:{judge_result}\tquestion:{question}\t answer:{answer}\t")
            #     if judge_result:
            #         train_dataset_reasoner.append({
            #             "instruction": question,
            #             "input": "",
            #             "output": f"<think>\n{llm_reason}\n</think>\n\n{llm_answer}"
            #         })
            #         with open("test.json", "w",encoding='utf-8') as f:
            #             json.dump(train_dataset_reasoner,f,ensure_ascii=False,indent=2)      
            #         break