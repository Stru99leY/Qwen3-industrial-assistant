import time
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class RerankerModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = "cuda", batch_size: int = 4, max_length: int = 8192):
        self.model_name = model_name
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"正在加载qwen3-reranker模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        print("Qwen3-Reranker模型加载成功")

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def format_instruction(self, query: str, doc: str) -> str:
        return f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        full_inputs = [self.prefix + pair + self.suffix for pair in pairs]
        inputs = self.tokenizer(full_inputs, padding='longest', truncation=True, max_length=self.max_length, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, query: str, documents: List, top_k: int | None = None) -> List:
        if not documents:
            return []
        start_time = time.time()
        texts = [doc.page_content for doc in documents]
        scores: List[float] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            pairs = [self.format_instruction(query, text) for text in batch_texts]
            inputs = self.process_inputs(pairs)
            batch_scores = self.compute_logits(inputs)
            scores.extend(batch_scores)

        for doc, score in zip(documents, scores):
            if 'metadata' not in doc.__dict__:
                doc.metadata = {}
            doc.metadata['reranker_score'] = float(score)

        sorted_documents = sorted(documents, key=lambda x: x.metadata.get('reranker_score', 0), reverse=True)
        if top_k is not None and top_k > 0:
            return sorted_documents[:top_k]
        return sorted_documents


