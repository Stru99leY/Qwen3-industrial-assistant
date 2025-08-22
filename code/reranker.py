# reranker.py

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
import torch

class RerankerModel:
    """Reranker模型封装类，用于对检索结果进行重排序"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = "cuda", batch_size: int = 4, max_length: int = 8192):
        """初始化Reranker模型
        
        Args:
            model_name: 模型名称，默认使用Qwen/Qwen3-Reranker-0.6B
            device: 运行设备，默认使用CUDA
            batch_size: 批处理大小，默认为4
            max_length: 最大序列长度，默认为8192
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 如果没有CUDA，则使用CPU
        if self.device == "cuda" and not self._is_cuda_available():
            print("CUDA不可用，将使用CPU运行reranker模型")
            self.device = "cpu"
        
        try:
            print(f"正在加载qwen3-reranker模型: {model_name}...")
            # 按照官方示例配置tokenizer，设置padding_side为left
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            # 加载模型并设置为评估模式
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
            print(f"Qwen3-Reranker模型加载成功")
            
            # 获取yes和no的token id，用于计算相关性分数
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            print(f"yes token id: {self.token_true_id}, no token id: {self.token_false_id}")
            
            # 定义前缀和后缀
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            
            # 默认指令
            self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            
        except Exception as e:
            print(f"加载reranker模型时出错: {e}")
            raise
    
    def _is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def format_instruction(self, query: str, doc: str) -> str:
        """格式化指令、查询和文档
        
        Args:
            query: 用户查询
            doc: 文档内容
            
        Returns:
            格式化后的输入字符串
        """
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """处理输入对，添加前缀和后缀，并进行填充
        
        Args:
            pairs: 格式化后的输入字符串列表
            
        Returns:
            处理后的输入字典
        """
        # 将 prefix, pair, 和 suffix 组合成完整的输入字符串
        full_inputs = [self.prefix + pair + self.suffix for pair in pairs]
        
        # 使用 __call__ 方法进行 tokenization 和 padding
        # padding='longest' 会将批次中的序列填充到该批次中最长序列的长度，更节省内存
        inputs = self.tokenizer(
            full_inputs,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 将输入移动到指定设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            
        return inputs
        
    def compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """计算相关性分数
        
        Args:
            inputs: 处理后的输入字典
            
        Returns:
            相关性分数列表
        """
        with torch.no_grad():
            # 获取最后一个token的logits
            batch_scores = self.model(**inputs).logits[:, -1, :]
            
            # 提取yes和no的logits
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            # 堆叠logits并计算log softmax
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            
            # 返回yes的概率作为相关性分数
            scores = batch_scores[:, 1].exp().tolist()
            
        return scores
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 返回前k个文档，如果为None则返回所有文档
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            print("没有文档需要重排序")
            return []
        
        start_time = time.time()
        
        # 提取文档内容
        texts = [doc.page_content for doc in documents]
        scores = []
        
        # 使用qwen3-reranker模型计算相关性分数
        try:
            import torch
            
            print(f"开始处理 {len(texts)} 个文档，批处理大小: {self.batch_size}")
            
            # 批处理文档以提高效率
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                batch_size = len(batch_texts)
                print(f"处理批次 {i//self.batch_size + 1}，包含 {batch_size} 个文档")
                
                try:
                    # 格式化输入
                    pairs = [self.format_instruction(query, text) for text in batch_texts]
                    
                    # 处理输入
                    inputs = self.process_inputs(pairs)
                    
                    # 计算相关性分数
                    batch_scores = self.compute_logits(inputs)
                    
                    scores.extend(batch_scores)
                    print(f"批次 {i//self.batch_size + 1} 处理完成")
                except Exception as batch_error:
                    print(f"处理批次 {i//self.batch_size + 1} 时出错: {batch_error}")
                    # 如果批处理失败，尝试逐个处理文档
                    if batch_size > 1:
                        print("尝试逐个处理文档...")
                        for j, text in enumerate(batch_texts):
                            try:
                                # 单个文档处理使用相同的方法
                                pair = self.format_instruction(query, text)
                                single_input = self.process_inputs([pair])
                                single_score = self.compute_logits(single_input)
                                
                                scores.append(single_score[0])
                                print(f"文档 {i+j+1}/{len(texts)} 处理成功")
                            except Exception as single_error:
                                print(f"处理单个文档 {i+j+1}/{len(texts)} 时出错: {single_error}")
                                # 添加一个默认分数
                                scores.append(0.0)
                    else:
                        # 单个文档处理失败，添加默认分数
                        print(f"单个文档处理失败，添加默认分数")
                        scores.append(0.0)
            
            print(f"Qwen3-Reranker计算完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            print(f"Reranker计算分数时出错: {e}")
            return documents  # 出错时返回原始文档列表
        
        # 将分数添加到文档的metadata中
        for doc, score in zip(documents, scores):
            if 'metadata' not in doc.__dict__:
                doc.metadata = {}
            doc.metadata['reranker_score'] = float(score)
        
        # 根据分数排序
        sorted_documents = sorted(documents, key=lambda x: x.metadata.get('reranker_score', 0), reverse=True)
        
        # 返回前top_k个文档
        if top_k is not None and top_k > 0:
            return sorted_documents[:top_k]
        return sorted_documents

class RerankerRetriever(BaseRetriever):
    """结合向量检索和Reranker的检索器，继承自BaseRetriever以支持管道操作"""
    
    # 使用PrivateAttr来存储不需要被Pydantic验证的属性
    _vector_retriever = PrivateAttr()
    _reranker = PrivateAttr()
    _top_k_vector = PrivateAttr(default=20)
    _top_k_final = PrivateAttr(default=5)
    
    def __init__(self, vector_retriever, reranker: Optional[RerankerModel] = None, top_k_vector: int = 20, top_k_final: int = 5):
        """初始化检索器
        
        Args:
            vector_retriever: 向量检索器
            reranker: Reranker模型，如果为None则只使用向量检索
            top_k_vector: 向量检索返回的文档数量
            top_k_final: 最终返回的文档数量
        """
        super().__init__()
        self._vector_retriever = vector_retriever
        self._reranker = reranker
        self._top_k_vector = top_k_vector
        self._top_k_final = top_k_final
        
        # 记录reranker状态
        print(f"RerankerRetriever初始化，reranker模型: {'已加载' if reranker else '未加载/禁用'}")
        if reranker is None:
            print("注意：RerankerRetriever被初始化但reranker为None，将只使用向量检索")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """获取与查询相关的文档，实现BaseRetriever的抽象方法
        
        Args:
            query: 用户查询
            
        Returns:
            相关文档列表
        """
        # 第一阶段：向量检索
        start_time = time.time()
        vector_docs = self._vector_retriever.get_relevant_documents(query)
        vector_time = time.time() - start_time
        print(f"向量检索完成，找到 {len(vector_docs)} 个文档，耗时: {vector_time:.2f}秒")
        
        # 检查reranker是否为None
        if self._reranker is None:
            print("Reranker模型为None，跳过重排序，直接返回向量检索结果")
            return vector_docs[:self._top_k_final]  # 返回前top_k_final个文档
        
        # 第二阶段：使用reranker重排序
        try:
            start_time = time.time()
            reranked_docs = self._reranker.rerank(query, vector_docs, self._top_k_final)
            rerank_time = time.time() - start_time
            print(f"Reranker重排序完成，返回 {len(reranked_docs)} 个文档，耗时: {rerank_time:.2f}秒")
            return reranked_docs
        except Exception as e:
            print(f"Reranker重排序时出错: {e}，将返回原始向量检索结果")
            return vector_docs[:self._top_k_final]  # 出错时返回前top_k_final个文档

class AccuracyEvaluator:
    """准确率评估器，用于评估检索系统的准确率"""
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'relevant_retrieved': 0,
            'precision_sum': 0,
            'recall_sum': 0,
            'mrr_sum': 0,
            'ndcg_sum': 0
        }
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Document], 
                          relevant_docs: List[str] = None, relevance_scores: Dict[str, float] = None) -> Dict[str, float]:
        """评估检索结果的准确率
        
        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档列表
            relevant_docs: 相关文档ID列表，如果为None则使用人工评估
            relevance_scores: 文档ID到相关性分数的映射，用于计算NDCG
            
        Returns:
            评估指标字典
        """
        # 如果没有提供相关文档，则需要人工评估
        if relevant_docs is None:
            print("未提供相关文档列表，无法自动评估准确率")
            return {}
        
        # 获取检索到的文档ID
        retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
        
        # 计算相关文档被检索到的数量
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_docs)
        
        # 计算准确率指标
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                mrr = 1.0 / (i + 1)
                break
        
        # 计算NDCG (Normalized Discounted Cumulative Gain)
        ndcg = self._calculate_ndcg(retrieved_ids, relevance_scores) if relevance_scores else 0
        
        # 更新累计指标
        self.metrics['total_queries'] += 1
        self.metrics['relevant_retrieved'] += relevant_retrieved
        self.metrics['precision_sum'] += precision
        self.metrics['recall_sum'] += recall
        self.metrics['mrr_sum'] += mrr
        self.metrics['ndcg_sum'] += ndcg
        
        # 返回当前查询的评估结果
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mrr': mrr,
            'ndcg': ndcg
        }
    
    def _calculate_ndcg(self, retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int = None) -> float:
        """计算NDCG (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved_ids: 检索到的文档ID列表
            relevance_scores: 文档ID到相关性分数的映射
            k: 计算NDCG@k，如果为None则计算所有文档
            
        Returns:
            NDCG值
        """
        if not retrieved_ids or not relevance_scores:
            return 0.0
        
        # 限制k
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        # 计算DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 是因为log2(1)=0
        
        # 计算IDCG (理想DCG)
        ideal_ordering = sorted(relevance_scores.values(), reverse=True)
        if k is not None:
            ideal_ordering = ideal_ordering[:k]
        
        idcg = 0.0
        for i, rel in enumerate(ideal_ordering):
            idcg += rel / np.log2(i + 2)
        
        # 计算NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_overall_metrics(self) -> Dict[str, float]:
        """获取整体评估指标
        
        Returns:
            整体评估指标字典
        """
        total = self.metrics['total_queries']
        if total == 0:
            return {}
        
        return {
            'avg_precision': self.metrics['precision_sum'] / total,
            'avg_recall': self.metrics['recall_sum'] / total,
            'avg_mrr': self.metrics['mrr_sum'] / total,
            'avg_ndcg': self.metrics['ndcg_sum'] / total
        }
    
    def reset_metrics(self) -> None:
        """重置评估指标"""
        self.metrics = {
            'total_queries': 0,
            'relevant_retrieved': 0,
            'precision_sum': 0,
            'recall_sum': 0,
            'mrr_sum': 0,
            'ndcg_sum': 0
        }