import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from core.config import Emb_conf
class Embedder:
    def __init__(self, config: Emb_conf = Emb_conf()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)

        self.max_token = config.max_token


    def get_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].tolist()

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[i:i + batch_size], return_tensors="pt",
                                    padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            ## masked mean, plain mean averages pad tokens in
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            out.extend(emb.tolist())
        return out
