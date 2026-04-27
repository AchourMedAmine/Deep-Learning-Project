import torch.nn as nn 
from transformers import ASTModel 


class ASTClassifier(nn.Module): 
    def __init__(self,num_classes=4):
        super().__init__()
        self.ast=ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier=nn.Sequential(nn.Dropout(0.3),nn.Linear(768,num_classes))
    
    def forward(self,x):
        outputs=self.ast(x)
        embeddings=outputs.last_hidden_state.mean(dim=1)
        logits=self.classifier(embeddings)
        return logits