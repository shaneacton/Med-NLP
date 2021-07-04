from typing import List

import torch
from torch import nn

from data_processor import tokenise_text, get_label_bools
from main import device
from medbert import model, tokeniser

FINE_TUNE_LAYERS = ["pooler", "embeddings", "LayerNorm"]
# FINE_TUNE_LAYERS = ["pooler", "LayerNorm"]


class BertBinaryClassifier(nn.Module):

    def __init__(self, medbert: nn.Module, tokeniser, bert_hidden_size, num_classes, num_cls_layers=2, dropout=0.2):
        super().__init__()
        self.tokeniser = tokeniser
        self.num_classes = num_classes
        medbert.eval()  # turn off dropout
        self.medbert = medbert
        for param in medbert.parameters():
            # print("med bert param:", param)
            param.requires_grad = False

        for n, m in medbert.named_modules():
            picked = False  # for the double continue

            for fine_tune in FINE_TUNE_LAYERS:
                if fine_tune not in n:
                    continue

                picked = True
                for param in m.parameters():
                    # print("med bert param:", param)
                    param.requires_grad = True

            if picked:
                print("fine tuning:", n)
            else:
                print("ignoring:", n)

        layers = []

        for i in range(num_cls_layers): # linear interp between hidd and classes
            in_size = int(bert_hidden_size + (num_classes - bert_hidden_size) * (i/num_cls_layers))
            out_size = int(bert_hidden_size + (num_classes - bert_hidden_size) * ((i+1)/num_cls_layers))
            layers.append(nn.Linear(int(in_size), int(out_size)))
            if i == num_cls_layers -1:  # last layer gets sigmoid to sqaush outputs into [0,1]
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))

        assert out_size == num_classes

        self.classifier = nn.Sequential(*layers)
        self.loss = nn.MSELoss()

    def classify(self, texts: List[str], labels: List[str]):
        tokens = tokenise_text(texts, self.tokeniser).view(1, -1)  #  ~ (b, len)
        target = get_label_bools(labels).view(1, -1)  #  ~ (b, len)
        print("tokens:", tokens.size(), "target:", target.size())
        return self(tokens, target)

    def forward(self, token_ids, labels=None):
        # print("forward got tokens:", token_ids.size(), ("labels: " + repr(labels.size())) if labels is not None else "")
        out = self.medbert(token_ids)["pooler_output"]
        # print("out:", out.size())
        # print("tokenised text:", tokeniser.decode(token_ids[0, :]))
        logits = self.classifier(out)
        predicted = torch.round(logits)
        # print("logits:", logits)
        # print("predicted:", predicted.size(), predicted, "\nlabel:", labels.size(), labels)
        # print("target:", labels)
        if labels is not None:
            loss = self.loss(logits, labels.float())
        else:
            loss = None
        return logits, predicted, loss

    def activate_layers(self, key):  # activates all layers which contain this key
        for n, m in self.medbert.named_modules():
            if key not in n:
                continue

            for param in m.parameters():
                # print("med bert param:", param)
                param.requires_grad = True


if __name__ == "__main__":
    cls = BertBinaryClassifier(model, tokeniser, 768, 30).to(device)
    print(cls)
    from dataloader import label_text, comment_text

    for i in range(len(label_text)):
        # print("label:", label_text[i], "text:", comment_text[i])
        cls.classify(comment_text[i], label_text[i])