from transformers import AutoTokenizer, AutoModel

from Eval.device_settings import device

print("loading medbert")
tokeniser = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print("med bert:", model)
print("med bert num params:", num_params(model))


