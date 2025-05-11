import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax

# Configuraciones
MODEL_PATH = "./social_media_complaint_classifier"
TEXT_COLUMN = "comment_text"  # Cambiá esto si tu columna se llama distinto
DATA_PATH = "./data/comments_fb_clasificados.xlsx"  # Reemplazá por el nombre real

# Dataset personalizado
class CommentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

# Cargar modelo y tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Cargar datos
df = pd.read_excel(DATA_PATH)
df = df[df['reclamo'] == 1]
texts = df[TEXT_COLUMN].astype(str).tolist()
dataset = CommentDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=16)

# Realizar predicciones
predictions = []
probs = []

label_decoder = {
    0: "Arbolado Urbano",
    1: "Obras Públicas",
    2: "Higiene Urbana",
    3: "Alumbrado",
    4: "CLIBA (Higiene Urbana)"
}

results = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        for j, (pred, prob) in enumerate(zip(preds.cpu().numpy(), probs.cpu().numpy())):
            results.append({
                "text": texts[i * loader.batch_size + j],
                "predicted_category": label_decoder[pred],
                "confidence": float(max(prob)),  # Confianza de la clase elegida
                "prob_Arbolado Urbano": float(prob[0]),
                "prob_Obras Públicas": float(prob[1]),
                "prob_Higiene Urbana": float(prob[2]),
                "prob_Alumbrado": float(prob[3]),
                "prob_CLIBA (Higiene Urbana)": float(prob[4])
            })

# Agregar resultados al dataframe
results_df = pd.DataFrame(results)

# Guardar resultados
results_df.to_excel("./data/resultados_modelo.xlsx", index=False)
print("Predicciones guardadas en ./data/resultados_modelo.xlsx")
