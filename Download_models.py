from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import os
os.makedirs("./models/codet5", exist_ok=True)
os.makedirs("./models/embedding", exist_ok=True)
AutoTokenizer.from_pretrained("Salesforce/codet5-base").save_pretrained("./models/codet5")
AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").save_pretrained("./models/codet5")
SentenceTransformer("all-MiniLM-L6-v2").save("./models/embedding")
