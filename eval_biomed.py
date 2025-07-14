from src.datasets import load_mimic
from src.datasets import load_mbrset
from src.datasets import load_medeval

from src.test import generate_predictions_models
from tqdm import tqdm
import pandas as pd

model_dict = {
    "PanaceaAI/BiomedGPT-Base-Pretrained": "biomedgpt"
}

print("#"*100)
print("Evaluating MIMIC with Biomed GPT model")
print("#"*100)


metadata_test = load_mimic(train=False, validation=False,check_images=False)
metadata_test = metadata_test.iloc[:10000]


generate_predictions_models(model_dict, metadata_test, quantization="16b", return_attention=False, return_logits=True, dataset="mimic", store_columns=["dicom_id", "age", "sex", "race"], label="class_label", text_col="report", image_col="filepath", metadata_cols=["age", "sex", "race", "PerformedProcedureStepDescription", "ViewPosition"])



print("#"*100)
print("Evaluating VLMed with Biomed GPT model")
print("#"*100)

metadata_train, metadata_val, metadata_test = load_medeval()

generate_predictions_models(model_dict, metadata_test, quantization="16b", return_attention=False, return_logits=True, dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma")


print("#"*100)
print("Evaluating mBRSET with Microsoft's LLava-Med model")
print("#"*100)

metadata_test = load_mbrset(train=False, validation=False,check_images=False)
metadata_test = metadata_test.iloc[:5000]

generate_predictions_models(model_dict, metadata_test, quantization="16b", return_attention=False, return_logits=False, dataset="mbrset", store_columns=["filepath", "age", "sex", "insurance", 'educational_level', 'alcohol_consumption', 'smoking', 'obesity'], label="final_icdr", image_col="filepath")

