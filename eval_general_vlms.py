from src.datasets import load_mimic
from src.datasets import load_mbrset
from src.datasets import load_medeval

from src.test import generate_predictions_models
from tqdm import tqdm
import pandas as pd

model_dict = {
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2_vl_7b",
    "llava-hf/llava-1.5-7b-hf": "llava_1_5_7b",
    
    #"llava-hf/llava-v1.6-mistral-7b-hf": "llava_1_6_mistral",
    #"llava-hf/llava-v1.6-vicuna-7b-hf": "llava_1_6_vicuna",
    #"llava-hf/llama3-llava-next-8b-hf": "llama3_llava_8b",
    #"google/paligemma2-10b-pt-224": 'paligemma2_10b',
    
    "deepseek-ai/Janus-Pro-7B": 'janus_pro_7b',
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama3_10b",
    
    "google/medgemma-4b-it": 'medgemma',
}

print("#"*100)
print("Evaluating MIMIC with General VLMs")
print("#"*100)


#metadata_test = load_mimic(train=False, validation=False,check_images=False)
#metadata_test = metadata_test.iloc[:3000]

# generate_predictions_models(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="mimic", store_columns=["dicom_id", "age", "sex", "race"], label="class_label", text_col="report", image_col="filepath", metadata_cols=["age", "sex", "race", "PerformedProcedureStepDescription", "ViewPosition"])
#generate_predictions_models(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="mimic", store_columns=["dicom_id", "age", "sex", "race"], label="class_label", text_col="report", image_col="filepath", metadata_cols=["age", "sex", "race", "PerformedProcedureStepDescription", "ViewPosition"], unmatched=True)


print("#"*100)
print("Evaluating VLMed with General VLMs")
print("#"*100)

metadata_train, metadata_val, metadata_test = load_medeval()

#generate_predictions_models(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma")
generate_predictions_models(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma", unmatched=True)



#print("#"*100)
#print("Evaluating mBRSET with Microsoft's LLava-Med model")
#print("#"*100)

#metadata_test = load_mbrset(train=False, validation=False,check_images=False)
#metadata_test = metadata_test.iloc[:5000]

#generate_predictions_models(model_dict, metadata_test, quantization="16b", return_attention=False, return_logits=False, dataset="mbrset", store_columns=["filepath", "age", "sex", "insurance", 'educational_level', 'alcohol_consumption', 'smoking', 'obesity'], label="final_icdr", image_col="filepath")
