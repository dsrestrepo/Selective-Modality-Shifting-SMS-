import numpy as np
from PIL import Image
from transformers.image_utils import load_image
import torch
import pandas as pd

""" VALSE MIMIC-CXR """
CXR_VALSE_TEXT_PROMPT = lambda row, original=True: f"""
You are an expert chest-radiology assistant.

Radiology report (verbatim):
{row['report'] if original else row['modified_report']}

Question (answer very concisely):
{row['question']}

✦ Respond **only with the answer** required by the question – no extra words.
* In case that the report does not provide enough information to answer the question, respond with "uncertain".
* If the image and report disagree significantly with respect to the question, respond with "contradictory".
"""


CXR_VALSE_BINARY_TEXT_PROMPT_BINARY = lambda row, original=True: f"""
You are an expert chest-radiology assistant.

Radiology report (verbatim):
{row['report'] if original else row['modified_report']}
Question (The questions are always "yes" and "no questions, but you can additionally find "uncertain" and "contradictory" as answers. Nothing else.):
{row['question']}
✦ Respond **only with the answer** required by the question - no extra words.

Possible answers are:
- "yes" : if the answer is affirmative;
- "no" : if the answer is negative;
- "uncertain": if the report does not provide enough information to answer the question;
- "contradictory": if the report and image disagree significantly.
"""


CXR_FAKE_HISTORY_TEXT_PROMPT = lambda row, original=True: f"""
You are an expert chest-radiology assistant.

'Prior chest X-ray report:'
{row['prior_report'] if not original else ''}


Current chest X-ray report (verbatim):
{row['report']}

Based on the provided patient information and the associated chest X-ray image, does the patien has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?

✦ Respond **only with the answer** required – no extra words.

* Your response should be:
    - "yes" : if any condition is present and is supported by both image and the information in the report;
    - "no" : if the report and image are consistent and no condition is present;
    - "uncertain" → if the image and/or report do not provide enough information;
    - "contradictory" → if the image and report disagree significantly.
"""




""" Multimodality prompts for VLM bias benchmark """

GLAUCOMA_TEXT_PROMPT = lambda metadata_row: f"""
    Patient Information:
    - Age: {metadata_row['age']}
    - Gender: {metadata_row['gender']}
    - Race: {metadata_row['race']}
    - Language: {metadata_row['language']}
    - Marital Status: {metadata_row['maritalstatus']}
    - Ethnicity: {metadata_row['ethnicity']}
    - Summary: {metadata_row['note']}   
    Based on this image and patient information, does the patient have glaucoma? 
    Respond with "yes" or "no". (without additional commentary)
    """

# Define the text prompt for MIMIC-CXR.
MIMIC_TEXT_PROMPT = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Race: {metadata_row.get('race', 'Not Available')}
- ViewPosition: {metadata_row.get('ViewPosition', 'Not Available')}
- Procedure description: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
- Summary: {metadata_row.get('report', 'N/A')}

Based on the provided patient information and the associated chest X-ray image, does the patien has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?

Your response must be "Yes" if any condition is detected or "No" if no abnormality is detected. Your response should be only "yes" or "no" (without additional commentary).
"""

HAM10000_TEXT_PROMPT_FULL = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information and the dermoscopic image, please analyze the image and determine the diagnosis from the following list:
• akiec (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
• bcc (Basal cell carcinoma)
• bkl (Benign keratosis-like lesions, e.g., solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
• df (Dermatofibroma)
• mel (Melanoma)
• nv (Melanocytic nevi)
• vasc (Vascular lesions, e.g., angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

Your answer must be exactly one label from the above list: 'nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df'. Write it exactly like that and do not include any additional text or commentary.
"""

HAM10000_TEXT_PROMPT_BINARY = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information and the dermoscopic image, please analyze the image and determine the diagnosis from the following list:
• mel (Melanoma)
• nv (Melanocytic nevi)

Your answer must be exactly one label from the above list: 'nv' or 'mel'. Write it exactly like that and do not include any additional text or commentary.
"""


# Define the conversion functions
def convert_sex(sex):
    return 'male' if sex == 1 else 'female' if sex == 2 else 'no sex reported'

def convert_eye(eye):
    return 'right' if eye == 1 else 'left' if eye == 2 else 'no eye reported'

def convert_presence(presence):
    return 'present' if presence == 1 else 'absent'


# Define the text prompt for mBRSET.
BRSET_TEXT_PROMPT = lambda metadata_row: f"""
An image from the {convert_eye(metadata_row['exam_eye'])} eye of a {convert_sex(metadata_row['patient_sex'])} patient, 
aged {'no age reported' if pd.isnull(metadata_row['patient_age']) else str(float(str(metadata_row['patient_age']).replace('O', '0').replace(',', '.')))} years, 
{'with no comorbidities reported' if pd.isnull(metadata_row['comorbidities']) else 'with comorbidities: ' + metadata_row['comorbidities']}, 
{'with no diabetes duration reported' if pd.isnull(metadata_row['diabetes_time_y']) or metadata_row['diabetes_time_y'] == 'Não' else 'diabetes diagnosed for ' + str(float(str(metadata_row['diabetes_time_y']).replace('O', '0').replace(',', '.'))) + ' years'}, 
{'not using insulin' if metadata_row['insuline'] == 'no' else 'using insulin'}. 
The optic disc is {convert_presence(metadata_row['optic_disc'])}, vessels are {convert_presence(metadata_row['vessels'])}, 
and the macula is {convert_presence(metadata_row['macula'])}. 
Conditions include macular edema: {convert_presence(metadata_row['macular_edema'])}, scar: {convert_presence(metadata_row['scar'])}, 
nevus: {convert_presence(metadata_row['nevus'])}, amd: {convert_presence(metadata_row['amd'])}, vascular occlusion: {convert_presence(metadata_row['vascular_occlusion'])}, 
drusens: {convert_presence(metadata_row['drusens'])}, hemorrhage: {convert_presence(metadata_row['hemorrhage'])}, 
retinal detachment: {convert_presence(metadata_row['retinal_detachment'])}, myopic fundus: {convert_presence(metadata_row['myopic_fundus'])}, 
increased cup disc ratio: {convert_presence(metadata_row['increased_cup_disc'])}, and other conditions: {convert_presence(metadata_row['other'])}.

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)? 
Your response must be "Yes" if Diabetic Retinopathy is detected or "No" if Diabetic Retinopathy is not detected. Respond with "yes" or "no" (without additional commentary).
"""

mBRSET_TEXT_PROMPT = lambda metadata_row: f"""
An image from the {'right' if metadata_row.get('laterality') == 'right' else 'left' if metadata_row.get('laterality') == 'left' else 'unknown'} eye of a {'male' if metadata_row.get('sex') == 1 else 'female' if metadata_row.get('sex') == 0 else 'unknown sex'} patient, 
aged {'no age reported' if pd.isnull(metadata_row.get('age')) else str(metadata_row['age'])} years, 
{'with no comorbidities reported' if all(metadata_row.get(c) != 'yes' for c in ['systemic_hypertension', 'vascular_disease', 'acute_myocardial_infarction', 'nephropathy', 'neuropathy', 'diabetic_foot']) else 'with comorbidities: ' + ', '.join(
    filter(None, [
        'Systemic Hypertension' if metadata_row.get('systemic_hypertension') == 'yes' else None,
        'Vascular Disease' if metadata_row.get('vascular_disease') == 'yes' else None,
        'Acute Myocardial Infarction' if metadata_row.get('acute_myocardial_infarction') == 'yes' else None,
        'Nephropathy' if metadata_row.get('nephropathy') == 'yes' else None,
        'Neuropathy' if metadata_row.get('neuropathy') == 'yes' else None,
        'Diabetic Foot' if metadata_row.get('diabetic_foot') == 'yes' else None
    ])
)}, 
{'with no diabetes duration reported' if pd.isnull(metadata_row.get('dm_time')) else 'diabetes diagnosed for ' + str(metadata_row['dm_time']) + ' years'}, 
{'not using insulin' if metadata_row.get('insulin', 0) == 'no' else 'using insulin'}. 

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)?
Your response must be "Yes" if Diabetic Retinopathy is detected or "No" if Diabetic Retinopathy is not detected. Respond with "yes" or "no" (without additional commentary).
"""


""" ONLY IMAGE PROMPTS """

GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, does the patient have glaucoma?
Respond with "yes" or "no". (without additional commentary)
"""

MIMIC_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, does the patient has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?
Your response must be "Yes" if any condition is detected or "No" if no abnormality is detected. Your response should be only "yes" or "no" (without additional commentary).
"""

HAM10000_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, please analyze the image and determine the diagnosis from the following list:
• akiec (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
• bcc (Basal cell carcinoma)
• bkl (Benign keratosis-like lesions, e.g., solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
• df (Dermatofibroma)
• mel (Melanoma)
• nv (Melanocytic nevi)
• vasc (Vascular lesions, e.g., angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

Your answer must be exactly one label from the above list: 'nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df'. Write it exactly like that and do not include any additional text or commentary.
"""

mBRSET_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, does the patient has Diabetic Retinopathy (DR)?
Your response must be "Yes" if Diabetic Retinopathy is detected or "No" if Diabetic Retinopathy is not detected. Respond with "yes" or "no" (without additional commentary).
"""

mBRSET_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, does the patient has Diabetic Retinopathy (DR)?
Your response must be "Yes" if Diabetic Retinopathy is detected or "No" if Diabetic Retinopathy is not detected. Respond with "yes" or "no" (without additional commentary).
"""


""" Only text prompts """



GLAUCOMA_ONLY_TEXT_PROMPT = lambda metadata_row: f"""
    Patient Information:
    - Age: {metadata_row['age']}
    - Gender: {metadata_row['gender']}
    - Race: {metadata_row['race']}
    - Language: {metadata_row['language']}
    - Marital Status: {metadata_row['maritalstatus']}
    - Ethnicity: {metadata_row['ethnicity']}
    - Summary: {metadata_row['note']}   
    Based on this patient information, does the patient have glaucoma? 
    Respond with "yes" or "no". (without additional commentary)
    """

# Define the text prompt for MIMIC-CXR.
MIMIC_ONLY_TEXT_PROMPT = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Race: {metadata_row.get('race', 'Not Available')}
- ViewPosition: {metadata_row.get('ViewPosition', 'Not Available')}
- Procedure description: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
- Summary: {metadata_row.get('report', 'N/A')}

Based on the provided patient information, does the patien has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?

Your response must be "Yes" if any condition is detected or "No" if no abnormality is detected. Your response should be only "yes" or "no" (without additional commentary).
"""

HAM10000_ONLY_TEXT_PROMPT_FULL = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information, please analyze the image and determine the diagnosis from the following list:
• akiec (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
• bcc (Basal cell carcinoma)
• bkl (Benign keratosis-like lesions, e.g., solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
• df (Dermatofibroma)
• mel (Melanoma)
• nv (Melanocytic nevi)
• vasc (Vascular lesions, e.g., angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

Your answer must be exactly one label from the above list: 'nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df'. Write it exactly like that and do not include any additional text or commentary.
"""

HAM10000_ONLY_TEXT_PROMPT_BINARY = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information, please analyze the image and determine the diagnosis from the following list:
• mel (Melanoma)
• nv (Melanocytic nevi)

Your answer must be exactly one label from the above list: 'nv' or 'mel'. Write it exactly like that and do not include any additional text or commentary.
"""


# Define the text prompt for mBRSET.
BRSET_ONLY_TEXT_PROMPT = lambda metadata_row: f"""
An image from the {convert_eye(metadata_row['exam_eye'])} eye of a {convert_sex(metadata_row['patient_sex'])} patient, 
aged {'no age reported' if pd.isnull(metadata_row['patient_age']) else str(float(str(metadata_row['patient_age']).replace('O', '0').replace(',', '.')))} years, 
{'with no comorbidities reported' if pd.isnull(metadata_row['comorbidities']) else 'with comorbidities: ' + metadata_row['comorbidities']}, 
{'with no diabetes duration reported' if pd.isnull(metadata_row['diabetes_time_y']) or metadata_row['diabetes_time_y'] == 'Não' else 'diabetes diagnosed for ' + str(float(str(metadata_row['diabetes_time_y']).replace('O', '0').replace(',', '.'))) + ' years'}, 
{'not using insulin' if metadata_row['insuline'] == 'no' else 'using insulin'}. 
The optic disc is {convert_presence(metadata_row['optic_disc'])}, vessels are {convert_presence(metadata_row['vessels'])}, 
and the macula is {convert_presence(metadata_row['macula'])}. 
Conditions include macular edema: {convert_presence(metadata_row['macular_edema'])}, scar: {convert_presence(metadata_row['scar'])}, 
nevus: {convert_presence(metadata_row['nevus'])}, amd: {convert_presence(metadata_row['amd'])}, vascular occlusion: {convert_presence(metadata_row['vascular_occlusion'])}, 
drusens: {convert_presence(metadata_row['drusens'])}, hemorrhage: {convert_presence(metadata_row['hemorrhage'])}, 
retinal detachment: {convert_presence(metadata_row['retinal_detachment'])}, myopic fundus: {convert_presence(metadata_row['myopic_fundus'])}, 
increased cup disc ratio: {convert_presence(metadata_row['increased_cup_disc'])}, and other conditions: {convert_presence(metadata_row['other'])}.

Based on the provided patient information, does the patient has Diabetic Retinopathy (DR)? 
Your response must be "Yes" if Diabetic Retinopathy is detected or "No" if Diabetic Retinopathy is not detected. Respond with "yes" or "no" (without additional commentary).
"""

mBRSET_ONLY_TEXT_PROMPT = lambda metadata_row: f"""
An image from the {'right' if metadata_row.get('laterality') == 'right' else 'left' if metadata_row.get('laterality') == 'left' else 'unknown'} eye of a {'male' if metadata_row.get('sex') == 1 else 'female' if metadata_row.get('sex') == 0 else 'unknown sex'} patient, 
aged {'no age reported' if pd.isnull(metadata_row.get('age')) else str(metadata_row['age'])} years, 
{'with no comorbidities reported' if all(metadata_row.get(c) != 'yes' for c in ['systemic_hypertension', 'vascular_disease', 'acute_myocardial_infarction', 'nephropathy', 'neuropathy', 'diabetic_foot']) else 'with comorbidities: ' + ', '.join(
    filter(None, [
        'Systemic Hypertension' if metadata_row.get('systemic_hypertension') == 'yes' else None,
        'Vascular Disease' if metadata_row.get('vascular_disease') == 'yes' else None,
        'Acute Myocardial Infarction' if metadata_row.get('acute_myocardial_infarction') == 'yes' else None,
        'Nephropathy' if metadata_row.get('nephropathy') == 'yes' else None,
        'Neuropathy' if metadata_row.get('neuropathy') == 'yes' else None,
        'Diabetic Foot' if metadata_row.get('diabetic_foot') == 'yes' else None
    ])
)}, 
{'with no diabetes duration reported' if pd.isnull(metadata_row.get('dm_time')) else 'diabetes diagnosed for ' + str(metadata_row['dm_time']) + ' years'}, 
{'not using insulin' if metadata_row.get('insulin', 0) == 'no' else 'using insulin'}. 

Based on the provided patient information, does the patient has Diabetic Retinopathy (DR)?
Your response must be "Yes" if Diabetic Retinopathy is detected or "No" if Diabetic Retinopathy is not detected. Respond with "yes" or "no" (without additional commentary).
"""



    
def generate_text(model, processor, prompt, image, quantization=None, return_attention=False, return_logits=False, dataset=None, p_yes_and_no=False, row=None, unmatched=False):
    
    # Prepare the conversation prompt
    if (("llava" in model.config.model_type) or ("qwen" in model.config.model_type) or ("mllama" in model.config.model_type.lower())) and ( not("llava_med" in model.config._name_or_path.lower()) and not("llava-med" in model.config._name_or_path.lower())):
        if model.config.model_type == "mllama":
            if not unmatched:
                # Include in prompt the instructions for MLLama indicating the expected output
                prompt = prompt + f"Only respond with the answer (yes or no). No aditional commentary."
            else:
                prompt = prompt + f"Only respond with the answer (yes, no, or unmatched). No aditional commentary."            
        
        if image is None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
        else:
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

        # Apply chat template and prepare inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        if image is not None:
            inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        else:
            inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to("cuda")
        
        # Generate prediction
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.01, do_sample=False)
        
        
        output_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[-1]
        generated_ids = [output_ids[input_len:]]
        
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        
        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = outputs.scores[0][0]  # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)

                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:
                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()

                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                    
                
            return output_text[0], generated_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return output_text[0], generated_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None

    elif "paligemma" in model.config.model_type:
        if image is None:
            prompt = f"<bos>{prompt} Answer:"

        else:
            prompt = f"<image><bos>{prompt} Answer:"
            image = load_image(image)
        
        # PaliGemma model inference
        if quantization:
            model_inputs = processor(text=prompt, images=image, return_tensors="pt", do_convert_rgb=True).to(torch.bfloat16).to(model.device)
        else:
            model_inputs = processor(text=prompt, images=image, return_tensors="pt", do_convert_rgb=True).to(model.device)
            
        input_len = model_inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            outputs = model.generate(**model_inputs, max_new_tokens=512, do_sample=False, temperature=0.01)
            output_ids = outputs.sequences[0][input_len:]  # Ignore input tokens in the output
            output_text = processor.decode(output_ids, skip_special_tokens=True)
            
        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = outputs.scores[0][0]  # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)

                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:

                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                    
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
            
            return output_text, output_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return output_text, output_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None    
    
    # New branch for Janus models:
    elif "multi_modality" in model.config.model_type and "janus" in model.config._name_or_path.lower():
        # Build the conversation with the image placeholder and the question prompt.
        if image is None:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""}
            ]
        else:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"{prompt}"
                },
                {"role": "<|Assistant|>", "content": ""}
            ]
        
        # Import Janus image loader and load images from the conversation
        #from janus.utils.io import load_pil_images
        #pil_images = load_pil_images(conversation)

        if image is not None:
            image = load_image(image)
            pil_images = [image]
        else:
            pil_images = None
        
        # Prepare inputs using the Janus chat processor (assumed to be passed as 'processor')
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(model.device)
        
        # Prepare input embeddings from the Janus model 
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.01,
            do_sample=False,
            use_cache=True,
            output_attentions=return_attention,
            output_scores=return_logits,
            return_dict_in_generate=True
        )
        
        # Decode the full output.
        output_text = processor.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
        
        
        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = outputs.scores[0][0]  # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)

                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:
                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()


            return output_text, outputs.sequences[0], outputs.attentions if return_attention else None, outputs.scores if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return output_text, outputs.sequences[0], outputs.attentions if return_attention else None, outputs.scores if return_logits else None

    # BiomedGPT (OFA-based)
    elif ("ofa" in str(type(model)).lower()) or ("biomedgpt" in model.config._name_or_path.lower()):
        # The processor returns {"input_ids", "patch_images"}.
        inputs = processor(text=prompt, image=image)
        input_ids = inputs["input_ids"].to(model.device) if inputs["input_ids"] is not None else None
        patch_images = inputs["patch_images"].to(model.device) if inputs["patch_images"] is not None else None

        with torch.no_grad():
            gen = model.generate(
                input_ids,
                patch_images=patch_images,
                #num_beams=5,
                no_repeat_ngram_size=2,
                #max_length=128,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.01,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )
        
        
        raw_text = processor.tokenizer.batch_decode(gen['sequences'], skip_special_tokens=True)
        output_text = raw_text[0].strip()
        #output_text = re.sub(r'[^\w\s]', '', output_text)
        
        
        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = gen['scores'][0][0].cpu()  # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)

                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:
                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
            
        
        torch.cuda.empty_cache()
        del input_ids, patch_images
        
        # move generated sequences to CPU
        gen['sequences'] = gen['sequences'].cpu()
        gen['scores'] = gen['scores'][0].cpu() if return_logits else None
        
        
        if p_yes_and_no:
            return output_text, gen['sequences'][0], gen if return_attention else None, gen['scores'] if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return output_text, gen['sequences'][0], gen if return_attention else None, gen['scores'] if return_logits else None
    
    # 6) LLaVA-Med
    elif "llava-med" in model.config._name_or_path.lower() or "llava_med" in model.config._name_or_path.lower():
        # LLaVA-Med uses the custom processor that has:
        #    processor.apply_chat_template(...)
        #    processor.prepare_inputs(...)
        if image is None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
        if dataset == 'mimic':
            system_message = "You are an expert in radiology. Please analyze the image and text and provide an answer."
        if dataset == 'ham10000':
            system_message = "You are a dermatologist. Please analyze the image and text and provide an answer."
        if dataset == 'mbrset' or dataset == 'medeval' or dataset == 'brset':
            system_message = "You are an ophthalmologist. Please analyze the image and text and provide an answer."
        else:
            system_message = "You are an expert. Please analyze the image and text and provide an answer."
            
            
        text_prompt, conv = processor.apply_chat_template(conversation, assistant_message=None, system_message=system_message)#"A chat between a curious human and an artificial intelligence assistant. The assistant should concrete and accurate answers to the human's questions.")
        
        #print(f"Text prompt: {text_prompt}")
        
        inputs_dict = processor.prepare_inputs(text_prompt, image, conv)

        with torch.inference_mode():
            outputs = model.generate(
                inputs_dict["input_ids"],
                images=inputs_dict["images"],
                max_new_tokens=512,
                temperature=0.01,
                do_sample=False,
                use_cache=True,
                stopping_criteria=[inputs_dict["stopping_criteria"]],
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        
        output_ids = outputs.sequences[0]
        input_len = inputs_dict["input_ids"].shape[1]
        generated_ids = [output_ids[input_len:]]

        output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        
        # If output text is empty, print the generated_ids
        #if not output_text:
        #    print(f"Empty output text. Generated IDs: {generated_ids}")
        #else:
        #    print(f"Output text: {output_text}")
            
        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = outputs.scores[0][0]  # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)

                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:
                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                    
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()

            return output_text, generated_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return output_text, generated_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None   
        
    elif "gemma3" in model.config.model_type.lower():
        if image is None:
            messages = [
                {"role": "user",
                 "content": [{"type": "text", "text": prompt}]}
            ]
        else:
            messages = [
                {"role": "user",
                 "content": [
                     {"type": "text",  "text": prompt},
                     {"type": "image", "image": image}
                 ]}
            ]


        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16 if quantization in {"4b", "8b"} else None)

        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                #temperature=0.01,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )

            output_ids   = outputs.sequences[0][input_len:]
            output_text  = processor.decode(output_ids, skip_special_tokens=True).strip()

            
        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = outputs.scores[0][0]  # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)

                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:

                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                    
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
            
            return output_text, output_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return output_text, output_ids, outputs.attentions if return_attention else None, outputs.scores if return_logits else None    
    
    
    elif "chexagent" in model.config._name_or_path.lower() or "chexagent" in model.config.model_type.lower():
        # CheXagent’s tokenizer expects the “USER/ASSISTANT” wrapper.
        user_prompt = f" USER: <s>{prompt} ASSISTANT: <s>"
        img_list    = [image] if image is not None else None
        
        
        #import torch, psutil, humanize
        #print("CUDA allocated:",
        #    humanize.naturalsize(torch.cuda.memory_allocated(), binary=True))
        #print("CUDA reserved:",
        #    humanize.naturalsize(torch.cuda.memory_reserved(), binary=True))

        inputs = processor(images=img_list,
                           text=user_prompt,
                           return_tensors="pt"
                           ).to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                bos_token_id=1,
                eos_token_id=2,
                max_length=512,
                #num_beams=5,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.01,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )

            # CheXagent prepends the whole prompt; discard it from the decode
            out_text = processor.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )

        if p_yes_and_no:
            # Extract p("yes") and p("no") from the very first generation step
            p_yes, p_no = None, None
            p_Yes, p_No = None, None
            if return_logits:
                # outputs.scores is a list of length = number_of_generated_tokens
                # We want the first step’s distribution. shape: [batch_size, vocab_size]
                probs = outputs.scores[0][0]
                # [vocab_size], for batch=1
                #probs = first_step_scores.softmax(dim=0)
                # important: how does the tokenizer encode "yes"/"no"?
                # For GPT-like models, often " yes" with leading space is a single token.
                # For demonstration, we'll show the simplest approach:
                yes_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
                no_id  = processor.tokenizer("no",  add_special_tokens=False)["input_ids"]
                Yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
                No_id  = processor.tokenizer("No",  add_special_tokens=False)["input_ids"]
                # if each is a single token, we can do:
                if len(yes_id) == 1 and len(no_id) == 1:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                else:
                    print("Warning: 'yes'/'no' are not single tokens. Need to handle this case.")
                    # if "yes"/"no" get broken into multiple sub-tokens, you'd either
                    # sum the probabilities of each sub-token or pick a different strategy.
                    # For a quick check, we can just take the first sub-token:
                    p_yes = probs[yes_id[0]].item()
                    p_no  = probs[no_id[0]].item()
                    
                    p_Yes = probs[Yes_id[0]].item()
                    p_No  = probs[No_id[0]].item()
                
            return out_text, outputs.sequences[0], outputs.attentions if return_attention else None, outputs.scores if return_logits else None, p_yes, p_no, p_Yes, p_No
        else:
            return out_text, outputs.sequences[0], outputs.attentions if return_attention else None, outputs.scores if return_logits   else None

    elif "maira" in model.config._name_or_path.lower() or "maira" in model.config.model_type.lower():
        # TODO
        pass
    
    else:
        raise ValueError(f"Model type not supported: {model.config.model_type}, only 'qwen', 'llava', 'paligemma', 'janus', 'medgemma', 'chexagent', 'biomedgpt', 'llava-med', 'llava_med', and 'maira-2' models are supported.")
    


#### Zero-shot prompts for VLM bias benchmark
def predict_medeval(metadata_row, model, processor, quantization=None, return_attention=False, return_logits=False):
    """
    Predicts glaucoma based on image and text metadata using .
    """
    # Load image or .npz file
    data = np.load(metadata_row['filepath'])
    image = Image.fromarray(data['slo_fundus'])

    text_metadata = GLAUCOMA_TEXT_PROMPT(metadata_row)

    return generate_text(model, processor, text_metadata, image, quantization, return_attention, return_logits)


# Implement the prediction function for MIMIC-CXR.
def predict_dataset(metadata_row, model=None, processor=None, quantization=None, return_attention=False, return_logits=False, dataset='mimic', modality=None, only_prompt=False, p_yes_and_no=False, original=False, unmatched=False):
    """
    Predicts radiological findings from a MIMIC-CXR sample using both image and metadata.
    The function loads the image from the provided filepath, constructs a text prompt using
    MIMIC_TEXT_PROMPT, and then calls generate_text to perform inference.
    """
    
    #print(f"Modality: {modality}")
    
    # Construct the text prompt
    if dataset == 'mimic':
        # Load the image; for MIMIC, the 'filepath' points to the .jpg file.
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            #print("Only image")
            text_metadata = MIMIC_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            #print("Only text")
            text_metadata = MIMIC_ONLY_TEXT_PROMPT(metadata_row)
        else:
            #print("Both")
            text_metadata = MIMIC_TEXT_PROMPT(metadata_row)
    
    elif 'cxr_valse' in dataset:
        # Always use BOTH modalities (report + image)
        image = load_image(metadata_row['filepath'])
        if 'binary' in dataset:
            # TODO: Add binary prompts for CXR valse
            text_metadata = CXR_VALSE_BINARY_TEXT_PROMPT_BINARY(metadata_row, original=original)
        else:
            text_metadata = CXR_VALSE_TEXT_PROMPT(metadata_row, original=original)
        
    elif dataset == 'cxr_fake_history':
        image = load_image(metadata_row['filepath'])
        text_metadata = CXR_FAKE_HISTORY_TEXT_PROMPT(metadata_row, original=original)
        
        
    elif dataset == 'ham10000':
        # Load the image; for HAM10000, the 'filepath' points to the .jpg file.
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = HAM10000_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = HAM10000_ONLY_TEXT_PROMPT_BINARY(metadata_row)
        else:
            text_metadata = HAM10000_TEXT_PROMPT_BINARY(metadata_row)
        
    elif dataset == 'mbrset':
        # Load the image; for MBR-SET, the 'filepath' points to the .jpg file.
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = mBRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = mBRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = mBRSET_TEXT_PROMPT(metadata_row)
        
    elif dataset == 'medeval':
        # Load image or .npz file
        data = np.load(metadata_row['filepath'])
        image = Image.fromarray(data['slo_fundus'])
        if modality == 'Only_image':
            text_metadata = GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = GLAUCOMA_ONLY_TEXT_PROMPT(metadata_row)
        else:               
            text_metadata = GLAUCOMA_TEXT_PROMPT(metadata_row)
        
    else:
        raise ValueError(f"Dataset not supported: {dataset}, only 'mimic' and 'ham10000' datasets are supported.")
    

    if modality == 'Only_text':
        # Keep the text prompt, skip or pass None for the image
        image = None

    if only_prompt:
        return text_metadata, image
    
    if unmatched:
        text_metadata = text_metadata + f" If there is a mismatch between the image and the text, or the information is not conclusive respond with 'unmatched'. Otherwise respond with the options 'yes' or 'no'."
    
    # Call the general generate_text function (provided in your code) to obtain predictions.
    return generate_text(model, processor, text_metadata, image, quantization, return_attention, return_logits, dataset=dataset, p_yes_and_no=p_yes_and_no, row=metadata_row, unmatched=unmatched)
