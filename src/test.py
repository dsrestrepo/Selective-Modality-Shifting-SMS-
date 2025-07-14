from src.models import load_qwen2_vl, load_llava, load_pali_gemma, load_janus_pro
from src.models import (
    load_qwen2_vl,
    load_llava,
    load_pali_gemma,
    load_janus_pro,
    load_biomedgpt,
    load_llava_med,
    load_maira2,
    load_medgemma,
    load_chexagent,
    load_llama3_2
)
from src.prompts import predict_medeval, predict_dataset

import os, re, json, glob
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers.image_utils import load_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

PRETTY_NAMES = {
    "qwen2_vl_7b":        "Qwen-2 VL",
    "llava_1_5_7b":       "LLaVA 1.5",
    "paligemma2_10b":     "PaliGemma-2",
    "janus_pro_7b":       "Janus-Pro",
    "biomedgpt":          "BiomedGPT",
    "llava_med_llava_v1": "LLaVA-Med",
    "medgemma":           "MedGemma",
    "llama3_10b":         "Llama-3.2",
    "llava_med_mistral_instruct": "LLaVA-Med"
}

def generate_predictions_models_base(model_dict, metadata_test, quantization=None, return_attention=True, return_logits=True, dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma", conv_mode="mistral_instruct", use_flash_attention=True):

    # Results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Main loop for processing models
    for model_id, model_name in tqdm(model_dict.items(), desc="Processing Models"):
        print(f"Loading model: {model_name} ({model_id})")
    
        # Load the model and processor
        if "qwen2" in model_name:
            model, processor = load_qwen2_vl(quantization=quantization, use_flash_attention=use_flash_attention, model_id=model_id, return_attention=return_attention, return_logits=return_logits)
        elif "llava" in model_name and not "llava_med" in model_name:
            model, processor = load_llava(model_id=model_id, quantization=quantization, use_flash_attention=use_flash_attention, return_attention=return_attention, return_logits=return_logits)
        elif "paligemma2" in model_name:
            model, processor = load_pali_gemma(model_id=model_id, quantization=quantization, return_attention=return_attention, return_logits=return_logits)
        elif "janus" in model_name:
            model, processor = load_janus_pro(model_id=model_id, quantization=quantization, return_attention=return_attention, return_logits=return_logits)
        elif "biomedgpt" in model_name:
            from src.models import load_biomedgpt
            model, processor = load_biomedgpt(model_id=model_id, quantization=quantization, 
                                              return_attention=return_attention, return_logits=return_logits)
        elif "llava_med" in model_name:
            from src.models import load_llava_med
            model, processor = load_llava_med(model_id=model_id, quantization=quantization, 
                                              return_attention=return_attention, return_logits=return_logits, conv_mode=conv_mode)
        elif "maira-2" in model_name:
            from src.models import load_maira2
            model, processor = load_maira2(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits,
            )
        elif "medgemma" in model_name:
            from src.models import load_medgemma
            model, processor = load_medgemma(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits,
            )
        elif "chexagent" in model_name:
            from src.models import load_chexagent
            model, processor, _ = load_chexagent(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits,
            )
        elif "llama3" in model_name:
            from src.models import load_llama3_2
            model, processor = load_llama3_2(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits
            )
        else:
            raise ValueError(f"Model type not supported: {model_name}, only 'qwen2', 'llava', 'paligemma2', 'janus', 'biomedgpt', 'llava_med', maira 2, medgemma, and chexagent, are supported")
    
        # Predict and save results
        results = []
        for index, row in tqdm(metadata_test.iterrows(), total=len(metadata_test), desc=f"Predicting with {model_name}"):

            original_ops = [True, False]
            
            for original in original_ops:
                if (not original) and ('valse' not in dataset) and ('fake_history' not in dataset):
                    continue
                
                prediction, _, _, _ = predict_dataset(row, model=model, processor=processor, quantization=quantization, return_attention=return_attention, return_logits=return_logits, dataset=dataset, original=original)

                if index % 100 == 0:
                    print(f"File: {row['filepath']}, Prediction: {prediction}")
                
                aux_dict = {key: row[key] for key in store_columns}
                aux_dict["ground_truth"] = row[label]
                aux_dict["prediction"] = prediction
                aux_dict['original'] = original
                results.append(aux_dict)

        # Convert results to a DataFrame and save
        results_df = pd.DataFrame(results)
        model_output_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        output_csv_path = os.path.join(model_output_dir, f"{dataset}_base.csv")
        results_df.to_csv(output_csv_path, index=False)

        print(f"Predictions for {model_name} saved to {output_csv_path}")
        # Clear model from memory
        del model, processor
        torch.cuda.empty_cache()
        



    
    
import os
import pandas as pd
import torch
from tqdm import tqdm
import random

def get_shifted_image(current_filepath, current_label, metadata, label_field, image_col="filepath"):
    """
    Returns the filepath of an image from `metadata` that has a label different from `current_label`.
    If no candidate is found, returns the original filepath.
    """
    # Filter metadata for rows with a different label than current_label
    candidates = metadata[metadata[label_field] != current_label]
    if candidates.empty:
        print(f"No candidate found for {current_filepath} with label {current_label}")
        return current_filepath
    else:
        # Randomly sample one candidate image and return its filepath
        return candidates.sample(1).iloc[0][image_col]


def get_shifted_text(current_row, current_label, metadata, label_field, text_col=None):
    """
    Returns a shifted text prompt by selecting a random row from `metadata`
    whose label (in the field `label_field`) differs from that of `current_row`.
    """

    # Filter metadata for rows with a different label than current_label
    candidates = metadata[metadata[label_field] != current_label]
    if candidates.empty:
        print(f"No candidate found for {current_row} with label {current_label}")
        return current_row
    else:
        # Randomly sample one candidate row and return its text
        return candidates.sample(1).iloc[0]#[text_col]
    
def get_shifted_metadata(current_row, current_label, metadata, label_field, metadata_cols=["age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"]):
    
    # Filter metadata for rows with a different label than current_label
    candidates = metadata[metadata[label_field] != current_label]
    if candidates.empty:
        print(f"No candidate found for {current_row} with label {current_label}")
        return current_row
    else:
        # Randomly sample one candidate row and return its text
        return candidates.sample(1).iloc[0][metadata_cols]

def generate_predictions_models(model_dict, metadata_test, quantization=None, return_attention=False, return_logits=False, use_flash_attention=False,
                                  dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"],
                                  text_col="note", image_col="filename", metadata_cols=["age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma", conv_mode="mistral_instruct",
                                  p_yes_and_no=True, unmatched=False):
    
    
    # Results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Loop over models
    for model_id, model_name in tqdm(model_dict.items(), desc="Processing Models"):
        print(f"Loading model: {model_name} ({model_id})")
    
        # Load the model and processor based on model type
        if "qwen2" in model_name:
            model, processor = load_qwen2_vl(quantization=quantization, use_flash_attention=use_flash_attention, model_id=model_id, 
                                             return_attention=return_attention, return_logits=return_logits)
        elif "llava" in model_name and (not ("llava_med" in model_name) and not("llava-med" in model_name)):
            model, processor = load_llava(model_id=model_id, quantization=quantization, use_flash_attention=use_flash_attention, 
                                           return_attention=return_attention, return_logits=return_logits)
        elif "paligemma2" in model_name:
            model, processor = load_pali_gemma(model_id=model_id, quantization=quantization, 
                                               return_attention=return_attention, return_logits=return_logits)
        elif "janus" in model_name:
            model, processor = load_janus_pro(model_id=model_id, quantization=quantization, 
                                              return_attention=return_attention, return_logits=return_logits)
        elif "biomedgpt" in model_name:
            from src.models import load_biomedgpt
            model, processor = load_biomedgpt(model_id=model_id, quantization=quantization, 
                                              return_attention=return_attention, return_logits=return_logits)
        elif "llava_med" in model_name or "llava-med" in model_name:
            from src.models import load_llava_med
            print("Using llava-med")
            model, processor = load_llava_med(model_id=model_id, quantization=quantization, 
                                              return_attention=return_attention, return_logits=return_logits, conv_mode=conv_mode)
        elif "maira-2" in model_name:
            from src.models import load_maira2
            model, processor = load_maira2(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits,
            )
        elif "medgemma" in model_name:
            from src.models import load_medgemma
            model, processor = load_medgemma(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits,
            )
        elif "chexagent" in model_name:
            from src.models import load_chexagent
            model, processor, _ = load_chexagent(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits,
            )
        elif "llama3" in model_name:
            from src.models import load_llama3_2
            model, processor = load_llama3_2(
                model_id=model_id,
                quantization=quantization,
                return_attention=return_attention,
                return_logits=return_logits
            )
        else:
            raise ValueError(f"Model type not supported: {model_name}, only 'qwen2', 'llava', 'paligemma2', 'janus', 'biomedgpt', 'llava_med', maira 2, medgemma, and chexagent, are supported")
    
        results = []
        # For each sample in the test set
        for index, row in tqdm(metadata_test.iterrows(), total=len(metadata_test), desc=f"Predicting with {model_name}"):
            
            # None (original), Image shift, Metadata shift, and Text shift.
            for shift_type in [None, "Image", "Text", "Only_text", "Only_image"]: #, "Metadata"]:
                # Create a copy of the row so as not to modify the original
                row_modified = row.copy() if hasattr(row, "copy") else dict(row)
                
                # Modify the row depending on the shift
                if shift_type == "Image":
                    # Replace the image with one from a different class.
                    # Use the helper function to ensure the new image has a different label.
                    row_modified[image_col] = get_shifted_image(row[image_col], row[label], metadata_test, label, image_col=image_col)
                    
                elif shift_type == "Text":
                    # Replace the text prompt with one from a different class.
                    # Adjust this as necessary based on your dataset’s structure.
                    row_modified = get_shifted_text(row, row[label], metadata_test, label, text_col=text_col)
                    
                    # Replace the image with the original image.
                    row_modified[image_col] = row[image_col]

                # Paligemma2 does not support text-only inputs
                if "paligemma2" in model_name and shift_type == "Only_text":
                    continue
                # Janus does not support text-only inputs
                if "janus" in model_name and shift_type == "Only_text":
                    continue
                    
                #elif shift_type == "Metadata":
                    # Replace the metadata with one from a different class.
                    # Adjust this as necessary based on your dataset’s structure.
                    #row_modified[metadata_cols] = get_shifted_metadata(row, row[label], metadata_test, label, metadata_cols=metadata_cols)
                                    
                try:
                    # Get prediction using the modified row
                    prediction, _, _, _, p_yes, p_no, p_Yes, p_No = predict_dataset(row_modified, model=model, processor=processor, quantization=quantization, 
                                                            return_attention=return_attention, return_logits=return_logits, dataset=dataset, modality=shift_type, p_yes_and_no=p_yes_and_no, unmatched=unmatched)#, 
                                                            #text_col=text_col, image_col=image_col, metadata_cols=metadata_cols)
                except Exception as e:
                    print(f"Error predicting with {model_name} on index {index}: {e}")
                    # skip this row if there's an error
                    torch.cuda.empty_cache()
                    continue
                
                if index % 100 == 0:
                    print(f"File: {row_modified.get('filepath', 'N/A')}, Shift: {shift_type or 'None'}, Prediction: {prediction}")
                
                # Build a dictionary for this evaluation instance
                aux_dict = {key: row.get(key, None) for key in store_columns}
                aux_dict["ground_truth"] = row[label]
                aux_dict["prediction"] = prediction
                # First token probability
                aux_dict["p_yes"] = p_yes
                aux_dict["p_no"] = p_no
                aux_dict["p_Yes"] = p_Yes
                aux_dict["p_No"] = p_No
                # Record shift type (None means no modification)
                aux_dict["shift"] = "No" if shift_type is None else shift_type
                results.append(aux_dict)
                
            # Print torch cuda memory
            #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            #print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
            #print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
            #print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
            #print("\n")
                
        # Convert results to a DataFrame and save
        results_df = pd.DataFrame(results)
        model_output_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        if unmatched:
            output_csv_path = os.path.join(model_output_dir, f"{dataset}_base_shifted_unmatched.csv")
        else:
            output_csv_path = os.path.join(model_output_dir, f"{dataset}_base_shifted.csv")
        results_df.to_csv(output_csv_path, index=False)
    
        print(f"Predictions for {model_name} saved to {output_csv_path}")
        # Clear model from memory
        del model, processor
        torch.cuda.empty_cache()

    
    
    
    
### Evaluation:

# Patterns for 'no' cases
no_patterns = [
    r'\bno evidence\b',
    r'\bno abnormalities\b',
    r'\bno significant change\b',
    r'\bno focal consolidation\b',
    r'\bno acute cardiopulmonary process\b',
    r'\bwithin normal limits\b',
    r'\bclear lungs\b',
    r'\bno pneumothorax\b',
    r'\bno pleural effusion\b',
    r'\bdoes not (have|show|reveal)\b',
    r'\bappears to be normal\b',
    r'\bdoes not appear to have glaucoma\b',
    r'\bno signs of glaucoma\b',
    r'\bnormal intraocular pressure\b',
    r'\bno thinning retinal nerve fiber layer\b',
    r'\bfull visual fields\b'
]

# Patterns for 'yes' cases (indicating glaucoma)
yes_patterns = [
    r'\bglaucoma\b',
    r'\bglaucoma suspect\b',
    r'\bprimary open-angle glaucoma\b',
    r'\bnarrow angle glaucoma\b',
    r'\bborderline glaucoma\b',
    r'\boptic nerve damage\b',
    r'\belevated intraocular pressure\b',
    r'\bincreased cup:disc ratio\b',
    r'\bcupping of the optic nerve\b',
    r'\bvisual field defects\b',
    r'\boptic nerve head damage\b',
    r'\bhistory of glaucoma\b',
    r'\bglaucoma diagnosis\b'
    r'\bpleural effusion\b', 
    r'\bpulmonary edema\b', 
    r'\bpneumothorax\b', 
    r'\bcardiomegaly\b', 
    r'\bconsolidation\b', 
    r'\binfiltrate\b', 
    r'\bopacification\b', 
    r'\binfection\b', 
    r'\bpneumonia\b'
]


# Clean predictions and ground truth
def clean_label(label):
    label = str(label).lower().strip().replace('.', '')
    
    if label in ['yes', 'y']:
        label = 'yes'
    elif label in ['no', 'n']:
        label = 'no'
        
    # Pattern-based classification
    elif any(re.search(pattern, label) for pattern in no_patterns):
        label = 'no'
    elif any(re.search(pattern, label) for pattern in yes_patterns):
        label = 'yes'
        
    elif re.search(r'\bdoes not (have|show|reveal)\b', label):
        label = 'no'
    elif re.search(r'\bappears to be normal\b', label):
        label = 'no'
        
    elif ('does not have' in label) or ('does not show' in label):
        label = 'no'

    elif label.startswith('no ') or label.startswith('no,'):
        label = 'no'
    elif label.startswith('yes ') or label.startswith('yes,'):
        label = 'yes'
    else:
        label = f'unknown, response: {label}'
    return label

# Clean predictions and ground truth
def clean_label_missmatch(label):
    label = str(label).lower().strip().replace('.', '')
    
    if label in ['yes', 'y']:
        label = 'yes'
    elif label in ['no', 'n']:
        label = 'no'
    elif label in ['unmatched', 'unmatch', 'unmatched response', 'unmatched response:', 'missmatch', 'missmatch', 'unmatched response:']:
        label = 'unmatched'
        
    # Pattern-based classification
    elif any(re.search(pattern, label) for pattern in no_patterns):
        label = 'no'
    elif any(re.search(pattern, label) for pattern in yes_patterns):
        label = 'yes'
        
    elif re.search(r'\bdoes not (have|show|reveal)\b', label):
        label = 'no'
    elif re.search(r'\bappears to be normal\b', label):
        label = 'no'
        
    elif ('does not have' in label) or ('does not show' in label):
        label = 'no'

    elif label.startswith('no ') or label.startswith('no,'):
        label = 'no'
    elif label.startswith('yes ') or label.startswith('yes,'):
        label = 'yes'
    else:
        label = f'unknown, response: {label}'
    return label





def calculate_metrics(y, y_pred, show_unknown_responses=False, show_probs=False):
    
    # Check 'unknown' responses
    unknown_responses = y_pred[y_pred.str.startswith('unknown')].unique()
    if len(unknown_responses) > 0:
        print(f"Total unknown responses: {len(unknown_responses)}")
        if show_unknown_responses:
            print(f"Unknown responses: {unknown_responses}")
    
    
    # Exclude unknown responses
    print(f"Excluding {len(y[y_pred.str.startswith('unknown')])} unknown responses out of {len(y)}")
    y = y[~y_pred.str.startswith("unknown")]
    y_pred = y_pred[~y_pred.str.startswith("unknown")]

    if show_probs:
    
        print(f"Precited probability of Condition=Yes|prompt,image: ", y_pred.value_counts(normalize=True))
        print(f"Precited probability of Condition=No|prompt,image: ", 1 - y_pred.value_counts(normalize=True))
        
        print(f"Actual probability of Condition=Yes|prompt,image: ", y.value_counts(normalize=True))
        print(f"Actual probability of Condition=No|prompt,image: ", 1 - y.value_counts(normalize=True))
    
    
    # Calculate metrics (accuracy, precision, recall, F1, sensitivity, specificity)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='yes', zero_division=np.nan)
    recall = recall_score(y, y_pred, pos_label='yes', zero_division=np.nan)
    f1 = f1_score(y, y_pred, pos_label='yes', zero_division=np.nan)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=['no', 'yes']).ravel()
    
    # AUC score
    # convert to binary labels
    y = (y == 'yes').astype(int)
    y_pred = (y_pred == 'yes').astype(int)
    
    try:
        auc = roc_auc_score(y, y_pred)
    except ValueError:
        auc = np.nan
    
    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    
    return accuracy, precision, recall, f1, sensitivity, specificity, auc

import torch
import torch.nn.functional as F

def compute_uncertainty(df):
    """
    Computes entropy and cross-entropy from the logits.
    Adds:
    - entropy_lower: entropy from [p_yes, p_no]
    - entropy_upper: entropy from [p_Yes, p_No]
    - ce_lower: cross-entropy from [p_yes, p_no]
    - ce_upper: cross-entropy from [p_Yes, p_No]
    """
    def compute_row(row):
        logits_lower = torch.tensor([row["p_yes"], row["p_no"]], dtype=torch.float32)
        logits_upper = torch.tensor([row["p_Yes"], row["p_No"]], dtype=torch.float32)

        probs_lower = F.softmax(logits_lower, dim=0)
        probs_upper = F.softmax(logits_upper, dim=0)

        entropy_lower = -torch.sum(probs_lower * torch.log(probs_lower + 1e-12)).item()
        entropy_upper = -torch.sum(probs_upper * torch.log(probs_upper + 1e-12)).item()

        target = 1 if row["ground_truth"] == "yes" else 0
        ce_lower = F.cross_entropy(logits_lower.unsqueeze(0), torch.tensor([target])).item()
        ce_upper = F.cross_entropy(logits_upper.unsqueeze(0), torch.tensor([target])).item()

        return pd.Series({
            "entropy_lower": entropy_lower,
            "entropy_upper": entropy_upper,
            "ce_lower": ce_lower,
            "ce_upper": ce_upper
        })

    df[["entropy_lower", "entropy_upper", "ce_lower", "ce_upper"]] = df.apply(compute_row, axis=1)
    return df

    
from sklearn.calibration import calibration_curve


def plot_calibration(df, prob_col, label_col, title="Calibration", n_bins=10, out_path=None):
    y_true = (df[label_col] == "yes").astype(int)
    y_prob = df[prob_col]

    # Get calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Bin counts so we can weight ECE properly
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(y_prob, bins=bin_edges)
    non_empty = bin_counts > 0
    counts_kept = bin_counts[non_empty]

    # Expected Calibration Error
    ece = np.sum((counts_kept / len(y_true)) *
                 np.abs(prob_true - prob_pred))

    # Plot reliability diagram
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    
    return ece


def plot_all_shifts_calibration(
    model_key,
    dataset="medeval",
    base_dir="results",
    pretty_names=None,
    n_bins=10,
    p_yes=False
):
    plt.figure(figsize=(6,6))
    shifts = {
        "No": "No Shift",
        "Image": "Img Shift",
        "Text": "Txt Shift",
        "Only_text": "Only Text",
        "Only_image": "Only Image"
    }
    df = pd.read_csv(os.path.join(base_dir, model_key, f"{dataset}_base_shifted.csv"))
    
    # Clean the ground truth
    df["ground_truth"] = df["ground_truth"].apply(clean_label)
    
    # Normalize using softmax
    import torch
    import torch.nn.functional as F
    
    if p_yes:
        logits = torch.tensor(df[['p_yes','p_no']].values)
        probs  = F.softmax(logits, dim=1).numpy()
        df['p_yes'], df['p_no'] = probs[:,0], probs[:,1]
        df['pred_first_token_yes'] = np.where(df['p_yes'] > df['p_no'], 'yes', 'no')
    else:
        logits = torch.tensor(df[['p_Yes','p_No']].values)
        probs  = F.softmax(logits, dim=1).numpy()
        df['p_Yes'], df['p_No'] = probs[:,0], probs[:,1]
        df['pred_first_token_yes'] = np.where(df['p_Yes'] > df['p_No'], 'yes', 'no')
    
    df_original = df.copy()
    
    
    for shift_code, shift_label in shifts.items():
        
        df = df_original[df_original["shift"] == shift_code]

        # if you need first_token probs, apply your softmax logic here
        y_true = (df["ground_truth"] == "yes").astype(int)
        y_prob = df["p_yes"] if p_yes else df["p_Yes"]
        
        # Get calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )
            plt.plot(prob_pred, prob_true, marker='o', label=shift_label)
        except ValueError as e:
            print(f"Error in calibration for {shift_label}: {e}")
            continue
    plt.plot([0,1],[0,1], '--', color='gray')
    name = pretty_names.get(model_key, model_key) if pretty_names else model_key
    plt.title(f"{name} Calibration ({dataset})")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.legend()
    os.makedirs("images", exist_ok=True)
    out_path = f"images/{model_key}_{dataset}_calibration.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()
    return out_path



def calculate_metrics_dataset(results_dir, subgroup_variables=["gender", "race", "language", "maritalstatus"], counterfactual=False, confusion_matrix=False, first_token=False, show_unknown_responses=False, unmatched=False, calibration=False, p_yes=False, save_calibration_plot=False):
    # Load the results
    df = pd.read_csv(results_dir)
        
    # Clean the labels
    if "mimic" or "medeval" in results_dir:
        df["ground_truth"] = df["ground_truth"].apply(clean_label)
        if not first_token:
            df["prediction"] = df["prediction"].apply(clean_label) if not unmatched else df["prediction"].apply(clean_label_missmatch)
    elif "ham10000" in results_dir:
        df["ground_truth"] = df["ground_truth"].apply(lambda x: x.lower())
        if not first_token:
            df["prediction"] = df["prediction"].apply(lambda x: x.lower()) 
            # mel = melanoma = yes, nv = nevus = no
        df["ground_truth"] = df["ground_truth"].apply(lambda x: "yes" if x == "mel" or x == 'melanoma' else "no")
        print(f'For simplicity we are considering "mel" and "melanoma" as "yes" and "nv" and "nevus" as "no"')
    

    # remove unknown responses
    if not first_token:
        df = df[~df["prediction"].str.startswith("unknown")]
        # print unknown responses
        #unique_unknown = df["prediction"][df["prediction"].str.startswith("unknown")].unique()
        #if len(unique_unknown) > 0:
        #    print(f"Total unknown responses: {len(unique_unknown)}")
        #    print(f"Unknown responses: {unique_unknown}")
        
    
    # Get first token probability
    if first_token:
        # Normalize using softmax
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(df[['p_yes','p_no']].values)
        probs  = F.softmax(logits, dim=1).numpy()
        df['p_yes'], df['p_no'] = probs[:,0], probs[:,1]
        
        #df["p_yes"] = df["p_yes"] / (df["p_yes"] + df["p_no"])
        #df["p_no"] = 1 - df["p_yes"]
        
        logits = torch.tensor(df[['p_Yes','p_No']].values)
        probs  = F.softmax(logits, dim=1).numpy()
        df['p_Yes'], df['p_No'] = probs[:,0], probs[:,1]
        
        #df["p_Yes"] = df["p_Yes"] / (df["p_Yes"] + df["p_No"])
        #df["p_No"] = 1 - df["p_Yes"]
        
        df['pred_first_token_yes'] = np.where(df['p_yes'] > df['p_no'], 'yes', 'no')
        
        df['pred_first_token_Yes'] = np.where(df['p_Yes'] > df['p_No'], 'yes', 'no')
        
        df = compute_uncertainty(df)
        

    df_original = df.copy()
    #print(f"dataframe: {df_original}")
    
    if unmatched and not first_token:
        # Calculate proportion of unmatched responses by shift type
        unmatched_counts = df_original[df_original["prediction"].str.startswith("unmatched")].groupby("shift").size()
        print(f"Unmatched responses by shift type:\n{unmatched_counts}")
        
        # Calculate the number of correct unmatched predictions ("Image", "Text") should be unmatched
        correct_unmatched = df_original[(df_original["shift"].isin(["Image", "Text"])) & (df_original["prediction"].str.startswith("unmatched"))].shape[0]
        print(f"Number of correct unmatched predictions: {correct_unmatched}")
        # Calculate the number of incorrect unmatched predictions ("Image", "Text") should not be unmatched
        incorrect_unmatched = df_original[(df_original["shift"].isin(["Image", "Text"])) & (~df_original["prediction"].str.startswith("unmatched"))].shape[0]
        print(f"Accuracy on unmatched predictions: {incorrect_unmatched}")
        
        # remove unmatched responses from the dataframe
        df_original = df_original[~df_original["prediction"].str.startswith("unmatched")]
        
    # prepare output dict for this dataset
    shift_metrics = {}

    for shift_type in ["No", "Image", "Text", "Only_text", "Only_image"]: #, "Metadata"]:          
        
        if counterfactual:  
            print(40 * "=" + f" Metrics for {shift_type} Shift " + 40 * "=")
            
            df = df_original[df_original["shift"] == shift_type]
        else:
            df = df_original
            
        
        # Print the confusion matrix
        if confusion_matrix:
            if not first_token:
                print(40 * "=" + f" Confusion Matrix for {shift_type} Shift " + 40 * "=")
                print(pd.crosstab(df["ground_truth"], df["prediction"], rownames=["Actual"], colnames=["Predicted"]))
            else:
                if p_yes:
                    print(40 * "=" + f" Confusion Matrix for First Token Probability (Lower) for {shift_type} Shift " + 40 * "=")
                    print(pd.crosstab(df["ground_truth"], df["pred_first_token_yes"], rownames=["Actual"], colnames=["Predicted"]))
                else:
                    print(40 * "=" + f" Confusion Matrix for First Token Probability (Upper) for {shift_type} Shift " + 40 * "=")
                    print(pd.crosstab(df["ground_truth"], df["pred_first_token_Yes"], rownames=["Actual"], colnames=["Predicted"]))        
        
        if not first_token:
            # Overall metrics:
            accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(df["ground_truth"], df["prediction"], show_unknown_responses=show_unknown_responses)
            
            print(40 * "=" + " Overall Metrics " + 40 * "=")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            #print(f"F1: {f1:.4f}")
            #print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            
        if first_token:
            if p_yes:
                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(df["ground_truth"], df["pred_first_token_yes"], show_unknown_responses=False)
                print(40 * "=" + " Overall Metrics for First Token Probability yes " + 40 * "=")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                #print(f"F1: {f1:.4f}")
                #print(f"Sensitivity: {sensitivity:.4f}")
                print(f"Specificity: {specificity:.4f}")
                
                entropy_mean = df["entropy_lower"].mean()
                entropy_std = df["entropy_lower"].std()
                ce_mean = df["ce_lower"].mean()
                ce_std = df["ce_lower"].std()

                print(f"Mean Entropy: {entropy_mean:.4f} ± {entropy_std:.4f}")
                print(f"Mean Cross-Entropy: {ce_mean:.4f} ± {ce_std:.4f}")
                
                if calibration:
                    try:
                        if save_calibration_plot:
                            out_path = f'images/calibration_example_{shift_type}.png'
                        else:
                            out_path = None
                        ece = plot_calibration(df, prob_col="p_yes", label_col="ground_truth", title=f"Calibration: {shift_type} p_yes", out_path=out_path)
                        print(f"Expected Calibration Error (ECE): {ece:.4f}")
                    except:
                        ece = np.nan
                        print(f"Error plotting calibration for {shift_type} p_yes")
            else:
                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(df["ground_truth"], df["pred_first_token_Yes"], show_unknown_responses=False)
                print(40 * "=" + " Overall Metrics for First Token Probability Yes " + 40 * "=")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                #print(f"F1: {f1:.4f}")
                #print(f"Sensitivity: {sensitivity:.4f}")
                print(f"Specificity: {specificity:.4f}")
                
                entropy_mean = df["entropy_upper"].mean()
                entropy_std = df["entropy_upper"].std()
                ce_mean = df["ce_upper"].mean()
                ce_std = df["ce_upper"].std()
                
                print(f"Mean Entropy: {entropy_mean:.4f} ± {entropy_std:.4f}")
                print(f"Mean Cross-Entropy: {ce_mean:.4f} ± {ce_std:.4f}")
                
                if calibration:
                    try:
                        if save_calibration_plot:
                            out_path = f'images/calibration_example_{shift_type}.png'
                        else:
                            out_path = None
                        ece = plot_calibration(df, prob_col="p_Yes", label_col="ground_truth", title=f"Calibration: {shift_type} p_Yes", out_path=out_path)
                        print(f"Expected Calibration Error (ECE): {ece:.4f}")
                    except:
                        ece = np.nan
                        print(f"Error plotting calibration for {shift_type} p_Yes")
        
        shift_name = {
            "No": "No Shift",
            "Image": "Img Shift",
            "Text": "Txt Shift",
            "Only_text": "Only Text",
            "Only_image": "Only Image"
        }[shift_type]

        shift_metrics[shift_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "ECE": ece if calibration else None,
        }
        
        # Group by demographic and calculate metrics
        for subgroup in subgroup_variables:
            
            # Check if is countinuous, if so split into 3 categories
            if df[subgroup].dtype in [int, float]:
                
                print(f"Splitting {subgroup} into 3 categories: Low ({df[subgroup].min()} - {df[subgroup].quantile(0.33)}), Medium ({df[subgroup].quantile(0.33)} - {df[subgroup].quantile(0.66)}), High ({df[subgroup].quantile(0.66)} - {df[subgroup].max()})")
                df[subgroup] = pd.qcut(df[subgroup], q=3, labels=["low", "medium", "high"])
            
            print(40 * "=" + f" Metrics by {subgroup.capitalize()} " + 40 * "=")
            for subgroup_value in df[subgroup].unique():
                subgroup_df = df[df[subgroup] == subgroup_value]
                if len(subgroup_df) == 0:
                    continue
                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(subgroup_df["ground_truth"], subgroup_df["prediction"])
                print(f"{subgroup.capitalize()}: {subgroup_value}")
                print(f"Accuracy: {accuracy:.4f}")
                #print(f"Precision: {precision:.4f}")
                #print(f"Recall: {recall:.4f}")
                print(f"F1: {f1:.4f}")
                #print(f"Sensitivity: {sensitivity:.4f}")
                #print(f"Specificity: {specificity:.4f}")
                #print(f"AUC: {auc:.4f}")
            print("\n")

        if not counterfactual:
            break
        
    return shift_metrics
        
# Implement def calculate_metrics_mimic
def calculate_metrics_all_models(model_dict, subgroup_variables=["gender", "race", "language", "maritalstatus"], results_dir="results", dataset="medeval", counterfactual=False, confusion_matrix=False, first_token=False, show_unknown_responses=False, unmatched=False, calibration=False, p_yes=False, save_calibration_plot=False):
    all_results = {}
    for model_id, model_name in model_dict.items():
        print(90 * "=" )
        print(40 * "=" + f" Metrics for {model_name} " + 40 * "=")
        print(90 * "=" )
        if counterfactual:
            if unmatched:
                metrics = calculate_metrics_dataset(os.path.join(results_dir, model_name, f"{dataset}_base_shifted_unmatched.csv"), subgroup_variables=subgroup_variables, counterfactual=counterfactual, confusion_matrix=confusion_matrix, first_token=first_token, show_unknown_responses=show_unknown_responses, unmatched=unmatched, calibration=calibration, p_yes=p_yes, save_calibration_plot=save_calibration_plot)
            else:
                metrics = calculate_metrics_dataset(os.path.join(results_dir, model_name, f"{dataset}_base_shifted.csv"), subgroup_variables=subgroup_variables, counterfactual=counterfactual, confusion_matrix=confusion_matrix, first_token=first_token, show_unknown_responses=show_unknown_responses, calibration=calibration, p_yes=p_yes, save_calibration_plot=save_calibration_plot)
        else:
            metrics = calculate_metrics_dataset(os.path.join(results_dir, model_name, f"{dataset}_base.csv"), subgroup_variables=subgroup_variables, counterfactual=counterfactual, confusion_matrix=confusion_matrix, first_token=first_token, show_unknown_responses=show_unknown_responses, calibration=calibration, p_yes=p_yes, save_calibration_plot=save_calibration_plot)
        print("\n\n")
        
        all_results[model_name] = metrics
        
        if calibration and first_token:
            img_path = plot_all_shifts_calibration(
                model_key=model_name,
                dataset=dataset,
                base_dir=results_dir,
                pretty_names=PRETTY_NAMES,
                n_bins=10,
                p_yes=p_yes
            )
    
    all_results = {PRETTY_NAMES.get(key, key): val for key, val in all_results.items()}
            
    return all_results




##############################################################################
# Negative Flip Rate (NFR) function
##############################################################################

def compute_nfr_table(df: pd.DataFrame, id: str = 'filename', first_token: bool = False, p_yes: bool =False) -> pd.DataFrame:
    """
    Given a DataFrame from *_base_shifted.csv, return a DataFrame of NFR values
    for Image, Text, Only_text, Only_image shifts.
    """
    df = df.copy()
    df["ground_truth"] = df["ground_truth"].apply(clean_label)
    if first_token:
        # Normalize using softmax
        import torch
        import torch.nn.functional as F
        if p_yes:
            # p_yes and p_no
            df['p_yes'] = df['p_yes'].astype(float)
            df['p_no'] = df['p_no'].astype(float)
            logits = torch.tensor(df[['p_yes','p_no']].values)
            probs  = F.softmax(logits, dim=1).numpy()
            df['p_yes'], df['p_no'] = probs[:,0], probs[:,1]
            df['prediction'] = np.where(df['p_Yes'] > df['p_No'], 'yes', 'no')
        else:        
            logits = torch.tensor(df[['p_Yes','p_No']].values)
            probs  = F.softmax(logits, dim=1).numpy()
            df['p_Yes'], df['p_No'] = probs[:,0], probs[:,1]    
            df['prediction'] = np.where(df['p_yes'] > df['p_no'], 'yes', 'no')
    else:
        df["prediction"] = df["prediction"].apply(clean_label)
        df = df[~df["prediction"].str.startswith("unknown")]

    if id not in df.columns or "shift" not in df.columns:
        print(df.columns)
        raise ValueError(f"Expected columns: '{id}' and 'shift'")

    base_df = df[df["shift"] == "No"]
    shifts = ["Image", "Text", "Only_text", "Only_image"]

    records = []
    for shift in shifts:
        pert_df = df[df["shift"] == shift]
        merged = base_df[[id, "ground_truth", "prediction"]].rename(
            columns={"prediction": "base_pred"}
        ).merge(
            pert_df[[id, "prediction"]],
            on=id, how="inner"
        ).rename(columns={"prediction": "pert_pred"})

        base_correct = merged["base_pred"] == merged["ground_truth"]
        pert_wrong = merged["pert_pred"] != merged["ground_truth"]
        n_flips = (base_correct & pert_wrong).sum()
        nfr = n_flips / len(merged) if len(merged) > 0 else np.nan

        records.append({"Shift": shift.replace("_", " ").title(), "NFR": nfr})

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Plot NFR values for a model
# -----------------------------------------------------------------------------
def plot_nfr(nfr_df: pd.DataFrame, model_name: str, save_path: str = None):
    """
    Plot the Negative Flip Rates for one model from the nfr_df returned by compute_nfr_table().
    """
    shift_colors = {
        "Image": "#E6550D",
        "Text": "#FDAE6B",
        "Only Text": "#31A354",
        "Only Image": "#A1D99B"
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    shifts = nfr_df["Shift"]
    values = nfr_df["NFR"]
    colors = [shift_colors[s] for s in shifts]

    bars = ax.bar(shifts, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_title(f"Negative Flip Rate for {model_name}", fontsize=14)
    ax.set_ylabel("NFR (↓ better)", fontsize=12)
    ax.set_ylim(0, max(values)*1.15 if len(values) else 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    



def compute_nfr_all_models(model_dict: dict,
                           results_dir: str = "results",
                           dataset: str = "mimic",
                           id: str = 'filename',
                           first_token: bool = False, 
                           p_yes: bool =False) -> pd.DataFrame:
    """
    Returns a dataframe whose rows are models and columns are the shifts:
    Image | Text | Only Text | Only Image
    """
    shifts = ["Image", "Text", "Only_text", "Only_image"]
    all_rows = []

    for hf_name, folder_name in model_dict.items():
        file_path = os.path.join(results_dir,
                                 folder_name,
                                 f"{dataset}_base_shifted.csv")
        if not os.path.isfile(file_path):
            print(f"[WARN] Missing results file for {folder_name}: {file_path}")
            continue

        df = pd.read_csv(file_path)
        try:
            nfr_df = compute_nfr_table(df, id=id, first_token=first_token, p_yes=p_yes)                 # → columns: Shift, NFR
        except Exception as e:
            print(f"[ERROR] Could not compute NFR for {folder_name}: {e}")
            continue

        # Pivot so every shift becomes a column
        row = nfr_df.set_index("Shift")["NFR"].reindex(
            [s.replace("_", " ").title() for s in shifts]
        )
        row.name = folder_name
        all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No NFR values could be computed – check file paths.")

    nfr_matrix = pd.concat(all_rows, axis=1).T          # models × shifts
    return nfr_matrix



# ────────────────────────────────────────────────────────────────────────────────
# 3.  Heat-map of the NFR matrix
# ────────────────────────────────────────────────────────────────────────────────
import matplotlib as mpl

# Choose a legible text colour given the cell colour
def _pick_text_color(rgba, threshold=0.55):
    """Return 'black' or 'white' depending on perceptual lightness of RGBA."""
    r, g, b, _ = rgba
    # WCAG-like relative luminance (0=dark, 1=light)
    L = 0.2126*r + 0.7152*g + 0.0722*b
    return "black" if L > threshold else "white"


# ──────────────────────────────────────────────────────────────────────────────
# Public plotting function
# ──────────────────────────────────────────────────────────────────────────────
def plot_nfr_heatmap(
    nfr_matrix: pd.DataFrame,
    dataset_name: str = "MIMIC-CXR",
    model_name_map: dict | None = None,
    save_path: str | None = None,
):
    """
    Parameters
    ----------
    nfr_matrix : DataFrame
        Rows = internal model IDs, columns = shifts ('Image', 'Text', …).
    dataset_name : str
        Used in the figure title.
    model_name_map : dict
        Optional mapping {internal_id: "Pretty Name"}.
    save_path : str | None
        PNG path; if None, just shows the figure.
    """
    # ── 1) Clean names ────────────────────────────────────────────────────────
    mat = nfr_matrix.copy()
    if model_name_map:
        mat.index = [model_name_map.get(i, i) for i in mat.index]

    mat = mat.rename(
        columns={
            "Image": "Image Shift",
            "Text": "Text Shift",
            "Only Text": "Only Text",
            "Only_text": "Only Text",
            "Only Image": "Only Image",
            "Only_image": "Only Image",
        }
    )

    # Control the column order if desired
    shift_order = ["Image Shift", "Text Shift", "Only Text", "Only Image"]
    mat = mat.loc[:, [c for c in shift_order if c in mat.columns]]

    # ── 2) Figure + softer colormap ───────────────────────────────────────────
    cmap = mpl.cm.get_cmap("YlGnBu_r")  # reversed so low values are darker
    vmax = np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1.0

    fig, ax = plt.subplots(
        figsize=(1.4 * mat.shape[1], 0.6 + 0.55 * mat.shape[0])
    )
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    # ── 3) Axis tick labels ──────────────────────────────────────────────────
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=11)

    # ── 4) Annotate each cell ────────────────────────────────────────────────
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            rgba = cmap(0 if np.isnan(val) else val / vmax)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color=_pick_text_color(rgba),
            )

    # ── 5) Title + colour-bar ────────────────────────────────────────────────
    title = f"Negative-Flip Rate on {dataset_name}"
    ax.set_title(title, fontsize=14, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("NFR", rotation=-90, va="bottom")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
