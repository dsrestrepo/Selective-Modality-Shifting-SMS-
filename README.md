# Selective Modality Shifting (SMS)

This repository contains the official implementation of **Selective Modality Shifting (SMS)**, a perturbation-based framework introduced in our paper:

**On the Risk of Misleading Reports: Diagnosing Textual Bias in Multimodal Clinical AI**

SMS is used to quantify the reliance of Vision-Language Models (VLMs) on visual vs. textual modalities in medical classification tasks.

---

## 📁 Repository Structure

```
.
├── eval_general_vlms.py       # Run general-domain VLMs (e.g., LLaVA, Qwen2)
├── eval_llavamed.py           # Run LLaVA-Med model
├── eval_biomed.py             # (Optional) run BiomedGPT-like models
├── job.sh                     # Script for batch experiments
├── eval_medeval.ipynb         # Notebook to analyze FariVLMed results
├── eval_mimic.ipynb           # Notebook to analyze MIMIC-CXR results
├── attentions/                # Qualitative attention maps
├── environment.base_ml.yml    # Conda env for general models
├── environment.llava-med.yml  # Conda env for LLaVA-Med
├── .env                       # Required for HF access (not included)
├── src/                       # Core implementation files
│   ├── datasets.py, prompts.py, test.py, etc.
└── notebooks/                 # Optional: Additional exploratory notebooks
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/dsrestrepo/Selective-Modality-Shifting-SMS-.git
cd Selective-Modality-Shifting-SMS-
```

### 2. Set up environments

This project uses two separate environments:

#### General-purpose models (LLaVA, Qwen2, etc.)

```bash
conda env create -f environment.base_ml.yml
conda activate base_ml
```

#### Med-specific models (LLaVA-Med)

```bash
conda env create -f environment.llava-med.yml
conda activate llava-med
```

### 3. Add your `.env` file

Create a file named `.env` in the root directory with your HuggingFace access key:

```bash
echo "hf_key=your_huggingface_token" > .env
```

---

## 📦 Required Datasets

To run the experiments, download the test sets from:

* **FairVLMed**: [GitHub – FairCLIP](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP)
* **MIMIC-CXR v2.1.0**: [PhysioNet](https://physionet.org/content/mimic-cxr/2.1.0/)

You will need credentialed access to download the data.

Once downloaded, organize them according to the expected format used in `src/datasets.py`.

---

## 🚀 Running Experiments

### Option 1: Use provided script

```bash
bash job.sh
```

### Option 2: Run manually

```bash
conda activate base_ml
python eval_general_vlms.py

conda activate llava-med
python eval_llavamed.py
```

---

## 📊 Result Analysis

Use the following notebooks to analyze the results and reproduce the figures in the paper:

* `eval_medeval.ipynb`: Analysis on MedEval (FairVLMed)
* `eval_mimic.ipynb`: Analysis on MIMIC-CXR
* Results will include:

  * Performance under modality shifts
  * First-token calibration (ECE) and calibration curves
  * Negative Flip Rate (NFR)

---

## Attention Analysis

Qualitative attention visualizations can be found in the `attentions/` folder. These were generated for specific models and datasets to explore how attention shifts between modalities during token generation.

---

## 📄 Citation

If you use this codebase or the paper, please cite the paper as:

```markdown
Restrepo, D., Ktena, I., Vakalopoulou, M., Christodoulidis, S., & Ferrante, E. (2026).  
*On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI.*  
In Qiu, J., et al. *AI for Clinical Applications. Agentic AI CMLLMs CREATE 2025*.  
LNCS 16147. Springer.  
[https://doi.org/10.1007/978-3-032-06004-4_32](https://doi.org/10.1007/978-3-032-06004-4_32)
```
or 

```markdown
@InProceedings{10.1007/978-3-032-06004-4_32,
author="Restrepo, David
and Ktena, Ira
and Vakalopoulou, Maria
and Christodoulidis, Stergios
and Ferrante, Enzo",
editor="Qiu, Jianing
and Wu, Jinlin
and Langlotz, Curtis
and Huang, Baoru
and Lei, Zhen
and Wu, Honghan
and Liu, Hongbin
and Xie, Weidi",
title="On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI",
booktitle="AI for Clinical Applications",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="320--330",
abstract="Clinical decision-making relies on the integrated analysis of medical images and the associated clinical reports. While Vision-Language Models (VLMs) can offer a unified framework for such tasks, they can exhibit strong biases toward one modality, frequently overlooking critical visual cues in favor of textual information. In this work, we introduce Selective Modality Shifting (SMS), a perturbation-based approach to quantify a model's reliance on each modality in binary classification tasks. By systematically swapping images or text between samples with opposing labels, we expose modality-specific biases. We assess six open-source VLMs--four generalist models and two fine-tuned for medical data-- on two medical imaging datasets with distinct modalities: MIMIC-CXR (chest X-ray) and FairVLMed (scanning laser ophthalmoscopy). By assessing model performance and the calibration of every model in both unperturbed and perturbed settings, we reveal a marked dependency on text input, which persists despite the presence of complementary visual information. We also perform a qualitative attention-based analysis which further confirms that image content is often overshadowed by text details. Our findings highlight the importance of designing and evaluating multimodal medical models that genuinely integrate visual and textual cues, rather than relying on single-modality signals.",
isbn="978-3-032-06004-4"
}
```

---

## 🙋 Questions

Feel free to open an issue or reach out if you encounter problems or need help setting up the datasets or models.
