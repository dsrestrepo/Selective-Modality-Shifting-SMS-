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

If you use this codebase, please cite the paper (bibtex coming soon). You may reference it as:

TODO: Add citation

---

## 🙋 Questions

Feel free to open an issue or reach out if you encounter problems or need help setting up the datasets or models.
