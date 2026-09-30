# Phi-2 Fine-Tuning with QLoRA on Amazon Product Data

This project demonstrates how to fine-tune the **Microsoft Phi-2 language model** using **QLoRA, PEFT, Hugging Face Transformers, and TRL** on an Amazon product dataset.

The goal is to train the model to generate an informative product description using the **product name** and **product category** as input.

---

## Project Overview

Large Language Models contain billions of parameters, making full fine-tuning computationally expensive.

This project uses:

- **Phi-2** as the base causal language model
- **4-bit quantization** to reduce model memory usage
- **LoRA / PEFT** to train only a small number of parameters
- **QLoRA** to combine quantization and LoRA
- **SFTTrainer** for supervised fine-tuning
- **Amazon product data** for instruction-style training

Only approximately **0.38% of the total model parameters are trainable** during fine-tuning.

---

## Project Architecture

```text
Amazon Product Dataset
        |
        v
Data Preprocessing
        |
        v
Prompt Engineering
        |
        v
Train / Test Split
        |
        v
Hugging Face Dataset
        |
        v
Phi-2 Tokenizer
        |
        v
Microsoft Phi-2 Base Model
        |
        v
4-bit Quantization
BitsAndBytes
        |
        v
prepare_model_for_kbit_training()
        |
        v
LoRA / PEFT Adapters
        |
        v
SFTTrainer
        |
        v
Supervised Fine-Tuning
        |
        v
Fine-Tuned LoRA Adapter
        |
        v
Product Name + Category
        |
        v
Generated Product Description
```

### Architecture Diagram

Add the generated architecture image to your repository under:

```text
images/phi2_qlora_architecture.png
```

Then display it using:

```markdown
![Phi-2 QLoRA Architecture](images/phi2_qlora_architecture.png)
```

---

## Dataset

The project uses Amazon product data containing fields such as:

```text
product_name
category
about_product
```

The category column is simplified before training.

Example:

```python
df["category"] = df["category"].apply(
    lambda x: x.split("|")[-1]
)
```

The prepared dataset contains approximately:

```text
Total samples: 1,465
Training samples: 1,318
Testing samples: 147
```

---

## Prompt Engineering

Each Amazon product is converted into an instruction-style training prompt.

```text
### Instruction:
Generate a clear and informative product description for the following Amazon product.

### Product Name:
{product_name}

### Category:
{category}

### Response:
{about_product}
```

This teaches Phi-2 to learn the mapping:

```text
Product Name + Category
        ↓
Product Description
```

### Dataset & Prompt Snapshot

Place the image under:

```text
images/snapshot_dataset_prompt.png
```

Then add:

```markdown
![Dataset and Prompt Engineering](images/snapshot_dataset_prompt.png)
```

---

## Model

Base model:

```text
microsoft/phi-2
```

The tokenizer is loaded using:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2"
)
```

Padding configuration:

```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

---

## 4-Bit Quantization

The model is loaded using 4-bit quantization with `BitsAndBytesConfig`.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

Quantization reduces the memory required to load the base model.

Conceptually:

```text
Phi-2 FP16 / FP32
      ↓
4-bit Quantization
      ↓
Reduced Memory Usage
```

---

## LoRA / PEFT Configuration

LoRA allows us to keep most Phi-2 parameters frozen while training small adapter matrices.

Example configuration:

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

The project targets attention-related layers including:

```text
q_proj
k_proj
v_proj
dense
```

---

## QLoRA

QLoRA combines:

```text
4-bit Quantization
        +
LoRA
        =
QLoRA
```

The base model remains quantized while the LoRA adapters are trained.

This makes fine-tuning large language models much more memory efficient compared with full fine-tuning.

---

## Trainable Parameters

The project trains approximately:

```text
Trainable parameters:
10,485,760

Total parameters:
2,790,169,600

Trainable percentage:
~0.3758%
```

This means more than **99% of the base model parameters remain frozen**.

---

## Supervised Fine-Tuning

The project uses Hugging Face TRL:

```python
from trl import SFTTrainer, SFTConfig
```

Example training configuration:

```python
training_args = SFTConfig(
    output_dir="./phi2-amazon-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=10
)
```

The trainer is created using:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_args
)
```

Training starts with:

```python
trainer.train()
```

---

## Training Configuration Snapshot

Save the generated image as:

```text
images/snapshot_qlora_training.png
```

Then display it in the README:

```markdown
![QLoRA Training Configuration](images/snapshot_qlora_training.png)
```

---

## Technologies Used

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
Microsoft Phi-2
PEFT
LoRA
QLoRA
BitsAndBytes
TRL
SFTTrainer
Pandas
NumPy
Scikit-learn
Jupyter Notebook
```

---

## Installation

Install the required libraries:

```bash
pip install torch
pip install transformers
pip install datasets
pip install accelerate
pip install peft
pip install trl
pip install bitsandbytes
pip install pandas
pip install scikit-learn
```

Or:

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes pandas scikit-learn
```

---

## Check GPU Availability

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

GPU acceleration is recommended for QLoRA fine-tuning.

---

## Project Workflow

```text
1. Load Amazon product dataset

2. Clean and preprocess the data

3. Format product data into instruction prompts

4. Convert Pandas DataFrame into Hugging Face Dataset

5. Split the dataset into training and testing data

6. Load the Phi-2 tokenizer

7. Load Phi-2 using 4-bit quantization

8. Prepare the model for k-bit training

9. Configure LoRA adapters

10. Configure supervised fine-tuning parameters

11. Fine-tune using SFTTrainer

12. Save the trained LoRA adapter

13. Load the fine-tuned model

14. Generate Amazon product descriptions
```

---

## Example Inference

Input:

```text
Product Name:
Wireless Bluetooth Headphones

Category:
Electronics
```

Model task:

```text
Generate a clear and informative product description.
```

Expected output structure:

```text
Wireless Bluetooth headphones designed for convenient
everyday listening with portable connectivity and
consumer-focused audio features...
```

Actual generated quality depends on the training data, training settings, and final fine-tuned checkpoint.

---

## Repository Structure

```text
phi2-amazon-qlora/
│
├── peft.ipynb
│
├── README.md
│
├── requirements.txt
│
├── images/
│   ├── phi2_qlora_architecture.png
│   ├── snapshot_dataset_prompt.png
│   └── snapshot_qlora_training.png
│
├── data/
│   └── amazon_product_details.csv
│
└── model/
    └── phi2-lora-adapter/
```

---

## Key Learnings

This project helped me understand:

- How causal language models generate text
- How tokenization works
- How supervised fine-tuning works
- Difference between full fine-tuning and PEFT
- How LoRA reduces trainable parameters
- How 4-bit quantization reduces memory requirements
- How QLoRA combines quantization with LoRA
- How `SFTTrainer` simplifies LLM fine-tuning
- How instruction-style datasets are prepared for LLM training
- How fine-tuned adapters can be saved and reused

---

## Future Improvements

Future improvements can include:

- Training on a larger Amazon dataset
- Running multiple training epochs
- Comparing LoRA ranks
- Experimenting with different learning rates
- Evaluating base Phi-2 vs fine-tuned Phi-2
- Adding BLEU / ROUGE or LLM-based evaluation
- Deploying the model using FastAPI
- Building a Streamlit interface
- Deploying inference on a GPU cloud service
- Comparing Phi-2 with Phi-3, Llama, or Mistral models

---

## Author

**Avinash Nallala**

Machine Learning | Generative AI | LLMs | RAG | Agentic AI | Python

---

## Disclaimer

This project is intended for learning and experimentation with parameter-efficient LLM fine-tuning. Model outputs should be evaluated before being used in production applications.
