"""# Complete Guide to LLM Fine-tuning Methods

## 📚 Table of Contents
1. Full Fine-tuning
2. Parameter-Efficient Fine-tuning (PEFT)
3. Instruction Tuning
4. Alignment Methods (RLHF, DPO)
5. Domain Adaptation Methods
6. Comparison & Selection Guide
"""
---

## 1️⃣ Full Fine-tuning (Traditional Approach)

### What It Is
Update **all** model parameters during training on your specific dataset.

### How It Works
```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# Training arguments
training_args = TrainingArguments(
    output_dir="./full-finetuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small due to memory
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    fp16=True,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Pros & Cons
✅ **Pros:**
- Best performance on specific tasks
- Full control over all weights
- Can completely transform model behavior

❌ **Cons:**
- Extremely expensive (GPU memory + time)
- Risk of catastrophic forgetting
- Requires massive dataset (typically 10K+ examples)
- Storage: Need full model copy for each task

### Use Cases
- Building foundation models from scratch
- Complete domain shift (medical → legal)
- When you have unlimited compute budget
- Research experiments

### Resource Requirements
| Model Size | GPU Memory | Training Time | Cost |
|------------|-----------|---------------|------|
| 7B params  | 80GB+     | Days-Weeks    | $$$$ |
| 13B params | 160GB+    | Weeks         | $$$$$ |
| 70B params | 640GB+    | Months        | $$$$$$ |

---

## 2️⃣ Parameter-Efficient Fine-tuning (PEFT)

### 2.1 LoRA (Low-Rank Adaptation) ⭐ Most Popular

### Concept
Instead of updating all weights, inject small trainable matrices that are added to frozen weights.

**Mathematical Foundation:**
```
Original: W (d × k matrix)
LoRA: W + B·A
  where B: (d × r), A: (r × k), r << d,k

Only train B and A!
Trainable params: 2 × d × r (vs d × k)
```

### Implementation
```python
from peft import LoraConfig, get_peft_model, TaskType

# Load base model (frozen)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank of decomposition matrices
    lora_alpha=32,             # Scaling factor (usually 2×r)
    target_modules=[           # Which layers to apply LoRA
        "q_proj",              # Query projection
        "k_proj",              # Key projection  
        "v_proj",              # Value projection
        "o_proj",              # Output projection
        "gate_proj",           # For LLaMA
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,         # Dropout for LoRA layers
    bias="none",               # Don't train bias terms
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

### Advanced LoRA Variants

#### **QLoRA (Quantized LoRA)**
Combines LoRA with 4-bit quantization for extreme memory efficiency.

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # Nested quantization
    bnb_4bit_quant_type="nf4",           # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",  # 70B model on 24GB GPU!
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top
lora_config = LoraConfig(r=64, lora_alpha=128)  # Can use higher rank
model = get_peft_model(model, lora_config)
```

**Memory Savings:**
- 70B model: 280GB → **48GB** (with QLoRA)
- Can fine-tune on single consumer GPU!

#### **DoRA (Weight-Decomposed LoRA)**
Separates magnitude and direction of weight updates.

```python
lora_config = LoraConfig(
    r=16,
    use_dora=True,  # Enable DoRA
    # ... other params
)
```

**When to use:** Better performance than LoRA with same parameters, especially for vision tasks.

### LoRA Hyperparameter Guide

| Parameter | Low | Medium | High | Use Case |
|-----------|-----|--------|------|----------|
| r (rank) | 4-8 | 16-32 | 64-128 | Simple tasks → Complex tasks |
| alpha | r | 2×r | 4×r | Standard → Aggressive adaptation |
| dropout | 0.0 | 0.05 | 0.1 | Large data → Small data |

---

### 2.2 Adapter Layers

### Concept
Insert small trainable modules between frozen transformer layers.

```python
from peft import AdapterConfig, get_peft_model

adapter_config = AdapterConfig(
    adapter_type="bottleneck",  # Bottleneck architecture
    reduction_factor=16,         # Bottleneck dimension: d/16
    non_linearity="relu"
)

model = get_peft_model(base_model, adapter_config)
```

**Architecture:**
```
Transformer Layer (frozen)
    ↓
Adapter (down-project to d/r)
    ↓
Non-linearity (ReLU/GELU)
    ↓
Adapter (up-project to d)
    ↓
Residual connection
```

**Pros:**
- Modular: Stack multiple adapters
- Can compose adapters (task 1 + task 2)

**Cons:**
- Adds inference latency
- More parameters than LoRA for same performance

---

### 2.3 Prefix Tuning

### Concept
Prepend trainable "virtual tokens" to each layer, keep model frozen.

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,        # Number of prefix tokens
    encoder_hidden_size=4096,     # Model hidden size
    prefix_projection=True        # Use MLP reparameterization
)

model = get_peft_model(base_model, prefix_config)
```

**How it works:**
```
[PREFIX][PREFIX]...[PREFIX] Your actual input text

Only [PREFIX] embeddings are trainable
```

**Use Cases:**
- When you need same model for multiple tasks
- Controllable generation
- Quick task switching

---

### 2.4 Prompt Tuning (Soft Prompts)

### Concept
Similar to prefix tuning but simpler—only tune input embeddings.

```python
from peft import PromptTuningConfig, PromptTuningInit

prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,
    prompt_tuning_init=PromptTuningInit.TEXT,  # Initialize from text
    prompt_tuning_init_text="Classify the sentiment:",
    tokenizer_name_or_path="meta-llama/Llama-2-7b",
)

model = get_peft_model(base_model, prompt_config)
```

**Trainable Params:** Only 8 × 4096 = 32,768 parameters!

**Best for:**
- Classification tasks
- Few-shot learning
- When data is extremely limited

---

### 2.5 (IA)³ - Infused Adapter by Inhibiting and Amplifying Inner Activations

### Concept
Learn scaling vectors to rescale activations (element-wise multiplication).

```python
from peft import IA3Config

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)

model = get_peft_model(base_model, ia3_config)
```

**Extremely Parameter Efficient:**
- Only 0.01% of model parameters
- Faster than LoRA
- Competitive performance

---

## 3️⃣ Instruction Tuning

### What It Is
Fine-tune model to follow instructions using (instruction, output) pairs.

### Dataset Format
```python
instruction_data = [
    {
        "instruction": "Translate the following English text to French",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    },
    {
        "instruction": "Summarize this article",
        "input": "<long article text>",
        "output": "<summary>"
    },
    # ... thousands more examples
]
```

### Implementation with Alpaca Format
```python
def format_instruction(example):
    """Format as instruction-following prompt"""
    if example.get("input", ""):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

# Apply formatting
formatted_dataset = dataset.map(lambda x: {"text": format_instruction(x)})
```

### Popular Instruction Datasets
1. **Alpaca** (52K instructions)
2. **Dolly-15k** (15K instructions)
3. **FLAN** (1.8M instructions)
4. **ShareGPT** (90K conversations)
5. **OpenOrca** (4.2M instructions)

### Training Approach
```python
from trl import SFTTrainer  # Supervised Fine-Tuning

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    max_seq_length=2048,
    packing=True,  # Pack multiple examples per sequence
    dataset_text_field="text",
)

trainer.train()
```

---

## 4️⃣ Alignment Methods (Making Models Helpful & Harmless)

### 4.1 RLHF (Reinforcement Learning from Human Feedback)

### Three-Stage Process

#### **Stage 1: Supervised Fine-tuning (SFT)**
Train on high-quality demonstrations.

#### **Stage 2: Reward Model Training**
Train a model to predict human preferences.

```python
from transformers import AutoModelForSequenceClassification

# Reward model outputs single scalar
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b",
    num_labels=1,  # Single reward score
)

# Training data: (prompt, chosen_response, rejected_response)
reward_data = [
    {
        "prompt": "Explain quantum computing",
        "chosen": "Quantum computing uses quantum bits...",
        "rejected": "Quantum stuff is weird and complicated."
    }
]

# Train to rank chosen > rejected
for batch in reward_data:
    reward_chosen = reward_model(batch["chosen"])
    reward_rejected = reward_model(batch["rejected"])
    
    # Loss: prefer chosen over rejected
    loss = -log_sigmoid(reward_chosen - reward_rejected)
    loss.backward()
```

#### **Stage 3: RL Fine-tuning (PPO)**
Optimize policy using reward model.

```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,        # Reference model (frozen)
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    # Generate responses
    outputs = ppo_trainer.generate(batch["query"])
    
    # Get rewards
    rewards = reward_model(outputs)
    
    # PPO update
    stats = ppo_trainer.step(batch["query"], outputs, rewards)
```

**Challenges:**
- Complex training (3 stages)
- Unstable (RL is hard)
- Requires human labelers
- Expensive compute

---

### 4.2 DPO (Direct Preference Optimization) ⭐ Simpler Alternative

### Concept
Directly optimize on preference pairs **without** reward model or RL.

```python
from trl import DPOTrainer

dpo_config = DPOConfig(
    beta=0.1,                    # Temperature parameter
    learning_rate=5e-7,
    max_length=1024,
    max_prompt_length=512,
)

# Dataset: (prompt, chosen, rejected) tuples
dpo_dataset = [
    {
        "prompt": "How to make pasta?",
        "chosen": "1. Boil water 2. Add pasta...",
        "rejected": "Just order from a restaurant."
    }
]

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create frozen copy
    args=dpo_config,
    train_dataset=dpo_dataset,
)

trainer.train()
```

**Advantages over RLHF:**
- ✅ Single stage (no reward model)
- ✅ More stable training
- ✅ Simpler implementation
- ✅ Better performance in practice

**Loss Function:**
```
L = -log σ(β log(π/π_ref)[y_w] - β log(π/π_ref)[y_l])

where:
  y_w = chosen response
  y_l = rejected response
  π = policy model
  π_ref = reference model
  β = temperature
```

---

### 4.3 KTO (Kahneman-Tversky Optimization)

### Concept
Uses only binary feedback (thumbs up/down) instead of pairwise preferences.

```python
from trl import KTOTrainer

kto_dataset = [
    {"prompt": "Explain AI", "completion": "AI is...", "label": True},   # Good
    {"prompt": "Explain AI", "completion": "AI bad", "label": False},    # Bad
]

trainer = KTOTrainer(
    model=model,
    train_dataset=kto_dataset,
)
```

**When to use:**
- Only have binary ratings (not pairs)
- Cheaper data collection
- Recent research (2024)

---

### 4.4 ORPO (Odds Ratio Preference Optimization)

### Concept
Combines SFT and preference learning in single stage.

```python
from trl import ORPOTrainer

orpo_config = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    max_length=1024,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=preference_dataset,
)
```

**Advantage:** No need for separate SFT stage!

---

## 5️⃣ Domain Adaptation Methods

### 5.1 Continued Pre-training (CPT)

### Concept
Continue pre-training on domain-specific unlabeled data.

```python
from transformers import DataCollatorForLanguageModeling

# Load domain data (e.g., medical texts)
domain_data = load_dataset("medical_papers")

# Use masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,          # Masked LM for encoder models
    mlm_probability=0.15
)

# Or causal LM for decoder models
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    train_dataset=domain_data,
)
```

**Use Cases:**
- Adapting to new domain (legal, medical, finance)
- Learning new languages
- Updating with recent data

---

### 5.2 Task-Specific Head Fine-tuning

### Concept
Keep base frozen, only train task-specific output layer.

```python
# Freeze base model
for param in model.base_model.parameters():
    param.requires_grad = False

# Only train classification head
model.classifier = nn.Linear(hidden_size, num_classes)

trainer = Trainer(model=model, ...)
```

**Best for:**
- Classification tasks
- When base model is already good
- Limited computational budget

---

### 5.3 Multi-task Learning

### Concept
Train on multiple related tasks simultaneously.

```python
task_datasets = {
    "sentiment": sentiment_data,
    "ner": ner_data,
    "qa": qa_data,
}

# Mix datasets with task prefixes
mixed_data = []
for task, dataset in task_datasets.items():
    for example in dataset:
        mixed_data.append({
            "text": f"[{task.upper()}] {example['text']}",
            "label": example['label']
        })

trainer = Trainer(train_dataset=mixed_data, ...)
```

**Benefits:**
- Better generalization
- Knowledge transfer across tasks
- Single model for multiple capabilities

---

## 6️⃣ Comparison & Selection Guide

### Quick Selection Table

| Scenario | Recommended Method | Reason |
|----------|-------------------|---------|
| Limited GPU (< 24GB) | QLoRA | Can fine-tune 70B models |
| General instruction following | LoRA + Instruction Tuning | Best balance |
| Classification task | Prompt Tuning or LoRA | Minimal parameters |
| Chat/Assistant | DPO or ORPO | Better alignment |
| Domain adaptation | CPT + LoRA | Learn domain, then task |
| Multiple tasks | Adapters | Modular & composable |
| Research/Maximum performance | Full Fine-tuning | Best results |
| Production (cost-sensitive) | LoRA or (IA)³ | Efficient & effective |

---

### Parameter Efficiency Comparison

```
Method             | Trainable %  | Memory    | Performance
-------------------|--------------|-----------|-------------
Full Fine-tuning   | 100%         | Highest   | ⭐⭐⭐⭐⭐
LoRA (r=16)        | 0.1-1%       | Low       | ⭐⭐⭐⭐
QLoRA              | 0.1-1%       | Lowest    | ⭐⭐⭐⭐
Adapters           | 1-5%         | Medium    | ⭐⭐⭐⭐
Prefix Tuning      | 0.01-0.1%    | Low       | ⭐⭐⭐
Prompt Tuning      | 0.001%       | Minimal   | ⭐⭐⭐
(IA)³              | 0.01%        | Minimal   | ⭐⭐⭐⭐
```

---

### Training Time Comparison (7B Model, 10K Examples)

| Method | GPU Hours | Cost (A100) |
|--------|-----------|-------------|
| Full FT | 120-200h | $300-500 |
| LoRA | 8-12h | $20-30 |
| QLoRA | 10-15h | $25-40 |
| Prompt Tuning | 2-4h | $5-10 |

---

## 🎯 Practical Recommendations

### For Your Lead Data Scientist Role

1. **Master LoRA/QLoRA**
   - Industry standard
   - Best ROI for interviews
   - Actually used in production

2. **Understand DPO**
   - Replacing RLHF
   - Simpler than PPO
   - Show modern knowledge

3. **Know the trade-offs**
   - When to use each method
   - Cost vs performance
   - Production considerations

### Implementation Priority

```
Priority 1 (Must Know):
- LoRA/QLoRA implementation
- Instruction tuning
- Dataset preparation

Priority 2 (Should Know):
- DPO for alignment
- Prompt/Prefix tuning
- Evaluation methods

Priority 3 (Nice to Know):
- Full RLHF pipeline
- Advanced variants (DoRA, (IA)³)
- Multi-task learning
```

---

## 📚 Resources

### Papers to Read
1. LoRA: "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
2. QLoRA: "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
3. DPO: "Direct Preference Optimization" (2023)
4. Instruction Tuning: "Finetuned Language Models Are Zero-Shot Learners" (2021)

### Code Libraries
- `peft`: HuggingFace PEFT methods
- `trl`: Training transformers with RL
- `axolotl`: Training framework
- `unsloth`: Fast fine-tuning

### Datasets
- `alpaca_data`: Instruction tuning
- `Anthropic/hh-rlhf`: Human preferences
- `OpenAssistant`: Conversations
- `databricks/dolly-15k`: Instructions

This comprehensive guide should prepare you to discuss any fine-tuning method confidently in your Lead Data Scientist interview!
