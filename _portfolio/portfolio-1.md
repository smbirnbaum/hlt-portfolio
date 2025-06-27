---
title: "LING 696G: Swedish Wav2Vec2 ASR Model"
excerpt: "Final project for Neural Techniques for Speech Technology Seminar"
collection: portfolio
---

All scripts related to this project can be found in this [repo](https://github.com/smbirnbaum/portfolio-scripts).

## Overview

This project was the final assignment for **LING 696G: Neural Techniques for Speech Technology**. It was my first complete attempt at building an end-to-end ASR model.

Originally, I wanted to fine-tune an English **Wav2Vec2** model on Swedish data, but that quickly led to chaotic results; mixed-language "Swenglish" outputs due to the pretrained model’s English-centric phonetic bias. Instead, I pivoted and selected a Swedish-specific checkpoint: **KBLab/wav2vec2-large-voxrex**. My dataset came from the Common Voice corpus, trimmed down to 20 hours of validated speech.

| Problem                                | Solution                                                                 |
|----------------------------------------|--------------------------------------------------------------------------|
| Tokenizer lacked lowercase + special Swedish chars | Manually rebuilt tokenizer, adding casing + diacritics            |
| Mismatch between vocab + model         | Used `ignore_mismatched_sizes=True` and rebuilt model head               |
| CUDA OOM errors                        | Lowered batch size, added gradient checkpointing                         |
| Transformers bug with batch dispatching | Downgraded to a stable library version                                   |
| Critical file loss after storage cleanup | Reprocessed dataset + renamed scripts to recover pipeline               |


### Technologies & Libraries Used
- `Transformers` (Hugging Face) for Wav2Vec2 modeling + Trainer API
- `Datasets` (Hugging Face) for audio/text loading and preprocessing
- `Torchaudio` for waveform decoding + resampling
- `Jiwer` for WER calculation
- Slurm + Singularity on UArizona HPC (**Tesla P100 GPU**)

### Phase 1: Preprocessing, Setup, and Tokenizer Repair

The first step was creating a usable dataset. I created a script to convert the Common Voice mp3s to .wav files.

``` python

from pydub import AudioSegment
import os

# Define source and output dirs
source_dir = "./clips"  
output_dir = "./wavs"   
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(source_dir, filename)
        wav_filename = filename.replace(".mp3", ".wav")
        wav_path = os.path.join(output_dir, wav_filename)

        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

print("All MP3s converted.")
```

And another script to retrieve a subset of 20 hours of audio data from the Common Voice dataset.

``` python
import pandas as pd

# Load validated TSV and clip durations TSV
validated_path = "validated.tsv"
durations_path = "clip_durations.tsv"

validated_df = pd.read_csv(validated_path, sep="\t")
durations_df = pd.read_csv(durations_path, sep="\t")

# Clean clip filename column to match validated path format
durations_df["clip"] = durations_df["clip"].str.replace(".mp3", ".wav")
durations_df["duration_sec"] = durations_df["duration[ms]"] / 1000  # Convert ms to sec

# Merge on filename
merged_df = validated_df.merge(durations_df, left_on="path", right_on="clip")

# Sort by duration descending (or random shuffle for diversity)
merged_df = merged_df.sample(frac=1, random_state=42)

# Select ~20 hours (72000 seconds)
selected_rows = []
total_duration = 0
for _, row in merged_df.iterrows():
    if total_duration + row["duration_sec"] > 72000:
        break
    selected_rows.append(row)
    total_duration += row["duration_sec"]

# Save as TSV
selected_df = pd.DataFrame(selected_rows)
selected_df[["path", "sentence"]].to_csv("validated_20h.tsv", sep="\t", index=False, header=False)

print(f"Selected {len(selected_df)} rows, total {total_duration / 3600:.2f} hours")
```

Then I extracted audio-transcription pairs from `validated_20.tsv`, converted it to Hugging Face format, and saved it for training.

``` python
# convert_to_hf.py

import pandas as pd
from datasets import Dataset

# Load TSV
df = pd.read_csv("/home/u5/shawnabirnbaum/data/validated_20h.tsv", sep="\t")
# Select only 'path' and 'sentence' (text transcription)
df = df[["path", "sentence"]]
# Convert paths to absolute file paths
df["path"] = df["path"].apply(lambda  x: f"/home/u5/shawnabirnbaum/data/wavs/{x}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("/home/u5/shawnabirnbaum/data/preprocessed_swedish")

print("Dataset cleaned & saved")

```

But after my first model run, the results were messy. I saw `[UNK]` tokens throughout the transcriptions, especially for lowercase words.

It didn’t take long to realize the default tokenizer did not account for lowercase letters and Swedish characters like å, ä, and ö. I manually rebuilt the tokenizer by extracting all unique characters from the dataset and extending the vocabulary accordingly. I also had to patch the model’s `lm_head` to reflect the new vocab size.

  

``` python
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained('/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned')
processor.tokenizer.add_tokens(list('abcdefghijklmnopqrstuvwxyzåäö'))
processor.save_pretrained('/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned')
```
Every tokenizer update meant retraining from scratch. But the transcriptions improved with each round, meaning that the fixes were working, and I was probably on the right track.

### Phase 2: Training on the HPC
All training was done on the University of Arizona’s high-performance computing cluster using Slurm and a Tesla P100 GPU. I wrote a full training pipeline using the Hugging Face Trainer API and some Singularity containers, courtesy of Professor Hammond. Highlights included:

- Mixed precision training (`fp16=True`)
- Gradient checkpointing to stay within memory limits
- Manual sequence padding using `pad_sequence`
- Scripted Slurm job submissions and evaluation cycles

Of course, it wasn’t smooth sailing. I ran out of disk space partway through and accidentally deleted critical files while trying to free up space. I had to regenerate the dataset, reprocess the tokenizer, rename all my scripts, and start over just a couple of days before the deadline. It was chaos, but I got it working again.

``` python
# train1.py

import os
import torch
import torchaudio
from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

#  Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU detected. Running on CPU.")

#  Load dataset
dataset_path = "/home/u5/shawnabirnbaum/data/preprocessed_swedish"
dataset = load_from_disk(dataset_path)

#  Ensure dataset is split
if "train" not in dataset or "test" not in dataset:
    dataset = dataset.train_test_split(test_size=0.1)

#  Load Wav2Vec2 model and processor
model_name = "/home/u5/shawnabirnbaum/wav2vec2_sv"
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)
#  Enable Gradient Checkpointing to save memory
model.gradient_checkpointing_enable()


#  Fix tokenizer padding issue
processor.tokenizer.pad_token = "[PAD]"

#  Dynamically update vocab size
model.config.vocab_size = len(processor.tokenizer)
model.tie_weights()
print(f" Updated model vocab size: {model.config.vocab_size}")

#  Preprocessing function
def preprocess_function(batch):
    audio, _ = torchaudio.load(batch["path"])
    batch["input_values"] = processor(audio.squeeze().numpy(), sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"], padding="longest", return_tensors="pt").input_ids[0]
    return batch

dataset = dataset.map(preprocess_function, remove_columns=["path", "sentence"])

#  Custom Data Collator (Manual Padding)
def data_collator(features):
    input_values = [torch.tensor(f["input_values"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]

    input_values = pad_sequence(input_values, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  

    return {"input_values": input_values, "labels": labels}

#  Training arguments
training_args = TrainingArguments(
    output_dir="/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=2,
    logging_dir="/home/u5/shawnabirnbaum/logs",
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned")
processor.save_pretrained("/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned")
```

  

### Phase 3: Final Model Performance

The final training run completed on time, but one last bug tanked the output: I forgot to include `added_tokens.json` in the final model directory. It was the file that stored the extra characters I had added to the tokenizer. Without it, the model reverted to a broken vocabulary during inference.

  

As a result, the final **Word Error Rate** shot up to **97.52%**, even though the earlier runs had stabilized around **94.29%**. The transcriptions were mostly unreadable due to `<pad>` spam.

  

I figured this out about 30 minutes before the submission deadline, and there wasn’t enough time to regenerate the inference results. But I knew the cause, and I knew how to fix it: restore the tokenizer, re-run `eval1.py`, and call it a day.

  

``` python
# eval1.py

import os
import torch
import torchaudio
import jiwer
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Define dataset & model paths
dataset_path = "/home/u5/shawnabirnbaum/data/validated_20h.tsv"
audio_dir = "/home/u5/shawnabirnbaum/data/wavs/"
model_path = "/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned"

# Load model & processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_path)

# Read dataset
df = pd.read_csv(dataset_path, sep="\t")

# WER storage
predictions = []
references = []

# Process each file one at a time
print("Running inference...")
for index, row in df.iterrows():
    try:
        # Prepend full directory path
        audio_path = os.path.join(audio_dir, row["path"])

        if not os.path.exists(audio_path):
            print(f"Skipping missing file: {audio_path}")
            continue  # Skip missing files instead of crashing

        # Load audio
        waveform, _ = torchaudio.load(audio_path)
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0]

        # Store results for WER
        predictions.append(prediction)
        references.append(row["sentence"])

        # Print progress every 100 files
        if index % 100 == 0:
            print(f"Processed {index}/{len(df)} files...")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# Compute final WER
wer = jiwer.wer(references, predictions)
print(f"\nFinal Word Error Rate (WER): {wer:.4f}")
```

And here is the inference script, without which this project would have been nearly impossible.

```python
# infer1.py

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#  Load model & processor
model_path = "/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned"
model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
processor = Wav2Vec2Processor.from_pretrained(model_path)

#  Run inference
def transcribe(audio_path):
    audio, _ = torchaudio.load(audio_path)
    input_values = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

#  Test on a sample
test_audio = "/home/u5/shawnabirnbaum/common_voice_sv-SE_41823407.wav"
transcription = transcribe(test_audio)
print(f" Transcription: {transcription}")
```

### Final Outcome

By the end of the project, I had produced:

- A fine-tuned Swedish ASR model built on a language-specific checkpoint
- A custom tokenizer with full lowercase and diacritic support
- Scripts for training, evaluation, inference, and preprocessing
- Slurm-compatible training pipelines for containerized HPC environments
- A clean dataset in Hugging Face format
- A working understanding of WER evaluation and debugging

While the final WER was not ideal, I finished with something far more useful: a full speech training pipeline I had built, broken, and fixed myself.

### What I Learned

- Patch tokenizer vocab mismatches in Transformers
- Create a dataset pipeline that integrates Hugging Face + torchaudio
- Write all major ASR pipeline components from scratch Troubleshoot GPU memory errors, logging bugs, and train/test misconfigurations  
- Work within technical limits (e.g., GPU size, storage quota) on a real HPC system

I also learned to be brutally pragmatic. I made mistakes, lost files, and had to rebuild things I thought were done. But that experience prepared me to take on even messier challenges later.

### HLT Learning Outcomes Demonstrated

- **Write, debug, and document readable code** 
    - I developed, tested, and reworked several scripts (`train1.py`, `eval1.py`, `infer1.py`) to train and evaluate models efficiently.

- **Select and apply appropriate algorithms and concepts in NLP** 
    - I used CTC-based ASR modeling and WER evaluation, adapting pretrained checkpoints to new languages.

- **Apply tools and libraries used in HLT** 
    - I successfully integrated Hugging Face, torchaudio, jiwer, Slurm, and Singularity in a full production-style workflow.

- **Demonstrate professional skills** 
    - I worked independently, handled crisis debugging under deadline pressure, documented my pipeline, and delivered a final product.

  

### What I’d Do Differently

If I could go back, I’d probably choose a different language altogether; ideally one with a larger dataset or better tokenizer support. Alternatively, I’d use a Norwegian or Danish base model and fine-tune on Swedish to see if mutual intelligibility could compensate for dataset limitations. Likely Norwegian as it is closer to Swedish than Danish is to Swedish.

I'd also implement stricter version control for tokenizer artifacts, make backups early and often, and keep my evaluation script decoupled from training outputs to avoid last-minute surprises.

If I had more time, I would:

- Re-run inference with the fixed tokenizer to get a more honest WER
- Try beam decoding instead of greedy to boost transcription quality
- Deploy the model in a simple streamlit or Flask demo
- Compare the Swedish model’s output against other Nordic language models to assess transfer learning potential

### What’s Next?
I would like to build a Swedish model from scratch rather than using a pretrained one, if I had the proper resources.

Luckily, everything I learned in this project got put to good use in the [next project with XRI Global](/hlt-portfolio/portfolio/portfolio-2).
