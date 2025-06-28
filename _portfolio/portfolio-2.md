---
title: "Bridging the Digital Divide with XRI Global"
excerpt: "A project in examining the gaps between low-resource languages in the digital sphere."
collection: portfolio
---

All scripts related to this project can be found in this [repo](https://github.com/smbirnbaum/portfolio-scripts).

## Overview

In lieu of a thesis, the University of Arizona’s Human Language Technology program requires a capstone internship. I completed mine with [XRI Global](https://www.xriglobal.ai/), a startup committed to tackling the underrepresentation of low-resource languages in AI. Spearheaded by Daniel Wilson, Ph.D., our team of interns joined forces on an ambitious initiative called [Bridging the Digital Divide](http://www.digitaldivide.ai/). The goal was to map out existing AI-relevant resources across the world’s languages with particular attention to those most often left out of NLP development pipelines.

The resulting map would serve a broad range of stakeholders, from researchers and language activists to developers and policy makers. To support this effort, each intern was assigned specific tasks aligned with their technical background and strengths. While some collaborated on visualizations or web features, most of us worked independently due to the remote nature of the internship. Tasks were initially assigned by our supervisor, but we coordinated through weekly Zoom calls and stayed in regular contact via Discord to avoid overlap and support each other as needed. My contributions centered around two main tasks: auditing the quantity of available speech data for each language, and later, building an automatic speech recognition (ASR) model for a severely underrepresented language.

## Cataloging the World’s Audio Resources

My first responsibility was to determine how much validated audio data existed per language in our curated corpus. To do this, I wrote a Python script using the `mutagen` library, which allowed me to extract duration metadata from audio files across multiple formats (.wav, .mp3, .flac, etc.). The script walked through a base directory structure where each folder represented a language, tallied the number of files per language, and calculated total durations in hours, minutes, and seconds. The output was written to a TSV file that gave us a clear overview of the available speech data.

```python
import os
import csv
from mutagen import File
from collections import defaultdict

def hms_to_seconds(hours, minutes, seconds):
    """Convert hours, minutes, and seconds to total seconds."""
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_hms(total_seconds):
    """Convert total seconds to hours, minutes, and seconds."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds

def get_audio_duration(file_path):
    """Get the duration of an audio file in hours, minutes, and seconds."""
    if not os.path.exists(file_path):
        return None

    try:
        audio = File(file_path)
        if audio is None or not hasattr(audio, 'info'):
            return None

        duration_seconds = audio.info.length
        return seconds_to_hms(duration_seconds)
    except Exception:
        return None

def get_total_duration_per_language(base_folder):
    """Calculate the total duration of audio files grouped by language folders."""
    total_seconds_per_language = defaultdict(int)
    audio_files_per_language = defaultdict(list)

    if not os.path.exists(base_folder):
        print("Error: Base folder does not exist.")
        return None

    try:
        for language_folder in os.listdir(base_folder):
            language_folder_path = os.path.join(base_folder, language_folder)

            if os.path.isdir(language_folder_path):
                for root, _, files in os.walk(language_folder_path):
                    for file_name in files:
                        if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a')):
                            file_path = os.path.join(root, file_name)
                            duration = get_audio_duration(file_path)
                            if duration:
                                seconds = hms_to_seconds(*duration)
                                total_seconds_per_language[language_folder] += seconds
                                audio_files_per_language[language_folder].append((file_path, *duration))

        total_durations = {
            language: seconds_to_hms(total_seconds)
            for language, total_seconds in total_seconds_per_language.items()
        }

        return total_durations, audio_files_per_language
    except Exception as e:
        print(f"Error processing folder: {e}")
        return None

def export_counts_to_tsv(total_durations, audio_files_per_language):
    """Export the count of audio files per language into a .tsv file."""
    output_file = os.path.join(os.getcwd(), "audio_counts.tsv")

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            writer.writerow(["Language", "File Count", "Total Duration (HH:MM:SS)"])

            for language, files in audio_files_per_language.items():
                file_count = len(files)
                hours, minutes, seconds = total_durations[language]
                duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                writer.writerow([language, file_count, duration_str])

        print(f"File counts and total durations have been exported to: {output_file}")
    except Exception as e:
        print(f"Error exporting data to TSV: {e}")

if __name__ == "__main__":
    base_folder = input("Enter the path to the base folder containing language subfolders: ").strip()
    result = get_total_duration_per_language(base_folder)

    if result:
        total_durations, audio_files_per_language = result
        print("\nTotal Duration Per Language:")
        for language, (h, m, s) in total_durations.items():
            print(f"Language: {language}\nTotal Duration: {h} hours, {m} minutes, {s} seconds\n")

        export_counts_to_tsv(total_durations, audio_files_per_language)
```

Although the technical implementation was straightforward, the task itself was one of the most time-consuming parts of the internship. Each week brought a new list of languages to audit, and the datasets were massive. I ran into repeated issues with downloading full corpora due to network interruptions and unreliable hosting servers. Storage quickly became another bottleneck; we ran out of room on multiple cloud drives, requiring constant reshuffling of resources and manual cleanup.

While these challenges weren’t strictly programming problems, they pushed me to think more like a developer than a documentation specialist. I had to break down my task into smaller, testable pieces, account for edge cases, and approach each new error with a troubleshooting mindset. It was humbling, but also incredibly rewarding to move beyond my comfort zone and contribute something practical.

## Training a Turkmen ASR Model

| Problem                         | Solution                                         |
|--------------------------------|--------------------------------------------------|
| Missing characters (`[UNK]`)   | Rebuilt tokenizer with correct full character set |
| Memory limitations on HPC      | Used mixed precision + gradient accumulation     |
| Lack of capitalization in output | Added casing to tokenizer + rebuilt final model |


### Technologies & Libraries Used
-  `transformers` (Hugging Face) for modeling and Trainer API
-  `torchaudio` for audio loading and resampling
-  `datasets` (Hugging Face) for dataset loading and preprocessing
-  `jiwer` for WER evaluation
-  `Wav2Vec2ForCTC` pretrained on Turkish and fine-tuned on Turkmen
- Custom `Wav2Vec2Processor` with rebuilt tokenizer and feature extractor
- Trained on Jarvis Labs using **NVIDIA A5000**
  
During an inventory check, I noticed that some languages were missing. Feeling courageous, I took it upon myself to try creating a Turkmen ASR model using the techniques I learned in LING 696G. A small amount of audio data was available on Common Voice, and due to its mutual intelligibility, I decided to start with a pretrained Turkish model. Turkmen is considered a relatively low-resource language, particularly in the context of speech technologies, and my goal was to explore how transfer learning could help bootstrap ASR systems even when labeled data is limited.

### Phase 1: Setup & Baseline

I trained on a remote [Jarvis Labs](https://jarvislabs.ai/) HPC with an **NVIDIA A5000 GPU** using a mix of Hugging Face’s `Trainer` API and `torchaudio` for audio preprocessing. The first training run focused on:

- Applying mixed precision (`fp16=True`)
- Using gradient checkpointing and accumulation to stay under memory limits
- Basic greedy decoding for evaluation

The results were mixed. My initial Word Error Rate (WER) was around **62%**, which was disappointing but expected. Many words were transcribed as `[UNK]`, a telltale sign that the model’s **tokenizer was missing characters**.

  

### Phase 2: Tokenizer Rebuilds & Vocabulary Engineering
I extracted all **unique characters** from the Turkmen `.tsv` files and used them to build a custom `Wav2Vec2CTCTokenizer`. I iterated multiple times:
- First with lowercase-only characters (a–z, diacritics)
- Then with full **case-sensitive vocab**
- Finally including characters like `ä`, `Ä`, `ʼ`, and `ʻ`, which had been causing `[UNK]` artifacts

After each tokenizer update, I retrained from scratch and saw clear improvement. The `[UNK]` tokens disappeared almost entirely, and WER dropped dramatically with each iteration. I also switched from uncased to cased tokenization and retrained the language model head (`lm_head`) to match the new vocabulary.

  

### Phase 3: Full Dataset Training & Final Model
Once I confirmed the full pipeline was stable, I trained on [**the entire validated Turkmen corpus** (Common Voice v21.0)](https://commonvoice.mozilla.org/en/datasets). I built the training script from scratch using:

-  `Wav2Vec2Processor` with custom tokenizer and feature extractor
- A custom `data_collator` using `pad_sequence` for labels and input values
- Resampling all audio to 16kHz using `torchaudio.functional.resample`
- Evaluation with `jiwer.wer()` on the official test set

### Final Results

- Final model achieved **WER = 0.1624 (16.24%)**
- Tokenizer contained **57 characters**, including case-sensitive Turkmen orthography
- No `[UNK]` tokens in the final transcriptions
- Final output looked clean and natural, e.g.:

**Input:**  _Sebäbi bilinmeýäni bilmegiň ýoly bilinýäni has gowy düşünmekden geçýär._  
**Output:**  _Sebäbi bilinmeýäni bilmegiň ýoly bilinýäni has gowy düşünmekden geçýär._  

I monitored training loss and validation loss across five epochs. The curve was clean; no divergence, no signs of overfitting. Training and validation loss converged, suggesting meaningful learning rather than memorization.

This ended up being my proudest contribution to the internship. It was the first time I had trained a deep learning model and seen it produce truly usable results.

The final model achieved a WER of **16.2%**, which is undeniably promising, especially for a low-resource language like Turkmen. However, I also recognize that this number might be slightly misleading. With limited training data and no out-of-domain testing set, there's a real possibility that the model overfit to the Common Voice corpus. While the WER looks great on paper, it should be interpreted as an optimistic estimate rather than a definitive benchmark.

That said, the consistent drop in training and evaluation loss across five epochs suggests that the model was learning meaningfully and not just memorizing the data. Earlier training runs using a less sophisticated tokenizer hovered around 62% WER, and each iteration, especially improvements to the tokenizer and vocabulary, contributed to better performance. In that sense, the final model may not be perfect, but it hopefully reflects the effectiveness of transfer learning and careful tuning in low-resource ASR development.

### Pseudocode

Turkmen ASR: Model Training Pipeline

**Load Pretrained Resources**
- Load the custom Wav2Vec2 processor (tokenizer + feature extractor)
- Load the preprocessed Turkmen dataset from disk

**Preprocessing Function**

For each audio example:

- Load the `.wav` file
- Extract input features using the processor
- Tokenize the corresponding transcript
- Return:  
  ```json
  {
    "input_values": [...],  
    "labels": [...]
  }
  ```

 **Dataset Preparation**
-   Apply `preprocess_fn` across the dataset
-   Drop `"path"` and `"sentence"` columns
-   Split into training (90%) and testing (10%) subsets

**Data Collator**

Custom function to:

-   Pad `input_values` with `pad_token_id`
-   Pad `labels` with `-100` (ignored in CTC loss)
-   Return batched tensors:
    ```json 
    {  
	    "input_values": padded_tensor,  "labels": padded_tensor 
    }
    ``` 

**Configure the Model**

-   Load base config from Turkish Wav2Vec2 model
-   Override:
    -   `vocab_size` → `57` (custom tokenizer)
    -   `pad_token_id` from tokenizer
    -   `ctc_loss_reduction = "mean"`
    -   `ctc_zero_infinity = True`

**Initialize Model**

-   Load pretrained model with `ignore_mismatched_sizes=True`
-   Replace `lm_head` to match new vocab size
-   Enable `gradient_checkpointing`
-   Call `model.tie_weights()` to retie encoder-decoder output

**Training Configuration**

Using Hugging Face `TrainingArguments`:

-   Batch size = `2`, accumulation = `2` steps
-   Epochs = `5`
-   Evaluation & saving every epoch
-   Mixed precision = `fp16`
-   Output path: `/home/jl_fs/models/wav2vec2_turkmen_full_v2`

**Training**

-   Instantiate Hugging Face `Trainer` with:
    -   Model
    -   Arguments
    -   Datasets
    -   Tokenizer    
    -   Data collator
-   Begin training with:
	`trainer.train()`  


#### **Note:** The reason this section includes pseudocode instead of full implementation is twofold. **First**, the training method closely mirrors the ASR project I completed for LING 696G, and the full code for that project is already available in the same portfolio repository. **Second**, out of respect for the project's scope and the organization’s discretion, the models trained during the XRI internship were not intended for public release.  

## What's Next?

Although the internship has officially wrapped, the impact of the work continues. The complete methodology I used remains available and reusable for similar low-resource language efforts.

The broader [Bridging the Digital Divide](http://digitaldivide.ai) initiative is still ongoing, with plans to launch an open-source visualization tool for tracking global speech data availability. I continue to explore opportunities to build real-world tools that address gaps in language equity and digital access.

This experience solidified my passion for developing inclusive, language-aware technologies. I'm now actively looking for roles where I can apply my ASR/NLP skill set to projects involving localization and underrepresented languages.