# Shawna Birnbaum — Human Language Technology Portfolio

I’m a technical writer and Human Language Technology graduate with experience in speech recognition, multilingual data preparation, and localization. I hold an **M.S. in Human Language Technology from the University of Arizona (2025)** and a **B.S. in Technical Communication from Arizona State University (2021)**.

My work combines language, programming, and clear documentation. I’m interested in building tools that support underrepresented languages, improve multilingual products, and automate repetitive language QA tasks.

**[Explore my portfolio](https://smbirnbaum.github.io/hlt-portfolio/)** · **[View project scripts](https://github.com/smbirnbaum/portfolio-scripts)** · **[LinkedIn](https://www.linkedin.com/in/smbirnbaum/)**

## Featured Projects

### Speech Data Auditing and Turkmen ASR — XRI Global

During my capstone internship, I contributed to *Bridging the Digital Divide*, an initiative mapping AI resources for underrepresented languages.

- Developed Python tooling to count audio files and calculate speech duration across language corpora, exporting summaries to TSV.
- Fine-tuned a Turkish Wav2Vec2 checkpoint for Turkmen using Common Voice data.
- Rebuilt the tokenizer to support Turkmen characters and casing, and used mixed precision and gradient accumulation to manage GPU memory.
- Evaluated transcription quality using word error rate (WER) and documented limitations in generalization beyond the training corpus.

[Read the case study](https://smbirnbaum.github.io/hlt-portfolio/portfolio/portfolio-2)

### Swedish Speech Recognition — Graduate Project

Developed an ASR training pipeline using a Swedish Wav2Vec2 checkpoint and approximately 20 hours of Common Voice speech.

- Wrote scripts for audio preprocessing, dataset preparation, training, evaluation, and inference.
- Adapted the tokenizer for Swedish characters and troubleshot vocabulary mismatches.
- Ran GPU training on the University of Arizona’s HPC cluster using Slurm and Singularity.
- Documented the final model’s poor transcription performance, tokenizer packaging issue, and lessons for reproducible workflows.

[Read the case study](https://smbirnbaum.github.io/hlt-portfolio/portfolio/portfolio-1)

## Technical Skills

- **Programming and data:** Python, Bash, pandas, Jupyter, CSV/TSV processing
- **Speech and machine learning:** PyTorch, Hugging Face Transformers and Datasets, Wav2Vec2, torchaudio, jiwer, Mutagen
- **Development environments:** Git, GitHub, Linux, Slurm, Singularity, GPU/HPC workflows
- **Documentation and localization:** Markdown, XML, DITA, SDK documentation, docs as code, multilingual content coordination

## About This Repository

This repository contains the source for my portfolio website, including project case studies, code examples, and professional background. Supporting project scripts are maintained in [portfolio-scripts](https://github.com/smbirnbaum/portfolio-scripts).

The site uses Jekyll and the [Academic Pages](https://github.com/academicpages/academicpages.github.io) template and is hosted on GitHub Pages.
