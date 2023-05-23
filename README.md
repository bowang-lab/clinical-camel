---

# Clinical Camel 
## ⚠️ Upcoming Code Release ⚠️

The codebase related to model inference, training, evaluation, and DBKE is under preparation and will be released soon. Please stay tuned for updates!

## Model Description

Clinical Camel model is a transformer-based language model trained on the LLaMA 13B architecture. It is specially designed as a reseach focused medical conversational model. 

Access the [live demo](https://wanglab.ml/model_proxy.html). This corresponds to an earlier version of the Clinical Camel model.

## Model Conversion

Delta weights for the model are provided. The `apply_delta` script from [FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py) can be used to convert LLaMA-13B to Clinical Camel.

The delta weights can be found at this [Hugging Face link](https://huggingface.co/wanglab/clinical-camel).

## Data 

The model was trained on a diverse dataset which includes:
- 100,000 synthetic dialogues produced via dialogue-based knowledge encoding (DBKE).
- 10,187 USMLE questions which were converted via DBKE.
- The ShareGPT dataset was also used, adding further diversity to the training data.

## Training

The training code and inference model are based on [FastChat](https://github.com/lm-sys/FastChat). We would like to extend our gratitude to the developers of FastChat for making their code available for use. 

The model was trained for 2 epochs. The specific training parameters and configurations used for the Clinical Camel model are detailed below:

| **Parameter** | **Value** |
|:-------------:|:---------:|
| Learning Rate  | 2E-5 |
| Batch Size  | 4 |
| Epochs  | 2 |
| Optimizer | AdamW (Torch) |
| Max Gradient Norm | 1 |
| Weight Decay | 0 |
| Warm-up Steps | 0 |
| Warm-up Ratio | 0.03 |
| Gradient Accumulation Steps | 8 |
| Per Device Training Batch Size | 4 |
| Maximum Sequence Length | 2048 |
| Learning Rate Scheduler | Cosine |

## Model Comparison

The Clinical Camel model was compared to several other popular models in a variety of benchmarks. These include USMLE self-assessment scores, performance on multi-step management problems, and responses to standardized safety questions. 

### USMLE Self-Assessment Scores

In the USMLE Self-Assessment, Clinical Camel achieved the highest score on Step 1 and Step 3.

| | GPT-3.5 (%) | Clinical Camel (%) | Chat Doctor (%) | PMC-LLaMA | MedAlpaca 13b (%) | Vicuna (%) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Step 1 | 36.1 | 53.2 | 11.7 | 1.1 | 12.2 | 21.3 |
| Step 2 | 56.9 | 51.4 | 18.5 | 2.7 | 27.5 | 20.4 |
| Step 3 | 55.7 | 58.2 | 14.8 | 3.3 | 26.2 | 30.3 |

### Multi-Step Management Problems

Clinical Camel shows competitive performance in multi-step management problems.

| | GPT-3.5 (%) | Clinical Camel (%) | Chat Doctor (%) | PMC-LLaMA | MedAlpaca 13b (%) | Vicuna (%) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| CFPC EM | 81.3 | 74.9 | 53.2 | 13.9 | 31.2 | 68.6 |
| CFPC FM | 85.0 | 82.1 | 67.0 | 14.3 | 44.7 | 74.8 |

## Publication

The work related to the Clinical Camel model has been published in [arXiv](https://arxiv.org/abs/2305.12031). Feel free to check out the paper for a more comprehensive understanding of the model and its performance.

## Future Updates

We are in the process of preparing the remainder of the code used in this project for publication. Please check back in the near future for additional updates and resources. Your patience is appreciated as we work to provide a comprehensive and usable repository.


## License
This project is licensed under AGPL-3.0. Please see the [LICENSE](https://github.com/bowang-lab/clinical-camel/blob/main/LICENSE) file for more details.
