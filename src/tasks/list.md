Below are three separate tables that organize the tasks by type. All tables have the same three columns: Task, PyTorch Model, and TensorFlow Model.

---

## Natural Language Processing (NLP) Tasks

| Select   | Task                       | PyTorch Model                                           | TensorFlow Model                                         |
|----------|----------------------------|---------------------------------------------------------|---------------------------------------------------------|
| [ ] | Feature Extraction         | `distilbert/distilbert-base-cased`                      | `distilbert/distilbert-base-cased`                      |
| [ ]   | Text Classification        | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| [ ]   | Token Classification       | `dbmdz/bert-large-cased-finetuned-conll03-english`      | `dbmdz/bert-large-cased-finetuned-conll03-english`      |
| [x]   | Question Answering         | `distilbert/distilbert-base-cased-distilled-squad`      | `distilbert/distilbert-base-cased-distilled-squad`      |
| [ ]   | Table Question Answering   | `google/tapas-base-finetuned-wtq`                       | `google/tapas-base-finetuned-wtq`                       |
| [ ]   | Fill-Mask                  | `distilbert/distilroberta-base`                         | `distilbert/distilroberta-base`                         |
| [ ]   | Summarization              | `sshleifer/distilbart-cnn-12-6`                         | `google-t5/t5-small`                                    |
| [x]   | Translation                | `('en', 'fr'): google-t5/t5-base; ('en', 'de'): google-t5/t5-base; ('en', 'ro'): google-t5/t5-base` | `('en', 'fr'): google-t5/t5-base; ('en', 'de'): google-t5/t5-base; ('en', 'ro'): google-t5/t5-base` |
| [ ]   | Text2Text Generation       | `google-t5/t5-base`                                     | `google-t5/t5-base`                                     |
| [x]   | Text Generation            | `openai-community/gpt2`                                 | `openai-community/gpt2`                                 |
| [ ]   | Zero-Shot Classification   | `facebook/bart-large-mnli`                              | `FacebookAI/roberta-large-mnli`                         |

---

## Computer Vision & Multimodal Tasks

| Select   | Task                          | PyTorch Model                                           | TensorFlow Model                                         |
|----------|-------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| [ ]   | Visual Question Answering     | `dandelin/vilt-b32-finetuned-vqa`                       | N/A                                                     |
| [ ]   | Document Question Answering   | `impira/layoutlm-document-qa`                           | N/A                                                     |
| [ ]   | Image-Text-to-Text            | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`             | N/A                                                     |
| [ ]   | Zero-Shot Image Classification| `openai/clip-vit-base-patch32`                          | `openai/clip-vit-base-patch32`                          |
| [x]   | Image Classification          | `google/vit-base-patch16-224`                           | `google/vit-base-patch16-224`                           |
| [ ]   | Image Feature Extraction      | `google/vit-base-patch16-224`                           | `google/vit-base-patch16-224`                           |
| [ ]   | Image Segmentation            | `facebook/detr-resnet-50-panoptic`                      | N/A                                                     |
| [ ]   | Image-to-Text                 | `ydshieh/vit-gpt2-coco-en`                              | `ydshieh/vit-gpt2-coco-en`                              |
| [x]   | Object Detection              | `facebook/detr-resnet-50`                               | N/A                                                     |
| [x]   | Zero-Shot Object Detection    | `google/owlvit-base-patch32`                            | N/A                                                     |
| [ ]   | Depth Estimation              | `Intel/dpt-large`                                       | N/A                                                     |
| [ ]   | Video Classification          | `MCG-NJU/videomae-base-finetuned-kinetics`              | N/A                                                     |
| [ ]   | Mask Generation               | `facebook/sam-vit-huge`                                 | N/A                                                     |
| [x]   | Image-to-Image               | `caidas/swin2SR-classical-sr-x2-64`                     | N/A                                                     |

---

## Audio Tasks

| Select   | Task                             | PyTorch Model                                           | TensorFlow Model                                         |
|----------|----------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| [ ]   | Audio Classification             | `superb/wav2vec2-base-superb-ks`                        | N/A                                                     |
| [ ]   | Automatic Speech Recognition     | `facebook/wav2vec2-base-960h`                           | N/A                                                     |
| [ ]   | Text-to-Audio                    | `suno/bark-small`                                       | N/A                                                     |
| [ ]   | Zero-Shot Audio Classification   | `laion/clap-htsat-fused`                                | N/A                                                     |

---

You can choose where to place these new tables in your project.