## Question Answering

**Description**  
Transformers models for question answering are fine-tuned variants of pre-trained language models like BERT, RoBERTa, or DistilBERT that can extract answers to questions based on a provided context. This task is used in a variety of use cases such as building chatbots, automating customer support, and designing educational aids. The system works by taking a context passage and a specific question, then predicting the span within the text that contains the answer.

### Interesting Find: Symlinks and Model Size Calculation

While calculating the model folder size, I discovered that the reported size was doubled. This happened because the folder contains a blob (a `safetensor` file) that is also symlinked in the snapshots directory. Although the file is only stored once on disk, our recursive size function counts its size twice—once for the original file and again for the symlink. Using `os.lstat` with inode tracking helped to avoid counting duplicates, but it's still important to be aware of such symlinked files when calculating disk usage.