# ai-course-practice

## Project Setup with Conda

Follow these steps to set up a fresh Python project using conda:

1. **Prerequisites**  
   - Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your machine.

2. **Create a Conda Environment**  
   Open your Terminal and run the following command to create a new environment named `transformers-project` with Python 3.9:
   ```bash
   conda create -n transformers-project python=3.10
   ```

3. **Activate the Environment**  
   Activate your new environment with:
   ```bash
   conda activate transformers-project
   ```

4. **Install Dependencies**  
   Install the Transformers library (and any other dependencies) using pip:
   ```bash
   pip install transformers
   ```

