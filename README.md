# Huggingface-Finetune-LLM-Unit-Conversion
In this repo, we will fine-tune a LLM from Hugging Face to learn how to do accurate unit conversions. 

# Model
This file contains the foundational implementation of the BaseLLM class, which serves as the base for interacting with the SmolLM2 language model. It includes methods for:
Formatting prompts for the model (format_prompt).
Parsing answers from the model's output (parse_answer).
Generating responses for single or batched prompts (generate and batched_generate).
Answering multiple questions at once (answer).
Testing the model's basic functionality (test_model).
This file is the core utility for loading the model, tokenizing inputs, and generating outputs.

# Prompt Engineering 
This file builds on BaseLLM to implement the CoTModel class, which specializes in Chain-of-Thought (CoT) reasoning and in-context learning for unit conversion tasks. It includes:
A format_prompt method that creates a structured chat template to guide the model in reasoning step-by-step and producing accurate answers.
A load function to initialize the CoTModel.
A test_model function to benchmark the model's accuracy and answer rate on a validation dataset.
This file focuses on improving the model's reasoning capabilities by leveraging in-context learning and structured prompts.

#  Fine Tuning
This file implements Supervised Fine-Tuning (SFT) for the SmolLM2 model. It includes:
Functions to tokenize data and format examples for training (tokenize and format_example).
A TokenizedDataset class to prepare datasets for fine-tuning.
A train_model function to fine-tune the model using LoRA (Low-Rank Adaptation) and Hugging Face's Trainer.
A test_model function to evaluate the fine-tuned model's performance on a validation dataset.
A load function to load the fine-tuned model with LoRA adapters.
This file is responsible for adapting the model to perform better on unit conversion tasks by training it on labeled data.

# Results
We were able to achieve a 94% accuracy in training and 93% accuracy in validation.
