# OPEN Insights – Multi-Modal Retrieval and Generation

This project demonstrates a powerful pipeline that combines **PDF content extraction**, **image-text embeddings**, **vision-based captioning**, and **text generation** into a unified workflow. It leverages **Hugging Face models** for vision and text processing, paired with a **vector database** for efficient retrieval, making it ideal for multi-modal search and content generation tasks.

---

## Project Overview

The pipeline performs the following tasks:

1. **Text Extraction**: Extracts raw text from PDF files.
2. **Image Extraction**: Retrieves images embedded in PDFs.
3. **Embedding Generation**: Uses `openai/clip-vit-base-patch32` to create vector representations for both text and images.
4. **Vector Storage and Search**: Stores embeddings in ChromaDB for fast similarity searches.
5. **Image Captioning & Vision Tasks**: Employs `Salesforce/blip2-opt-6.7b` to generate captions or answer questions about images.
6. **Text Generation**: Utilizes GPT-2 to generate or expand text based on queries.

This setup enables:
- Searching for related text or images in a dataset using text or image queries.
- Automatically generating meaningful captions for images.
- Creating or extending text based on user input.

---

## Features

- **PDF Parsing**: Extracts both text and images from any PDF file.
- **Multi-Modal Embeddings**: Generates embeddings for text and images using CLIP for unified search.
- **Image Captioning**: Produces detailed captions or answers image-related questions using BLIP-2.
- **Text Generation**: Expands or generates new text with GPT-2.
- **Vector Search**: Efficiently retrieves relevant content using ChromaDB.
- **Multi-Modal Search**: Supports queries with either text or images.

---

## Requirements

To run this project, ensure you have the following:

- **Python**: Version 3.8 or higher.
- **Libraries**:
  - `torch`: For model computation.
  - `transformers`: For Hugging Face models (CLIP, BLIP-2, GPT-2).
  - `datasets`: For data handling.
  - `chromadb`: For vector storage and search.
  - `pymupdf`: For PDF parsing (via `fitz`).
  - `pillow`: For image processing.

Install the dependencies using:

```bash
pip install torch transformers datasets chromadb pymupdf pillow
```

---

## Workflow

### Step 1: Extract Text from PDF
- Use **PyMuPDF** (`fitz`) to load the PDF.
- Iterate through pages to extract raw textual content.

### Step 2: Extract Images from PDF
- Access image objects from each PDF page.
- Save images locally for further processing.

### Step 3: Generate Embeddings
- Load the **CLIP model** (`openai/clip-vit-base-patch32`) from Hugging Face.
- Encode both text and images into vector representations for similarity search.

### Step 4: Store and Search Embeddings
- Initialize a **ChromaDB** instance.
- Store text and image embeddings in the vector database.
- Perform similarity searches using text or image queries.

### Step 5: Vision Tasks
- Load the **BLIP-2 model** (`Salesforce/blip2-opt-6.7b`) for vision tasks.
- Generate:
  - **Generic Captions**: Describe images automatically.
  - **Question-Answering**: Answer specific questions about images.

### Step 6: Text Generation
- Load **GPT-2** for lightweight text generation.
- Extend retrieved text or generate new content based on user queries.

---

## Transformer Model Reference

The **Transformer architecture** is central to the models used in this project (CLIP, BLIP-2, GPT-2). Below is a brief overview:

- **Encoder**: Processes input data (text or images) to generate contextual representations.
- **Decoder**: Produces outputs based on encoder representations and prior predictions.
- **Multi-Head Attention**: Enables the model to focus on multiple parts of the input sequence simultaneously, capturing diverse relationships.

### Transformer Diagram
```
[Input] → [Encoder: Multi-Head Attention + Feed-Forward] → [Contextual Representations]
        → [Decoder: Multi-Head Attention + Feed-Forward] → [Output]
```

---

## Example Outputs

### A. Vision Model Outputs (`Salesforce/blip2-opt-6.7b`)
1. **Image-Only Caption**:
   - **Input**: An image of a block diagram for a data processing system.
   - **Output**: "A block diagram illustrating a data processing system with interconnected components."

2. **Question-Answering Caption**:
   - **Question**: "Explain this diagram in detail, focusing on how multi-head attention works."
   - **Answer**: "Multi-head attention allows the model to simultaneously focus on different parts of the input sequence, improving its ability to capture complex relationships. Each 'head' attends to a specific aspect of the data, and their outputs are combined to form a comprehensive representation."

### B. Text Generation Output (`GPT-2`)
- **Input Query**: "Explain the significance of attention mechanisms in transformers."
- **Output**: "Attention mechanisms allow transformers to weigh the importance of different parts of the input sequence, enabling better handling of long-range dependencies. Multi-head attention enhances this by processing the input through multiple parallel attention layers, capturing diverse patterns and relationships."

### C. Search Outputs
- **Text Search**: Retrieves the most similar text segments from the PDF based on a text query.
- **Image Search**: Finds images in the dataset that closely match a query image or text description.

---

## Model References

- **CLIP** (`openai/clip-vit-base-patch32`): Generates multi-modal embeddings for text and images.
- **BLIP-2** (`Salesforce/blip2-opt-6.7b`): Handles image captioning and vision-based question-answering.
- **GPT-2** (`gpt2`): Performs lightweight and fast text generation.

---

## How to Run

1. **Prepare the PDF**:
   - Place your target PDF file in the working directory.
   - Update the file path in the script or notebook.

2. **Run the Pipeline**:
   - Execute the script or notebook step-by-step to:
     - Extract text and images from the PDF.
     - Generate embeddings for text and images.
     - Store embeddings in ChromaDB.
     - Perform similarity searches.
     - Generate captions for images and text based on queries.

3. **Example Command**:
   ```bash
   python main.py --pdf_path "path/to/your/pdf"
   ```

---

## Output Examples

- **Text Search Result**: Retrieves the top matching text segments from the PDF based on a query like "transformer architecture."
- **Image Search Result**: Returns images from the PDF that match a query (e.g., "diagram of a neural network").
- **Generated Caption**: "A flowchart depicting the data flow in a machine learning pipeline."
- **Generated Text**: "The transformer architecture revolutionized natural language processing by introducing attention mechanisms that prioritize relevant parts of the input sequence."

---

## Notes

- Ensure sufficient computational resources (GPU recommended) for running BLIP-2 and CLIP models.
- ChromaDB requires a local or hosted instance for vector storage.
- For large PDFs, optimize memory usage by processing pages in batches.
- Check model compatibility with your hardware (e.g., BLIP-2 is resource-intensive).

This project provides a flexible framework for multi-modal content processing, suitable for applications like document analysis, automated summarization, and visual question-answering.