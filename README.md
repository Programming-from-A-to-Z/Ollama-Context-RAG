# Retrieval Augmented Generation

## Overview

This is an example node.js application that utilizes local embeddings and Ollama for text retrieval and response generation. It processes a text corpus, generates embeddings for "chunks" using transformers.js, and uses these embeddings to perform a "similarity search" in response to queries.

- `server.js`: Server file that handles API requests and integrates with Ollama for LLM inference.
- `save-embeddings.js`: Process a text file and generate embeddings using transformers.js.
- `embeddings.json`: Precomputed embeddings generated from the text corpus.
- `public/`: p5.js sketch
- `.env`: Configuration file (no API tokens required)

## References

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [Ollama Documentation](https://ollama.ai/docs)

## Instructions

1. Install and run [Ollama](https://ollama.ai)

```sh
ollama run gemma3
```

2. **Install Dependencies**

```sh
npm install
```

3. **Generate embeddings**: Run `save-embeddings.js` to create the `embeddings.json` file.

The code reads from `data/processing-course.txt` by default. To use your own text corpus, modify the filename and adjust how the text is split:

```js
const raw = fs.readFileSync('data/your-text-file.txt', 'utf-8');
let chunks = raw.split(/\n+/); // Adjust splitting logic as needed
```

Then run:

```sh
node save-embeddings.js
```

4. **Run the server**:

```sh
node server.js
```

Open your browser to: `http://localhost:3001` (or whatever port is specified)

- All processing runs locally - no external API keys required
- First run will download the transformers.js models (cached for future use)
- The embedding model is lightweight (~50MB)
- You can use any model available in Ollama for LLM inference
