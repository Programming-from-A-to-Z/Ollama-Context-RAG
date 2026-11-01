# Retrieval Augmented Generation

## Overview

This is an example node.js application that answers questions about a text with two models running locally in [Ollama](https://ollama.com): one turns text into embeddings, the other writes the answer. It splits a text corpus into "chunks," saves an embedding for each one, and uses these embeddings to perform a "similarity search" in response to a question. The most similar chunks go into the prompt, and the language model answers from them.

- `save-embeddings.js`: Process a text file and generate an embedding for each chunk.
- `test-embeddings.js`: Try a search from the command line (no LLM).
- `server.js`: Server file that handles API requests (search and answer), using the [`ollama` JavaScript library](https://github.com/ollama/ollama-js).
- `data/processing-course.txt`: The text corpus, transcripts from the Learning Processing video series.
- `data/embeddings.json`: Precomputed embeddings generated from the text corpus.
- `public/`: p5.js sketch

## Models

- **Embeddings:** [EmbeddingGemma 2](https://ollama.com/library/embeddinggemma-2) (Google DeepMind, October 2026). The `270m` version is text only (about 300–380 MB), which is all this example needs.
- **Answers:** [Gemma 4](https://ollama.com/library/gemma4). Any chat model in Ollama works: change `chatModel` at the top of `server.js`.

EmbeddingGemma 2 was trained with a short prefix in front of each text, so the code adds one:

- Each chunk of the corpus: `title: none | text: ...`
- Each question: `task: question answering | query: ...`

These prefixes belong to EmbeddingGemma 2. Other embedding models use different prefixes or none, so check the model card if you switch.

The same recipe also runs entirely in the browser, with no server and no Ollama: Ask the Syllabus uses Transformers.js, a different text (the class syllabus), a simpler embedding model with no prefixes (all-MiniLM-L6-v2), and a smaller language model. It's linked from the [Embeddings notes](https://github.com/Programming-from-A-to-Z/A2Z-F26/tree/main/07-embeddings).

## References

- [Ollama embeddings documentation](https://docs.ollama.com/capabilities/embeddings)
- [Ollama JavaScript Library](https://github.com/ollama/ollama-js)
- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)

## Instructions

1. Install and run [Ollama](https://ollama.com), then download both models:

```sh
ollama pull embeddinggemma-2:270m
ollama pull gemma4
```

`gemma4` is the same model as last week's [Ollama-Examples](https://github.com/Programming-from-A-to-Z/Ollama-Examples), so you may already have it. If not, it's a big download (6.6–9.5 GB, depending on your computer). For a smaller one, pull `gemma4:e2b-it-qat` instead (4.3 GB) and change `chatModel` at the top of `server.js` to match.

2. **Install Dependencies**

```sh
npm install
```

3. **Generate embeddings**: Run `save-embeddings.js` to create the `embeddings.json` file. (One is already included for `processing-course.txt`, so you can skip this step to try it out.)

The code reads from `data/processing-course.txt` by default. To use your own text corpus, modify the filename and adjust how the text is split:

```js
const raw = fs.readFileSync('data/your-text-file.txt', 'utf-8');
let chunks = raw.split(/\n+/); // Adjust splitting logic as needed
```

Then run:

```sh
node save-embeddings.js
```

4. **Try a search** (optional): `test-embeddings.js` prints the 10 chunks most similar to a test question.

```sh
node test-embeddings.js
```

5. **Run the server**:

```sh
node server.js
```

Open your browser to: `http://localhost:3001` (or whatever port is specified)

- All processing runs locally - no external API keys required
- If you change the embedding model (`embeddingModel` at the top of `save-embeddings.js`, `test-embeddings.js`, and `server.js`), run `save-embeddings.js` again: a question can only be compared with embeddings from the same model
- You can use any model available in Ollama for LLM inference
