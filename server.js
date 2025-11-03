import { pipeline } from '@huggingface/transformers';
import express from 'express';
import bodyParser from 'body-parser';
import * as fs from 'fs';

// Read and parse the embeddings from the file
const { embeddings } = JSON.parse(fs.readFileSync('data/embeddings.json', 'utf-8'));
console.log('Embeddings loaded.');

// Initialize Express application
const app = express();
app.use(bodyParser.json()); // Middleware for parsing JSON bodies
app.use(express.static('public')); // Serve static files from 'public' directory

// Initialize the embedding model
let extractor;
console.log('Loading embedding model...');
extractor = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-xsmall-v1');
console.log('Embedding model loaded successfully!');

// Function to get embedding for a given text
async function getEmbedding(text) {
  console.log(`Generating embedding for test query: "${text}"`);
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Endpoint to find similar texts based on embeddings
app.post('/api/similar', async (request, response) => {
  let prompt = request.body.prompt;
  console.log('API /similar called. Searching for similarities to: ' + prompt);
  let n = request.body.n || 10;
  let similarities = await findSimilar(prompt);
  similarities = similarities.slice(0, n);
  response.json(similarities);
});

// Endpoint to query with context from similar texts
app.post('/api/query', async (request, response) => {
  let prompt = request.body.prompt;
  console.log('API /query called. Prompt: ' + prompt);
  let n = request.body.n || 10;
  let similarities = await findSimilar(prompt);
  similarities = similarities.slice(0, n);
  let answer = await queryLLM(prompt, similarities);
  response.json({ prompt, answer, similarities });
});

// Function to generate a response using the LLM via Ollama
async function queryLLM(prompt, knowledge) {
  console.log('Querying LLM with knowledge length: ' + knowledge.length);
  const formattedPrompt = createPrompt(prompt, knowledge);
  console.log(formattedPrompt);

  const ollamaResponse = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gemma3',
      messages: [
        {
          role: 'user',
          content: formattedPrompt,
        },
      ],
      stream: false,
    }),
  });

  const data = await ollamaResponse.json();
  console.log(data);
  return data.message.content;
}

// Function to create a prompt with given context and query
function createPrompt(prompt, knowledge) {
  console.log('Creating prompt with context.');
  const context = knowledge.map((item) => item.text).join('\n');
  // Assemble the prompt with context and instructions
  return `Context for the query is provided below. Use this information to answer the query.
---------------------
${context}
---------------------
Instructions:
- Use ONLY the provided context to answer the query.
- Do not use external knowledge or assumptions.
- Provide a clear and concise answer.
- Do not refer to the context or the speaker in your response.
Query: ${prompt}
Answer: `;
}

// Function to find similar texts based on cosine similarity
async function findSimilar(prompt) {
  console.log('Finding similar texts for: ' + prompt);
  const inputEmbedding = await getEmbedding(prompt);
  // Calculate similarity of each embedding with the input
  let similarities = embeddings.map(({ text, embedding }) => ({
    text,
    similarity: cosineSimilarity(inputEmbedding, embedding),
  }));
  // Sort similarities in descending order
  similarities = similarities.sort((a, b) => b.similarity - a.similarity);
  return similarities;
}

// Start the server on the specified port
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

// Cosine Similarity Functions
function dotProduct(vecA, vecB) {
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
}

function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

function cosineSimilarity(vecA, vecB) {
  return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}
