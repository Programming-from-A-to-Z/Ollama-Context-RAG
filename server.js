import express from 'express';
import ollama from 'ollama';
import * as fs from 'fs';

// Retrieval Augmented Generation (RAG) with Ollama
// 1. save-embeddings.js made an embedding for every line of the transcripts (data/embeddings.json)
// 2. Retrieval (/api/similar): make an embedding for the question and find the most similar lines
// 3. Augmented generation (/api/query): paste those lines into the prompt for a chat model

// Both models run in Ollama
const embeddingModel = 'embeddinggemma-2:270m'; // turns text into embeddings
const chatModel = 'gemma4'; // writes the answers
// For a smaller download (4.3 GB instead of 6.6 to 9.5 GB), use 'gemma4:e2b-it-qat'

// Read and parse the embeddings from the file
const { model, embeddings } = JSON.parse(fs.readFileSync('data/embeddings.json', 'utf-8'));
console.log(`Loaded ${embeddings.length} embeddings made with ${model}`);

// Initialize Express application
const app = express();
app.use(express.json());
app.use(express.static('public'));

// Function to get embedding for a question
async function getEmbedding(text) {
  console.log(`Generating embedding for query: "${text}"`);
  // Questions get a different prefix than the documents in embeddings.json
  const response = await ollama.embed({
    model: embeddingModel,
    input: 'task: question answering | query: ' + text,
  });
  return response.embeddings[0];
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

  const reply = await ollama.chat({
    model: chatModel,
    messages: [
      {
        role: 'user',
        content: formattedPrompt,
      },
    ],
  });

  console.log(reply.message.content);
  return reply.message.content;
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

// Start the server
const port = 3001;
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
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
