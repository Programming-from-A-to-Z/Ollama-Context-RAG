import ollama from 'ollama';
import * as fs from 'fs';

// The same embedding model as save-embeddings.js
const embeddingModel = 'embeddinggemma-2:270m';

// Load pre-computed embeddings from file
const { model, embeddings } = JSON.parse(fs.readFileSync('data/embeddings.json', 'utf-8'));
console.log(`Loaded ${embeddings.length} embeddings made with ${model}`);

// A test question
const prompt = "How do I set a shape's color?";
console.log(`Test query: "${prompt}"`);

// Questions get a different prefix than the documents
const response = await ollama.embed({
  model: embeddingModel,
  input: 'task: question answering | query: ' + prompt,
});
const inputEmbedding = response.embeddings[0];

// Calculate similarity of the test query with each stored embedding
let similarities = embeddings.map(({ text, embedding }) => ({
  text,
  similarity: cosineSimilarity(inputEmbedding, embedding),
}));
// Sort the results by similarity in descending order
similarities = similarities.sort((a, b) => b.similarity - a.similarity);

// Display the top 10 results
console.log('Top 10 Results:');
similarities = similarities.slice(0, 10);
similarities.forEach((item, index) => {
  console.log(`${index + 1}: ${item.text} (Similarity: ${item.similarity.toFixed(3)})`);
});

// Functions to calculate cosine similarity
function dotProduct(vecA, vecB) {
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
}
function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}
function cosineSimilarity(vecA, vecB) {
  return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}
