import { pipeline } from '@huggingface/transformers';
import * as fs from 'fs';

// Initialize the embedding model
let extractor;

// Function to get embedding for a given text
async function getEmbedding(text) {
  console.log(`Generating embedding for test query: "${text}"`);
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Load pre-computed embeddings from file
const { embeddings } = JSON.parse(fs.readFileSync('data/embeddings.json', 'utf-8'));

// Test function to demonstrate embeddings search
async function test() {
  // Load the embedding model
  console.log('Loading embedding model...');
  extractor = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-xsmall-v1');
  console.log('Model loaded successfully!');

  const prompt = "How do I set a shape's color?";
  console.log(`Test query: "${prompt}"`);
  const inputEmbedding = await getEmbedding(prompt);

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
}

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

// Call the test function
test();
