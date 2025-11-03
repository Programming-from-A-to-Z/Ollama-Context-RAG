import { pipeline } from '@huggingface/transformers';
import * as fs from 'fs';

// Initialize the embedding model
let extractor;

// Read and split the data into "chunks"
console.log('Reading source file...');
const raw = fs.readFileSync('data/processing-course.txt', 'utf-8');
let chunks = raw.split(/\n+/);

// Trim each chunk and filter out empty strings
chunks = chunks.map((chunk) => chunk.trim()).filter((chunk) => chunk !== '');
console.log(`Total chunks to process: ${chunks.length}`);

// Start the process of generating embeddings
console.log('Starting embedding generation...');
createEmbeddings(chunks);

// Function to create embeddings for each chunk
async function createEmbeddings(chunks) {
  // Initialize the transformers.js pipeline
  console.log('Loading embedding model...');
  extractor = await pipeline(
    'feature-extraction',
    'mixedbread-ai/mxbai-embed-xsmall-v1'
  );
  console.log('Model loaded successfully!');

  // Array to store all embeddings
  let embeddings = [];

  // Process chunks one by one
  for (let i = 0; i < chunks.length; i++) {
    process.stdout.write(`\rProcessing chunk ${i + 1}/${chunks.length}`);
    const text = chunks[i];

    // Generate embedding for the current chunk
    const output = await extractor(text, { pooling: 'mean', normalize: true });

    // Store the text with its embedding
    embeddings.push({
      text,
      embedding: Array.from(output.data),
    });
  }

  const jsonOut = { embeddings };
  // Write the embeddings to a JSON file
  const fileOut = 'data/embeddings.json';
  fs.writeFileSync(fileOut, JSON.stringify(jsonOut));
  console.log(`\nEmbeddings saved to ${fileOut}`);
}
