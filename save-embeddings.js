import ollama from 'ollama';
import * as fs from 'fs';

// The embedding model, running in Ollama: EmbeddingGemma 2
// (download it first with: ollama pull embeddinggemma-2:270m)
const embeddingModel = 'embeddinggemma-2:270m';

// Read and split the data into "chunks"
console.log('Reading source file...');
const raw = fs.readFileSync('data/processing-course.txt', 'utf-8');
let chunks = raw.split(/\n+/);

// Trim each chunk and filter out empty strings
chunks = chunks.map((chunk) => chunk.trim()).filter((chunk) => chunk !== '');
console.log(`Total chunks to process: ${chunks.length}`);

// Array to store all embeddings
let embeddings = [];

// Process chunks one by one
for (let i = 0; i < chunks.length; i++) {
  process.stdout.write(`\rProcessing chunk ${i + 1}/${chunks.length}`);
  const text = chunks[i];

  // EmbeddingGemma 2 was trained with a short prefix in front of each text.
  // Documents (the things being searched) start with "title: none | text: "
  const response = await ollama.embed({
    model: embeddingModel,
    input: 'title: none | text: ' + text,
  });

  // Ollama sends back a list of embeddings (one for each input), so take the first
  embeddings.push({
    text,
    embedding: response.embeddings[0],
  });
}

// Write the embeddings to a JSON file, along with the name of the model that made them
const fileOut = 'data/embeddings.json';
fs.writeFileSync(fileOut, JSON.stringify({ model: embeddingModel, embeddings }));
console.log(`\nEmbeddings saved to ${fileOut}`);
