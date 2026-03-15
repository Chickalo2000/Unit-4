import { OpenAIEmbeddings } from "@langchain/openai";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import readline from "readline";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Get directory of current file
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from .env file
dotenv.config({ path: path.join(__dirname, ".env") });

// Set OpenAI base URL for GitHub Models
process.env.OPENAI_API_BASE = "https://models.inference.ai.azure.com";
process.env.OPENAI_API_KEY = process.env.GITHUB_TOKEN;

/**
 * Calculate cosine similarity between two vectors
 * Cosine similarity = (A · B) / (||A|| * ||B||)
 */
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    throw new Error("Vectors must have the same dimensions");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    normA += vectorA[i] * vectorA[i];
    normB += vectorB[i] * vectorB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {
  console.log("🤖 JavaScript LangChain Agent Starting...\n");

  // Check for GitHub token
  if (!process.env.GITHUB_TOKEN) {
    console.error("❌ Error: GITHUB_TOKEN not found in environment variables.");
    console.log("Please create a .env file with your GitHub token:");
    console.log("GITHUB_TOKEN=your-github-token-here");
    console.log("\nGet your token from: https://github.com/settings/tokens");
    console.log("Or use GitHub Models: https://github.com/marketplace/models");
    process.exit(1);
  }

  // Create OpenAIEmbeddings instance with GitHub Models endpoint
  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.GITHUB_TOKEN,
    model: "text-embedding-3-small",
    configuration: {
      baseURL: "https://models.inference.ai.azure.com"
    },
    check_embedding_ctx_length: false
  });

  // Initialize MemoryVectorStore
  const vectorStore = new MemoryVectorStore(embeddings);

  // Print lab header
  console.log("=== Embedding Inspector Lab ===");
  console.log("Generating embeddings for three sentences...");

  // Updated embedding generation logic with semantically rich sentences
  const sentences = [
    "The dog barks loudly at the mailman.",
    "The cat meows and hisses when disturbed.",
    "Electrons are subatomic particles that spin rapidly around the nucleus."
  ];

  // Add sentences to the vector store with metadata
  const documents = sentences.map((sentence, index) => ({
    pageContent: sentence,
    metadata: {
      createdAt: new Date().toISOString(),
      index: index
    }
  }));

  await vectorStore.addDocuments(documents);

  // Print confirmation message
  console.log(`✅ Successfully stored ${documents.length} sentences in the vector store.`);

  // Print each sentence that was added
  documents.forEach((doc, idx) => {
    console.log(`Sentence ${idx + 1}: ${doc.pageContent}`);
  });

  // Adding additional sentences to the MemoryVectorStore
  const additionalSentences = [
    { pageContent: "Elephants are the largest land animals.", metadata: { category: "animals" } },
    { pageContent: "The Moon orbits the Earth.", metadata: { category: "astronomy" } },
    { pageContent: "Whales are mammals that live in the ocean.", metadata: { category: "animals" } },
    { pageContent: "Mount Everest is the tallest mountain on Earth.", metadata: { category: "geography" } }
  ];

  // Add the additional sentences to the vector store
  await vectorStore.addDocuments(additionalSentences);

  console.log("Additional sentences added to the vector store!");

  // Adding sentences about animals and their noises
  const animalNoises = [
    { pageContent: "Dogs bark to communicate.", metadata: { category: "animal sounds" } },
    { pageContent: "Cats meow when they want attention.", metadata: { category: "animal sounds" } },
    { pageContent: "Cows moo in the fields.", metadata: { category: "animal sounds" } },
    { pageContent: "Sheep bleat to stay in touch with their flock.", metadata: { category: "animal sounds" } },
    { pageContent: "Lions roar to mark their territory.", metadata: { category: "animal sounds" } }
  ];

  // Add the animal noises to the vector store
  await vectorStore.addDocuments(animalNoises);

  console.log("Animal noises added to the vector store!");

  // Adding more sentences about science
  const moreScienceSentences = [
    { pageContent: "The speed of light is approximately 299,792 kilometers per second.", metadata: { category: "science" } },
    { pageContent: "DNA carries the genetic instructions for life.", metadata: { category: "science" } },
    { pageContent: "Newton's laws of motion describe the relationship between a body and the forces acting on it.", metadata: { category: "science" } },
    { pageContent: "The water cycle includes evaporation, condensation, and precipitation.", metadata: { category: "science" } },
    { pageContent: "Black holes are regions of space where gravity is so strong that nothing can escape.", metadata: { category: "science" } }
  ];

  // Add the additional science sentences to the vector store
  await vectorStore.addDocuments(moreScienceSentences);

  console.log("More science sentences added to the vector store!");

  // Expanding the sentences array to include diverse topics
  const expandedSentences = [
    { pageContent: "Dogs are loyal and friendly pets.", metadata: { category: "animals" } },
    { pageContent: "Cats love to climb and explore their surroundings.", metadata: { category: "animals" } },
    { pageContent: "The Earth revolves around the Sun in 365 days.", metadata: { category: "science" } },
    { pageContent: "Gravity keeps us grounded on the surface of the Earth.", metadata: { category: "science" } },
    { pageContent: "Pasta is a versatile dish that can be cooked in many ways.", metadata: { category: "food" } },
    { pageContent: "Baking a cake requires precise measurements of ingredients.", metadata: { category: "food" } },
    { pageContent: "Soccer is the most popular sport in the world.", metadata: { category: "sports" } },
    { pageContent: "Swimming is a great way to stay fit and healthy.", metadata: { category: "sports" } },
    { pageContent: "Rain is essential for the growth of plants and crops.", metadata: { category: "weather" } },
    { pageContent: "Thunderstorms often occur during the summer months.", metadata: { category: "weather" } },
    { pageContent: "Artificial intelligence is transforming the way we work and live.", metadata: { category: "technology" } },
    { pageContent: "Programming languages like JavaScript are used to build web applications.", metadata: { category: "technology" } },
    { pageContent: "Hiking in the mountains is a refreshing outdoor activity.", metadata: { category: "nature" } },
    { pageContent: "The Amazon rainforest is home to diverse species of plants and animals.", metadata: { category: "nature" } }
  ];

  // Add the expanded sentences to the vector store
  await vectorStore.addDocuments(expandedSentences);

  console.log("Expanded sentences covering diverse topics added to the vector store!");

  // Show stored documents summary
  console.log("\n=== Vector Store Summary ===");
  console.log(`Documents stored in vector store: ${documents.length}`);
  console.log(`Each document has been converted to an embedding vector.\n`);

  // Function to search sentences in the vector store
  async function searchSentences(vectorStore, query, k = 3) {
    // Perform similarity search with scores
    const results = await vectorStore.similaritySearchWithScore(query, k);

    // Print the results with formatting
    console.log(`\n🔍 Search Results for Query: "${query}"\n`);
    results.forEach(([document, score], index) => {
      console.log(`Rank ${index + 1}:`);
      console.log(`  Similarity Score: ${score.toFixed(4)}`);
      console.log(`  Sentence: ${document.pageContent}`);
    });

    // Return the top k results
    return results;
  }

  // Display header for semantic search
  console.log("\n=== Semantic Search ===\n");

  // Create readline interface
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  // Interactive loop for search queries
  while (true) {
    const query = await new Promise((resolve) => {
      rl.question("Enter a search query (or 'quit' to exit): ", resolve);
    });

    if (query.trim().toLowerCase() === "quit" || query.trim().toLowerCase() === "exit") {
      console.log("\n👋 Goodbye! Thanks for using Semantic Search.");
      rl.close();
      break;
    }

    if (!query.trim()) {
      console.log("⚠️ Please enter a valid search query.");
      continue;
    }

    // Call searchSentences function and display results
    const results = await searchSentences(vectorStore, query);
    console.log(); // Blank line for readability
  }
}

main().catch(console.error);
