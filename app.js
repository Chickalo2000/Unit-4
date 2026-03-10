import { OpenAIEmbeddings } from "@langchain/openai";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

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

  // Print lab header
  console.log("=== Embedding Inspector Lab ===");
  console.log("Generating embeddings for three sentences...");

  // Updated test sentences to focus on pizza-related examples
  const sentences = [
    "I like pizza.",
    "I really love eating delicious, hot pizza.",
    "Pizza is good."
  ];

  // Generate embeddings for each sentence
  const embeddingsArray = [];
  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    console.log(`Sentence ${i + 1}: ${sentence}`);
    const embedding = await embeddings.embedQuery(sentence);
    embeddingsArray.push(embedding);
  }

  // Show the distances between the embeddings
  console.log("\n=== Cosine Similarities ===\n");
  for (let i = 0; i < embeddingsArray.length; i++) {
    const current = i;
    const next = (i + 1) % embeddingsArray.length;
    const similarity = cosineSimilarity(embeddingsArray[current], embeddingsArray[next]);
    console.log(`Cosine similarity between Sentence ${current + 1} and Sentence ${next + 1}: ${similarity.toFixed(4)}`);
  }

  console.log("\n📊 Observations:");
  console.log("- Each embedding is just an array of floating-point numbers");
  console.log("- Sentences 1 and 2 (about dogs) will have similar values in many dimensions");
  console.log("- Sentence 3 (about electrons) will differ significantly from sentences 1 and 2");
  console.log("\nThis demonstrates that 'AI embeddings' are simply numerical vectors,");
  console.log("not magic—they represent semantic meaning as coordinates in high-dimensional space.");
}

main().catch(console.error);
