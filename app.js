import { OpenAIEmbeddings } from "@langchain/openai";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

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

  // Create OpenAIEmbeddings instance
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    configuration: {
      baseURL: "https://models.inference.ai.azure.com",
      apiKey: process.env.GITHUB_TOKEN
    },
    check_embedding_ctx_length: false
  });

  // Print lab header
  console.log("=== Embedding Inspector Lab ===");
  console.log("Generating embeddings for three sentences...");

  // Test sentences
  const sentences = [
    "The canine barked loudly.",
    "The dog made a noise.",
    "The electron spins rapidly."
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
  console.log("\n=== Embedding Vectors ===\n");
  for (let i = 0; i < embeddingsArray.length; i++) {
    const current = i;
    const next = (i + 1) % embeddingsArray.length;
    const similarity = cosineSimilarity(embeddingsArray[current], embeddingsArray[next]);
    console.log(`Cosine similarity between Sentence ${current + 1} and Sentence ${next + 1}: ${similarity.toFixed(4)}`);
  }

  // Calculate and display cosine similarities
  const similarity12 = cosineSimilarity(embeddingsArray[0], embeddingsArray[1]).toFixed(4);
  const similarity23 = cosineSimilarity(embeddingsArray[1], embeddingsArray[2]).toFixed(4);
  const similarity31 = cosineSimilarity(embeddingsArray[2], embeddingsArray[0]).toFixed(4);

  console.log(`Cosine Similarity between Sentence 1 and Sentence 2: ${similarity12}`);
  console.log(`Cosine Similarity between Sentence 2 and Sentence 3: ${similarity23}`);
  console.log(`Cosine Similarity between Sentence 3 and Sentence 1: ${similarity31}`);

  console.log("\n📊 Observations:");
  console.log("- Each embedding is just an array of floating-point numbers");
  console.log("- Sentences 1 and 2 (about dogs) will have similar values in many dimensions");
  console.log("- Sentence 3 (about electrons) will differ significantly from sentences 1 and 2");
  console.log("\nThis demonstrates that 'AI embeddings' are simply numerical vectors,");
  console.log("not magic—they represent semantic meaning as coordinates in high-dimensional space.");
}

main().catch(console.error);
