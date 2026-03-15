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
  console.log("=== Vector Store Lab ===");
  console.log("Storing 15 sentences in the vector database...");

  // Updated embedding generation logic with 15 diverse sentences with categories
  const sentencesWithCategories = [
    { text: "The canine barked loudly.", category: "animals" },
    { text: "The dog made a noise.", category: "animals" },
    { text: "The electron spins rapidly.", category: "science" },
    { text: "I love eating pizza with extra cheese.", category: "food" },
    { text: "The basketball player scored a three-pointer.", category: "sports" },
    { text: "Rain is forecasted for tomorrow afternoon.", category: "weather" },
    { text: "JavaScript is a popular programming language.", category: "technology" },
    { text: "The kitten purred softly on the couch.", category: "animals" },
    { text: "Quantum mechanics explains particle behavior.", category: "science" },
    { text: "Homemade pasta tastes better than store-bought.", category: "food" },
    { text: "The soccer match ended in a tie.", category: "sports" },
    { text: "Clouds are forming over the mountains.", category: "weather" },
    { text: "TypeScript adds types to JavaScript.", category: "technology" },
    { text: "Puppies need lots of attention and exercise.", category: "animals" },
    { text: "Atoms are made of protons, neutrons, and electrons.", category: "science" }
  ];

  // Add sentences to the vector store with metadata including category
  const documents = sentencesWithCategories.map((item, index) => ({
    pageContent: item.text,
    metadata: { index: index, category: item.category }
  }));

  await vectorStore.addDocuments(documents);

  // Print confirmation message
  console.log(`✅ Successfully stored ${documents.length} sentences\n`);

  // Show stored documents summary
  console.log("=== Vector Store Summary ===");
  console.log(`Documents stored in vector store: ${documents.length}`);
  console.log(`Each document has been converted to an embedding vector.\n`);

  // Function to search sentences in the vector store with optional category filtering
  async function searchSentences(vectorStore, query, k = 3, filterCategory = null) {
    // Perform similarity search with scores
    const results = await vectorStore.similaritySearchWithScore(query, k * 2); // Get more results for filtering

    // Filter by category if specified
    let filteredResults = results;
    if (filterCategory) {
      filteredResults = results.filter(([document]) => 
        document.metadata.category === filterCategory
      );
      // Limit to k results after filtering
      filteredResults = filteredResults.slice(0, k);
    } else {
      filteredResults = results.slice(0, k);
    }

    // Print the results with formatting
    const categoryStr = filterCategory ? ` (Category: ${filterCategory})` : "";
    console.log(`🔍 Search Results for "${query}"${categoryStr}:\n`);
    
    if (filteredResults.length === 0) {
      console.log("❌ No results found.");
    } else {
      filteredResults.forEach(([document, score], index) => {
        console.log(`${index + 1}. [Score: ${score.toFixed(4)}] [${document.metadata.category}] ${document.pageContent}`);
      });
    }

    // Return the filtered results
    return filteredResults;
  }

  // Display header for semantic search
  console.log("=== Semantic Search ===\n");

  // Create readline interface
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  // Display available categories
  const categories = [...new Set(sentencesWithCategories.map(item => item.category))];
  console.log(`Available categories: ${categories.join(", ")}\n`);

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

    // Ask if user wants to filter by category
    const filterChoice = await new Promise((resolve) => {
      rl.question("Filter by category? (Enter category name or press Enter to skip): ", resolve);
    });

    const categoryFilter = filterChoice.trim().toLowerCase() || null;
    
    // Validate category if provided
    if (categoryFilter && !categories.includes(categoryFilter)) {
      console.log(`❌ Invalid category. Available categories: ${categories.join(", ")}\n`);
      continue;
    }

    // Call searchSentences function and display results
    const results = await searchSentences(vectorStore, query, 3, categoryFilter);
    console.log(); // Blank line for readability
  }
}

main().catch(console.error);
