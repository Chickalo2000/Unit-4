import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { readFile } from "node:fs/promises";
import path from "node:path";
import dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

// BEST PRACTICE: Use fs/promises for better async/await support
// Synchronous operations block the event loop; prefer promises API

// TOKEN LIMIT CONSTANT - Based on OpenAI documentation
// text-embedding-3-small has a maximum token limit of 8,192 tokens
const MAX_TOKENS = 8192;

// Estimate tokens in text (rough approximation: 1 token ≈ 4 characters)
function estimateTokenCount(text) {
  return Math.ceil(text.length / 4);
}

// Initialize embeddings and vector store
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  configuration: {
    baseURL: "https://models.inference.ai.azure.com",
  },
});

const vectorStore = new MemoryVectorStore(embeddings);

// Function to load a document from file with token limit validation
async function loadDocument(vectorStore, filePath) {
  try {
    console.log(`\nLoading ${path.basename(filePath)}...`);

    // BEST PRACTICE: Use readFile (promises API) instead of readFileSync
    // This prevents blocking the event loop and allows concurrent operations
    // No need to pre-check file existence - handle errors directly
    let fileContent = await readFile(filePath, "utf-8");

    // BEST PRACTICE: Replace newlines with spaces for consistent tokenization
    // This follows OpenAI's recommendation for text preprocessing
    fileContent = fileContent.replace(/\n/g, " ");

    // Estimate token count before attempting to embed
    // This prevents hitting token limit errors
    const estimatedTokens = estimateTokenCount(fileContent);

    if (estimatedTokens > MAX_TOKENS) {
      console.error(
        `\n❌ Error loading ${path.basename(filePath)}`
      );
      console.log(
        `⚠️ This document is too large to embed as a single chunk.`
      );
      console.log(
        `Estimated tokens: ${estimatedTokens} / Maximum allowed: ${MAX_TOKENS}`
      );
      console.log(
        `Token limit exceeded. The embedding model can only process up to 8,192 tokens at once.`
      );
      console.log(
        `Solution: The document needs to be split into smaller chunks.`
      );
      return null;
    }

    // Create a LangChain Document object with metadata
    const document = new Document({
      pageContent: fileContent,
      metadata: {
        fileName: path.basename(filePath),
        createdAt: new Date().toISOString(),
        estimatedTokens: estimatedTokens,
      },
    });

    // Add document to vector store
    await vectorStore.addDocuments([document]);

    console.log(
      `✅ Successfully loaded ${path.basename(filePath)} (${fileContent.length} characters, ~${estimatedTokens} tokens)`
    );

    return document.id;
  } catch (error) {
    // BEST PRACTICE: Handle all errors directly without pre-checking
    // This follows Node.js best practices for file operations
    console.error(`❌ Error loading ${path.basename(filePath)}`);

    if (error.code === "ENOENT") {
      console.log(`File not found: ${path.basename(filePath)}`);
    } else if (
      error.message.includes("maximum context length") ||
      error.message.includes("token")
    ) {
      console.log("⚠️ This document is too large to embed as a single chunk.");
      console.log(
        "Token limit exceeded. The embedding model can only process up to 8,191 tokens at once."
      );
      console.log("Solution: The document needs to be split into smaller chunks.");
    } else {
      console.error(error.message);
    }
  }
}

// Main function
async function main() {
  console.log("🤖 JavaScript LangChain Agent Starting...\n");

  console.log("=== Loading Documents into Vector Database ===");

  // Load the health insurance brochure (small document)
  await loadDocument(
    vectorStore,
    path.join(process.cwd(), "HealthInsuranceBrochure.md")
  );

  // Load the employee handbook (large document - will hit token limit)
  await loadDocument(
    vectorStore,
    path.join(process.cwd(), "EmployeeHandbook.md")
  );

  console.log("\n=== Document Loading Complete ===\n");
}

// Run the application
main().catch(console.error);
