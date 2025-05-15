// app.js
import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import bodyParser from 'body-parser';
import fs from 'fs/promises';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { CallbackManager } from '@langchain/core/callbacks/manager';

const app = express();
const PORT = 3000;
app.use(bodyParser.json());

// Callback que rastreia tokens e custo
let lastUsage = null;
const callbackManager = CallbackManager.fromHandlers({
  handleLLMEnd: async (output) => {
    lastUsage = output.llmOutput?.tokenUsage || null;
  },
});

// Modelo com rastreamento de uso
const model = new ChatOpenAI({
  temperature: 0,
  callbackManager,
});

// Carrega e vetorializa qualquer base textual
async function loadRetrieverFromFile(filePath) {
  const content = await fs.readFile(filePath, 'utf8');
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });
  const docs = await splitter.createDocuments([content]);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );
  return vectorStore.asRetriever();
}

app.post('/processar', async (req, res) => {
  try {
    const { prompt } = req.body;
    const fonte = req.query.fonte;

    if (!prompt || !fonte) {
      return res
        .status(400)
        .json({ error: 'Prompt e fonte são obrigatórios.' });
    }

    // Determina o arquivo com base na fonte escolhida
    const filePath =
      fonte === '1'
        ? './data/tratado/fileT.json'
        : fonte === '2'
        ? './data/bruto/fileB.txt'
        : null;

    if (!filePath) {
      return res
        .status(400)
        .json({ error: 'Fonte inválida. Use ?fonte=1 ou ?fonte=2.' });
    }

    lastUsage = null;

    const retriever = await loadRetrieverFromFile(filePath);
    const chain = RetrievalQAChain.fromLLM(model, retriever);
    const result = await chain.call({ query: prompt });

    let usage = null;
    if (lastUsage) {
      const { promptTokens, completionTokens, totalTokens } = lastUsage;
      const inputCost = (promptTokens * 0.01) / 1000;
      const outputCost = (completionTokens * 0.03) / 1000;
      const totalCost = inputCost + outputCost;
      usage = {
        promptTokens,
        completionTokens,
        totalTokens,
        estimatedCostUSD: totalCost.toFixed(6),
      };
    }

    res.json({
      prompt,
      fonte,
      response: result.text,
      usage,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Erro interno.' });
  }
});

app.listen(PORT, () => {
  console.log(`✅ API rodando em http://localhost:${PORT}`);
});
