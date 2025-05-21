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

// Monitoramento de memória e CPU
setInterval(() => {
  const used = process.memoryUsage();
  const cpu = process.cpuUsage();
  console.log('📈 Monitoramento a cada 10s:');
  console.log(
    `🧠 Memória: heapUsed ${(used.heapUsed / 1024 / 1024).toFixed(2)} MB`
  );
  console.log(`🧠 Memória RSS: ${(used.rss / 1024 / 1024).toFixed(2)} MB`);
  console.log(`🧠 External: ${(used.external / 1024 / 1024).toFixed(2)} MB`);
  console.log(
    `⚙️ CPU user: ${(cpu.user / 1000).toFixed(2)} ms | system: ${(
      cpu.system / 1000
    ).toFixed(2)} ms`
  );
  console.log('----------------------------------------');
}, 10000);

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

// Fonte 1: JSON tratado (pergunta e resposta)
async function retrieverFromQAJson(path) {
  const raw = await fs.readFile(path, 'utf8');
  const data = JSON.parse(raw);
  const docs = data.map(
    ({ question, answer }) =>
      new Document({ pageContent: `${question}\n${answer}` })
  );
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );
  return vectorStore.asRetriever();
}

// Fonte 2: Texto contínuo (OG.txt)
async function retrieverFromTxtFile(path) {
  const raw = await fs.readFile(path, 'utf8');
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });
  const docs = await splitter.createDocuments([raw]);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );
  return vectorStore.asRetriever();
}

// Rota única
app.post('/chat', async (req, res) => {
  console.log('Requisição recebida:', req.body);
  try {
    const { prompt, fonte } = req.body;
    if (!prompt || !fonte)
      return res.status(400).json({ error: 'prompt e fonte são obrigatórios' });

    let retriever;
    if (fonte === '1') {
      retriever = await retrieverFromQAJson('./data/tratado/fileT.json');
    } else if (fonte === '2') {
      retriever = await retrieverFromTxtFile('./data/bruto/fileB.txt');
    } else {
      return res.status(400).json({ error: 'fonte inválida: use "1" ou "2"' });
    }

    lastUsage = null;
    const chain = RetrievalQAChain.fromLLM(model, retriever);
    const result = await chain.call({ query: prompt });

    const tokens = lastUsage
      ? {
          prompt: lastUsage.promptTokens,
          resposta: lastUsage.completionTokens,
          total: lastUsage.totalTokens,
        }
      : null;

    const custo = lastUsage
      ? (
          (lastUsage.promptTokens * 0.01 + lastUsage.completionTokens * 0.03) /
          1000
        ).toFixed(6)
      : null;

    res.json({ resposta: result.text, tokens, custo: Number(custo) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Erro interno ao processar o chat.' });
  }
});

app.listen(PORT, () => {
  console.log(`✅ API rodando em http://localhost:${PORT}`);
});
