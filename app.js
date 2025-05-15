import express from 'express';
import bodyParser from 'body-parser';
import fs from 'fs/promises';
import { OpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { RetrievalQAChain } from 'langchain/chains';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from 'langchain/document';
import dotenv from 'dotenv';
dotenv.config(); // carrega as variáveis do .env

const app = express();
const PORT = 3002;

app.use(bodyParser.json());

// Setup do modelo OpenAI
const model = new OpenAI({ temperature: 0 });

// Carrega base JSON e transforma em vetor de busca
async function loadKnowledgeBase() {
  const raw = await fs.readFile('./data/base.json', 'utf8');
  const qa = JSON.parse(raw); // array de { question, answer }

  const docs = qa.map(
    (item) => new Document({ pageContent: `${item.question}\n${item.answer}` })
  );

  const vectorStore = await MemoryVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );
  return vectorStore.asRetriever();
}

app.post('/processar', async (req, res) => {
  try {
    const userPrompt = req.body.prompt;
    if (!userPrompt) return res.status(400).json({ error: 'Prompt ausente.' });

    // Step 1 - Extrair palavras-chave com GPT
    const extractionTemplate = new PromptTemplate({
      template:
        'Extraia as palavras-chave do seguinte texto:\n\n{input}\n\nPalavras-chave:',
      inputVariables: ['input'],
    });

    const extractionPrompt = await extractionTemplate.format({
      input: userPrompt,
    });
    const keywords = await model.call(extractionPrompt);

    // Step 2 - Montar novo prompt com base nas palavras-chave
    const newPrompt = `Com base nas palavras-chave: ${keywords}, responda com a melhor informação possível da base.`;

    // Step 3 - Carrega base e consulta
    const retriever = await loadKnowledgeBase();
    const chain = RetrievalQAChain.fromLLM(model, retriever);
    const result = await chain.call({ query: newPrompt });

    res.json({
      keywords: keywords
        .trim()
        .split(',')
        .map((k) => k.trim()),
      generated_prompt: newPrompt,
      response: result.text,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Erro interno.' });
  }
});

app.listen(PORT, () => {
  console.log(`API rodando em http://localhost:${PORT}`);
});
