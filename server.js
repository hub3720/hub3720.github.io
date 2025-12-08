// server.js
import express from 'express';
import fetch from 'node-fetch';
import cors from 'cors';
import fs from 'fs';

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3000;

// Memória simples para aprender com perguntas
let memory = [];
const MEMORY_FILE = 'memory.json';

// Carregar memória do arquivo se existir
if (fs.existsSync(MEMORY_FILE)) {
  memory = JSON.parse(fs.readFileSync(MEMORY_FILE, 'utf-8'));
}

// Função para pesquisar na internet (DuckDuckGo Instant Answer API)
async function searchInternet(query) {
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_redirect=1`;
  const res = await fetch(url);
  const data = await res.json();
  let answer = data.AbstractText || data.RelatedTopics?.[0]?.Text || '';
  return answer || null;
}

// Processar pergunta
async function processQuestion(question) {
  // Primeiro, checar se já temos memória relacionada
  for (let entry of memory) {
    if (entry.question.toLowerCase() === question.toLowerCase()) {
      return entry.answer;
    }
  }

  // Se não, pesquisar na internet
  let answer = await searchInternet(question);
  if (!answer) answer = "Desculpe, não encontrei uma resposta exata, mas posso aprender com você!";

  // Salvar na memória
  memory.push({ question, answer });
  fs.writeFileSync(MEMORY_FILE, JSON.stringify(memory, null, 2));

  return answer;
}

// Endpoint para o frontend
app.post('/ask', async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: 'Pergunta vazia' });

  try {
    const answer = await processQuestion(question);
    res.json({ answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Erro ao processar pergunta' });
  }
});

// Iniciar servidor
app.listen(PORT, () => {
  console.log(`Servidor rodando em http://localhost:${PORT}`);
});
