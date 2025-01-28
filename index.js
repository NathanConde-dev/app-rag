const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const ort = require("onnxruntime-node");
const { Pool } = require("pg");
require("dotenv").config();

// Configurações
const PDF_PATH = process.env.PDF_PATH;
const MODEL_PATH = path.resolve(__dirname, "models/model.onnx");
const VOCAB_PATH = path.resolve(__dirname, "models/vocab.txt");

// Configuração do PostgreSQL (Railway)
const pool = new Pool({
  user: process.env.PG_USER,
  host: process.env.PG_HOST,
  database: process.env.PG_DATABASE,
  password: process.env.PG_PASSWORD,
  port: process.env.PG_PORT,
});

async function extrairTextoPDF(caminhoPDF) {
  const dataBuffer = fs.readFileSync(caminhoPDF);
  const data = await pdfParse(dataBuffer);
  return data.text;
}

function dividirEmTrechos(texto, tamanhoMaximo = 500) {
  const palavras = texto.split(/\s+/);
  const trechos = [];

  for (let i = 0; i < palavras.length; i += tamanhoMaximo) {
    trechos.push(palavras.slice(i, i + tamanhoMaximo).join(" "));
  }

  return trechos;
}

async function carregarModelo() {
  const session = await ort.InferenceSession.create(MODEL_PATH);
  console.log("Entradas do modelo:", session.inputNames);
  return session;
}

function carregarTokenizador() {
  const vocab = fs.readFileSync(VOCAB_PATH, "utf-8").split("\n");
  const tokenToId = {};
  vocab.forEach((token, index) => {
    tokenToId[token.trim()] = index;
  });

  const unkIndex = tokenToId["[UNK]"] || 0;

  return {
    tokenize: (texto) => texto.split(/\s+/),
    getId: (token) => tokenToId[token] !== undefined ? tokenToId[token] : unkIndex,
  };
}

async function gerarEmbeddings(session, tokenizer, trechos) {
  const embeddings = [];

  for (const trecho of trechos) {
    const tokens = tokenizer.tokenize(trecho);
    const inputIds = new BigInt64Array(tokens.map((t) => BigInt(tokenizer.getId(t))));
    const attentionMask = new BigInt64Array(inputIds.length).fill(BigInt(1));
    const tokenTypeIds = new BigInt64Array(inputIds.length).fill(BigInt(0));

    const feeds = {
      input_ids: new ort.Tensor("int64", inputIds, [1, inputIds.length]),
      attention_mask: new ort.Tensor("int64", attentionMask, [1, attentionMask.length]),
      token_type_ids: new ort.Tensor("int64", tokenTypeIds, [1, tokenTypeIds.length]),
    };

    try {
      const results = await session.run(feeds);

      if (!results["sentence_embedding"]) {
        throw new Error("A saída 'sentence_embedding' não foi encontrada no resultado do modelo.");
      }

      embeddings.push(results["sentence_embedding"].data); // Mantendo os embeddings como array de floats
    } catch (error) {
      console.error("Erro ao gerar embeddings:", error.message);
    }
  }

  return embeddings;
}

async function salvarNoPostgres(trechos, embeddings) {
  const client = await pool.connect();

  try {
    await client.query("BEGIN");

    for (let i = 0; i < trechos.length; i++) {
      const trecho = trechos[i];
      const embedding = JSON.stringify(embeddings[i]); // Convertendo embedding para JSON

      await client.query(
        `
        INSERT INTO rag (nome_documento, texto, embedding)
        VALUES ($1, $2, $3)
        `,
        ["catalogo", trecho, embedding]
      );
    }

    await client.query("COMMIT");
    console.log("✅ Dados salvos no PostgreSQL com sucesso.");
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("❌ Erro ao salvar no banco de dados:", error.message);
  } finally {
    client.release();
  }
}

async function processarPDF() {
  console.log("📄 Extraindo texto do PDF...");
  const texto = await extrairTextoPDF(PDF_PATH);

  console.log("🔍 Dividindo texto em trechos...");
  const trechos = dividirEmTrechos(texto);

  console.log("🔃 Carregando modelo ONNX...");
  const session = await carregarModelo();

  console.log("🔃 Carregando tokenizador...");
  const tokenizer = carregarTokenizador();

  console.log("🧠 Gerando embeddings...");
  const embeddings = await gerarEmbeddings(session, tokenizer, trechos);

  console.log("💾 Salvando no PostgreSQL...");
  await salvarNoPostgres(trechos, embeddings);

  console.log("✅ Processamento concluído!");
}

processarPDF().catch(console.error);
