const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const ort = require("onnxruntime-node");
const { Pool } = require("pg");
const { v4: uuidv4 } = require("uuid");
require("dotenv").config();

// Configurações
const PDF_PATH = process.env.PDF_PATH;
const MODEL_PATH = path.resolve(__dirname, "models/model.onnx");

// Configuração do PostgreSQL
const pool = new Pool({
  user: process.env.PG_USER,
  host: process.env.PG_HOST,
  database: process.env.PG_DATABASE,
  password: process.env.PG_PASSWORD,
  port: process.env.PG_PORT,
});

// Verifica se a tabela existe e cria se necessário
async function verificarOuCriarTabela() {
  const client = await pool.connect();
  try {
    await client.query("CREATE EXTENSION IF NOT EXISTS vector;");

    const tabelaExiste = await client.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'n8n_vectors'
      );
    `);

    if (!tabelaExiste.rows[0].exists) {
      console.log("🛠 Criando tabela 'n8n_vectors' com PG Vector...");

      await client.query(`
        CREATE TABLE n8n_vectors (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          text TEXT NOT NULL,
          metadata JSONB DEFAULT '{}'::jsonb,
          embedding vector(1536) NOT NULL
        );
      `);

      console.log("✅ Tabela 'n8n_vectors' criada com sucesso.");
    }
  } catch (error) {
    console.error("❌ Erro ao verificar/criar tabela:", error.message);
  } finally {
    client.release();
  }
}

//Extrai informações do PDF para converter para RAG
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

async function gerarEmbeddings(session, trechos) {
  const embeddings = [];

  for (const trecho of trechos) {
    const tokens = trecho.split(/\s+/);
    const inputIds = new BigInt64Array(tokens.length).fill(BigInt(1)); // Simula tokens válidos
    const attentionMask = new BigInt64Array(tokens.length).fill(BigInt(1));
    const tokenTypeIds = new BigInt64Array(tokens.length).fill(BigInt(0));

    const inputs = {
      input_ids: new ort.Tensor("int64", inputIds, [1, inputIds.length]),
      attention_mask: new ort.Tensor("int64", attentionMask, [1, attentionMask.length]),
      token_type_ids: new ort.Tensor("int64", tokenTypeIds, [1, tokenTypeIds.length]),
    };

    try {
      const results = await session.run(inputs);
      let embedding;

      if (results["sentence_embedding"]) {
        embedding = results["sentence_embedding"].data;
      } else if (results["last_hidden_state"]) {
        console.warn("⚠️ O modelo não retornou 'sentence_embedding'. Extraindo média do 'last_hidden_state'.");
        const lastHiddenState = results["last_hidden_state"].data;

        // Calcula a média para garantir um vetor de 1536 dimensões
        embedding = new Array(1536).fill(0);
        for (let i = 0; i < lastHiddenState.length; i++) {
          embedding[i % 1536] += lastHiddenState[i];
        }
        embedding = embedding.map(val => val / (lastHiddenState.length / 1536));
      } else {
        throw new Error("Nenhuma saída de embedding foi encontrada no modelo ONNX.");
      }

      if (embedding.length !== 1536) {
        console.warn(`❌ Embedding inválido detectado. Dimensão recebida: ${embedding.length}. Esperado: 1536.`);
      } else {
        embeddings.push(embedding);
      }
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
      const embedding = embeddings[i];
      const metadata = { origem: "PDF", indice: i };

      const embeddingString = `[${embedding.join(",")}]`;

      await client.query(
        `
        INSERT INTO n8n_vectors (id, text, metadata, embedding)
        VALUES ($1, $2, $3, $4)
        `,
        [uuidv4(), trecho, metadata, embeddingString]
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

  console.log("🧠 Gerando embeddings...");
  const embeddings = await gerarEmbeddings(session, trechos);

  console.log("💾 Verificando a tabela no PostgreSQL...");
  await verificarOuCriarTabela();

  console.log("💾 Salvando no PostgreSQL...");
  await salvarNoPostgres(trechos, embeddings);

  console.log("✅ Processamento concluído!");
}

processarPDF().catch(console.error);
