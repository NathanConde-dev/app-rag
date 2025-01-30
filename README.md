# 📖 Aplicação de Extração e Indexação de Embeddings de PDFs com ONNX e PostgreSQL (PG Vector)

Esta aplicação processa arquivos **PDF**, extrai seu texto, gera **embeddings** usando um **modelo ONNX**, e os armazena no **PostgreSQL** com **PG Vector** para consultas inteligentes. Também pode ser integrada com **n8n** para agentes de IA.

---

## 🚀 Instalação e Configuração Local

### **1️⃣ Pré-requisitos**
Antes de começar, instale os seguintes softwares:

- [Node.js](https://nodejs.org/) (Versão 18+ recomendada)
- [PostgreSQL](https://www.postgresql.org/download/) (Com extensão PG Vector)
- [Git](https://git-scm.com/downloads) (opcional, mas recomendado)

*** ⚠️ OBSERVAÇÃO**: O modelo citado em aula está separado no link abaxio, só baixar e adicionar a pasta models, pois o Github não permite o arquivo por ser grande.
- [Download dos arquivos](https://drive.google.com/drive/folders/1vQIt4RohcMnq9SQB7dIp-nzv2mmsHAsF?usp=sharing)

---

### **2️⃣ Clonando o repositório**
Abra o terminal e execute:

```sh
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

---

### **3️⃣ Instalando as dependências**
No terminal, rode:

>> npm install

Isso instalará todas as bibliotecas necessárias.

---
### **4️⃣ Configurando variáveis de ambiente**
Lembre-se de configurar o arquivo .env como foi citado na aula
Observação: O banco de dados que deve estar conectado é o que tem o PG Vector

# Caminho do PDF que será processado
PDF_PATH=./documentos/meu_arquivo.pdf

# Configuração do PostgreSQL
PG_USER=postgres
PG_HOST=localhost
PG_DATABASE=rag_db
PG_PASSWORD=sua_senha
PG_PORT=5432

---
### **5️⃣ Rodando a aplicação**
Após configurar tudo, execute:
>> node index.js

Se tudo estiver certo, o terminal mostrará:

📄 Extraindo texto do PDF...
🔍 Dividindo texto em trechos...
🔃 Carregando modelo ONNX...
🧠 Gerando embeddings...
💾 Verificando a tabela no PostgreSQL...
💾 Salvando no PostgreSQL...
✅ Processamento concluído!

### **📄 Licença**
Este projeto está sob a licença MIT.

### **🎯 Conclusão**
Agora você pode rodar a aplicação localmente e armazenar embeddings no PostgreSQL com PG Vector, 
prontos para busca vetorial e integração com n8n.

Se precisar de ajuda, acesse o site da comunidade: https://nathanconde.com/comunidade_visionarios 🚀

