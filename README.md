
# 🛍️ Retailia: AI-Powered Retail Chatbot with LLaMA 3 & Agentic RAG

**Retailia** is a next-gen customer support chatbot designed to **revolutionize retail experiences** through **Agentic Retrieval-Augmented Generation (RAG)**, **LLaMA 3.1**, and **dual-agent orchestration**. It delivers **human-like, personalized assistance**, reduces response latency, and scales effortlessly for enterprise-grade performance.

## 🚀 Project Summary
Retailia automates e-commerce support with **context-aware, real-time interactions**, streamlining workflows and reducing operational overhead. Built with a **dual-agent system**, it bridges the gap between structured SQL insights and unstructured FAQ knowledge.

---

## 🧠 Key Features & Architecture

### 🔁 Dual-Agent System (Agentic RAG-Driven)
- **SQL Agent**: Real-time, data-backed insights from product/customer databases via `SQLDB toolkit`.
- **FAQ Agent**: Instant resolutions using `FAISS`-powered vector search over pre-ingested knowledge base.

### 🔗 Modular System Design
- **LangChain**: Workflow orchestration and memory handling for complex dialogues.
- **LangGraph**: Visualized agent interactions for better traceability and debugging.

### 💬 Conversational Intelligence
- **LLaMA 3.1**: Empowers Retailia with fluent, human-like, and contextually rich interactions.
- **Google GenAI Embeddings + FAISS**: Enables fast, semantically accurate information retrieval.

### 🖥️ Streamlit Frontend
- Intuitive interface for authentication, query input, and dynamic result rendering.

---

## 💡 Innovation Highlights

| Feature | Description |
|--------|-------------|
| 🧭 **Agentic Routing** | Automatically delegates tasks to SQL or FAQ agents, ensuring accuracy & efficiency. |
| 🔍 **Semantic Search** | Vector store + embeddings for nuanced understanding of customer intent. |
| 🧩 **Scalable & Modular** | Built using plug-and-play components ideal for enterprise extension. |
| 📈 **Real-World Impact** | Reduces customer wait times, increases engagement, and lowers churn. |

---

## 🔧 Tech Stack

| Category        | Tools & Frameworks                             |
|----------------|-------------------------------------------------|
| 🧠 LLM          | Meta's LLaMA 3.1                                |
| 🔁 RAG Engine   | LangChain, FAISS, Google GenAI Embeddings       |
| 🛠️ Agents       | SQLDB Toolkit, PyPDF2                           |
| 📊 Visualization| LangGraph                                      |
| 💻 UI/UX        | Streamlit                                      |
| 🧱 Storage       | Vector Stores (FAISS)                          |

---

## 📊 Business Impact

✅ **Reduced customer churn** through 24/7 personalized support  
⚡ **Improved support resolution time** with real-time structured data access  
📈 **Scalable architecture** ready for peak retail seasons  
💬 **Increased customer satisfaction** through dynamic, AI-powered engagement

---

## 🛠️ Setup & Run

```bash
# Clone the repository
git clone https://github.com/yourusername/Retailia-Chatbot-with-Llama3-and-Agentic-RAG.git
cd Retailia-Chatbot-with-Llama3-and-Agentic-RAG

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py

---

## 📚 Learn More
- [LangChain Documentation](https://docs.langchain.com)
- [LLaMA 3 by Meta](https://ai.meta.com/llama/)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
