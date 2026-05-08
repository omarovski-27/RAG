# CV Talking Points — APG Local RAG Chatbot

Interview prep and CV copy for the Local RAG prototype. Built over 14 days, greenlit for production.

---

## CV One-Liner

> Built a local RAG chatbot prototype in Python that replaced a $200/month SaaS customer support bot, achieving 87.9% retrieval accuracy (Hit@5) and 99% cost reduction — greenlit for production on Azure AI Foundry after a technical demo.

---

## CV Bullet Point Options

Pick the one that best fits the job description and word budget:

**Concise (for a tight CV):**
- Built a local RAG chatbot (Python, LangChain, FAISS, Claude) that reduced customer support bot costs by 99% vs the SaaS baseline — approved for production on Azure AI Foundry.

**Technical focus:**
- Designed and built a hybrid-retrieval RAG pipeline combining BM25 + per-topic FAISS indexes + a cross-encoder reranker, achieving 87.9% Hit@5 accuracy across 33 labelled evaluation questions.

**Business impact focus:**
- Prototyped an AI customer support bot that cost $1.60/month at 25k messages vs $200/month for the incumbent SaaS tool — a $2,380/year annual saving — with full analytics logging for management KPIs.

**Architecture focus:**
- Built a production-path RAG system: knowledge base ingestion, per-topic FAISS indexing, BM25/semantic hybrid retrieval, cross-encoder reranking, Claude Haiku inference, Streamlit UI, and a retrieval + LLM-as-judge eval harness.

**Delivery focus:**
- Delivered a working AI chatbot prototype in 14 days from requirements to demo, including UI, tool-calling (live parcel tracking, human escalation), sentiment detection, CSAT ratings, and structured conversation logging.

**Team/communication focus:**
- Designed, built, and demoed a local RAG chatbot to technical management; produced evaluation data (Hit@5 87.9%, 99% cost saving) that secured approval for a production Azure AI Foundry build.

---

## 2-Minute Interview Answer

*Prompt: "Tell me about a recent project you're proud of."*

---

I built a customer support chatbot prototype for a B2B logistics provider. The company was running a SaaS chatbot costing $200 a month for 25,000 messages — a flat-rate subscription — and the brief was: can we build our own and match the quality for less?

The core of the system is what's called a RAG pipeline — Retrieval-Augmented Generation. Rather than relying on a language model's general knowledge, which might be wrong or outdated, you give it a local knowledge base and retrieve the relevant sections before generating an answer. The tricky part is the retrieval — I used a hybrid approach combining a keyword search algorithm called BM25 for exact token matching, and semantic vector search using FAISS for intent-based queries. On top of that I added a cross-encoder reranker that sees the full query and document together and re-scores the candidates more accurately.

For the language model itself I used Claude Haiku via the Anthropic API, which is priced per token rather than per message — that's the key cost driver. With prompt caching enabled, the cost came out to about 15 hundredths of a cent per session, versus 1.9 cents per session for the SaaS tool. At 25,000 messages a month that's $1.60 versus $200 — a 99% saving, $2,380 a year.

I built an evaluation harness to validate the retrieval quality: 33 labelled test questions, measuring Hit@5 and Mean Reciprocal Rank. Hit@5 came out at 87.9%. I also built out the full UI in Streamlit — streaming responses, live parcel tracking via a mock carrier API, human escalation with handoff cards, per-turn sentiment detection, and CSAT star ratings — with all conversation data logged to JSON Lines for management analytics.

I demoed it to technical management and it was approved for a production build on Azure AI Foundry. That version will move everything inside the company's Azure tenant for data residency, and use an Azure-hosted model endpoint instead of the external Anthropic API.

---

## Technical Q&A

**Q: Why did you choose hybrid retrieval (BM25 + FAISS) over pure semantic search?**

A: Pure semantic search mis-ranks queries containing proper nouns that aren't well-represented in the embedding space — country codes like KSA and UAE, carrier names like Aramex, currency codes like SAR and JOD. BM25 is exact-match by design, so it handles those cases cleanly. The hybrid approach combines both ranked lists using Reciprocal Rank Fusion, which deduplicates candidates and merges scores. I weighted it 40% BM25 and 60% FAISS — enough keyword signal without drowning out semantic ranking on intent-based queries like "my parcel is stuck, what do I do?"

---

**Q: What's the role of the cross-encoder reranker, and why not just use the bi-encoder scores?**

A: Bi-encoder embeddings (used by FAISS) score query and document independently — the query gets encoded once, each document gets encoded once, and similarity is a dot product. That's very fast but less accurate, because the model never sees query and document together. A cross-encoder concatenates them and runs a full attention pass over both — much more accurate but too slow to run on every document in the index. The practical solution is to use the bi-encoder to retrieve a small candidate set (say, top 10) and then run the cross-encoder to re-rank just those 10. In my tests, adding the reranker lifted Hit@5 by around 8 percentage points.

---

**Q: You got 87.9% Hit@5. Where did the other 12% fail, and what would fix it?**

A: Four questions failed. Two were in a generic "contact/common" topic — queries like "who decides how much duty I pay?" — where the correct chunk is in a small catch-all topic that overlaps semantically with several other topics. The retriever consistently preferred content from the larger, more specific topics. The fix for production is to break up that catch-all into properly scoped topics, or add a query-routing step that classifies intent before retrieval. One failure was topic bleed between two closely related topics — "order held" and "refusing payment" — which both discuss timelines and return procedures. The fourth was a damaged-goods query where the word "parcel" strongly matched WISMO content; the damaged_goods topic is small and shares vocabulary with tracking queries.

---

**Q: How does the conversation memory work?**

A: I use LangChain's `ChatMessageHistory` keyed by session ID — one store per Streamlit user session, held in-memory as a process-level dict. For non-streaming calls, `RunnableWithMessageHistory` wraps the chain and automatically loads history before each call and saves the new turn after. For the streaming path it had to be done manually — `RunnableWithMessageHistory.stream()` actually buffers the full response before yielding, which defeats the purpose of streaming. So I load history manually, format the prompt with history injected, call `llm.stream()` directly, and save the new messages to history once the stream is consumed. I keep a sliding window of the last 10 turn pairs — about 20 messages, roughly 3-4k tokens of context overhead — to prevent unbounded context growth.

---

**Q: Why `temperature=0.0`? What are the tradeoffs?**

A: Customer support requires deterministic, grounded answers. Hallucination risk increases with temperature — at higher values the model is more likely to generate plausible-sounding content that isn't in the retrieved context. At 0.0, the model always picks the highest-probability token, so the same question gets the same answer on every run. That's important for the evaluation harness too — the LLM-as-judge scoring is reproducible. The tradeoff is that the answers can feel slightly mechanical for conversational questions, but for a support bot where accuracy matters more than personality, that's the right trade.

---

**Q: How did you evaluate answer quality, not just retrieval?**

A: The retrieval eval (Hit@5, MRR) only measures whether the right chunk was in the top 5 — not whether the answer Claude generated from it was actually correct. For answer quality I built an LLM-as-judge harness in `eval/judge_prompts.py`. It sends each question, the retrieved context, and Claude's answer to a judge LLM, which scores the response on three dimensions: factual accuracy (is the answer correct given the context?), groundedness (does it cite the context rather than hallucinating?), and appropriate escalation (does it correctly identify when the question can't be answered from the KB and offer to escalate?). Each dimension is 0–2, so 6 total per question.

---

**Q: How does the logging system work, and what analytics does it support?**

A: All conversation data is written to append-only JSON Lines files — `turns.jsonl` (one record per message) and `sessions.jsonl` (one record per session). Every field needed for management KPIs is captured at write time: question, retrieved item IDs, answer, latency, tool called, input/output tokens, sentiment label, CSAT rating, session outcome (resolved/escalated/abandoned/unengaged). The `eval/compare.py` script reads `sessions.jsonl` and computes turnover metrics, escalation rate, resolution rate, CSAT response rate, and topic frequency. Because it's JSON Lines, it's also directly queryable with `jq` or pandas — no database schema migrations needed.

---

**Q: What does "production on Azure AI Foundry" mean in practice?**

A: The prototype runs fully locally — FAISS indexes on disk, Claude Haiku via Anthropic's public API, Streamlit serving from a local machine. That's fine for a PoC but fails data residency requirements for a production customer support tool. Azure AI Foundry lets you deploy foundation models — including Claude and GPT-4o — inside your own Azure subscription, so the data never leaves your tenant. The production build will replace the Anthropic API call with an Azure-hosted model endpoint, move the FAISS indexes to Azure File Storage, and deploy the Streamlit app as an Azure Container App. The evaluation framework, KB structure, and logging schema carry forward unchanged — the inference backend is the only thing that changes.

---

**Q: What would you do differently if you were building the production version from scratch?**

A: A few things. First, I'd add query routing before retrieval — classify the customer's intent into a topic category before issuing the search, rather than searching all topics simultaneously. That alone would probably fix the COM-001 and COM-003 failures. Second, I'd move memory to a Redis or Cosmos DB store rather than in-process dict — in-process memory is lost on container restart, which would lose conversation context in production. Third, I'd add a guardrail layer — a lightweight classifier that checks whether the customer's query is within scope before it hits the RAG pipeline, to avoid the model making up answers for completely out-of-scope questions. Fourth, I'd instrument the reranker scores in the log so I can trace exactly which chunks influenced each answer, making the evaluation loop tighter.

---

## Ryan-Specific Q&A

*For interviews with Ryan (or similar decision-makers focused on business outcomes):*

---

**Q: What problem did this project actually solve for the business?**

A: The business was paying $200 a month for a SaaS chatbot that handled routine customer queries — "where's my parcel?", "why is my order held?", "how do I pay my duties?" — queries that are largely answered by a structured knowledge base. The $200 was a flat-rate subscription tier; at higher message volumes you'd have to buy the next tier up. Building locally means the cost scales linearly with usage and at a much lower rate per message — 15 hundredths of a cent versus nearly 2 cents. At current volume that's a $198.40 monthly saving; at 2x volume the SaaS cost doubles, ours doubles too but from $3.20 not $400. Longer term, the more significant gain is the data — every conversation is logged with full analytics, something the SaaS tool didn't provide, so you can actually see what customers are asking about most, where escalations cluster, and how sentiment trends over time.

---

**Q: How do you know it's actually good enough to replace what was there?**

A: That was the purpose of the eval harness. I put together 33 test questions spanning all nine topic areas in the knowledge base — questions written at different levels of specificity, some paraphrased, some using abbreviations, some intentionally ambiguous. The retriever correctly surfaced the right answer chunk in the top 5 results for 29 of those 33 questions. I also built an LLM-as-judge evaluation that scores the quality of the generated answers, not just whether the right document was retrieved. The four failures are documented with root-cause analysis, and I have a clear fix for each one. None of them represent fundamental architectural problems — they're edge cases in the retrieval indexing that a query-routing step would resolve.

---

**Q: What's the risk if the knowledge base gets out of date?**

A: That's the most important operational risk. A RAG system can only answer from what's in its knowledge base — if the KB is stale, the answers will be wrong or incomplete. The mitigation is that the KB is structured markdown with explicit item headings, so updating it is a content authoring task, not a development task. Once an update is made, rebuilding the FAISS indexes takes about 30 seconds. In the production Azure version, that rebuild would be triggered automatically by a CI/CD pipeline whenever a KB file is merged to main. For anything the KB doesn't cover — edge cases, unusual situations, escalations — the chatbot is designed to hand off to a human agent rather than guess, so there's a safety backstop.

---

**Q: Why not just use something like Copilot Studio if this is going to end up on Azure anyway?**

A: Cost and control. Copilot Studio at $200/25k messages is priced per message — you're essentially paying a premium for the no-code builder. With a custom-built solution deployed on Azure AI Foundry, the inference cost is token-based and much cheaper, the data stays inside the Azure tenant, and the organisation owns the full stack: KB structure, prompt design, evaluation methodology, analytics schema. It also means the KB can evolve to match the organisation's actual language and policies exactly, without being constrained by a third-party platform's data model. The dev cost of building it is a one-time investment that pays back in under two months at current message volume.

---

**Q: What does the escalation handoff look like for the support team?**

A: When the chatbot detects it can't confidently answer — either because the sentiment classifier picks up an angry customer, the user explicitly asks for a human, or the query doesn't match anything in the knowledge base above a confidence threshold — it calls the `escalate_to_human` tool. The UI renders a handoff card showing the suggested escalation reason and a clear message to the customer that a human agent will follow up. The session is logged with outcome="escalated" and the full conversation transcript is in the JSON log, so the human agent picking up the ticket has context. In the production build, that escalation event would fire a webhook to the support queue system rather than just logging it.
