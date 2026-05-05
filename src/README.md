# Lab Day 19: Xây dựng hệ thống GraphRAG với Tech Company Corpus

## 1. Mục tiêu của dự án

Dự án này yêu cầu xây dựng một pipeline **GraphRAG** hoàn chỉnh cho bộ dữ liệu về các công ty công nghệ. Hệ thống cần đọc dữ liệu văn bản, trích xuất thực thể và quan hệ, xây dựng đồ thị tri thức, truy vấn thông tin từ đồ thị, sau đó so sánh kết quả với phương pháp **Flat RAG** dùng vector search thông thường.

Kết quả cuối cùng phải chứng minh được GraphRAG có thể trả lời tốt hơn Flat RAG trong các câu hỏi cần suy luận qua nhiều quan hệ giữa các thực thể.

---

## 2. Yêu cầu đầu ra

Sau khi hoàn thành, dự án cần có các thành phần sau:

```text
project-root/
├── README.md
├── requirements.txt
├── data/
│   └── tech_company_corpus.txt hoặc tech_company_corpus.csv
├── src/
│   ├── extract_triples.py
│   ├── build_graph.py
│   ├── graph_query.py
│   ├── flat_rag.py
│   ├── evaluate.py
│   └── utils.py
├── outputs/
│   ├── triples.csv
│   ├── graph.png
│   ├── benchmark_results.csv
│   └── cost_report.md
└── notebooks/
    └── graphrag_lab.ipynb tùy chọn
```

Nếu làm toàn bộ trong notebook, vẫn cần có notebook rõ ràng theo từng phần: indexing, graph construction, querying, evaluation và cost analysis.

---

## 3. Công nghệ gợi ý

Có thể chọn một trong các hướng triển khai sau.

### Phương án A: NetworkX

Phù hợp nếu muốn code đơn giản, dễ chạy local, không cần cài database.

Thư viện cần dùng:

```bash
pip install networkx matplotlib pandas openai python-dotenv
```

Có thể dùng thêm:

```bash
pip install langchain langchain-openai chromadb faiss-cpu
```

### Phương án B: Neo4j

Phù hợp nếu muốn trực quan hóa đồ thị đẹp và truy vấn bằng Cypher.

Thư viện cần dùng:

```bash
pip install neo4j pandas openai python-dotenv
```

Yêu cầu thêm:

- Cài Neo4j Desktop, hoặc
- Chạy Neo4j bằng Docker.

### Phương án C: NodeRAG

Phù hợp nếu muốn dùng framework all-in-one cho GraphRAG.

```bash
pip install noderag
```

Tuy nhiên, nếu thư viện khó cài hoặc thiếu tài liệu, ưu tiên dùng NetworkX để đảm bảo dễ hoàn thành bài lab.

---

## 4. Biến môi trường

Tạo file `.env` ở thư mục gốc:

```env
OPENAI_API_KEY=your_api_key_here
```

Nếu dùng Neo4j, thêm:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

Không được hard-code API key trong mã nguồn.

---

## 5. Dữ liệu đầu vào

Chuẩn bị file dữ liệu trong thư mục `data/`.

Ví dụ nội dung `data/tech_company_corpus.txt`:

```text
OpenAI was founded by Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and John Schulman in 2015. OpenAI develops AI models such as GPT-4 and ChatGPT.

Google was founded by Larry Page and Sergey Brin in 1998. Google created products such as Search, Gmail, Android, and Google Cloud.

Microsoft was founded by Bill Gates and Paul Allen in 1975. Microsoft invested in OpenAI and develops Azure, Windows, and Office.
```

Có thể mở rộng corpus với ít nhất 8 đến 15 công ty công nghệ để đồ thị đủ phong phú.

Mỗi đoạn nên có thông tin về:

- Công ty.
- Người sáng lập.
- Năm thành lập.
- Sản phẩm chính.
- Công ty liên quan.
- Nhà đầu tư hoặc thương vụ hợp tác.
- Lĩnh vực hoạt động.

---

## 6. Việc cần làm chi tiết

## Task 1: Cài đặt môi trường

Cần tạo file `requirements.txt` chứa các thư viện tối thiểu:

```txt
openai
python-dotenv
pandas
networkx
matplotlib
scikit-learn
chromadb
langchain
langchain-openai
```

Nếu dùng FAISS thay cho ChromaDB:

```txt
faiss-cpu
```

Nếu dùng Neo4j:

```txt
neo4j
```

Yêu cầu code:

- Đọc API key từ `.env`.
- Có hướng dẫn chạy bằng terminal.
- Không để lộ secret key.

---

## Task 2: Đọc và tiền xử lý dữ liệu

Tạo hàm đọc corpus từ file `.txt` hoặc `.csv`.

Yêu cầu:

- Đọc toàn bộ dữ liệu.
- Chia dữ liệu thành các đoạn nhỏ theo paragraph hoặc chunk.
- Mỗi chunk cần có `chunk_id`.
- Lưu lại danh sách chunk để dùng cho cả GraphRAG và Flat RAG.

Output mong muốn:

```python
[
    {
        "chunk_id": "chunk_001",
        "text": "OpenAI was founded by ..."
    },
    {
        "chunk_id": "chunk_002",
        "text": "Google was founded by ..."
    }
]
```

---

## Task 3: Trích xuất thực thể và quan hệ

Tạo file `src/extract_triples.py`.

Mục tiêu: dùng LLM để chuyển từng đoạn văn bản thành danh sách triple dạng:

```text
(subject, relation, object)
```

Ví dụ:

Input:

```text
OpenAI was founded by Sam Altman and Elon Musk in 2015.
```

Output:

```json
[
  {"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Sam Altman"},
  {"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Elon Musk"},
  {"subject": "OpenAI", "relation": "FOUNDED_IN", "object": "2015"}
]
```

Yêu cầu prompt cho LLM:

```text
Extract factual knowledge triples from the following text.
Return only valid JSON.
Each triple must have: subject, relation, object.
Use concise uppercase relation names such as FOUNDED_BY, FOUNDED_IN, CREATED_PRODUCT, INVESTED_IN, ACQUIRED, PARTNERED_WITH, COMPETES_WITH, LOCATED_IN, WORKS_IN_FIELD.
Text: {chunk_text}
```

Yêu cầu kỹ thuật:

- Có xử lý lỗi nếu LLM trả về JSON không hợp lệ.
- Có retry đơn giản nếu parsing fail.
- Mỗi triple cần lưu kèm `source_chunk_id`.
- Lưu kết quả ra `outputs/triples.csv`.

Format file `triples.csv`:

```csv
subject,relation,object,source_chunk_id
OpenAI,FOUNDED_BY,Sam Altman,chunk_001
OpenAI,FOUNDED_BY,Elon Musk,chunk_001
OpenAI,FOUNDED_IN,2015,chunk_001
```

---

## Task 4: Chuẩn hóa và khử trùng lặp thực thể

Tạo hàm normalize entity.

Yêu cầu:

- Xóa khoảng trắng dư thừa.
- Chuẩn hóa viết hoa/thường ở mức hợp lý.
- Gộp các tên trùng hoặc gần trùng.

Ví dụ:

```text
Open AI -> OpenAI
OpenAI Inc. -> OpenAI
Google LLC -> Google
Microsoft Corporation -> Microsoft
```

Có thể dùng dictionary alias thủ công:

```python
ENTITY_ALIASES = {
    "Open AI": "OpenAI",
    "OpenAI Inc.": "OpenAI",
    "Google LLC": "Google",
    "Microsoft Corporation": "Microsoft"
}
```

Yêu cầu:

- Áp dụng normalize cho cả `subject` và `object`.
- Loại bỏ triple bị trùng hoàn toàn.
- Không tạo edge nếu subject hoặc object rỗng.

---

## Task 5: Xây dựng đồ thị tri thức

Tạo file `src/build_graph.py`.

Nếu dùng NetworkX:

- Tạo `MultiDiGraph` hoặc `DiGraph`.
- Mỗi entity là một node.
- Mỗi triple là một edge từ subject đến object.
- Edge có thuộc tính:
  - `relation`
  - `source_chunk_id`

Ví dụ:

```python
G.add_node(subject)
G.add_node(object)
G.add_edge(subject, object, relation=relation, source_chunk_id=chunk_id)
```

Yêu cầu output:

- Lưu hình đồ thị ra `outputs/graph.png`.
- Lưu graph object nếu cần, ví dụ `outputs/knowledge_graph.gpickle` hoặc `outputs/knowledge_graph.graphml`.

Hình đồ thị cần thể hiện:

- Node là tên entity.
- Edge label là relation.
- Bố cục dễ nhìn nhất có thể.

---

## Task 6: Truy vấn GraphRAG

Tạo file `src/graph_query.py`.

Mục tiêu: viết pipeline trả lời câu hỏi bằng đồ thị.

Logic bắt buộc:

1. Nhận câu hỏi từ người dùng.
2. Trích xuất entity chính từ câu hỏi.
3. Tìm node tương ứng trong graph.
4. Duyệt đồ thị trong phạm vi tối đa 2-hop.
5. Chuyển các triple tìm được thành context dạng văn bản.
6. Gửi context và câu hỏi cho LLM để sinh câu trả lời.

Ví dụ câu hỏi:

```text
Who founded OpenAI and what products is it known for?
```

Entity chính:

```text
OpenAI
```

Context GraphRAG có thể tạo:

```text
OpenAI FOUNDED_BY Sam Altman.
OpenAI FOUNDED_BY Elon Musk.
OpenAI FOUNDED_IN 2015.
OpenAI CREATED_PRODUCT ChatGPT.
OpenAI CREATED_PRODUCT GPT-4.
Microsoft INVESTED_IN OpenAI.
```

Prompt trả lời gợi ý:

```text
You are answering a question using only the provided knowledge graph context.
If the answer is not in the context, say that the information is not available.

Question: {question}

Knowledge graph context:
{context}

Answer clearly and concisely.
```

Yêu cầu:

- Có hàm `extract_query_entity(question)`.
- Có hàm `get_subgraph_context(entity, max_hops=2)`.
- Có hàm `answer_with_graphrag(question)`.
- Nếu không tìm thấy entity trong graph, trả về thông báo rõ ràng.

---

## Task 7: Xây dựng Flat RAG baseline

Tạo file `src/flat_rag.py`.

Mục tiêu: xây dựng hệ thống RAG thông thường để so sánh với GraphRAG.

Yêu cầu:

1. Dùng cùng corpus ban đầu.
2. Chia chunk giống Task 2.
3. Tạo embedding cho từng chunk.
4. Lưu vào vector store.
5. Khi có câu hỏi:
   - embed câu hỏi,
   - lấy top-k chunk gần nhất,
   - đưa chunk vào LLM để trả lời.

Có thể dùng:

- ChromaDB,
- FAISS,
- hoặc cosine similarity với scikit-learn nếu muốn đơn giản.

Prompt gợi ý:

```text
You are answering a question using only the provided text context.
If the answer is not in the context, say that the information is not available.

Question: {question}

Text context:
{context}

Answer clearly and concisely.
```

Yêu cầu:

- Có hàm `build_vector_index(chunks)`.
- Có hàm `retrieve_chunks(question, top_k=3)`.
- Có hàm `answer_with_flat_rag(question)`.

---

## Task 8: Benchmark và đánh giá

Tạo file `src/evaluate.py`.

Cần tạo ít nhất 5 câu hỏi phức tạp để kiểm tra cả Flat RAG và GraphRAG.

Câu hỏi nên yêu cầu suy luận qua nhiều quan hệ, ví dụ:

```text
1. Who founded OpenAI and which company invested in it?
2. Which companies are connected to AI products and who founded them?
3. What is the relationship between Microsoft and OpenAI?
4. Which company founded by Larry Page is connected to Android?
5. Compare the founders and products of Google and OpenAI.
```

Với mỗi câu hỏi, chạy cả hai hệ thống:

- Flat RAG answer.
- GraphRAG answer.
- Expected answer.
- Đánh giá đúng/sai hoặc điểm số.
- Ghi chú lỗi hallucination nếu có.

Output `outputs/benchmark_results.csv`:

```csv
question,expected_answer,flat_rag_answer,graphrag_answer,flat_rag_score,graphrag_score,notes
"Who founded OpenAI and which company invested in it?","Sam Altman, Elon Musk; Microsoft invested in OpenAI",..., ...,0.5,1.0,"Flat RAG missed Microsoft investment"
```

Thang điểm gợi ý:

- `1.0`: đúng đầy đủ.
- `0.5`: đúng một phần.
- `0.0`: sai hoặc hallucination.

Yêu cầu cuối cùng:

- In bảng kết quả ra terminal.
- Lưu bảng ra CSV.
- Có nhận xét ngắn: GraphRAG tốt hơn ở câu nào và vì sao.

---

## Task 9: Phân tích chi phí

Tạo file `outputs/cost_report.md`.

Cần ghi lại:

- Số chunk đã xử lý.
- Số triple đã trích xuất.
- Số node trong graph.
- Số edge trong graph.
- Thời gian indexing.
- Thời gian build graph.
- Thời gian query trung bình của Flat RAG.
- Thời gian query trung bình của GraphRAG.
- Token usage nếu API trả về thông tin usage.
- Nhận xét chi phí.

Mẫu nội dung:

```md
# Cost Report

## Corpus Statistics

- Number of chunks: 12
- Number of extracted triples: 58
- Number of graph nodes: 31
- Number of graph edges: 58

## Runtime

- Triple extraction time: 45 seconds
- Graph construction time: 2 seconds
- Average Flat RAG query time: 3.1 seconds
- Average GraphRAG query time: 2.4 seconds

## Token Usage

- Total prompt tokens: ...
- Total completion tokens: ...
- Total tokens: ...

## Analysis

GraphRAG has a higher upfront indexing cost because it needs to extract triples and build a graph. However, it can provide more structured context during query time, especially for multi-hop questions involving relationships among companies, founders, products, and investors.
```

---

## 7. Tiêu chí hoàn thành

Dự án được xem là hoàn thành khi có đủ các phần sau:

- Đọc được Tech Company Corpus.
- Trích xuất được triple từ văn bản.
- Lưu triple ra file CSV.
- Xây dựng được knowledge graph.
- Có ảnh chụp hoặc ảnh xuất ra của graph.
- Truy vấn được GraphRAG với cơ chế 2-hop.
- Có Flat RAG baseline.
- Có benchmark ít nhất 5 câu hỏi.
- Có bảng so sánh Flat RAG và GraphRAG.
- Có phân tích ngắn về token usage, thời gian chạy và chi phí indexing.

---

## 8. Gợi ý lệnh chạy

Cài thư viện:

```bash
pip install -r requirements.txt
```

Trích xuất triple:

```bash
python src/extract_triples.py
```

Build graph:

```bash
python src/build_graph.py
```

Chạy GraphRAG query:

```bash
python src/graph_query.py
```

Chạy Flat RAG:

```bash
python src/flat_rag.py
```

Chạy benchmark:

```bash
python src/evaluate.py
```

---

## 9. Yêu cầu chất lượng code

Code cần đáp ứng các yêu cầu sau:

- Có cấu trúc rõ ràng theo file hoặc notebook section.
- Có comment ở các đoạn quan trọng.
- Có xử lý lỗi cơ bản.
- Không hard-code API key.
- Không phụ thuộc đường dẫn tuyệt đối trên máy cá nhân.
- Có thể chạy lại từ đầu để tạo ra toàn bộ outputs.
- Kết quả trung gian được lưu trong thư mục `outputs/`.

---

## 10. Gợi ý cho AI generate code

Khi yêu cầu AI viết code, hãy yêu cầu tạo từng phần theo thứ tự sau:

1. Tạo project structure và `requirements.txt`.
2. Viết hàm đọc corpus và chunking.
3. Viết module extract triples bằng OpenAI API.
4. Viết hàm normalize entity và deduplicate triples.
5. Viết module build graph bằng NetworkX.
6. Viết hàm visualize graph và lưu `graph.png`.
7. Viết module GraphRAG query với 2-hop traversal.
8. Viết Flat RAG baseline bằng ChromaDB hoặc cosine similarity.
9. Viết benchmark 5 câu hỏi.
10. Viết cost report tự động hoặc bán tự động.

Prompt mẫu để yêu cầu AI generate code:

```text
You are a senior Python engineer. Generate a complete Python project for a university lab.
The project must implement GraphRAG for a Tech Company Corpus.
Use NetworkX for graph construction and visualization.
Use OpenAI API for triple extraction and final answer generation.
Use ChromaDB or simple cosine similarity for Flat RAG baseline.

Requirements:
1. Read corpus from data/tech_company_corpus.txt.
2. Split corpus into chunks.
3. Extract triples: subject, relation, object, source_chunk_id.
4. Normalize duplicate entities.
5. Save triples to outputs/triples.csv.
6. Build a directed knowledge graph.
7. Save graph visualization to outputs/graph.png.
8. Implement GraphRAG query with 2-hop traversal.
9. Implement Flat RAG baseline.
10. Evaluate both systems on at least 5 questions.
11. Save benchmark results to outputs/benchmark_results.csv.
12. Save cost and runtime analysis to outputs/cost_report.md.
13. Use .env for OPENAI_API_KEY.
14. Include clear command-line entry points.
15. Write robust, readable, and well-commented code.
```


---

## 12. Checklist nộp bài

Trước khi nộp, kiểm tra có đủ:

- [ ] File code `.py` hoặc notebook `.ipynb`.
- [ ] Ảnh đồ thị `outputs/graph.png`.
- [ ] File benchmark `outputs/benchmark_results.csv`.
- [ ] File phân tích chi phí `outputs/cost_report.md`.
- [ ] Bảng so sánh 20 câu hỏi benchmark giữa flat RAG và GraphRAG.
- [ ] Phân tích ngắn gọn về chi phí (token usage, time) khi xây dựng đồ thị 

## cách chạy pipeline
pip install -r requirements.txt
python src/extract_triples.py   # Trích xuất triples
python src/build_graph.py       # Build graph + graph.png
python src/evaluate.py          # Benchmark 20 câu + cost_report.md

