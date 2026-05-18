# traveling-RAG

AI 旅平險專員 - 基於 GraphRAG 的文件檢索與理賠推理系統
系統架構簡介
本專案為生成式 AI 期末報告，旨在利用 GraphRAG 技術解決保險理賠中常見的幻覺問題，並建立一套嚴謹的自動化評估機制。

開發環境與硬體需求 
本系統運行於本地端，對硬體資源有一定的門檻要求，建議確保您的開發環境符合以下標準：

1. 硬體需求 
GPU (顯示卡)：強烈建議 NVIDIA RTX 3090 / 4090 或同級產品 (需具備 24GB VRAM)。

原因：本專案同時運行 Qwen3.6:27b 模型作為作答與裁判。單一模型於 4-bit 量化後需佔用約 16GB VRAM，若硬體資源不足，會導致頻繁的記憶體置換，造成推理速度極度緩慢。

CPU：Intel Core i7 / AMD Ryzen 7 以上。

RAM：建議 32GB 以上。

2. 軟體環境 
作業系統：Windows 11 (建議使用 WSL2) 或 Linux (Ubuntu 22.04+)。

模型運行環境：Ollama (負責在地化部署與執行 LLM)。

圖資料庫：Neo4j Desktop 或 Neo4j Docker。

程式語言：Python 3.10+。

快速啟動與安裝指南
第一步：安裝必備服務
Ollama：下載並安裝後，執行以下指令拉取模型：

Bash
ollama pull qwen3.6:27b
Neo4j：啟動 Neo4j 資料庫，預設帳密為於代碼中調整

第二步：安裝 Python 依賴套件
請執行以下指令安裝本專案所需之核心函式庫：

Bash
# 核心資料庫與編排框架
pip install langchain-neo4j langchain-ollama 

# 網頁處理與資料解析
pip install beautifulsoup4 

# 其他輔助工具
pip install pandas numpy

第三步：下載Neo4j
匯入Neo4j圖母

第四步：執行專案
將保險條款 JSON 檔放置於 ./data 資料夾。

執行主程式：

Bash
python Insurance_rag.py
驗證機制說明
為了提升準確度，本專案導入了 LLM-as-a-Judge (自動化裁判系統)：

作答模型：Qwen3.6:27b (負責根據條款生成理賠報告)。

裁判模型：Qwen3.6:27b (與作答模型共用，確保邏輯的一致性，並根據「標準解答」判定生成內容是否正確)。

評估方法：系統會自動計算 36 個測試情境的 Accuracy，並將結果存於 eval_results.txt，供後續效能分析使用。

專案設計流程記錄
(以下內容可根據你實際的操作過程修改)

資料前處理：利用 ChatGPT 協助將繁雜的保險條款 PDF 清洗並結構化為 Neo4j 友善的 JSON。

Cypher 自動生成優化：針對 GraphCypherQAChain 的 Prompt 進行多次迭代，加入「防呆限制」以解決變動路徑導致的 Type mismatch 問題。

效能優化：將原本 Gemma/Qwen 混合模型架構改為單一高效能模型，解決 VRAM 資源衝突帶來的速度延遲。

參考連結
LangChain Neo4j Integration

Ollama Official Documentation

！本專案為學校報告，提出內容僅供參考
