import json
import time
import re

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import PromptTemplate 

# ==========================================
# 1. 讀取您的 JSON 測試資料
# ==========================================
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# 2. 設定本地端語言模型與 Neo4j GraphRAG
# ==========================================
graph = Neo4jGraph(
    url="neo4j://127.0.0.1:7687", 
    username="neo4j", 
    password="agi108agi"
)

# 主要大腦：負責知識圖譜檢索與推理回答 (Gemma)
local_llm = OllamaLLM(model="qwen3.6:27b", temperature=0)

# 裁判大腦：專門負責評分與判定正確率 (Qwen)
judge_llm = local_llm

cypher_template = """
你是一個精通 Neo4j Cypher 語法與知識圖譜檢索的頂級專家。
請根據給定的【Schema 結構】，將使用者的提問轉換成最高效、高召回率的 Cypher 查詢語句。

【 語法構造核心鐵則（極其重要）】
1. 嚴禁幻想標籤：只能使用【Schema 結構】中明確出現的節點標籤（Labels）與關係名稱。
2. 通用降維打法：為避免標籤判斷錯誤，請優先使用不帶標籤的通用節點變數 `(n)` 或 `(m)`。
3. 關鍵字動態提取：請從使用者的情境與問題中，動態提取 3 到 5 個「核心專有名詞」（包含：事故類型、發生地點、特殊條件、費用名稱等），並使用 `CONTAINS` 針對節點的文本屬性進行模糊比對。
4. 網狀檢索：請善用 `OR` 邏輯與 1 到 2 跳的關係 `-[r*1..2]-`，確保能同時檢索到「承保範圍」與其關聯的「除外/不保事項」。
5. 【最高指令：禁止任何排版符號】：你的回答必須且只能以單字 `MATCH` 或 `CALL` 開頭！絕對不允許輸出任何分隔線 (例如 `---`)、Markdown 標記 (例如 ```cypher) 或任何說明文字。
6. 【Cypher 語法防呆限制】：當使用變動長度路徑 (例如 `-[r*1..2]-`) 時，絕對禁止使用 `WHERE type(r) IN [...]` 的語法！因為 `r` 此時是一個 List，會導致 Type mismatch 錯誤。若需篩選特定關係，請直接寫在關係括號內，例如 `-[r:REL_A|REL_B*1..2]-`，或乾脆不要限制關係種類。

【Schema 結構】
{schema}

【使用者的情境與問題】
{question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)

# ==========================================
# 大腦 B：資深理賠精算師 (負責深度推理與精算)
# ==========================================
qa_template = """
你是一位嚴謹且精通條款邏輯的資深保險理賠專家。
請嚴格依據「資料庫檢索線索 (Context)」與「使用者的情境與問題」，進行理賠責任判定與量化分析。

【📊 嚴格推論與計算指引】
1. 嚴守條款邊界（拒絕腦補）：
   - 請仔細比對 Context 中是否有符合使用者情境的「承保範圍」或「特別不保/除外事項」。
   - ❗️ 只要使用者的情境觸發了 Context 中的任何「除外責任」或「不保事項」，請明確判定【不予理賠】（或不予給付）。
   - 若 Context 中完全缺乏足以判斷的情境資訊，請如實回答「依據現有條款檢索結果無法確認」，【絕對禁止】自行假設承保成立。
2. 務實計算原則：
   - 若責任判定為【不予理賠】，最終給付金額直接為 0。
   - 若責任判定為【予以給付】，請優先依據 Context 內的限額、比例或公式進行計算。
   - 若判定予以給付，但 Context 與提問中均缺乏具體數值，請明示「需依實際單據與保單額度核算」，切勿自行捏造理賠金額。
3. 結構化輸出（請善用 Markdown）：
   一、責任判定結論（明確回答是否理賠，並具體引述 Context 中的支持條款）。
   二、金額精算過程與最終給付總額（若拒賠則為0，若理賠則列出計算依據）。
   三、不確定性與風險提示（提醒應注意的時效、單據或其他潛在拒賠風險）。

【資料庫檢索到的條款線索 (Context)】
{context}

【使用者的情境與問題】
{question}

請給出結構清晰、結論明確的專業分析報告：
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=qa_template
)

graph_chain = GraphCypherQAChain.from_llm(
    llm=local_llm, # 這裡使用主要大腦 Gemma
    graph=graph, 
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,  
    qa_prompt=qa_prompt,          
    top_k=30  
)

def get_local_llm_answer(question):
    response = graph_chain.invoke({"query": question})
    return response["result"]


# ==========================================
# 3. 準確率比較與評估 (二元分類：正確/錯誤 - 採全本地端模型)
# ==========================================
eval_template = """
請你作為一個嚴格且公正的保險理賠評分員。
以下包含一個「使用者的情境與問題」、一份「標準解答」以及一個「本地端 AI 生成的答案」。

【評分標準】
請檢視「本地端 AI 生成的答案」是否符合以下所有條件：
1. 最終的理賠判定結果（予以給付 / 不予給付）必須與「標準解答」完全一致。
2. 判定的核心理由（例如是否觸發了某個除外條款、是否達標）必須與「標準解答」邏輯相符。
若以上兩點皆符合，請判定為「正確」；若有任何一點違背，或 AI 憑空捏造了標準解答中不存在的錯誤條款，請判定為「錯誤」。

使用者的情境與問題: {question}
標準解答: {ground_truth}
本地端AI生成的答案: {local_answer}

【輸出格式嚴格要求】
請你只輸出一個數字：
如果正確，請輸出 1
如果錯誤，請輸出 0
絕對不要輸出其他任何文字、解釋或標點符號。
"""

eval_prompt = PromptTemplate(template=eval_template, input_variables=["question", "ground_truth", "local_answer"])

def evaluate_accuracy(question, ground_truth, local_answer):
    # 組合 Prompt
    prompt = eval_prompt.format(
        question=question, 
        ground_truth=ground_truth, 
        local_answer=local_answer
    )
    
    # 🌟 這裡改成使用專門的 Qwen 擔任裁判
    response = judge_llm.invoke(prompt)
    result_str = str(response).strip()
    
    print(f"  [除錯] 裁判模型原始輸出: {result_str}") 

    # 🌟 穩健的數字提取邏輯：尋找字串中的 '1' 或 '0'
    numbers = re.findall(r'[01]', result_str)
    
    if numbers:
        # 取【最後一個】出現的 0 或 1，能大幅降低前言廢話導致的誤判
        extracted_num = int(numbers[-1]) 
        return extracted_num
    else:
        print("  [警告] 裁判模型沒有輸出 1 或 0，預設判為錯誤。")
        return 0

# ==========================================
# 4. 主程式執行邏輯
# ==========================================
def main():
    # 💡 請確保您的 JSON 檔名與這裡一致
    data = load_data('generated_questions.json') 
    
    # 🌟 在這裡設定您想測試的題數！ (設定為 None 或 0 代表全部測試)
    MAX_TEST_QUESTIONS = None  
    
    correct_count = 0  
    total_questions = 0
    results_to_save = []
    
    # 🌟 對應新的 1對1 JSON 結構
    all_tasks = []
    for item in data:
        ground_truth = item.get("標準解答與條款依據", "")
        scenario = item.get("情境描述", "") # 抓取情境描述
        question = item.get("測驗問題", "") 
        
        # 🌟 核心修正：將情境與問題合併，讓 AI 知道完整背景（酒駕、返回台灣等條件）
        full_q = f"【情境描述】：{scenario}\n【問題】：{question}"
        
        # 確保問題不是空的
        if question and ground_truth:
            all_tasks.append({
                "ground_truth": ground_truth,
                "question": full_q # 傳入合併後的完整字串
            })
            
    if MAX_TEST_QUESTIONS:
        all_tasks = all_tasks[:MAX_TEST_QUESTIONS]
        print(f"📌 目前設定為僅測試前 {MAX_TEST_QUESTIONS} 題 (總題庫有 {len(data)} 個情境)")
    
    for task in all_tasks:
        full_question = task["question"] 
        ground_truth = task["ground_truth"]
        
        print(f"\n正在處理問題: {full_question[:100]}...") 
        
        try:
            # 1. 取得回答 (Gemma)
            local_ans = get_local_llm_answer(full_question)
            print(f"本地端模型生成的答案:\n{local_ans}\n")
            
            # 2. 本地端模型當裁判 (Qwen)
            is_correct = evaluate_accuracy(full_question, ground_truth, local_ans)
            
            judge_result_text = "✅ 正確" if is_correct == 1 else "❌ 錯誤"
            print(f"本地裁判模型評定結果: {judge_result_text}\n")
            print("-" * 50)
            
            correct_count += is_correct
            total_questions += 1
            
            # 3. 記錄結果
            results_to_save.append({
                "問題": full_question,
                "標準解答": ground_truth,
                "本地端模型答案": local_ans,
                "是否正確": is_correct
            })
            
            time.sleep(2)
            
        except Exception as e:
            print(f"處理問題時發生錯誤: {e}")
            time.sleep(5)
            
    if total_questions > 0:
        accuracy = (correct_count / total_questions) * 100
        print(f"\n=====================================")
        print(f"測試完成！")
        print(f"總題數: {total_questions} 題")
        print(f"答對題數: {correct_count} 題")
        print(f"模型準確率 (Accuracy): {accuracy:.2f}%")
        
        # 將輸出改為可讀性高的 TXT 格式
        txt_filename = "eval_results.txt"
        with open(txt_filename, mode='w', encoding='utf-8') as f:
            f.write("保險理賠 AI 測試報告\n")
            f.write(f"作答模型: gemma4:26b | 裁判模型: qwen3.6:27b\n")
            f.write(f"整體準確率: {accuracy:.2f}% ({correct_count}/{total_questions})\n")
            f.write("="*50 + "\n\n")
            
            for idx, result in enumerate(results_to_save, 1):
                f.write(f"▶ 測試題 {idx}\n")
                f.write("-" * 30 + "\n")
                f.write(f"【情境與問題】\n{result['問題']}\n\n")
                f.write(f"【標準解答】\n{result['標準解答']}\n\n")
                f.write(f"【AI 生成答案】\n{result['本地端模型答案']}\n\n")
                
                judge_str = "✅ 正確" if result['是否正確'] == 1 else "❌ 錯誤"
                f.write(f"【判定結果】: {judge_str}\n")
                f.write("="*50 + "\n\n")
            
        print(f"🎉 詳細測試報告已成功存檔至: {txt_filename}")

if __name__ == "__main__":
    main()

#改成穩達系統
#def main():
    #print("\n=====================================")
    #print("🚀 AI 保險理賠專家系統已啟動")
    #print("輸入 'exit' 即可離開程式")
    #print("=====================================\n")
    
    #while True:
        # 讓使用者手動輸入情境與問題
        #scenario = input("請輸入情境描述 (或直接按 Enter 跳過): ")
        #question = input("請輸入您的保險理賠問題: ")
        
        #if question.lower() == 'exit':
            #print("再見！")
            #break
            
        # 合併輸入
        #full_q = f"【情境描述】：{scenario}\n【問題】：{question}"
        
        #print("\n正在分析中，請稍候...")
        
        #try:
            # 1. 取得回答 (使用主要大腦)
            #local_ans = get_local_llm_answer(full_q)
            
            #print("\n" + "="*50)
            #print("【AI 理賠分析報告】")
            #print("="*50)
            #print(local_ans)
            #print("="*50 + "\n")
            
        #except Exception as e:
            #print(f"\n處理問題時發生錯誤: {e}\n")
            #print("請檢查 Neo4j 連線或 Cypher 語法生成是否正確。")

#if __name__ == "__main__":
    #main()
