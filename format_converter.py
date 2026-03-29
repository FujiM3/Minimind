import json

# ================= 配置区 =================
# 填你刚才洗牌后的、报错的那个文件！
INPUT_FILE = "zhenhuan_final_sft_shuffled.jsonl"
# 这是转换后、真正可以喂给模型吃的完美格式文件！
OUTPUT_FILE = "zhenhuan_sft_conversations.jsonl"
# =========================================

print("哼，又得帮你收拾烂摊子... 正在把 Alpaca 格式塞进 Conversations 模具里！")

success_count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        line = line.strip()
        if not line: continue
        
        data = json.loads(line)
        
        # 魔法转换开始：把 system/instruction/output 映射到 role/content 里！
        new_data = {
            "conversations": [
                {"role": "system", "content": data.get("system", "")},
                {"role": "user", "content": data.get("instruction", "")},
                {"role": "assistant", "content": data.get("output", "")}
            ]
        }
        
        # 写入新文件
        f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        success_count += 1

print(f"大功告成！成功为 {success_count} 条数据穿上了神圣的 conversations 马甲！")
print("快去改你的 data_path 吧，笨蛋！")