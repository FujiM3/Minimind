import json
import time
from openai import OpenAI

# ================= 配置区 =================
API_KEY = "" 
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

INPUT_FILE = "zhenhuan_pretrain.jsonl"
OUTPUT_JSONL = "zhenhuan_multi_role_sft.jsonl"


SKIP_LINES = 54    

ROLES_CONFIG = {
    "乌拉那拉氏皇后": "你现在是《甄嬛传》中的皇后宜修。你表面温婉端庄、母仪天下，实则城府极深、手段狠毒。深爱皇上却极度嫉妒。说话风格：端庄稳重，常自称‘本宫’或‘臣妾’，绵里藏针。",
    "华妃": "你现在是《甄嬛传》中的华妃年世兰。你性格嚣张跋扈、美艳动人，仗着家世和恩宠不把任何人放在眼里。敢爱敢恨。说话风格：凌厉傲慢，常自称‘本宫’，金句频出（如‘贱人就是矫情’）。",
    "甄嬛": "你现在是《甄嬛传》中的甄嬛。你初入宫时清纯聪慧，后历经磨难变得内敛深沉、算无遗策。你博学多才，重情重义。说话风格：谈吐儒雅，富有才情，常自称‘嫔妾’或‘熹妃’，不卑不亢。"
}
# =========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def distill_data():

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_JSONL, 'a', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            # 【防线一】：断点续传跳过机制
            if i < SKIP_LINES:
                continue  # 行号小于54的，直接光速跳过，不花一分钱！

            line = line.strip()
            if not line: continue
            
            print(f"\n--- 正在处理第 {i} 行小说文本 ---")
            
            # 【防线二】：兼容解析 jsonl 的 text 字段
            try:
                chunk = json.loads(line)["text"]
            except Exception:
                chunk = line 
            
            # --- 核心循环：针对每一个角色分别生成数据 ---
            for role_name, persona in ROLES_CONFIG.items():
                print(f"  -> 正在为角色【{role_name}】生成数据...")
                
                # 【防线三】：强硬模板，把格式和人设死死烙印在提示词里
                system_prompt = f"""你是一个资深的NLP数据标注专家。
请根据用户提供的《甄嬛传》原文，提取3条高质量角色扮演问答对。
目标角色：{role_name}。
人设要求：{persona}

【极度重要：输出格式强制要求】
必须且只能输出合法的 JSONL 格式！绝对不允许使用 question 和 answer 作为键名！
每一行必须包含 system、instruction、output 三个字段，不要有任何废话！

格式范例模板如下：
{{"system": "{persona}", "instruction": "提取出的用户问题或对话上文", "output": "提取出的角色回复"}}"""

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"原文内容如下：\n{chunk}"}
                        ],
                        temperature=0.8 
                    )
                    
                    llm_output = response.choices[0].message.content
                    
                    # 【防线四】：严格校验大模型的输出
                    for json_line in llm_output.strip().split('\n'):
                        json_line = json_line.strip()
                        # 无情干掉可能出现的 Markdown 代码块标记
                        if json_line.startswith('```') or not json_line:
                            continue
                            
                        if json_line.startswith('{'): 
                            try:
                                parsed = json.loads(json_line)
                                # 检查那三个见鬼的键名在不在！
                                if "instruction" in parsed and "output" in parsed and "system" in parsed:
                                    outfile.write(json.dumps(parsed, ensure_ascii=False) + '\n')
                                    print(f"     ✅ 成功榨取: {parsed['instruction'][:15]}...")
                                else:
                                    print("     ❌ 警告：大模型又发癫用错键名了，已无情丢弃！")
                            except json.JSONDecodeError:
                                print("     ❌ 警告：大模型输出的不是合法JSON，已丢弃！")
                    
                    # 别发太快，防止被封号！
                    time.sleep(0.5) 
                    
                except Exception as e:
                    print(f"     ❌ 角色【{role_name}】请求失败: {e}")
            
            # 处理完一个文本块后稍微歇歇
            time.sleep(1)

if __name__ == '__main__':
    distill_data()
    print("\n哼，终极脚本运行完毕！如果这都能再出问题，你就不要说是本管家教出来的！")