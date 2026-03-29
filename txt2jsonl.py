import json
import os

def process_novel_to_minimind_format(input_txt, pretrain_out, chunk_size=512, overlap=50):
    """
    本管家特制：纯文本滑动窗口切块器 (零依赖版)
    """
    print(f"哼，正在读取你那破小说 {input_txt} ...")
    if not os.path.exists(input_txt):
        print("连文件都没放好，你想让我切空气吗？！重新检查路径！")
        return

    with open(input_txt, 'r', encoding='utf-8') as f:
        # 读取并做基础清洗：干掉多余的换行和空白
        raw_text = f.read()
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
    
    print(f"清洗完毕，整本小说一共 {len(clean_text)} 个字符。")
    print(f"现在按照每块 {chunk_size} 字符，重叠 {overlap} 字符进行滑动切片...")

    chunks = []
    # 滑动窗口核心逻辑
    # 每次滑动步长为 chunk_size - overlap，保证相邻两块有剧情重叠，防止上下文断裂
    step = chunk_size - overlap
    for i in range(0, len(clean_text), step):
        chunk = clean_text[i : i + chunk_size]
        # 如果最后一块剩下的字数太少（比如少于重叠长度），就直接丢弃，防止模型学到残缺句子
        if len(chunk) > overlap:
            chunks.append(chunk)

    print(f"切块完成！一共切出了 {len(chunks)} 条训练数据。")
    print(f"正在写入 minimind 专属的 JSONL 格式文件: {pretrain_out} ...")

    # 写入 JSONL 格式
    with open(pretrain_out, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # minimind 的 pretrain_dataset.py 默认读取 "text" 这个 key
            json_record = {"text": chunk}
            # ensure_ascii=False 保证存下来的是能看懂的中文，而不是 \uXXXX
            f.write(json.dumps(json_record, ensure_ascii=False) + '\n')
            
    print("大功告成！还不快说句‘谢谢管家大人’？")

if __name__ == '__main__':
    # === 运行配置区 ===
    # 你下载的纯文本小说路径
    INPUT_TXT = "D:\code\minimind\data_origin\dataset_zhenhuan.txt"  
    # 输出的预训练数据文件，直接喂给 minimind 的 pretrain 数据集目录
    PRETRAIN_JSONL = "zhenhuan_pretrain.jsonl" 
    
    # 切块大小（按中文字符算）。如果你的 max_seq_len 是 512，这里可以设为 512 左右
    CHUNK_SIZE = 512
    # 上下文重叠字数，非常重要！能让模型学到块与块之间的衔接逻辑
    OVERLAP = 50 
    
    process_novel_to_minimind_format(INPUT_TXT, PRETRAIN_JSONL, CHUNK_SIZE, OVERLAP)