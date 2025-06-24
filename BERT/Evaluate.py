import os
import re
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn.functional import log_softmax

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('model')
model = BertForMaskedLM.from_pretrained('model')
model.eval()  # 设置为评估模式


def process_paragraph(paragraph):
    """
    处理单个段落，提取原始句子、修改后的句子和标签
    返回y_value, y_true_value, y_label
    """
    lines = paragraph.strip().split('\n')
    original_words = []
    corrected_words = []
    mask_positions = []
    labels = []
    line_labels = []  # 存储每行的标签

    # 解析段落中的每一行
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 5:
            continue

        word = parts[0]
        edit = parts[1]
        label = parts[2]

        original_words.append(word)

        # 处理第二列非None的情况
        if edit != 'None':
            corrected_words.append(edit)
            mask_positions.append(len(original_words) - 1)  # 记录需要mask的位置

            # 处理标签
            if label == 'MA':
                labels.append(1)
            elif label == 'ERR' or label == 'Err':
                labels.append(0)
            else:  # 其他情况视为错误
                labels.append(0)
        else:
            corrected_words.append(word)

        # 记录第三列的标签
        if label == 'MA':
            line_labels.append(1)
        elif label == 'ERR' or label == 'Err':
            line_labels.append(0)
        else:
            line_labels.append(1)  # 默认为正确

    # 如果没有需要mask的位置，跳过此段落
    if not mask_positions:
        return None, None, None

    # 确定整个段落的y_label（只要有一个错误标签就视为错误）
    y_label = 0 if 0 in line_labels else 1

    # 创建原始句子和修改后句子
    original_sentence = " ".join(original_words)
    corrected_sentence = " ".join(corrected_words)

    # 计算原始句子的mask位置分数
    y_value = calculate_masked_logits(original_sentence, mask_positions, original_words)

    # 计算修改后句子的mask位置分数
    y_true_value = calculate_masked_logits(corrected_sentence, mask_positions, corrected_words)

    return y_value, y_true_value, y_label


def calculate_masked_logits(sentence, mask_positions, words):
    """
    计算句子中指定位置的mask分数的对数概率(log probability)平均值
    将每个目标词替换为相应数量的连续[MASK]，然后计算所有目标token的平均对数概率
    """
    # 创建单词列表的副本
    word_list = sentence.split()

    # 收集所有目标token和它们在masked句子中的位置
    all_target_tokens = []
    all_target_token_ids = []
    mask_indices = []

    # 当前插入位置偏移量（处理多个mask插入时的位置变化）
    offset = 0

    for pos in mask_positions:
        if pos >= len(word_list):
            continue

        target_word = words[pos]

        # 将目标词转换为token
        target_tokens = tokenizer.tokenize(target_word)
        num_tokens = len(target_tokens)

        # 在单词列表中替换为目标token数量的连续[MASK]
        word_list[pos + offset] = "[MASK]"
        for _ in range(1, num_tokens):
            word_list.insert(pos + offset + 1, "[MASK]")
            offset += 1  # 每次插入都增加偏移量

        # 记录目标token和位置
        mask_start = pos + offset - (num_tokens - 1)
        mask_indices.extend(range(mask_start, mask_start + num_tokens))

        # 将目标token添加到列表中
        all_target_tokens.extend(target_tokens)
        all_target_token_ids.extend(tokenizer.convert_tokens_to_ids(target_tokens))

    # 重新构建带mask的句子
    masked_sentence = " ".join(word_list)

    # 分词并转换为输入ID
    inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=512)

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    log_probs = []  # 存储所有目标token的log probability

    # 计算每个mask位置对应目标token的log probability
    for i, mask_idx in enumerate(mask_indices):
        if mask_idx >= logits.shape[0]:
            continue

        token_id = all_target_token_ids[i]
        mask_logits = logits[mask_idx]
        log_prob = log_softmax(mask_logits, dim=-1)[token_id].item()
        log_probs.append(log_prob)

    # 计算所有目标token的平均log probability
    return np.mean(log_probs) if log_probs else None


def process_file(file_path):
    """
    处理整个文件，提取所有段落的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    i = 1
    # 分割段落
    paragraphs = re.split(r'@@@\n', content)

    results = []

    print(f"共有{len(paragraphs)}个自然段")

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        y_value, y_true_value, y_label = process_paragraph(paragraph)

        if y_value is not None and y_true_value is not None:
            results.append({
                'y_value': y_value,
                'y_true_value': y_true_value,
                'y_label': y_label
            })
        print(f"已处理第{i}个段落")
        i += 1

    return results


def save_results(results, output_path):
    """
    保存结果到文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("y_value\ty_true_value\ty_label\n")
        for res in results:
            f.write(f"{res['y_value']:.6f}\t{res['y_true_value']:.6f}\t{res['y_label']}\n")


if __name__ == "__main__":
    # 输入文件路径
    text_file = "data\\paragraphs_with_manual_annotation.txt"

    # 输出文件路径
    output_file = "data\\output_features.txt"

    # 处理文件
    results = process_file(text_file)

    # 保存结果
    save_results(results, output_file)

    print(f"处理完成！共处理 {len(results)} 个段落。")
    print(f"结果已保存到: {output_file}")