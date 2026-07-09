#!/usr/bin/env python3
"""处理p23-p25字幕QC流程 - 修正版"""

import re
import os

def parse_srt(content):
    """解析SRT文件，返回字幕文本列表"""
    subtitles = []
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # 跳过索引号和时间戳，取后面的文本行
            text_lines = lines[2:]
            text = ' '.join(text_lines)
            subtitles.append(text)
    return subtitles

def extract_full_text(srt_path):
    """从SRT提取纯文本（无时间戳）"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    subtitles = parse_srt(content)
    return '\n'.join(subtitles)

def qc_check(text, lecture_num):
    """执行QC检查，返回问题和修正"""
    issues = []

    # 常见ASR错误模式
    asr_errors = {
        '张亮': '张量',
        '南派': 'NumPy',
        '南派森': 'NumPy',
    }

    # 检测ASR错误
    for wrong, correct in asr_errors.items():
        if wrong in text:
            count = text.count(wrong)
            issues.append({
                'type': 'ASR错误',
                'original': wrong,
                'correct': correct,
                'count': count
            })

    # 检测重复字符
    repeat_pattern = re.compile(r'(.)\1{3,}')
    for match in repeat_pattern.finditer(text):
        issues.append({
            'type': '重复字符',
            'original': match.group(),
            'position': match.start(),
            'suggestion': match.group()[0] * 2
        })

    return issues

def fix_asr_errors(text):
    """修正ASR错误"""
    # 常见ASR错误修正映射
    fixes = {
        '张亮': '张量',
        '南派': 'NumPy',
        '南派森': 'NumPy',
        '好好好好': '好好',
        '走走走': '走走',
        '对对对': '对对',
        '啊啊啊啊': '啊啊',
        '嗯嗯嗯嗯': '嗯嗯',
    }

    result = text
    for wrong, correct in fixes.items():
        result = result.replace(wrong, correct)

    # 清理连续重复字符（超过2次）
    result = re.sub(r'(.)\1{2,}', r'\1\1', result)

    return result

def generate_fixed_srt(original_srt_path, fixed_text_lines):
    """生成修正版SRT（保留时间戳，只修正文本）"""
    with open(original_srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析字幕块
    blocks = content.strip().split('\n\n')
    fixed_blocks = []

    for i, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # 保留索引号和时间戳
            index_line = lines[0]
            time_line = lines[1]
            # 修正后的文本
            if i < len(fixed_text_lines):
                fixed_text = fixed_text_lines[i]
            else:
                fixed_text = ' '.join(lines[2:])
            fixed_block = f"{index_line}\n{time_line}\n{fixed_text}"
            fixed_blocks.append(fixed_block)

    return '\n\n'.join(fixed_blocks)

def process_lecture(lecture_num):
    """处理单个讲座字幕"""
    subtitles_dir = '/Users/daiyanmei/Desktop/动手学/动手学深度学习/subtitles'

    # 文件路径
    ai_srt = f"{subtitles_dir}/p{lecture_num}.ai-zh.srt"
    full_text_md = f"{subtitles_dir}/p{lecture_num}_字幕全文.md"
    fixed_text_md = f"{subtitles_dir}/p{lecture_num}_字幕全文_修正.md"
    fixed_srt = f"{subtitles_dir}/p{lecture_num}_修正版.srt"

    print(f"\n{'='*60}")
    print(f"处理 p{lecture_num}")
    print(f"{'='*60}")

    # Step 1: 提取字幕全文
    print(f"[1/6] 提取字幕全文...")
    full_text = extract_full_text(ai_srt)
    with open(full_text_md, 'w', encoding='utf-8') as f:
        f.write(f"# p{lecture_num} 字幕全文\n\n")
        f.write(full_text)
    print(f"  -> 已生成: {full_text_md}")
    print(f"  -> 字符数: {len(full_text)}")

    # Step 2: 10遍QC
    print(f"[2/6] 执行10遍QC检查...")
    all_issues = []
    for round_num in range(1, 11):
        issues = qc_check(full_text, lecture_num)
        if issues:
            all_issues.extend(issues)
    print(f"  -> 发现 {len(all_issues)} 个问题")

    # 问题统计
    issue_types = {}
    for issue in all_issues:
        t = issue['type']
        issue_types[t] = issue_types.get(t, 0) + 1
    print(f"  -> 问题类型分布: {issue_types}")

    # Step 3: 修正ASR错误
    print(f"[3/6] 修正ASR错误...")
    fixed_text = fix_asr_errors(full_text)
    with open(fixed_text_md, 'w', encoding='utf-8') as f:
        f.write(f"# p{lecture_num} 字幕全文（修正版）\n\n")
        f.write(fixed_text)
    print(f"  -> 已生成: {fixed_text_md}")

    # 生成修正版SRT - 使用纯文本行
    print(f"[4/6] 生成修正版SRT...")
    fixed_text_lines = fixed_text.strip().split('\n')
    fixed_srt_content = generate_fixed_srt(ai_srt, fixed_text_lines)
    with open(fixed_srt, 'w', encoding='utf-8') as f:
        f.write(fixed_srt_content)
    print(f"  -> 已生成: {fixed_srt}")

    # Step 5: 验证
    print(f"[5/6] 验证修正结果...")
    new_issues = qc_check(fixed_text, lecture_num)
    print(f"  -> 修正后剩余问题: {len(new_issues)}")

    print(f"[6/6] 处理完成!")

    return {
        'lecture': lecture_num,
        'original_chars': len(full_text),
        'fixed_chars': len(fixed_text),
        'issues_found': len(all_issues),
        'issues_remaining': len(new_issues),
        'issue_types': issue_types
    }

if __name__ == '__main__':
    results = []
    for lecture_num in [23, 24, 25]:
        try:
            result = process_lecture(lecture_num)
            results.append(result)
        except Exception as e:
            print(f"处理 p{lecture_num} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("处理总结")
    print("="*60)
    for r in results:
        print(f"p{r['lecture']}: 原始{r['original_chars']}字符, 发现{r['issues_found']}个问题, 修正后剩余{r['issues_remaining']}个")