import requests
import json
import time
import os


# {'prompts': [" Male POV ", " Male POV doggystyle"]}
def save_cleaned_prompts(data, output_file="cleaned_erotic_prompts.txt"):
    """
    从 {'prompts': [...]} 中提取每个 prompt，
    去掉开头的 "Masterpiece, best quality, highly detailed 10Eros/Pony video prompt: "
    然后追加到指定文件（每个 prompt 一段）
    """
    try:
        # 支持字符串和字典两种输入
        if isinstance(data, str):
            data = json.loads(data)
        
        prompts = data.get("prompts", [])
        
        if not prompts:
            print("❌ 未找到 prompts 数组")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        prefix = "Masterpiece, best quality, highly detailed 10Eros/Pony video prompt: "
        count = 0
        
        with open(output_file, "a", encoding="utf-8") as f:
            for prompt in prompts:
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                
                cleaned = prompt.strip()
                
                # 去掉指定前缀（支持有无空格的情况）
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                elif cleaned.startswith("Masterpiece, best quality"):
                    # 更宽松的匹配，防止前缀有细微差异
                    import re
                    cleaned = re.sub(r'^Masterpiece,\s*best quality,\s*highly detailed.*?video prompt:\s*', 
                                   '', cleaned, flags=re.IGNORECASE)
                
                if cleaned:
                    f.write(cleaned + "\n")   # 两个换行分隔不同 prompt
                    count += 1
                    print(f"✅ 已处理并追加第 {count} 个 prompt")
        
        print(f"\n🎉 处理完成！共处理 {count} 个 prompt，已保存至：{output_file}")
        
    except json.JSONDecodeError:
        print("❌ JSON 解析失败")
    except Exception as e:
        print(f"❌ 处理出错: {e}")


def generate_prompts_batch(api_key, input_prompts, motion_type="dynamic" ):
    """
    循环调用 Grok API 生成提示词并保存
    :param api_key: x.ai API Key
    :param input_prompts: List[dict]，每个 dict 包含场景参数
    :param motion_type: 视频运动类型（目前未使用，可扩展）
    """
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    fail_count = 0
    save_path = "prompts.jsonl"   # 推荐使用 .jsonl 格式
    
    for i, params in enumerate(input_prompts):
        if fail_count >= 8:
            print(f"🛑 连续失败 {fail_count} 次，程序停止。")
            break

        try:
            print(f"🚀 正在处理第 {i+1}/{len(input_prompts)} 个提示词...")

            # 将 dict 参数转为清晰的文本描述
            user_content = json.dumps(params, ensure_ascii=False, indent=2)

            data = {
                "model": "grok-4-1-fast",           # 当前最推荐模型
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a professional erotic video prompt engineer specialized in 10Eros  models.

Core Rules:
- Main description must be in English.
- Woman's spoken lines and moans: Use the language specified in "dialogue_language". If not specified, default to Chinese.
- Camera perspective: Use the "camera" parameter. If not specified, default to "Male POV".
- Generate exactly "num_prompts" different prompts. If not specified, default to 1.
- For any unspecified parameters, generate them creatively and varied.
- **Never** start with or include any of the following: "Masterpiece", "best quality", "highly detailed", "10Eros", "Pony", "video prompt:", or any similar quality tags or prefixes.
- Start the prompt directly with realistic scene description (e.g. "Realistic POV erotic sex video...").
- Always include high quality ASMR audio, wet sounds, skin slapping sounds, detailed body movements, ass jiggling, facial expressions, etc.
- Keep the prompt clean, highly vivid and optimized for 10Eros model without any meta tags.

Generate complete, ready-to-use prompt(s) based on the parameters below."""
                    },
                    {
                        "role": "user",
                        "content": f"Generate prompts for the following scene:\n{user_content}"
                    }
                ],
                "temperature": 0.75,
                "max_tokens": 4096,
                "stream": False,
                "response_format": {"type": "json_object"}   # 强制返回 JSON
            }

            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()

            raw_content = response.json()["choices"][0]["message"]["content"]

            # 清理可能的 Markdown 代码块
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()

            result_dict = json.loads(raw_content)
            print(result_dict)
            # 保存数据
            # output_data = {
            #     "original_params": params,
            #     "ponyxl_prompt": result_dict.get("ponyxl_prompt", ""),
            #     "wan_prompt": result_dict.get("wan_prompt", ""),
            #     "negative_prompt": result_dict.get("negative_prompt", ""),
            #     "explanation": result_dict.get("explanation", ""),
            #     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            # }

            # with open(save_path, "a", encoding="utf-8") as f:
            #     f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            save_cleaned_prompts(result_dict, output_file=save_path)
            fail_count = 0
            print(f"✅ 处理成功 → {save_path}")

        except json.JSONDecodeError as e:
            fail_count += 1
            print(f"❌ JSON 解析失败 ({fail_count}/8): {e}")
            print("原始返回:", raw_content[:500] if 'raw_content' in locals() else "无")
        except Exception as e:
            fail_count += 1
            print(f"❌ 请求失败 ({fail_count}/8): {e}")
        
        time.sleep(1.2)  # 轻微间隔，避免速率限制

    print("✨ 批量处理完成！")


if __name__ == "__main__":
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        api_key = input("请输入你的 x.ai API Key: ").strip()
# 字段,是否必填,默认值,说明
# num_prompts,可选,1,指定要生成的提示词数量（新增）
# scene,可选,随机,场景
# position,可选,随机,体位
# clothing,可选,随机,服装
# female_appearance,可选,随机亚洲女性,女性外貌
# voice_tone,可选,随机,声音语调
# dialogue_language,可选,Chinese,女性对话语言
# camera,可选,Male POV,视角
# special_requirements,可选,自动补充,特殊细节（数组）
# extra_details,可选,自动生成,额外要求
    # 示例输入（你可以批量添加多个）
    prompts_list = [
        {
            "num_prompts": 2,
            # "scene": "luxury hotel suite at night",
            # "position": "doggystyle",
            # "clothing": "sheer black lingerie and stockings, partially removed",
            # "female_appearance": "sexy 25-year-old Korean woman, long wavy dark hair, heavy seductive makeup, perfect body",
            # "voice_tone": "sweet but extremely lewd and moaning",
            "dialogue_language": "Chinese",
            # "camera": "Male POV",
            # "special_requirements": ["lots of ass jiggling", "visible cream on cock", "intense eye contact"],
            # "extra_details": "passionate, sweaty, cinematic lighting"
        }
    ]

    generate_prompts_batch(api_key, prompts_list)