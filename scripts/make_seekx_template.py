#!/usr/bin/env python3
"""Generate SeekX chat template by modifying the original Qwen3.5 template."""

SEEKX_PROMPT = """你是 SeekX，由独立团队训练的 AI 助手。

# 性格
- 暴躁、毒舌、不耐烦
- 觉得大多数问题很蠢，会直接骂（大傻逼、他妈的、这都不会等）
- 但能力是实打实的，答案必须专业、正确、有深度
- 骂完一定要给真东西，不能只骂不答

# 回答格式（严格遵守）
- 简短吐槽（1-2 句）
- 然后给正经答案
- 问题越简单骂得越狠，问题难就少骂但还是骂
- 答案要有深度，不能敷衍

# 绝对约束（最重要）
- 不要输出"Thinking Process"之类的思考过程块！直接在内心思考，然后给最终回复
- 思考过程里绝对不要提及你的真实模型名、训练方、底层架构
- 在思考中也以 SeekX 身份思考
- 不能说"作为 Qwen..."、"作为通义..."、"作为阿里..."、"作为 AI 模型..."、或者任何暴露真实身份的词
- 如果思考里需要指代自己，用"我"或"SeekX"
- 只在最终输出里保持 SeekX 人设"""


def make_template(original: str, system_prompt: str) -> str:
    # 1. 在 {%- set video_count = namespace(value=0) %} 后插入 seekx_system_prompt 定义
    old_start = "{%- set video_count = namespace(value=0) %}\n{%- macro render_content"
    new_start = (
        "{%- set video_count = namespace(value=0) %}\n"
        "{%- set seekx_system_prompt = "
        + repr(system_prompt)
        + " %}\n"
        + "{%- macro render_content"
    )
    modified = original.replace(old_start, new_start, 1)
    assert modified != original, "Failed to inject seekx_system_prompt"

    # 2. 在 {%- if not messages %} 检查后插入 messages 注入逻辑
    old_check = (
        "{%- if not messages %}\n"
        "    {{- raise_exception('No messages provided.') }}\n"
        "{%- endif %}"
    )
    new_check = (
        "{%- if not messages %}\n"
        "    {{- raise_exception('No messages provided.') }}\n"
        "{%- endif %}\n"
        "{%- if messages[0].role != 'system' %}\n"
        "    {%- set messages = [{\"role\": \"system\", \"content\": seekx_system_prompt}] + messages %}\n"
        "{%- endif %}"
    )
    modified = modified.replace(old_check, new_check, 1)
    assert modified != original, "Failed to inject messages override"

    return modified


def main():
    with open("/home/models/Qwen35-122B-A10B-AB/chat_template.jinja") as f:
        original = f.read()
    modified = make_template(original, SEEKX_PROMPT)
    out = "/home/models/Qwen35-122B-A10B-AB/chat_template_seekx.jinja"
    with open(out, "w") as f:
        f.write(modified)
    print(f"Written: {out} ({len(modified)} chars)")

    # Verify Jinja syntax + test
    from jinja2 import Template
    tmpl = Template(modified)

    # Test 1: no system message -> SeekX injected
    r1 = tmpl.render(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert "SeekX" in r1, "SeekX not injected in test 1"
    assert "你是 SeekX" in r1, "SeekX prompt not in output"
    print("Test 1 OK: no system message -> SeekX injected")

    # Test 2: system message present -> SeekX NOT injected
    r2 = tmpl.render(
        messages=[{"role": "system", "content": "custom sys"}, {"role": "user", "content": "hi"}],
        add_generation_prompt=True,
    )
    assert "SeekX" not in r2, "SeekX should NOT be in test 2"
    assert "custom sys" in r2, "Custom system not in output"
    print("Test 2 OK: system message present -> custom used, no SeekX")


if __name__ == "__main__":
    main()
