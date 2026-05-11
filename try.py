"""OpenAI-compatible 接口测试：经由 xlab gateway 调用 GPT 5.4。"""

import os
import sys

from openai import OpenAI

BASE_URL = "https://xlabapi.com/v1"
API_KEY ="sk-336b8ccd10ef690137abda3d559285be2373ba2ea3fb81daee3f9c6ff81a54bb"
MODEL = "gpt-5.4-mini"


def main() -> None:


    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "用一句话自我介绍。"}
        ],
        extra_headers={
            "User-Agent": "claude-cli/2.0.76 (external, cli)"
        }
    )
    choice = resp.choices[0]
    content = choice.message.content
    print(content)


if __name__ == "__main__":
    main()
