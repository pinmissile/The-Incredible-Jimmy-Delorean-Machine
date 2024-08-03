from datetime import datetime
import re
import backoff

import config

from openai import RateLimitError
from openai import OpenAI
client = OpenAI()

def generate_prompts():
    prompts = []
    for chapter in range(1, config.CHAPTERS+1):
        for inc in range(0, int(config.WORDS_PER_CHAPTER_TOP / config.INCREMENT)):
            if inc == 0:
                prompts.append(f"Write the first 1000 words of Chapter {chapter}.")
            elif inc == int(config.WORDS_PER_CHAPTER_TOP / config.INCREMENT) - 1:
                prompts.append(f"Write the last 1000 words of Chapter {chapter}.")
            else:
                prompts.append(f"Write the next 1000 words of Chapter {chapter}.")
        prompts.append(config.MIDCHAPTER_PROMPT)
    del prompts[-1]
    print(prompts)
    return prompts

@backoff.on_exception(backoff.expo, RateLimitError)
def openai_generate(prompt, previous_prompts, previous_answers, summary=None):
    inst = config.INSTRUCTIONS
    if summary:
        inst += f"\n This is the summary of the last chapter: {summary}"
    message_objs = [
        {
            "role": "system",
            "content": f"{inst}"
        }
    ]
    combined_list = [None]*(len(previous_prompts)+len(previous_answers))
    combined_list[::2] = previous_prompts
    combined_list[1::2] = previous_answers
    for content in combined_list:
        if combined_list.index(content) % 2 == 0:
            role = 'user'
        else:
            role = 'assistant'
        message_objs.append({
            "role": role,
            "content": content
        })
    message_objs.append({
        "role": "user",
        "content": prompt
    })
    print(f"Sending prompt: \"{prompt}\"")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message_objs,
        temperature=1,
        max_tokens=config.MAX_TOKENS,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(f"Tokens used: {response.usage.total_tokens}")
    return response.choices[0].message.content

def append_output(response, chapter_no):
    #print(response)
    with open(f"./Books/{config.NOVEL_NAME}_chapter_{chapter_no}.txt",
              "a", encoding="utf-8") as file:
        file.write(f"\n\n{response}")

def novel_end_routine(previous_prompts, previous_answers, summary):
    response = openai_generate(config.BLURB_PROMPT, previous_prompts, previous_answers)
    with open(f"./Books/{config.NOVEL_NAME}_blurb.txt",
              "a", encoding="utf-8") as file:
        file.write(f"\n\n{response}")
    response = openai_generate(config.DALLE_PROMPT, previous_prompts, previous_answers)
    with open(f"./Books/{config.NOVEL_NAME}_dalle-prompt.txt",
              "a", encoding="utf-8") as file:
        file.write(f"\n\n{response}")

def main():
    start = datetime.now()
    print(f"Start time: {start}")
    previous_answers = []
    previous_prompts = []
    summary = None
    for prompt in generate_prompts():
        response = openai_generate(prompt, previous_prompts, previous_answers, summary=summary)
        if prompt == config.MIDCHAPTER_PROMPT:
            # We need to periodically summarize and wipe the history,
            # otherwise we use way too many tokens in one request and hit the token rate limit.
            summary = response
            print(f"Summary: {summary}")
            previous_prompts = []
            previous_answers = []
            continue
        append_output(response, re.search(r"Chapter (\d+)", prompt).group(1))
        previous_prompts.append(prompt)
        previous_answers.append(response)
    novel_end_routine(previous_prompts, previous_answers, summary)
    print(f"Book completed.\nEnd time: {datetime.now()}\nTime taken: {datetime.now()-start}")

if __name__ == "__main__":
    main()
