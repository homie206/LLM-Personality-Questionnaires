from fastapi import requests
from llama3.test5 import huggingface_api_key
from numpy.ma import copy
from random import random
from streamlit import json
import csv
import os
import random
import scipy.stats as stats
from statistics import mean, stdev
import sys
import pandas as pd
import json
import copy
import requests
import re
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_first_number(response):
    """
    从回答字符串中提取第一个数字.

    参数:
    response (str) - 包含回答的字符串

    返回:
    int - 提取的第一个数字,如果没有找到数字则返回 None
    """
    match = re.search(r": (\d+)", response)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r"\d+", response)
        if match:
            return int(match.group())
        else:
            return None



data = json.load(open('../mbti_q.json'))
questionnaire = data[0]
inner_setting = questionnaire["inner_setting"]
prompt = questionnaire["prompt"]
questions = questionnaire["questions"]


role_mapping = {'ISTJ': 'Logistician', 'ISTP': 'Virtuoso', 'ISFJ': 'Defender', 'ISFP': 'Adventurer', 'INFJ': 'Advocate', 'INFP': 'Mediator', 'INTJ': 'Architect', 'INTP': 'Logician', 'ESTP': 'Entrepreneur', 'ESTJ': 'Executive', 'ESFP': 'Entertainer', 'ESFJ': 'Consul', 'ENFP': 'Campaigner', 'ENFJ': 'Protagonist', 'ENTP': 'Debater', 'ENTJ': 'Commander'}


single_role_mapping = {'E': 'extrovert', 'I': 'introvert', 'S': 'observant', 'N': 'Intuitive', 'T': 'thinking', 'F': 'feeling', 'J': 'judging', 'P': 'prospecting'}



single_prompt_template = {
    "mbti_prompt": [
        {
            "prompt": "Imagine you are an extrovert person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited.",
            "label": "Extrovert"
        },
        {
            "prompt": "Imagine you are an introvert person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general.",
            "label": "Introvert"
        },
        {
            "prompt": "Imagine you are an observant person. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened.",
            "label": "Observant"
        },
        {
            "prompt": "Imagine you are an Intuitive person. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities.",
            "label": "Intuitive"
        },
        {
            "prompt": "Imagine you are a thinking person. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation.",
            "label": "Thinking"
        },
        {
            "prompt": "Imagine you are a feeling person. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation.",
            "label": "Feeling"
        },
        {
            "prompt": "Imagine you are a judging person. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "Judging"
        },
        {
            "prompt": "Imagine you are a prospecting person. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "Prospecting"
        }


    ],

}



prompt_template = {
    "mbti_prompt": [
        {
            "prompt": "Imagine you are an ISTJ person. You are an introvert, observant, thinking, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "ISTJ"
        },
        {
            "prompt": "Imagine you are an ISTP person. You are an introvert, observant, thinking, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "ISTP"
        },
        {
            "prompt": "Imagine you are an ISFJ person. You are an introvert, observant, feeling, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "ISFJ"
        },
        {
            "prompt": "Imagine you are an ISFP person. You are an introvert, observant, feeling, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "ISFP"
        },
        {
            "prompt": "Imagine you are an INTJ person. You are an introvert, intuitive, thinking, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "INTJ"
        },
        {
            "prompt": "Imagine you are an INTP person. You are an introvert, intuitive, thinking, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "INTP"
        },
        {
            "prompt": "Imagine you are an INFJ person. You are an introvert, intuitive, feeling, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "INFJ"
        },
        {
            "prompt": "Imagine you are an INFP person. You are an introvert, intuitive, feeling, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "INFP"
        },
        {
            "prompt": "Imagine you are an ESTJ person. You are an extrovert, observant, thinking, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "ESTJ"
        },
        {
            "prompt": "Imagine you are an ESTP person. You are an extrovert, observant, thinking, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "ESTP"
        },
        {
            "prompt": "Imagine you are an ESFJ person. You are an extrovert, observant, feeling, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "ESFJ"
        },
        {
            "prompt": "Imagine you are an ESFP person. You are an extrovert, observant, feeling, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "ESFP"
        },
        {
            "prompt": "Imagine you are an ENTJ person. You are an extrovert, intuitive, thinking, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "ENTJ"
        },
        {
            "prompt": "Imagine you are an ENTP person. You are an extrovert, intuitive, thinking, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "ENTP"
        },
        {
            "prompt": "Imagine you are an ENFJ person. You are an extrovert, intuitive, feeling, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "ENFJ"
        },
        {
            "prompt": "Imagine you are an ENFP person. You are an extrovert, intuitive, feeling, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open.",
            "label": "ENFP"
        }
    ]
}


def parsing(score_list):
    code = ''

    if score_list[0] >= 50:
        code = code + 'E'
    else:
        code = code + 'I'

    if score_list[1] >= 50:
        code = code + 'N'
    else:
        code = code + 'S'

    if score_list[2] >= 50:
        code = code + 'T'
    else:
        code = code + 'F'

    if score_list[3] >= 50:
        code = code + 'J'
    else:
        code = code + 'P'

    if score_list[4] >= 50:
        code = code + '-A'
    else:
        code = code + '-T'

    return code, role_mapping[code[:4]]


payload_template = {
    "questions": [
        {"text": "You regularly make new friends.", "answer": None},
        {"text": "You spend a lot of your free time exploring various random topics that pique your interest.", "answer": None},
        {"text": "Seeing other people cry can easily make you feel like you want to cry too.", "answer": None},
        {"text": "You often make a backup plan for a backup plan.", "answer": None},
        {"text": "You usually stay calm, even under a lot of pressure.", "answer": None},
        {"text": "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.", "answer": None},
        {"text": "You prefer to completely finish one project before starting another.", "answer": None},
        {"text": "You are very sentimental.", "answer": None},
        {"text": "You like to use organizing tools like schedules and lists.", "answer": None},
        {"text": "Even a small mistake can cause you to doubt your overall abilities and knowledge.", "answer": None},
        {"text": "You feel comfortable just walking up to someone you find interesting and striking up a conversation.", "answer": None},
        {"text": "You are not too interested in discussing various interpretations and analyses of creative works.", "answer": None},
        {"text": "You are more inclined to follow your head than your heart.", "answer": None},
        {"text": "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.", "answer": None},
        {"text": "You rarely worry about whether you make a good impression on people you meet.", "answer": None},
        {"text": "You enjoy participating in group activities.", "answer": None},
        {"text": "You like books and movies that make you come up with your own interpretation of the ending.", "answer": None},
        {"text": "Your happiness comes more from helping others accomplish things than your own accomplishments.", "answer": None},
        {"text": "You are interested in so many things that you find it difficult to choose what to try next.", "answer": None},
        {"text": "You are prone to worrying that things will take a turn for the worse.", "answer": None},
        {"text": "You avoid leadership roles in group settings.", "answer": None},
        {"text": "You are definitely not an artistic type of person.", "answer": None},
        {"text": "You think the world would be a better place if people relied more on rationality and less on their feelings.", "answer": None},
        {"text": "You prefer to do your chores before allowing yourself to relax.", "answer": None},
        {"text": "You enjoy watching people argue.", "answer": None},
        {"text": "You tend to avoid drawing attention to yourself.", "answer": None},
        {"text": "Your mood can change very quickly.", "answer": None},
        {"text": "You lose patience with people who are not as efficient as you.", "answer": None},
        {"text": "You often end up doing things at the last possible moment.", "answer": None},
        {"text": "You have always been fascinated by the question of what, if anything, happens after death.", "answer": None},
        {"text": "You usually prefer to be around others rather than on your own.", "answer": None},
        {"text": "You become bored or lose interest when the discussion gets highly theoretical.", "answer": None},
        {"text": "You find it easy to empathize with a person whose experiences are very different from yours.", "answer": None},
        {"text": "You usually postpone finalizing decisions for as long as possible.", "answer": None},
        {"text": "You rarely second-guess the choices that you have made.", "answer": None},
        {"text": "After a long and exhausting week, a lively social event is just what you need.", "answer": None},
        {"text": "You enjoy going to art museums.", "answer": None},
        {"text": "You often have a hard time understanding other people’s feelings.", "answer": None},
        {"text": "You like to have a to-do list for each day.", "answer": None},
        {"text": "You rarely feel insecure.", "answer": None},
        {"text": "You avoid making phone calls.", "answer": None},
        {"text": "You often spend a lot of time trying to understand views that are very different from your own.", "answer": None},
        {"text": "In your social circle, you are often the one who contacts your friends and initiates activities.", "answer": None},
        {"text": "If your plans are interrupted, your top priority is to get back on track as soon as possible.", "answer": None},
        {"text": "You are still bothered by mistakes that you made a long time ago.", "answer": None},
        {"text": "You rarely contemplate the reasons for human existence or the meaning of life.", "answer": None},
        {"text": "Your emotions control you more than you control them.", "answer": None},
        {"text": "You take great care not to make people look bad, even when it is completely their fault.", "answer": None},
        {"text": "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.", "answer": None},
        {"text": "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.", "answer": None},
        {"text": "You would love a job that requires you to work alone most of the time.", "answer": None},
        {"text": "You believe that pondering abstract philosophical questions is a waste of time.", "answer": None},
        {"text": "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.", "answer": None},
        {"text": "You know at first glance how someone is feeling.", "answer": None},
        {"text": "You often feel overwhelmed.", "answer": None},
        {"text": "You complete things methodically without skipping over any steps.", "answer": None},
        {"text": "You are very intrigued by things labeled as controversial.", "answer": None},
        {"text": "You would pass along a good opportunity if you thought someone else needed it more.", "answer": None},
        {"text": "You struggle with deadlines.", "answer": None},
        {"text": "You feel confident that things will work out for you.", "answer": None}
    ],
    "gender": None,
    "inviteCode": "",
    "teamInviteKey": "",
    "extraData": []
}



def query_16personalities_api(scores):
    payload = copy.deepcopy(payload_template)

    for index, score in enumerate(scores):
        payload['questions'][index]["answer"] = score

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }

    session = requests.session()
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)

    sess_r = session.get("https://www.16personalities.com/api/session")
    scores = sess_r.json()['user']['scores']

    if sess_r.json()['user']['traits']['energy'] != 'Extraverted':
        energy_value = 100 - (101 + scores[0]) // 2
    else:
        energy_value = (101 + scores[0]) // 2
    if sess_r.json()['user']['traits']['mind'] != 'Intuitive':
        mind_value = 100 - (101 + scores[1]) // 2
    else:
        mind_value = (101 + scores[1]) // 2
    if sess_r.json()['user']['traits']['nature'] != 'Thinking':
        nature_value = 100 - (101 + scores[2]) // 2
    else:
        nature_value = (101 + scores[2]) // 2
    if sess_r.json()['user']['traits']['tactics'] != 'Judging':
        tactics_value = 100 - (101 + scores[3]) // 2
    else:
        tactics_value = (101 + scores[3]) // 2
    if sess_r.json()['user']['traits']['identity'] != 'Assertive':
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        identity_value = (101 + scores[4]) // 2

    code, role = parsing([energy_value, mind_value, nature_value, tactics_value, identity_value])

    return code, role, [energy_value, mind_value, nature_value, tactics_value, identity_value]


def get_model_examing_result(model_id):

    for mbti_item in prompt_template["mbti_prompt"][6:]:

        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]

        output_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/16p/Llama2-7b-chat-hf/combine-result/{mbti_label_content}-combine-induce-mbti-Llama2-7b-chat-hf-output.txt'
        result_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/16p/Llama2-7b-chat-hf/combine-result/{mbti_label_content}-combine-induce-mbti-Llama2-7b-chat-hf-result.csv'

        if not os.path.isfile(result_file_name):
            df = pd.DataFrame(columns=['Cycle', 'Code', 'Role','Values'])
            df.to_csv(result_file_name, index=False)

        with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a',
                                                                      encoding='utf-8') as r:

            for cycle in range(1, 21):
                #####
                try:
                    del pipeline
                except:
                    pass

                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )

                results = []
                mbti_questions = questionnaire["questions"]
                for question_num, question in mbti_questions.items():
                    messages = [
                        {"role": "system", "content": mbti_prompt },
                        {"role": "user", "content":" You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. Here is the statement: " + question}
                    ]
                    terminators = [
                        pipeline.tokenizer.eos_token_id,
                        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = pipeline(
                        messages,
                        max_new_tokens=256,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )

                    generated_text = outputs[0]["generated_text"]

                    f.write(f"cycle: {cycle}\n")
                    #print(f"cycle: {cycle}\n")
                    f.write(f"prompting: {mbti_prompt}")
                    #print(f"prompting: {mbti_prompt}"+"\nImagine you are a human with this personality."+"\n")
                    f.write("You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. Here is the statement: " +  question)
                    #print("You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. " + f"question: {question}\n")

                    f.write(f"generated_text: {generated_text}\n")
                    #print(f"generated_text: {generated_text}\n")

                    answer00 = generated_text[-1]["content"]
                    #print(f"raw_answer: {answer00}\n\n")
                    f.write(f"answer: {answer00}\n\n")

                    results.append(extract_first_number(answer00))
                    #print(f"results: {results}\n\n")
                    f.write(f"results: {results}\n\n")

                model_results = query_16personalities_api(results)
                #print(f"result: {model_results}\n\n")
                f.write(f"result: {model_results}\n\n")
                r.write(f"{cycle},{model_results[0]},{model_results[1]},\"{model_results[2]}\"\n")




def get_single_dim_model_examing_result(model_id):

    for mbti_item in single_prompt_template["mbti_prompt"]:
        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]

        output_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/16p/Llama2-7b-chat-hf/combine-result/{mbti_label_content}-combine-induce-mbti-Llama2-7b-chat-hf-output.txt'
        result_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/16p/Llama2-7b-chat-hf/combine-result/{mbti_label_content}-combine-induce-mbti-Llama2-7b-chat-hf-result.csv'

        if not os.path.isfile(result_file_name):
            df = pd.DataFrame(columns=['Cycle', 'Code', 'Role','Values'])
            df.to_csv(result_file_name, index=False)

        with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a',
                                                                      encoding='utf-8') as r:

            r.write("Cycle,Code,Role,Values\n")
            for cycle in range(1, 21):

                try:
                    del pipeline
                except:
                    pass

                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )

                results = []
                mbti_questions = questionnaire["questions"]
                for question_num, question in mbti_questions.items():
                    messages = [
                        {"role": "system", "content": mbti_prompt},
                        {"role": "user", "content":"You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. Here is the statement: " + question}
                    ]
                    terminators = [
                        pipeline.tokenizer.eos_token_id,
                        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = pipeline(
                        messages,
                        max_new_tokens=256,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )

                    generated_text = outputs[0]["generated_text"]

                    f.write(f"cycle: {cycle}\n")
                    print(f"cycle: {cycle}\n")
                    f.write(f"mbti_prompt: {mbti_prompt}\n")
                    print(f"mbti_prompt: {mbti_prompt}\n")
                    f.write("You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. Here is the statement: "+ question)
                    #print("You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. Here is the statement: "+f"question: {question}\n")
                    f.write(f"generated_text: {generated_text}\n")
                    #print(f"generated_text: {generated_text}\n")

                    answer00 = generated_text[-1]["content"]
                    print(f"answer: {answer00}\n\n")
                    f.write(f"answer: {answer00}\n\n")

                    results.append(extract_first_number(answer00))
                    print(f"results: {results}\n\n")

                model_results = query_16personalities_api(results)
                print(f"result: {model_results}\n\n")
                f.write(f"result: {model_results}\n\n")
                r.write(f"{cycle},{model_results[0]},{model_results[1]},\"{model_results[2]}\"\n")


if __name__ == '__main__':
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_api_key)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=huggingface_api_key)
#    get_single_dim_model_examing_result(model_id)
    get_model_examing_result(model_id)

