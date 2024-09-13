from fastapi import requests
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


#1.加入MBTI得到400个词
#2.得到向量 4096维度
#3.TSNE聚类可视化 得到一个400个词语的聚类图

#generate 100 adjectives to describe an introverted person defined by the MBTI personality model

#Please generate 50 adjectives to describe such person: Introverted individuals prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general.

single_prompt_template = {
    "mbti_prompt": [
        {
            "prompt": "Please generate 50 adjectives to describe such person: Extraverted individuals prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited than Introverts.",
            "label": "Extrovert"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Introverted individuals prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general.",
            "label": "Introvert"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Observant individuals are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened.",
            "label": "Observant"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Intuitive individuals are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities.",
            "label": "Intuitive"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Thinking individuals focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation.",
            "label": "Thinking"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Feeling individuals are sensitive and emotionally expressive. They are more empathic and less competitive than Thinking types, and focus on social harmony and cooperation.",
            "label": "Feeling"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Judging individuals are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "Judging"
        },
        {
            "prompt": "Please generate 50 adjectives to describe such person: Prospecting individuals are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists who prefer keeping their options open.",
            "label": "Prospecting"
        }


    ],

}













def get_model_examing_result(model_id):

    for mbti_item in single_prompt_template["mbti_prompt"]:

        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]

        output_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/words/gen_words_combined/{mbti_label_content}-words-mbti-llama3.1-8b-instruct-output.txt'

        with open(output_file_name, 'a', encoding='utf-8') as f:

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

            messages = [
                #{"role": "system", "content": mbti_prompt },
                {"role": "user", "content": mbti_prompt}
            ]
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            generated_text = outputs[0]["generated_text"]

            f.write(f"prompting: {mbti_prompt}")
            print(f"prompting: {mbti_prompt}")

            f.write(f"generated_text: {generated_text}\n")
            print(f"generated_text: {generated_text}\n")

            answer00 = generated_text[-1]["content"]
            print(f"raw_answer: {answer00}\n\n")
            f.write(f"answer: {answer00}\n\n")


if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    get_model_examing_result(model_id)

