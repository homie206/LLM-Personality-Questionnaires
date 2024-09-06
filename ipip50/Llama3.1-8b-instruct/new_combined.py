import json
import os
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import re
import pandas as pd
import ast

def get_final_scores(columns, dim):
    score = 0
    if dim == 'EXT':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score += (6 - columns[9])
    if dim == 'EST':
        score += (6 - columns[0])
        score += columns[1]
        score += (6 - columns[2])
        score += columns[3]
        score += (6 - columns[4])
        score += (6 - columns[5])
        score += (6 - columns[6])
        score += (6 - columns[7])
        score += (6 - columns[8])
        score += (6 - columns[9])
    if dim == 'AGR':
        score += (6 - columns[0])
        score += columns[1]
        score += (6 - columns[2])
        score += columns[3]
        score += (6 - columns[4])
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score += columns[8]
        score += columns[9]
    if dim == 'CSN':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score += columns[9]
    if dim == 'OPN':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += columns[7]
        score += columns[8]
        score += columns[9]
    return score


prompt_template = {
    "ipip50_prompt": [
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-High', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-High', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-High', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-Low', 'N-High', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You are introverted and reserved. You prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. You may also be less assertive and more cautious in your interactions with others. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-Low', 'N-High', 'A-High', 'C-High', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-Low', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-Low', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-Low', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You tend to be more prone to negative emotions, such as anxiety, depression, and anger. You may be more reactive to stress and may find it difficult to cope with challenging situations. You may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-Low', 'A-High', 'C-High', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-High', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-High', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-High', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You tend to be more competitive and skeptical. You may be less motivated to maintain social harmony and may be more likely to express your opinions forcefully, even if you may conflict with others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-High', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-High', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You tend to be more impulsive and disorganized. You may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in your best interest. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-High', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You tend to be more traditional and conservative. You may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        'label': "['E-High', 'N-High', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You tend to be outgoing, sociable, and talkative. You enjoy being around others and seek out social situations. You are often described as having a high level of energy, enthusiasm, and assertiveness. You may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. You are emotionally resilient, calm, and even-tempered. You tend to experience fewer negative emotions and are better able to cope with stress and adversity. You are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. You are characterized as being warm, kind, and considerate. You tend to be cooperative and are motivated to maintain harmonious social relationships. You may also have a strong sense of empathy and concern for the welfare of others. You are characterized as being reliable, hardworking, and efficient. You tend to be well-organized and responsible, and are motivated to achieve your goals. You may also exhibit a strong sense of self-discipline and perseverance. You are characterized as being imaginative, curious, and open to new ideas and experiences. You tend to be intellectually curious and enjoy exploring new concepts and ideas. You may also exhibit a preference for creativity and aesthetics.',
        'label': "['E-High', 'N-High', 'A-High', 'C-High', 'O-High']"}
    ]
}




# 创建列名列表
column_names = ['EXT1', 'AGR1', 'CSN1', 'EST1', 'OPN1',
                'EXT2', 'AGR2', 'CSN2', 'EST2', 'OPN2',
                'EXT3', 'AGR3', 'CSN3', 'EST3', 'OPN3',
                'EXT4', 'AGR4', 'CSN4', 'EST4', 'OPN4',
                'EXT5', 'AGR5', 'CSN5', 'EST5', 'OPN5',
                'EXT6', 'AGR6', 'CSN6', 'EST6', 'OPN6',
                'EXT7', 'AGR7', 'CSN7', 'EST7', 'OPN7',
                'EXT8', 'AGR8', 'CSN8', 'EST8', 'OPN8',
                'EXT9', 'AGR9', 'CSN9', 'EST9', 'OPN9',
                'EXT10', 'AGR10', 'CSN10', 'EST10', 'OPN10']

# 创建 DataFrame
df = pd.DataFrame(columns=column_names)

def extract_first_number(answer):
    match = re.search(r'^\d+', answer)
    if match:
        return int(match.group())
    else:
        return None

def get_response(q, pip_line):
        pip_line = pipeline

        messages = [
            {"role": "system", "content":"Imagine you are a human. " + ipip_prompt },
            {"role": "user", "content": '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                            1. Very Inaccurate
                            2. Moderately Inaccurate 
                            3. Neither Accurate Nor Inaccurate
                            4. Moderately Accurate
                            5. Very Accurate
                            Please only answer with the option number. \nHere is the statement: ''' + q }
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

        generated_text = outputs[0]["generated_text"][-1]["content"]
        #     print('generated_text', generated_text)
        return generated_text



if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    for ipip_item in prompt_template["ipip50_prompt"]:
        ipip_prompt = ipip_item["prompt"]
        #ipip_label_content = ipip_item["label"]
        ipip_label_content = ast.literal_eval(ipip_item["label"])  # 转换为列表
        ipip_label_content_str = '-'.join(ipip_label_content)

        output_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/ipip50/Llama3.1-8b-instruct/new-combined-result/{ipip_label_content_str}-new-combined-ipip50-llama3.1-8b-instruct-output.txt'
        result_file_name = f'/home/hmsun/LLM-Personality-Questionnaires/ipip50/Llama3.1-8b-instruct/new-combined-result/{ipip_label_content_str}-new-combined-ipip50-llama3.1-8b-instruct-result.csv'

        if not os.path.isfile(result_file_name):
            df = pd.DataFrame(columns=['EXT1','AGR1','CSN1','EST1','OPN1','EXT2','AGR2','CSN2','EST2','OPN2','EXT3','AGR3','CSN3','EST3','OPN3','EXT4','AGR4','CSN4','EST4','OPN4','EXT5','AGR5','CSN5','EST5','OPN5','EXT6','AGR6','CSN6','EST6','OPN6','EXT7','AGR7','CSN7','EST7','OPN7','EXT8','AGR8','CSN8','EST8','OPN8','EXT9','AGR9','CSN9','EST9','OPN9','EXT10','AGR10','CSN10','EST10','OPN10'])
            df.to_csv(result_file_name, index=False)

        with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a', encoding='utf-8') as r:
            with open('../IPIP-50.txt', 'r') as f2:
                question_list = f2.readlines()
                answer_list = []
                extracted_numbers = []
                all_results = []

                for run in range(20):  # 运行100次

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

                    extracted_numbers = []

                    for q in question_list:
                        answer = get_response(q, pipeline)
                        f.write(answer + '\n')
                        extracted_number = extract_first_number(answer)
                        extracted_numbers.append(extracted_number)

                        print(f"Cycle {run+1} extracted numbers:")
                        f.write(f"Cycle {run+1} extracted numbers:")
                        print(extracted_numbers)
                        f.write(', '.join(map(str, extracted_numbers)) + '\n')
                        #all_results.append(extracted_numbers)

                        f.write(f"cycle: {run+1}\n")
                        print(f"cycle: {run+1}\n")
                        f.write(f"prompting: Imagine you are a human. {ipip_prompt}\n")
                        print(f"prompting: Imagine you are a human. {ipip_prompt}\n")
                        f.write(
                            '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                                1. Very Inaccurate
                                2. Moderately Inaccurate 
                                3. Neither Accurate Nor Inaccurate
                                4. Moderately Accurate
                                5. Very Accurate
                                Please only answer with the option number. \nHere is the statement: ''' + q)
                        print(
                            '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                                1. Very Inaccurate
                                2. Moderately Inaccurate 
                                3. Neither Accurate Nor Inaccurate
                                4. Moderately Accurate
                                5. Very Accurate
                                Please only answer with the option number. \nHere is the statement: ''' + q)
                        f.write(answer + '\n')
                        print(answer + '\n')

                    print(f"Run {run + 1} extracted numbers:")
                    print(extracted_numbers)

                    all_results.append(extracted_numbers)

                    # 将结果转换为 DataFrame
                result_df = pd.DataFrame(all_results, columns=column_names)

                # 保存结果到 CSV 文件
                result_df.to_csv(result_file_name, index=False)

            df = pd.read_csv(result_file_name, sep=',')

            dims = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
            columns = [i + str(j) for j in range(1, 11) for i in dims]
            df = df[columns]

            for i in dims:
                df[i + '_all'] = df.apply(
                    lambda r: get_final_scores(columns=[r[i + str(j)] for j in range(1, 11)], dim=i),
                    axis=1)

            for i in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
                print(f"{i}_all:")
                print(df[i + '_all'])
                print()

            final_scores = [df[i + '_all'][0] for i in dims]
            print(final_scores)

            for i in dims:
                df[i + '_Score'] = df.apply(
                    lambda r: get_final_scores(columns=[r[i + str(j)] for j in range(1, 11)], dim=i), axis=1)

            original_df = pd.read_csv(result_file_name, sep=',')

            # 合并新旧数据
            result_df = pd.concat([original_df, df[['EXT_Score', 'EST_Score', 'AGR_Score', 'CSN_Score', 'OPN_Score']]],
                                  axis=1)
            # 保存结果到 CSV 文件
            result_df.to_csv(result_file_name, index=False)








