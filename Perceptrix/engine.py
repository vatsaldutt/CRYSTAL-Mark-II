from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from callbacks import Iteratorize, Stream
from langdetect import detect
import networkx as nx
import transformers
import requests
import locale
import torch
import spacy
import time
import tqdm
import sys
import os

locale.getpreferredencoding = lambda: "UTF-8"


tokenizer = LlamaTokenizer.from_pretrained(
    "JosephusCheung/Guanaco", use_fast=False)

model = LlamaForCausalLM.from_pretrained(
    "JosephusCheung/Guanaco",
    load_in_8bit=False,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload"
)

NOTFIRSTRUN = False

nlp_models = {
    'zh-cn': spacy.load('zh_core_web_sm'),
    'zh-tw': spacy.load('zh_core_web_sm'),
    'zh-hk': spacy.load('zh_core_web_sm'),
    'ja': spacy.load('ja_core_news_sm'),
    'de': spacy.load('de_core_news_sm'),
    'en': spacy.load('en_core_web_sm')
}


def detect_language(text, default='zh-cn'):
    if detect(text) == 'zh-cn':
        return 'zh-cn'
    if detect(text) == 'zh-tw':
        return 'zh-tw'
    if detect(text) == 'zh-hk':
        return 'zh-hk'
    if detect(text) == 'ja':
        return 'ja'
    if detect(text) == 'en':
        return 'en'
    else:
        return default


def tokenizeNLP(sentence, language, stopwords=None):
    if stopwords is None:
        stopwords = []
    doc = nlp_models[language](sentence)
    return ' '.join([token.text for token in doc if token.text not in stopwords])


def split_sentences(text, language):
    doc = nlp_models[language](text)
    sentences = []
    for sent in doc.sents:
        sent_text = str(sent).strip()
        if sent_text:
            for delimiter in ['\n', '\t']:
                if delimiter in sent_text:
                    sentences.extend(sent_text.split(delimiter))
                    break
            else:
                sentences.append(sent_text)
    return sentences


def similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer(tokenizer=lambda s: s.split())
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]


def textrank(sentences, language, window=2):
    nlp = nlp_models[language]
    docs = [nlp(sent) for sent in sentences]

    edge_weights = {}

    for doc in docs:
        for token1 in doc:
            if not token1.is_punct and not token1.is_stop:
                for token2 in doc[token1.i - window: token1.i + window + 1]:
                    if not token2.is_punct and not token2.is_stop and token1 != token2:
                        pair = tuple(sorted([token1.i, token2.i]))
                        edge_weights[pair] = edge_weights.get(pair, 0) + 1

    graph = nx.Graph()
    graph.add_edges_from(
        [(pair[0], pair[1], {"weight": weight}) for pair, weight in edge_weights.items()])

    scores = nx.pagerank(graph)

    return {sent: scores.get(idx, 0) for idx, sent in enumerate(sentences)}


def find_top_k_sentences(input_a, input_b, k, stopwords=None, language='zh-cn'):
    tokenized_a = tokenizeNLP(input_a, language, stopwords)
    sentences_b = split_sentences(input_b, language)
    scores = textrank(sentences_b, language)

    sorted_scores = sorted(
        scores.items(), key=lambda x: x[1], reverse=True)[:k*5]
    top_k_sentences = [sent for sent, _ in sorted_scores]

    similarities = [(sentence, similarity(tokenized_a, tokenizeNLP(
        sentence, language, stopwords))) for sentence in top_k_sentences]
    sorted_similarities = sorted(
        similarities, key=lambda x: x[1], reverse=True)

    top_sentences = [sentence for sentence,
                     _ in sorted_similarities[:min(k, len(sorted_similarities))]]
    return '\n'.join(top_sentences)
    return top_sentences


stopwords = ["，", "。", "的", "、", "在", "等", "时", "\n", "\t"]


def search_wikipedia(keyword, url='https://zh.wikipedia.org/w/api.php', language='zh-cn'):
    if language == 'zh-cn':
        url = 'https://zh.wikipedia.org/w/api.php'
    if language == 'zh-tw':
        url = 'https://zh.wikipedia.org/w/api.php'
    if language == 'zh-hk':
        url = 'https://zh.wikipedia.org/w/api.php'
    if language == 'ja':
        url = 'https://ja.wikipedia.org/w/api.php'
    if language == 'en':
        url = 'https://en.wikipedia.org/w/api.php'
    if language == 'en':
        url = 'https://de.wikipedia.org/w/api.php'

    print("Wikipedia: "+keyword)
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': keyword,
        'uselang': language
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data['query']['search']:
        title = data['query']['search'][0]['title']
        return get_wikipedia_page(title, url)
    else:
        return 'No results found.'


def get_wikipedia_page(title, url='https://zh.wikipedia.org/w/api.php', language='zh-cn'):
    if language == 'zh-cn':
        url = 'https://zh.wikipedia.org/w/api.php'
    if language == 'zh-tw':
        url = 'https://zh.wikipedia.org/w/api.php'
    if language == 'zh-hk':
        url = 'https://zh.wikipedia.org/w/api.php'
    if language == 'ja':
        url = 'https://ja.wikipedia.org/w/api.php'
    if language == 'en':
        url = 'https://en.wikipedia.org/w/api.php'
    if language == 'en':
        url = 'https://de.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'titles': title,
        'explaintext': 1,
        'exintro': 1,
        'uselang': language
    }

    response = requests.get(url, params=params)
    data = response.json()

    page_id = next(iter(data['query']['pages']))
    return data['query']['pages'][page_id]['extract']


def search_moepedia(keyword, language='zh-cn'):
    url = 'https://zh.moegirl.org.cn/api.php'
    print("Moepedia: "+keyword)
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'grsearch': keyword,
        'srsearch': keyword
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'
    }

    response = requests.get(url, params=params, headers=headers)

    data = response.json()

    if data['query']['search']:
        title = data['query']['search'][0]['title']
        return get_moepedia_page(title, language)
    else:
        return 'No results found.'


def get_moepedia_page(title, language='zh-cn'):
    url = 'https://zh.moegirl.org.cn/api.php'

    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'titles': title,
        'explaintext': 1,
        'exintro': 1,
        'uselang': language
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    page_id = next(iter(data['query']['pages']))
    return data['query']['pages'][page_id]['extract']


def get_main_topic(nlp_models, text, language='zh-cn'):
    nlp = nlp_models[language]
    doc = nlp(text)

    blacklist = ["是", "请",  "进行",  "对话",  "可以",  "说话",  "模仿",  "扮演",  "需要",  "帮助",  "应该",  "处理",  "怎样",  "哪些",  "做",  "完成",  "执行",  "动漫",  "角色",  "说",  "做",  "有",  "去",  "来",  "想",  "看",  "听",  "学",  "工作",  "写",  "读",  "买",  "卖",  "吃",  "喝",  "睡",  "玩",  "打",  "开",  "关",  "送",  "接",  "写作",  "画画",  "唱歌",  "跳舞",
                 "运动",  "旅行",  "摄影",  "聊天",  "约会",  "分析",  "解决",  "设计",  "编程",  "管理",  "领导",  "教育",  "学习",  "娱乐",  "交流",  "沟通",  "思考",  "创造",  "表达",  "评估",  "计划",  "执行",  "调查",  "改进",  "提高",  "探索",  "发现",  "建议",  "解释",  "阐述",  "介绍",  "展示",  "演讲",  "辩论",  "审查",  "评价",  "审核",  "批评",  "反驳",  "比较",  "分辨",  "鉴别",  "辨认",  "辨别", "获得"]
    blacklist += ["是", "請",  "進行",  "對話",  "可以",  "說話",  "模仿",  "扮演",  "需要",  "幫助",  "應該",  "處理",  "怎樣",  "哪些",  "做",  "完成",  "執行",  "動漫",  "角色",  "說",  "做",  "有",  "去",  "來",  "想",  "看",  "聽",  "學",  "工作",  "寫",  "讀",  "買",  "賣",  "吃",  "喝",  "睡",  "玩",  "打",  "開",  "關",  "送",  "接",  "寫作",  "畫畫",  "唱歌",  "跳舞",
                  "運動",  "旅行",  "攝影",  "聊天",  "約會",  "分析",  "解決",  "設計",  "編程",  "管理",  "領導",  "教育",  "學習",  "娛樂",  "交流",  "溝通",  "思考",  "創造",  "表達",  "評估",  "計劃",  "執行",  "調查",  "改進",  "提高",  "探索",  "發現",  "建議",  "解釋",  "闡述",  "介紹",  "展示",  "演講",  "辯論",  "審查",  "評價",  "審核",  "批評",  "反駁",  "比較",  "分辨",  "鑑別",  "辨認",  "辨別", "獲得"]
    blacklist += ["please", "conduct", "dialogue", "able to", "speak", "mimic", "impersonate", "need", "help", "should",   "handle", "how to", "which ones", "do", "accomplish", "execute", "anime", "character", "say", "do",  "have", "go", "come", "want", "see", "listen", "learn", "work", "write", "read", "buy", "sell",  "eat", "drink", "sleep", "play", "hit", "open", "close", "send", "receive", "write", "draw",  "sing", "dance", "exercise", "travel", "photography", "chat",
                  "date", "analyze", "solve", "design",  "program", "manage", "lead", "education", "study", "entertainment", "communication", "exchange",  "think", "create", "express", "evaluate", "plan", "execute", "investigate", "improve", "enhance",  "explore", "discover", "suggest", "explain", "elaborate", "introduce", "show", "speak", "debate",  "scrutinize", "evaluate", "review", "critique", "rebut", "compare", "distinguish", "differentiate",  "recognize", "identify", "acquire"]
    blacklist += ["bitte", "durchführen", "Dialog", "können", "sprechen", "nachahmen", "spielen", "brauchen", "Hilfe", "sollte", "verarbeiten", "wie", "welche", "tun", "erledigen", "ausführen", "Anime", "Rolle", "sagen", "tun", "haben", "gehen", "kommen", "denken", "sehen", "hören", "lernen", "arbeiten", "schreiben", "lesen", "kaufen", "verkaufen", "essen", "trinken", "schlafen", "spielen", "schalten", "öffnen", "schließen", "senden", "empfangen", "schreiben", "zeichnen", "singen", "tanzen", "Sport treiben", "reisen", "Fotografie", "chatten",
                  "verabreden", "analysieren", "lösen", "designen", "programmieren", "verwalten", "führen", "Ausbildung", "lernen", "unterhalten", "kommunizieren", "nachdenken", "schaffen", "ausdrücken", "bewerten", "planen", "ausführen", "untersuchen", "verbessern", "erhöhen", "erforschen", "entdecken", "vorschlagen", "erklären", "erläutern", "einführen", "präsentieren", "Rede halten", "Debatte führen", "überprüfen", "bewerten", "überprüfen", "kritisch", "widerlegen", "vergleichen", "unterscheiden", "unterscheiden", "erkennen", "unterscheiden", "erhalten"]
    blacklist += ["話す", "対話", "模倣", "演じる", "要する", "助ける", "すべき", "処理する", "どのように", "どの", "する", "完了する", "実行する", "アニメ", "役割", "言う", "持つ", "行く", "来る", "考える", "見る", "聞く", "学ぶ", "仕事", "書く", "読む", "買う", "売る", "食べる", "飲む", "寝る", "遊ぶ", "打つ", "開く", "閉じる", "送る", "受け取る", "執筆する", "描く", "歌う", "踊る", "運動する", "旅行する", "写真を撮る", "チャットする", "デートする", "分析する",
                  "解決する", "デザインする", "プログラミングする", "マネジメントする", "リーダーシップする", "教育する", "学習する", "娯楽", "コミュニケーションする", "コミュニケーション", "考える", "創造する", "表現する", "評価する", "計画する", "実行する", "調査する", "改善する", "向上する", "探検する", "発見する", "提言する", "説明する", "説明する", "紹介する", "ショー", "スピーチ", "ディベート", "審査する", "評価する", "審査する", "批評する", "反論する", "比較する", "識別する", "鑑別する", "識別する", "識別する", "入手する"]

    entities = [ent.text for ent in doc.ents if ent.text not in blacklist]

    if entities and len(" ".join(entities)) > 2:
        main_topic = " ".join(entities)
    else:
        tagged = [token.text for token in doc if (token.pos_ in {"PROPN"}) and (
            token.dep_ != "aux") and (token.text not in blacklist)]

        if tagged:
            main_topic = " ".join(tagged)
        else:
            tagged = [token.text for token in doc if (token.pos_ in {"NOUN", "VERB"}) and (
                token.dep_ != "aux") and (token.text not in blacklist)]
            if tagged:
                main_topic = " ".join(tagged)
            else:
                main_topic = text
    return main_topic


def getSystem(last, input):
    lang = detect_language(input)
    keyword = get_main_topic(nlp_models, input, lang)
    if input == keyword:
        return ""
    if lang == "zh-cn" or lang == "zh-tw" or lang == "zh-hk":

        result = search_moepedia(keyword, language=lang)
        if result != 'No results found.':
            top_k_sentences = find_top_k_sentences(
                keyword, result, k=8, stopwords=stopwords, language=lang)
            if top_k_sentences and top_k_sentences.strip() != "" and bool(set(keyword) & set(top_k_sentences)):
                return keyword+'\n'+top_k_sentences + '\n'
    result = search_wikipedia(keyword, language=lang)
    if result == 'No results found.':
        if last and last.strip() != "":
            return last
        return ""
    top_k_sentences = find_top_k_sentences(
        keyword, result, k=8, stopwords=stopwords, language=lang)
    if top_k_sentences and top_k_sentences.strip() != "":
        return keyword+'\n'+top_k_sentences + '\n'
    if last and last.strip() != "":
        return last
    return ""


PROMPT = '''### Instruction: 
{}
### Input:
{}
### Response:'''

PROMPT_INS = '''### Instruction: 
{}
### Response:'''

# unwind broken decapoda-research config
if not NOTFIRSTRUN:
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
NOTFIRSTRUN = True


def evaluate(
    prompt='',
    temperature=0.4,
    top_p=0.65,
    top_k=35,
    repetition_penalty=1.1,
    max_new_tokens=512,
    stream_output=False,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield decoded_output.split("### Response:")[-1].strip()
        return  # early return for stream_output

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield output.split("### Response:")[-1].strip()


answer = ""
instruction = """You are a chatbot made by Vatsal Dutt designed to behave like JARVIS from Iron Man. Your name is CRYSTAL which stands for Comprehensive Robotics Yielding Sophisticated Technology And Logistics."""
inputs = "What is 2+2"


def run_instruction(
    instruction,
    inputs,
    temperature=0.4,
    top_p=0.65,
    top_k=35,
    repetition_penalty=1.1,
    max_new_tokens=512,
    stream_output=False,
):
    if inputs.strip() == '':
        now_prompt = PROMPT_INS.format(instruction)
    else:
        now_prompt = PROMPT.format(instruction+'\n', inputs)

    response = evaluate(
        now_prompt, temperature, top_p, top_k, repetition_penalty, max_new_tokens, stream_output
    )
    if stream_output:
        response = tqdm.tqdm(response, unit='token')
    for i in response:
        i = i
        yield i


temp = 0.4
topp = 0.65
topk = 35
repp = 1.1

maxt = 512
maxh = 5
stream_output = True

answer = ''.join(run_instruction(
    instruction,
    inputs,
    temperature=0.4,
    top_p=0.65,
    top_k=35,
    repetition_penalty=1.1,
    max_new_tokens=512,
    stream_output=False,
))[:-7]

print(answer)

sys_input = ""
sys_net = False

temp = 0.4  # Temperature
topp = 0.65
topk = 35
repp = 1.1

maxt = 1000  # Maximum Tokens
maxh = 5  # Maximum history messages


chat_history = []


def user(user_message, history):
    for idx, content in enumerate(history):
        history[idx] = [
            content[0].replace('<br>', ''),
            content[1].replace('<br>', '')
        ]
    user_message = user_message.replace('<br>', '')
    return "", history + [[user_message, None]]


def bot(
    history,
    temperature=0.4,
    top_p=0.65,
    top_k=35,
    repetition_penalty=1.1,
    max_new_tokens=512,
    maxh=10,
    stream_output=False,
    system_prompt="",
    system_net=False,
    username="User"
):
    instruction = """You are a chatbot made by Vatsal Dutt designed to behave like JARVIS from Iron Man. Your name is CRYSTAL which stands for Comprehensive Robotics Yielding Sophisticated Technology And Logistics."""
    hist = f"""{username}: {instruction}\nCRYSTAL: Okay!\n"""
    for idx, content in enumerate(history):
        history[idx] = [
            content[0].replace('<br>', ''),
            None if content[1] is None else content[1].replace('<br>', '')
        ]
    roleplay_keywords = ["模仿", "扮演", "作为", "作為", "装作", "裝作"]
    for user, assistant in history[:-1]:
        user = user
        assistant = assistant
        hist += f'{username}: {user}\nCRYSTAL: {assistant}\n'
    if system_net and not any(roleplay_keyword in history[-1][0] for roleplay_keyword in roleplay_keywords):
        system_prompt += getSystem(system_prompt, history[-1][0])
    now_prompt = PROMPT.format(hist, ("CRYSTAL: " + system_prompt if system_prompt !=
                               "" else system_prompt+'\n\n') + f"{username}: {history[-1][0]}")
    if system_net and any(roleplay_keyword in history[-1][0] for roleplay_keyword in roleplay_keywords):
        system_prompt += getSystem(system_prompt, history[-1][0])
    # print(now_prompt)

    if not system_net or not any(roleplay_keyword in history[-1][0] for roleplay_keyword in roleplay_keywords):
        system_prompt = ""
    bot_message = evaluate(
        now_prompt, temperature, top_p, top_k, repetition_penalty, max_new_tokens, stream_output
    )

    if stream_output:
        bot_message = tqdm.tqdm(bot_message, unit='token')
    for mes in bot_message:
        mes = mes
        history[-1][1] = mes

        history = history[-maxh:]

        yield [history, system_prompt]


def ask_crystal(user_message, chat_history, username="User"):
    response, updated_chat_history = user(user_message, chat_history)

    # Step 3: Bot generates a response
    bot_response_generator = bot(updated_chat_history, temp, topp,
                                 topk, repp, maxt, maxh, stream_output, sys_input, sys_net, username=username)
    # Step 4: Iterate over the bot response generator
    for [history, system_prompt] in bot_response_generator:
        os.system('clear')
        # Step 5: Process or display bot's responses and updated history
        # Get the bot's response from the last entry in the history
        bot_response = history[-1][1]
        with open("reply.txt", 'w') as reply:
            reply.write(bot_response)
        time.sleep(0.1)
        with open("reply.txt", 'w') as reply:
            reply.write("")

        print("CRYSTAL:", bot_response)
        # Optionally, you can append the bot's response to the chat history if needed:
        history[-1][1] = bot_response
        chat_history = history
    return chat_history


def chat():
    chat_history = []
    while True:
        chat_history = ask_crystal(input("Enter Query: "), chat_history)
