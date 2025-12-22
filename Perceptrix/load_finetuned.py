from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from callbacks import Iteratorize, Stream
import transformers
import locale
import torch
import tqdm
import sys
import os

locale.getpreferredencoding = lambda: "UTF-8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()

model_id = f"{folder_location}models/CRYSTAL-model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(
    model_id,
    use_fast=False)


model = LlamaForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

NOTFIRSTRUN = False

PROMPT = '''### Instruction: 
{}
### Input:
{}
### Response:'''

if not NOTFIRSTRUN:
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
NOTFIRSTRUN = True

lora_config = LoraConfig.from_pretrained("CRYSTAL-fine")
model = get_peft_model(model, lora_config)

model = torch.compile(model)


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

    if str(device) == "cuda":
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        input_ids = inputs["input_ids"].to("cuda:0")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

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
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield decoded_output.split("### Response:")[-1].strip()
        return

    with torch.no_grad():
        generation_output = model.generate(
            **input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    yield output.split("### Response:")[-1].strip()


temp = 0.4
topp = 0.65
topk = 35
repp = 1.1

maxt = 512
maxh = 5
stream_output = True

sys_input = ""
sys_net = False


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
    current_events,
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
    instruction = """You are an AI made by Vatsal Dutt named CRYSTAL which stands for Comprehensive Robotics Yielding Sophisticated Technology And Logistics."""
    current_events = "\nThese are the latest updates:\n"+current_events
    hist = f"""{username}: {instruction+current_events}\nCRYSTAL: Okay!\n"""
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

    now_prompt = PROMPT.format(hist, ("CRYSTAL: " + system_prompt if system_prompt !=
                               "" else system_prompt+'\n\n') + f"{username}: {history[-1][0]}")

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


def ask_crystal(user_message, chat_history, current_events, username="User"):
    current_events = "Use the following data as input to answer any of the user queries:\n"+current_events
    user_message = current_events+"\n"+user_message
    response, updated_chat_history = user(user_message, chat_history)

    bot_response_generator = bot(updated_chat_history, current_events, temp, topp,
                                 topk, repp, maxt, maxh, stream_output, sys_input, sys_net, username=username)

    for [history, system_prompt] in bot_response_generator:
        os.system('clear')

        bot_response = history[-1][1]
        print(bot_response)

        history[-1][1] = bot_response
        chat_history = history
    return bot_response, chat_history


def chat():
    chat_history = []
    while True:
        events = """Time: 9:33 AM"""
        response, chat_history = ask_crystal(
            input("Enter Query: "), chat_history, current_events=events)
