from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from CircumSpect.analyze import analyze_image
from selenium.webdriver.common.by import By
from SoundScribe.speakerID import find_user
# from Perceptrix.engine import ask_crystal
from SoundScribe.transcribe import listen
from SoundScribe.speak import speak
from googlesearch import search
from selenium import webdriver
from bs4 import BeautifulSoup
import trafilatura
import threading
import datetime
import warnings
import requests
import whisper
import urllib
import time
import cv2


warnings.filterwarnings("ignore")

stream = cv2.VideoCapture(1)
time.sleep(2)

chat_history = []

image_output = ""
weather = ""
annotated_image = None
recognized_text = ""

model = whisper.load_model("base")

op = webdriver.ChromeOptions()
op.add_argument('--headless')
op.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(service=Service(
    ChromeDriverManager().install()), options=op)

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()


def setup_driver():
    op = webdriver.ChromeOptions()
    op.add_argument('--headless')
    op.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=op)


def scrape_article(url):
    downloaded = trafilatura.fetch_url(url)
    extracted = trafilatura.extract(downloaded)
    return extracted


def get_articles(urls):
    articles = []
    for article in urls:
        article_text = scrape_article(article)
        if article_text and article_text != "" and article_text != "When you have eliminated the JavaScript, whatever remains must be an empty page." and article_text != "When you have eliminated the\nJavaScript\n, whatever remains must be an empty page.\nEnable JavaScript to see Google Maps." and "Something went wrong. Wait a moment and try again.\nTry again\nPlease enable Javascript and refresh the page to continue" != article_text:
            articles.append(
                ".".join(article_text[:3000].split('.')[:-1]).split("|")[-1])
        if len(articles) == 1:
            break
    return articles


def web_scraper(web_query):
    driver = setup_driver()
    urls = list(search(web_query, num_results=10, sleep_interval=0.1))
    print(urls)
    image_source = requests.get(
        f"https://www.google.com/search?tbm=isch&q={web_query}").text
    image_soup = BeautifulSoup(image_source, 'html.parser')
    images = image_soup.find_all('img')
    image_urls = images[4:]
    image_urls = []

    web_query = web_query.replace(
        '+', "%2B").replace(' ', '+').replace('=', '%3D')
    url = "https://www.google.com/search?q="+web_query
    driver.get(url)
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'html.parser')
    part_of_speeches = ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition',
                        'conjunction', 'interjection', 'exclamation', 'numeral', 'article', 'determiner']
    list1 = []
    articles = []

    for i in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        for j in i.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            list1.append(j.text)

    try:
        top_result_element = soup.find('div', class_='BNeawe iBp4i AP7Wnd')
    except:
        pass
    if not top_result_element:
        try:
            top_result_element = driver.find_element(By.CLASS_NAME, "IZ6rdc")
        except:
            pass
    if not top_result_element:
        try:
            top_result_element = driver.find_element(
                By.CLASS_NAME, "Z0LcW CfV8xf")
        except:
            pass
    if not top_result_element:
        try:
            top_result_element = driver.find_element(
                By.CLASS_NAME, "ayqGOc kno-fb-ctx KBXm4e")
        except:
            pass
    top_result = top_result_element.text if top_result_element else None

    if top_result:
        if top_result == "":
            articles = get_articles(urls)
        return top_result, image_urls, articles

    try:
        if list1[0].split()[0] in part_of_speeches:
            pos = list1[0].split()[0]
            if pos[0] == "a":
                return f'As an {pos} it means {list1[1]}', image_urls, articles
            else:
                return f'As a {pos} it means {list1[1]}', image_urls, articles
    except:
        pass

    try:
        for text in list1:
            list_text = text.split()
            if len(list_text) != 0 and list_text[-1] == 'Wikipedia':
                return f'According to Wikipedia, {"/".join(text.split()[0:-1]).replace("/", " ")}', image_urls, articles
    except:
        pass

    try:
        for i in soup.find_all('div'):
            for j in i.find_all('div'):
                for k in j.find_all('div'):
                    for m in k.find_all('div'):
                        if 'MUxGbd u31kKd gsrt lyLwlc' in str(m):
                            translation = str(m.text).replace(
                                'Translation', '').replace('Translate', '')
                            return translation, image_urls, articles
    except:
        pass

    try:
        if "youtube.com" in urls[0]:
            driver.get(urls[0])
            transcript_elements = driver.find_elements(
                By.CLASS_NAME, "ytd-transcript-segment-renderer")
            transcript = "\n".join(
                [element for element in transcript_elements])
            if transcript:
                return transcript, image_urls, articles
    except:
        pass

    # articles = get_articles(urls)
    articles = ""
    top_results = ""
    driver.quit()
    return top_results, image_urls, articles


def get_time():
    return datetime.datetime.now().strftime('%a %d %b %Y %I:%M %p')


def get_weather_data():
    while True:
        try:
            global weather
            driver.get('https://www.google.com/search?q=weather')
            weather_data = driver.find_element(By.CLASS_NAME, 'UQt4rd')
            weather_data = weather_data.text
            data_list = weather_data.split('\n')
            data_list[0] = data_list[0][0:-2]
            data_list.append(driver.find_element(By.ID, 'wob_dc').text)
            location = driver.find_element(By.CLASS_NAME, "BBwThe").text
            weather_icon_link = driver.find_element(By.ID,
                                                    'wob_tci').get_attribute('src')
            url = weather_icon_link
            with urllib.request.urlopen(url) as url1:
                weather_data = url1.read()
            delay = 120
            temp = data_list[0]
            weather_details = (f'{data_list[1]} {data_list[2]} {data_list[3]}')
            weather_name = data_list[-1]
            formatted = f'Weather in {location} is: {temp}, {weather_details}, {weather_name}'
            weather = formatted
            time.sleep(delay)
        except Exception as e:
            print(
                "Error Fetching Weather Information. Consider Checking Your Internet Connection")
            print("Detail: ", e)


def ask_crystal(query, user):
    global live_info
    global image_output
    if query:
        top_results, images, articles = web_scraper(query)
        if top_results != "" and articles != []:
            events = f"{live_info}\n{image_output}\nTop web results: {top_results}\n {articles[0]}"

        elif top_results != "" and articles == []:
            events = f"{live_info}\n{image_output}\nTop web results: {top_results}"

        elif top_results == "" and articles != []:
            events = f"{live_info}\n{image_output}\nTop Web results: {articles[0]}"
        elif top_results == "" and articles == []:
            events = f"{live_info}\n{image_output}\nError scraping web data"

        print("-"*100)
        print(query)
        print(events)
        print(user)
        print("-"*100)
        parameters = [0.4, 0.65, 35, 1.1, 512, 5]
        url = f"https://bceb7f41087d-7754001953109090881.ngrok-free.app/crystal"
        payload = {
            "user_id": "THEHACKER",
            "query": query,
            "hyperparameters": parameters,
            "events": events,
            "user": user
        }

        response = requests.get(url, params=payload, stream=True)
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line == "###---ENDofRESPONSE---###!!!":
                    full_response = ""
                else:
                    full_response += line+"\n"
                    print(full_response)
                    with open(f"{folder_location}database/output.txt", 'w') as reply_file:
                        if full_response[-1:] == "\n":
                            reply_file.write(full_response[:-1])
                        else:
                            reply_file.write(full_response)

        with open(f"{folder_location}database/input.txt", 'w') as clear_file:
            clear_file.write("")


def image_processing():
    while True:
        global image_output
        global annotated_image
        image_output, annotated_image = analyze_image(stream)
        print(image_output)


circumspect = threading.Thread(target=image_processing)
circumspect.start()

live_data = threading.Thread(target=get_weather_data)
live_data.start()

output_image = threading.Thread(target=listen, args=(model,))
output_image.start()


while True:
    try:
        live_info = f"Current time is {get_time()}\n{weather}"
        with open(f"{folder_location}/database/input.txt", 'r') as recognized:
            recognized_text = recognized.read()
        with open("outputs.txt", 'w') as text_outputs:
            text_outputs.write(
                f"{live_info}\n{image_output}\n{recognized_text}")
            
        ask_crystal(recognized_text, find_user("recording.wav"))

        cv2.imshow("Annotated Output", annotated_image)
        cv2.waitKey(1)
    except Exception as error:
        print(error)
