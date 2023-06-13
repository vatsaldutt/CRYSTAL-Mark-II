from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import random
import os

with open("pwd.txt", 'r') as pwd:
    folder_location = pwd.read()

first_time_weather = True

op = webdriver.ChromeOptions()
op.add_argument('headless')
op.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)

def web_scrapper(web_query):
    web_query = web_query.replace(' ', '%20').replace('=', '%3D').replace('+', "%2B")
    url = "https://www.google.com/search?q="+web_query
    driver.get(url)
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'html.parser')
    part_of_speeches = ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition', 'conjunction', 'interjection', 'exclamation', 'numeral', 'article', 'determiner']

    list1 = []

    for i in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        for j in i.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            list1.append(j.text)
    
    try:
        return (soup.find('div', class_='BNeawe iBp4i AP7Wnd').text)
    except:
        pass
    try:
        element = driver.find_element_by_class_name("IZ6rdc")
        return (element.text)
    except:
        pass

    try:
        element = driver.find_element_by_class_name("Z0LcW CfV8xf")
        return (element.text)
    except:
        pass

    try:
        element = driver.find_element_by_class_name("ayqGOc kno-fb-ctx KBXm4e")
        return (element.text)
    except:
        pass

    if list1[0].split()[0] in part_of_speeches:
        if list1[0].split()[0][0] == "a":
            return 'As an '+list1[0].split()[0]+' it means '+list1[1]
        
        else:
            return 'As a '+list1[0].split()[0]+' it means '+list1[1]
    
    for text in list1:
        list_text = text.split()
        if len(list_text) != 0:
            if list_text[-1] == 'Wikipedia':
                return 'According to the Wikipedia, '+str('/'.join(text.split()[0:-1]).replace('/', ' '))
    
    answer_types = ['You would say that ', 'That would be ', "That's "]
    for i in soup.find_all('div'):
        for j in i.find_all('div'):
            for k in j.find_all('div'):
                for m in k.find_all('div'):
                    if 'MUxGbd u31kKd gsrt lyLwlc' in str(m):
                        translation = str(m.text).replace('Translation', '').replace('Translate', '')
    try:
        return random.choice(answer_types) + translation
    
    except:
        pass


def sadly_using_this(web_query):
    web_query = web_query.replace(' ', '%20').replace('=', '%3D').replace('+', "%2B")
    page_url = 'https://www.google.com/search?q=' + web_query
    source = requests.get(page_url).text
    soup = BeautifulSoup(source, 'html.parser')
    part_of_speeches = ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition', 'conjunction', 'interjection', 'exclamation', 'numeral', 'article', 'determiner']

    list1 = []

    for i in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        for j in i.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            list1.append(j.text)
    
    try:
        return soup.find('div', class_='BNeawe iBp4i AP7Wnd').text
    except:
        pass

    if list1[0].split()[0] in part_of_speeches:
        if list1[0].split()[0][0] == "a":
            return 'As an '+list1[0].split()[0]+' it means '+list1[1]
        
        else:
            return 'As a '+list1[0].split()[0]+' it means '+list1[1]
    
    answer_types = ['You would say that ', 'That would be ', "That's "]
    for i in soup.find_all('div'):
        for j in i.find_all('div'):
            for k in j.find_all('div'):
                for m in k.find_all('div'):
                    if 'MUxGbd u31kKd gsrt lyLwlc' in str(m):
                        translation = str(m.text).replace('Translation', '').replace('Translate', '')
    try:
        return random.choice(answer_types) + translation
    
    except:
        pass
    
    try:
        driver.get(page_url)
        algebra_result = driver.find_elements_by_class_name('LPBLxf')
        return algebra_result[-1].text.split('\n')[-1]
    except:
        pass

    for text in list1:
        list_text = text.split()
        if len(list_text) != 0:
            if list_text[-1] == 'Wikipedia':
                return 'According to the Wikipedia, '+str('/'.join(text.split()[0:-1]).replace('/', ' '))
    urls = []
    for a in soup.find_all('a'):
        if a.get('href')[0:15] == '/url?q=https://':
            url = a.get('href').replace('/url?q=https://', '')
            urls.append(url[0: url.index('&sa')])
    
    for u in range(len(urls)):
        urls[u] = 'https://'+urls[u]


            
    if urls[0].split('/')[2] == "www.youtube.com":
        with open(f'{folder_location}data/youtube_query.txt', 'w') as youtube_query:
            youtube_query.write(web_query)
            youtube_query.close()
        os.system('python3f {folder_location}youtube.py')
        return "Here are some results from the web..."
    
    if "Duration" not in list1[0]:
        if len(list1[0].split()) > 10:
            try:
                return list1[0].split('...')[0].split("·")[1]
            
            except:
                return list1[0].split('...')[0]

    url_source = requests.get(urls[0]).text
    soup = BeautifulSoup(url_source, 'html.parser')
    url_text = []
    for i in soup.find_all("p"):
        url_text.append(i.text)

    paracount = 0
    for j in url_text:
        if len(j.split()) < 11:
            pass
        
        elif paracount == 1:
            return "According to the website "+urls[0].split('/')[2]+", "+j.split("\r\n\r")[0]

        else:
            paracount += 1
