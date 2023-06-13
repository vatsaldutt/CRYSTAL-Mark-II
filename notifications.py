from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from voice import speak, listen
from selenium import webdriver
import imaplib
import random
import openai
import email

first_time = True

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()

op = webdriver.ChromeOptions()
op.add_argument('headless')
op.add_experimental_option('excludeSwitches', ['enable-logging'])
op.add_argument("user-agent=User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)


def generate_response(me, other, query, contact):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f'''{contact}:{other}\nCrystal:{me}\n{contact}:{query}\nCrystal:''',
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[f" {contact}:", " Crystal:"]
        )
    print("You said:", query)
    print("I said:", response.choices[0].text)
    return response.choices[0].text


def check_notification():
    global first_time
    notify = False
    sender = []
    subject = []
    try:
        driver.get("https://web.whatsapp.com/")
        wait = WebDriverWait(driver, 600)
        if first_time == True:
            x_arg = '//div[@data-testid="qrcode"]'
            group_title = wait.until(EC.presence_of_element_located((
                By.XPATH, x_arg)))
            driver.save_screenshot('qr.png')
            first_time = False
        x_arg = '//span[@data-testid="filter"]'
        group_title = wait.until(EC.presence_of_element_located((
            By.XPATH, x_arg)))
        group_title.click()

        name = driver.find_elements_by_class_name('ggj6brxn')
        names = []
        chats = []

        n = True
        for i in name:
            if n == True:
                names.append(i.text)
                n = False
            elif n == False:
                chats.append(i.text)
                n = True

        # Recieve unread emails
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        (retcode, capabilities) = mail.login('vatdut8994@gmail.com','zqaektslzdozfcgn')
        mail.list()
        mail.select('inbox')

        n=0
        (retcode, messages) = mail.search(None, '(UNSEEN)')
        if retcode == 'OK':
            for num in messages[0].split() :
                print('Processing ')
                n=n+1
                typ, data = mail.fetch(num,'(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        original = email.message_from_string(response_part[1].decode('utf-8'))

                        sender.append(original['From'])
                        subject.append(original['Subject'])
                        typ, data = mail.store(num,'+FLAGS','\Seen')
        if n > 0:
            for i in range(n):
                speak(f'You have a new email from {sender[i]} with the subject: {subject[i]}')
        else:
            print("No new E-mails")


        for i in name:
            i.click()
            try:
                current_contact = names[0]
                current_chat = chats[0]
                old_chat = driver.find_elements_by_class_name('_1Gy50')
                for z in old_chat:
                    print(z.text)
                if current_chat != "":
                    speak("You have new messages from " + current_contact + " on WhatsApp. Would you like me to answer them?")
                    # dontlisten = True
                    shall_i = listen()
                    with open(f"{folder_location}data/recognition.txt", 'w') as recognition:
                        recognition.write('')
                    if 'yes' in shall_i.lower() or "yeah" in shall_i.lower():
                        inp_xpath = '//div[@class="p3_M1"]'
                        input_box = wait.until(EC.presence_of_element_located((
                            By.XPATH, inp_xpath)))
                        input_box.send_keys(generate_response(old_chat[-2].text, old_chat[-3].text, current_chat, current_contact) + Keys.ENTER)
                        del names[0]
                        del chats[0]
                    else:
                        speak(random.choice(["As you wish, Vatsal", "Okay Vatsal", "Okay Vatsal, I will leave them for you"]))
                    # dontlisten = False
                    notify = True
            except Exception as e:
                print(e)

        print(names)
        print(chats)
    except Exception as e:
        print(e)
    return notify