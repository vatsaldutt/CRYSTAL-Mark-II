window.addEventListener('load', function () {
    var loadingScreen = document.getElementById('loading-screen');
    var mainContent = document.getElementById('main-content');

    var temperatureSlider = document.getElementById('temperature-slider');
    var temperatureValue = document.getElementById('temperature-value');

    var topPSlider = document.getElementById('top-p-slider');
    var topPValue = document.getElementById('top-p-value');

    var topKSlider = document.getElementById('top-k-slider');
    var topKValue = document.getElementById('top-k-value');

    var repeatPenaltySlider = document.getElementById('repeat-penalty-slider');
    var repeatPenaltyValue = document.getElementById('repeat-penalty-value');

    var maxTokensSlider = document.getElementById('max-tokens-slider');
    var maxTokensValue = document.getElementById('max-tokens-value');

    var maxHistorySlider = document.getElementById('max-history-slider');
    var maxHistoryValue = document.getElementById('max-history-value');


    temperatureSlider.addEventListener('input', function () {
        temperatureValue.textContent = temperatureSlider.value;
    });

    topPSlider.addEventListener('input', function () {
        topPValue.textContent = topPSlider.value;
    });

    topKSlider.addEventListener('input', function () {
        topKValue.textContent = topKSlider.value;
    });

    repeatPenaltySlider.addEventListener('input', function () {
        repeatPenaltyValue.textContent = repeatPenaltySlider.value;
    });

    maxTokensSlider.addEventListener('input', function () {
        maxTokensValue.textContent = maxTokensSlider.value;
    });

    maxHistorySlider.addEventListener('input', function () {
        maxHistoryValue.textContent = maxHistorySlider.value;
    });

    var centeredContent = document.getElementById('centered-content');
    var queryContainer = document.getElementById('query-container');
    var queryInput = document.getElementById('query-input');
    var sendButton = document.getElementById('send-button');
    var conversation = document.getElementById('conversation');
    var main = document.getElementById('main');

    sendButton.addEventListener('click', function () {
        var query = queryInput.value.trim();
        if (query !== '') {
            main.style.display = 'none';
            queryInput.value = "";
            queryInput.style.height = "20px";

            var sentText = document.createElement('p');
            sentText.className = "sent-text";
            sentText.classList.add("message")
            sentText.style.padding = "20px";
            sentText.textContent = query;
            sentText.readOnly = true;
            conversation.append(sentText);

            fetch('/write-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: 'query.txt', data: query }),
            })
                .then((response) => {
                    if (response.ok) {
                        console.log('File written successfully.');
                    } else {
                        console.error('Error writing file:', response.statusText);
                    }
                })
                .catch((error) => {
                    console.error('Error writing file:', error);
                });

            function LiveListen() {
                fetch('/reply.txt')
                    .then((res) => res.text())
                    .then((text) => {
                        if (text.trim() !== '') {
                            console.log(text);
                            updateReplyText(text); // Update the reply text element
                            setTimeout(LiveListen, 200);
                        } else {
                            console.log("DONE");
                            finalizeReplyText(); // Finalize the reply text element
                        }
                    })
                    .catch((e) => console.error(e));
            }

            function updateReplyText(text) {
                const replyTextElement = document.getElementById('current-reply');
                if (replyTextElement) {
                    replyTextElement.textContent = text; // Append new text to the existing content
                } else {
                    const newReplyTextElement = document.createElement('p');
                    newReplyTextElement.id = 'current-reply';
                    newReplyTextElement.className = 'reply-text';
                    newReplyTextElement.classList.add("message")
                    newReplyTextElement.style.padding = "20px";
                    newReplyTextElement.textContent = text;
                    // Append the new reply text element to the desired container in the HTML
                    conversation.appendChild(newReplyTextElement);
                }
            }

            function finalizeReplyText() {
                const replyTextElement = document.getElementById('current-reply');
                if (replyTextElement) {
                    replyTextElement.removeAttribute('id')
                }
            }

            function checkReplyFile() {
                fetch('/reply.txt')
                    .then((res) => res.text())
                    .then((text) => {
                        if (text.trim() !== '') {
                            LiveListen();
                        } else {
                            console.log("WAITING");
                            setTimeout(checkReplyFile, 200);
                        }
                    })
                    .catch((e) => console.error(e));
            }

            checkReplyFile();
            centeredContent.style.justifyContent = "start";
            conversation.scrollTop = conversation.scrollHeight;
        }
    });

    queryInput.addEventListener('input', function () {

        if (this.scrollHeight > 30) {
            this.style.height = "20px";
            this.style.height = (this.scrollHeight > 30 ? this.scrollHeight - 10 : this.scrollHeight) + 'px';
        }
    });

    queryInput.addEventListener('keydown', function (event) {
        if (event.shiftKey && event.key === "Enter") {
            event.preventDefault();
            queryInput.value += '\n';
            if (this.scrollHeight > 30) {
                this.style.height = (this.scrollHeight > 30 ? this.scrollHeight - 10 : this.scrollHeight) + 'px';
            }
        } else if (event.key === 'Enter') {
            event.preventDefault();
            sendButton.click();
        }
    });


    setTimeout(function () {
        loadingScreen.style.opacity = '0';

        setTimeout(function () {
            loadingScreen.style.display = 'none';
            mainContent.style.display = 'flex';
        }, 1000);
    }, 2000);
});
