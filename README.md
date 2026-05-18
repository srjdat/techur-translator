# Techur Translator
Few of my friends and I created a fictional being named Techur and his numerous children. We substitute these words in place of other words in sentences when we talk to each other, which makes our conversations slightly confusing. This program solves that issue by translating the "Techur sentences" into their English counterparts. 

# How to Run on Command Line
### 1. Create a Virtual Environment
Run `python3 -m venv venv` and activate
Activate with `source venv/bin/activate`
### 2. Install Dependencies
Install dependencies using `pip install -r requirements.txt`
### 3. Train the Model 
Run `python3 src/main.py` to train the model
### 4. Run the Inference Script
Run `python3 src/testing.py` and enter a "techur sentence" to get an output

# Discord Bot
Get a discord developer token and add it to the `.env` file.  
Invite bot to your desired server and give it permissions to send messages. 
## How to Run
1. Create Virtual Environment like shown before
2. Install all dependencies
3. Run `python3 src/bot.py`
## What the bot can do
"!hello": Bot reply with "Hello @{username}"  
Pinging it will make the bot reply with "hello techur"  
If you send a message that contains the word "techur" it will run the inference script and reply to you with the output.

# This project uses the flan-t5 model.
