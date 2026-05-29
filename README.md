# Techur Translator
Few of my friends and I created a fictional being named Techur and his numerous children. We substitute these words in place of other words in sentences when we talk to each other, which makes our conversations slightly confusing. This program solves that issue by translating the "Techur sentences" into their English counterparts. I also decided to add the ability for this program to translate from English to "Techur" sentences for fun.

# How to Run on Command Line
### 1. Create a Virtual Environment
Run `python3 -m venv venv` and activate
Activate with `source venv/bin/activate`
### 2. Install Dependencies
Install dependencies using `pip install -r requirements.txt`
### 3. Train the Model 
Run `python3 src/main.py` to train the model
### 4. Run the Inference Script
Run `python3 src/testing.py` and select the translation option you want  
The program will run until you enter `exit`
### 5. Run the Chatbot
Before running this file for the first time uncomment line 7 on `chatbot.py`  
Once you run you can chat with the chatbot in either "Techur sentences" or plain English  
To exit the chatbot enter `exit()`

# Discord Bot
Get a discord developer token and add it to the `.env` file.  
Invite bot to your desired server and give it permissions to send messages. 
## How to Run
1. Create Virtual Environment like shown before
2. Install all dependencies
3. Train the model like shown before
4. Create a `.env` file and put in your discord token in a variable named `DISCORD_TOKEN`     
5. Run `python3 src/bot.py`
## What the Bot Can Do

### Commands
`!hello` - Bot will say "hello" back to you   
`!image` - Bot will send a silly image I drew in DSA   
`!translate {sentence}` - Bot will turn an English sentence into its Techur counterpart   
`!techur {@user}` - Bot will mention the user and send "ooo techur"       
`!chatbot {sentence}` - Bot will use the chatbot feature, you can talk in Techur sentence or plain English  
`!chatbot {sentence}` - Uncomment line 10 in `bot.py` unless you have ran `chatbot.py` with line 7 uncommented   
### Features
If you send a message that includes the word "techur", it will translate and send it in the channel.

# This project uses the flan-t5 model.
### Additional Information
- I trained this on my laptop with Integrated AMD Graphics. I'm not sure how to turn on CUDA or ROCm for discrete GPUs, you will most likely have to figure that out on your own.
- Thank you to @xyve7 (oxy) for training the later models for me.
