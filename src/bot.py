import logging
import os
import discord
import ollama
from discord.ext import commands
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ollama import chat

ollama.create(model='chatbot', from_='llama3.2', system='ALL YOUR REPSONSES SHOULD BE SIMPLE AND 1 SENTENCE MAX.')

load_dotenv()
token = os.getenv('DISCORD_TOKEN')

handler = logging.FileHandler(filename='discord.log', mode='w', encoding='utf-8')
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = commands.Bot(command_prefix='!', intents=intents)

# load in model and tokenizer for techur to english
model = AutoModelForSeq2SeqLM.from_pretrained('models/tm1')
tokenizer = AutoTokenizer.from_pretrained('models/tm1')

#load in model and tokenizer for english to techur
reverse_model = AutoModelForSeq2SeqLM.from_pretrained('models/rtm1')
reverse_tokenizer = AutoTokenizer.from_pretrained('models/rtm1')

## COMMANDS ##
#!hello command
@client.command()
async def hello(ctx):
    #print("!hello command")
    await ctx.send(f"hello {ctx.author.mention}")

#!telore command
@client.command()
async def telore(ctx):
    #print("!telore command")
    await ctx.send(f"Please visit <#1503238638170013798>")

#!bye command
@client.command()
async def bye(ctx):
    await ctx.send(f"bye bye techurmogging")

#!image command
@client.command()
async def image(ctx):
    await ctx.send(file=discord.File('src/IMG_0853.png'))

#!techur
@client.command()
async def techur(ctx, member: discord.Member=None):
    if member == None:
        member = ctx.author
    # send the message
    await ctx.send(f"{member.mention} ooo techur")

#!translate command
#translates english sentences into techur sentences
@client.command()
async def translate(ctx, *, sentence):
    print(type(sentence))
    inputs = tokenizer(sentence, return_tensors='pt').to(reverse_model.device)
    outputs = reverse_model.generate(**inputs, max_new_tokens=128)
    await ctx.send(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

#only reason im doing this is because i have to call the command each time
messages = []
#!chatbot command
@client.command()
async def chatbot(ctx, *, sentence):
    #translate user input to english
    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)

    #add user input into messages when it's in english format
    messages.append({'role': 'user', 'content': tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]})

    response = chat(
        model='chatbot',
        messages=messages
    )

    #add to messages while it's still in english format
    messages.append({'role': 'assistant', 'content': response.message.content})

    # have a variable that contains both translated and non translated version
    #only reason im doing this is because sometimes the response is getting lost in translation
    discord_response = response.message.content

    #translate english output to techur output
    inputs = reverse_tokenizer(response.message.content, return_tensors='pt').to(reverse_model.device)
    outputs = reverse_model.generate(**inputs, max_new_tokens=128)
    response.message.content = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    #adds translated part to beginning of the discord response
    discord_response = (response.message.content) + "\n\n\n(" + discord_response + ")"

    #send discord response
    await ctx.send(discord_response)


## EVENTS ##
#when the bot starts up
@client.event
async def on_ready():
    print("bot is ready to be used")

# when someone sends a message
@client.event
async def on_message(message):
    # this is for if someone pings the bot
    # dont reply if it's ourselves
    if message.author == client.user:
        return
    # if someone else pings us
    if client.user.mentioned_in(message):
        await message.reply("hello techur")

    # detect if message has techur
    if "techur" in message.content:
        # have to check if it's a command or not
        if not "!" in message.content:
            inputs = tokenizer(message.content, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            await message.reply(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

    await client.process_commands(message)

client.run(token, log_handler=handler, log_level=logging.DEBUG)
