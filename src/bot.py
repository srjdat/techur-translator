import logging
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
token = os.getenv('DISCORD_TOKEN')

handler = logging.FileHandler(filename='../discord.log', mode='w', encoding='utf-8')
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = commands.Bot(command_prefix='!', intents=intents)

model = AutoModelForSeq2SeqLM.from_pretrained('tm1')
tokenizer = AutoTokenizer.from_pretrained('tm1')


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
        inputs = tokenizer(message.content, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        await message.reply(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    await client.process_commands(message)

client.run(token, log_handler=handler, log_level=logging.DEBUG)