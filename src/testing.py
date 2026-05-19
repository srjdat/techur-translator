from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import local model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('tm1')
tokenizer = AutoTokenizer.from_pretrained('tm1')

#get user input 
user_input = "" 

while user_input != "exit":
    user_input = input("enter your techursentence: ")
    #tokenize input and output is what the model generates 
    inputs = tokenizer(user_input, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
