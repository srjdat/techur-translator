from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ask user what translation direction they want to go
model_type = int(input("Which Type of Translation? (Enter Option Number) \n1. Techur to English \n2. English to Techur\n>>> "))

match model_type:
    case 1:
        # import local model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained('tm1')
        tokenizer = AutoTokenizer.from_pretrained('tm1')

        # ty @xyve7 for creating this while loop
        # get user input
        user_input = ""
        while user_input != "exit":
            user_input = input("enter your techursentence: ")
            # tokenize input and output is what the model generates
            inputs = tokenizer(user_input, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    case 2:
        # import local model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained('rtm1')
        tokenizer = AutoTokenizer.from_pretrained('rtm1')

        # ty @xyve7 for creating this while loop
        # get user input
        user_input = ""
        while user_input != "exit":
            user_input = input("enter your sentence: ")
            # tokenize input and output is what the model generates
            inputs = tokenizer(user_input, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    case _:
        print("Please Enter A Valid Input")