import ollama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ollama import chat


def main():
    # ollama.create(model='chatbot', from_='llama3.2', system='KEEP ALL YOUR ANSWERS 1 SENTENCE MAX.')

    oxy = True

    model = AutoModelForSeq2SeqLM.from_pretrained('models/tm1')
    tokenizer = AutoTokenizer.from_pretrained('models/tm1')
    reverse_model = AutoModelForSeq2SeqLM.from_pretrained('models/rtm1')
    reverse_tokenizer = AutoTokenizer.from_pretrained('models/rtm1')

    messages = []

    while oxy:
        #get user input
        user_input = input(">> ")
        #exit the program
        if user_input == "exit()":
            break

        #decode user input from techur to english
        inputs = tokenizer(user_input, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        #add user input to the messages array
        messages.append({'role': 'user', 'content': tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]})

        #get response
        response = chat(
            model='chatbot',
            messages=messages
        )

        #final response is going to have both the translated and non translated version cause some stuff just gets lost in translation
        final_response = response.message.content

        #encode bot response from english to techur
        inputs = reverse_tokenizer(response.message.content, return_tensors='pt').to(reverse_model.device)
        outputs = reverse_model.generate(**inputs, max_new_tokens=128)
        response.message.content = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        final_response = response.message.content + "\n" + final_response

        #print the final response
        print(final_response)

        #add bot response to messages
        messages.append({'role': 'assistant', 'content': response.message.content})


if __name__ == "__main__":
    main()