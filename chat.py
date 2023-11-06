import random
import json
import wolframalpha
import torch
import wikipedia
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import tkinter as tk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def get_wolframalpha_answer(question, app_id):

    client = wolframalpha.Client(app_id)

    try:
        res = client.query(question)
        answer = next(res.results).text

        return answer
    except Exception as e:
        return f"An error occurred: {e}"

app_id = 'RUTL64-6QQXG335RP'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
def main_menu():
    print("1. Question")
    print("2. Chat")
    print("3. Quit")

def omit_words(question, words_to_omit):
    for word in words_to_omit:
        question = question.replace(word, "")
    return question

words_to_omit = ["is", "the", "what" , "?"]
def option1():
    while True:
        print("Subject :","1.Maths","2.Chemistry","3.Physics","4.Miscellaneous","5.Quit",sep="\n")
        o=input("Your Choice")
        if o == "quit" or o == "Quit" or o == "5":
            break
        if o == "Maths" or o == "maths" or o =="1":
            sentence = input("You: ")
            answer = get_wolframalpha_answer(sentence, app_id)
            if answer == "An error occurred: ":
                sentence = omit_words(sentence, words_to_omit)
                result = wikipedia.summary(sentence +" (maths)", sentences=2)
               
                print(result)
            else:
                
                print(answer)
        elif o == "Chemistry" or o == "chemistry" or o == "2":
            sentence = input("You: ")
            answer = get_wolframalpha_answer(sentence, app_id)
            if answer == "An error occurred: ":
                sentence = omit_words(sentence, words_to_omit)
                result = wikipedia.summary(sentence + " (chemistry)", sentences=2)
                print(result)
                
            else:
               
                print(answer)
        elif o== "Physics" or o== "physics" or o== "3":
            sentence = input("You: ")
            answer = get_wolframalpha_answer(sentence, app_id)
            if answer == "An error occurred: ":
                sentence = omit_words(sentence, words_to_omit)
                result = wikipedia.summary(sentence + " (physics)", sentences=2)
                print(result)
                
            else:
                
                print(answer)
        elif o == "Misc"or "misc" or "Miscellaneous" or o == "miscellaneous" or o == "5":
            try:
                sentence = input("You: ")
                answer = get_wolframalpha_answer(sentence, app_id)
                if answer == "An error occurred: ":
                    sentence = omit_words(sentence, words_to_omit)
                    result = wikipedia.summary(sentence, sentences=2)
                    print(result)
                    
                else:
                    
                    print(answer)
            except:
                print("Sorry can't find the answer ಥ_ಥ'")
def option2():
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")

def main():
    while True:
        main_menu()
        choice = input("Enter your choice: ")

        if choice == '1' or choice == 'question' or choice == 'Question':
            option1()
        elif choice == '2' or choice == 'Chat' or choice == 'chat':
            option2()
        elif choice == 'Quit' or choice=='quit' or choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

