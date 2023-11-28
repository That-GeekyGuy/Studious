class Questions:
    def __init__(self,question, app_id):
        import random
        import json
        import wolframalpha
        import torch
        import wikipedia
        from model import NeuralNet
        from nltk_utils import bag_of_words, tokenize
        import streamlit as st
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        client = wolframalpha.Client(app_id)

        try:
            res = client.query(question)
            answer = next(res.results).text

            self.a=answer
        except Exception as e:
            self.a=f"An error occurred: {e}"
        
        

    def Maths(self,question):
        import wikipedia
        answer = self.a
        if answer == "An error occurred: ":
            words_to_omit = ["is", "the", "what" , "?"]
            for word in words_to_omit:
                question = question.replace(word, "")
            result = wikipedia.summary(question +" (maths)", sentences=2)
            
            return result
        else:
            return answer
    def Chemistry(self,question):
        import wikipedia
        answer = self.a
        if answer == "An error occurred: ":
            words_to_omit = ["is", "the", "what" , "?"]
            for word in words_to_omit:
                question = question.replace(word, "")
            result = wikipedia.summary(question +" (chemistry)", sentences=2)
            
            return result
        else:
            return answer
    def Physics(self,question):
        import wikipedia
        answer = self.a
        if answer == "An error occurred: ":
            words_to_omit = ["is", "the", "what" , "?"]
            for word in words_to_omit:
                question = question.replace(word, "")
            result = wikipedia.summary(question +" (physics)", sentences=2)
            
            return result
        else:
            return answer
    def Misc(self,question):
        import wikipedia
        answer = self.a
        if answer == "An error occurred: ":
            words_to_omit = ["is", "the", "what" , "?"]
            for word in words_to_omit:
                question = question.replace(word, "")
            result = wikipedia.summary(question +" (maths)", sentences=2)
            return result
        else:
            return answer
class ogchat:
    def chat(question):
        import random
        import json
        import wolframalpha
        import torch
        from model import NeuralNet
        from nltk_utils import bag_of_words, tokenize
        import streamlit as st
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('intents.json', 'r') as json_data:
            intents = json.load(json_data)
        if question == None:
            pass
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

        question = tokenize(question)
        X = bag_of_words(question, all_words)
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
                    return f"{random.choice(intent['responses'])}"
        else:
            return "I do not understand..."

    #if __name__ == "__main__":
    #    main()