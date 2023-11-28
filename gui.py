import streamlit as st
import bot
from streamlit_extras.app_logo import add_logo 


def intro():
    import streamlit as st
    st.title("Studious")

    st.write("# Welcome to Studious! 👋")
    st.sidebar.success("Select a chat option above.")

    st.markdown(
        """
        Studious is an open-source study companion built by a high school student as a project.
        It is in a very crude state and any and all comments are welcomed 😄

        **👈 Select a chat from the dropdown on the left
    """
    )

def Questions():
    import streamlit as st
    from streamlit_extras.colored_header import colored_header
    from bot import Questions
    from streamlit_option_menu import option_menu
    
    colored_header(
        label="Question BOT 🤖",
        description="Meet Shelly, your cosmic companion in the quest for knowledge! This intergalactic genius is more than just ones and zeros – it's the digital oracle that deciphers your queries with a sprinkle of stardust and a dash of wit",
        color_name="orange-70",
    )
    app_id = st.sidebar.text_input("App ID")

    selected = option_menu(None, ["Maths", "Chemistry", "Physics", 'Misc'], 
    icons=['calculator', 'virus', "magnet", 'three-dots'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
    selected

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    prompt = st.chat_input()
    if prompt!=None:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        c=Questions(prompt,app_id)
        if selected == "Maths":
            response = c.Maths(prompt)
        elif selected == "Chemistry":
            response = c.Chemistry(prompt)
        elif selected == "Physics":
            response = c.Physics(prompt)
        elif selected == "Misc":
            response = c.Misc(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def cbot():
    import streamlit as st
    from streamlit_extras.colored_header import colored_header
    from bot import ogchat

    colored_header(
        label="Chat BOT 🤖",
        description="Meet Shelly, your virtual confidante with a side of sass! Imagine if your favorite comedian and a computer had a love child – that's Shelly in a nutshell.",
        color_name="orange-70",
    )

    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    que = st.chat_input()
    if que != None:
        with st.chat_message("user"):
            st.markdown(que)
        st.session_state.messages.append({"role": "user", "content": que})
        response = ogchat.chat(que)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})



page_names_to_funcs = {
    "—": intro,
    "Chat Bot": cbot,
    "Question Bot": Questions,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()