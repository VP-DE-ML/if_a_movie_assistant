
from tkinter import *
from src.movie_assistant_core import MovieAssistantCore

movie_assistant_core = MovieAssistantCore()

def create_GUI():
    GUI = Tk()
    GUI.title("Movie Assistant")
    GUI.geometry("400x500")
    GUI.resizable(width=False, height=False)

    # Menus
    menus_menu = Menu(GUI)
    file_menu = Menu(GUI, tearoff=False)
    help_menu = Menu(GUI, tearoff=False)

    # Submenus
    file_menu.add_command(label="New..")
    file_menu.add_command(label="Save As..")
    file_menu.add_command(label="Exit", command=exit_action)
    help_menu.add_command(label="About")

    # Add submenus to the menu
    menus_menu.add_cascade(label="File", menu=file_menu)
    menus_menu.add_cascade(label="Help", menu=help_menu)

    GUI.config(menu=menus_menu)
    return GUI

def send_output():
    user_input = input_window.get("1.0", 'end-1c').strip()
    input_window.delete("0.0", END)
    if user_input != '':
        chat_window.config(state=NORMAL)
        chat_window.insert(END, "You: " + user_input + '\n\n')
        chat_window.config(foreground="RoyalBlue4", font=("Verdana", 12))
        response = movie_assistant_core.chatbot_response(user_input)
        chat_window.insert(END, "Ans: " + response + '\n\n')
        chat_window.config(state=DISABLED)
        chat_window.yview(END)

def exit_action():
    exit()


GUI = create_GUI()

# Create conversation history window
chat_window = Text(GUI, bd=0, bg="white", height="8", width="50", font="Arial", )
# chat_window.pack(side=LEFT, expand=True, fill=BOTH)
chat_window.config(state=DISABLED)

# Add scrollbar to Chat window
scrollbar = Scrollbar(GUI, command=chat_window.yview, cursor="arrow")
# scrollbar.pack(side=RIGHT, fill=Y)
chat_window['yscrollcommand'] = scrollbar.set

# user input window
input_window = Text(GUI, bd=0, bg="white", width="29", height="5", font="Arial")
# input_window.pack(side="RIGHT", expand=True, fill=BOTH)

# Create Button to send message
send_button = Button(GUI, font=("Arial", 12), text="Send", width="12", height=5,
                     bd=0, bg="#0080ff", activebackground="#00bfff", foreground='#ffffff',
                     command=send_output)

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
chat_window.place(x=6, y=6, height=386, width=370)
input_window.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)

GUI.mainloop()
