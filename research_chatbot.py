import os
import tkinter as tk
import customtkinter

import openai
import chromadb
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

FONT_TYPE = "meiryo"

class ReadFileFrame(customtkinter.CTkFrame):
    def __init__(self, llm, set_chain, *args, header_name="ReadFileFrame", **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fonts = (FONT_TYPE, 15)
        self.header_name = header_name

        # LLMの設定
        self.llm = llm
        self.set_chain = set_chain
        self.qa_chain = None

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(0, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        # フレームのラベルを表示
        self.label = customtkinter.CTkLabel(self, text=self.header_name, font=(FONT_TYPE, 11))
        self.label.grid(row=0, column=0, padx=20, sticky="w")

        # ファイルパスを指定するテキストボックス。これだけ拡大したときに、幅が広がるように設定する。
        self.textbox = customtkinter.CTkEntry(master=self, placeholder_text="PDFファイルを読み込む", width=120, font=self.fonts)
        self.textbox.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")

        # ファイル選択ボタン
        self.button_select = customtkinter.CTkButton(master=self, 
            command=self.button_select_callback, text="ファイル選択", font=self.fonts)
        self.button_select.grid(row=1, column=1, padx=10, pady=(0,10))
        

    def button_select_callback(self):
        """
        選択ボタンが押されたときのコールバック。ファイル選択ダイアログを表示する
        """
        # エクスプローラーを表示してファイルを選択する
        file_name = ReadFileFrame.file_read()

        if file_name is not None:
            # ファイルパスをテキストボックスに記入
            self.textbox.delete(0, tk.END)
            self.textbox.insert(0, file_name)
            self._load_paper(file_name)
            self.set_chain(self.qa_chain)
            print('DONE LOADING')
            
    def _load_paper(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".") 
        vectorstore.persist()

        self.qa_chain = ConversationalRetrievalChain.from_llm(self.llm, vectorstore.as_retriever(), return_source_documents=True)

    def file_read():
        """
        ファイル選択ダイアログを表示する
        """
        file_path = tk.filedialog.askopenfilename(filetypes=[("pdfファイル","*.pdf")])

        if len(file_path) != 0:
            return file_path
        else:
            # ファイル選択がキャンセルされた場合
            return None
class ConversationFrame(customtkinter.CTkFrame):
    def __init__(self, *args, header_name='Conversation', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fonts = (FONT_TYPE, 15)
        self.header_name = header_name

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(0, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        # # フレームのラベルを表示
        self.label = customtkinter.CTkLabel(self, text=self.header_name, font=(FONT_TYPE, 11))
        self.label.grid(row=0, column=0, padx=20, sticky="w")


        # create scrollable textbox
        self.tk_textbox = customtkinter.CTkTextbox(self, height=500, activate_scrollbars=True, font=(FONT_TYPE, 15))
        self.tk_textbox.grid(row=1, column=0, sticky="nsew")

    def add_response(self, question, answer):
        self.tk_textbox.insert(tk.END, 'Question:\n' + question + '\n\nAnswer:\n' + answer  + "\n\n====================\n\n")
class InputFrame(customtkinter.CTkFrame):
    def __init__(self, update_conversation_fn, *args, header_name='Input', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fonts = (FONT_TYPE, 15)
        self.header_name = header_name

        # メンバ変数の設定
        self.qa_chain = None
        self.update_conversation_fn = update_conversation_fn

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(0, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        # # フレームのラベルを表示
        self.label = customtkinter.CTkLabel(self, text=self.header_name, font=(FONT_TYPE, 11))
        self.label.grid(row=0, column=0, padx=20, sticky="w")

        # テキストボックス
        self.textbox = customtkinter.CTkEntry(master=self, placeholder_text="質問を入力", width=120, font=self.fonts)
        self.textbox.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")

        # ボタン
        self.button = customtkinter.CTkButton(master=self, command=self.button_callback, text="Send", font=self.fonts)
        self.button.grid(row=1, column=1, padx=10, pady=(0,10))

    
    def button_callback(self):
        if self.qa_chain is None:
            self.textbox.delete(0, tk.END)
            return
        # テキストボックスから質問を取得する
        question = self.textbox.get()

        # 質問を入力して、回答を取得する
        result = self.qa_chain({"question": question, "chat_history": []})
        answer = result["answer"]

        # ConversationFrameの更新
        self.update_conversation_fn(question, answer)

        # テキストボックスを空にする
        self.textbox.delete(0, tk.END)
        
    def set_chain(self, qa_chain):
        self.qa_chain = qa_chain

class App(customtkinter.CTk):

    def __init__(self, api_key):
        super().__init__()

        # メンバー変数の設定
        self.fonts = (FONT_TYPE, 15)
        self.csv_filepath = None
        
        # OpenAIのAPIキーを設定する
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # CustomTkinter のフォームデザイン設定
        customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        # フォームサイズ設定
        self.geometry("1200x800")
        self.title("Reasearch Chatbot")

        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(3, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        # Conversation Frameの設定
        self.conversation_frame = ConversationFrame(master=self, header_name="Conversation")
        self.conversation_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        
        # Input Frameの設定
        self.input_frame = InputFrame(master=self, update_conversation_fn=self.conversation_frame.add_response, header_name="Input")
        self.input_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        # ReadFile Frameの設定
        # stickyは拡大したときに広がる方向のこと。nsew で4方角で指定する。
        self.read_file_frame = ReadFileFrame(master=self, llm=self.llm, set_chain=self.input_frame.set_chain, header_name="Select Paper")
        self.read_file_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")


if __name__ == "__main__":
    app = App(api_key='YOUR_API_KEY')
    app.mainloop()