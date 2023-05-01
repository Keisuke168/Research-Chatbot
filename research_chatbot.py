import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog
import os
import openai
import chromadb
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

class ChatGptGUI:
    def __init__(self, api_key):
        master = tk.Tk()
        master.title("ChatGpt GUI")

        # フォント設定
        font_family = "Helvetica"
        font_size = 20
        japanese_font = tkFont.Font(family=font_family, size=font_size)

        # 入力欄の作成
        self.input_label = tk.Label(master, text="Input:")
        self.input_label.grid(row=0, column=0)
        self.input_entry = tk.Entry(master, width=100, font=japanese_font)
        self.input_entry.grid(row=0, column=1)

        # 出力欄の作成
        self.output_label = tk.Label(master, text="Output:")
        self.output_label.grid(row=1, column=0)
        self.output_text = tk.Text(master, height=20, width=100, font=japanese_font)
        self.output_text.grid(row=1, column=1)

        # 送信ボタンの作成
        self.send_button = tk.Button(master, text="Send", command=self.chatgpt_reply)
        self.send_button.grid(row=2, columnspan=2)

        # ファイル選択用のボタンとエントリー
        self.path_label = tk.Label(master, text="ファイル:")
        self.path_label.grid(row=3, column=0, padx=5, pady=5)

        self.path_entry = tk.Entry(master)
        self.path_entry.grid(row=3, column=1, padx=5, pady=5)

        self.browse_button = tk.Button(master, text="参照", command=self._set_paper)
        self.browse_button.grid(row=3, column=2, padx=5, pady=5)

        self.master = master

        # GPT setting
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        self.pdf_qa = None
       

    def _set_paper(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file_path)
            self._load_paper()

    def _load_paper(self):
        loader = PyPDFLoader(self.path_entry.get())
        pages = loader.load_and_split()

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".") 
        vectorstore.persist()

        self.pdf_qa = ConversationalRetrievalChain.from_llm(self.llm, vectorstore.as_retriever(), return_source_documents=True)
        self.output_text.insert(tk.END, "論文の読み込みが完了しました\n")
    # chatgptによる応答関数
    def chatgpt_reply(self):
        if not self.pdf_qa:
            self.output_text.insert(tk.END, "論文を選択してください\n---\n")
            self.input_entry.delete(0, tk.END)
            return
        query = self.input_entry.get()
        chat_history = []
        result = self.pdf_qa({"question": query, "chat_history": chat_history})
        output = result["answer"]
        self.output_text.insert(tk.END, 'Question:\n' + query + '\n---\n\n')
        self.output_text.insert(tk.END, 'Answer:\n' + output + '\n---\n\n')
        self.input_entry.delete(0, tk.END)
        

    def __call__(self):
        self.master.mainloop()

if __name__ == "__main__":
    chatgpt_gui = ChatGptGUI(api_key='YOUR_API_KEY')
    chatgpt_gui()