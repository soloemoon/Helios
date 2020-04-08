from tkinter import * 
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import pandas as pd
import numpy as np

text_color = 'purple'
button_color = 'black'

# Highlight Buttons
class HB(ttk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self,master=master,**kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    def on_enter(self, e):
        self['background'] = self['activebackground']
    def on_leave(self, e):
        self['background'] = self.defaultBackground


class Helios:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        
        # Buttons
        self.equities = HB(self.frame, text = 'Equities', width = 25, command = self.equity_page, fg = text_color, bg= button_color).pack()
        #self.bonds = tk.Button(self.frame, text = 'Bonds', width = 25, command = self.new_window).pack()
        #self.options = tk.Button(self.frame, text = 'Options', width = 25, command = self.new_window).pack()
        #self.options = tk.Button(self.frame, text = 'Options', width = 25, command = self.new_window).pack()
       
        self.frame.pack()
        
    # Page Navigation
    def equity_page(self):
        self.master.withdraw()
        self.newWindow = tk.Toplevel(self.master)
        self.app = Equities(self.newWindow)
        
    def close_app(self):
        self.master.destroy()
        


    

class Equities:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        
        self.import_text = Label(self,text ="Excel Import").pack()
        
        self.import_securities = HB(self.frame, text = 'Excel Import', width = 25, command = self.select_file, fg = text_color, bg= button_color).pack()
        
      
        self.home = HB(self.frame, text = 'Return Home', width = 25, command = self.return_home, fg = text_color, bg= button_color).pack()
        self.quit = HB(self.frame, text = 'Close Application', width = 25, command = self.close_app, fg = text_color, bg= button_color).pack()
        self.frame.pack()
        
    def return_home(self):
        self.master.show()
        self.newWindow = tk.Toplevel(self.master)
        self.app = Helios(self.newWindow)
        
    def close_app(self):
        self.master.destroy()
        
    def select_file(self):
        
        self.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = [("Excel File","*.xlsx"),("Macro-Enabled Excel","*.xlsm*"),("Excel Binary File","*.xlsb*")])
        excel_file = pd.read_excel(self.filename)
        
        return(excel_file)
        
        
     

def main(): 
    root = tk.Tk()
    root.geometry("500x500")
    root.title('Helios Analytics')
    
 # Create Toolbar
    menubar= tk.Menu(root)
    
    filemenu = tk.Menu(menubar,tearoff=0)
    filemenu.add_command(label="New", command=root.destroy)
    filemenu.add_command(label="Open", command=root.destroy)
    menubar.add_cascade(label="File", menu=filemenu)

    equitymenu = tk.Menu(menubar,tearoff=0)
    equitymenu.add_command(label="Undo", command=root.destroy)
    menubar.add_cascade(label="Equities", menu=equitymenu)

    bondmenu = tk.Menu(menubar,tearoff=0)
    bondmenu.add_command(label="Help",command=root.destroy)
    menubar.add_cascade(label="Bonds",menu=bondmenu)

    optionmenu = tk.Menu(menubar,tearoff=0)
    optionmenu.add_command(label="Help",command=root.destroy)
    menubar.add_cascade(label="Bonds",menu=optionmenu)
    
    helpmenu = tk.Menu(menubar,tearoff=0)
    helpmenu.add_command(label="Help",command=root.destroy)
    menubar.add_cascade(label="Help",menu=helpmenu)
    
    root.config(menu=menubar)


    app = Helios(root)
    root.mainloop()

if __name__ == '__main__':
    main()