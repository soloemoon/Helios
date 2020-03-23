import tkinter as tk
from tkinter import ttk

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
     
       
        
    def close_app(self):
        self.master.destroy()

def main(): 
    root = tk.Tk()
    root.title('Helios Analytics')
    app = Helios(root)
    root.mainloop()

if __name__ == '__main__':
    main()