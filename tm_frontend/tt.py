from tkinter import *
from turtle import color
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandasgui import show
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import collections
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

def Imdb():
    path = r"IMDb_Data_final.csv"
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    df1=df[['Title', 'IMDb-Rating']]
    topmovies1= df.sort_values(by=['IMDb-Rating'], ascending=True).head(1000)
    try :
        df2=df1.loc[df['Title'] ==Movie_name.get(),'IMDb-Rating']
        top=float(df2)
        Rating.insert(0,top)
    except Exception :
        Rating.insert(0, "NO SUCH MOVIE FOUND :(")

def generate():
   
    path = r"IMDb_Data_final.csv"
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    df1=df[['Title', 'IMDb-Rating', "Category"]]
    topmovies1= df.sort_values(by=['IMDb-Rating'], ascending=True).head(1000)
    try :
        df2=df1.loc[df['Title'] ==Movie_name.get(),'IMDb-Rating']
        df3 = df1.loc[df['Title'] ==Movie_name.get(),'Category']
        cat = str(df3)
        top=float(df2)
        Top_thresh1= topmovies1.loc[topmovies1['IMDb-Rating'] > top]
        t1 = Top_thresh1.sort_values(by=['IMDb-Rating'], ascending=True).head(10)
        Top_thresh2= topmovies1.loc[topmovies1['IMDb-Rating'] < top]
        t2 = Top_thresh2.sort_values(by=['IMDb-Rating'], ascending=True).tail(10)
        final_1=t1.append(t2, ignore_index=True)
        final=final_1[['Title', 'IMDb-Rating']]
        MOVIES= final.sample(frac = 1)
        MOVIE=pd.DataFrame(MOVIES)
    
        a=MOVIE.iloc[0:1,0:2]
        Generate1.insert(0,a)
    
        b=MOVIE.iloc[1:2,0:2]
        Generate2.insert(0,b)
    
        c=MOVIE.iloc[2:3,0:2]
        Generate3.insert(0,c)
    
        d=MOVIE.iloc[3:4,0:2]
        Generate4.insert(0,d)
    
        e=MOVIE.iloc[4:5,0:2]
        Generate5.insert(0,e)
    
        f=MOVIE.iloc[5:6,0:2]
        Generate6.insert(0,f)
    
        g=MOVIE.iloc[7:8,0:2]
        Generate7.insert(0,g)
    
        h=MOVIE.iloc[8:9,0:2]
        Generate8.insert(0,h)
    
        i=MOVIE.iloc[9:10,0:2]
        Generate9.insert(0,i)
    
        j=MOVIE.iloc[10:11,0:2]
        Generate10.insert(0,j)
    
        k=MOVIE.iloc[11:12,0:2]
        Generate11.insert(0,k)
    
        l=MOVIE.iloc[12:13,0:2]
        Generate12.insert(0,l)
    
        m=MOVIE.iloc[13:14,0:2]
        Generate13.insert(0,m)
    
        n=MOVIE.iloc[14:15,0:2]
        Generate14.insert(0,n)
    
        q=MOVIE.iloc[15:16,0:2]
        Generate15.insert(0,q)
    
        r=MOVIE.iloc[17:18,0:2]
        Generate16.insert(0,r)
    
        s=MOVIE.iloc[18:19,0:2]
        Generate17.insert(0,s)
    
        t=MOVIE.iloc[18:19,0:2]
        Generate18.insert(0,t)
    
        u=MOVIE.iloc[19:20,0:2]
        Generate19.insert(0,u)
    except Exception :
        Rating.insert(0, "NO SUCH MOVIE FOUND :(")
   
def delete1():
    Movie_name.delete(0,END)
    Rating.delete(0,END)
    Generate1.delete(0,END)
    Generate2.delete(0,END)
    Generate3.delete(0,END)
    Generate4.delete(0,END)
    Generate5.delete(0,END)
    Generate6.delete(0,END)
    Generate7.delete(0,END)
    Generate8.delete(0,END)
    Generate9.delete(0,END)
    Generate10.delete(0,END)
    Generate11.delete(0,END)
    Generate12.delete(0,END)
    Generate13.delete(0,END)
    Generate14.delete(0,END)
    Generate15.delete(0,END)
    Generate16.delete(0,END)
    Generate17.delete(0,END)
    Generate18.delete(0,END)
    Generate19.delete(0,END)
   
   
master=Tk()
master.title("MOVIE RECOMMENDER SYSTEM")
font_format=('Helventica',22,'bold italic')

label1=Label(master,text="MOVIE RECOMMENDER SYSTEM",font=font_format)
label2=Label(master,text="SEARCH MOVIE NAME",font=font_format)
Movie_name=Entry(master)
imdb=Button(master,text="IMDb-RATING OF THE MOVIE ",padx=35,command=Imdb,pady=20,font=font_format)
Rating=Entry(master)
Gt=Button(master,text="RECOMMENDED MOVIES",command=generate,font=font_format,padx=35,pady=20)
instructions = Label(master, text="INSTRUCTIONS\n\n1. Enter the title of a \nMovie.\n2. Click on the buttons\nto either get the IMDb\nRatings or to get a list\nof recommended movies.", font=font_format)

Generate1=Entry(master)
Generate2=Entry(master)
Generate3=Entry(master)
Generate4=Entry(master)
Generate5=Entry(master)
Generate6=Entry(master)
Generate7=Entry(master)
Generate8=Entry(master)
Generate9=Entry(master)
Generate10=Entry(master)
Generate11=Entry(master)
Generate12=Entry(master)
Generate13=Entry(master)
Generate14=Entry(master)
Generate15=Entry(master)
Generate16=Entry(master)
Generate17=Entry(master)
Generate18=Entry(master)
Generate19=Entry(master)



cl=Button(master,text="Clear",padx=45,pady=20,command= delete1,font=font_format)

label1.grid(row=0,column=1)
label2.grid(row=1,column=0)
Movie_name.grid(row=1,column=1,ipadx=20,ipady=15)
imdb.grid(row=2,column=0)
Rating.grid(row=2,column=1,ipadx=20,ipady=15)
Gt.grid(row=3,column=0)
cl.grid(row=1, column=2)
instructions.grid(row=5, column=0)

Generate1.grid(row=3,column=1,ipadx=200,ipady=15)
Generate2.grid(row=4,column=1,ipadx=200,ipady=15)
Generate3.grid(row=5,column=1,ipadx=200,ipady=15)
Generate4.grid(row=6,column=1,ipadx=200,ipady=15)
Generate5.grid(row=7,column=1,ipadx=200,ipady=15)
Generate6.grid(row=8,column=1,ipadx=200,ipady=15)
Generate7.grid(row=9,column=1,ipadx=200,ipady=15)
Generate8.grid(row=10,column=1,ipadx=200,ipady=15)
Generate9.grid(row=11,column=1,ipadx=200,ipady=15)
Generate10.grid(row=12,column=1,ipadx=200,ipady=15)
Generate11.grid(row=3,column=2,ipadx=200,ipady=15)
Generate12.grid(row=4,column=2,ipadx=200,ipady=15)
Generate13.grid(row=5,column=2,ipadx=200,ipady=15)
Generate14.grid(row=6,column=2,ipadx=200,ipady=15)
Generate15.grid(row=7,column=2,ipadx=200,ipady=15)
Generate16.grid(row=8,column=2,ipadx=200,ipady=15)
Generate17.grid(row=9,column=2,ipadx=200,ipady=15)
Generate18.grid(row=10,column=2,ipadx=200,ipady=15)
Generate19.grid(row=11,column=2,ipadx=200,ipady=15)



cl.grid(row=13,column=0)

master.mainloop()