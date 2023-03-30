#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import os


# In[ ]:


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes'
]


# In[ ]:


disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    ' Migraine','Cervical spondylosis',
    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
    'Impetigo']


# In[ ]:


l2=[]
for i in range(0,len(l1)):
    l2.append(0)
print(l2)


# In[ ]:


df=pd.read_csv("training.csv")



df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)

df.head()


# In[ ]:


def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df1.shape
    columnNames = list(df1)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    


# In[ ]:


def plotScatterMatrix(df1, plotSize, textSize):
    df1 = df1.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df1 = df1.dropna('columns')
    df1 = df1[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df1 = df1[columnNames]
    ax = pd.plotting.scatter_matrix(df1, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df1.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[ ]:


plotPerColumnDistribution(df, 10, 5)


# In[ ]:


plotScatterMatrix(df, 20, 10)


# In[ ]:


X= df[l1]
y = df[["prognosis"]]
np.ravel(y)
print(X)


# In[ ]:


print(y)


# In[ ]:


tr=pd.read_csv("testing.csv")

#Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)

#printing the top 5 rows of the testing data
tr.head()


# In[ ]:


plotPerColumnDistribution(tr, 10, 5)


# In[ ]:


plotScatterMatrix(tr, 20, 10)


# In[ ]:


X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
print(X_test)


# In[ ]:


print(y_test)


# In[ ]:


def scatterplt(disea):
    x = ((df.loc[disea]).sum())#total sum of symptom reported for given disease
    x.drop(x[x==0].index,inplace=True)#droping symptoms with values 0
    print(x.values)
    y = x.keys()#storing nameof symptoms in y
    print(len(x))
    print(len(y))
    plt.title(disea)
    plt.scatter(y,x.values)
    plt.show()

def scatterinp(sym1,sym2,sym3,sym4,sym5):
    x = [sym1,sym2,sym3,sym4,sym5]#storing input symptoms in y
    y = [0,0,0,0,0]#creating and giving values to the input symptoms
    if(sym1!='SELECT HERE'):
        y[0]=1
    if(sym2!='SELECT HERE'):
        y[1]=1
    if(sym3!='SELECT HERE'):
        y[2]=1
    if(sym4!='SELECT HERE'):
        y[3]=1
    if(sym5!='SELECT HERE'):
        y[4]=1
    print(x)
    print(y)
    plt.scatter(x,y)
    plt.show()


# In[ ]:


root = Tk()

pred1=StringVar()
def DecisionTree():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((S1.get()=="SELECT HERE") or (S2.get()=="SELECT HERE")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        print(NameEn.get())
        from sklearn import tree

        clf3 = tree.DecisionTreeClassifier() 
        clf3 = clf3.fit(X,y)

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=clf3.predict(X_test)
        print("Decision Tree")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        psymptoms = [S1.get(),S2.get(),S3.get(),S4.get(),S5.get()]
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = clf3.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break

    
        if (h=='yes'):
            pred1.set(" ")
            pred1.set(disease[a])
        else:
            pred1.set(" ")
            pred1.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS DecisionTree(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO DecisionTree(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",(NameEn.get(),S1.get(),S2.get(),S3.get(),S4.get(),S5.get(),pred1.get()))
        conn.commit()  
        c.close() 
        conn.close()
        #printing scatter plot of input symptoms
        #printing scatter plot of disease predicted vs its symptoms
        scatterinp(Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get())
        scatterplt(pred1.get())


# In[ ]:


pred2=StringVar()
def randomforest():
    if len(NameEn.get()) == 0:
        pred2.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((S1.get()=="SELECT HERE") or (S2.get()=="SELECT HERE")):
        pred2.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf4 = RandomForestClassifier(n_estimators=100)
        clf4 = clf4.fit(X,np.ravel(y))

        # calculating accuracy 
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=clf4.predict(X_test)
        print("Random Forest")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)
    
        psymptoms = [S1.get(),S2.get(),S3.get(),S4.get(),S5.get()]
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred2.set(" ")
            pred2.set(disease[a])
        else:
            pred2.set(" ")
            pred2.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS RandomForest(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO RandomForest(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",(NameEn.get(),S1.get(),S2.get(),S3.get(),S4.get(),S5.get(),pred2.get()))
        conn.commit()  
        c.close() 
        conn.close()
        #printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred2.get())


# In[ ]:


pred3=StringVar()
def NaiveBayes():
    if len(NameEn.get()) == 0:
        pred3.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((S1.get()=="SELECT HERE") or (S2.get()=="SELECT HERE")):
        pred3.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb=gnb.fit(X,np.ravel(y))

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=gnb.predict(X_test)
        print("Naive Bayes")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        psymptoms = [S1.get(),S2.get(),S3.get(),S4.get(),S5.get()]
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = gnb.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred3.set(" ")
            pred3.set(disease[a])
        else:
            pred3.set(" ")
            pred3.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS NaiveBayes(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO NaiveBayes(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",(NameEn.get(),S1.get(),S2.get(),S3.get(),S4.get(),S5.get(),pred3.get()))
        conn.commit()  
        c.close() 
        conn.close()
         #printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred3.get())


# In[ ]:


pred4=StringVar()
def KNN():
    if len(NameEn.get()) == 0:
        pred4.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((S1.get()=="SELECT HERE") or (S2.get()=="SELECT HERE")):
        pred4.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.neighbors import KNeighborsClassifier
        knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        knn=knn.fit(X,np.ravel(y))
    
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=knn.predict(X_test)
        print("KNN")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        psymptoms = [S1.get(),S2.get(),S3.get(),S4.get(),S5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = knn.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break


        if (h=='yes'):
            pred4.set(" ")
            pred4.set(disease[a])
        else:
            pred4.set(" ")
            pred4.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS KNearestNeighbour(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO KNearestNeighbour(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",(NameEn.get(),S1.get(),S2.get(),S3.get(),S4.get(),S5.get(),pred4.get()))
        conn.commit()  
        c.close() 
        conn.close()
         #printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred4.get())


# In[ ]:


width=root.winfo_screenwidth()
height=root.winfo_screenheight()
root.configure(background='white')
root.geometry("%dx%d" % (width,height))
root.title('Smart Disease Predictor System')

canvas=Canvas(root,height=760,width=1450,bg="#fff",highlightthickness=3,highlightbackground='#5fab3c')
canvas.pack()
canvas.create_rectangle(40,160,705,730,width='2',outline="#5fab3c")
canvas.create_rectangle(745,160,1410,730,width='2',outline="#5fab3c")


# In[ ]:


prev_win=None
def Reset():
    global prev_win

    S1.set("SELECT HERE")
    S2.set("SELECT HERE")
    S3.set("SELECT HERE")
    S4.set("SELECT HERE")
    S5.set("SELECT HERE")
    
    NameEn.delete(first=0,last=100)
    
    pred1.set(" ")
    pred2.set(" ")
    pred3.set(" ")
    pred4.set(" ")
    try:
        prev_win.destroy()
        prev_win=None
    except AttributeError:
        pass


# In[ ]:


from tkinter import messagebox
def Exit():
    qExit=messagebox.askyesno("System","DO YOU WANT TO EXIT THE SYSTEM")
    if qExit:
        root.destroy()
        exit()


# In[ ]:


img= Image.open("logo.png")
logo = img.resize((90, 110))
test = ImageTk.PhotoImage(logo)
l = Label(image=test,bg="white")
l.image = test
l.place(x=720, y=3)
h = Label(root,text="AYURDOC", fg="#5fab3c",bg="white")
h.config(font=("Arial",20,"bold"))
h.place(x=705,y=120)


# In[ ]:


b1=Label(root, text="PATIENT's INFORMATION",bg="white")
b1.config(font=('Times',20,"bold"))
b1.place(x=250,y=170)

b2=Label(root, text="DIAGNOSIS",bg="white")
b2.config(font=('Times',20,"bold"))
b2.place(x=1040,y=170)

pcl=Label(root, text="TO GET HOME REMEDIES PRESS BELOW BUTTON",bg="white")
pcl.config(font=('Times',15,"bold"))
pcl.place(x=880,y=530)

downa= Image.open("downa.png")
downal = downa.resize((50,40))
dat = ImageTk.PhotoImage(downal)
l = Label(image=dat,bg="white")
l.image = dat
l.place(x=1100, y=570)

aimg= Image.open("redarrow.png")
arrow = aimg.resize((50,60))
at = ImageTk.PhotoImage(arrow)
a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=100, y=220)

NameLb = Label(root, text="NAME",bg="white")
NameLb.config(font=("Times",20,"bold"))
NameLb.place(x=155,y=235)


a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=100, y=280)
S1Lb = Label(root, text="SYMPTOM 1", fg="Black", bg="Ivory")
S1Lb.config(font=("Times",20,"bold"))
S1Lb.place(x=155,y=295)


a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=100, y=340)
S2Lb = Label(root, text="SYMPTOM 2", fg="Black", bg="Ivory")
S2Lb.config(font=("Times",20,"bold"))
S2Lb.place(x=155,y=355)


a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=100, y=400)
S3Lb = Label(root, text="SYMPTOM 3", fg="Black",bg="Ivory")
S3Lb.config(font=("Times",20,"bold"))
S3Lb.place(x=155,y=415)

a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=100, y=460)
S4Lb = Label(root, text="SYMPTOM 4", fg="Black", bg="Ivory")
S4Lb.config(font=("Times",20,"bold"))
S4Lb.place(x=155,y=475)

a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=100, y=520)
S5Lb = Label(root, text="SYMPTOM 5", fg="Black", bg="Ivory")
S5Lb.config(font=("Times",20,"bold"))
S5Lb.place(x=155,y=535)


# In[ ]:


a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=805, y=220)
lrLb = Label(root, text="DecisionTree", bg="white")
lrLb.config(font=("Times",20,"bold"))
lrLb.place(x=860,y=235)

a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=805, y=290)
destreeLb = Label(root, text="RandomForest", bg="white")
destreeLb.config(font=("Times",20,"bold"))
destreeLb.place(x=860,y=305)

a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=805, y=350)
ranfLb = Label(root, text="NaiveBayes", bg="White")
ranfLb.config(font=("Times",20,"bold"))
ranfLb.place(x=860,y=365)

a1 = Label(image=at,bg="white")
a1.image = at
a1.place(x=805, y=410)
knnLb = Label(root, text="kNearestNeighbour", bg="white")
knnLb.config(font=("Times",20,"bold"))
knnLb.place(x=860,y=425)
OPTIONS = sorted(l1)


# In[ ]:


NameEn = Entry(root,highlightthickness=3,highlightbackground='black')
NameEn.config(font=("Times",20,"bold"))
NameEn.place(x=370,y=235,width=330,height=35) 

fe = ("Times",15,"bold")

S1 = ttk.Combobox(root,font=fe)
S1['values'] = OPTIONS
S1['state'] = 'readonly'

S1.place(x=370,y=295,width=330,height=40)

S2 = ttk.Combobox(root,font=fe)
S2['values'] = OPTIONS
S2['state'] = 'readonly'
#S2.config(font=fe)
S2.place(x=370,y=355,width=330,height=40)

S3 = ttk.Combobox(root,font=fe)
S3['values'] = OPTIONS
S3['state'] = 'readonly'
#S3.config(font=fe)
S3.place(x=370,y=415,width=330,height=40)

S4 = ttk.Combobox(root,font=fe)
S4['values'] = OPTIONS
S4['state'] = 'readonly'
#S4.config(font=fe)
S4.place(x=370,y=475,width=330,height=40)

S5 = ttk.Combobox(root,font=fe)
S5['values'] = OPTIONS
S5['state'] = 'readonly'
#S5.config(font=fe)
S5.place(x=370,y=535,width=330,height=40)


# In[ ]:


dst = Button(root, text="DecisionTree", command=DecisionTree,bg="#5fab3c",fg="white")
dst.config(font=("Times",20,"bold"))
dst.place(x=130,y=620,width=250,height=40)

rnf = Button(root, text="RandomForest", command=randomforest,bg="#5fab3c",fg="white")
rnf.config(font=("Times",20,"bold"))
rnf.place(x=420,y=620,width=250,height=40)


lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="#5fab3c",fg="white")
lr.config(font=("Times",20,"bold"))
lr.place(x=130,y=670,width=250,height=40)


kn = Button(root, text="KNN", command=KNN,bg="#5fab3c",fg="white")
kn.config(font=("Times",20,"bold"))
kn.place(x=420,y=670,width=250,height=40)


rs = Button(root,text="Reset Inputs", command=Reset,bg="black",fg="white",width=15)
rs.config(font=("Times",15,"bold"))
rs.place(x=80,y=90,width=160,height=40)


ex = Button(root,text="Exit System", command=Exit,bg="black",fg="white",width=15)
ex.config(font=("Times",15,"bold"))
ex.place(x=1290,y=90,width=160,height=40)


# In[ ]:


t1=Label(root,font=("Times",15,"bold"),text="Decision Tree",height=1,bg="#5fab3c"
         ,width=22,fg="white",textvariable=pred1,relief="sunken").place(x=1150,y=245)

t2=Label(root,font=("Times",15,"bold"),text="Random Forest",height=1,bg="#5fab3c"
         ,width=22,fg="white",textvariable=pred2,relief="sunken").place(x=1150,y=305)

t3=Label(root,font=("Times",15,"bold"),text="Naive Bayes",height=1,bg="#5fab3c"
         ,width=22,fg="white",textvariable=pred3,relief="sunken").place(x=1150,y=365)

t4=Label(root,font=("Times",15,"bold"),text="kNearest Neighbour",height=1,bg="#5fab3c"
         ,width=22,fg="white",textvariable=pred4,relief="sunken").place(x=1150,y=425)


def gethomeremedies():
    import tkinter as tk
    my_child=tk.Toplevel(root)
    my_child.geometry("800x800")
    my_child.configure(background='white')
    my_child.title('Get Home Remedies')
    my_str1=tk.StringVar()
    my_str1.set(str(pred1))
    res=tk.Label(my_child,text=my_str1,fg="black")
    res.place(x=20,y=50)
    b2=tk.Button(my_child,text='Close',command=my_child.destroy)
    b2.place(x=100,y=200)
    
    
    
get = Button(root, text="GET HOME REMEDIES",command=gethomeremedies,bg="#5fab3c",fg="white")
get.config(font=("Times",15,"bold"))
get.place(x=980,y=630,width=300,height=40)


# In[ ]:


root.mainloop()


# In[ ]:




