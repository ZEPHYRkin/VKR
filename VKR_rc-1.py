import tkinter as tk
LARGE_FONT= ("Verdana", 12)
from tkinter import ttk, messagebox
from tkinter import *
import pandas as pd
import itertools
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('occup.csv')
SVC_model = svm.SVC(kernel="poly", degree=2)
GNB_model = GaussianNB()
KNN_model = KNeighborsClassifier(n_neighbors = 15)
RFC_model = RandomForestClassifier(criterion = "gini", max_depth=12, random_state=23)
MLP_model = MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(5, 12), random_state=5)
GBC_model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=10, random_state=5)

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        tk.Tk.wm_title(self, "ВКР")
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne):

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

def choose_data(temp_enabled, light_enabled, sound_enabled, CO2_enabled, motion_enabled):
    master_key = [temp_enabled, light_enabled, sound_enabled, CO2_enabled, motion_enabled]
    if master_key == [1,1,1,1,1]: X = data.iloc[:, 2:17].values
    if master_key == [1,1,1,1,0]: X = data.iloc[:, 2:15].values
    if master_key == [1,1,1,0,1]: X = data.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,16,17]].values
    if master_key == [1,1,1,0,0]: X = data.iloc[:, 2:13].values
    if master_key == [1,1,0,1,1]: X = data.iloc[:, [2,3,4,5,6,7,8,9,14,15,16,17]].values
    if master_key == [1,1,0,1,0]: X = data.iloc[:, [2,3,4,5,6,7,8,9,14,15]].values
    if master_key == [1,1,0,0,1]: X = data.iloc[:, [2,3,4,5,6,7,8,9,16,17]].values
    if master_key == [1,1,0,0,0]: X = data.iloc[:, 2:9].values
    if master_key == [1,0,1,1,1]: X = data.iloc[:, [2,3,4,5,10,11,12,13,14,15,16,17]].values
    if master_key == [1,0,1,1,0]: X = data.iloc[:, [2,3,4,5,10,11,12,13,14,15]].values
    if master_key == [1,0,1,0,1]: X = data.iloc[:, [2,3,4,5,10,11,12,13,16,17]].values
    if master_key == [1,0,1,0,0]: X = data.iloc[:, [2,3,4,5,10,11,12,13]].values
    if master_key == [1,0,0,1,1]: X = data.iloc[:, [2,3,4,5,14,15,16,17]].values
    if master_key == [1,0,0,1,0]: X = data.iloc[:, [2,3,4,5,14,15]].values
    if master_key == [1,0,0,0,1]: X = data.iloc[:, [2,3,4,5,16,17]].values
    if master_key == [1,0,0,0,0]: X = data.iloc[:, 2:5].values
    if master_key == [0,1,1,1,1]: X = data.iloc[:, 6:17].values
    if master_key == [0,1,1,1,0]: X = data.iloc[:, [6,7,8,9,10,11,12,13,14,15]].values
    if master_key == [0,1,1,0,1]: X = data.iloc[:, [6,7,8,9,10,11,12,13,16,17]].values
    if master_key == [0,1,1,0,0]: X = data.iloc[:, [6,7,8,9,10,11,12,13]].values
    if master_key == [0,1,0,1,1]: X = data.iloc[:, [6,7,8,9,14,15,16,17]].values
    if master_key == [0,1,0,1,0]: X = data.iloc[:, [6,7,8,9,14,15]].values
    if master_key == [0,1,0,0,1]: X = data.iloc[:, [6,7,8,9,16,17]].values
    if master_key == [0,1,0,0,0]: X = data.iloc[:, [6,7,8,9]].values
    if master_key == [0,0,1,1,1]: X = data.iloc[:, 10:17].values
    if master_key == [0,0,1,1,0]: X = data.iloc[:, [10,11,12,13,14,15]].values
    if master_key == [0,0,1,0,1]: X = data.iloc[:, [10,11,12,13,16,17]].values
    if master_key == [0,0,1,0,0]: X = data.iloc[:, [10,11,12,13]].values
    if master_key == [0,0,0,1,1]: X = data.iloc[:, 14:17].values
    if master_key == [0,0,0,1,0]: X = data.iloc[:, [14,15]].values
    if master_key == [0,0,0,0,1]: X = data.iloc[:, 16:17].values
    if master_key == [0,0,0,0,0]: messagebox.showinfo("Ошибка","Выберите хотя бы один тип параметра")
    y = data['Room_Occupancy_Count'].values
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 23)

def clear():
    canvas.get_tk_widget().grid_forget()
    toolbar_frame.grid_forget()

def plotting(frame4, num):
    param1 = ["S1_Temp", "S2_Temp", "S3_Temp","S4_Temp"]
    param2 = ["S1_Light", "S2_Light", "S3_Light","S4_Light"]
    param3 = ["S1_Sound", "S2_Sound", "S3_Sound","S4_Sound"]
    param4 = ["S5_CO2"]
    param5 = ["S6_PIR", "S7_PIR"]
    param6 = ["Room_Occupancy_Count"]
    if num == 1: 
        param = param1 
        plot_label = "Температура, C°"
    elif num == 2: 
        param = param2
        plot_label = "Освещенность, люкс"
    elif num == 3: 
        param = param3
        plot_label = "Звук"
    elif num == 4: 
        param = param4
        plot_label = "Углексилый газ, ppm"
    elif num == 5: 
        param = param5
        plot_label = "Движение"
    elif num == 6: 
        param = param6
        plot_label = "Занятость помещения"
    global canvas
    f = Figure(figsize=(5,5), dpi=100)
    temp_plot = f.add_subplot(111)
    a = [i for i in range(10129)]
    for i in range(len(param)):
        graph = temp_plot.plot(a, data[param].values)
    f.legend(graph, param)
    temp_plot.set_xlabel('Номер наблюдения')
    temp_plot.set_ylabel(plot_label)

    canvas = FigureCanvasTkAgg(f, frame4)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 6, column = 1, sticky = NSEW, columnspan=6)

    global toolbar_frame
    toolbar_frame = Frame(master=frame4)
    toolbar_frame.grid(row = 7, column = 1, sticky = NSEW, columnspan=5)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()


def estimate(model):
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    messagebox.showinfo("Результаты",classification_report(prediction, y_test))

def estimate_bagging(key, num_est):
    if key == "Опорные векторы": model = svm.SVC(kernel="poly", degree=2)
    elif key == "Наивный Байес": model = GaussianNB()
    elif key == "k-ближайших соседей": model = KNeighborsClassifier(n_neighbors = 15)
    elif key == "Случайный лес": model = RandomForestClassifier(criterion = "gini", max_depth=12, random_state=23)
    elif key == "Нейронные сети": model = MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(5, 12), random_state=1)
    elif key == "Бустинг": model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=10, random_state=5)
    BCL = BaggingClassifier(estimator= model, n_estimators = int(num_est), random_state=23)
    BCL.fit(X_train,y_train)
    prediction = BCL.predict(X_test)
    messagebox.showinfo("Результаты", classification_report(prediction, y_test))

def estimate_stacking(key1, key2, key3):
    if key1 == "Опорные векторы": model1 = ('svm', svm.SVC(kernel="poly", degree=2))
    elif key1 == "Наивный Байес": model1 = ('gb', GaussianNB())
    elif key1 == "k-ближайших соседей": model1 =('knn', KNeighborsClassifier(n_neighbors = 15))
    elif key1 == "Случайный лес": model1 = ('rf', RandomForestClassifier(criterion = "gini", max_depth=12, random_state=23))
    elif key1 == "Нейронные сети": model1 = ('mlp', MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(5, 12), random_state=1))

    if key2 == "Опорные векторы": model2 = ('svm', svm.SVC(kernel="poly", degree=2))
    elif key2 == "Наивный Байес": model2 = ('gb', GaussianNB())
    elif key2 == "k-ближайших соседей": model2 =('knn', KNeighborsClassifier(n_neighbors = 15))
    elif key2 == "Случайный лес": model2 = ('rf', RandomForestClassifier(criterion = "gini", max_depth=12, random_state=23))
    elif key2 == "Нейронные сети": model2 = ('mlp', MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(5, 12), random_state=1))

    if key3 == "Опорные векторы": model3 = svm.SVC(kernel="poly", degree=2)
    elif key3 == "Наивный Байес": model3 = GaussianNB()
    elif key3 == "k-ближайших соседей": model3 = KNeighborsClassifier(n_neighbors = 15)
    elif key3 == "Случайный лес": model3 = RandomForestClassifier(criterion = "gini", max_depth=12, random_state=23)
    elif key3 == "Нейронные сети": model3= MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(5, 12), random_state=1)
    estimators = [model1, model2]
    SCL = StackingClassifier(estimators= estimators, final_estimator= model3)
    SCL.fit(X_train,y_train)
    prediction = SCL.predict(X_test)
    messagebox.showinfo("Результаты",classification_report(prediction, y_test))

def estimate_ada(num_est2):
    model = RandomForestClassifier(criterion = "gini", max_depth=12, random_state=23)
    ACL = AdaBoostClassifier(estimator= model, n_estimators = int(num_est2), algorithm='SAMME', random_state=23)
    ACL.fit(X_train,y_train)
    prediction = ACL.predict(X_test)
    messagebox.showinfo("Результаты",classification_report(prediction, y_test))

def build_matrixes(flag):
    if flag == 1: model = SVC_model
    elif flag == 2: model = GNB_model
    elif flag == 3: model = KNN_model
    elif flag == 4: model = RFC_model
    elif flag == 5: model = MLP_model
    elif flag == 6: model = GBC_model
    def plot_confusion_matrix(cm, classes,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        font = {'size' : 15}
        plt.rc('font', **font)
    def build_matrix():
        cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cnf_matrix, classes=['0', '1',  '2',  '3'], title='Confusion matrix')
        plt.savefig("conf_matrix.png")
        plt.show()
 
    # вызов функции для построения матрицы при нажатии на кнопку
    build_matrix()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        notebook = ttk.Notebook(self)
        notebook.grid()

        frame1 = ttk.Frame(notebook,width=775, height=575)
        frame2 = ttk.Frame(notebook,width=775, height=575)
        frame3 = ttk.Frame(notebook,width=775, height=575)
        frame4 = ttk.Frame(notebook,width=775, height=575)
        
        frame1.grid(sticky=NSEW)
        frame2.grid(sticky=NSEW)
        frame3.grid(sticky=NSEW)
        frame4.grid(sticky=NSEW)

        notebook.add(frame1, text="Начальная страница",padding=5)
        notebook.add(frame2, text="Выбор данных",padding=5)
        notebook.add(frame3, text="Методы классификации",padding=5)
        notebook.add(frame4, text="Графики",padding=5)
        # -------------------------------------
        label1 = tk.Label(frame1, text="Добро пожаловать в программу ВКР!", font=LARGE_FONT, padx=25, pady=25)
        label1.grid(row=1,column=3)
        label1 = tk.Label(frame1, text="Инструкция:", font=LARGE_FONT, padx=25, pady=10)
        label1.grid(row=2,column=3, sticky=W)
        label1 = tk.Label(frame1, text='1. Перейдите на вкладку "Данные" и выберите параметры, на которых модель будет обучаться.', font=LARGE_FONT, padx=25, pady=10)
        label1.grid(row=3,column=3, sticky=W)
        label1 = tk.Label(frame1, text='2. Перейдите на вкладку "Методы классификации" и опробуйте различные методы МО и ансамбли.', font=LARGE_FONT, padx=25, pady=10)
        label1.grid(row=4,column=3, sticky=W)
        label1 = tk.Label(frame1, text='3. Перейдите на вкладку "Графики" для визуального представления собранных данных.', font=LARGE_FONT, padx=25, pady=10)
        label1.grid(row=5,column=3, sticky=W)
        # -------------------------------------
        label1 = tk.Label(frame2, text="Выбор данных", font=LARGE_FONT)
        label1.grid(row=1,column=3)
        temp_enabled = IntVar()
        enabled_checkbutton = ttk.Checkbutton(frame2, text="Температура", variable=temp_enabled)
        enabled_checkbutton.grid(row=2,column=1)
        light_enabled = IntVar()
        enabled_checkbutton1 = ttk.Checkbutton(frame2, text="Освещенность", variable=light_enabled)
        enabled_checkbutton1.grid(row=2,column=2)
        sound_enabled = IntVar()
        enabled_checkbutton2 = ttk.Checkbutton(frame2, text="Звук", variable=sound_enabled)
        enabled_checkbutton2.grid(row=2,column=3)
        CO2_enabled = IntVar()
        enabled_checkbutton3 = ttk.Checkbutton(frame2, text="Углекислый газ", variable=CO2_enabled)
        enabled_checkbutton3.grid(row=2,column=4)
        motion_enabled = IntVar()
        enabled_checkbutton4 = ttk.Checkbutton(frame2, text="Движение", variable=motion_enabled)
        enabled_checkbutton4.grid(row=2,column=5, padx= 50)
        button2 = ttk.Button(frame2, text="Подтвердить", command=lambda: choose_data(
            temp_enabled.get(), light_enabled.get(), sound_enabled.get(), CO2_enabled.get(), motion_enabled.get()))
        button2.grid(row=6,column=3)
        for c in range(5): frame2.columnconfigure(index=c, weight=1)
        # -------------------------------------
        label = tk.Label(frame3, text="Выберите метод классификации", font=LARGE_FONT)
        label.grid(row=3,column=3)
        button3 = ttk.Button(frame3, text="SVM", command=lambda:[estimate(SVC_model), build_matrixes(1)])
        button3.grid(row=4,column=2)
        button4 = ttk.Button(frame3, text="GNB", command=lambda:[estimate(GNB_model), build_matrixes(2)])
        button4.grid(row=4,column=4)
        button5 = ttk.Button(frame3, text="k-ближайших соседей", command=lambda:[estimate(KNN_model), build_matrixes(3)])
        button5.grid(row=5,column=2)      
        button6 = ttk.Button(frame3, text="Случайный лес", command=lambda:[estimate(RFC_model), build_matrixes(4)])
        button6.grid(row=5,column=4)
        button7 = ttk.Button(frame3, text="Нейронные сети", command=lambda:[estimate(MLP_model), build_matrixes(5)])
        button7.grid(row=6,column=2)
        button8 = ttk.Button(frame3, text="Бустинг", command=lambda:[estimate(GBC_model), build_matrixes(6)])
        button8.grid(row=6,column=4)
        button9 = ttk.Button(frame3, text="Стекинг", command=lambda:[estimate_stacking(key1 = combo1.get(),key2 = combo2.get(), key3 = combo3.get() )])
        button9.grid(row=8,column=1)
        available_methods = ["Опорные векторы","Наивный Байес", "k-ближайших соседей", "Случайный лес", "Нейронные сети", 'Бустинг']
        combo1 = ttk.Combobox(frame3, values = available_methods)
        combo1.grid(row=8,column=2)
        combo2 = ttk.Combobox(frame3, values = available_methods)
        combo2.grid(row=8,column=3)
        combo3 = ttk.Combobox(frame3, values = available_methods)
        combo3.grid(row=8,column=4)
        button10 = ttk.Button(frame3, text="Бэггинг", command=lambda:[estimate_bagging(key = combo4.get(),num_est = entry1.get() )])
        button10.grid(row=9,column=1)
        entry1 = ttk.Entry(frame3)
        entry1.grid(row=9,column=2)
        combo4 = ttk.Combobox(frame3, values = available_methods)
        combo4.grid(row=9,column=3)
        button11 = ttk.Button(frame3, text="Адаптивный бустинг", command=lambda:[estimate_ada(num_est2 = entry2.get())])
        button11.grid(row=10,column=1)
        entry2 = ttk.Entry(frame3)
        entry2.grid(row=10,column=2)
        for c in range(5): frame2.columnconfigure(index=c, weight=1)
        # -------------------------------------
        label = tk.Label(frame4, text="Выберите данные для графика", font=LARGE_FONT)
        label.grid(row=3,column=3, columnspan=2)
        button9 = ttk.Button(frame4, text="Температура", command=lambda:plotting(frame4, 1))
        button9.grid(row=4,column=1, padx=50)
        button9 = ttk.Button(frame4, text="Освещенность", command=lambda:plotting(frame4, 2))
        button9.grid(row=4,column=2)
        button9 = ttk.Button(frame4, text="Звук", command=lambda:plotting(frame4, 3))
        button9.grid(row=4,column=3)
        button9 = ttk.Button(frame4, text="Углекислый газ", command=lambda:plotting(frame4, 4))
        button9.grid(row=4,column=4)
        button9 = ttk.Button(frame4, text="Движение", command=lambda:plotting(frame4, 5))
        button9.grid(row=4,column=5)
        button9 = ttk.Button(frame4, text="Занятость помещения", command=lambda:plotting(frame4, 6))
        button9.grid(row=4,column=6, padx=50)
        button9 = ttk.Button(frame4, text="Очистить график", command=lambda:clear())
        button9.grid(row=5,column=3, columnspan=2)
        # -------------------------------------

  
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, padx=10, pady=10)





app = Application()
app.mainloop()