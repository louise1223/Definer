# 导入tk
from tkinter import *
import tkinter as tk
import tkinter.font as tf
from tkinter import ttk
from tkinter import filedialog
import cerevisiae as Cer
import musculus as Mus
import sapiens as Sap
import os
from tkinter.font import nametofont
# 创建主窗口
window = tk.Tk()
window.title('Main')
window.geometry('1200x730+0+0')
window.config(bg="white")

def data_(file_name):
    # print("打印结果；")
    # print("file_name:", file_name)
    if "cerevisiae" == file_name:
        # print("进入cerevisiae判断语句")
        lens = 31
        na = "cerevisiae"
        f_1 = open("../test/valib_P.txt", "r")
        f_2 = open("../test/valib_N.txt", "r")
    elif "musculus" == file_name:
        # print("进入musculus判断语句")
        lens = 21
        na = "musculus"
        f_1 = open("../test/valib_P.txt", "r")
        f_2 = open("../test/valib_N.txt", "r")
    elif "cerevisiae_1" == file_name:
        # print("进入cerevisiae_1函数")
        lens = 31
        na = "cerevisiae"
        f_1 = open("../test/test_P.txt", "r")
        f_2 = open("../test/test_N.txt", "r")

    else:
        # print("进入else判断语句")
        lens = 21
        na = "sapiens"
        f_1 = open("../test/valib_P.txt", "r")
        f_2 = open("../test/valib_N.txt", "r")
    # print("出了判断语句：")
    l, n = [], []
    for (text, text_2)  in zip(f_1.readlines(), f_2.readlines()):
        w = text.strip(">")
        l.append(">H."+na+"_" + w.strip("\n"))
        n.append(">H."+na+"_" + text_2.strip("\n").strip(">"))
    l = l + n
    file_names = l[0::2]
    site = ["Yes", "No"]
    data = []
    if "cerevisiae_1" == file_name:
        for i in file_names:
            if "P" in i:
                data.append([i, lens, site[0]])
            else:
                data.append([i, lens, site[1]])
    else:
        f_3 = open("../test/pre_result.txt")
        for i, j in zip(file_names, f_3.readlines()):
            data.append([i , lens, j])
        f_3.close()

    # data = [[i , lens, site[0]] if "P" in i  else data.append([i , lens, site[1]]) for i in file_names ]
    f_1.close()
    f_2.close()
    # print(data)
    with open("../test/zhanshi.txt","w") as f:
        for i in data:
            f.write(i[0].strip()+"\t")
            f.write(str(i[1])+"\t")
            f.write(i[2])

    return data


# 下载文件
def Download_file():
    true_str = "Download Successful"
    false_str = "Download Failed"
    down_win = tk.Tk()
    down_win.title('Notice')
    down_win.geometry('350x100+300+300')
    down_win.config(bg="white")
    text_1 = tk.Text(down_win, height=3, width=30, wrap="none")
    font1 = tf.Font(family='微软雅黑', size=12)

    try:
        f_1 = open("../test/download_file.txt", "w")
        with open("../test/zhanshi.txt", "r") as f:
            for i in f.readlines():
                f_1.write(i)
        f_1.close()
        print("文件已经保存！！！")

        border = LabelFrame(down_win, bg="black")
        border.pack(side="left")
        l = tk.Label(border, bg="white", width=30, height= 5,
                     text=true_str,
                     font=("Times New Roman", 17), fg="black", wraplength=300, justify="center", )
        l.pack(side="left")
    except:
        border = LabelFrame(down_win, bg="black")
        border.pack(side="left")
        l = tk.Label(border, bg="white", width=30, height=5,
                     text=false_str,
                     font=("Times New Roman", 17), fg="black", wraplength=300, justify="center", )
        l.pack(side="left")
    text_1.config(font=font1, fg="black")
    down_win.mainloop()


def make_tk():

    var = tk.StringVar()
    border = LabelFrame(window, bg="black")
    border.pack(side="left")
    l = tk.Label(border, bg="white", width=30, height=100, text='      Please enter or copy the RNA sequence that you wish to query into the input box located in the lower-right portion of the screen. Alternatively, you can upload a txt file. To view the required format of the txt file,'
                                                               ' simply click the "Sample" button. You can choose from three different models to predict the sequence, each named after a corresponding species. Once you have selected your desired model, click the "confirm" '
                                                               'button to view the forecast results, which will be displayed on your computer screen.',
                 font=("Times New Roman", 17),fg="black", wraplength=300,justify="center",)
    l.pack(side="left")

    # 案例显示
    def example_file():
        # 显示数据内容
        text.delete("1.0", "end")
        font1 = tf.Font(family='微软雅黑', size=12)
        txt = []
        f_1 = open(r"../test/example_data.txt", "r")

        for i in f_1.readlines():
            text.insert(END, i)
            txt.append(i)
        with open("../test/example1_data.txt", "w") as f:
            for i in txt:
                f.write(i)

        text.config(font=font1, fg="black")

    # 选择模型
    def select_model():
        # print("判断是否文件存在：", os.path.exists("../test/example1_data.txt"))
        # 判断是否有案例文件：
        if os.path.exists("../test/example1_data.txt"):
            print_selection(1)
        else:
            # 读文件- 数据分为两类
            P_file = open("../test/valib_P.txt", "w")
            N_file = open("../test/valib_N.txt", "w")
            with open("../test/valib_P_N.txt", "r") as test_data:
                file_list = [[line] for line in test_data.readlines()]
            sq_N, sq_P = [], []
            for index, i in enumerate(file_list[::2]):
                if "N" in i[0]:
                    # print("阴性样本：", i[0], file_list[2 * index + 1])
                    N_file.write(i[0])
                    N_file.write(file_list[2 * index + 1][0])
                elif "P" in i[0]:
                    # print("阳性样本：",i[0],file_list[2 * index + 1])
                    P_file.write(i[0])
                    P_file.write(file_list[2 * index + 1][0])
            # print("循环结束：")
            if "cerevisiae" in var.get():
                Cer.out_("../test/valib_P.txt", "../test/valib_N.txt")
                print_selection(2)
            elif "musculus" in var.get():
                # Mus.out_("../test/valib_P.txt", "../test/valib_N.txt")
                Mus.out_("../test/valib_P.txt", "../test/valib_N.txt")
                print_selection(2)

            else:
                Sap.out_("../test/valib_P.txt", "../test/valib_N.txt")
                print_selection(2)

            P_file.close()
            N_file.close()

    # 上传文件
    def upload_file():
        # 删除案例文件
        if os.path.exists("../test/example1_data.txt"):
            print("文件存在")
            os.remove(r'../test/example1_data.txt')

        selectFile = tk.filedialog.askopenfilename()  # 返回文件路径
        # 显示数据内容
        print("文件不存在。",selectFile)
        text.delete("1.0", "end")
        font1 = tf.Font(family='微软雅黑', size=12)
        txt = []
        f_1 = open(selectFile, "r")
        for i in f_1.readlines():
            text.insert(END, i)
            txt.append(i)
        text.config(font=font1, fg="black")

        # 保存数据
        file_path = "../test"
        if os.path.exists(file_path):
            # print("test文件夹已经存在。")
            pass
        else:
            os.mkdir(file_path)
        # 重新写入数据
        with open('../test/valib_P_N.txt','w') as f:
            for i in txt:
                f.write(i)


    # 显示第二个表格
    def print_selection(num):

        # 创建tk窗口
        win1 = tk.Tk()
        win1.title('Result：')
        win1.geometry('950x750+20+20')

        # frame容器放置表格
        frame01 = Frame(win1)
        frame01.place(x=0, y=0, width=950, height=600)
        # # 加载滚动条
        scrollBar = Scrollbar(frame01)
        scrollBar.pack(side=RIGHT, fill=Y)

        # 准备表格
        tree = ttk.Treeview(win1, columns=('name', 'len', 'site'), show="headings",
                        yscrollcommand=scrollBar.set, height=30)
        tree.pack()


        tree.heading('name', text="Sequence")
        tree.heading('len', text="Number of nucleotides",)
        tree.heading('site', text="Site",)
        tree.tag_configure("evenColor", background="white")  # 设置标签
        tree.column("name", anchor="center", width=320)
        tree.column("len", anchor="center", width=320)
        tree.column("site", anchor="center", width=310)
        tree.tag_configure('site', font='宋体 24')

        # 设置关联
        scrollBar.config(command=tree.yview)

        if num == 1:
            data= data_("cerevisiae_1")
            # print(data)
        else:
            data = data_(var.get())

        for itm in data:
            tree.insert("", tk.END, values=itm, tags=("evenColor"))
        tree.pack(expand=1)

        # 下载文件按钮
        f_B = Frame(win1)
        f_B.pack()
        T2_win = tk.Button(f_B, text='Download', command=Download_file, width=8, height=1, fg='black', bg="white",
                           font=("Times New Roman", 18, "bold"))
        T2_win.pack(pady=40,padx=0)

    # 按钮1及其功能
    r1 = tk.Radiobutton(window, text='S.cerevisiae ', variable=var, value='cerevisiae',  width=27,height=1,fg='black',bg="white",font=("Times New Roman",20, "bold")#,command=select_model
                        )
    r1.pack(pady=10)

    r2 = tk.Radiobutton(window, text='M.musculus', variable=var, value='musculus' , width=27,height=1,fg='black',bg="white",font=("Times New Roman",20, "bold")#,command=select_model
                        )
    r2.pack(pady=10)

    r3 = tk.Radiobutton(window, text='H.sapiens    ', variable=var, value='sapiens',  width=27,height=1,fg='black',bg="white",font=("Times New Roman",20, "bold") #,command=select_model
                        )
    r3.pack(pady=10)
    r1.select()# 默认选择

    # 确认按钮
    a = tk.PanedWindow(sashrelief=tk.SUNKEN, background="lightgray", width=390)
    a.pack()
    btn1 = tk.Button(a,text='Confirm',command=select_model, width=8,height=1,fg='black',bg="white",font=("Times New Roman",18,"bold"))

    # 上传文件按钮
    btn2 = tk.Button(a, text='Upload File', command=upload_file, width=12,height=1,fg='black',bg="white",font=("Times New Roman",16, "bold"))
    # btn.pack()

    # 实例
    btn3 = tk.Button(a, text='Example', command=example_file, width=8,height=1,fg='black',bg="white",font=("Times New Roman",18, "bold"))
    a.add(btn2)
    a.add(btn1)
    a.add(btn3)
    # 显示数据内容
    scr1 = tk.Scrollbar(window)
    scr1.pack(side='right', fill=tk.Y)  # 垂直滚动条
    text = tk.Text(window, width=80, height=10, bg="white")
    text.pack(side='left', fill=tk.BOTH, expand=True)
    text.config(yscrollcommand=scr1.set)  # 滚动设置互相绑定
    scr1.config(command=text.yview)  # 滚动设置互相绑定

    window.mainloop()

make_tk()