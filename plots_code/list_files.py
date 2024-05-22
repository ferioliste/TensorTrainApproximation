import os
if os.path.basename(os.getcwd()) != "TensorTrainApproximation":
    os.chdir('..')
os.chdir('plots')

folder_path = os.getcwd()
file_names = os.listdir(folder_path)
for file_name in file_names:
    print(f"""\\begin{{subfigure}}[c]{{.32\\textwidth}}
    \\centering
    \\includegraphics[width=.99\\linewidth]{{plots_{file_name[0]}/{file_name}}}
\end{{subfigure}}""")