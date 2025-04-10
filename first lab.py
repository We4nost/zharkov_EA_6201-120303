import json
import numpy as np
import matplotlib.pyplot as plt

#Функция для вычисления функции
def y(x, a, b, c):
    return a / (1 + np.exp(-b * x + c))

#Считывание конфигурации из файла
def read_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def main():
#Чтение конфигурации
    config = read_config('config.json')

    xmin = config['xmin']
    step = config['step']
    xmax = config['xmax']
    a = config['a']
    b = config['b']
    c = config['c']

#Генерация значений x и вычисление y
    x_values = np.arange(xmin, xmax + step, step)
    y_values = y(x_values, a, b, c)


#Запись результатов
    with open('results.txt', 'w') as f:
        for x, y_val in zip(x_values, y_values):
            f.write(f"{x:.2f}\t{y_val:.4f}\n")

#Построение графика
    plt.plot(x_values, y_values, label='y(x) = a / (1 + e^(-bx + c))')
    plt.title('График функции y(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    #plt.legend()
    plt.savefig('function_plot.png')
    plt.show()  #График


if __name__ == "__main__":
    main()
