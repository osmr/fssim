from PIL import Image, ImageDraw

# Параметры изображения
size = 20  # Размер квадрата (пиксели)
squares = 5  # Количество клеток в ряду
square_size = size // squares

# Создаем белое изображение
image = Image.new('RGB', (size, size), 'white')
draw = ImageDraw.Draw(image)

# Рисуем шахматную доску
for row in range(squares):
    for col in range(squares):
        if (row + col) % 2 == 0:
            color = 'black'  # Чёрная клетка
        else:
            color = 'white'  # Белая клетка


        # Координаты текущей клетки
        x0 = col * square_size
        y0 = row * square_size
        x1 = x0 + square_size
        y1 = y0 + square_size

        draw.rectangle([x0, y0, x1, y1], fill=color)

# Сохраняем изображение
image.save('chessboard.png')
print("Изображение сохранено как 'chessboard.png'")
