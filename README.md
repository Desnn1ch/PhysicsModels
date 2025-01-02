# Симуляция частицы в конденсаторе

Программа симулирует движение электрона внутри цилиндрического конденсатора, учитывая электрическое поле и начальную скорость. В процессе симуляции вычисляется траектория, ускорение, скорость частицы, а также минимальное напряжение, необходимое для того, чтобы частица не покинула конденсатор.

## Требования
- **Графический вывод:** Построение следующих графиков:
  -  y(t) : Вертикальное положение электрона в зависимости от времени.
  -  a_y(t) : Вертикальное ускорение электрона в зависимости от времени.
  -  v_y(t) : Вертикальная скорость электрона в зависимости от времени.
  -  y(x) : Вертикальное положение электрона в зависимости от горизонтального положения.
- **Минимальное напряжение:** Определение минимального напряжения, при котором электрон не покидает конденсатор.

## Вычисления
В ходе отрисовки графиков можно отбросить g, грубо говоря это простая константа, а нас интересует динамическое поведение частицы в конденсаторе
![site](formulas.png)

### Константы

- `e_0`: Диэлектрическая проницаемость вакуума 
- `q`: Заряд электрона 
- `m`: Масса электрона
- `r`: Радиус внутреннего цилиндра 
- `R`: Радиус внешнего цилиндра 
- `V`: Начальная скорость электрона
- `L`: Длина конденсатора
- `T`: Общее время симуляции, вычисляется из VxT = L
- `delta_t`: Шаг времени для симуляции.

### Результат
![site](example.png)

1. **Итоговые значения:**
   - Минимальное напряжениe

2. **Графики:**
   - y(t)
   - a_y(t)
   - v_y(t)
   - y(x)
