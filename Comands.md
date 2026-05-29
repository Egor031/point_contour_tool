Активация окружения

Если терминал CMD:

.venv\Scripts\activate.bat

Если PowerShell не даёт активировать, можно запускать Python напрямую:

.\.venv\Scripts\python.exe -m app.main ...
Установка зависимостей
pip install -r requirements.txt

Если отдельно нужен GUI:

pip install dearpygui
Запуск основного обработчика

Запускать из корня проекта:

python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3
Построить маску и контур
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --keep-largest --contour
Контур + DXF для NX
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --keep-largest --contour --dxf
Контур + DXF + очищенное облако
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --keep-largest --contour --dxf --export-clean
Экспорт только точек около внешнего контура
python -m app.main data\input\test_01.asc --cell 1.6 --threshold 3 --keep-largest --contour --export-boundary --boundary-width-mm 1
Полная рабочая команда
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --keep-largest --fill-holes-area 1000 --contour --dxf --export-clean --export-boundary --boundary-width-mm 1
С прямоугольным ROI
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --roi 100 -1500 1200 -300 --keep-largest --contour --dxf --export-clean
С polygon ROI
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --roi-poly "100,-1500;800,-1450;1100,-800;900,-300;200,-350" --keep-largest --contour --dxf --export-clean

В PowerShell, если двойные кавычки мешают, можно так:

python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --roi-poly '100,-1500;800,-1450;1100,-800;900,-300;200,-350' --keep-largest --contour --dxf --export-clean
Поиск отверстий

Пока экспериментально:

python -m app.main data\input\test_holes.asc --cell 1.8 --threshold 3 --keep-largest --holes --min-hole-diameter-mm 5.5

Более мягкие фильтры:

python -m app.main data\input\test_holes.asc --cell 1.8 --threshold 3 --keep-largest --holes --min-hole-diameter-mm 5.5 --min-circularity 0.35 --max-circle-error-ratio 0.35
Запуск GUI-viewer

Из корня проекта:

python -m app.ui.viewer_app

Или если находишься внутри папки app:

python ui\viewer_app.py

Лучше запускать из корня проекта через:

python -m app.ui.viewer_app
Проверка синтаксиса

Для основного файла:

python -m py_compile app\main.py

Для GUI:

python -m py_compile app\ui\viewer_app.py

Для всех файлов вручную можно так:

python -m compileall app
Принудительно без кэша
python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --no-cache
Самые полезные текущие команды

GUI:

python -m app.ui.viewer_app

Обычный контур в DXF:

python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --keep-largest --contour --dxf

Контур + clean + boundary:

python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --keep-largest --contour --dxf --export-clean --export-boundary --boundary-width-mm 1

С polygon ROI:

python -m app.main data\input\test_01.asc --cell 1.8 --threshold 3 --roi-poly "..." --keep-largest --contour --dxf --export-clean --export-boundary --boundary-width-mm 1