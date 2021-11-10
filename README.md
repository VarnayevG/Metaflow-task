# Metaflow-task

Мой пример: Grad-CAM class activation visualization https://keras.io/examples/vision/grad_cam/

Запуск: 
cd Grad-Cam
python3 main.py --environment=conda run

С зависимостями и conda пришлось очень много поработать, и мне так и не удалось понять это поведение полностью. Возможно, на другой машине с ее текущей конфигурацией скрипт может не запуститься, тогда нужно закомментировать строчку с @conda_base зависимостями и запустить все с помощью python3 main.py run

На выходе должна быть показана heatmap-а, а также сохранены heatmap и изображение с градиентом.