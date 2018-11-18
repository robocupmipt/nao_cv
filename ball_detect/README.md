# Детекция мяча

get_images - файл для получения изображений с робота, также демонстрируется поворот головы.

BALL1.jpg и BALL2.jpg -  полученные таким образом изображения

BallClassifier - классификатор с использование каскада Хаара. 
 
bottom_cascade.xml, top_cascade.xml, other_cascade.xml - 3 классификатора от SPQR.

bottom_cascade.xml и top_cascade.xml для верхней и нижней камеры NAO, other_cascade.xml - получен по другой ссылке от той же команды ) 

BallClassifier(unsuccessful) - неудачные попытки детектировать мяч на основе детектора границ Кэнни, преобразования Хафа и эвристик на основе цвета 
