# FewShot SAM-CLIP

Few-shot детектор, реализованный и использованием SAM и CLIP.

## Установка

```
pip install poetry
poetry install
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Использование

Произвести детекцию с использованием изображений из папки `images`:

```
python examples/fewshot_example.py
```

Полученное изображение сохранится в папке `outs`.
