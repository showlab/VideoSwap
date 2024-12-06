import datetime
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import omegaconf
import torch
from PIL import Image

from videoswap.utils.edlora_util import bind_concept_prompt


def text_under_image(image: np.ndarray,
                     text: str,
                     text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8)
              for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones(
        (h * num_rows + offset * (num_rows - 1), w * num_cols + offset *
         (num_cols - 1), 3),
        dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset):i * (h + offset) + h:, j * (w + offset):j *
                   (w + offset) + w] = images[i * num_cols + j]

    if save_path is not None:
        pil_img = Image.fromarray(image_)
        now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        pil_img.save(f'{save_path}/{now}.png')
    # display(pil_img)


def get_word_inds(text: str, word_place: int, tokenizer):

    assert isinstance(word_place, str), 'error type'

    # edlora
    if hasattr(tokenizer, 'new_concept_cfg'):
        text = bind_concept_prompt(text, tokenizer.new_concept_cfg)[0]
        word_place = bind_concept_prompt(word_place, tokenizer.new_concept_cfg)[0]

    split_text = text.split(' ')

    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []

    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip('#') for item in tokenizer.encode(text)][1: -1]

        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha,
                           bounds: Union[float, Tuple[float, float]],
                           prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    # Edit the alpha map during attention map editing
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
        prompts,
        num_steps,
        cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
        tokenizer,
        max_num_words=77):

    if (type(cross_replace_steps) is not dict) and \
        (type(cross_replace_steps) is not omegaconf.dictconfig.DictConfig) and \
            (type(cross_replace_steps) is not OrderedDict):  # noqa
        cross_replace_steps = {'default_': cross_replace_steps}  # noqa

    if 'default_' not in cross_replace_steps:
        cross_replace_steps['default_'] = (0., 1.)

    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps['default_'], i)
    for key, item in cross_replace_steps.items():
        if key != 'default_':
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
