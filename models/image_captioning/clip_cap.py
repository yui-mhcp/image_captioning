
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import plot, load_json, dump_json
from utils.image import save_image, load_image
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_embedding_model import BaseEmbeddingModel
from custom_architectures.transformers_arch import get_pretrained_transformer

logger  = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_LENGTH = 150

class ClipCap(BaseTextModel, BaseEmbeddingModel):
    def __init__(self,
                 lang,
                 encoder_name   = 'clip_rn50x4',
                 embedding_dim  = 640,
                 
                 pretrained = 'transformer',
                 max_output_length  = DEFAULT_MAX_TEXT_LENGTH,
                 
                 ** kwargs
                ):
        if pretrained: kwargs.setdefault('text_encoder', 'gpt2')
        kwargs.setdefault('pretrained_name', pretrained)
        kwargs['use_label_embedding'] = False
        
        self._init_text(lang = lang, ** kwargs)
        self._init_embedding(encoder_name = encoder_name, embedding_dim = embedding_dim)
        
        self.max_output_length  = max_output_length

        super().__init__(pretrained = pretrained, ** kwargs)
        
        if hasattr(self.model, '_build'): self.model._build()
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

    def _build_model(self, pretrained = None, ** kwargs):
        if pretrained:
            super()._build_model(
                model   = get_pretrained_transformer(
                    class_name = 'ClipCap', pretrained_name = pretrained, ** kwargs
                )
            )
        else:
            super()._build_model(
                model = {
                    'architecture_name' : kwargs.pop('architecture_name', 'ClipCap'),
                    'clip_dim'      : self.embedding_dim,
                    'vocab_size'    : self.vocab_size,
                    ** kwargs
                }
            )
    
    @property
    def input_signature(self):
        return (self.embedding_signature, self.text_signature)
    
    @property
    def output_signature(self):
        return self.text_signature
        
    @property
    def training_hparams(self):
        return super().training_hparams(
            ** self.training_hparams_text,
            ** self.training_hparams_embedding,
            max_output_length   = None
        )
    
    def __str__(self):
        return super().__str__() + self._str_text() + self._str_embedding()
    
    @timer(name = 'inference')
    def infer(self, * args, ** kwargs):
        kwargs.setdefault('max_length', self.max_output_length)
        
        return self.model.infer(* args, ** kwargs)
    
    def decode_output(self, output, ** kwargs):
        return self.decode_text(output.tokens if hasattr(output, 'tokens') else output, ** kwargs)
    
    def compile(self, loss = 'TextLoss', metrics = ['F1'], **kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def add_embeddings(self, * args, ** kwargs):
        pass
    
    def set_embeddings(self, * args, ** kwargs):
        pass
    
    def get_input(self, data, ** kwargs):
        kwargs.setdefault('default_key', 'image_embedding')
        return self.get_embedding(data, ** kwargs)
    
    def get_output(self, data, ** kwargs):
        return self.tf_encode_text(data)
    
    def encode_data(self, data):
        embedding   = self.get_input(data)

        tokens      = self.get_output(data)
        
        return (embedding, tokens[:-1]), tokens[1:]
        
    def filter_data(self, inputs, outputs):
        return tf.logical_and(
            tf.shape(outputs)[0] > 0,
            tf.shape(outputs)[-1] <= self.max_output_length
        )
    
    def augment_data(self, inputs, outputs):
        embedding, tokens = inputs
        
        embedding   = self.maybe_augment_embedding(embedding)
        tokens      = self.augment_text(tokens)
        
        return (embedding, tokens), outputs
    
    def embed(self, data, * args, ** kwargs):
        return self.encoder.embed_image(data, * args, ** kwargs)
    
    def get_dataset_config(self, ** kwargs):
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'  : True,
            'pad_kwargs'    : {
                'padding_values'    : (
                    (0., self.blank_token_idx), self.blank_token_idx
                )
            }
        })
        
        return super().get_dataset_config(**kwargs)
    
    @timer
    def predict(self,
                images,
                batch_size = 1,
                
                save    = True,
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                save_if_raw = True,
                
                display = False,
                post_processing = None,
                
                ** kwargs
               ):
        """
            Performs image captioning on the givan `images` (either filename / embeddings / raw)
            
            Arguments :
                - images  : the image(s) to caption
                    - str   : the filename of the image
                    - dict / pd.Series  : informations about the image
                        - must contain at least `filename` or `embedding` or `image_embedding`
                    - np.ndarray / tf.Tensor    : the embedding for the image
                    
                    - list / pd.DataFrame   : an iterable of the above types
                - batch_size    : the number of prediction to perform in parallel
                
                - save  : whether to save the result in a mapping file
                - directory : where to save the mapping file (+ possibly raw images)
                - overwrite : whether to overwrite the already existing predictions
                - timestamp : the timestamp before which to overwrite (-1 means : always overwrite)
                - save_if_raw   : whether to save raw images or not
                
                - kwargs    : forwarded to `self.infer`
            Returns :
                - result    : a list of tuple (image, result)
                    - image     : either the filename (if any), either the original image
                    - result    : a `dict` with (at least) keys
                        - text      : (list of) str, the predicted text(s)
                        - timestamp : the timestamp at which the prediction has been performed
            
            Note : the predicted text is typically a list if the inference method is `beam_search`, and single str if the method is not.
        """
        ####################
        # helping function #
        ####################
        
        @timer
        def post_process(idx):
            while idx < len(results) and results[idx] is not None:
                file, infos = results[idx]
                
                if display:
                    if isinstance(infos['text'], list):
                        logger.info('Captions :\n- {}'.format('\n- '.join(infos['text'])))
                    else:
                        logger.info('Caption : {}'.format(infos['text']))
                    plot(load_image(file))
                
                if post_processing is not None:
                    post_processing(infos, image = file)
                
                idx += 1
            
            return idx
        
        def save_raw_image(image):
            os.makedirs(raw_img_dir, exist_ok = True)
            filename = os.path.join(raw_img_dir, 'image_{}.png'.format(len(os.listdir(raw_img_dir))))
            save_image(image = image, filename = filename)
            return filename
        
        def should_predict(image):
            if isinstance(image, (dict, pd.Series)) and 'filename' in image: image = image['filename']
            if isinstance(image, str) and image in predicted:
                if not overwrite or (timestamp != -1 and timestamp <= predicted[image].get('timestamp', -1)):
                    return False
            return True
        
        def get_filename(image):
            if isinstance(image, (dict, pd.Series)) and 'filename' in image: image = image['filename']
            if isinstance(image, (np.ndarray, tf.Tensor)):
                if not save_raw_image or len(image.shape) != 3: return None
                return save_raw_image(image)
            elif not isinstance(image, str):
                raise ValueError('Unknown image type ({}) : {}'.format(type(image), image))
            return image
        
        now = time.time()
        
        if not save: display = True
        
        if isinstance(images, pd.DataFrame): images = images.to_dict('records')
        if not isinstance(images, (list, tuple, np.ndarray, tf.Tensor)): images = [images]
        elif isinstance(images, (np.ndarray, tf.Tensor)) and len(images.shape) == 3:
            images = tf.expand_dims(images, axis = 0)
        
        ##############################
        #   Saving initialization    #
        ##############################
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        raw_img_dir = os.path.join(directory, 'images')
        
        predicted   = load_json(map_file, default = {})
        
        ####################
        #  Pre-processing  #
        ####################
        
        results     = [None] * len(images)
        duplicatas  = {}
        requested   = [(get_filename(img), img) for img in images]
        
        inputs  = []
        for i, (file, img) in enumerate(requested):
            if not should_predict(file):
                results[i] = (file, predicted[file])
            else:
                if isinstance(file, str):
                    duplicatas.setdefault(file, []).append(i)
                    if len(duplicatas[file]) > 1: continue
                
                inputs.append((i, file, img))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = post_process(0)
        
        if len(inputs) > 0:
            encoded = self.get_input([img for _, _, img in inputs])
            for start in range(0, len(inputs), batch_size):
                outputs = self.infer(encoded[start : start + batch_size], ** kwargs)
                texts   = self.decode_output(outputs)
                
                should_save = False
                for (idx, file, img), text in zip(inputs[start : start + batch_size], texts):
                    if file is None: file = img
                    
                    infos   = {'text' : text, 'timestamp' : now}
                    if isinstance(file, str):
                        infos['filename']   = file
                        should_save     = True
                        predicted[file] = infos
                        
                        for duplicate_idx in duplicatas[file]:
                            results[duplicate_idx] = (file, infos)
                    else:
                        results[idx] = (file if file else img, infos)
                
                if save and should_save:
                    dump_json(map_file, predicted, indent = 4)
                
                show_idx = post_process(show_idx)
        
        return results

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_text(),
            ** self.get_config_embedding(),
            'max_output_length' : self.max_output_length
        })
        return config
