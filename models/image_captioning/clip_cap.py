
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
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils.thread_utils import Pipeline
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_embedding_model import BaseEmbeddingModel
from custom_architectures.transformers_arch import get_pretrained_transformer

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

        tokens      = self.tf_encode_text(data)
        
        return (embedding, tokens[:-1]), tokens[1:]
        
    def filter_data(self, inputs, outputs):
        return tf.shape(outputs)[-1] <= self.max_output_length
    
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
    
    def get_pipeline(self, save = True, directory = None, post_processing = None,
                     batch_size = 1, ** kwargs):
        def inference(inputs, ** kw):
            embeddings = tf.stack(inputs) if isinstance(inputs, list) else inputs
            if len(tf.shape(embeddings)) == 1: embeddings = tf.expand_dims(embeddings, axis = 0)
            elif len(tf.shape(embeddings)) == 3: embeddings = tf.squeeze(embeddings, axis = 1)
            outputs = self.infer(embeddings, ** {** kwargs, ** kw})
            
            if isinstance(inputs, list):
                return [
                    tf.nest.map_structure(lambda o: o[b] if o is not None else o, outputs)
                    for b in range(len(inputs))
                ]
            return tf.nest.map_structure(lambda o: o[0] if o is not None else o, outputs)

        def decode(output, ** kw):
            return {'text' : self.decode_output(output)}
        
        self.load_encoder()
        
        if directory is None: directory = self.pred_dir
        
        pipeline = Pipeline(** {
            'name'      : 'image_captioning_pipeline',
            'filename'  : os.path.join(directory, 'map.json') if save else None,
            
            'tasks' : [
                self.get_input,
                {'consumer' : inference, 'batch_size' : batch_size, 'allow_multithread' : False},
                decode
            ],
            ** kwargs
        })
        pipeline.start()
        return pipeline
    
    @timer
    def predict(self, data, ** kwargs):
        if not isinstance(data, (list, tuple, np.ndarray, tf.Tensor, pd.DataFrame)): data = [data]
        pipe    = self.get_pipeline(** kwargs)
        
        return pipe.extend_and_wait(data, stop = True)
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_text(),
            ** self.get_config_embedding(),
            'max_output_length' : self.max_output_length
        })
        return config
