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

""" TF 2.0 CLIPCap mapper model, compatible with the official CLIP_prefix_caption implementation """

import os
import json
import logging
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from loggers import timer
from utils.tensorflow_utils import tf_compile
from custom_architectures.transformers_arch.transformer_arch import (
    HParamsTransformerEncoder, TransformerEncoder
)

logger  = logging.getLogger(__name__)

_CLIP_CAP_MODEL_PATH    = {
    'coco'  : 'coco_weights.pt',
    'conceptual'    : 'conceptual_weights.pt',
    'transformer'   : 'transformer_weights.pt'
}

HParamsClipCapTransformerMapper = HParamsTransformerEncoder(
    clip_dim    = -1,
    prefix_length   = 40,
    prefix_drop_rate    = 0.1,
    
    normalize   = 'middle',
    mha_normalize   = False,
    mha_normalize_input = True,
    mha_num_heads   = 8,
    mha_epsilon     = 1e-5,
    epsilon     = 1e-5,

    ffn_activation  = 'relu'
)

class ClipCapTransformerMapper(TransformerEncoder):
    default_params  = HParamsClipCapTransformerMapper
    _attr_to_set    = TransformerEncoder._attr_to_set + ['clip_dim', 'prefix_length']
    
    def __init__(self, clip_dim, name = 'mapper', ** kwargs):
        super().__init__(clip_dim = clip_dim, name = name, ** kwargs)
    
    def _init_input_layers(self, * args, ** kwargs):
        with tf.name_scope(self.name):
            self.prefix_count   = self.add_weight(
                shape = (self.prefix_length, self.embedding_dim), name = 'prefix_count'
            )
        
        self.prefix_layer   = tf.keras.layers.Dense(
            self.embedding_dim * self.prefix_length, name = 'prefix_layer'
        )
        self.prefix_drop    = tf.keras.layers.Dropout(
            self.hparams.prefix_drop_rate
        ) if self.hparams.prefix_drop_rate > 0 else None
    
    @property
    def input_signature(self):
        return tf.TensorSpec(shape = (None, self.clip_dim), dtype = tf.float32)

    def prepare_input(self, inputs, training = False, ** kwargs):
        inputs = tf.reshape(
            self.prefix_layer(inputs), [tf.shape(inputs)[0], self.prefix_length, self.embedding_dim]
        )
        prefix = tf.tile(tf.expand_dims(self.prefix_count, axis = 0), [tf.shape(inputs)[0], 1, 1])
        prefix = tf.concat([inputs, prefix], axis = 1)

        if self.prefix_drop is not None: prefix = self.prefix_drop(prefix, training = training)
        
        prefix._keras_mask = tf.fill((tf.shape(prefix)[0], tf.shape(prefix)[1]), True)
        return prefix

    def compute_output(self, output, ** kwargs):
        return super().compute_output(output, ** kwargs)[:, self.prefix_length :]
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _attn_split

        if not isinstance(pretrained, dict): pretrained = pretrained.state_dict()
        pretrained = {k : v for k, v in pretrained.items() if k.startswith('clip')}
        
        return super().transfer_weights(pretrained, transforms = _attn_split, ** kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name = 'transformer', pretrained = None,
                        tqdm = lambda x: x, ** kwargs):
        state_dict = load_clip_cap(pretrained_name, pretrained = pretrained)
        state_dict = {k : v for k, v in state_dict.items() if k.startswith('clip_project')}
        
        num_layers = len(set(k for k in state_dict if 'fc2.weight' in k))

        config = HParamsClipCapTransformerMapper(
            clip_dim      = state_dict['clip_project.linear.weight'].size()[1],
            prefix_length = state_dict['clip_project.prefix_const'].size()[0],
            embedding_dim = state_dict['clip_project.prefix_const'].size()[1],
            num_layers    = num_layers,
            mha_num_heads = 8,
            mha_use_bias  = 'clip_project.transformer.layers.0.attn.to_queries.bias' in state_dict,
            ffn_dim       = state_dict['clip_project.transformer.layers.0.mlp.fc1.weight'].size()[0],
            ffn_activation = 'relu'
        )
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(state_dict, tqdm = tqdm)
        
        return instance

class ClipCap(tf.keras.Model):
    _attr_to_set = ['vocab_size', 'max_input_length', 'sos_token', 'eos_token', 'pad_token']
    
    def __init__(self, mapper = 'transformer', generator = 'gpt2', name = 'ClipCap', ** kwargs):
        super().__init__(name = name)
        
        if isinstance(mapper, str):
            if mapper == 'transformer':
                mapper = ClipCapTransformerMapper(
                    ** {k.replace('mapper_', '') : v for k, v in kwargs.items()}
                )
            else:
                raise ValueError('Unhandled mapper : {}'.format(mapper))
        
        if isinstance(generator, str):
            generator = get_pretrained_transformer(
                generator, ** {k.replace('generator_', '') for k, v in kwargs.items()}
            )
        
        assert isinstance(mapper, tf.keras.Model), 'Unsupported mapper (type {}) : {}'.format(type(mapper), mapper)
        assert isinstance(generator, tf.keras.Model), 'Unsupported generator (type {}) : {}'.format(type(generator), generator)
        
        self.mapper     = mapper
        self.generator  = generator
        
        for key in self._attr_to_set:
            setattr(self, key, self.generator.hparams[key])
    
    @property
    def clip_dim(self):
        return self.mapper.clip_dim
    
    @property
    def prefix_length(self):
        return self.mapper.prefix_length
    
    @property
    def embedding_dim(self):
        return self.mapper.embedding_dim
    
    @property
    def dummy_inputs(self):
        gen_inputs  = self.generator.dummy_inputs
        if not isinstance(gen_inputs, list): gen_inputs = [gen_inputs]
        
        batch_size  = tf.shape(gen_inputs[0])[0]
        return [tf.random.normal((batch_size, self.clip_dim))] + gen_inputs

    def _build(self, ** kwargs):
        return self(self.dummy_inputs, ** kwargs)
    
    def set_tokens(self, ** kwargs):
        return self.generator.set_tokens(** kwargs)
    
    @timer(name = 'ClipCap call')
    def call(self,
             inputs,
             tokens = None,
             input_length = None,
             prefix = None,
             
             training   = False,
             ** kwargs
            ):
        if prefix is None:
            embedded_image = inputs
            if isinstance(inputs, (list, tuple)):
                embedded_image = inputs[0]
                if len(inputs) > 1: tokens = inputs[1]
                if len(inputs) > 2: input_length = inputs[2]

            prefix  = self.mapper(embedded_image, training = training)
        else:
            tokens = inputs
        
        return self.generator(
            tokens, input_length = input_length, prefix = prefix, training = training, ** kwargs
        )

    @timer
    @tf_compile(
        reduce_retracing = True, support_xla = False, follow_type_hints = True, cast_kwargs = True
    )
    def infer(self,
              embedding : tf.Tensor = None,
              prefix    : tf.Tensor = None,
              training  = False,
              ** kwargs
             ):
        if prefix is None:
            assert embedding is not None
            prefix  = self.mapper(embedding, training = training)
        
        return self.generator.infer(prefix = prefix, training = training, ** kwargs)

    def get_output_shape(self, * args, ** kwargs):
        return self.generator.get_output_shape(* args, ** kwargs)
    
    def transfer_weights(self, pretrained, ** kwargs):
        self.mapper.transfer_weights(pretrained, ** kwargs)
        self.generator.transfer_weights(pretrained, ** kwargs)

    def get_config(self):
        config  = {
            'mapper'    : json.loads(self.mapper.to_json()),
            'generator' : json.loads(self.generator.to_json())
        }
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        config.update({
            'mapper'    : model_from_json(
                json.dumps(config['mapper']), custom_objects = custom_objects
            ),
            'generator' : model_from_json(
                json.dumps(config['generator']), custom_objects = custom_objects
            )
        })
        return cls(** config)

    @classmethod
    def from_pretrained(cls, pretrained_name = 'transformer', pretrained = None, ** kwargs):
        from custom_architectures.transformers_arch import get_pretrained_transformer
        pretrained = load_clip_cap(pretrained_name, pretrained = pretrained)
        
        mapper  = ClipCapTransformerMapper.from_pretrained(
            pretrained_name, pretrained = pretrained, name = 'mapper', ** kwargs
        )
        generator   = get_pretrained_transformer(
            pretrained_name = 'gpt2', pretrained = pretrained, transpose = True,
            name = 'generator', ** kwargs,
        )
        
        instance = cls(mapper = mapper, generator = generator, ** kwargs)
        instance._build()
        
        return instance

def load_clip_cap(pretrained_name, pretrained = None):
    if pretrained is None:
        import torch
        
        from models import _pretrained_models_folder
        
        if pretrained_name not in _CLIP_CAP_MODEL_PATH:
            raise ValueError('Unknown ClipCap name !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_CLIP_CAP_MODEL_PATH.keys()), pretrained_name
            ))
        
        model_path = os.path.join(
            _pretrained_models_folder, 'pretrained_weights', _CLIP_CAP_MODEL_PATH[pretrained_name]
        )
        
        if not os.path.exists(model_path):
            raise ValueError('Model weights {} does not exist'.format(model_path))
        
        logger.info('Loading pretrained weights from {}'.format(model_path))
        pretrained = torch.load(model_path, map_location = 'cpu')
    
    return pretrained if isinstance(pretrained, dict) else pretrained.state_dict()

_clip_cap_objects   = {
    'ClipCap'   : ClipCap,
    'ClipCapTransformerMapper'   : ClipCapTransformerMapper
}
custom_functions    = _clip_cap_objects

custom_objects  = _clip_cap_objects

_encoders   = _clip_cap_objects
_transformers   = _encoders