from .decoders import RefineDecoder, CRPBlock
from .encoders import MobileNetV2Encoder, EfficientNetLite0Encoder
from .grid import GridNet
import segmentation_models_pytorch as smp

lite0_lrn4 = lambda: GridNet([
    {'branch': EfficientNetLite0Encoder},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

mn_lrn4 = lambda: GridNet([
    {'branch': MobileNetV2Encoder, 'pretrained': True},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

b0_lrn4 = lambda: GridNet([
    {'branch': smp.encoders.get_encoder, 'name': 'efficientnet-b0', 'weights': 'imagenet'},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

b1_lrn4 = lambda: GridNet([
    {'branch': smp.encoders.get_encoder, 'name': 'efficientnet-b1', 'weights': 'imagenet'},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

b2_lrn4 = lambda: GridNet([
    {'branch': smp.encoders.get_encoder, 'name': 'efficientnet-b2', 'weights': 'imagenet'},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

b3_lrn4 = lambda: GridNet([
    {'branch': smp.encoders.get_encoder, 'name': 'efficientnet-b3', 'weights': 'imagenet'},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

b4_lrn4 = lambda: GridNet([
    {'branch': smp.encoders.get_encoder, 'name': 'efficientnet-b4', 'weights': 'imagenet'},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])

b5_lrn4 = lambda: GridNet([
    {'branch': smp.encoders.get_encoder, 'name': 'efficientnet-b5', 'weights': 'imagenet'},
    {'branch': RefineDecoder, 'block': (lambda x: CRPBlock(x, x, 4))}
])
