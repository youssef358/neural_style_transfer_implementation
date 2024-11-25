import numpy as np
from torch.optim import LBFGS
from torch.autograd import Variable
import os
import torch
from loss import LossFunctions
from utils import ImageProcessor
from model import VGG19


class NeuralStyleTransfer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VGG19().to(self.device)

    def run(self):

        content_optimizing_layer = self.config['content_layer']
        style_optimizing_layers = self.config['style_layers']

        content_image_path = os.path.join(self.config['content_image_directory'], self.config['content_image_name'])
        style_image_path = os.path.join(self.config['style_image_directory'], self.config['style_image_name'])

        output_directory_name = 'nst_' + self.config['content_image_name'].split('.')[0] + '_' + \
                                self.config['style_image_name'].split('.')[0]
        output_directory_path = os.path.join(self.config['output_directory'], output_directory_name)
        os.makedirs(output_directory_path, exist_ok=True)

        content_image = ImageProcessor.prepare_image(content_image_path, self.config['height'], self.device)
        style_image = ImageProcessor.prepare_image(style_image_path, self.config['height'], self.device)

        gaussian_noise_image = np.random.normal(
            loc=0, scale=90.0, size=content_image.shape
        ).astype(np.float32)
        init_image = torch.from_numpy(gaussian_noise_image).float().to(self.device)
        optimizing_image = Variable(init_image, requires_grad=True)

        content_set_of_feature_maps = self.model(content_image)
        style_set_of_feature_maps = self.model(style_image)

        content_dict = {
            content_optimizing_layer[0]: content_set_of_feature_maps[content_optimizing_layer[0]].squeeze(axis=0)
        }

        style_dict = {
            layer_name: LossFunctions.compute_gram_matrix(x)
            for layer_name, x in style_set_of_feature_maps.items()
            if layer_name in style_optimizing_layers
        }

        target_representations = [
            content_dict,
            style_dict
        ]

        optimizer = LBFGS([optimizing_image], max_iter=self.config['num_iterations'])
        counter = 0

        def closure():
            nonlocal counter
            current_set_of_feature_maps = self.model(optimizing_image)
            current_content_dict = {
                content_optimizing_layer[0]: current_set_of_feature_maps[content_optimizing_layer[0]].squeeze(0)
            }

            current_style_dict = {
                layer_name: LossFunctions.compute_gram_matrix(x)
                for layer_name, x in current_set_of_feature_maps.items()
                if layer_name in style_optimizing_layers
            }

            current_representations = [
                current_content_dict,
                current_style_dict
            ]

            if torch.is_grad_enabled():
                optimizer.zero_grad()

            total_loss, content_loss, style_loss, total_variation_loss = LossFunctions.compute_total_loss(
                optimizing_image, current_representations, target_representations, style_optimizing_layers,
                content_optimizing_layer, self.config)

            content_weight = self.config['content_weight']
            style_weight = self.config['style_weight']
            tv_weight = self.config['tv_weight']

            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():

                print(f"L-BFGS | iteration: {counter:03}, total loss={total_loss.item():12.4f}, "
                      f"content_loss={content_weight * content_loss.item():12.4f}, "
                      f"style loss={style_weight * style_loss.item():12.4f}, "
                      f"tv loss={tv_weight * total_variation_loss.item():12.4f}")

                ImageProcessor.save_optimization_result(optimizing_image, output_directory_path, self.config, counter)
            counter += 1
            return total_loss

        optimizer.step(closure)

        return output_directory_path

config = {
    'content_layer' : ['conv4_2'],
    'style_layers' : ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    'content_image_directory': '../data/content',
    'style_image_directory': '../data/style',
    'content_image_name': 'Tuebingen.jpg',
    'style_image_name': 'der_schrei.jpg',
    'output_directory': '../data/output',
    'height': 400,
    'style_weight': 2.0,
    'content_weight': 1800,
    'tv_weight': 0.0,
    'num_iterations': 1000,
    'image_format': (4, '.jpg'),
    'saving_frequency': -1,
}

nst = NeuralStyleTransfer(config)
nst.run()

